"""HTTP shim between ART's tau-bench client and tau2-bench v1.0.1.

ART's TauBenchClient speaks the env-server protocol of the original
tau-bench: string actions, a per-step reward/terminated pair, a user-first
initial observation. tau2 v1.0.1 ships no compatible server -- its
simulation service runs whole episodes with tau2's own agent, and its
environment manager exposes bare tool routes with no user simulator and no
rewards (see UPSTREAM.md; the full contract analysis lives in
planning/notes/upstream_art_contract.md). This module composes tau2's own
registry, Environment, UserSimulator, and evaluator behind that older
protocol.

Two choices here are correctness-critical. First, ART sums step rewards
into the trajectory reward, so every intermediate step returns 0.0 and the
official evaluator's reward is returned exactly once, on the terminating
step -- anything else over-counts. Second, every message is a real tau2
message stamped at event time and appended in event order, mirroring what
the Orchestrator would have recorded, so the official evaluator can
strict-replay the trajectory against a fresh environment.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator
from tau2.utils.llm_utils import get_cost
from tau2.utils.tools import parse_action_string
from tau2.utils.utils import get_now

from service_agent.splits import load_frozen_dev_ids, train_core_ids

# tau2 splits that contain the official test tasks. Serving them for
# training would contaminate the protocol (CLAUDE.md hard rule 2), so they
# are refused unless explicitly unlocked for a final evaluation run.
EVAL_SPLITS = frozenset({"test", "full", "base"})

# Splits defined by this repo's frozen data protocol rather than tau2's
# split file. They partition tau2's train split, so they are only
# meaningful for the task sets that share the telecom split file.
PROTOCOL_SPLITS = frozenset({"dev", "train-core"})
PROTOCOL_DOMAINS = frozenset({"telecom", "telecom-workflow"})

# ART only ever reads task.id; everything else in the task is either the
# user simulator's script or the evaluator's labels, and none of it may
# leave the server (CLAUDE.md hard rule 3).
REDACTED_TASK_FIELDS = (
    "description",
    "user_scenario",
    "ticket",
    "initial_state",
    "evaluation_criteria",
)

# Mirrors the Orchestrator's max_errors default: ten errored tool
# executions end the episode with TOO_MANY_ERRORS.
MAX_ERRORS = 10

DEFAULT_MAX_STEPS = 200


class CreateEnvironmentRequest(BaseModel):
    domain: str
    task_id: str
    user_llm: Optional[str] = None
    user_llm_args: Optional[dict[str, Any]] = None
    retrieval_config: Optional[str] = None
    retrieval_config_kwargs: Optional[dict[str, Any]] = None
    idle_timeout_seconds: Optional[float] = None


class StepRequest(BaseModel):
    action: str


@dataclass
class EnvSession:
    """One live environment plus everything needed to finalize and grade it."""

    id: str
    domain: str
    task: Task
    env: Environment
    env_kwargs: dict[str, Any]
    user: UserSimulator
    user_state: Any
    trajectory: list[Message]
    start_time: str
    start_perf: float
    idle_timeout_seconds: Optional[float]
    last_touched: float
    # ART runs many rollouts concurrently against distinct environments,
    # but nothing stops a client from stepping one environment from two
    # threads; the lock keeps the trajectory append-only and ordered.
    lock: threading.Lock = field(default_factory=threading.Lock)
    step_count: int = 0
    num_errors: int = 0
    done: bool = False
    termination_reason: Optional[TerminationReason] = None
    pending_termination: Optional[TerminationReason] = None
    last_user_text: str = ""
    simulation: Optional[SimulationRun] = None


class EnvironmentStore:
    """In-process session registry with lazy idle collection.

    There is no background reaper: every access sweeps sessions whose
    idle_timeout_seconds elapsed since they were last touched. ART deletes
    environments in a finally block, so the sweep only matters when a
    client crashes hard and its environments would otherwise leak.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, EnvSession] = {}
        self._lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        expired = [
            env_id
            for env_id, session in self._sessions.items()
            if session.idle_timeout_seconds is not None
            and now - session.last_touched > session.idle_timeout_seconds
        ]
        for env_id in expired:
            del self._sessions[env_id]

    def add(self, session: EnvSession) -> None:
        with self._lock:
            self._sweep(time.monotonic())
            self._sessions[session.id] = session

    def get(self, env_id: str) -> Optional[EnvSession]:
        with self._lock:
            now = time.monotonic()
            self._sweep(now)
            session = self._sessions.get(env_id)
            if session is not None:
                session.last_touched = now
            return session

    def delete(self, env_id: str) -> bool:
        with self._lock:
            self._sweep(time.monotonic())
            return self._sessions.pop(env_id, None) is not None


def parse_action(action: str) -> AssistantMessage:
    """Translate ART's string action into a tau2 assistant message.

    ART encodes tool calls as "name(k=v, ...)" with Python-repr values
    (rollout.py builds them with f"{key}={value!r}"); tau2's
    parse_action_string already AST-parses exactly that shape, including
    True/False/None, negative numbers, and nested lists/dicts, so it is
    reused rather than reimplemented. Anything that does not parse as a
    call is the assistant's text turn. Parsed tool calls get fresh unique
    ids because parse_action_string leaves the id empty and the evaluator's
    replay pairs calls with results by id.
    """
    if not action.strip():
        # ART sends "" when the policy returns neither text nor tool calls.
        # Keep it as an (empty) text turn: the evaluator ignores it and the
        # user simulator responds without recording it.
        return AssistantMessage(role="assistant", content=action)
    message = parse_action_string(action, requestor="assistant")
    if message.is_tool_call():
        for tool_call in message.tool_calls:
            tool_call.id = uuid.uuid4().hex
    return message


# Loading a task set means parsing the full tasks JSON (2285 tasks for
# telecom); doing that on every environment creation is pure waste, and the
# Task objects are treated as read-only everywhere in tau2's evaluator, so
# a process-wide cache is safe.
_TASKS_BY_DOMAIN: dict[str, dict[str, Task]] = {}


def _all_tasks(domain: str) -> dict[str, Task]:
    if domain not in _TASKS_BY_DOMAIN:
        try:
            loader = registry.get_tasks_loader(domain)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown domain: {domain}")
        try:
            tasks = loader(task_split_name=None)
        except TypeError:
            tasks = loader()
        _TASKS_BY_DOMAIN[domain] = {task.id: task for task in tasks}
    return _TASKS_BY_DOMAIN[domain]


def _resolve_split_tasks(domain: str, split: str) -> list[Task]:
    try:
        loader = registry.get_tasks_loader(domain)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown domain: {domain}")
    if split in PROTOCOL_SPLITS:
        if domain not in PROTOCOL_DOMAINS:
            raise HTTPException(
                status_code=400,
                detail=f"Split '{split}' is defined by the telecom data protocol "
                f"and is not available for domain '{domain}'.",
            )
        ids = set(load_frozen_dev_ids()) if split == "dev" else set(train_core_ids())
        tasks = [task for task in loader(task_split_name="train") if task.id in ids]
        if len(tasks) != len(ids):
            # The frozen ids no longer match upstream's train split; that is
            # protocol drift, not a client error, so fail loudly.
            raise HTTPException(
                status_code=500,
                detail=f"Frozen '{split}' ids do not match the upstream train split.",
            )
        return tasks
    try:
        return loader(task_split_name=split)
    except TypeError:
        raise HTTPException(
            status_code=400, detail=f"Task set '{domain}' does not support splits."
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _redact_task(task: Task) -> dict[str, Any]:
    data = task.model_dump(mode="json")
    for field_name in REDACTED_TASK_FIELDS:
        data[field_name] = None
    return data


def _drive_user(session: EnvSession, message: Message) -> tuple[UserMessage, float]:
    """Run the user's turn until the simulator produces a text message.

    Mirrors the Orchestrator's AGENT/ENV->USER and USER->ENV branches: user
    tool calls are executed against the environment and their results fed
    back to the simulator until it says something to the agent. Every
    message lands in the trajectory in event order. Returns the final user
    text message and the summed LLM cost of the user messages this turn
    (ART reads it as info["user_message_cost"]).
    """
    cost = 0.0
    while True:
        user_msg, session.user_state = session.user.generate_next_message(
            message, session.user_state
        )
        session.trajectory.append(user_msg)
        cost += user_msg.cost or 0.0
        if not user_msg.is_tool_call():
            session.last_user_text = user_msg.content or ""
            return user_msg, cost
        results = []
        for tool_call in user_msg.tool_calls:
            result = session.env.get_response(tool_call)
            if result.error:
                session.num_errors += 1
            results.append(result)
        session.trajectory.extend(results)
        if len(results) == 1:
            message = results[0]
        else:
            message = MultiToolMessage(role="tool", tool_messages=results)


def _final_messages(session: EnvSession) -> list[Message]:
    # Same transformation as Orchestrator.get_trajectory: order by the
    # event-time stamps, then number the turns for the evaluator.
    messages = sorted(deepcopy(session.trajectory), key=lambda m: m.timestamp)
    for turn_idx, message in enumerate(messages):
        message.turn_idx = turn_idx
    return messages


def _finalize(session: EnvSession, reason: TerminationReason) -> float:
    """Close the episode and grade it with tau2's official evaluator.

    Builds the SimulationRun the way Orchestrator._finalize does and runs
    evaluate_simulation exactly once. The returned reward is what the
    terminating step reports to ART; because ART sums step rewards, this is
    the only step allowed to return a non-zero value.
    """
    messages = _final_messages(session)
    costs = get_cost(messages)
    agent_cost, user_cost = costs if costs is not None else (None, None)
    simulation = SimulationRun(
        id=str(uuid.uuid4()),
        task_id=session.task.id,
        start_time=session.start_time,
        end_time=get_now(),
        duration=time.perf_counter() - session.start_perf,
        termination_reason=reason.value,
        agent_cost=agent_cost,
        user_cost=user_cost,
        messages=messages,
        seed=None,
    )
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=session.task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=session.domain,
        env_kwargs=session.env_kwargs or None,
    )
    simulation.reward_info = reward_info
    session.simulation = simulation
    session.termination_reason = reason
    session.done = True
    return float(reward_info.reward)


def create_app() -> FastAPI:
    app = FastAPI(title="tau2 environment shim for ART's tau-bench client")
    store = EnvironmentStore()
    max_steps = int(os.environ.get("SHIM_MAX_STEPS", str(DEFAULT_MAX_STEPS)))

    def _get_session(env_id: str) -> EnvSession:
        session = store.get(env_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown environment: {env_id}")
        return session

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/scenarios")
    def get_scenarios(
        domain: str = Query(...), split: Optional[str] = Query(None)
    ) -> dict[str, Any]:
        if split is None:
            # tau2's loaders default to "base", which contains the official
            # test tasks; an implicit default must not be able to leak them.
            raise HTTPException(
                status_code=422,
                detail="split is required (train, dev, train-core, small, base, full, test)",
            )
        if split in EVAL_SPLITS and os.environ.get("SHIM_ALLOW_EVAL_SPLITS") != "1":
            raise HTTPException(
                status_code=403,
                detail=f"Split '{split}' contains official test tasks and is "
                "locked to prevent training contamination. Set "
                "SHIM_ALLOW_EVAL_SPLITS=1 only for a final evaluation run.",
            )
        tasks = _resolve_split_tasks(domain, split)
        return {
            "scenarios": [{"domain": domain, "task": _redact_task(task)} for task in tasks]
        }

    @app.post("/environments")
    def create_environment(request: CreateEnvironmentRequest) -> dict[str, Any]:
        if request.user_llm is None:
            raise HTTPException(
                status_code=422,
                detail="user_llm is required: the user simulator runs server-side",
            )
        task = _all_tasks(request.domain).get(request.task_id)
        if task is None:
            raise HTTPException(
                status_code=404,
                detail=f"Task '{request.task_id}' not found in domain '{request.domain}'",
            )
        initial_state = task.initial_state
        if initial_state is not None and initial_state.message_history:
            # Mid-conversation starts would need the agent- and user-state
            # seeding logic of Orchestrator.initialize; telecom tasks never
            # use it, so refuse loudly instead of mis-running the episode.
            raise HTTPException(
                status_code=422,
                detail="Tasks with initial_state.message_history are not supported",
            )
        env_kwargs: dict[str, Any] = {}
        if request.retrieval_config is not None:
            # Same mapping tau2's runner uses for banking_knowledge
            # (runner/helpers.py get_info); other domains take no retrieval
            # kwargs, so only pass them when requested.
            env_kwargs["retrieval_variant"] = request.retrieval_config
            if request.retrieval_config_kwargs:
                env_kwargs["retrieval_kwargs"] = dict(request.retrieval_config_kwargs)
        try:
            env = registry.get_env_constructor(request.domain)(**env_kwargs)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Unknown domain: {request.domain}"
            )
        env.set_state(
            initialization_data=(
                initial_state.initialization_data if initial_state is not None else None
            ),
            initialization_actions=(
                initial_state.initialization_actions if initial_state is not None else None
            ),
            message_history=[],
        )
        try:
            user_tools = env.get_user_tools(include=task.user_tools) or None
        except Exception:
            # Same fallback as tau2's build_user: domains without user tools
            # raise, and the simulator then runs tool-less.
            user_tools = None
        user = UserSimulator(
            llm=request.user_llm,
            llm_args=request.user_llm_args,
            instructions=str(task.user_scenario),
            tools=user_tools,
        )
        session = EnvSession(
            id=uuid.uuid4().hex,
            domain=request.domain,
            task=task,
            env=env,
            env_kwargs=env_kwargs,
            user=user,
            user_state=user.get_init_state(),
            trajectory=[],
            start_time=get_now(),
            start_perf=time.perf_counter(),
            idle_timeout_seconds=request.idle_timeout_seconds,
            last_touched=time.monotonic(),
        )
        # tau2 is agent-first, ART expects a user-first observation. Feeding
        # tau2's canonical greeting through the user simulator produces that
        # first observation while keeping the trajectory identical to what
        # the Orchestrator would record (greeting included).
        greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        greeting.timestamp = get_now()
        session.trajectory.append(greeting)
        user_msg, _ = _drive_user(session, greeting)
        if UserSimulator.is_stop(user_msg):
            # The create response carries no terminated flag, so an
            # immediate stop is deferred: the first step call finalizes.
            session.pending_termination = TerminationReason.USER_STOP
        store.add(session)
        return {
            "id": session.id,
            "observation": f"user: {user_msg.content}",
            "info": {
                "policy": env.get_policy(),
                "tools": [tool.openai_schema for tool in env.get_tools()],
            },
        }

    @app.post("/environments/{env_id}/step")
    def step_environment(env_id: str, request: StepRequest) -> dict[str, Any]:
        session = _get_session(env_id)
        with session.lock:
            if session.done:
                raise HTTPException(
                    status_code=409, detail="Environment episode already terminated"
                )
            if session.pending_termination is not None:
                # The user ended the conversation during creation; the agent
                # action never happened in tau2 terms, so drop it and close.
                reward = _finalize(session, session.pending_termination)
                return {
                    "id": session.id,
                    "observation": f"user: {session.last_user_text}",
                    "info": {},
                    "reward": reward,
                    "terminated": True,
                    "truncated": False,
                }
            session.step_count += 1
            message = parse_action(request.action)
            session.trajectory.append(message)
            stop: Optional[TerminationReason] = None
            if message.is_tool_call():
                results = []
                for tool_call in message.tool_calls:
                    result = session.env.get_response(tool_call)
                    if result.error:
                        session.num_errors += 1
                    results.append(result)
                session.trajectory.extend(results)
                observation = f"tool: {results[-1].content or ''}"
                info: dict[str, Any] = {}
            else:
                user_msg, user_cost = _drive_user(session, message)
                observation = f"user: {user_msg.content}"
                info = {"user_message_cost": user_cost}
                if UserSimulator.is_stop(user_msg):
                    stop = TerminationReason.USER_STOP
            # A real user stop takes precedence over the guards: it is the
            # only reason here the evaluator fully grades instead of
            # returning 0.0 for a premature ending.
            if stop is None and session.num_errors >= MAX_ERRORS:
                stop = TerminationReason.TOO_MANY_ERRORS
            if stop is None and session.step_count >= max_steps:
                stop = TerminationReason.MAX_STEPS
            reward = _finalize(session, stop) if stop is not None else 0.0
            return {
                "id": session.id,
                "observation": observation,
                "info": info,
                "reward": reward,
                "terminated": stop is not None,
                "truncated": False,
            }

    @app.get("/environments/{env_id}/trajectory")
    def get_trajectory(env_id: str) -> dict[str, Any]:
        # Debug/parity surface, not part of ART's contract: exposes the
        # exact message list the evaluator saw (or would see) so tests can
        # independently strict-replay it.
        session = _get_session(env_id)
        with session.lock:
            if session.simulation is not None:
                messages = session.simulation.messages
            else:
                messages = _final_messages(session)
            return {
                "id": session.id,
                "task_id": session.task.id,
                "domain": session.domain,
                "termination_reason": (
                    session.termination_reason.value
                    if session.termination_reason is not None
                    else None
                ),
                "messages": [message.model_dump(mode="json") for message in messages],
            }

    @app.delete("/environments/{env_id}")
    def delete_environment(env_id: str) -> dict[str, Any]:
        if not store.delete(env_id):
            raise HTTPException(status_code=404, detail=f"Unknown environment: {env_id}")
        return {"id": env_id, "deleted": True}

    return app


def main() -> None:
    """Run the shim under uvicorn; port from SHIM_PORT, default 8000."""
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("SHIM_HOST", "127.0.0.1"),
        port=int(os.environ.get("SHIM_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
