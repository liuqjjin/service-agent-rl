"""ART tau-bench rollout with a fail-closed multi-tool boundary.

ART's pinned rollout sends multiple tool calls from one assistant choice as
separate HTTP steps. tau2 instead records one AssistantMessage followed by all
matching ToolMessages. The old protocol cannot preserve that grouping, so
training must stop before executing any of those calls rather than create a
trajectory that the native evaluator would interpret differently.
"""

from __future__ import annotations

import json
from typing import Any


class MultipleToolCallsError(RuntimeError):
    """One model choice contained a message the shim cannot preserve."""


def require_single_tool_call(choice: Any) -> None:
    tool_calls = getattr(getattr(choice, "message", None), "tool_calls", None) or []
    if len(tool_calls) > 1:
        raise MultipleToolCallsError(
            "ART's string-action protocol cannot preserve one assistant message "
            "containing multiple tool calls; no call was executed"
        )


def _tool_call_action(tool_call: Any) -> str:
    arguments = json.loads(tool_call.function.arguments)
    args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{tool_call.function.name}({args})"


def _is_max_tokens_error(exc: Exception) -> bool:
    message = getattr(exc, "message", str(exc))
    return "max_tokens" in message or "max_completion_tokens" in message


async def rollout(
    scenario: Any,
    model: Any,
    *,
    client: Any,
    max_turns: int,
    max_completion_tokens: int,
    max_model_len: int,
    temperature: float,
    policy_seed: int,
    user_model_name: str,
    user_chat_completion_kwargs: dict[str, Any],
) -> Any:
    """Run one official-reward episode without ART's multi-call split bug."""

    from art import Trajectory
    from openai import BadRequestError

    async with client.environment(
        domain=scenario.domain,
        task_id=scenario.task.id,
        user_llm=user_model_name,
        user_llm_args=user_chat_completion_kwargs,
    ) as env:
        openai_client = model.openai_client()
        trajectory = Trajectory(
            messages_and_choices=[
                {"role": "system", "content": env.info["policy"]},
                {
                    "role": "user",
                    "content": env.observation.removeprefix("user: "),
                },
            ],
            tools=env.info.get("tools"),
            reward=0.0,
            metrics={
                "cost/user": 0.0,
                "strict_replay": 0.0,
                "reward_finalized_once": 0.0,
                "multi_tool_calls": 0.0,
            },
            metadata={"scenario_id": scenario.task.id, "policy_seed": policy_seed},
        )
        terminated = False
        turns = 0
        while not terminated and turns < max_turns:
            try:
                completion = await openai_client.chat.completions.create(
                    messages=trajectory.messages(),
                    model=model.get_inference_name(),
                    stream=False,
                    tool_choice="auto",
                    tools=trajectory.tools or [],
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                    # Native tau2 calls set_seed once on the agent and reuse
                    # that seed for each generation in the episode.
                    seed=policy_seed,
                )
            except BadRequestError as exc:
                if _is_max_tokens_error(exc):
                    break
                raise

            choice = completion.choices[0]
            require_single_tool_call(choice)
            trajectory.messages_and_choices.append(choice)
            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls:
                step = await client.step_environment(
                    env.id,
                    _tool_call_action(tool_calls[0]),
                )
                trajectory.messages_and_choices.append(
                    {
                        "role": "tool",
                        "content": step.observation.removeprefix("tool: "),
                        "tool_call_id": tool_calls[0].id,
                    }
                )
            else:
                step = await client.step_environment(
                    env.id,
                    choice.message.content or "",
                )
                trajectory.messages_and_choices.append(
                    {
                        "role": "user",
                        "content": step.observation.removeprefix("user: "),
                    }
                )
                trajectory.metrics["cost/user"] += float(
                    step.info.get("user_message_cost", 0.0)
                )

            trajectory.reward += step.reward
            terminated = step.terminated or step.truncated
            if step.info.get("strict_replay") is True:
                trajectory.metrics["strict_replay"] = 1.0
            if step.info.get("reward_finalized_once") is True:
                trajectory.metrics["reward_finalized_once"] = 1.0
            turns += 1

            usage = completion.usage
            if (
                usage is not None
                and usage.total_tokens + max_completion_tokens > max_model_len
            ):
                break

        trajectory.metrics["num_turns"] = float(turns)
        trajectory.metrics["terminated"] = float(terminated)
        return trajectory
