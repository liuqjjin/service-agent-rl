"""Parity: the same scripted episode through the native Orchestrator and
through the shim must agree on what happened and what it scored.

The RL trajectories come from the shim; the ablation and final evaluation
come from the native runner. If the two paths disagreed on tool sequences or
rewards, the 2x2 table would be comparing apples to oranges. Here one agent
script and one user script drive both paths; the test asserts identical
policy text, identical executed tool sequences, and identical official
reward."""

import json
import uuid

from fastapi.testclient import TestClient
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator_base import STOP
from tau2.utils.utils import get_now

from service_agent.serve import create_app

TASK_ID = "[service_issue]break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:Hard]"
CUSTOMER, OVERDUE_BILL = "C1001", "B1234321"

# One episode, expressed as scripts shared by both paths.
AGENT_SCRIPT = [
    ("get_customer_by_id", {"customer_id": CUSTOMER}),
    ("get_bills_for_customer", {"customer_id": CUSTOMER}),
    ("send_payment_request", {"customer_id": CUSTOMER, "bill_id": OVERDUE_BILL}),
    "I have sent a payment request for your overdue bill.",
]
USER_SCRIPT = ["I want to pay my overdue bill.", STOP]


class ScriptedUserLLM:
    def __init__(self, items):
        self.items = list(items)

    def __call__(self, model, messages, tools=None, **kwargs):
        item = self.items.pop(0) if self.items else STOP
        msg = AssistantMessage(role="assistant", content=item, cost=0.0)
        msg.timestamp = get_now()
        return msg


class ScriptedAgentLLM:
    def __init__(self, items):
        self.items = list(items)

    def __call__(self, model, messages, tools=None, **kwargs):
        item = self.items.pop(0) if self.items else STOP
        if isinstance(item, str):
            msg = AssistantMessage(role="assistant", content=item, cost=0.0)
        else:
            name, arguments = item
            msg = AssistantMessage(
                role="assistant",
                cost=0.0,
                tool_calls=[
                    ToolCall(
                        id=uuid.uuid4().hex,
                        name=name,
                        arguments=arguments,
                        requestor="assistant",
                    )
                ],
            )
        msg.timestamp = get_now()
        return msg


def art_action(item) -> str:
    """Encode an agent script item exactly the way ART's rollout does."""
    if isinstance(item, str):
        return item
    name, arguments = item
    return f"{name}(" + ", ".join(f"{k}={v!r}" for k, v in arguments.items()) + ")"


def run_native(monkeypatch, telecom_tasks):
    from tau2.agent.llm_agent import LLMAgent

    monkeypatch.setattr(
        "tau2.agent.llm_agent.generate", ScriptedAgentLLM(AGENT_SCRIPT)
    )
    monkeypatch.setattr(
        "tau2.user.user_simulator.generate", ScriptedUserLLM(USER_SCRIPT)
    )
    from tau2.user.user_simulator import UserSimulator

    environment = registry.get_env_constructor("telecom")()
    task = telecom_tasks[TASK_ID]
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="scripted/none",
    )
    user = UserSimulator(
        tools=None, instructions=task.user_scenario, llm="scripted/none"
    )
    orchestrator = Orchestrator(
        domain="telecom",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=40,
        seed=7,
    )
    simulation = orchestrator.run()
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="telecom",
    )
    tool_sequence = [
        tc.name
        for m in simulation.messages
        if isinstance(m, AssistantMessage) and m.is_tool_call()
        for tc in m.tool_calls
    ]
    return agent.system_prompt, tool_sequence, float(reward_info.reward)


def run_shim(monkeypatch):
    monkeypatch.setattr(
        "tau2.user.user_simulator.generate", ScriptedUserLLM(USER_SCRIPT)
    )
    client = TestClient(create_app())
    created = client.post(
        "/environments",
        json={
            "domain": "telecom",
            "task_id": TASK_ID,
            "user_llm": "scripted/none",
            "user_llm_args": {"temperature": 0.0},
        },
    )
    assert created.status_code == 200, created.text
    env_id = created.json()["id"]
    policy = created.json()["info"]["policy"]

    reward = 0.0
    terminated = False
    for item in AGENT_SCRIPT:
        step = client.post(
            f"/environments/{env_id}/step", json={"action": art_action(item)}
        )
        assert step.status_code == 200, step.text
        payload = step.json()
        reward += payload["reward"]
        terminated = payload["terminated"]
        if terminated:
            break

    assert terminated, "scripted episode must terminate via user STOP"
    trajectory = client.get(f"/environments/{env_id}/trajectory").json()["messages"]
    tool_sequence = [
        tc["name"]
        for m in trajectory
        if m.get("role") == "assistant" and m.get("tool_calls")
        for tc in m["tool_calls"]
    ]
    return policy, tool_sequence, reward


def test_native_and_shim_agree(monkeypatch, telecom_tasks):
    native_policy, native_tools, native_reward = run_native(monkeypatch, telecom_tasks)
    shim_policy, shim_tools, shim_reward = run_shim(monkeypatch)

    assert shim_policy == native_policy
    assert shim_tools == native_tools == [
        "get_customer_by_id",
        "get_bills_for_customer",
        "send_payment_request",
    ]
    assert shim_reward == native_reward
    # Both paths were graded by the same official evaluator; agreement here
    # plus the strict-replay test in test_shim.py is the parity guarantee the
    # final 2x2 rests on.
    assert isinstance(json.dumps(shim_reward), str)
