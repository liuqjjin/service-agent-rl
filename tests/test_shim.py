"""The ART tau-bench contract shim against real tau2 components.

Everything is real except the user simulator's LLM: tau2's generate is
patched at the user_simulator module seam with a scripted queue, so no test
needs an API key. The environment, the task data, the split protocol, and
the evaluator (including strict trajectory replay) all run for real.
"""

import time
import uuid

import pytest
from fastapi.testclient import TestClient
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.user.user_simulator_base import STOP
from tau2.utils.utils import get_now

from service_agent.serve.tau2_shim import create_app, parse_action
from service_agent.splits import load_frozen_dev_ids, load_split_ids, train_core_ids

TASK_ID = "[service_issue]break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:Hard]"

REDACTED_FIELDS = (
    "description",
    "user_scenario",
    "ticket",
    "initial_state",
    "evaluation_criteria",
)

USER_MESSAGE_COST = 0.25


class ScriptedUserLLM:
    """Stands in for llm_utils.generate inside the shim's UserSimulator.

    Items are specs (a text string or a (tool_name, arguments) pair), not
    prebuilt messages: tau2 sorts trajectories by timestamp, so messages
    must be stamped when the call happens, not when the script is declared
    (the same trap ScriptedModel documents in test_governed_agent_replay).
    When the script runs out the user stops, like a satisfied customer.
    """

    def __init__(self, items):
        self.items = list(items)
        self.calls = 0

    def __call__(self, model, messages, tools=None, **kwargs):
        self.calls += 1
        item = self.items.pop(0) if self.items else STOP
        if isinstance(item, str):
            return AssistantMessage(
                role="assistant", content=item, cost=USER_MESSAGE_COST
            )
        name, arguments = item
        return AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=uuid.uuid4().hex,
                    name=name,
                    arguments=arguments,
                    requestor="assistant",
                )
            ],
            cost=USER_MESSAGE_COST,
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("SHIM_ALLOW_EVAL_SPLITS", raising=False)
    return TestClient(create_app())


def script_user(monkeypatch, items):
    scripted = ScriptedUserLLM(items)
    monkeypatch.setattr("tau2.user.user_simulator.generate", scripted)
    return scripted


def create_env(client, **overrides):
    body = {
        "domain": "telecom",
        "task_id": TASK_ID,
        "user_llm": "scripted/none",
        "user_llm_args": {"temperature": 0.0},
    }
    body.update(overrides)
    response = client.post("/environments", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# -- scenarios ----------------------------------------------------------------


def test_scenarios_train_split_redacts_labels(client):
    response = client.get("/scenarios", params={"domain": "telecom", "split": "train"})
    assert response.status_code == 200
    scenarios = response.json()["scenarios"]
    assert {s["task"]["id"] for s in scenarios} == load_split_ids()["train"]
    for scenario in scenarios:
        assert scenario["domain"] == "telecom"
        for field in REDACTED_FIELDS:
            assert scenario["task"][field] is None


def test_scenarios_dev_split_is_the_frozen_ids(client):
    response = client.get("/scenarios", params={"domain": "telecom", "split": "dev"})
    assert response.status_code == 200
    ids = [s["task"]["id"] for s in response.json()["scenarios"]]
    assert len(ids) == 20
    assert set(ids) == set(load_frozen_dev_ids())


def test_scenarios_eval_splits_are_locked(client, monkeypatch):
    for split in ("test", "full", "base"):
        response = client.get("/scenarios", params={"domain": "telecom", "split": split})
        assert response.status_code == 403
        assert "SHIM_ALLOW_EVAL_SPLITS" in response.json()["detail"]
    # The unlock is explicit and env-scoped, for the one final eval run.
    monkeypatch.setenv("SHIM_ALLOW_EVAL_SPLITS", "1")
    response = client.get("/scenarios", params={"domain": "telecom", "split": "test"})
    assert response.status_code == 200
    ids = {s["task"]["id"] for s in response.json()["scenarios"]}
    assert ids == load_split_ids()["test"]


# -- environment creation -----------------------------------------------------


def test_create_environment(client, monkeypatch, telecom_env_constructor):
    from tau2.agent.llm_agent import LLMAgent

    script_user(
        monkeypatch, ["Hi, my mobile data is not working and my line is suspended."]
    )
    payload = create_env(client)
    assert payload["observation"].startswith("user: ")
    assert "mobile data" in payload["observation"]

    env = telecom_env_constructor()
    native_agent = LLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="scripted/none",
    )
    assert payload["info"]["policy"] == native_agent.system_prompt

    tools = payload["info"]["tools"]
    assert isinstance(tools, list) and tools
    assert all(tool["type"] == "function" for tool in tools)
    names = {tool["function"]["name"] for tool in tools}
    assert names == {tool.name for tool in env.get_tools()}
    assert "get_customer_by_id" in names


def test_training_contract_matches_native_agent(client, telecom_env_constructor):
    from tau2.agent.llm_agent import LLMAgent

    response = client.get("/training-contract", params={"domain": "telecom"})
    assert response.status_code == 200
    payload = response.json()

    env = telecom_env_constructor()
    native_agent = LLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="contract/none",
    )
    assert payload["system_prompt"] == native_agent.system_prompt
    assert payload["tools"] == [tool.openai_schema for tool in env.get_tools()]


def test_create_environment_unknown_task(client):
    response = client.post(
        "/environments",
        json={"domain": "telecom", "task_id": "no-such-task", "user_llm": "scripted/none"},
    )
    assert response.status_code == 404
    assert "no-such-task" in response.json()["detail"]


def test_create_environment_refuses_test_split_tasks(client, monkeypatch):
    # Locking /scenarios is not enough: test task ids are readable straight out
    # of split_tasks.json, so a client can name one without ever listing it.
    body = {
        "domain": "telecom",
        "task_id": sorted(load_split_ids()["test"])[0],
        "user_llm": "scripted/none",
    }
    response = client.post("/environments", json=body)
    assert response.status_code == 403
    assert "SHIM_ALLOW_EVAL_SPLITS" in response.json()["detail"]

    # Same env-scoped unlock as /scenarios, for the one final evaluation run.
    script_user(monkeypatch, ["Hello."])
    monkeypatch.setenv("SHIM_ALLOW_EVAL_SPLITS", "1")
    assert client.post("/environments", json=body).status_code == 200


def test_create_environment_allows_train_core_tasks(client, monkeypatch):
    # The lock covers the test split only: a task from full/base that is not in
    # test is legitimate training material and must still instantiate.
    script_user(monkeypatch, ["Hello."])
    response = client.post(
        "/environments",
        json={
            "domain": "telecom",
            "task_id": train_core_ids()[0],
            "user_llm": "scripted/none",
        },
    )
    assert response.status_code == 200, response.text


# -- stepping -----------------------------------------------------------------


def test_step_tool_call(client, monkeypatch):
    script_user(monkeypatch, ["My line is suspended, please help."])
    env_id = create_env(client)["id"]
    response = client.post(
        f"/environments/{env_id}/step",
        json={"action": "get_customer_by_id(customer_id='C1001')"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["observation"].startswith("tool: ")
    assert "C1001" in payload["observation"]
    assert payload["reward"] == 0.0
    assert payload["terminated"] is False
    assert payload["truncated"] is False


def test_parse_action_python_reprs():
    # The exact encoding ART's rollout produces: f"{key}={value!r}" over
    # json.loads-ed arguments.
    arguments = {
        "customer_id": "C1001",
        "enabled": True,
        "missing": None,
        "ratio": 0.5,
        "count": -3,
        "tags": ["a", "it's"],
        "nested": {"k": [1, 2], "flag": False},
    }
    action = "configure(" + ", ".join(f"{k}={v!r}" for k, v in arguments.items()) + ")"
    message = parse_action(action)
    assert message.is_tool_call()
    (tool_call,) = message.tool_calls
    assert tool_call.name == "configure"
    assert tool_call.arguments == arguments
    assert tool_call.requestor == "assistant"
    assert tool_call.id  # replay pairs calls with results by id


def test_parse_action_text():
    text = "Could you check the status bar on your phone?"
    message = parse_action(text)
    assert not message.is_tool_call()
    assert message.content == text


# -- full episode -------------------------------------------------------------


def run_episode(client, monkeypatch):
    """Drive a short but complete episode: agent reads and writes, the user
    answers once via a device tool, then stops. Returns (env_id, steps)."""
    script_user(
        monkeypatch,
        [
            "Hi, my mobile data is broken and my line seems suspended.",
            ("check_status_bar", {}),
            "My status bar shows no service.",
            STOP,
        ],
    )
    env_id = create_env(client)["id"]
    actions = [
        "get_customer_by_id(customer_id='C1001')",
        "send_payment_request(customer_id='C1001', bill_id='B1234321')",
        "Could you check the status bar on your phone for me?",
        "Thanks. Is there anything else I can help you with?",
    ]
    steps = []
    for action in actions:
        response = client.post(f"/environments/{env_id}/step", json={"action": action})
        assert response.status_code == 200, response.text
        steps.append(response.json())
    return env_id, steps


def test_full_episode_reward_only_on_termination(client, monkeypatch):
    _, steps = run_episode(client, monkeypatch)

    assert [s["terminated"] for s in steps] == [False, False, False, True]
    # ART sums step rewards, so every non-terminating step must be 0.0 and
    # the final reward must come from the real evaluator exactly once.
    assert [s["reward"] for s in steps[:3]] == [0.0, 0.0, 0.0]
    assert isinstance(steps[3]["reward"], float)
    assert all(s["truncated"] is False for s in steps)

    # Tool steps report tool observations; text steps report user ones.
    assert steps[0]["observation"].startswith("tool: ")
    assert steps[1]["observation"].startswith("tool: ")
    assert steps[2]["observation"].startswith("user: ")
    assert steps[3]["observation"].startswith("user: ")
    assert STOP in steps[3]["observation"]

    # The status-bar turn cost two user LLM calls (tool call + text), the
    # stop turn one; tool steps carry no user cost key.
    assert steps[2]["info"]["user_message_cost"] == pytest.approx(2 * USER_MESSAGE_COST)
    assert steps[3]["info"]["user_message_cost"] == pytest.approx(USER_MESSAGE_COST)


# -- deletion and GC ----------------------------------------------------------


def test_delete_environment(client, monkeypatch):
    script_user(monkeypatch, ["Hello."])
    env_id = create_env(client)["id"]
    response = client.delete(f"/environments/{env_id}")
    assert response.status_code == 200
    assert response.json() == {"id": env_id, "deleted": True}

    second = client.delete(f"/environments/{env_id}")
    assert second.status_code == 404
    assert "detail" in second.json()


def test_idle_environment_is_collected(client, monkeypatch):
    script_user(monkeypatch, ["Hello."])
    env_id = create_env(client, idle_timeout_seconds=0.01)["id"]
    time.sleep(0.05)
    response = client.post(f"/environments/{env_id}/step", json={"action": "hi"})
    assert response.status_code == 404


# -- trajectory replay safety -------------------------------------------------

MESSAGE_TYPES = {"user": UserMessage, "assistant": AssistantMessage, "tool": ToolMessage}


def test_trajectory_survives_strict_replay(client, monkeypatch, telecom_tasks):
    env_id, steps = run_episode(client, monkeypatch)

    response = client.get(f"/environments/{env_id}/trajectory")
    assert response.status_code == 200
    data = response.json()
    assert data["termination_reason"] == "user_stop"

    # Rebuild the SimulationRun the way the shim did and grade it with the
    # official evaluator in strict-replay mode: any drift between what the
    # live environment did and what the trajectory records would raise.
    messages = [MESSAGE_TYPES[m["role"]].model_validate(m) for m in data["messages"]]
    simulation = SimulationRun(
        id="replay",
        task_id=data["task_id"],
        start_time=get_now(),
        end_time=get_now(),
        duration=0.0,
        termination_reason="user_stop",
        messages=messages,
    )
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=telecom_tasks[TASK_ID],
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="telecom",
    )
    assert reward_info.reward is not None
    # Deterministic evaluation: replaying yields the reward the shim served.
    assert reward_info.reward == steps[3]["reward"]
