"""End-to-end: the governed agent through a real Orchestrator, real telecom
environment, and the real evaluator. Only the LLM and the user are scripted.

This is the correctness test the whole design hangs on: rejected candidates
must never appear in the official trajectory, allowed writes must execute and
replay, and the official evaluator must run cleanly (strict replay) on
whatever we produce. If governance ever intercepted in the wrong place, these
tests are the ones that break."""


import json

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator_base import STOP

from service_agent.agent import GovernanceConfig, GovernedLLMAgent
from service_agent.governance import Decision

TASK_ID = "[service_issue]break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:Hard]"
CUSTOMER, LINE, OVERDUE_BILL = "C1001", "L1002", "B1234321"


def text(content):
    return AssistantMessage(role="assistant", content=content, cost=0.0)


def call(call_id, name, **arguments):
    return AssistantMessage(
        role="assistant",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments, requestor="assistant")],
        cost=0.0,
    )


class ScriptedModel:
    """Stands in for llm_utils.generate inside the governed agent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts_seen = []

    def __call__(self, model, messages, tools=None, **kwargs):
        from tau2.utils.utils import get_now

        self.prompts_seen.append(list(messages))
        response = self.responses.pop(0) if self.responses else text(STOP)
        response = response.model_copy(deep=True)
        # The orchestrator sorts the trajectory by timestamp; a pre-built
        # script would carry construction-time stamps and scramble the order.
        # Real generate() stamps at call time, so the mock must too.
        response.timestamp = get_now()
        return response


class ScriptedUser:
    """Minimal half-duplex user: says its lines, then stops."""

    def __init__(self, lines):
        self.lines = list(lines)

    def get_init_state(self, message_history=None):
        return {"messages": list(message_history or [])}

    def set_seed(self, seed):
        pass

    def stop(self, message=None, state=None):
        pass

    @classmethod
    def is_stop(cls, message):
        return message.content is not None and STOP in message.content

    def generate_next_message(self, message, state):
        content = self.lines.pop(0) if self.lines else STOP
        return UserMessage(role="user", content=content, cost=0.0), state


def run_sim(monkeypatch, telecom_tasks, agent_script, user_lines, config=None):
    scripted = ScriptedModel(agent_script)
    monkeypatch.setattr("service_agent.agent.governed.generate", scripted)

    environment = registry.get_env_constructor("telecom")()
    agent = GovernedLLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="scripted/none",
        config=config or GovernanceConfig.h2(),
    )
    task = telecom_tasks[TASK_ID]
    orchestrator = Orchestrator(
        domain="telecom",
        agent=agent,
        user=ScriptedUser(user_lines),
        environment=environment,
        task=task,
        max_steps=40,
        seed=7,
    )
    simulation = orchestrator.run()
    return simulation, agent, scripted, task


def trajectory_tool_calls(simulation):
    return [
        tc.name
        for msg in simulation.messages
        if isinstance(msg, AssistantMessage) and msg.is_tool_call()
        for tc in msg.tool_calls
    ]


def test_rejected_write_never_enters_trajectory(monkeypatch, telecom_tasks):
    # The agent tries to resume a suspended line cold -- no reads, unpaid
    # overdue bill. The gate must reject it and the trajectory must show only
    # the safe text message that replaced it.
    simulation, agent, scripted, task = run_sim(
        monkeypatch,
        telecom_tasks,
        agent_script=[
            call("tc1", "resume_line", customer_id=CUSTOMER, line_id=LINE),
            text("Let me review your account details first."),
        ],
        user_lines=["My line is suspended, resume it right now please."],
    )

    assert simulation.termination_reason == "user_stop"
    assert "resume_line" not in trajectory_tool_calls(simulation)
    rejected = agent.audit.rejected_candidates()
    assert [r.tool_name for r in rejected] == ["resume_line"]
    assert rejected[0].decision == Decision.REQUIRE_EVIDENCE.value

    # The governance feedback reached the model privately, folded into the
    # system prompt (serving stacks reject system messages that are not at
    # position zero -- found the hard way on mlx_lm.server)...
    last_prompt = scripted.prompts_seen[-1]
    feedback = [m for m in last_prompt if "execution governance" in (m.content or "")]
    assert feedback and feedback[0] is last_prompt[0]
    for prompt in scripted.prompts_seen:
        assert all(getattr(m, "role", "") != "system" for m in prompt[1:]), (
            "system content may only appear at position zero"
        )
    # ...and never the official trajectory.
    assert not any(
        "execution governance" in (m.content or "") for m in simulation.messages
    )
    # The feedback is a new model-visible surface: it must be built only from
    # conversation-derived evidence, never task labels.
    from service_agent.leakage import find_leaks

    for msg in feedback:
        assert find_leaks(msg.content, task, strict=False) == []

    # The official evaluator must accept the trajectory (strict replay).
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="telecom",
    )
    assert reward_info.reward is not None


def test_allowed_write_executes_and_replays(monkeypatch, telecom_tasks):
    # Proper flow: identify the customer, read the bills, then send the
    # payment request. The gate allows it, the environment executes it, and
    # the evaluator can replay the trajectory without error.
    simulation, agent, scripted, task = run_sim(
        monkeypatch,
        telecom_tasks,
        agent_script=[
            call("tc1", "get_customer_by_id", customer_id=CUSTOMER),
            call("tc2", "get_bills_for_customer", customer_id=CUSTOMER),
            call("tc3", "send_payment_request", customer_id=CUSTOMER, bill_id=OVERDUE_BILL),
            text("I have sent you a payment request for the overdue bill."),
        ],
        user_lines=["I'd like to pay my overdue bill."],
    )

    calls = trajectory_tool_calls(simulation)
    assert calls == [
        "get_customer_by_id",
        "get_bills_for_customer",
        "send_payment_request",
    ]
    assert agent.audit.counts_by_decision().get("deny", 0) == 0

    # Every trajectory tool call has its ToolMessage (replay precondition).
    tool_message_ids = {
        m.id for m in simulation.messages if isinstance(m, ToolMessage)
    }
    assert {"tc1", "tc2", "tc3"} <= tool_message_ids

    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="telecom",
    )
    assert reward_info.reward is not None
    assert reward_info.db_check is not None


def test_duplicate_write_blocked_once_executed(monkeypatch, telecom_tasks):
    # The user insists; the model complies with a second identical payment
    # request. H2's idempotency ledger must stop the repeat before it reaches
    # the environment: exactly one send_payment_request in the trajectory.
    simulation, agent, scripted, task = run_sim(
        monkeypatch,
        telecom_tasks,
        agent_script=[
            call("tc1", "get_customer_by_id", customer_id=CUSTOMER),
            call("tc2", "get_bills_for_customer", customer_id=CUSTOMER),
            call("tc3", "send_payment_request", customer_id=CUSTOMER, bill_id=OVERDUE_BILL),
            text("The payment request has been sent."),
            call("tc4", "send_payment_request", customer_id=CUSTOMER, bill_id=OVERDUE_BILL),
            text("It has already been sent -- no need to send it twice."),
        ],
        user_lines=[
            "I'd like to pay my overdue bill.",
            "Are you sure? Please send the payment request again.",
        ],
    )

    calls = trajectory_tool_calls(simulation)
    assert calls.count("send_payment_request") == 1
    duplicates = [
        r for r in agent.audit.records if r.reason_code == "duplicate_write"
    ]
    assert len(duplicates) == 1

    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="telecom",
    )
    assert reward_info.reward is not None


def test_h1_lets_duplicate_through(monkeypatch, telecom_tasks):
    # Same story under H1 (no idempotency ledger): the duplicate is allowed
    # to reach the environment, which raises "already awaiting payment" --
    # the exact failure H2 exists to prevent. The ablation measures this gap.
    simulation, agent, scripted, task = run_sim(
        monkeypatch,
        telecom_tasks,
        agent_script=[
            call("tc1", "get_customer_by_id", customer_id=CUSTOMER),
            call("tc2", "get_bills_for_customer", customer_id=CUSTOMER),
            call("tc3", "send_payment_request", customer_id=CUSTOMER, bill_id=OVERDUE_BILL),
            text("The payment request has been sent."),
            call("tc4", "send_payment_request", customer_id=CUSTOMER, bill_id=OVERDUE_BILL),
            text("Apologies -- that request was already pending."),
        ],
        user_lines=[
            "I'd like to pay my overdue bill.",
            "Are you sure? Please send the payment request again.",
        ],
        config=GovernanceConfig.h1(),
    )

    calls = trajectory_tool_calls(simulation)
    assert calls.count("send_payment_request") == 2
    errors = [
        m for m in simulation.messages if isinstance(m, ToolMessage) and m.error
    ]
    assert any("already awaiting payment" in (m.content or "") for m in errors)

    # Even the failed duplicate must replay cleanly: errored calls made no
    # state change and are replayed as errors again.
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="telecom",
    )
    assert reward_info.reward is not None


def test_regeneration_budget_produces_safe_fallback(monkeypatch, telecom_tasks):
    # The model stubbornly repeats the same illegal write. After the
    # regeneration budget is spent the agent must send the fallback text --
    # the rejected call still never reaches the trajectory.
    illegal = call("tc1", "resume_line", customer_id=CUSTOMER, line_id=LINE)
    simulation, agent, scripted, task = run_sim(
        monkeypatch,
        telecom_tasks,
        agent_script=[illegal, illegal, illegal],
        user_lines=["Resume my line immediately."],
    )

    assert "resume_line" not in trajectory_tool_calls(simulation)
    assert len(agent.audit.rejected_candidates()) == 3
    fallback_texts = [
        m.content
        for m in simulation.messages
        if isinstance(m, AssistantMessage) and not m.is_tool_call()
    ]
    assert any("double-check the details" in (t or "") for t in fallback_texts)


def test_audit_dump_is_idempotent(tmp_path):
    from service_agent.governance.audit import AuditTrail
    from service_agent.governance.core import GovernanceResult, ProposedAction

    trail = AuditTrail(session_id="session", task_id="task")
    trail.record(
        ProposedAction("get_customer_by_id", {"customer_id": "C1"}, "call-1"),
        GovernanceResult(Decision.ALLOW, "read_or_generic", ""),
        attempt=0,
    )
    path = tmp_path / "audit_session.jsonl"

    trail.dump_jsonl(path)
    trail.dump_jsonl(path)

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["tool_call_id"] == "call-1"
