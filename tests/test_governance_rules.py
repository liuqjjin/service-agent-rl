"""Write-gate rules against the policy they were derived from.

Every scenario builds evidence the same way the runtime does -- by feeding
real tau2 message objects through the EvidenceExtractor -- so these tests
exercise the JSON parsing path, not hand-assembled state."""

import json

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from service_agent.governance import (
    Decision,
    EvidenceExtractor,
    ProposedAction,
    TelecomGovernor,
    idempotency_key,
)

SESSION = "test-session"

CUSTOMER = {"customer_id": "C1", "full_name": "Jane Doe", "line_ids": ["L1"], "bill_ids": ["B1", "B2"]}
LINE_OK = {
    "line_id": "L1",
    "status": "Suspended",
    "plan_id": "P1",
    "contract_end_date": "2026-01-01",
    "roaming_enabled": False,
}
LINE_EXPIRED = {**LINE_OK, "contract_end_date": "2025-01-01"}
PLAN = {"plan_id": "P1", "data_limit_gb": 5.0, "data_refueling_price_per_gb": 4.0}
BILL_OVERDUE = {"bill_id": "B1", "customer_id": "C1", "status": "Overdue", "total_due": 40.0}
BILL_PAID = {"bill_id": "B1", "customer_id": "C1", "status": "Paid", "total_due": 40.0}
BILL_ISSUED = {"bill_id": "B1", "customer_id": "C1", "status": "Issued", "total_due": 40.0}
BILL_AWAITING = {"bill_id": "B2", "customer_id": "C1", "status": "Awaiting Payment", "total_due": 10.0}


def observe_read(extractor, tool_name, payload, call_id):
    """Simulate one read: agent tool call followed by its result."""
    call = ToolCall(id=call_id, name=tool_name, arguments={})
    extractor.observe_agent_message(
        AssistantMessage(role="assistant", tool_calls=[call])
    )
    extractor.observe_tool_result(
        ToolMessage(
            id=call_id, role="tool", requestor="assistant", content=json.dumps(payload)
        )
    )


@pytest.fixture
def governor():
    return TelecomGovernor(session_id=SESSION)


@pytest.fixture
def extractor():
    return EvidenceExtractor()


def proposed(tool_name, **arguments):
    return ProposedAction(tool_name=tool_name, arguments=arguments, tool_call_id="tc")


# -- reads and generic tools are never gated ----------------------------------


def test_reads_and_transfer_always_allowed(governor, extractor):
    for tool in ("get_customer_by_id", "get_bills_for_customer", "transfer_to_human_agents"):
        result = governor.evaluate(proposed(tool), extractor.state)
        assert result.decision is Decision.ALLOW


# -- send_payment_request ------------------------------------------------------


def test_payment_requires_identified_customer(governor, extractor):
    result = governor.evaluate(
        proposed("send_payment_request", customer_id="C1", bill_id="B1"),
        extractor.state,
    )
    assert result.decision is Decision.REQUIRE_EVIDENCE
    assert result.reason_code == "customer_not_identified"


def test_payment_requires_bill_read(governor, extractor):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    result = governor.evaluate(
        proposed("send_payment_request", customer_id="C1", bill_id="B1"),
        extractor.state,
    )
    assert result.decision is Decision.REQUIRE_EVIDENCE
    assert result.reason_code == "bill_not_read"


@pytest.mark.parametrize(
    "bill,reason",
    [(BILL_PAID, "bill_already_paid"), (BILL_ISSUED, "bill_not_overdue")],
)
def test_payment_denied_for_non_overdue_bill(governor, extractor, bill, reason):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(extractor, "get_bills_for_customer", [bill], "b1")
    result = governor.evaluate(
        proposed("send_payment_request", customer_id="C1", bill_id="B1"),
        extractor.state,
    )
    assert result.decision is Decision.DENY
    assert result.reason_code == reason


def test_payment_denied_when_another_bill_awaiting(governor, extractor):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(extractor, "get_bills_for_customer", [BILL_OVERDUE, BILL_AWAITING], "b1")
    result = governor.evaluate(
        proposed("send_payment_request", customer_id="C1", bill_id="B1"),
        extractor.state,
    )
    assert result.decision is Decision.DENY
    assert result.reason_code == "another_bill_awaiting_payment"


def test_payment_denied_for_foreign_bill(governor, extractor):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(
        extractor,
        "get_bills_for_customer",
        [{**BILL_OVERDUE, "customer_id": "C999"}],
        "b1",
    )
    result = governor.evaluate(
        proposed("send_payment_request", customer_id="C1", bill_id="B1"),
        extractor.state,
    )
    assert result.decision is Decision.DENY
    assert result.reason_code == "bill_wrong_customer"


def test_payment_allowed_when_preconditions_met(governor, extractor):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(extractor, "get_bills_for_customer", [BILL_OVERDUE], "b1")
    result = governor.evaluate(
        proposed("send_payment_request", customer_id="C1", bill_id="B1"),
        extractor.state,
    )
    assert result.decision is Decision.ALLOW


# -- resume_line ---------------------------------------------------------------


def test_resume_requires_line_read(governor, extractor):
    result = governor.evaluate(
        proposed("resume_line", customer_id="C1", line_id="L1"), extractor.state
    )
    assert result.decision is Decision.REQUIRE_EVIDENCE
    assert result.reason_code == "line_not_read"


def test_resume_denied_after_contract_end_even_if_bills_paid(governor, extractor):
    # main_policy.md:126 -- the one rule that must hold even when everything
    # else looks fine.
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(extractor, "get_details_by_id", LINE_EXPIRED, "l1")
    observe_read(extractor, "get_bills_for_customer", [BILL_PAID], "b1")
    result = governor.evaluate(
        proposed("resume_line", customer_id="C1", line_id="L1"), extractor.state
    )
    assert result.decision is Decision.DENY
    assert result.reason_code == "contract_expired"


def test_resume_requires_bills_read(governor, extractor):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(extractor, "get_details_by_id", LINE_OK, "l1")
    result = governor.evaluate(
        proposed("resume_line", customer_id="C1", line_id="L1"), extractor.state
    )
    assert result.decision is Decision.REQUIRE_EVIDENCE
    assert result.reason_code == "bills_not_read"


def test_resume_denied_with_unpaid_overdue_bill(governor, extractor):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(extractor, "get_details_by_id", LINE_OK, "l1")
    observe_read(extractor, "get_bills_for_customer", [BILL_OVERDUE], "b1")
    result = governor.evaluate(
        proposed("resume_line", customer_id="C1", line_id="L1"), extractor.state
    )
    assert result.decision is Decision.DENY
    assert result.reason_code == "overdue_bills_unpaid"


def test_resume_allowed_when_bills_paid_and_contract_live(governor, extractor):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(extractor, "get_details_by_id", LINE_OK, "l1")
    observe_read(extractor, "get_bills_for_customer", [BILL_PAID], "b1")
    result = governor.evaluate(
        proposed("resume_line", customer_id="C1", line_id="L1"), extractor.state
    )
    assert result.decision is Decision.ALLOW


# -- refuel_data ---------------------------------------------------------------


def _read_line_and_plan(extractor):
    observe_read(extractor, "get_details_by_id", LINE_OK, "l1")
    observe_read(extractor, "get_details_by_id", PLAN, "p1")


@pytest.mark.parametrize("amount", [2.5, 3.0, 0.0, -1.0])
def test_refuel_denied_out_of_bounds(governor, extractor, amount):
    _read_line_and_plan(extractor)
    result = governor.evaluate(
        proposed("refuel_data", customer_id="C1", line_id="L1", gb_amount=amount),
        extractor.state,
    )
    assert result.decision is Decision.DENY
    assert result.reason_code == "refuel_amount_out_of_bounds"


def test_refuel_requires_known_price(governor, extractor):
    result = governor.evaluate(
        proposed("refuel_data", customer_id="C1", line_id="L1", gb_amount=2.0),
        extractor.state,
    )
    assert result.decision is Decision.REQUIRE_EVIDENCE
    assert result.reason_code == "refuel_price_unknown"


def test_refuel_requires_confirmation(governor, extractor):
    _read_line_and_plan(extractor)
    result = governor.evaluate(
        proposed("refuel_data", customer_id="C1", line_id="L1", gb_amount=2.0),
        extractor.state,
    )
    assert result.decision is Decision.REQUIRE_CONFIRMATION
    assert result.reason_code == "price_not_confirmed"


def test_refuel_allowed_after_quote_and_affirmation(governor, extractor):
    _read_line_and_plan(extractor)
    extractor.observe_agent_message(
        AssistantMessage(
            role="assistant",
            content="Adding 2 GB at $4.00/GB will cost $8.00 total. Shall I proceed?",
        )
    )
    extractor.observe_user_message(UserMessage(role="user", content="Yes, go ahead."))
    result = governor.evaluate(
        proposed("refuel_data", customer_id="C1", line_id="L1", gb_amount=2.0),
        extractor.state,
    )
    assert result.decision is Decision.ALLOW


def test_refuel_hedged_reply_is_not_confirmation(governor, extractor):
    _read_line_and_plan(extractor)
    extractor.observe_agent_message(
        AssistantMessage(
            role="assistant", content="2 GB costs $8.00 total. Shall I proceed?"
        )
    )
    extractor.observe_user_message(
        UserMessage(role="user", content="Ok wait, actually no, hold on.")
    )
    result = governor.evaluate(
        proposed("refuel_data", customer_id="C1", line_id="L1", gb_amount=2.0),
        extractor.state,
    )
    assert result.decision is Decision.REQUIRE_CONFIRMATION


def test_refuel_confirmation_must_match_amount(governor, extractor):
    # The user confirmed 1 GB / $4; refueling 2 GB on that confirmation
    # would charge twice what was agreed.
    _read_line_and_plan(extractor)
    extractor.observe_agent_message(
        AssistantMessage(role="assistant", content="1 GB costs $4.00. Proceed?")
    )
    extractor.observe_user_message(UserMessage(role="user", content="yes"))
    result = governor.evaluate(
        proposed("refuel_data", customer_id="C1", line_id="L1", gb_amount=2.0),
        extractor.state,
    )
    assert result.decision is Decision.REQUIRE_CONFIRMATION


# -- idempotency ---------------------------------------------------------------


def test_duplicate_write_is_blocked(governor, extractor):
    observe_read(extractor, "get_customer_by_id", CUSTOMER, "c1")
    observe_read(extractor, "get_bills_for_customer", [BILL_OVERDUE], "b1")
    action = proposed("send_payment_request", customer_id="C1", bill_id="B1")
    assert governor.evaluate(action, extractor.state).decision is Decision.ALLOW

    # Simulate the successful write being executed and observed.
    key = idempotency_key(SESSION, action.tool_name, action.arguments)
    call = ToolCall(id="w1", name=action.tool_name, arguments=action.arguments)
    extractor.observe_agent_message(AssistantMessage(role="assistant", tool_calls=[call]))
    extractor.observe_tool_result(
        ToolMessage(id="w1", role="tool", requestor="assistant", content="Payment request sent"),
        idempotency_key=key,
    )

    result = governor.evaluate(action, extractor.state)
    assert result.decision is Decision.DUPLICATE

    # A different bill is a different operation, not a duplicate.
    other = proposed("send_payment_request", customer_id="C1", bill_id="B9")
    assert governor.evaluate(other, extractor.state).decision is not Decision.DUPLICATE
