"""Evidence extraction mechanics and bounded recovery."""

import json

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from service_agent.governance import (
    ErrorClass,
    EvidenceExtractor,
    RecoveryBudget,
    classify_tool_error,
)


def _pair(extractor, tool_name, payload, call_id, error=False):
    call = ToolCall(id=call_id, name=tool_name, arguments={})
    extractor.observe_agent_message(AssistantMessage(role="assistant", tool_calls=[call]))
    content = payload if isinstance(payload, str) else json.dumps(payload)
    extractor.observe_tool_result(
        ToolMessage(id=call_id, role="tool", requestor="assistant", content=content, error=error)
    )


def test_results_joined_to_calls_by_id():
    ex = EvidenceExtractor()
    _pair(ex, "get_customer_by_id", {"customer_id": "C1", "bill_ids": []}, "a")
    assert "C1" in ex.state.customers
    assert ex.state.pending_calls == {}


def test_error_results_produce_no_evidence():
    ex = EvidenceExtractor()
    _pair(ex, "get_customer_by_id", "Error: Customer with ID C9 not found", "a", error=True)
    assert ex.state.customers == {}


def test_partial_read_merges_instead_of_clobbering():
    # get_data_usage returns a partial dict keyed by line_id; it must not
    # erase status/contract_end_date learned from an earlier full read.
    ex = EvidenceExtractor()
    _pair(
        ex,
        "get_details_by_id",
        {"line_id": "L1", "status": "Active", "contract_end_date": "2026-01-01"},
        "a",
    )
    _pair(ex, "get_data_usage", {"line_id": "L1", "data_used_gb": 9.5}, "b")
    line = ex.state.lines["L1"]
    assert line["status"] == "Active"
    assert line["data_used_gb"] == 9.5


def test_customer_list_lookup_records_all():
    ex = EvidenceExtractor()
    _pair(
        ex,
        "get_customer_by_name",
        [{"customer_id": "C1"}, {"customer_id": "C2"}],
        "a",
    )
    assert set(ex.state.customers) == {"C1", "C2"}


def test_resume_result_records_nested_line():
    ex = EvidenceExtractor()
    _pair(
        ex,
        "resume_line",
        {"message": "Line resumed successfully", "line": {"line_id": "L1", "status": "Active"}},
        "a",
    )
    assert ex.state.lines["L1"]["status"] == "Active"


def test_confirmation_only_binds_to_next_user_message():
    ex = EvidenceExtractor()
    ex.observe_agent_message(
        AssistantMessage(role="assistant", content="2 GB will cost $8.00. Proceed?")
    )
    # The user asks a question instead of confirming: quote expires.
    ex.observe_user_message(UserMessage(role="user", content="Is that per month?"))
    ex.observe_user_message(UserMessage(role="user", content="yes"))
    assert not ex.state.has_confirmed_amount(2.0, 8.0)


def test_confirmation_binds_amount_and_price():
    ex = EvidenceExtractor()
    ex.observe_agent_message(
        AssistantMessage(role="assistant", content="Refueling 1.5 GB at $4/GB is $6.00.")
    )
    ex.observe_user_message(UserMessage(role="user", content="sounds good"))
    assert ex.state.has_confirmed_amount(1.5, 6.0)
    assert not ex.state.has_confirmed_amount(2.0, 8.0)


# -- recovery ------------------------------------------------------------------


def test_error_classification():
    assert (
        classify_tool_error("Error: Customer with ID C9 not found")
        is ErrorClass.INVALID_ARGUMENT
    )
    assert (
        classify_tool_error("Error: Line must be active to suspend")
        is ErrorClass.PRECONDITION_FAILED
    )
    assert (
        classify_tool_error("Error: A bill is already awaiting payment for this customer")
        is ErrorClass.PRECONDITION_FAILED
    )
    assert classify_tool_error("Error: something odd") is ErrorClass.UNKNOWN


def test_recovery_budget_exhausts():
    budget = RecoveryBudget(max_recoveries=2)
    assert budget.spend(ErrorClass.INVALID_ARGUMENT)
    assert budget.spend(ErrorClass.UNKNOWN)
    assert not budget.spend(ErrorClass.PRECONDITION_FAILED)
    assert budget.exhausted
    assert budget.by_class == {"invalid_argument": 1, "unknown": 1, "precondition_failed": 1}
