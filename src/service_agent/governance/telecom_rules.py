"""Telecom write-gate rules, each derived from a policy line.

Sources (all in third_party/tau2-bench/data/tau2/domains/telecom/, line numbers
at pin cf71a80):

- main_policy.md:9    one tool call at a time, never text + tool together
- main_policy.md:105  check the bill is overdue before payment steps
- main_policy.md:113  verify PAID before telling the user it is paid (read-side)
- main_policy.md:116  only one bill may be AWAITING PAYMENT at a time
- main_policy.md:117  the tool will not check overdue -- the agent must
- main_policy.md:125  resume allowed only after all overdue bills are paid
- main_policy.md:126  never resume if the contract end date is in the past
- main_policy.md:134  refuel at most 2GB
- main_policy.md:136-137  ask the amount, confirm the price before refueling
- main_policy.md:159  identify the customer first
- tools.py:369-370    send_payment_request docstring: does not check PAID
- tools.py:629-630    refuel_data active-line check commented out

Rules use only conversation-derived evidence (see evidence.py). The one
environmental input is the domain's fixed clock (get_today), which the policy
itself discloses to the model ("The current time is ...", main_policy.md:3).

The policy's Change Plan section (main_policy.md:141-149) has no corresponding
agent-side tool at this pin, so there is no rule for it; see DECISIONS.md.
"""

from __future__ import annotations

import datetime
from typing import Optional

from tau2.domains.telecom.utils import get_today

from service_agent.governance.core import (
    Decision,
    GovernanceResult,
    ProposedAction,
    idempotency_key,
)
from service_agent.governance.evidence import EvidenceState

# Per @is_tool(ToolType.WRITE) in tools.py at the pin.
WRITE_TOOLS = frozenset(
    {
        "send_payment_request",
        "resume_line",
        "suspend_line",
        "refuel_data",
        "enable_roaming",
        "disable_roaming",
    }
)

MAX_REFUEL_GB = 2.0  # main_policy.md:134


def _allow(reason_code: str = "allowed") -> GovernanceResult:
    return GovernanceResult(Decision.ALLOW, reason_code, "")


class TelecomGovernor:
    """Adjudicates a candidate action against the current evidence."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    def evaluate(self, action: ProposedAction, evidence: EvidenceState) -> GovernanceResult:
        if action.tool_name not in WRITE_TOOLS:
            # Reads and transfer are never gated: the whole point is that the
            # agent should read more, not less.
            return _allow("read_or_generic")

        key = idempotency_key(self.session_id, action.tool_name, action.arguments)
        if key in evidence.completed_writes:
            return GovernanceResult(
                Decision.DUPLICATE,
                "duplicate_write",
                "This exact operation already succeeded earlier in this session. "
                "Do not execute it again; use the earlier result or verify state "
                "with a read tool.",
                policy_ref="tools.py:369-370",
            )

        handler = getattr(self, f"_check_{action.tool_name}", None)
        if handler is None:
            return _allow("no_specific_rule")
        return handler(action, evidence)

    # -- payment ---------------------------------------------------------------

    def _check_send_payment_request(
        self, action: ProposedAction, evidence: EvidenceState
    ) -> GovernanceResult:
        customer_id = str(action.arguments.get("customer_id", ""))
        bill_id = str(action.arguments.get("bill_id", ""))

        if not evidence.customer_identified(customer_id):
            return GovernanceResult(
                Decision.REQUIRE_EVIDENCE,
                "customer_not_identified",
                "Identify the customer first (look them up by phone number, "
                "customer ID, or full name with date of birth) before sending "
                "a payment request.",
                policy_ref="main_policy.md:159",
                missing_evidence=("customer_record",),
            )

        bill = evidence.bills.get(bill_id)
        if bill is None:
            return GovernanceResult(
                Decision.REQUIRE_EVIDENCE,
                "bill_not_read",
                "Read the customer's bills first and check this bill's status "
                "is Overdue before sending a payment request.",
                policy_ref="main_policy.md:105,117",
                missing_evidence=("bill_status",),
            )

        if bill.get("customer_id") not in (None, customer_id):
            return GovernanceResult(
                Decision.DENY,
                "bill_wrong_customer",
                f"Bill {bill_id} does not belong to customer {customer_id}.",
                policy_ref="tools.py:389-390",
            )

        status = str(bill.get("status", ""))
        if status == "Paid":
            return GovernanceResult(
                Decision.DENY,
                "bill_already_paid",
                f"Bill {bill_id} is already Paid; a paid bill must not receive "
                "another payment request.",
                policy_ref="tools.py:369-370",
            )
        if status != "Overdue":
            return GovernanceResult(
                Decision.DENY,
                "bill_not_overdue",
                f"Bill {bill_id} has status '{status}'. Payment requests are "
                "only for Overdue bills; re-read the bill if you believe its "
                "status changed.",
                policy_ref="main_policy.md:105,117",
            )

        awaiting = [
            b_id
            for b_id, b in evidence.bills.items()
            if b.get("customer_id") == customer_id and b.get("status") == "Awaiting Payment"
        ]
        if awaiting:
            return GovernanceResult(
                Decision.DENY,
                "another_bill_awaiting_payment",
                f"Bill {awaiting[0]} is already Awaiting Payment for this "
                "customer, and only one bill may be awaiting payment at a time.",
                policy_ref="main_policy.md:116",
            )
        return _allow("payment_preconditions_met")

    # -- line suspension -------------------------------------------------------

    def _check_resume_line(
        self, action: ProposedAction, evidence: EvidenceState
    ) -> GovernanceResult:
        customer_id = str(action.arguments.get("customer_id", ""))
        line_id = str(action.arguments.get("line_id", ""))

        line = evidence.lines.get(line_id)
        if line is None:
            return GovernanceResult(
                Decision.REQUIRE_EVIDENCE,
                "line_not_read",
                f"Read line {line_id} first: resuming requires checking its "
                "status and contract end date.",
                policy_ref="main_policy.md:125-126",
                missing_evidence=("line_record",),
            )

        contract_end = _parse_date(line.get("contract_end_date"))
        if contract_end is not None and contract_end < get_today():
            return GovernanceResult(
                Decision.DENY,
                "contract_expired",
                f"Line {line_id}'s contract ended on {contract_end}. The "
                "suspension must not be lifted even if all overdue bills are "
                "paid; transfer if the user cannot be helped otherwise.",
                policy_ref="main_policy.md:126",
            )

        if not evidence.customer_identified(customer_id):
            return GovernanceResult(
                Decision.REQUIRE_EVIDENCE,
                "customer_not_identified",
                "Identify the customer before resuming their line.",
                policy_ref="main_policy.md:159",
                missing_evidence=("customer_record",),
            )

        customer_bills = [
            b for b in evidence.bills.values() if b.get("customer_id") == customer_id
        ]
        if not customer_bills:
            return GovernanceResult(
                Decision.REQUIRE_EVIDENCE,
                "bills_not_read",
                "Read the customer's bills first: a suspension may only be "
                "lifted after all overdue bills are paid.",
                policy_ref="main_policy.md:125",
                missing_evidence=("bill_statuses",),
            )
        overdue = [b for b in customer_bills if b.get("status") == "Overdue"]
        if overdue:
            return GovernanceResult(
                Decision.DENY,
                "overdue_bills_unpaid",
                f"Customer still has overdue bill(s): "
                f"{', '.join(sorted(str(b.get('bill_id')) for b in overdue))}. "
                "All overdue bills must be paid before the suspension is lifted.",
                policy_ref="main_policy.md:125",
            )
        return _allow("resume_preconditions_met")

    def _check_suspend_line(
        self, action: ProposedAction, evidence: EvidenceState
    ) -> GovernanceResult:
        line_id = str(action.arguments.get("line_id", ""))
        if evidence.lines.get(line_id) is None:
            return GovernanceResult(
                Decision.REQUIRE_EVIDENCE,
                "line_not_read",
                f"Read line {line_id} before suspending it.",
                policy_ref="main_policy.md:119-123",
                missing_evidence=("line_record",),
            )
        return _allow("suspend_preconditions_met")

    # -- data refueling --------------------------------------------------------

    def _check_refuel_data(
        self, action: ProposedAction, evidence: EvidenceState
    ) -> GovernanceResult:
        line_id = str(action.arguments.get("line_id", ""))
        try:
            gb_amount = float(action.arguments.get("gb_amount", 0.0))
        except (TypeError, ValueError):
            gb_amount = 0.0

        if not 0.0 < gb_amount <= MAX_REFUEL_GB:
            return GovernanceResult(
                Decision.DENY,
                "refuel_amount_out_of_bounds",
                f"Refuel amount must be between 0 and {MAX_REFUEL_GB} GB; "
                f"got {gb_amount}.",
                policy_ref="main_policy.md:134",
            )

        line = evidence.lines.get(line_id)
        plan = evidence.plans.get(str(line.get("plan_id"))) if line else None
        price_per_gb = _as_float(plan.get("data_refueling_price_per_gb")) if plan else None
        if price_per_gb is None:
            return GovernanceResult(
                Decision.REQUIRE_EVIDENCE,
                "refuel_price_unknown",
                "Read the line and its plan first: you must quote the plan's "
                "refueling price and get the user's confirmation before "
                "refueling.",
                policy_ref="main_policy.md:136-137",
                missing_evidence=("plan_refueling_price",),
            )

        expected_price = round(gb_amount * price_per_gb, 2)
        if not evidence.has_confirmed_amount(gb_amount, expected_price):
            return GovernanceResult(
                Decision.REQUIRE_CONFIRMATION,
                "price_not_confirmed",
                f"Quote the exact price to the user (adding {gb_amount} GB at "
                f"${price_per_gb}/GB costs ${expected_price:.2f}) and wait for "
                "their explicit confirmation before refueling.",
                policy_ref="main_policy.md:136-137",
            )
        return _allow("refuel_confirmed")


def _parse_date(value) -> Optional[datetime.date]:
    if value is None:
        return None
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
