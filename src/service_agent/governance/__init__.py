"""Execution governance: deterministic adjudication of candidate tool calls.

A model's tool call is an intent, not an authorization. The telecom write
tools execute whatever they are asked (see UPSTREAM.md: send_payment_request
does not check PAID, refuel_data's active check is commented out), while the
policy places the business preconditions on the agent. This package closes
that gap in code: every candidate write is checked against evidence gathered
from the conversation itself before it is allowed to execute.
"""

from service_agent.governance.core import (
    Decision,
    GovernanceResult,
    ProposedAction,
    idempotency_key,
)
from service_agent.governance.evidence import EvidenceExtractor, EvidenceState
from service_agent.governance.recovery import ErrorClass, RecoveryBudget, classify_tool_error
from service_agent.governance.telecom_rules import WRITE_TOOLS, TelecomGovernor

__all__ = [
    "Decision",
    "GovernanceResult",
    "ProposedAction",
    "idempotency_key",
    "EvidenceExtractor",
    "EvidenceState",
    "ErrorClass",
    "RecoveryBudget",
    "classify_tool_error",
    "TelecomGovernor",
    "WRITE_TOOLS",
]
