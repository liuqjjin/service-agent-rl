"""Bounded recovery: classify tool failures and cap the retry budget.

The telecom tools fail by raising ValueError, which the environment converts
to a ToolMessage with error=True and content "Error: <message>"
(environment.py:465-490). Only two failure families actually occur in this
in-memory environment -- bad identifiers and unmet state preconditions --
so that is what we classify. Transport-level failures (429/timeouts) belong
to the LLM client layer, not tool execution, and are handled there by
LiteLLM's retries.

The classification decides what a sane retry looks like:
- INVALID_ARGUMENT: one corrected attempt is reasonable (typo'd ID).
- PRECONDITION_FAILED: retrying the same call cannot succeed; the agent must
  read state or do something else first. Retrying around a precondition is
  exactly the failure mode governance exists to stop.
- UNKNOWN: treat like INVALID_ARGUMENT but count it against the budget.

The budget is session-global: an agent burning attempt after attempt is the
max-steps death spiral; past the cap the right move is to transfer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ErrorClass(str, Enum):
    INVALID_ARGUMENT = "invalid_argument"
    PRECONDITION_FAILED = "precondition_failed"
    UNKNOWN = "unknown"


_PRECONDITION_PATTERNS = re.compile(
    r"must be (active|suspended|positive)|already (has|awaiting|an overdue)|"
    r"already awaiting payment",
    re.IGNORECASE,
)
_INVALID_ARGUMENT_PATTERNS = re.compile(
    r"not found|unknown id|invalid|format", re.IGNORECASE
)


def classify_tool_error(content: str) -> ErrorClass:
    text = content or ""
    if _PRECONDITION_PATTERNS.search(text):
        return ErrorClass.PRECONDITION_FAILED
    if _INVALID_ARGUMENT_PATTERNS.search(text):
        return ErrorClass.INVALID_ARGUMENT
    return ErrorClass.UNKNOWN


@dataclass
class RecoveryBudget:
    max_recoveries: int = 4
    used: int = 0
    by_class: dict[str, int] = field(default_factory=dict)

    def spend(self, error_class: ErrorClass) -> bool:
        """Record a recovery attempt. Returns False when the budget is gone
        and the agent should transfer instead of trying again."""
        self.used += 1
        self.by_class[error_class.value] = self.by_class.get(error_class.value, 0) + 1
        return self.used <= self.max_recoveries

    @property
    def exhausted(self) -> bool:
        return self.used > self.max_recoveries
