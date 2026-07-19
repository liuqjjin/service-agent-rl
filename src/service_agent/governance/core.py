"""Decision vocabulary and idempotency keys."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_EVIDENCE = "require_evidence"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DUPLICATE = "duplicate"
    TRANSFER = "transfer"


@dataclass(frozen=True)
class ProposedAction:
    """A candidate tool call, before anyone has agreed to execute it."""

    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str = ""

    @classmethod
    def from_tool_call(cls, tool_call: Any) -> "ProposedAction":
        return cls(
            tool_name=tool_call.name,
            arguments=dict(tool_call.arguments or {}),
            tool_call_id=tool_call.id,
        )


@dataclass(frozen=True)
class GovernanceResult:
    decision: Decision
    reason_code: str
    # Written for the model: it is fed back verbatim as private guidance when
    # the candidate is rejected, so it should say what to do, not just what
    # went wrong.
    guidance: str
    policy_ref: str = ""
    missing_evidence: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def idempotency_key(session_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
    """One key per business write. Two calls with the same tool and canonicalized
    arguments in the same session are the same operation: executing the second
    one would double the side effect (the PAID-bill warning and the refuel
    double-charge are exactly this failure)."""
    payload = f"{session_id}|{tool_name}|{canonical_json(arguments)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
