"""Evidence: what the conversation has actually established.

The governor may only use information the model itself could have seen --
the policy, tool schemas, tool results, and what the user said. It must not
peek at task labels or hidden environment state; a gate that cheats is not
a harness result, it is leakage (the same rule the leak tests enforce for
prompts). This module turns the message stream into that evidence.

Structure notes for the parser:
- ToolMessage carries only the tool_call_id, so calls are recorded when the
  assistant message is observed and joined with their results by id.
- Tool results are JSON produced by Environment.to_json_str over pydantic
  dumps: bills have bill_id/status/customer_id, lines have line_id/status/
  contract_end_date/plan_id, plans have plan_id/data_refueling_price_per_gb.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Quote:
    """Amounts the agent stated in one message: the raw material of a price
    confirmation (main_policy.md:136-137 requires asking the amount and
    confirming the price before refueling)."""

    dollars: tuple[float, ...]
    gigabytes: tuple[float, ...]


@dataclass
class EvidenceState:
    customers: dict[str, dict[str, Any]] = field(default_factory=dict)
    bills: dict[str, dict[str, Any]] = field(default_factory=dict)
    lines: dict[str, dict[str, Any]] = field(default_factory=dict)
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    # tool_call_id -> (tool_name, arguments), for joining results to calls
    pending_calls: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    # Quote awaiting the user's next message
    pending_quote: Optional[Quote] = None
    # Quotes the user affirmed
    confirmed_quotes: list[Quote] = field(default_factory=list)
    # idempotency_key -> recorded tool result content
    completed_writes: dict[str, str] = field(default_factory=dict)

    def customer_identified(self, customer_id: str) -> bool:
        return customer_id in self.customers

    def has_confirmed_amount(self, gigabytes: float, dollars: float) -> bool:
        """Was a quote containing this GB amount and this price affirmed?"""
        for quote in self.confirmed_quotes:
            gb_ok = any(abs(g - gigabytes) < 1e-6 for g in quote.gigabytes)
            price_ok = any(abs(d - dollars) < 0.005 for d in quote.dollars)
            if gb_ok and price_ok:
                return True
        return False


_DOLLARS = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
_GIGABYTES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:GB|gigabytes?)\b", re.IGNORECASE)
_AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|sure|ok|okay|confirm|confirmed|go ahead|please do|do it|"
    r"proceed|sounds good|that works|that's fine|i agree|absolutely)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(no|not|don't|do not|wait|hold|instead|actually|rather|cancel|"
    r"nevermind|never mind)\b",
    re.IGNORECASE,
)


class EvidenceExtractor:
    """Feeds messages into an EvidenceState. Call in conversation order."""

    def __init__(self, state: Optional[EvidenceState] = None):
        self.state = state or EvidenceState()

    # -- assistant ------------------------------------------------------------

    def observe_agent_message(self, message: Any) -> None:
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                self.state.pending_calls[tc.id] = (tc.name, dict(tc.arguments or {}))
            return
        text = getattr(message, "content", None) or ""
        dollars = tuple(float(m) for m in _DOLLARS.findall(text))
        gigabytes = tuple(float(m) for m in _GIGABYTES.findall(text))
        if dollars and gigabytes:
            self.state.pending_quote = Quote(dollars=dollars, gigabytes=gigabytes)

    # -- user -----------------------------------------------------------------

    def observe_user_message(self, message: Any) -> None:
        if getattr(message, "tool_calls", None):
            return
        text = getattr(message, "content", None) or ""
        pending = self.state.pending_quote
        if pending is not None:
            # Only the user message immediately after the quote can confirm it,
            # and hedged affirmatives ("yes but...", "ok wait") do not count.
            if _AFFIRMATIVE.search(text) and not _NEGATION.search(text):
                self.state.confirmed_quotes.append(pending)
            self.state.pending_quote = None

    # -- environment ----------------------------------------------------------

    def observe_tool_result(self, tool_message: Any, idempotency_key: str = "") -> None:
        call = self.state.pending_calls.pop(tool_message.id, None)
        if call is None or getattr(tool_message, "error", False):
            return
        tool_name, _arguments = call
        payload = _parse(tool_message.content)
        self._route(tool_name, payload)
        if idempotency_key:
            self.state.completed_writes[idempotency_key] = tool_message.content or ""

    def _route(self, tool_name: str, payload: Any) -> None:
        if payload is None:
            return
        if tool_name in ("get_customer_by_phone", "get_customer_by_id"):
            self._record(payload)
        elif tool_name == "get_customer_by_name":
            for item in _as_list(payload):
                self._record(item)
        elif tool_name == "get_bills_for_customer":
            for item in _as_list(payload):
                self._record(item)
        elif tool_name in ("get_details_by_id", "get_data_usage"):
            self._record(payload)
        elif tool_name in ("suspend_line", "resume_line"):
            # These return {"message": ..., "line": {...}} with the updated line
            if isinstance(payload, dict):
                self._record(payload.get("line"))

    def _record(self, obj: Any) -> None:
        """File an object under the right index based on its identifying key.

        Records are merged, not replaced: get_data_usage returns a partial
        dict keyed by line_id, and a later read must not erase the fields an
        earlier one established (status, contract_end_date, ...)."""
        if not isinstance(obj, dict):
            return
        for key, index in (
            ("customer_id", self.state.customers),
            ("bill_id", self.state.bills),
            ("line_id", self.state.lines),
            ("plan_id", self.state.plans),
        ):
            if key == "customer_id" and "bill_id" in obj:
                continue  # bills carry customer_id too; index them as bills
            if key in obj and isinstance(obj.get(key), str):
                index[obj[key]] = {**index.get(obj[key], {}), **obj}
                return


def _parse(content: Optional[str]) -> Any:
    if not content:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def _as_list(payload: Any) -> list:
    return payload if isinstance(payload, list) else []
