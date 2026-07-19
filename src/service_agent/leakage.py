"""Label-leak detection for anything that reaches a model prompt.

The dangerous fields on a tau2 Task, in decreasing order of blast radius:

- evaluation_criteria: the reference action trajectory -- the literal answer key.
- initial_state: the injected faults, i.e. the diagnosis the agent is graded on.
- task id: telecom IDs encode the fault chain in plain text
  ("[mms_issue]airplane_mode_on|break_apn_settings|...").
- user_scenario: the user simulator's script, including `unknown_info` (facts the
  user is explicitly not supposed to know) and `task_instructions` (how the user
  will behave, e.g. when they consider the issue resolved).
- description / ticket: annotator metadata and the solo-mode ticket.

Two audit modes, because information legitimately flows during a conversation:

- strict (pre-conversation): nothing task-derived may appear at all. Applied to
  system prompts and initial observations.
- lax (mid-conversation): the user may have voluntarily said things from
  known_info / reason_for_call, so those are exempt; everything else --
  criteria, initial state, id, unknown_info, task_instructions -- must still
  never appear.

Checks match on serialized substrings (str() and JSON forms), not field names,
so a leak through any formatting path still trips the assertion.
"""

from __future__ import annotations

import json
from typing import Any

# Substrings shorter than this are skipped: values like "None" or "easy" would
# false-positive on ordinary prose. Task IDs are always checked regardless.
MIN_MATCH_LEN = 12


class LabelLeakError(AssertionError):
    pass


def _as_dict(task: Any) -> dict:
    if hasattr(task, "model_dump"):
        return task.model_dump(exclude_none=True)
    return task


def _collect(value: Any, label: str, out: list[tuple[str, str]]) -> None:
    """Flatten a task field into checkable substrings: raw strings and JSON blobs."""
    if value is None:
        return
    if isinstance(value, str):
        out.append((label, value))
    elif isinstance(value, (dict, list)):
        out.append((label, json.dumps(value, ensure_ascii=False)))
        if isinstance(value, dict):
            for k, v in value.items():
                _collect(v, f"{label}.{k}", out)
        else:
            for i, v in enumerate(value):
                _collect(v, f"{label}[{i}]", out)
    else:
        out.append((label, str(value)))


def _collect_composite(value: Any, label: str, out: list[tuple[str, str]]) -> None:
    """Composite serializations only: the whole field as JSON plus each list
    element as JSON. Individual leaf values are deliberately not checked --
    reference actions are made of tool names, and tool names legitimately
    appear in every prompt via the policy and tool schemas. A leak of the
    answer key surfaces as a structured blob, not as one bare tool name."""
    if value is None:
        return
    out.append((label, json.dumps(value, ensure_ascii=False)))
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, list):
                for i, item in enumerate(v):
                    out.append((f"{label}.{k}[{i}]", json.dumps(item, ensure_ascii=False)))


def forbidden_strings(task: Any, strict: bool = True) -> list[tuple[str, str]]:
    """(label, substring) pairs that must not appear in model-visible text."""
    d = _as_dict(task)
    out: list[tuple[str, str]] = []

    _collect_composite(d.get("evaluation_criteria"), "evaluation_criteria", out)
    _collect_composite(d.get("initial_state"), "initial_state", out)
    _collect(d.get("description"), "description", out)

    scenario = d.get("user_scenario") or {}
    instructions = scenario.get("instructions")
    if isinstance(instructions, dict):
        _collect(instructions.get("unknown_info"), "user_scenario.unknown_info", out)
        _collect(
            instructions.get("task_instructions"), "user_scenario.task_instructions", out
        )
        if strict:
            _collect(instructions.get("known_info"), "user_scenario.known_info", out)
            _collect(
                instructions.get("reason_for_call"), "user_scenario.reason_for_call", out
            )
    else:
        _collect(instructions, "user_scenario.instructions", out)
    if strict:
        _collect(scenario.get("persona"), "user_scenario.persona", out)
        _collect(d.get("ticket"), "ticket", out)

    # tau2's own __str__ serializers, when a real Task object is passed: cover
    # the exact formatting an accidental str(task) would produce.
    if hasattr(task, "model_dump"):
        out.append(("str(task)", str(task)))
        for attr in ("evaluation_criteria", "initial_state", "user_scenario"):
            value = getattr(task, attr, None)
            if value is not None:
                out.append((f"str(task.{attr})", str(value)))

    out = [(label, s) for label, s in out if len(s) >= MIN_MATCH_LEN]

    # The ID encodes the fault chain; check it and its segments unconditionally.
    task_id = d.get("id") or ""
    if task_id:
        out.append(("task.id", task_id))
        for seg in task_id.replace("[", "|").replace("]", "|").split("|"):
            if len(seg) >= MIN_MATCH_LEN:
                out.append(("task.id.segment", seg))
    return out


def find_leaks(text: str, task: Any, strict: bool = True) -> list[str]:
    return sorted(
        {label for label, s in forbidden_strings(task, strict=strict) if s in text}
    )


def assert_no_leak(text: str, task: Any, strict: bool = True, context: str = "") -> None:
    leaks = find_leaks(text, task, strict=strict)
    if leaks:
        raise LabelLeakError(
            f"Task labels leaked into model-visible text{f' ({context})' if context else ''}: "
            f"{', '.join(leaks)}"
        )
