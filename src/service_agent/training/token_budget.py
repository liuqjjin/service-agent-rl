"""Measure the final vLLM context limit from committed dev trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from service_agent.training.contracts import CHAT_TEMPLATE_KWARGS

GOVERNANCE_FEEDBACK_BUFFER = 512


def _template_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise TypeError("tool-call arguments must be a JSON object")
    return {
        "id": call.get("id", ""),
        "type": "function",
        "function": {
            "name": call["name"],
            # ART normalizes OpenAI's JSON string to a mapping before applying
            # Qwen's template, which iterates over tool_call.arguments|items.
            "arguments": dict(arguments),
        },
    }


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        out["content"] = message["content"]
    if message.get("tool_calls"):
        out["tool_calls"] = [_template_tool_call(call) for call in message["tool_calls"]]
    return out


def _agent_visible(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    if role == "assistant":
        return _assistant_message(message)
    if role == "user" and not message.get("tool_calls"):
        return {"role": "user", "content": message.get("content") or ""}
    if role == "tool" and message.get("requestor") == "assistant":
        return {
            "role": "tool",
            "tool_call_id": message.get("id", ""),
            "content": message.get("content") or "",
        }
    return None


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def measure_dev_token_budget(
    *,
    repo_root: Path,
    tokenizer: Any,
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Tokenize every model-generation prefix in all committed dev arms."""

    prompt_lengths: list[int] = []
    simulations = 0
    for arm in ("h0", "h1", "h2"):
        payload = json.loads((repo_root / f"results/dev/{arm}/results.json").read_text())
        for simulation in payload["simulations"]:
            simulations += 1
            history: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
            for index, message in enumerate(simulation["messages"]):
                visible = _agent_visible(message)
                if visible is None:
                    continue
                # The native orchestrator's fixed greeting is trajectory data,
                # not LLMAgent state, so the policy never receives it.
                if (
                    index == 0
                    and visible["role"] == "assistant"
                    and visible.get("content") == "Hi! How can I help you today?"
                ):
                    continue
                if visible["role"] == "assistant":
                    token_ids = tokenizer.apply_chat_template(
                        history,
                        tools=tools,
                        add_generation_prompt=True,
                        **CHAT_TEMPLATE_KWARGS,
                    )
                    prompt_lengths.append(len(token_ids))
                history.append(visible)

    if not prompt_lengths:
        raise RuntimeError("no policy-generation prefixes found in committed dev results")
    maximum = max(prompt_lengths)
    return {
        "source": "results/dev/{h0,h1,h2}/results.json",
        "simulations": simulations,
        "generation_prefixes": len(prompt_lengths),
        "prompt_tokens_p50": _percentile(prompt_lengths, 0.50),
        "prompt_tokens_p95": _percentile(prompt_lengths, 0.95),
        "prompt_tokens_p99": _percentile(prompt_lengths, 0.99),
        "prompt_tokens_max": maximum,
        "governance_feedback_buffer": GOVERNANCE_FEEDBACK_BUFFER,
        "required_prompt_capacity": maximum + GOVERNANCE_FEEDBACK_BUFFER,
    }


def validate_context_capacity(
    report: dict[str, Any],
    *,
    max_model_len: int,
    max_completion_tokens: int,
) -> None:
    available = max_model_len - max_completion_tokens
    required = int(report["required_prompt_capacity"])
    if available < required:
        raise RuntimeError(
            "configured context is smaller than committed dev evidence plus "
            f"governance buffer: {available} < {required}"
        )
