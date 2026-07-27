"""Step-0 token and logprob parity for ART's actual rollout server.

The gate samples from the registered ART vLLM endpoint before any update and
captures the exact prompt/completion token IDs returned by ART's runtime. The
backend is then closed before the bf16 Hugging Face reference is loaded, so a
48 GB card never holds the vLLM KV cache and the reference model together.
Prompt IDs must match exactly; completion logprobs must pass the configured
importance-ratio thresholds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any

from service_agent.training.contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHAT_TEMPLATE_KWARGS,
)


@dataclass(frozen=True)
class ProbeRecord:
    messages: list[dict[str, str]]
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    rollout_logprobs: list[float]


def build_probe_messages(system_prompt: str, n: int) -> list[list[dict[str, str]]]:
    users = [
        "Hi, my phone says No Service and I need it fixed today.",
        "I want to pay my overdue bill, my customer id is C1001.",
        "My mobile data is very slow when I travel abroad.",
        "Can you add 2 GB of data to my line?",
        "I can't send picture messages since yesterday.",
        "My line got suspended, please turn it back on.",
    ]
    system = {"role": "system", "content": system_prompt}
    return [[system, {"role": "user", "content": user}] for user in users[:n]]


def _choice_token_ids(response: Any, choice: Any) -> tuple[list[int], list[int]]:
    extra = choice.model_extra or {}
    prompt_ids = extra.get("prompt_token_ids")
    completion_ids = extra.get("token_ids")
    if prompt_ids is None:
        prompt_ids = (response.model_extra or {}).get("prompt_token_ids")
    if completion_ids is None:
        raw_choices = (response.model_extra or {}).get("choices") or []
        if raw_choices:
            completion_ids = raw_choices[0].get("token_ids")
    if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
        raise RuntimeError("ART vLLM response omitted exact prompt/completion token IDs")
    return [int(token) for token in prompt_ids], [int(token) for token in completion_ids]


def _choice_logprobs(choice: Any) -> list[float]:
    logprobs = choice.logprobs
    if logprobs is None:
        raise RuntimeError("ART vLLM response omitted completion logprobs")
    entries = list(logprobs.content or []) + list(logprobs.refusal or [])
    return [float(entry.logprob) for entry in entries]


async def sample_probe_records(
    *,
    openai_client: Any,
    served_model: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
    n_prompts: int = 6,
    max_tokens: int = 128,
    temperature: float = 1.0,
    seed: int = 42,
) -> list[ProbeRecord]:
    """Capture rollout-side IDs and logprobs while ART vLLM is live."""

    records: list[ProbeRecord] = []
    for index, messages in enumerate(build_probe_messages(system_prompt, n_prompts)):
        response = await openai_client.chat.completions.create(
            model=served_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed + index,
            logprobs=True,
        )
        choice = response.choices[0]
        prompt_ids, completion_ids = _choice_token_ids(response, choice)
        rollout_logprobs = _choice_logprobs(choice)
        if len(completion_ids) != len(rollout_logprobs):
            raise RuntimeError(
                "completion token/logprob length mismatch: "
                f"{len(completion_ids)} IDs vs {len(rollout_logprobs)} logprobs"
            )
        if completion_ids:
            records.append(
                ProbeRecord(
                    messages=messages,
                    prompt_token_ids=prompt_ids,
                    completion_token_ids=completion_ids,
                    rollout_logprobs=rollout_logprobs,
                )
            )
    if not records:
        raise RuntimeError("step-0 server produced no probe tokens")
    return records


def _local_logprobs(model: Any, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
    import torch

    all_ids = prompt_ids + completion_ids
    input_ids = torch.tensor([all_ids], device="cuda")
    keep = len(completion_ids) + 1
    with torch.no_grad():
        try:
            logits = model(input_ids, logits_to_keep=keep).logits[0]
        except TypeError:
            logits = model(input_ids).logits[0, -keep:]
    # The first retained position is prompt[-1], which predicts completion[0].
    logits = logits[: len(completion_ids)].float()
    target = torch.tensor(completion_ids, device=logits.device)
    selected = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    values = selected - torch.logsumexp(logits, dim=-1)
    return [float(value) for value in values.cpu().tolist()]


def evaluate_probe_records(
    records: list[ProbeRecord],
    *,
    tools: list[dict[str, Any]],
    ratio_mean_tol: float = 0.02,
    clip_epsilon: float = 0.2,
    max_clip_fraction: float = 0.02,
) -> dict[str, Any]:
    """Load the pinned bf16 reference after vLLM closes and evaluate the gate."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("step-0 logprob gate requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("step-0 logprob gate requires bf16 support")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()

    exact_prompt_ids = True
    deltas: list[float] = []
    ratios: list[float] = []
    try:
        for record in records:
            local_prompt_ids = tokenizer.apply_chat_template(
                record.messages,
                tools=tools,
                add_generation_prompt=True,
                **CHAT_TEMPLATE_KWARGS,
            )
            local_prompt_ids = [int(token) for token in local_prompt_ids]
            if local_prompt_ids != record.prompt_token_ids:
                exact_prompt_ids = False
                continue
            trainer_logprobs = _local_logprobs(
                model,
                record.prompt_token_ids,
                record.completion_token_ids,
            )
            for rollout_lp, trainer_lp in zip(
                record.rollout_logprobs,
                trainer_logprobs,
                strict=True,
            ):
                delta = trainer_lp - rollout_lp
                deltas.append(abs(delta))
                ratios.append(math.exp(delta))
    finally:
        del model
        torch.cuda.empty_cache()

    if not ratios:
        return {
            "status": "failed",
            "prompt_token_ids_exact": exact_prompt_ids,
            "tokens": 0,
            "reason": "no comparable tokens",
        }

    ordered = sorted(ratios)

    def percentile(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    lo, hi = 1.0 - clip_epsilon, 1.0 + clip_epsilon
    clip_fraction = sum(ratio < lo or ratio > hi for ratio in ratios) / len(ratios)
    ratio_mean = statistics.fmean(ratios)
    passed = (
        exact_prompt_ids
        and abs(ratio_mean - 1.0) <= ratio_mean_tol
        and clip_fraction <= max_clip_fraction
    )
    return {
        "status": "passed" if passed else "failed",
        "prompt_token_ids_exact": exact_prompt_ids,
        "prompts": len(records),
        "tokens": len(ratios),
        "ratio_mean": ratio_mean,
        "ratio_median": percentile(0.5),
        "ratio_p95": percentile(0.95),
        "ratio_p99": percentile(0.99),
        "mean_abs_logprob_delta": statistics.fmean(deltas),
        "clip_fraction_before_first_update": clip_fraction,
        "thresholds": {
            "abs_ratio_mean_minus_one": ratio_mean_tol,
            "clip_fraction": max_clip_fraction,
            "clip_window": [lo, hi],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--contract-json", required=True)
    parser.add_argument("--n-prompts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


async def _cli(args: argparse.Namespace) -> dict[str, Any]:
    from openai import AsyncOpenAI

    contract = json.loads(args.contract_json)
    client = AsyncOpenAI(base_url=args.api_base, api_key="local")
    try:
        records = await sample_probe_records(
            openai_client=client,
            served_model=args.served_model,
            system_prompt=contract["system_prompt"],
            tools=contract["tools"],
            n_prompts=args.n_prompts,
            max_tokens=args.max_tokens,
        )
    finally:
        await client.close()
    return evaluate_probe_records(records, tools=contract["tools"])


def main() -> None:
    report = asyncio.run(_cli(parse_args()))
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
