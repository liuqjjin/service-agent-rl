"""Step-0 logprob consistency between the rollout server and the trainer.

Before the first parameter update, the importance ratio
exp(logprob_trainer - logprob_rollout) should be ~1.0 for every sampled
token: same weights, so any gap is a tokenization, chat-template, or
numerics mismatch -- and PPO/GRPO clipping will silently mangle updates
built on it. If this check fails, fix the template; do not touch the
learning rate (planning docs, and every hard-won war story in agentic RL).

Method: sample completions from the vLLM OpenAI server that will serve
rollouts, with logprobs and token ids enabled; rebuild the exact same token
sequence locally via tokenizer.apply_chat_template; run one forward pass of
the same weights in HF transformers; compare per-token logprobs of the
completion segment.

Prompts are built from the real telecom policy and tool schemas so the check
covers the token distribution training will actually see (system prompt with
tool definitions is where template mismatches hide).

Run on the training box (needs torch + transformers + the vLLM server up):

    uv run python -m service_agent.training.logprob_check \
        --api-base http://127.0.0.1:8300/v1 --served-model Qwen/Qwen3.5-4B \
        --hf-model Qwen/Qwen3.5-4B

Exit code 1 on failure; thresholds are printed with the report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--n-prompts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ratio-mean-tol", type=float, default=0.02)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-clip-fraction", type=float, default=0.02)
    parser.add_argument(
        "--chat-template-kwargs",
        default='{"enable_thinking": false}',
        help="JSON; must match what the rollout server applies",
    )
    return parser.parse_args()


def build_probe_messages(n: int) -> list[list[dict]]:
    """Real telecom prompt surfaces: policy system prompt + short dialogues."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.registry import registry

    env = registry.get_env_constructor("telecom")()
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="probe")
    system = {"role": "system", "content": agent.system_prompt}
    users = [
        "Hi, my phone says No Service and I need it fixed today.",
        "I want to pay my overdue bill, my customer id is C1001.",
        "My mobile data is very slow when I travel abroad.",
        "Can you add 2 GB of data to my line?",
        "I can't send picture messages since yesterday.",
        "My line got suspended, please turn it back on.",
    ]
    return [[system, {"role": "user", "content": u}] for u in users[:n]]


def openai_tool_schemas() -> list[dict]:
    # Same conversion tau2's generate() applies (llm_utils.py:389).
    from tau2.registry import registry

    env = registry.get_env_constructor("telecom")()
    return [tool.openai_schema for tool in env.get_tools()]


def sample_with_logprobs(args, messages: list[dict], tools: list[dict]):
    from openai import OpenAI

    client = OpenAI(base_url=args.api_base, api_key="local")
    response = client.chat.completions.create(
        model=args.served_model,
        messages=messages,
        tools=tools,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        logprobs=True,
        extra_body={
            # vLLM: report tokens as "token_id:<int>" so we get exact ids,
            # immune to detokenization artifacts.
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": json.loads(args.chat_template_kwargs),
        },
    )
    choice = response.choices[0]
    content = choice.logprobs.content or []
    token_ids, logprobs = [], []
    for entry in content:
        token = entry.token
        if token.startswith("token_id:"):
            token_ids.append(int(token.split(":", 1)[1]))
        else:
            raise RuntimeError(
                "server did not honor return_tokens_as_token_ids; "
                "exact-id comparison is impossible"
            )
        logprobs.append(entry.logprob)
    return token_ids, logprobs


def local_logprobs(tokenizer, model, prompt_ids, completion_ids, device):
    import torch

    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    logsoft = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for i, token_id in enumerate(completion_ids):
        # Token at position p is predicted by logits at p-1.
        pos = len(prompt_ids) + i - 1
        out.append(logsoft[pos, token_id].item())
    return out


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    tools = openai_tool_schemas()
    template_kwargs = json.loads(args.chat_template_kwargs)

    deltas, ratios = [], []
    for messages in build_probe_messages(args.n_prompts):
        completion_ids, rollout_lps = sample_with_logprobs(args, messages, tools)
        if not completion_ids:
            continue
        prompt_ids = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, **template_kwargs
        )
        trainer_lps = local_logprobs(tokenizer, model, prompt_ids, completion_ids, device)
        for rollout_lp, trainer_lp in zip(rollout_lps, trainer_lps):
            deltas.append(abs(trainer_lp - rollout_lp))
            ratios.append(math.exp(trainer_lp - rollout_lp))

    if not ratios:
        print("no tokens sampled; check the server")
        sys.exit(1)

    ratios_sorted = sorted(ratios)

    def pct(p: float) -> float:
        return ratios_sorted[min(len(ratios_sorted) - 1, int(p * len(ratios_sorted)))]

    lo, hi = 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon
    clip_fraction = sum(1 for r in ratios if r < lo or r > hi) / len(ratios)
    report = {
        "tokens": len(ratios),
        "ratio_mean": statistics.fmean(ratios),
        "ratio_median": pct(0.5),
        "ratio_p95": pct(0.95),
        "ratio_p99": pct(0.99),
        "mean_abs_logprob_delta": statistics.fmean(deltas),
        "clip_fraction_before_first_update": clip_fraction,
        "thresholds": {
            "abs(ratio_mean - 1)": args.ratio_mean_tol,
            "clip_fraction": args.max_clip_fraction,
            "clip_window": [lo, hi],
        },
    }
    print(json.dumps(report, indent=2))

    ok = (
        abs(report["ratio_mean"] - 1.0) <= args.ratio_mean_tol
        and clip_fraction <= args.max_clip_fraction
    )
    print("PASS" if ok else "FAIL: fix chat template/tokenization before training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
