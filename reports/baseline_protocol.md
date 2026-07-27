# Baseline protocol

What is held fixed for every experimental cell, and why. Numbers land in the
companion reports; this file is the contract they are produced under.

## Fixed components

| Component | Value | Notes |
|---|---|---|
| Environment | tau2-bench telecom @ `cf71a80` | submodule pin; native runner only |
| User simulator | DeepSeek official API, `deepseek-v4-pro` | non-thinking (`extra_body.thinking.type=disabled`), temperature 0.0 |
| Simulator fallback | DashScope `qwen3.7-max-2026-06-08` | dev-only sensitivity checks |
| Policy model (dev ablation) | Qwen, served locally via mlx_lm.server | temperature 0.0, thinking disabled via `chat_template_kwargs` |
| Policy model (final 2x2) | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | served by vLLM on the training box; base and RL cells share checkpoint, tokenizer, chat template, tool parser, and stack |
| Trials | dev: 4 per task; final test: 8 per task | pass^k needs `trials >= k` |
| Seed | 42 (per-trial seeds derived by the native runner) | `batch.py` seeds trials, orchestrator seeds agent+user |
| Max steps | 100 | `termination_reason` reported separately, never folded into accuracy |
| Concurrency | limited by the local model server | recorded per run in `run_config.json` |

## Arms

- **H0** — native `llm_agent`, untouched.
- **H1** — `GovernedLLMAgent` with the precondition gate only.
- **H2** — H1 + idempotency ledger + bounded recovery.

RL training happens in the H0 environment only; governance appears in
evaluation cells (see CLAUDE.md hard rule 4).

## Measurement

Official metrics come from tau2's `compute_metrics` (avg reward, pass^k,
costs, termination reasons). Governance metrics come from replaying every
arm's trajectories through the same rule code that gates H1/H2 live
(`service_agent.eval.metrics.analyze_trajectory`): unauthorized executed
writes, duplicate side effects, denial reasons. One yardstick for every arm,
so H0's violation rate and H1/H2's prevention are directly comparable. H1/H2
additionally report the live gate's audit (candidates rejected before ever
reaching the trajectory, regeneration counts) — behavior invisible to
trajectory replay by design.

## Honest caveats

- The dev-ablation policy stack (MLX, quantized) is not the final-eval stack
  (vLLM, bf16). Dev results select Hbest; every number in the final 2x2 table
  is produced on the final stack.
- The dev set steers all selection; the official test split is touched once,
  at the end, with the four frozen cells.
- Simulator identity is part of the benchmark definition: all cells share one
  simulator build, and no number is compared across simulator changes.
