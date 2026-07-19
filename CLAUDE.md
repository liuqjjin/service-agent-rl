# service-agent

Execution governance + GRPO post-training for LLM agents on tau2-bench telecom (dual-control).
The research question: how much reliability comes from a deterministic governance harness vs.
from RL post-training, measured with a 2x2 factorial (base/RL model x native/governed harness).

Status: base-model row of the 2x2 measured on dev (reports/governance_ablation.md, Hbest = H2).
Shim, training path, logprob gate, SFT bridge, and AutoDL runbook built and tested. Not yet run:
the GRPO training and the final 2x2 on the test split (GPU, runbooks/autodl.md). No real weights
updated yet, so the repo stays named `service-agent`.

## Commands

```bash
uv sync                 # Python 3.12; tau2 installed editable from third_party/tau2-bench
uv run pytest           # our tests only (tests/)
uv run pytest third_party/tau2-bench/tests/test_gym/test_gym.py  # upstream gym tests

# Serve a local policy model (separate uv tool env, not in the project lock)
mlx_lm.server --model /Users/lqj/local-llm/models/Qwen3.6-35B-A3B-4bit --port 8399
mlx_lm.server --model mlx-community/Qwen3.5-4B-MLX-8bit --port 8398

# One ablation arm on the frozen dev set (keys come from .env)
uv run python -m service_agent.eval.run_ablation --arm h1 --tasks smoke3 --trials 1 \
    --agent-llm "openai//Users/lqj/local-llm/models/Qwen3.6-35B-A3B-4bit" \
    --agent-api-base http://127.0.0.1:8399/v1 --out results/dev/h1_smoke
```

Upstream gym tests spawn a real LLM user simulator even for the mock domain: without
`OPENAI_API_KEY` the orchestrator thread dies instantly and 3 tests fail. That is
environmental, not a code bug.

Model-serving traps, all found empirically (details in DECISIONS.md):
- The LiteLLM model string after `openai/` must be the id mlx_lm.server actually
  serves (the exact `--model` value). Asking for any other id makes the server
  silently download that repo from HF instead of using the loaded model.
- Local Qwen agents must disable thinking via
  `extra_body={"chat_template_kwargs": {"enable_thinking": false}}` — otherwise
  long tasks can return think-only (empty) completions that fail tau2's message
  validation and burn all retries.
- deepseek-v4-pro disables thinking only via
  `extra_body={"thinking": {"type": "disabled"}}`; the top-level `thinking`
  parameter does not work through LiteLLM.
- run_ablation handles all three automatically; direct generate() calls must not
  forget them.

## Layout

- `src/service_agent/` — everything we own:
  - `splits.py`, `leakage.py` — data protocol and label-leak detection
  - `governance/` — decision core, evidence extraction, telecom rules (every rule
    cites a policy line), idempotency, bounded recovery, audit trail
  - `agent/governed.py` — GovernedLLMAgent; adjudicates candidates before they
    enter the official trajectory (the evaluator replays trajectory writes, so
    interception anywhere later corrupts the reward)
  - `eval/` — registry arms (h1/h2), offline governance replay metrics (one
    yardstick for every arm incl. H0), run_ablation CLI
- `third_party/tau2-bench` — submodule pinned at `cf71a80` (+ a minimal gym-fix branch, see UPSTREAM.md).
  Never edit outside that fix; a hash-guard test enforces it.
- `third_party/ART` — submodule pinned at `828b839`. Read-only reference for the RL client contract.
  Not installed in the Mac venv; only the AutoDL trainer venv installs it.
- `planning/` — private design manuals (Chinese), gitignored, the source of truth for scope.
- `data_protocol/` — frozen split/dev IDs and hygiene reports. Committed, never regenerated silently.

## Hard rules

These are correctness constraints, not preferences. Breaking any of them invalidates results.

1. Governance must reject a candidate tool call **before** it enters the official trajectory.
   The evaluator replays trajectory writes via `set_state`; a logged-but-unexecuted write
   corrupts the DB-match reward. Rejected candidates go to the audit trace only.
2. Never train on `test`, `full`, or `base` splits (`full`/`base` contain `test`).
   Training uses train-core only; dev (20 frozen IDs) for all selection; test touched once.
3. Model prompts are built by whitelist. `str(task)`, `task.evaluation_criteria`,
   `task.initial_state`, `user_scenario`, and task IDs must never reach a prompt.
   Leak tests assert on serialized substrings, not field names.
4. RL trains in the native H0 environment; governance is layered on only at evaluation.
   Otherwise the model-vs-harness attribution in the 2x2 is confounded.
5. Fixed user simulator: official DeepSeek API `deepseek-v4-pro`, non-thinking, temperature 0,
   identical parameters across every cell. Fallback (recorded in DECISIONS.md):
   DashScope `qwen3.7-max-2026-06-08`. Keys live in `.env`, never committed.
6. Upstream commits are pinned; do not track main. Version claims must cite file:line and be
   reproducible with a command (UPSTREAM.md style).

## Style

Repository language is English, written plainly. Comments explain why, not what.
No marketing adjectives, no emoji, no bullet-point walls in docs. Commits tell a story:
what changed and why, not "Add feature". Run `uv run ruff check` before committing.
