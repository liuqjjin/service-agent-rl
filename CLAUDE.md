# service-agent

Execution governance + GRPO post-training for LLM agents on tau2-bench telecom (dual-control).
The research question: how much reliability comes from a deterministic governance harness vs.
from RL post-training, measured with a 2x2 factorial (base/RL model x native/governed harness).

## Commands

```bash
uv sync                 # Python 3.12; tau2 installed editable from third_party/tau2-bench
uv run pytest           # our tests only (tests/)
uv run pytest third_party/tau2-bench/tests/test_gym/test_gym.py  # upstream gym tests
```

Upstream gym tests spawn a real LLM user simulator even for the mock domain: without
`OPENAI_API_KEY` the orchestrator thread dies instantly and 3 tests fail. That is
environmental, not a code bug.

## Layout

- `src/service_agent/` — everything we own: governance, governed agent, audit, eval, shim, training.
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
