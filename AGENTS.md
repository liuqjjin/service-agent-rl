# service-agent — guide for coding agents

Execution governance + GRPO post-training for LLM agents on tau2-bench telecom (dual-control).
The research question: how much of an agent's reliability comes from a deterministic governance
harness and how much from RL post-training, measured with a 2x2 factorial
(base/RL model x native/governed harness).

`CLAUDE.md` is the source of truth for the six hard rules and the project's scope; the private
manuals under `planning/` (gitignored) are the source of truth for what is in and out of scope.
This file is the operational guide: what is where, which commands actually work, what the code
does, and what is still broken. Where this file and `CLAUDE.md` ever disagree, `CLAUDE.md` wins
and this file is the one to fix. Do not edit `CLAUDE.md` to resolve a disagreement.

## Status

The base-model row of the 2x2 is measured on the frozen dev set:
`reports/governance_ablation.md`, Hbest = H2. The GPU path now has separate
zero-update preflight, one-update smoke, and formal lineages; exact model,
tokenization, ART/tau2, runtime, split, replay, and resume contracts are tested
locally. It has not yet passed a real GPU preflight and no weights have been
trained, which is why the package is still named `service-agent`.

Local Mac gate at the time of writing: 99 tests green, lint clean, both
submodules at their pins. GPU work happens only on `codex/autodl-grpo-final`;
`main` stays untouched.

## Repository map

Everything under `src/service_agent/`, `tests/`, `data_protocol/`, `reports/`, and `runbooks/`
is written for this project. Everything under `third_party/` is someone else's work, pinned.

```
src/service_agent/
  splits.py                data protocol: split hygiene, deterministic dev selection, fingerprints
  leakage.py               label-leak detection over serialized task fields
  governance/
    core.py                Decision enum, ProposedAction, GovernanceResult, idempotency keys
    evidence.py            EvidenceState + EvidenceExtractor: what the conversation established
    telecom_rules.py       TelecomGovernor: one rule per policy line, cited by file:line
    recovery.py            tool-error classification and the session-global retry budget
    audit.py               AuditTrail: every candidate, allowed or not; JSONL persistence
  agent/governed.py        GovernedLLMAgent: adjudication at the generation boundary
  eval/
    registration.py        registers the h1/h2 arms with tau2's registry
    run_ablation.py        CLI: one arm over the native runner, self-describing run directory
    metrics.py             offline governance replay (one yardstick incl. H0) + audit aggregation
    factorial.py           paired bootstrap, 2x2 effects, mechanical failure taxonomy
    report_ablation.py     regenerates both dev reports from committed artifacts
  serve/tau2_shim.py       FastAPI shim: ART's tau-bench protocol over tau2 v1.0.1 natives
  training/
    contracts.py           frozen model/runtime/API contracts and manifest gates
    model_snapshot.py      exact-revision download plus offline pin after verification
    tau_rollout.py         ART rollout with fail-closed multi-tool handling
    token_budget.py        context floor measured from every committed dev prefix
    art_tau_train.py       preflight/smoke/formal GRPO driver (imports ART lazily)
    logprob_check.py       step-0 rollout/trainer logprob consistency gate
    sft_prepare.py         teacher-trajectory filter and chat-JSONL export (GRPO fallback)
data_protocol/             frozen dev IDs and the split report. Committed, never regenerated silently
reports/                   baseline protocol, governance ablation, failure taxonomy
runbooks/autodl.md         the GPU sequence end to end
results/dev/{h0,h1,h2}/    the measured dev ablation: run_config, results, metrics, audit JSONL
results/compat/            the compatibility-matrix smoke runs behind DECISIONS.md D7
logs/dev_h*.log            raw stdout of the three dev runs
UPSTREAM.md                pins, provenance, every code-level claim with a reproducing command
DECISIONS.md               execution-time judgment calls (D1-D8), with reasoning
```

Submodule gitlinks, as checked out:

| Path | Commit | Note |
|---|---|---|
| `third_party/tau2-bench` | `2822d90` | branch `fix/gym-seed-and-thread-lifecycle`, merge-base `cf71a80` (v1.0.1) |
| `third_party/ART` | `828b839` | unmodified; read-only reference for the RL client contract |

`2822d90` is one commit on top of the `cf71a80` pin touching only `src/tau2/gym/gym_agent.py`.
Do not "restore" the submodule to `cf71a80`: `tests/test_gym_lifecycle.py` needs the fix commit,
and `tests/test_upstream_pristine.py` enforces that nothing else differs from the pin.

`.gitmodules` points tau2-bench at the fork `liuqjjin/tau2-bench`, because `2822d90` does not
exist in sierra-research's repo and a clone pointed there cannot check the submodule out. Locally
the submodule keeps `upstream` as a second remote for provenance. Do not repoint `.gitmodules`
back at sierra-research; that is what broke fresh clones before.

## Environment and commands

Python 3.12 (`.python-version`; `requires-python = ">=3.12,<3.14"`), uv-managed. tau2 is
installed editable from the submodule via `[tool.uv.sources]`.

```bash
uv sync                  # create/refresh .venv
uv run pytest            # our tests only (testpaths = ["tests"]); 99 tests, ~5s, no API keys
uv run ruff check        # line-length 100, rules E4/E7/E9/F/I
uv run pytest third_party/tau2-bench/tests/test_gym/test_gym.py   # upstream; see below
```

Serving a local policy model happens in a separate uv tool environment that is deliberately
not in the project lock:

```bash
mlx_lm.server --model /Users/lqj/local-llm/models/Qwen3.5-4B-MLX-8bit  --port 8398  # dev policy
mlx_lm.server --model /Users/lqj/local-llm/models/Qwen3.6-35B-A3B-4bit --port 8399  # teacher/smoke
```

The model id after `openai/` in `--agent-llm` must be byte-identical to the server's `--model`
value, or mlx_lm.server silently downloads that id from Hugging Face instead of using the loaded
weights. The recorded dev-ablation runs used the absolute local paths above; see
`results/dev/h0/run_config.json` for the exact strings that produced the committed numbers.

One ablation arm (keys come from `.env`, loaded by `python-dotenv`):

```bash
uv run python -m service_agent.eval.run_ablation --arm h2 --tasks dev --trials 4 \
    --agent-llm "openai//Users/lqj/local-llm/models/Qwen3.5-4B-MLX-8bit" \
    --agent-api-base http://127.0.0.1:8398/v1 --out results/dev/h2
uv run python -m service_agent.eval.report_ablation    # regenerates both dev reports
```

`--arm` is one of `h0`/`h1`/`h2`; `--tasks` is one of `smoke3`/`dev`/`train-core`. There is no
`test` option, on purpose (`runbooks/autodl.md` §7). Adding one is a protocol decision, not a
convenience change.

Model-serving traps, all found empirically and all handled inside `run_ablation`; a direct
`generate()` call must not forget them:

- Local Qwen agents need `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`.
  Otherwise long tasks return think-only (empty) completions that fail tau2's message validation
  and burn every runner retry.
- `deepseek-v4-pro` disables thinking only via `extra_body={"thinking": {"type": "disabled"}}`;
  the top-level `thinking` parameter is ignored through LiteLLM, and the default mode leaks
  `reasoning_content`.

The shim and the training entry points:

```bash
uv run python -m service_agent.serve.tau2_shim              # SHIM_HOST/SHIM_PORT, default :8000
curl -s "localhost:8000/scenarios?domain=telecom&split=train-core" | ...   # 54 scenarios
curl -s "localhost:8000/scenarios?domain=telecom&split=test" -o /dev/null -w "%{http_code}\n"  # 403
python -m service_agent.training.art_tau_train --smoke      # trainer venv on the GPU box only
python -m service_agent.training.logprob_check --api-base ... --served-model ... --hf-model ...
```

`SHIM_ALLOW_EVAL_SPLITS=1` unlocks the test split on **both** `GET /scenarios` (which refuses to
list `test`/`full`/`base`) and `POST /environments` (which refuses to instantiate any task in the
official test split). It exists for the single final evaluation run and must never be set during
training. Both endpoints have to agree: task ids are readable straight out of `split_tasks.json`,
so a client can name a test task without ever listing one.

## Architecture and data flow

**The governance layer.** A model's tool call is an intent, not an authorization. tau2's telecom
write tools execute whatever they are asked — `send_payment_request` admits in its docstring that
it does not check PAID (`tools.py:369-370`), `refuel_data`'s active-line check is commented out
(`tools.py:630`) — while `main_policy.md` places those preconditions on the agent. `TelecomGovernor`
closes the gap: for each of the six `ToolType.WRITE` tools it checks the candidate against
`EvidenceState`, which is built only from things the model itself could have seen (tool results
joined to calls by `tool_call_id`, prices the agent quoted, the user's next reply). Every rule
cites the policy line it came from. A gate that peeked at task labels would be leakage, not a
harness result.

**Where adjudication happens, and why it cannot move.** tau2's evaluator does not read the live
environment. It recomputes the database-match reward by replaying the write tool calls found in
the *official trajectory* against a fresh environment via `Environment.set_state`, which also
requires every trajectory tool call to be followed by its matching `ToolMessage`. So a rejected
action must never enter the trajectory at all. The only place that satisfies this is inside the
agent, before `generate_next_message` returns: `GovernedLLMAgent` generates a candidate,
adjudicates it, and returns it only if allowed. Rejections go to the audit trail and become an
ephemeral amendment to the system prompt for the next attempt (position zero — serving stacks
reject system content anywhere else). Regeneration is bounded at 2; past that the agent sends a
fixed safe text instead of an unauthorized write.

**Arms.** H0 is tau2's native `llm_agent`, untouched. H1 is `GovernedLLMAgent` with the
precondition gate only. H2 adds the idempotency ledger and the bounded recovery budget.
`GovernanceConfig.h1()`/`.h2()` are the only difference; `registration.py` exposes them as two
registry names because `build_agent`'s factory contract has no channel for arbitrary config.

**One yardstick.** `eval/metrics.analyze_trajectory` replays *any* arm's official trajectory
through the same governor that gates H1/H2 live. For H0 that counts executed writes the gate
would have blocked; for H1/H2 it counts what leaked past (zero). This is what makes the
"unauthorized writes" column comparable across arms. H1/H2 additionally report their live audit,
which contains candidates that never became trajectory messages and are therefore invisible to
replay by construction.

**The shim.** ART's `TauBenchClient` speaks the original tau-bench env-server protocol
(`GET /scenarios`, `POST /environments`, `POST /environments/{id}/step`), which tau2 v1.0.1 does
not serve. `serve/tau2_shim.py` composes tau2's registry, `Environment`, `UserSimulator`, and
evaluator behind it. Two details are correctness-critical: ART *sums* step rewards, so every
intermediate step returns `0.0` and the official reward is returned exactly once on the
terminating step; and every message is a real tau2 message stamped at event time and appended in
event order, so the official evaluator can strict-replay the trajectory.
`tests/test_shim_native_parity.py` drives one scripted episode down both paths and asserts the
same policy text, the same tool sequence, and the same reward.

**Training.** GRPO runs in the native H0 environment through the shim, so the 2x2's model effect
stays attributable to training alone. `art_tau_train.py` asserts the scenario IDs are a subset of
frozen train-core and that no scenario carries evaluation labels, before any step. ART's own
bundled example trains on `split="test"`; those asserts exist so that mistake is impossible here.

## Invariants you must not break

These are the six hard rules from `CLAUDE.md`, restated with where they live in code. Breaking
any of them invalidates results, which is worse than a bug: the numbers still look fine.

1. **Governance rejects before the trajectory.** Rejected candidates go to `AuditTrail` only.
   Never "log it and skip execution" — the evaluator's replay would re-execute it or crash.
   Guarded by `tests/test_governed_agent_replay.py`, which drives the real orchestrator and the
   real evaluator.
2. **Never train on `test`, `full`, or `base`** (`full` and `base` both contain `test`). Training
   uses train-core (54 IDs); dev (20 frozen IDs) drives every selection decision; test is touched
   exactly once, at the end. Guarded by `tests/test_splits.py`, the shim's 403 on both
   `/scenarios` and `/environments`, and `assert_training_split_clean`.
3. **Model prompts are built by whitelist.** `str(task)`, `task.evaluation_criteria`,
   `task.initial_state`, `user_scenario`, and task IDs must never reach a prompt. Leak tests
   assert on serialized substrings, not field names, so a leak through any formatting path trips.
   Any new model-visible surface (a new feedback string, a new observation field) needs a
   `find_leaks` assertion in the same commit.
4. **RL trains in native H0; governance is layered on only at evaluation.** Mixing them confounds
   the model-vs-harness attribution the whole project exists to measure.
5. **The user simulator is fixed:** official DeepSeek API `deepseek-v4-pro`, non-thinking,
   temperature 0, identical parameters in every cell. Fallback, recorded in DECISIONS.md D1:
   DashScope `qwen3.7-max-2026-06-08`, dev-only. Keys live in `.env`, never committed.
6. **Upstream commits are pinned; do not track `main`.** Any version claim must cite `file:line`
   and be reproducible by a command, UPSTREAM.md style.

Two more that are project practice rather than numbered rules: `data_protocol/` is frozen output
that is regenerated only deliberately (`python -m service_agent.splits` rewrites it), and
`reports/*.md` are generated — edit `report_ablation.py`, not the markdown.

## Test gates

`uv run pytest` must be green before any commit. As of HEAD: 99 passed in ~5s, no API keys
needed, models mocked or driven by scripted stand-ins.

| File | What it protects |
|---|---|
| `test_splits.py` | split hygiene, frozen dev IDs equal regeneration, train-core disjointness |
| `test_leakage.py` | the detector fires on planted leaks and stays quiet on the real prompts |
| `test_upstream_pristine.py` | only `gym/gym_agent.py` differs from `cf71a80`; submodule clean |
| `test_gym_lifecycle.py` | the two upstream gym bugs stay fixed (seed, thread leak) |
| `test_governance_rules.py` | each rule allows/blocks per the policy line it cites |
| `test_evidence_and_recovery.py` | extraction mechanics, quote binding, retry budget |
| `test_governed_agent_replay.py` | the correctness test: rejected calls never enter the trajectory, allowed ones replay, feedback stays private and leak-free |
| `test_eval_harness.py` | arm registration, offline analyzer verdicts |
| `test_shim.py` | ART contract, split lock on both endpoints, reward-once, strict replay |
| `test_shim_native_parity.py` | native and shim agree on tools and reward |
| `test_factorial.py` | bootstrap and 2x2 arithmetic |
| `test_training_contract.py` | pins, ART API, bf16/runtime config, CUDA-runtime bootstrap, phase/resume gates, multi-tool fail-close, true within-group reward variance |

`uv run pytest third_party/tau2-bench/tests/test_gym/test_gym.py` yields **10 passed, 3 failed**
without `OPENAI_API_KEY`: those tests spin up a real LLM user simulator even for the mock domain,
so the orchestrator thread dies on a credentials error. That happens at the pristine pin too. It
is environmental, not a regression — do not "fix" it.

The reports are also a gate of sorts: `service_agent.eval.report_ablation` regenerates
`reports/governance_ablation.md` and `reports/failure_taxonomy.md` byte-identically from the
committed run artifacts. If a change makes them differ, either the change is wrong or the reports
need regenerating in the same commit, with the difference explained.

## Conventions

Repository language is English, written plainly. No marketing adjectives, no emoji, no
bullet-point walls in prose docs. Comments explain why, not what — the codebase's module
docstrings are the model: they explain the constraint that forced the design, usually with an
upstream `file:line`. Commits tell a story ("Sanitize mixed candidates instead of fighting the
model's formatting"), not "Add feature". Run `uv run ruff check` before committing.

Numbers in docs carry their provenance: the command that produced them and the artifact path
they can be recomputed from.

## Do not

- Edit anything in `third_party/` outside `src/tau2/gym/gym_agent.py`, and prefer not to touch
  even that; a hash-guard test enforces the boundary.
- Add a `--tasks test` option, unlock `SHIM_ALLOW_EVAL_SPLITS`, or run anything against the test
  split without an explicit decision recorded in `DECISIONS.md`.
- Hand-edit `reports/*.md` or `data_protocol/*.json`.
- Commit `.env`, `planning/`, weights, or `wandb/` (all gitignored — keep it that way).
- Substitute the local 35B for the formal 4B policy in any reported cell (DECISIONS.md D2), or
  compare numbers across a user-simulator change (D1).
- Bump submodule pins, or resolve a dependency conflict by loosening `[tool.uv.sources]`.

## Known gaps

Each one is a real blocker or a real inaccuracy, not a style preference. Fixed
entries are removed rather than retained as stale warnings: the submodule fork
and both test-split locks are in place; audit normalization and persistence are
now unambiguous; the final model revision is pinned; and first-party imports are
declared directly.

- **The project repo is private.** The current GPU box receives the exact
  branch by rsync over the configured SSH connection. A future independent
  clone still needs GitHub credentials. Making the repo public is a portfolio
  decision, not a code one.
- **The GPU contracts are locally tested but not yet empirically cleared.**
  The pinned ART/vLLM/Unsloth stack must still pass exact-token/logprob
  preflight and one real optimizer update on the RTX 4090 before formal GRPO.
- **`enable_roaming`/`disable_roaming` have no rule.** They are in `WRITE_TOOLS` and fall through
  to `no_specific_rule` (28 such records per dev arm). `main_policy.md:155` only says to enable
  roaming at no cost for a traveling user, so an unconditional allow is defensible — but
  `telecom_rules.py` documents the Change Plan omission and not this one.
- **Dev and final serving stacks differ**, disclosed rather than smoothed over: dev ablations ran
  quantized MLX, the final 2x2 is specified on vLLM bf16. Dev results only select Hbest.
- **Scope caveats that more compute would address**: one domain, one model size, one simulator
  build, and a `pass^4` computed on 4 trials per dev task.
