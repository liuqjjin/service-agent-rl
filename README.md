# service-agent

An LLM agent's tool call is an intent, not an authorization. This project makes
that distinction concrete on tau2-bench's telecom domain: a deterministic
execution-governance layer that decides whether a proposed write is allowed to
run, and a 2x2 experiment that separates how much reliability comes from that
governance layer versus from RL post-training of the model itself.

The telecom write tools execute whatever they are asked. `send_payment_request`
does not check whether the bill is already paid; `refuel_data`'s active-line
check is commented out; neither enforces the business preconditions the policy
places on the agent. A capable model still gets those preconditions wrong, and
task reward hides it: on the dev set, the base model runs **36 policy-violating
writes, 34 of them in episodes that still scored full reward.** Reward alone
certifies the model clean while it breaks policy dozens of times. That gap is
the whole problem, and closing it is the whole project.

## The argument

The claim is verifiable, not rhetorical. Every assertion below reproduces from
the pinned upstream code (`third_party/tau2-bench` at `cf71a80`):

```bash
# The write tools admit they do not enforce policy
grep -n "does not check\|Always check" \
  third_party/tau2-bench/src/tau2/domains/telecom/tools.py
grep -n "Line must be active to refuel" \
  third_party/tau2-bench/src/tau2/domains/telecom/tools.py   # commented out

# The policy puts those checks on the agent
grep -n "will not check\|not allowed to lift\|maximum amount\|one tool call" \
  third_party/tau2-bench/data/tau2/domains/telecom/main_policy.md
```

UPSTREAM.md lists every such finding with file and line. The governance layer
closes the gap in code: each candidate write is checked against evidence
gathered from the conversation itself — the customer read, the bill status seen,
the price the user actually confirmed — before it is allowed to execute. Rules
are derived from the policy line by line (`src/service_agent/governance/telecom_rules.py`),
never from task answers.

## What this measures: a 2x2

The research question is not "can an agent do telecom support" — the frontier
already trains tau2 RL agents. It is: **when reliability improves, how much is
the governance harness and how much is the model?** A 2x2 factorial answers it.

| | Native harness | Governed harness |
|---|---|---|
| **Base model** | H0 | Hbest |
| **RL model** | RL | Hbest + RL |

From the four cells: harness effect (`Hbest - H0`), model effect (`RL - H0`),
and the interaction — whether governance and training fix the same failures or
different ones. The base-model row is measured below. The RL row runs on GPU
(`runbooks/autodl.md`); everything needed for it is built and tested.

## Dev findings (base model)

20 frozen dev tasks x 4 trials per arm, one fixed policy model (Qwen3.5-4B,
8-bit, thinking off) and one fixed user simulator (deepseek-v4-pro, non-thinking,
temperature 0). Three harness arms: H0 native, H1 precondition gate, H2 gate +
idempotency ledger + bounded recovery. Full report: `reports/governance_ablation.md`.

| Arm | avg reward | pass^4 | unauthorized writes | max-steps failures |
|---|---:|---:|---:|---:|
| H0 (native) | 0.912 | 0.850 | 36 | 7 |
| H1 (gate) | 0.850 | 0.750 | 0 | 12 |
| H2 (gate + ledger + recovery) | 0.900 | 0.850 | 0 | 8 |

The gate removes every unauthorized write, and the safety is not free: forcing
the compliant path pushes a 4B model to max-steps where it used to cut corners
(H1-H0 reward = -0.062, 95% CI [-0.125, -0.013], significant over a 10k paired
bootstrap on tasks). Bounded recovery earns most of it back by turning
tool-error loops into completed episodes (H2-H0 = -0.013, CI [-0.037, +0.000],
not significant), at zero violations. Hbest = H2.

"Unauthorized writes" is one yardstick across all arms: every arm's official
trajectories are replayed through the same governor that gates H1/H2 live
(`src/service_agent/eval/metrics.py`). For H0 the gate never ran, so it counts
writes it would have blocked; for H1/H2 it counts what leaked past (zero). The
violations break down as 24 unconfirmed refuel prices, 8 resumes over unpaid
overdue bills, 3 resumes past a contract-end date, 1 refuel with an unread
price — all business rules the tools do not enforce.

Reproduce a single arm (needs a served policy model and `DEEPSEEK_API_KEY` in
`.env`):

```bash
uv run python -m service_agent.eval.run_ablation --arm h2 --tasks dev --trials 4 \
    --agent-llm "openai/<served-model>" --agent-api-base http://127.0.0.1:8398/v1 \
    --out results/dev/h2
uv run python -m service_agent.eval.report_ablation   # regenerates both reports
```

## What is mine, what is upstream

Everything under `src/service_agent/` and `tests/` is written for this project.
`third_party/tau2-bench` (`cf71a80`) and `third_party/ART` (`828b839`) are pinned
submodules, not vendored into the tree. ART is unmodified; tau2-bench carries one
local fix commit to the gym wrapper (the seed/thread fixes below), isolated by a
test that asserts nothing else differs from the pin. UPSTREAM.md draws the line
precisely. The pieces I built:

- **Execution governance** (`governance/`, `agent/governed.py`): the evidence
  extractor, the policy-derived rule table, idempotency keys, bounded recovery,
  and the `GovernedLLMAgent` that adjudicates a candidate *before* it enters the
  official trajectory. That placement is a correctness requirement, not a
  preference — see below.
- **Data hygiene** (`splits.py`, `leakage.py`): the split protocol that keeps
  training off the test set, deterministic stratified dev selection, and
  leak detection that matches serialized task labels against every
  model-visible surface.
- **A FastAPI shim** (`serve/tau2_shim.py`): ART's tau-bench client speaks the
  original tau-bench env-server protocol, which tau2 v1.0.1 does not serve; the
  shim bridges it to tau2's native environment, user simulator, and evaluator.
- **Two upstream bug fixes** to tau2's gym wrapper (seed propagation and a
  thread leak), with reproducing tests, on a branch ready to submit as a PR.
- **The evaluation and analysis harness**: ablation arms over the native runner,
  one-yardstick governance metrics, paired bootstrap, mechanical failure
  taxonomy, and the RL training path.

I do not reimplement the environment, the user simulator, the evaluator, or the
RL trainer. The value is in understanding where enterprise agents actually break
and measuring it rigorously, not in rewriting a benchmark.

## The one correctness subtlety worth knowing

tau2's evaluator does not read the live environment. It computes the
database-match reward by **replaying the write tool calls found in the official
trajectory** against a fresh environment. So a rejected action must never enter
the trajectory — if it did, the replay would re-execute it and the reward would
be wrong, or strict replay would crash. The governance layer therefore
adjudicates inside the agent, before the message is returned to the
orchestrator: rejected candidates go only to an audit trail, the model gets
private feedback and regenerates, and the official trajectory contains only
authorized actions. An end-to-end test drives the real orchestrator and
evaluator to prove replay stays valid (`tests/test_governed_agent_replay.py`).

## Reproduce

```bash
uv sync                 # Python 3.12; tau2 installed editable from the submodule
uv run pytest           # 93 tests: splits, leakage, gym fixes, governance,
                        # replay safety, shim, shim/native parity, statistics
```

Tests need no API keys — models are mocked or driven by scripted stand-ins.
The ablation and RL runs need a served policy model and keys in `.env`;
`CLAUDE.md` lists the serving commands and the traps found along the way.

## Limitations

- **The RL row is not yet run.** The training path, the step-0 logprob
  consistency gate, the SFT fallback, and the AutoDL runbook are built and unit
  tested, but the GRPO run and the final 2x2 on the test split require a GPU and
  have not executed. Every number here is the base-model row on the dev set.
- **The dev serving stack is not the final stack.** Dev ablations run the policy
  on quantized MLX; the final 2x2 runs it on vLLM in bf16. Dev results select
  Hbest; the final table will be produced entirely on the final stack, and the
  difference is disclosed rather than smoothed over.
- **One domain, one model size.** Whether the governance/training split holds
  across enterprise domains and model scales is the first thing more compute
  would test.
- **The user simulator is part of the benchmark.** All cells share one simulator
  build; no number is compared across a simulator change.

## Layout

```
src/service_agent/
  splits.py leakage.py     data protocol and label-leak detection
  governance/              decisions, evidence, telecom rules, idempotency, recovery, audit
  agent/governed.py        pre-trajectory adjudication
  serve/tau2_shim.py       ART client protocol over tau2 natives
  eval/                    ablation arms, one-yardstick metrics, factorial stats, reports
  training/                GRPO driver, logprob gate, SFT bridge
reports/                   baseline protocol, governance ablation, failure taxonomy
runbooks/autodl.md         the GPU training sequence, end to end
UPSTREAM.md                pins, provenance, every verified code-level claim
DECISIONS.md               execution-time judgment calls, with reasoning
```
