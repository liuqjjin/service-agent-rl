# Upstream provenance

Everything in `third_party/` is someone else's work, pinned and used as-is except
where stated below. Everything in `src/service_agent/`, `tests/`, `data_protocol/`,
`reports/`, and `runbooks/` is written for this project.

## Pins

| Project | Pin | License | Role |
|---|---|---|---|
| [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | `cf71a80` (v1.0.1) + 1 local fix commit (`2822d90`) | MIT | Environment, telecom domain, user simulator, evaluator, metrics, native runner |
| [OpenPipe/ART](https://github.com/OpenPipe/ART) | `828b839` (0.5.18) | Apache-2.0 | RL trainer (GRPO + LoRA); its tau-bench client/rollout define the HTTP contract our shim serves |

Pins are frozen for the life of the project. tau2 v1.0.1 changed
banking_knowledge scoring relative to earlier releases, so results are not
comparable across versions; nothing here tracks `main`.

ART is not installed in the development environment on macOS. It is a read-only
reference here; the training environment on the GPU box installs it per
`runbooks/autodl.md`.

## Local modification to tau2-bench

One commit on branch `fix/gym-seed-and-thread-lifecycle`, based on `cf71a80`,
touching only `src/tau2/gym/gym_agent.py`. A test
(`tests/test_upstream_pristine.py`) asserts that no other file differs from the
pin, so every result-bearing path (runner, orchestrator, evaluator, domains,
data) is byte-identical to upstream.

That commit does not exist in sierra-research's repository, so `.gitmodules`
points the submodule at a fork, [liuqjjin/tau2-bench](https://github.com/liuqjjin/tau2-bench),
which carries the full upstream history. Without this, `git clone
--recurse-submodules` cannot check the submodule out at all. The pin stays
verifiable from the fork: `git merge-base HEAD cf71a80` returns the pin and
`git diff --name-only cf71a80` returns one file.

The native runner does not import the gym wrapper; all benchmark results are
unaffected by this commit.

### Bug 1: `reset(seed=...)` never reached the Orchestrator

- Native path (correct): `runner/batch.py:517-518` derives per-trial seeds,
  `runner/build.py:384-427` passes `seed=` to the Orchestrator,
  `orchestrator/orchestrator.py:123` stores it and `:527-528` call
  `agent.set_seed()` / `user.set_seed()`.
- Gym path (broken at the pin): `gym/gym_agent.py` `reset()` calls only
  `super().reset(seed=seed)` (gymnasium RNG); `_get_orchestrator()` (lines
  1080 and 1531 at the pin — both env classes) constructs `Orchestrator`
  without `seed=`. Seeded gym runs were silently unseeded.

### Bug 2: every `reset()` leaked a permanently blocked thread

The orchestrator thread blocks in `GymAgent.generate_next_message()` waiting
for `set_action()`. `reset()` joined it with `timeout=1.0` (`gym_agent.py:684`
at the pin), which cannot succeed against a thread waiting on an event that
will never fire, then started a new daemon thread anyway. Each reset leaked one
thread pinning the abandoned orchestrator's object graph. The fix adds
cooperative cancellation (`abort()` raises `SimulationAborted` inside the
thread), `close()` on both envs, and a guard so a straggler thread cannot
clobber the replacement simulation's state.

### Reproduce both bugs on the pristine pin

```bash
git -C third_party/tau2-bench checkout cf71a8070269883e38a365ffa85f78f46844c1f4
uv run pytest tests/test_gym_lifecycle.py   # 5 of 6 fail
git -C third_party/tau2-bench checkout fix/gym-seed-and-thread-lifecycle
uv run pytest tests/test_gym_lifecycle.py   # all pass
```

PR to upstream: branch pushed to the fork, description written, not opened.

## Verified findings at the pin

Each claim was checked by reading the pinned code; the command reproduces it.
Line numbers refer to `cf71a80`. Run from `third_party/tau2-bench`.

**Splits: `train`∩`test`=0, but `full` and `base` both contain `test`.**
Sizes: small=20, train=74, test=40, full=2285, base=114; test⊂full, test⊂base.
```bash
python3 -c "import json,itertools; j=json.load(open('data/tau2/domains/telecom/split_tasks.json')); S={k:set(map(str,v)) for k,v in j.items()}; [print(k,len(v)) for k,v in S.items()]; [print(a,b,len(S[a]&S[b])) for a,b in itertools.combinations(S,2)]"
```

**Write tools do not enforce the policy's business preconditions.**
`send_payment_request` docstring admits it does not check PAID
(`src/tau2/domains/telecom/tools.py:369-370`); `refuel_data`'s active-line
check is commented out (`tools.py:629-630`); the policy places those checks on
the agent (`data/tau2/domains/telecom/main_policy.md:117` overdue check,
`:126` contract-end resume ban, `:134` 2GB cap, `:9` one tool call at a time).
```bash
grep -n "does not check\|Always check" src/tau2/domains/telecom/tools.py
grep -n "Line must be active to refuel" src/tau2/domains/telecom/tools.py
grep -n "will not check\|not allowed to lift\|maximum amount\|one tool call" data/tau2/domains/telecom/main_policy.md
```

**Tools are typed READ/WRITE** (`@is_tool(ToolType.WRITE)`, 6 write + 6 read in
telecom `tools.py`), which is what lets a governance layer gate writes only.
```bash
grep -c "is_tool(ToolType.WRITE)" src/tau2/domains/telecom/tools.py
```

**The evaluator replays trajectory writes.** `EnvironmentEvaluator.calculate_reward`
rebuilds a predicted environment by replaying mutating tool calls from the
official trajectory (`evaluator/evaluator_env.py:85-129` via
`Environment.set_state`, `environment/environment.py:293-410`) and compares DB
hashes against a gold environment. `set_state` requires every tool call in the
trajectory to be followed by its matching ToolMessage (id-checked) and, with
`strict=True`, raises if a replayed call returns different content. A tool call
that entered the trajectory but was not executed live therefore breaks
evaluation — this is why our governance layer rejects candidate actions
*before* they enter the trajectory, not after.
```bash
grep -n "predicted_environment\|set_state\|get_db_hash" src/tau2/evaluator/evaluator_env.py | head
grep -n "Tool message expected\|Tool call id mismatch" src/tau2/environment/environment.py
```

**Premature termination scores 0; reward is a product of components.**
`evaluator/evaluator.py:57` ("product of all applicable component rewards"),
`:118-128` (premature termination note). `termination_reason` distinguishes
max-steps from wrong-answer failures and must be reported separately.

**pass^k** = C(c,k)/C(m,k), raises if `num_trials < k`
(`metrics/agent_metrics.py:113-126`). pass^4 therefore needs ≥4 trials per task.

**All model calls go through LiteLLM** (`utils/llm_utils.py:14-15`; agents call
`generate()` from `agent/llm_agent.py`).

**The gym info dict returns the full Task** (`gym/gym_agent.py:728` at the pin),
and `Task.__str__` (`data_model/tasks.py:624-641`) prints "Evaluation
Criteria:" including the reference actions
(`EvaluationCriteria.actions`, `tasks.py:377-389` — "One reference trajectory
that solves the task"). Any adapter that does `str(info["task"])` leaks the
answer key. Telecom task IDs additionally spell out the injected fault chain.
Our leak tests (`tests/test_leakage.py`) assert against both.

**tau2 v1.0.1's own HTTP services do not match ART's client contract.**
tau2 serves `/api/v1/get_tasks`, `/api/v1/run_domain` (batch-style,
`api_service/simulation_service.py:30,42`) and an environment manager with
`POST /{env_id}/tools/{tool_name}` (`orchestrator/environment_manager.py:171`).
ART's `TauBenchClient` (`ART/src/art/tau_bench/client.py:109`) expects
`GET /scenarios`, `POST /environments`, `POST /environments/{id}/step`
(`client.py:168,192,218`). Hence the FastAPI shim in `src/service_agent/serve/`.

**ART's bundled example trains on the test split.**
`ART/dev/tau-bench-minimal.py:37`: `get_scenarios(domain="telecom",
split="test")`. Useful as a smoke example, forbidden as an experimental
protocol; our training entry points assert train-core only.

**ART defaults local vLLM tool parsing to Hermes, but permits an exact
override.** `ART/src/art/unsloth/service.py:276-286` constructs server
arguments with `tool_call_parser="hermes"` and then overlays the caller's
`server_args`. [Qwen's model card at the frozen revision](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md)
specifies `qwen3_coder` for vLLM tool use, so the project passes that value
explicitly and records it in the semantic contract.

```bash
sed -n '276,286p' third_party/ART/src/art/unsloth/service.py
```

**An ART step can advance without gradient work.**
`ART/src/art/local/backend.py:1466-1521` handles a batch with no trainable
reward groups by copying the current checkpoint to the next numbered
directory, incrementing the logical step, and emitting zero trainable groups
and zero gradient steps. Project manifests therefore report checkpoint
positions and gradient-bearing work separately.

```bash
sed -n '1466,1521p' third_party/ART/src/art/local/backend.py
```

**ART's mask patch predates the installed Transformers 5 signature.**
The pinned wrapper
(`ART/src/art/transformers/patches.py:14-34`) omits the newer
`cache_position` argument and forwards all values positionally. The project
adapter described in DECISIONS.md D21 verifies both signatures before model
registration and leaves ART unmodified. Run this command from the
superproject root in the trainer environment:

```bash
.venv-trainer/bin/python - <<'PY'
import inspect
import art
from art.transformers import patches
print(inspect.signature(patches._preprocess_mask_arguments))
print(inspect.signature(patches._patched_preprocess_mask_arguments))
PY
```

**ART's rollout builds prompts safely.** `ART/src/art/tau_bench/rollout.py:99-102`
uses `env.info["policy"]` + observation + tool schemas, not the Task object.
Its default user simulator is `gpt-4.1-2025-04-14` (`rollout.py:31`), which we
override (see DECISIONS.md).

## Environmental quirk

Upstream `tests/test_gym/test_gym.py` spins up a real LLM user simulator even
for the mock domain; without `OPENAI_API_KEY` the orchestrator thread dies
instantly and 3 tests fail. That happens at the pristine pin too — it is not
caused by our fix (same 3 failures before and after).
