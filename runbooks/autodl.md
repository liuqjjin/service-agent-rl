# AutoDL GRPO runbook

This is the reproducible GPU procedure. The current instance is one NVIDIA RTX
PRO 6000 Blackwell Server Edition with 97,887 MiB VRAM and a 200 GB data disk at
`/root/autodl-tmp`. All large state lives on that disk. Services bind to
localhost only.

The training protocol has three lineages:

1. `preflight-qwen3coder-r1`: step-0 token/logprob gate, then rollout-only — passed.
2. `smoke-qwen3coder-r1`: one disposable backend training call — passed.
3. `grpo-4b-qwen3coder-r1`: fresh formal run, authorized by both gates — terminal
   `stopped_sparse_reward` at checkpoint position 0024.

The official test split stays locked throughout all three.

## Frozen inputs

| Input | Value |
|---|---|
| Superproject | commit recorded in each manifest |
| ART | `828b839b1139ac780725f0a22a9bde70a82b4878` |
| tau2 fork | `2822d9030b621e6f13a190fb14fa08cf1c9c4ca4` |
| Base model | `Qwen/Qwen3.5-4B` |
| Model revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Policy template | tokenizer at that revision, `enable_thinking=false` |
| Tool-call parser | `qwen3_coder` |
| User simulator | `deepseek/deepseek-v4-pro`, thinking disabled, temperature 0 |
| Training split | frozen train-core, 54 tasks |
| Selection split | frozen dev, 20 tasks |

## 1. Data-disk layout

```bash
mkdir -p /root/autodl-tmp/{work,cache/huggingface,cache/uv,cache/wandb,cache/unsloth_compiled_cache,wandb,tmp,logs,runs,art}
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_CACHE_DIR=/root/autodl-tmp/cache/wandb
export WANDB_DIR=/root/autodl-tmp/wandb
export UNSLOTH_COMPILE_LOCATION=/root/autodl-tmp/cache/unsloth_compiled_cache
export TMPDIR=/root/autodl-tmp/tmp
export TOKENIZERS_PARALLELISM=false
```

Persist those non-secret exports in the shell profile. Do not put API keys
there. The project `.env` contains `DEEPSEEK_API_KEY` and `WANDB_API_KEY`,
has mode `0600`, and is never printed or committed.

The AutoDL network cannot reach `huggingface.co:443`, so `HF_ENDPOINT` changes
transport only. `prepare_pinned_snapshot` still requires the mirror's `main`
metadata to resolve to the frozen revision, requires the downloaded snapshot
directory to carry that exact SHA, passes that local directory to both ART
training and vLLM, and then forces all model loads offline. The step-0
reference tokenizer and bf16 model also receive that manifest-recorded path
with local-only loading; they never resolve the canonical model ID.

## 2. Runtime installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv python install 3.12

cd /root/autodl-tmp/work/service-agent-rl
uv sync --python 3.12

uv venv /root/autodl-tmp/work/service-agent-rl/.venv-trainer --python 3.12
uv pip install --python .venv-trainer/bin/python -e 'third_party/ART[backend]'
uv pip install --python .venv-trainer/bin/python --no-deps -e .
uv pip install --python .venv-trainer/bin/python python-dotenv

# An editable ART checkout launches vLLM from its separately locked runtime.
uv sync --project third_party/ART/vllm_runtime --frozen --no-dev
```

The service environment owns tau2. The trainer environment owns ART,
Transformers, Unsloth, and the training torch stack. ART's source checkout
owns a third, isolated environment at `third_party/ART/vllm_runtime/.venv`;
that environment owns vLLM and its separately locked inference dependencies.
The first-party package is installed without dependencies in the trainer so
it does not resolve another tau2 version from PyPI. Preflight fails before
model registration if the isolated runtime or any required package metadata is
missing.

Record the environment without dumping variables:

```bash
nvidia-smi
df -h /root/autodl-tmp
free -h
uv run python -V
.venv-trainer/bin/python -c \
  'from importlib.metadata import version; import torch; print(version("openpipe-art"), torch.__version__, version("transformers"), version("unsloth"), torch.cuda.is_bf16_supported())'
third_party/ART/vllm_runtime/.venv/bin/python -c \
  'from importlib.metadata import version; import torch; print(version("art-vllm-runtime"), torch.__version__, version("transformers"), version("vllm"), torch.cuda.is_bf16_supported())'
test -x third_party/ART/vllm_runtime/.venv/bin/art-vllm-runtime-server
git status --short --branch
git submodule status
```

Then run the Mac-side gates again on the GPU box:

```bash
uv run pytest
uv run ruff check
uv run python -m service_agent.eval.report_ablation
git diff --exit-code -- reports/governance_ablation.md reports/failure_taxonomy.md
```

## 3. Shim

Run in its own tmux window:

```bash
cd /root/autodl-tmp/work/service-agent-rl
export SHIM_HOST=127.0.0.1
export SHIM_PORT=8000
export SHIM_MAX_STEPS=30
uv run python -m service_agent.serve.tau2_shim \
  2>&1 | tee /root/autodl-tmp/logs/shim.log
```

Health checks:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS 'http://127.0.0.1:8000/scenarios?domain=telecom&split=train-core' \
  | uv run python -c 'import json,sys; print(len(json.load(sys.stdin)["scenarios"]))'
curl -sS -o /dev/null -w '%{http_code}\n' \
  'http://127.0.0.1:8000/scenarios?domain=telecom&split=test'
```

The outputs must be healthy, `54`, and `403`. `SHIM_ALLOW_EVAL_SPLITS` must
not exist in the training process.

## 4. Preflight: zero updates

The command downloads and verifies the pinned model snapshot. It then:

1. probes ART's isolated runtime and requires its verified real CUDA library,
   rather than TileLang's linker stub, to provide `cudaDeviceReset`;
2. requires the `ninja` executable from that same locked runtime to be visible
   for FlashInfer JIT compilation;
3. verifies ART's pinned attention-mask wrapper against the installed
   Transformers 5 signature and installs the recorded project adapter;
4. registers ART at step 0;
5. samples exact vLLM prompt/completion token IDs and logprobs;
6. closes vLLM before loading the tokenizer and bf16 reference model from the
   exact manifest-recorded snapshot, with local-only loading;
7. requires byte-identical prompt token IDs, mean importance ratio within
   2% of 1, and at most 2% outside the PPO clip window;
8. reopens the untouched step-0 checkpoint;
9. runs eight train-core episodes through strict replay with reward finalized
   exactly once;
10. exits with `final_step=0`.

The manifest records the exact Python argv, download endpoint,
`qwen3_coder` parser, composite semantic-contract hash, and separate hashes for
the system prompt, tools, and tokenizer chat template. It also records the
selected CUDA runtime and bootstrap plus locked `ninja` paths, versions, and
SHA-256 values; smoke and formal phases reject drift in any of them. The
installed attention-mask adapter and both upstream parameter orders are
recorded and matched too. Manifest schema 3 makes the parser explicit; an
earlier Hermes manifest cannot authorize any phase.

```bash
cd /root/autodl-tmp/work/service-agent-rl
source .venv-trainer/bin/activate
set -o pipefail
python -m service_agent.training.art_tau_train \
  --phase preflight --run-name preflight-qwen3coder-r1 \
  --art-path /root/autodl-tmp/art/preflight-qwen3coder-r1 \
  --out /root/autodl-tmp/runs/preflight-qwen3coder-r1 \
  --hf-cache /root/autodl-tmp/cache/huggingface \
  --group-size 4 --max-turns 30 \
  --max-completion-tokens 1024 --max-model-len 16384 \
  --rollout-concurrency 4 --gpu-memory-utilization 0.68 \
  --logprob-calculation-chunk-size 512 \
  2>&1 | tee /root/autodl-tmp/logs/preflight-qwen3coder-r1.log
preflight_rc=${PIPESTATUS[0]}
printf '%s\n' "$preflight_rc" \
  > /root/autodl-tmp/logs/preflight-qwen3coder-r1.exit
deactivate
test "$preflight_rc" -eq 0
```

Gate artifact:

`/root/autodl-tmp/runs/preflight-qwen3coder-r1/preflight_manifest.json`

Do not continue unless its status is `passed`, both token and logprob gates
pass, strict replay is true, test locking is true, and both steps are zero.

## 5. Single-train-call disposable smoke

```bash
source .venv-trainer/bin/activate
set -o pipefail
python -m service_agent.training.art_tau_train \
  --phase smoke --run-name smoke-qwen3coder-r1 \
  --art-path /root/autodl-tmp/art/smoke-qwen3coder-r1 \
  --out /root/autodl-tmp/runs/smoke-qwen3coder-r1 \
  --hf-cache /root/autodl-tmp/cache/huggingface \
  --preflight-manifest \
    /root/autodl-tmp/runs/preflight-qwen3coder-r1/preflight_manifest.json \
  --group-size 4 --groups-per-step 2 --max-turns 30 \
  --max-completion-tokens 1024 --max-model-len 16384 \
  --rollout-concurrency 4 --gpu-memory-utilization 0.68 \
  --logprob-calculation-chunk-size 512 \
  2>&1 | tee /root/autodl-tmp/logs/smoke-qwen3coder-r1.log
smoke_rc=${PIPESTATUS[0]}
printf '%s\n' "$smoke_rc" \
  > /root/autodl-tmp/logs/smoke-qwen3coder-r1.exit
deactivate
test "$smoke_rc" -eq 0
```

The smoke submits four groups: formal slots 0-3, exactly the first two
two-group formal batches selected by `_scenario_for_slot`, with the same
policy/user seed mapping (42-57 at the frozen defaults). It must observe at
least one mixed-reward group before calling the backend. Its manifest must then
show checkpoint step 0 to 1, at least one ART trainable group, positive
`data/step_num_gradient_steps`, one checkpoint, strict replay, a W&B URL, and
no OOM. If the fixed sample has no variance, stop; do not search for another
seed. After its checkpoint and log are backed up, this ART lineage is
disposable and must never seed the formal run.

## 6. Formal GRPO

The formal configuration starts with group size 4, two task groups per
rollout/checkpoint step, four concurrent rollouts, bf16 LoRA, and a
16,384-token context. Preflight records p50/p95/p99/max prompt lengths from all
240 committed dev episodes and refuses the run if the context cannot cover the
observed maximum plus a governance-feedback buffer and 1,024 completion
tokens.

```bash
tmux new-session -d -s grpo-qwen3coder bash
tmux send-keys -t grpo-qwen3coder \
  'set -o pipefail; cd /root/autodl-tmp/work/service-agent-rl || exit $?; source .venv-trainer/bin/activate || exit $?; python -m service_agent.training.art_tau_train --phase train --run-name grpo-4b-qwen3coder-r1 --art-path /root/autodl-tmp/art/grpo-4b-qwen3coder-r1 --out /root/autodl-tmp/runs/grpo-4b-qwen3coder-r1 --hf-cache /root/autodl-tmp/cache/huggingface --preflight-manifest /root/autodl-tmp/runs/preflight-qwen3coder-r1/preflight_manifest.json --smoke-manifest /root/autodl-tmp/runs/smoke-qwen3coder-r1/smoke_manifest.json --group-size 4 --groups-per-step 2 --max-turns 30 --max-completion-tokens 1024 --max-model-len 16384 --rollout-concurrency 4 --gpu-memory-utilization 0.68 --logprob-calculation-chunk-size 512 --steps 60 --learning-rate 5e-6 --loss-fn ppo --val-every 5 --val-trials 2 2>&1 | tee /root/autodl-tmp/logs/grpo-4b-qwen3coder-r1.log; grpo_rc=${PIPESTATUS[0]}; printf "%s\n" "$grpo_rc" > /root/autodl-tmp/logs/grpo-4b-qwen3coder-r1.exit; exit "$grpo_rc"' C-m
```

Recorded outcome: the official lineage completed 24 of the requested 60
rollout/checkpoint positions. It reported five trainable groups, five
gradient-bearing checkpoint positions, 19 skipped positions, and 445 ART
gradient steps. Scheduled frozen-dev means were 0.850 at 0005, 0.850 at 0010,
0.925 at 0015, and 0.900 at 0020, so the fixed rule selected checkpoint 0015.
The last ten positions contained one mixed-reward group in total and triggered
the predeclared sparse-reward stop at 0024.

The driver persisted `status=stopped_sparse_reward` atomically and then raised
the protocol error that ends the process. The resulting shell exit 1 is
expected for this terminal manifest and is not an OOM, CUDA error, or
infrastructure crash. Rerunning the command to chase the requested 60
positions is forbidden: the exact lineage is terminal. The Bash `PIPESTATUS`
capture is required because `tee` otherwise hides the Python process's exit;
the three `.exit` files are committed under each phase as `process.exit`.

ART writes each position's submitted/trainable group counters on both the
rollout log record and the following backend-metrics record. Its cumulative
W&B state therefore displays 96 submitted and 10 trainable groups. Count
unique training work from the 24 manifest `train_steps` or the 24 backend
records: both give 48 submitted, 5 trainable, and 445 gradient steps.
Gradient steps occur only on backend records, so that counter is not doubled.
`results/gpu/WANDB_COUNTER_AUDIT.json` records this reconciliation.

The command is resume-safe: rerunning the exact command requires the same
semantic-contract hash and resolved ART lineage path, and requires the
manifest's checkpoint path and step to equal ART's. A passed or
sparse-reward-stopped lineage is terminal. Every rollout/checkpoint step writes
the manifest atomically.

`--steps 60` fixes 60 ART rollout/checkpoint positions, not 60 optimizer
steps. At the pinned ART commit, a batch with no mixed-reward group copies the
current checkpoint to the next step without gradient work. Each train-step
record and the top-level `progress` therefore report checkpoint steps,
gradient-bearing checkpoint steps, skipped checkpoint steps, submitted and
trainable groups, and `data/step_num_gradient_steps` separately. The selected
checkpoint records the same totals only through its own step.

The selected checkpoint rule is fixed before training: highest frozen-dev
average reward among scheduled checkpoints; ties choose the earliest step.
Every scheduled validation checkpoint is evaluated with the same dev
task/trial seeds (common random numbers). The test split is not involved.

Stop immediately on any of these:

- token/logprob gate failure;
- test endpoint no longer returns 403;
- prompt/tool/template/revision hash drift;
- multi-tool choice (fails before any call executes);
- missing strict replay or reward-finalized-once marker;
- evaluator or shim 5xx;
- at most one mixed-reward group over ten consecutive rollout/checkpoint steps;
- a second controlled OOM.

For the first OOM only: terminate residual GPU processes, keep the last good
checkpoint and logs, and retry once after reducing only rollout concurrency,
vLLM memory utilization, context length while retaining the measured context
floor, or the mathematically equivalent logprob calculation chunk size. The
chunk size is recorded in every phase manifest and must match across gates.
Do not change learning rate, reward, prompts, tools, model revision, group
size, or split. A second OOM is a hard gate.

SFT is not automatic. D26 freezes checkpoint 0015 as the current GRPO
candidate and does not authorize teacher data or SFT. Any fallback would
require a new decision and a fresh lineage.

## 7. Training handoff and backup

Before any final test run, copy these to the Mac and verify checksums:

- `/root/autodl-tmp/runs/preflight-qwen3coder-r1/`
- `/root/autodl-tmp/runs/smoke-qwen3coder-r1/`
- `/root/autodl-tmp/runs/grpo-4b-qwen3coder-r1/`
- all three ART lineages, including selected checkpoint 0015 and latest
  terminal checkpoint 0024 under `/root/autodl-tmp/art/`
- the three official logs and process-exit files plus `shim.log` under
  `/root/autodl-tmp/logs/`
- W&B run URLs
- a SHA-256 manifest for every copied file

Commit byte-identical copies of the three phase manifests and process exits
under `results/gpu/`; keep LoRA weights, trajectories, tensors, and raw logs in
the ignored local backup. Regenerate `reports/grpo_training.md` from the
committed evidence and require its tests to pass.

Recorded handoff: `checkpoints/autodl-backup-2026-07-28-qwen3coder-r1/`
contains the 292 indexed training-source files (about 3.3 GB) plus the later
recovery-proof manifest. Independently generated remote and local SHA-256
lists for those 292 sources match byte for byte; the canonical list has SHA-256
`a5c9d4c9e630dbfc422a41f4fb37298eef781466e0877d2b185226665107f724`.
`results/gpu/BACKUP_SHA256SUMS` is its committed copy. The backup directory is
ignored by Git; deleting the AutoDL instance before verifying that local
directory would discard the recoverable LoRA weights.

The backup does not include the bf16 base-model snapshot. It is recoverable,
but not a standalone offline bundle: download the exact frozen revision first,
load that snapshot explicitly, and then attach checkpoint 0015. Do not rely on
the source-machine path embedded in `adapter_config.json`. The following
minimal check is intentionally unrelated to tau2 and does not access a split:

```bash
export RESTORE_BASE=/path/to/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export RESTORE_ADAPTER=/path/to/backed-up/checkpoints/0015
.venv-trainer/bin/python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = os.environ["RESTORE_BASE"]
adapter = os.environ["RESTORE_ADAPTER"]
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    base,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(
    model,
    adapter,
    local_files_only=True,
    is_trainable=False,
)
inputs = tokenizer("Reply with one word: ready", return_tensors="pt").to("cuda:0")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=1, do_sample=False)
assert output.shape[-1] == inputs["input_ids"].shape[-1] + 1
PY
```

Recorded recovery smoke: checkpoint 0015 was copied to
`/root/autodl-tmp/restore-proof-cp0015-r1/`, loaded against the explicit pinned
base on the RTX PRO 6000, and generated one token. The adapter hash remained
`1018931f9483c71ae20fbd59c76ab6a0c73137d4aefe9c8ad823175931b2c898`;
`results/gpu/restore-cp0015-r1/restore_manifest.json` is the byte-identical
result copied back to the Mac.

Confirm again that the test endpoint returns 403 and that no final-results
directory exists. Stop here and request the exact approval string
`FINAL_TEST_APPROVED`.

## 8. Final 2x2: approval-gated

The exact approval string was received on 2026-07-29. D27 freezes the dedicated
final-evaluation entry point before the official simulation campaign.
`run_ablation` still has no test option, and the ART shim remains locked: it is
a training protocol bridge and cannot layer H2 onto a native evaluation.
Confirm that its test endpoint continues to return 403 before and after the
campaign.

D28 records one narrower deviation after approval and after every experimental
choice was frozen: a legacy dev-report reproducibility check instantiated
tau2's default `base` task objects, which include the 40 test tasks, although it
used only committed dev IDs and produced no test episode, metric, or selection
signal. The dev reporter now requests `train` explicitly. Do not describe the
state before the formal campaign as test-object-unread; the defensible claim is
one disclosed post-approval object load and one official test simulation
campaign.

The one final experiment is 40 official test tasks × 8 trials × four frozen
cells: base/H0, base/H2, RL/H0, RL/H2. All cells share the pinned bf16 base,
vLLM process, tokenizer, chat template, tool parser, simulator parameters,
seeds, hardware, and concurrency. There is no retuning after any test output.
Only a logged infrastructure failure may rerun the same seed.

Start one pinned ART vLLM runtime process that exposes the base and selected
LoRA as two static aliases. Run the final entry point's preflight mode first;
it does not resolve or instantiate a benchmark split. It must verify repository
and submodule commits, model and adapter hashes, the serving manifest, both
model cards, and one automatic non-benchmark tool call from each alias.

The formal command receives only paths and the exact approval through the
environment. Split, cells, order, trials, seeds, temperatures, completion
length, simulator, and concurrency are constants rather than CLI choices. It
writes raw trajectories outside the repository. Existing output is accepted
only for an exact protocol-hash resume; otherwise the runner fails without
overwriting anything. Do not inspect native-runner reward lines or compute
cross-cell summaries until all four cells pass the completeness gate.

Use these exact paths and keep the serving process alive for preflight, smoke,
and the full campaign:

```bash
cd /root/autodl-tmp/work/service-agent-rl
FINAL_BASE=/root/autodl-tmp/cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
FINAL_ADAPTER=/root/autodl-tmp/art/grpo-4b-qwen3coder-r1/service-agent/models/grpo-4b-qwen3coder-r1/checkpoints/0015
FINAL_SERVING_DIR=/root/autodl-tmp/runs/final-serving-r1
FINAL_SERVING=$FINAL_SERVING_DIR/serving_manifest.json
FINAL_RAW=/root/autodl-tmp/runs/final-2x2-r1
FINAL_SMOKE=/root/autodl-tmp/runs/final-native-smoke-r1
mkdir -p "$FINAL_SERVING_DIR" /root/autodl-tmp/logs

PREPARED_TMP=$(mktemp "$FINAL_SERVING_DIR/.prepared.XXXXXX")
uv run python -m service_agent.eval.final_serving manifest \
  --snapshot "$FINAL_BASE" --adapter "$FINAL_ADAPTER" > "$PREPARED_TMP"
mv "$PREPARED_TMP" "$FINAL_SERVING_DIR/prepared_manifest.json"

tmux new-session -d -s final-serving-r1 \
  "cd /root/autodl-tmp/work/service-agent-rl && exec uv run python \
  -m service_agent.eval.final_serving launch --snapshot '$FINAL_BASE' \
  --adapter '$FINAL_ADAPTER' > /root/autodl-tmp/logs/final-serving-r1.log 2>&1"
```

Wait for `curl -fsS http://127.0.0.1:8100/health`, then bind the two
non-benchmark tool-call probes to the exact launch manifest:

```bash
PROBE_TMP=$(mktemp "$FINAL_SERVING_DIR/.probe.XXXXXX")
uv run python -m service_agent.eval.final_serving probe \
  --snapshot "$FINAL_BASE" --adapter "$FINAL_ADAPTER" > "$PROBE_TMP"
mv "$PROBE_TMP" "$FINAL_SERVING_DIR/probe_manifest.json"

SERVING_TMP=$(mktemp "$FINAL_SERVING_DIR/.serving.XXXXXX")
uv run python -m service_agent.eval.final_serving finalize \
  --prepared-manifest "$FINAL_SERVING_DIR/prepared_manifest.json" \
  --probe-manifest "$FINAL_SERVING_DIR/probe_manifest.json" > "$SERVING_TMP"
mv "$SERVING_TMP" "$FINAL_SERVING"

curl -s -o /dev/null -w "%{http_code}\n" \
  "http://127.0.0.1:8000/scenarios?domain=telecom&split=test"  # must print 403
```

Preflight does not load a task. Smoke uses the three frozen dev tasks and is an
operational gate only:

```bash
uv run python -m service_agent.eval.run_final --preflight \
  --out "$FINAL_RAW" --serving-manifest "$FINAL_SERVING" \
  --base-snapshot "$FINAL_BASE" --adapter "$FINAL_ADAPTER"
uv run python -m service_agent.eval.run_final --smoke \
  --out "$FINAL_SMOKE" --serving-manifest "$FINAL_SERVING" \
  --base-snapshot "$FINAL_BASE" --adapter "$FINAL_ADAPTER"
jq '{status,completed_cells}' "$FINAL_SMOKE/smoke_manifest.json"
```

Run the approved campaign in a separate tmux session. The log contains native
progress and must not be tailed or summarized before all four cells complete:

```bash
tmux new-session -d -s final-2x2-r1 \
  "cd /root/autodl-tmp/work/service-agent-rl && \
  FINAL_TEST_APPROVAL=FINAL_TEST_APPROVED uv run python \
  -m service_agent.eval.run_final --out '$FINAL_RAW' \
  --serving-manifest '$FINAL_SERVING' --base-snapshot '$FINAL_BASE' \
  --adapter '$FINAL_ADAPTER' \
  --smoke-manifest '$FINAL_SMOKE/smoke_manifest.json' \
  > /root/autodl-tmp/logs/final-2x2-r1.log 2>&1"
```

While it runs, inspect only process liveness, GPU use, `status`,
`completed_cells`, and the number of persisted simulations. Do not read
per-episode rewards. If an infrastructure failure requires a resume, use the
same command while the same serving process remains alive; the runner refuses
every protocol, process, task, or result-hash drift.

Afterward, confirm the shim is still locked, create a SHA-256 index over the
private raw directory, and copy it to the ignored Mac backup. Generate public
artifacts only from that verified backup:

```bash
uv run python -m service_agent.eval.report_factorial generate \
  --raw-root checkpoints/autodl-final-2x2-r1 \
  --protocol-manifest checkpoints/autodl-final-2x2-r1/final_manifest.json \
  --serving-manifest checkpoints/autodl-final-serving-r1/serving_manifest.json
uv run python -m service_agent.eval.report_factorial check \
  --raw-root checkpoints/autodl-final-2x2-r1 \
  --protocol-manifest checkpoints/autodl-final-2x2-r1/final_manifest.json \
  --serving-manifest checkpoints/autodl-final-serving-r1/serving_manifest.json
uv run pytest
uv run ruff check
```
