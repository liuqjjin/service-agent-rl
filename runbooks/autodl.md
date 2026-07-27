# AutoDL GRPO runbook

This is the reproducible GPU procedure. The current instance is one RTX 4090
with 49,140 MiB VRAM and a 200 GB data disk at `/root/autodl-tmp`. All large
state lives on that disk. Services bind to localhost only.

The training protocol has three lineages:

1. `preflight-r1`: step-0 token/logprob gate, then rollout-only; no update.
2. `smoke-r1`: one disposable update, authorized by preflight.
3. `grpo-4b-r1`: fresh formal run, authorized by both earlier manifests.

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
3. registers ART at step 0;
4. samples exact vLLM prompt/completion token IDs and logprobs;
5. closes vLLM before loading the tokenizer and bf16 reference model from the
   exact manifest-recorded snapshot, with local-only loading;
6. requires byte-identical prompt token IDs, mean importance ratio within
   2% of 1, and at most 2% outside the PPO clip window;
7. reopens the untouched step-0 checkpoint;
8. runs eight train-core episodes through strict replay with reward finalized
   exactly once;
9. exits with `final_step=0`.

The manifest records the exact Python argv, download endpoint, composite
semantic-contract hash, and separate hashes for the system prompt, tools, and
tokenizer chat template. It also records the selected CUDA runtime and
bootstrap plus locked `ninja` paths, versions, and SHA-256 values; smoke and
formal phases reject drift in any of them.

```bash
cd /root/autodl-tmp/work/service-agent-rl
source .venv-trainer/bin/activate
python -m service_agent.training.art_tau_train \
  --phase preflight --run-name preflight-r1 \
  --art-path /root/autodl-tmp/art/preflight \
  --out /root/autodl-tmp/runs/preflight-r1 \
  --hf-cache /root/autodl-tmp/cache/huggingface \
  --group-size 4 --max-turns 30 \
  --max-completion-tokens 1024 --max-model-len 16384 \
  --rollout-concurrency 4 --gpu-memory-utilization 0.68 \
  2>&1 | tee /root/autodl-tmp/logs/preflight-r1.log
deactivate
```

Gate artifact:

`/root/autodl-tmp/runs/preflight-r1/preflight_manifest.json`

Do not continue unless its status is `passed`, both token and logprob gates
pass, strict replay is true, test locking is true, and both steps are zero.

## 5. One-update disposable smoke

```bash
source .venv-trainer/bin/activate
python -m service_agent.training.art_tau_train \
  --phase smoke --run-name smoke-r1 \
  --art-path /root/autodl-tmp/art/smoke \
  --out /root/autodl-tmp/runs/smoke-r1 \
  --hf-cache /root/autodl-tmp/cache/huggingface \
  --preflight-manifest \
    /root/autodl-tmp/runs/preflight-r1/preflight_manifest.json \
  --group-size 4 --max-turns 30 \
  --max-completion-tokens 1024 --max-model-len 16384 \
  --rollout-concurrency 4 --gpu-memory-utilization 0.68 \
  2>&1 | tee /root/autodl-tmp/logs/smoke-r1.log
deactivate
```

The smoke runs two groups and must observe within-group reward variance before
training. Its manifest must show step 0 to step 1, at least one trainable group,
a real optimizer update, one checkpoint, strict replay, a W&B URL, and no OOM.
After its checkpoint and log are backed up, this ART lineage is disposable and
must never seed the formal run.

## 6. Formal GRPO

The formal configuration starts with group size 4, two task groups per update,
four concurrent rollouts, bf16 LoRA, and a 16,384-token context. Preflight
records p50/p95/p99/max prompt lengths from all 240 committed dev episodes and
refuses the run if the context cannot cover the observed maximum plus a
governance-feedback buffer and 1,024 completion tokens.

```bash
tmux new-session -d -s grpo
tmux send-keys -t grpo \
  'cd /root/autodl-tmp/work/service-agent-rl && source .venv-trainer/bin/activate && python -m service_agent.training.art_tau_train --phase train --run-name grpo-4b-r1 --art-path /root/autodl-tmp/art/formal --out /root/autodl-tmp/runs/grpo-4b-r1 --hf-cache /root/autodl-tmp/cache/huggingface --preflight-manifest /root/autodl-tmp/runs/preflight-r1/preflight_manifest.json --smoke-manifest /root/autodl-tmp/runs/smoke-r1/smoke_manifest.json --group-size 4 --groups-per-step 2 --max-turns 30 --max-completion-tokens 1024 --max-model-len 16384 --rollout-concurrency 4 --gpu-memory-utilization 0.68 --steps 60 --learning-rate 5e-6 --loss-fn ppo --val-every 5 --val-trials 2 2>&1 | tee /root/autodl-tmp/logs/grpo-4b-r1.log' C-m
```

The command is resume-safe: rerunning the exact command requires the same
semantic-contract hash and the manifest step to equal ART's checkpoint step.
Every update writes the manifest atomically.

The selected checkpoint rule is fixed before training: highest frozen-dev
average reward among scheduled checkpoints; ties choose the earliest step.
Every checkpoint is evaluated with the same dev task/trial seeds (common random
numbers). The test split is not involved.

Stop immediately on any of these:

- token/logprob gate failure;
- test endpoint no longer returns 403;
- prompt/tool/template/revision hash drift;
- multi-tool choice (fails before any call executes);
- missing strict replay or reward-finalized-once marker;
- evaluator or shim 5xx;
- at most one mixed-reward group over ten consecutive update steps;
- a second controlled OOM.

For the first OOM only: terminate residual GPU processes, keep the last good
checkpoint and logs, and retry once after reducing only rollout concurrency,
vLLM memory utilization, or context length while retaining the measured
context floor. Do not change learning rate, reward, prompts, tools, model
revision, group size, or split. A second OOM is a hard gate.

SFT is not automatic. Sparse reward stops the formal driver and requires a
recorded protocol decision before any teacher data or SFT update is created.

## 7. Training handoff and backup

Before any final test run, copy these to the Mac and verify checksums:

- `/root/autodl-tmp/runs/preflight-r1/`
- `/root/autodl-tmp/runs/smoke-r1/`
- `/root/autodl-tmp/runs/grpo-4b-r1/`
- the selected LoRA checkpoint under `/root/autodl-tmp/art/formal/`
- `/root/autodl-tmp/logs/`
- W&B run URLs
- a SHA-256 manifest for every copied file

Confirm again that the test endpoint returns 403 and that no final-results
directory exists. Stop here and request the exact approval string
`FINAL_TEST_APPROVED`.

## 8. Final 2x2: approval-gated

The final runner intentionally has no test option yet. Only after the exact
approval string is received may one commit add the locked final-evaluation
entry point and set `SHIM_ALLOW_EVAL_SPLITS=1` for that run.

The one final experiment is 40 official test tasks × 8 trials × four frozen
cells: base/H0, base/H2, RL/H0, RL/H2. All cells share the pinned bf16 base,
vLLM process, tokenizer, chat template, tool parser, simulator parameters,
seeds, hardware, and concurrency. There is no retuning after any test output.
Only a logged infrastructure failure may rerun the same seed.

Afterward, relock the shim, generate the factorial report and bootstrap
intervals from the four raw result directories, back up all artifacts and
checksums to the Mac, and rerun tests, Ruff, and report reproducibility.
