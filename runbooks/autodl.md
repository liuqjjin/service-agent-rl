# AutoDL training runbook

Everything the GPU box runs, in order, with what you should see at each step.
The Mac never trains; the GPU box never decides protocol. Both follow this file.

Topology:

```
art_tau_train.py (trainer venv)         # rollouts + GRPO updates
  ├── HTTP → shim :8000 (service venv)  # tau2 env + deepseek user sim + evaluator
  └── ART LocalBackend (same process)   # vLLM serving the 4B + LoRA, Unsloth training
W&B                                     # curves, configs, evidence
```

## 1. Instance

- GPU: one A100 80GB (safest for 4B LoRA GRPO + vLLM on one card). A 48GB card
  can work with `--group-size 4` and reduced `max_tokens`; try only after the
  80GB path is proven.
- Image: newest PyTorch 2.x / CUDA 12.x image with Python 3.12 available
  (`python3.12 -V`; otherwise install via uv below, which manages its own).
- Disk: ≥ 100 GB (base model ~8 GB, vLLM cache, LoRA checkpoints, results).
- Region/price: whatever is available; nothing here depends on it.

## 2. One-time setup

```bash
# tools
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# repo, pinned submodules included. service-agent-rl is a private repo, so the
# box needs credentials first: `gh auth login` (or a PAT in the clone URL).
# tau2-bench resolves to a fork because the pinned commit is upstream cf71a80
# plus one gym fix that does not exist in sierra-research's repo (UPSTREAM.md).
git clone --recurse-submodules https://github.com/liuqjjin/service-agent-rl.git
cd service-agent-rl
git submodule status   # expect tau2-bench at 2822d90, ART at 828b839

# secrets -- create .env in the repo root (never committed):
#   DEEPSEEK_API_KEY=...      # fixed user simulator
#   WANDB_API_KEY=...         # public evidence
# China network: also export HF_ENDPOINT=https://hf-mirror.com in ~/.bashrc

# venv 1: environment service (tau2 + this repo, no training deps)
uv sync                                   # creates .venv used by uv run

# venv 2: trainer (ART with training backend), isolated from the service venv
uv venv .venv-trainer --python 3.12
source .venv-trainer/bin/activate
uv pip install -e third_party/ART[backend] -e . 
deactivate
# If ART's backend extra conflicts with tau2's pins inside one venv, this
# two-venv split is exactly why: the trainer talks to the service over HTTP
# and never imports tau2.
```

## 3. tmux layout

```bash
tmux new -s train
# window 0: shim        window 1: trainer        window 2: watch
```

Window 0 — the environment service:

```bash
cd service-agent-rl
uv run python -m service_agent.serve.tau2_shim     # SHIM_PORT=8000 default
```

Expected log: uvicorn startup on 127.0.0.1:8000. Then verify from window 2:

```bash
curl -s localhost:8000/health                       # {"status":"ok"}
curl -s "localhost:8000/scenarios?domain=telecom&split=train-core" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["scenarios"]))'   # 54
curl -s "localhost:8000/scenarios?domain=telecom&split=test" -o /dev/null -w "%{http_code}\n"  # 403 -- must stay 403 during training
```

## 4. Preflight (do not skip; ~20 minutes total)

### 4a. Smoke rollout + one update

```bash
source .venv-trainer/bin/activate
python -m service_agent.training.art_tau_train --smoke \
    --base-model Qwen/Qwen3.5-4B --model-name smoke-$(date +%m%d)
```

Expected: model registers (first run downloads the base model), a
`step 0: {"groups": 1, "mixed": ...}` line, one train step completing without
OOM, and a checkpoint under ART's art/ directory. Rollouts hitting the shim
appear in window 0's log. Cost: a few cents of DeepSeek.

### 4b. Step-0 logprob consistency (the gate that saves you a week)

ART's LocalBackend serves the policy over an OpenAI endpoint. Get its address
and the inference model name:

```bash
python - <<'EOF'
import asyncio, art
from art.local import LocalBackend
async def main():
    backend = LocalBackend()
    model = art.TrainableModel(name="smoke-<date>", project="service-agent", base_model="Qwen/Qwen3.5-4B")
    await model.register(backend)
    print("base_url:", model.inference_base_url)
    print("model:", model.get_inference_name())
asyncio.run(main())
EOF

python -m service_agent.training.logprob_check \
    --api-base <base_url> --served-model <model> --hf-model Qwen/Qwen3.5-4B
```

PASS criteria are printed (ratio mean within 2% of 1.0, <2% of tokens outside
the 0.8-1.2 clip window). On FAIL: the mismatch is chat template or
tokenization, not hyperparameters. Compare `--chat-template-kwargs` with what
vLLM applies (thinking flags are the usual culprit), fix, re-run until PASS.
Do not start training on a FAIL.

## 5. Main run (direct dual-control GRPO)

```bash
source .venv-trainer/bin/activate
python -m service_agent.training.art_tau_train \
    --model-name grpo-4b-r1 --base-model Qwen/Qwen3.5-4B \
    --group-size 8 --groups-per-step 4 --max-turns 30 \
    --steps 60 --learning-rate 5e-6 --loss-fn ppo --val-every 10 \
    2>&1 | tee logs/grpo-4b-r1.log
```

Rough expectations (estimates, not promises):
- 32 rollouts per step, gathered concurrently; a step is dominated by
  user-simulator latency: ~10-20 min/step, so 60 steps ≈ 10-20 h.
- VRAM: ~35-55 GB (vLLM KV cache + LoRA training). OOM → `--group-size 4`
  first, then lower vLLM `max_model_len` via ART config.
- DeepSeek cost: ~$0.003/rollout ≈ $6-8 for the full run.
- Healthy logs: `mixed` groups > 0 most steps; dev avg_reward drifting up by
  step 20-30; W&B shows reward, KL, loss curves under project `service-agent`.

### Go / no-go while it runs

| Symptom | Meaning | Action |
|---|---|---|
| `mixed: 0` with `all_zero` dominating for >10 steps | reward too sparse for GRPO | stop; go to §6 SFT bridge |
| logprob check failed earlier | template mismatch | fix template, never tune lr around it |
| shim window shows 5xx / trainer stalls | env service died | restart shim; training resumes (ART checkpoints steps) |
| DeepSeek 429s | user-sim rate limit | lower `--groups-per-step`; resume |
| loss spikes + reward collapse | update too hot | halve lr; consider `--kl-penalty-coef 0.01` |
| CUDA OOM | memory | `--group-size 4`, restart (resumes from last step) |

Resume after any interruption: re-run the same command; the script continues
from `model.get_step()`.

## 6. Fallback: teacher SFT bridge, then GRPO

Only if direct GRPO shows all-zero groups persistently.

```bash
# 1. Teacher episodes (can run on the Mac against the local 35B, or here):
uv run python -m service_agent.eval.run_ablation --arm h0 --tasks train-core \
    --trials 2 --agent-llm "openai//Users/lqj/local-llm/models/Qwen3.6-35B-A3B-4bit" \
    --agent-api-base http://127.0.0.1:8399/v1 --out results/teacher/r1

# 2. Filter to reward-1.0, governance-clean episodes and export chat JSONL:
uv run python -m service_agent.training.sft_prepare \
    --results results/teacher/r1/results.json --out data_protocol/sft_teacher.jsonl

# 3. SFT on the GPU box, then GRPO with the warm-started checkpoint:
source .venv-trainer/bin/activate
python - <<'EOF'
import asyncio, json, art
from art.local import LocalBackend

async def main():
    backend = LocalBackend()
    model = art.TrainableModel(name="grpo-4b-r1", project="service-agent", base_model="Qwen/Qwen3.5-4B")
    await model.register(backend)
    trajectories = []
    for line in open("data_protocol/sft_teacher.jsonl"):
        s = json.loads(line)
        trajectories.append(art.Trajectory(
            messages_and_choices=s["messages"], tools=s["tools"], reward=1.0))
    await model.train_sft(trajectories)
asyncio.run(main())
EOF
python -m service_agent.training.art_tau_train --model-name grpo-4b-r1 ...  # as §5
```

## 7. Final 2x2 evaluation (test split, exactly once)

All four cells run on THIS box with the same vLLM stack. Do not run until the
Mac-side dev ablation has frozen Hbest and this file's owner says go.

```bash
# Serve both policies with one vLLM instance (base + trained LoRA):
source .venv-trainer/bin/activate
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-4B \
    --enable-lora --lora-modules rl=<path-to-final-lora-checkpoint> \
    --port 8300 --chat-template-kwargs '{"enable_thinking": false}' &

# Four cells, native runner, 40 test tasks x 8 trials each:
for arm_model in "h0 Qwen/Qwen3.5-4B" "h2 Qwen/Qwen3.5-4B" "h0 rl" "h2 rl"; do
  set -- $arm_model
  uv run python -m service_agent.eval.run_ablation --arm $1 --tasks test --trials 8 \
      --agent-llm "openai/$2" --agent-api-base http://127.0.0.1:8300/v1 \
      --user-llm deepseek/deepseek-v4-pro --seed 42 --max-concurrency 4 \
      --out results/final/$1_$2
done
```

(`--tasks test` requires a one-line addition to run_ablation that is
deliberately absent today; it lands together with the final-eval sign-off so
the test split cannot be touched by accident. If Hbest turns out to be H1,
substitute h1 above.)

Ship back: `results/final/`, the W&B project link, and the last LoRA
checkpoint path. The Mac side computes the factorial table, bootstrap CIs,
and the reports.

## 8. What to send back after training

- W&B run URLs (train + dev curves)
- `logs/grpo-4b-r1.log`
- The final checkpoint directory (or its AutoDL path)
- `results/final/` if §7 was run
