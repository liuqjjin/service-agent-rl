# Official GPU artifacts

These are the byte-identical manifests and process exit codes from the three
official manifest-v3 lineages on the RTX PRO 6000:

- `preflight-qwen3coder-r1`: passed, process exit 0;
- `smoke-qwen3coder-r1`: passed, process exit 0;
- `grpo-4b-qwen3coder-r1`: terminal `stopped_sparse_reward`, process exit 1.

The formal exit is nonzero by design: the driver persists the terminal sparse
manifest and then raises the protocol error that prevents an accidental
resume. Interpret the manifest and exit together.

`BACKUP_SHA256SUMS` indexes the 292 files copied from the complete backed-up
state of all three ART lineage directories, including the terminal
sparse-stopped formal state, their run directories, the official logs, and the
shim log. The remote and local lists matched byte for byte; the list itself
has SHA-256
`a5c9d4c9e630dbfc422a41f4fb37298eef781466e0877d2b185226665107f724`.
The ignored Mac backup is
`checkpoints/autodl-backup-2026-07-28-qwen3coder-r1/` (about 3.3 GB).
Checkpoint directories 0000–0024 each contain all eight expected files, and
both the selected 0015 and latest 0024 safetensors open with 256 tensors and
LoRA rank 16.

`WANDB_COUNTER_AUDIT.json` reconciles one ART logging detail against the hashed
formal `history.jsonl` and `state.json`. The 24 rollout records and 24
backend-train records both carry group counters, so ART/W&B cumulatively shows
96 submitted and 10 trainable groups. The unique manifest positions and the
backend-record subset both give 48 submitted, 5 trainable, and 445 gradient
steps. Gradient steps are backend-only and are not doubled.

Weights, tensors, trajectories, and raw logs are intentionally not committed.
The selected adapter is present in the backup at checkpoint 0015 with SHA-256
`1018931f9483c71ae20fbd59c76ab6a0c73137d4aefe9c8ad823175931b2c898`.
It is loaded on the pinned base-model revision recorded in every manifest.
The backup does not contain that base model and is therefore not a standalone
offline bundle. A post-backup recovery smoke copied 0015 to a separate
directory, explicitly loaded the exact pinned bf16 base, attached the relocated
adapter, and generated one non-benchmark token on the RTX PRO 6000.
`restore-cp0015-r1/restore_manifest.json` is the byte-identical remote result.

Verify the committed evidence and generated report from the repository root:

```bash
shasum -a 256 -c results/gpu/CHECKSUMS.sha256
uv run python -m service_agent.eval.report_grpo --check
```

On the Mac that owns the ignored backup, rehash all 292 source files too:

```bash
uv run python -m service_agent.eval.report_grpo --check \
  --backup-root checkpoints/autodl-backup-2026-07-28-qwen3coder-r1
```

All three manifests record `test_split_locked=true`. This directory contains
training and frozen-dev selection evidence only, not a final test result.
