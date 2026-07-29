# Decisions

Execution-time judgment calls, with the reasoning. Newest last.

## D1. Fixed user simulator: DeepSeek official API, `deepseek-v4-pro`

Non-thinking mode, temperature 0.0, identical parameters across every
experimental cell (H0/H1/H2, base and RL). Provider, model name, request
parameters, date, and raw outputs are logged with every run. Fallback if it
proves incompatible with the pinned tau2/LiteLLM path: DashScope
`qwen3.7-max-2026-06-08` (dated snapshot), recorded here with the failure.
Secondary simulator-sensitivity checks (dev split only) use that same Qwen
snapshot. Rationale: the user simulator is part of the benchmark definition —
changing it mid-project invalidates every earlier number; tau2's default
(gpt-4.1) is replaced before any formal run rather than after.

## D2. Local Qwen3.6-35B-A3B (4-bit) is a tool, not a subject

Used for smoke tests and failure analysis. It is reserved as a possible
teacher for the SFT bridge only after a separate protocol decision. Never
substituted for the formal Qwen 4B policy in any reported cell: within each
model row, native and governed cells share identical weights. The base and RL
rows share the base revision, tokenizer, chat template, tool parser, and
inference stack; the RL row additionally loads its selected adapter.

## D3. Dev selection is deterministic, not seeded-random

Stratified by (issue family, needs-agent-write, needs-user-action) with
largest-remainder allocation, then systematic sampling over reference-action
count inside each stratum. A seeded random sample would have been defensible
but invites "why this seed"; systematic sampling has no free parameter and
provably covers the short/long spectrum.

## D4. No separate "whitelist prompt builder" module

The native LLMAgent already builds its prompt from policy + tool schemas +
conversation only, and ART's rollout builds from env.info policy +
observation — both are whitelist-by-construction. Wrapping them in another
builder would add a layer without adding safety. The enforcement point is
`leakage.py` (serialized-substring detection) applied in tests to every prompt
surface we ship, and later to the shim's observations. If a future surface
needs assembly from raw Task fields, that is when a builder earns its place.

## D5. Leak detection matches composite serializations, not leaf values

Reference actions are made of tool names; tool names legitimately appear in
every prompt via the policy and tool schemas. Matching individual leaf values
produced false positives on the clean native prompt (tool names, the user's
own phone number). Matching whole-field JSON, per-action JSON, str() forms,
and task-ID fault segments catches every realistic leak path while staying
quiet on legitimate text. Trade-off accepted: a leak that copies a single
argument value verbatim without structure would slip through; no such surface
exists in this codebase.

## D6. One submodule fix commit, not two

Seed and thread-lifecycle fixes share the file, the tests, and the PR; the
combined commit message tells the full story. Upstream can squash or split as
they prefer.

## D7. Compatibility matrix results and the serving stack

Fixed simulator (deepseek-v4-pro, official API) verified through tau2's
LiteLLM path: tool calls parse, user tools execute (8-13 device-tool calls
per smoke simulation), multi-turn history holds over 40-68 message episodes,
temperature-0 repeats are stable, and non-thinking mode works only via
`extra_body={"thinking": {"type": "disabled"}}` (the top-level `thinking`
parameter is ignored; default mode leaks reasoning_content). User-simulator
cost measured at ~$0.003 per simulation.

Local policy serving via mlx_lm.server (uv tool, v0.31.3): works, with two
traps. The request's model id must be exactly the served `--model` value or
the server downloads that id from HF instead; and Qwen thinking mode must be
disabled per request via `chat_template_kwargs`, or long tasks
deterministically produce think-only empty completions that fail tau2's
message validation (found on the 8-fault MMS smoke task, reproduced across
all 4 runner retries, fixed by disabling thinking).

Smoke evidence (3 dev-family tasks, local Qwen3.6-35B agent + deepseek user):
2/3 reward=1.0 with thinking accidentally on; the failure was the empty
completion above, not a protocol issue. Full matrix rerun with thinking off
recorded in results/compat/.

## D8. Mixed text+tool-call candidates are sanitized, not denied

Qwen3.5-4B habitually narrates while calling a tool (~4 mixed messages per
episode). The policy says one or the other (main_policy.md:9), and the first
H1 run denied such candidates with feedback and regeneration: 103 denials
over a 3-episode smoke, models rarely complied, turns burned to max-steps,
and reward halved versus H0. That punished a formatting habit, not a
business violation -- the native orchestrator routes a mixed message to the
environment and the user never sees its text, and telecom's reward basis has
no COMMUNICATE component. The gate now strips the text, keeps the call,
records the event in the audit (reason mixed_text_stripped), and reserves
deny-and-regenerate for actual business-precondition failures. Smoke after
the change: reward back to H0 parity, and the sole rejection in the run was
a genuine price_not_confirmed with a successful regeneration.

## D9. The formal checkpoint and tokenization contract are one pinned object

The final policy is `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Before GPU work, the downloader
checks that the repository's `main` still resolves to that SHA, downloads it
into a dedicated cache, and verifies the returned snapshot path. ART's
training loader, vLLM engine, and tokenizer are configured for the same
revision; thinking is disabled with `enable_thinking=false`; training and
serving both use bf16. The explicit check is necessary because ART's pinned
training-tokenizer helper does not forward `revision` even though its model
and engine loaders do.

## D10. Step 0 is an update-free gate, not a smoke-training by-product

Preflight registers an untouched LoRA lineage, captures vLLM's exact prompt
and completion token IDs plus rollout logprobs, closes vLLM, and only then
loads the bf16 reference model. Prompt token IDs must match byte-for-byte and
importance ratios must pass before any gradient work. A second registration of the
same untouched step-0 checkpoint runs official train-core episodes through
strict replay without training. Only that manifest can authorize a separate
single-train-call smoke lineage; only preflight plus smoke can authorize a fresh
formal lineage. This replaces the earlier runbook order, which performed a
smoke training call before the purported step-0 check.

The pinned ART tau-bench client also split multiple tool calls from one model
message into separate shim steps. That changes tau2 trajectory semantics. The
training rollout now raises before executing any call when a choice contains
more than one; fail-closed is preferable to optimizing on a trajectory the
native evaluator would interpret differently.

## D11. Audit normalizations are events, not extra allow verdicts

`mixed_text_stripped` remains a separate audit record so the formatting rate
is measurable, but report aggregation removes it from decision counts and
shows it in its own column. The committed H1/H2 allow totals therefore mean
actual candidate verdicts rather than “verdicts plus sanitizer events.”
Audit files are snapshots written with replacement, so repeated cleanup of
one session cannot append duplicate records.

## D12. GPU phases share one complete protocol, and dev uses common seeds

A semantic prompt hash alone is not enough to authorize training. Preflight,
smoke, formal startup, and formal resume now compare the superproject, ART and
tau2 commits; model and revision; tokenizer-derived context budget; bf16
runtime and package versions; user simulator; rollout limits; and optimization
settings. Run names are the only phase-specific field omitted from the
cross-phase comparison. A formal resume additionally requires the same run
name, resolved ART lineage path, checkpoint path, and checkpoint step.
Completed and sparse-reward-stopped lineages are terminal rather than
restartable by repeating the command.

ART reports a group as trainable only when rewards vary within that group.
The driver now uses that exact definition: constant fractional rewards such as
`[0.5, 0.5]` are not called mixed. The single-train-call smoke refuses to train
unless at least one group has variance, and its gate requires ART to report both
a trainable group and positive gradient work.
Formal checkpoint selection uses the same task/trial seeds at every scheduled
dev evaluation, so a checkpoint is not selected merely because it received an
easier random draw.

## D13. ART training and vLLM inference keep separate locked environments

The editable ART checkout does not install vLLM into the trainer environment.
Its launcher resolves
`third_party/ART/vllm_runtime/.venv/bin/art-vllm-runtime-server`, whose
dependencies come from ART's own `vllm_runtime/uv.lock`. Treating vLLM as a
trainer dependency made the original runbook verification command fail and
would have recorded `vllm: null` in the experiment manifest. The GPU setup now
builds that isolated environment explicitly. Every phase records both the
trainer package set and the runtime's Python, ART runtime, FlashInfer, torch,
Transformers, and vLLM versions; phase and resume gates reject drift in either
environment. ART's lock intentionally overrides NumPy to `<2`; the resulting
OpenCV metadata warning is not resolved by changing NumPy outside the lock.

## D14. The AutoDL host uses a verified Hugging Face transport mirror

The instance cannot connect to `huggingface.co:443`; an independent request
times out before TLS. `https://hf-mirror.com` is reachable and its model-info
response resolves `Qwen/Qwen3.5-4B` `main` to the frozen
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` revision. The mirror is only the
download transport: the driver still rejects any different commit, validates
the snapshot directory name against the pin, and forces later loads offline.
Manifest schema 2 records the endpoint and exact invocation, plus separate
system-prompt, tool-schema, and tokenizer-template hashes in addition to the
composite semantic hash.

## D15. Token-budget replay uses the template-level tool-call shape

Committed tau2 trajectories store tool arguments as JSON objects. The first
GPU preflight exposed that the budget replayer converted those objects to
OpenAI JSON strings, while the pinned Qwen template iterates
`tool_call.arguments|items`. ART performs the same string-to-object
normalization before training tokenization in
`third_party/ART/src/art/preprocessing/tokenize.py`. The budget replayer now
builds that template-level mapping directly. This changes neither the
trajectory nor the OpenAI request; it makes the context measurement follow the
actual ART/Qwen rendering path instead of failing before model registration.

## D16. Transformers 5 chat rendering returns a `BatchEncoding`

The GPU trainer has Transformers 5.2.0. Its Qwen tokenizer returns a
`BatchEncoding` with `input_ids` and `attention_mask` from
`apply_chat_template`, whereas the Mac service stack and older APIs commonly
return the ID list directly. Taking `len()` of that mapping produced a
plausible-looking but false two-token budget for all 4,919 measured prefixes;
iterating it in the reference logprob gate would likewise have read field
names instead of IDs. One shared adapter now extracts and validates
`input_ids` from either API shape. The context budget and the exact-token
logprob comparison both use that adapter.

## D17. ART and vLLM load the verified local snapshot path

Unsloth 2026.3.3 calls `HfFileSystem.glob` for a model ID even when its
`revision` argument is pinned; that metadata request fails after the driver
intentionally enables offline mode. Relaxing offline mode would also reopen
ART's training-tokenizer helper, which does not pass a revision. The
`TrainableModel` therefore receives the already verified local snapshot path
as `base_model`. Unsloth recognizes it as a directory, ART's tokenizer reads
the same files, and the dedicated vLLM process sees that identical path.
Manifests and W&B config continue to identify the canonical model ID and
revision separately.

## D18. vLLM sleep mode binds a verified CUDA runtime, not TileLang's stub

ART's isolated vLLM runtime applies its DeepSeek-V4 patches before it knows
which model will be served. Those patches import TileLang, which loads
`libcudart_stub.so`. vLLM 0.23 implements sleep mode by taking the first
`libcudart` path in `/proc/self/maps`; on the AutoDL image that was the stub,
which has no `cudaDeviceReset`, rather than the real CUDA 12 runtime already
installed in the same environment. Preloading another library does not change
that map order, and disabling sleep mode is not valid on the single-GPU shared
training path.

Before ART registration, the driver now locates the isolated runtime's own
`libcudart.so.12`, verifies its SHA-256 and `cudaDeviceReset` symbol, and
injects a project-owned `sitecustomize` that overrides only vLLM's
`find_loaded_library("libcudart")` result. A separate runtime process then
loads all ART patches and instantiates `CudaRTLibrary`; model loading cannot
start unless that probe selects the verified file. The CUDA library and
bootstrap paths and hashes are part of the manifest's runtime provenance, so
preflight, smoke, formal training, and resume all reject drift.

## D19. FlashInfer JIT uses the `ninja` already locked with vLLM

The isolated ART runtime contains `ninja==1.13.0` and its executable, but ART
starts `art-vllm-runtime-server` by absolute path while copying the trainer's
`PATH`. That does not activate the runtime virtual environment. FlashInfer's
first sampling-kernel warmup therefore loaded the model and then failed with
`FileNotFoundError: ninja`, even though the required build tool was already
installed from `vllm_runtime/uv.lock`.

The driver now prepends only the isolated runtime's `bin` directory to the
child process path. The same pre-GPU probe resolves `ninja`, runs
`ninja --version`, and requires its base version to match the locked Python
distribution (`1.13.0`; the wheel's binary adds a Kitware feature suffix).
The distribution version, full binary version, executable path, and SHA-256
are recorded beside the CUDA bootstrap provenance. No system package or
unlocked Python dependency is added.

## D20. The logprob reference loads only the verified local snapshot

The first real preflight showed that passing the canonical model ID to the
reference tokenizer and model could start a second download in the default
Hugging Face hub cache. The verified snapshot deliberately lives at an
explicit cache path outside that default layout. In addition, setting
`HF_HUB_OFFLINE` after `huggingface_hub` has already been imported does not
retroactively change every module-level offline constant.

The reference gate now receives the snapshot path already recorded in the
preflight manifest, requires its directory name to equal the frozen model
revision, and passes that path to both Transformers loaders with
`local_files_only=True`. It no longer resolves the canonical ID or accepts a
revision argument at this boundary. The canonical ID remains in the manifest
for provenance; the verified path is the only source from which the
tokenizer and bf16 reference weights can be loaded.

## D21. A project adapter closes ART's Transformers 5 mask-signature drift

ART's pinned mask patch
(`third_party/ART/src/art/transformers/patches.py:14-34`) expects the older
argument order without `cache_position`.
Transformers 5.2 inserts `cache_position` before `past_key_values`. During the
first reference forward, ART's wrapper therefore received a
`Qwen3_5DynamicCache` as `position_ids` and tried to read its nonexistent
`shape`. Disabling the cache would hide that one failure while leaving ART's
intended 3D `position_ids` normalization ineffective during training.

The submodule stays untouched. After importing ART, the training driver now
requires both exact pinned signatures and requires ART's wrapper to be the
active function. It then installs a project-owned adapter with the
Transformers 5 parameter order, preserves the 3D normalization, and delegates
to the original Transformers implementation by keyword. The two observed
parameter orders and installation status are recorded in every manifest and
are part of the cross-phase protocol contract. Unknown patches or future API
drift fail before model registration.

## D22. Bound the exact logprob workspace after the first controlled OOM

The first real smoke attempt completed all eight official rollouts with
strict replay, reward finalized once, no multi-tool calls, and one mixed
reward group. Its first training forward then failed at
`torch.logsumexp(chunk_logits)` while requesting a 970 MiB temporary tensor.
ART had packed 102 trainable exchanges into 102 rows of length 12,288, so
reducing rollout concurrency or vLLM memory would not reduce this learner-side
workspace; the shared-GPU runtime was already asleep at 520 MiB.

`LocalBackend.train` exposes `logprob_calculation_chunk_size` specifically for
this calculation. The driver now fixes it at 512 instead of ART's 1,024
default. This partitions the same vocabulary log-sum-exp without changing
tokens, logits, rewards, advantages, optimizer settings, or model weights.
The value is written into the cross-phase training contract, passed to every
smoke and formal learner call, and rejected if preflight, smoke, training, or resume
drifts. Because the failed attempt used the earlier contract, all gates are
rerun from fresh lineages rather than treating the failed smoke as reusable
evidence.

## D23. Qwen3.5 tool calls use `qwen3_coder`, not ART's Hermes default

Pinned ART initializes its local vLLM server with `hermes` unless `server_args`
overrides it (`third_party/ART/src/art/unsloth/service.py:276-286`). The
[frozen Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md)
instead specifies `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
for vLLM tool use.

On the RTX PRO 6000, Hermes treated Qwen's native function/parameter
serialization incorrectly: the failed smoke sample produced 0/8 reward.
Changing only the parser to `qwen3_coder` made the same scenarios and seeds
8/8. The parser is explicit in manifest schema 3 and remains part of the
semantic-contract hash, so every Hermes-era gate is invalid and all official
phases restart from fresh lineages.

## D24. Smoke exercises the first two formal batches

With the corrected parser, formal slots 0-1 and their formal seeds were all-one
(8/8), so a two-group smoke would ask ART to advance a checkpoint without
gradient work. A rollout-only scan of the fixed first ten formal steps found
three mixed groups and would not trigger the sparse-reward stop; the first
mixed group was slot 2.

Smoke therefore uses the contiguous prefix
`_scenario_for_slot(train, 0..3, seed=42)` and the identical formal seed
mapping: four groups and policy/user seeds 42-57, exactly the first two formal
batches. This is the minimum contiguous prefix that exercises gradient work,
not a selected successful task. The diagnostic lineage produced one
mixed/trainable group, one `backend.train` call, checkpoint transition 0 to 1,
and 88 reported gradient steps
([W&B](https://wandb.ai/lqj-physics-nudt/service-agent/runs/diag-smoke-qwen3coder-formal-slots0-3-r1)).
That lineage is disposable; formal training still starts fresh from slot 0. If
the official smoke has no variance, it fails rather than searching new seeds.

## D25. ART checkpoint steps and gradient work are reported separately

Pinned ART advances its logical step even when no group has reward variance: it
copies the current checkpoint to the next directory and emits zero for both
`data/step_num_groups_trainable` and `data/step_num_gradient_steps`
(`third_party/ART/src/art/local/backend.py:1466-1521`). Therefore
`checkpoint_step` and `last_completed_step` are lineage positions, not proof
that weights changed.

Smoke requires positive trainable groups and gradient steps. Formal manifests
report checkpoint steps, gradient-bearing checkpoint steps, skipped checkpoint
steps, submitted/trainable groups, and ART gradient steps separately. The
selected checkpoint also records those totals only through its own step. A
60-step lineage is never described as 60 optimizer updates without that
evidence.

## D26. Freeze checkpoint 0015 as the candidate from the sparse-stopped lineage

The official schema-3 preflight and smoke both passed under semantic contract
`91fa4cb5c06414976cf029003ad621b36becfe154ee86201c726f331ec9d6fb6`.
Formal GRPO requested 60 rollout/checkpoint positions and completed 24. Across
the complete lineage, ART reported five trainable groups, five
gradient-bearing checkpoint positions, 19 skipped positions, and 445 gradient
steps. The final ten positions contained one mixed-reward group, so the
predeclared sparsity rule wrote terminal status `stopped_sparse_reward` and
then ended the process with its expected protocol error. This was not a passed
60-position run, an OOM, or an infrastructure crash.

The fixed checkpoint rule selected 0015: the scheduled frozen-dev means were
0.850 at 0005, 0.850 at 0010, 0.925 at 0015, and 0.900 at 0020. Checkpoint
0024 is the latest terminal lineage position, not the selected model.
Checkpoint 0015 is therefore frozen as the current GRPO candidate for the
approval-gated final 2x2. Its 0.925 value is selection telemetry within this
bf16 lineage, not an RL evaluation result and not evidence of a +0.013 gain
over the separate quantized-MLX dev ablation.

The byte-identical formal manifest is
`results/gpu/grpo-4b-qwen3coder-r1/train_manifest.json` (SHA-256
`a011cdc352ef360d63e8e3f4db81b4b8152bac71bc84a7184407c8c35b2bdea3`).
The backed-up selected adapter is SHA-256
`1018931f9483c71ae20fbd59c76ab6a0c73137d4aefe9c8ad823175931b2c898`.
The recorded W&B run is
[grpo-4b-qwen3coder-r1](https://wandb.ai/lqj-physics-nudt/service-agent/runs/grpo-4b-qwen3coder-r1),
and `reports/grpo_training.md` regenerates the analysis from all three phase
manifests.

The ART history contains 24 rollout records, 24 backend-train records, and four
dev records. Both records for each formal position carry submitted/trainable
group counters, so ART's cumulative W&B state sums them to 96/10; the manifest
and the backend-record subset each reconstruct the authoritative 48 submitted,
five trainable, and 445 gradient-step totals. Gradient steps exist only on the
backend records and remain 445. This is a disclosed logging aggregation
artifact, not evidence of 96 distinct groups or a second training pass. The
committed `results/gpu/WANDB_COUNTER_AUDIT.json` is recomputed from the hashed
raw history and state when the ignored backup is available.

The 3.3 GB backup deliberately excludes the pinned bf16 base weights and is not
a standalone offline bundle. Recovery must download
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, load that snapshot
explicitly, and attach the backed-up adapter; the source-machine absolute path
inside `adapter_config.json` is not the recovery contract. This was exercised
after backup with checkpoint 0015 copied to a separate directory: the explicit
base and adapter loaded on the RTX PRO 6000 and generated one non-benchmark
token. `results/gpu/restore-cp0015-r1/restore_manifest.json` records the proof.

The exact formal lineage remains terminal: do not resume, reseed, retune, or
run it to 60 positions. This decision does not authorize teacher-data
generation or SFT. Either action would require a separate decision and a fresh
lineage before the final-test approval; absent that decision, checkpoint 0015
remains the RL candidate.
