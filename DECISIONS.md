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

Used for smoke tests, teacher-trajectory generation (SFT bridge), and failure
analysis. Never substituted for the formal Qwen 4B policy in any reported cell:
the 2x2 requires base and RL cells to share one checkpoint, tokenizer, chat
template, tool parser, and inference stack.

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

## D10. Step 0 is a zero-update gate, not a smoke-training by-product

Preflight registers an untouched LoRA lineage, captures vLLM's exact prompt
and completion token IDs plus rollout logprobs, closes vLLM, and only then
loads the bf16 reference model. Prompt token IDs must match byte-for-byte and
importance ratios must pass before any update. A second registration of the
same untouched step-0 checkpoint runs official train-core episodes through
strict replay without training. Only that manifest can authorize a separate
one-update smoke lineage; only preflight plus smoke can authorize a fresh
formal lineage. This replaces the earlier runbook order, which performed a
smoke update before the purported step-0 check.

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
name and checkpoint step.

ART reports a group as trainable only when rewards vary within that group.
The driver now uses that exact definition: constant fractional rewards such as
`[0.5, 0.5]` are not called mixed. The one-update smoke refuses to train unless
at least one group has variance and ART confirms at least one trainable group.
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
