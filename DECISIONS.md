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
