"""Frozen contracts shared by GPU preflight, training, and final serving.

The model revision, chat-template mode, and ART API are protocol inputs, not
convenient defaults. Keeping them here lets local tests reject drift before a
GPU process downloads weights or performs an update.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
BASE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ART_COMMIT = "828b839b1139ac780725f0a22a9bde70a82b4878"
TAU2_COMMIT = "2822d9030b621e6f13a190fb14fa08cf1c9c4ca4"
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
TOOL_CALL_PARSER = "hermes"
MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RuntimeConfig:
    """The parts of a local ART run that affect weights or token sequences."""

    run_name: str
    project: str = "service-agent"
    base_model: str = BASE_MODEL_ID
    max_model_len: int = 16_384
    max_completion_tokens: int = 1_024
    rollout_concurrency: int = 4
    gpu_memory_utilization: float = 0.68
    lora_rank: int = 16
    lora_alpha: int = 32
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name is required")
        if self.base_model != BASE_MODEL_ID:
            raise ValueError(
                f"formal protocol requires {BASE_MODEL_ID}; got {self.base_model}"
            )
        if self.max_model_len <= self.max_completion_tokens:
            raise ValueError("max_model_len must exceed max_completion_tokens")
        if self.rollout_concurrency < 1:
            raise ValueError("rollout_concurrency must be positive")
        if not 0.1 <= self.gpu_memory_utilization <= 0.9:
            raise ValueError("gpu_memory_utilization must be between 0.1 and 0.9")


def apply_chat_template_token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
) -> list[int]:
    """Return one prompt's IDs across Transformers list/BatchEncoding APIs."""

    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    tolist = getattr(encoded, "tolist", None)
    if callable(tolist):
        encoded = tolist()
    if (
        isinstance(encoded, list)
        and len(encoded) == 1
        and isinstance(encoded[0], list)
    ):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)):
        raise RuntimeError("chat template did not return token IDs")
    return [int(token) for token in encoded]


def build_internal_model_config(config: RuntimeConfig) -> dict[str, Any]:
    """Configuration consumed by the pinned ART LocalBackend.

    Training is bf16 LoRA rather than ART's default 4-bit QLoRA. That makes
    the step-0 trainer weights the same bf16 checkpoint served by vLLM, so the
    logprob gate measures the actual optimization lineage.
    """

    return {
        "init_args": {
            "revision": BASE_MODEL_REVISION,
            "max_seq_length": config.max_model_len,
            "dtype": "bfloat16",
            "load_in_4bit": False,
            "load_in_16bit": True,
            "random_state": config.seed,
            "use_gradient_checkpointing": "unsloth",
        },
        "engine_args": {
            "revision": BASE_MODEL_REVISION,
            "tokenizer_revision": BASE_MODEL_REVISION,
            "dtype": "bfloat16",
            "max_model_len": config.max_model_len,
            "max_num_seqs": config.rollout_concurrency,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enable_sleep_mode": True,
            "enforce_eager": True,
            "max_logprobs": 1,
        },
        "trainer_args": {
            "bf16": True,
            "fp16": False,
            "seed": config.seed,
            "data_seed": config.seed,
            "max_completion_length": config.max_completion_tokens,
            "max_prompt_length": config.max_model_len - config.max_completion_tokens,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "report_to": "none",
        },
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
    }


def build_trainable_model_kwargs(config: RuntimeConfig) -> dict[str, Any]:
    """Keyword arguments accepted by TrainableModel at the pinned ART commit."""

    return {
        "name": config.run_name,
        "run_name": config.run_name,
        "project": config.project,
        "base_model": config.base_model,
        "lora_config": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": 0.0,
            "random_state": config.seed,
            "max_seq_length": config.max_model_len,
            "use_gradient_checkpointing": "unsloth",
        },
        "_internal_config": build_internal_model_config(config),
    }


def _function_kwonly_args(path: Path, class_name: str, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if class_node is None:
        raise RuntimeError(f"{class_name} is missing from {path}")
    function = next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise RuntimeError(f"{class_name}.{function_name} is missing from {path}")
    return {arg.arg for arg in function.args.kwonlyargs}


def assert_pinned_art_api(art_root: Path) -> None:
    """Fail before GPU work if the checked-out ART contract has drifted."""

    completed = subprocess.run(
        ["git", "-C", str(art_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = completed.stdout.strip()
    if actual_commit != ART_COMMIT:
        raise RuntimeError(f"ART commit drift: expected {ART_COMMIT}, got {actual_commit}")

    accepted = _function_kwonly_args(
        art_root / "src/art/model.py", "TrainableModel", "__init__"
    )
    required = {
        "name",
        "run_name",
        "project",
        "base_model",
        "lora_config",
        "_internal_config",
    }
    if missing := required - accepted:
        raise RuntimeError(f"pinned ART TrainableModel is missing arguments: {sorted(missing)}")

    rollout_tree = ast.parse((art_root / "src/art/tau_bench/rollout.py").read_text())
    rollout = next(
        (
            node
            for node in rollout_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "rollout"
            and not (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and node.body[0].value.value is Ellipsis
            )
        ),
        None,
    )
    if rollout is None:
        raise RuntimeError("pinned ART tau_bench.rollout is missing")
    rollout_args = {
        arg.arg
        for arg in (
            rollout.args.posonlyargs
            + rollout.args.args
            + rollout.args.kwonlyargs
        )
    }
    required_rollout = {"client", "max_turns", "chat_completion_kwargs"}
    if missing := required_rollout - rollout_args:
        raise RuntimeError(f"pinned ART rollout is missing arguments: {sorted(missing)}")


def semantic_input_hashes(
    *,
    system_prompt: str,
    tools: list[dict[str, Any]],
    tokenizer_chat_template: str,
) -> dict[str, str]:
    """Hash the three model-visible surfaces separately for provenance."""

    tools_payload = json.dumps(
        tools,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return {
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "tools_sha256": hashlib.sha256(tools_payload.encode("utf-8")).hexdigest(),
        "tokenizer_chat_template_sha256": hashlib.sha256(
            tokenizer_chat_template.encode("utf-8")
        ).hexdigest(),
    }


def semantic_contract_sha256(
    *,
    system_prompt: str,
    tools: list[dict[str, Any]],
    tokenizer_chat_template: str,
) -> str:
    """Hash every semantic input that must match across step 0 and training."""

    payload = {
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "tool_call_parser": TOOL_CALL_PARSER,
        "system_prompt": system_prompt,
        "tools": tools,
        "tokenizer_chat_template": tokenizer_chat_template,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _shared_protocol_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Return inputs that must not drift between GPU phases."""

    runtime = dict(payload.get("runtime") or {})
    runtime.pop("run_name", None)
    system = payload.get("system") or {}
    stable_system = {
        key: system.get(key)
        for key in (
            "python",
            "torch",
            "cuda_runtime",
            "gpu",
            "bf16_supported",
            "packages",
            "vllm_runtime",
        )
    }
    return {
        "schema_version": payload.get("schema_version"),
        "repo_commit": payload.get("repo_commit"),
        "art_commit": payload.get("art_commit"),
        "tau2_commit": payload.get("tau2_commit"),
        "base_model": payload.get("base_model"),
        "base_model_revision": payload.get("base_model_revision"),
        "semantic_contract_sha256": payload.get("semantic_contract_sha256"),
        "semantic_input_hashes": payload.get("semantic_input_hashes"),
        "runtime": runtime,
        "training": payload.get("training"),
        "token_budget": payload.get("token_budget"),
        "system": stable_system,
        "user_simulator": payload.get("user_simulator"),
    }


def validate_matching_protocol(
    gate: dict[str, Any],
    current: dict[str, Any],
    description: str,
) -> None:
    """Require a gate to have been produced by the current full protocol."""

    expected = _shared_protocol_contract(current)
    actual = _shared_protocol_contract(gate)
    different = [key for key in expected if actual[key] != expected[key]]
    if different:
        raise RuntimeError(
            f"{description} protocol does not match the current run: "
            + ", ".join(different)
        )


def validate_resume_contract(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Resume only the exact formal run, never a nearby configuration."""

    different = [
        key
        for key in ("phase", "run_name")
        if previous.get(key) != current.get(key)
    ]
    if different:
        raise RuntimeError(
            "resume manifest does not match the current run: " + ", ".join(different)
        )
    validate_matching_protocol(previous, current, "resume manifest")


def validate_preflight_gate(payload: dict[str, Any], expected_contract: str) -> None:
    """Authorize an update only after an unchanged, update-free preflight."""

    rollout = payload.get("rollout_only") or {}
    logprob = payload.get("logprob_gate") or {}
    checks = {
        "schema version": payload.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        "phase": payload.get("phase") == "preflight",
        "status": payload.get("status") == "passed",
        "ART commit": payload.get("art_commit") == ART_COMMIT,
        "base model": payload.get("base_model") == BASE_MODEL_ID,
        "base model revision": (
            payload.get("base_model_revision") == BASE_MODEL_REVISION
        ),
        "semantic contract": payload.get("semantic_contract_sha256") == expected_contract,
        "fresh step": payload.get("initial_step") == 0,
        "no update": payload.get("final_step") == 0,
        "logprob gate": logprob.get("status") == "passed",
        "exact prompt token ids": logprob.get("prompt_token_ids_exact") is True,
        "rollout-only gate": rollout.get("status") == "passed",
        "test split lock": rollout.get("test_split_locked") is True,
        "strict replay": rollout.get("strict_replay") is True,
        "reward finalized once": rollout.get("reward_finalized_once") is True,
        "multi-tool safety": rollout.get("multi_tool_calls") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("preflight gate is not valid: " + ", ".join(failed))
