"""Frozen, benchmark-free serving support for the final factorial evaluation.

The final base and RL rows must share one vLLM process.  The process exposes
the untouched bf16 snapshot under one alias and checkpoint 0015 as a static
LoRA under a second alias.  This module freezes that launch contract and
provides health probes which do not instantiate or list any tau2 scenario.

The ``launch`` action validates the exact command and environment, then
replaces itself with the serving process.  The remaining actions only prepare
or validate public manifests and probe an already-running localhost server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from service_agent.training.contracts import (
    ART_COMMIT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHAT_TEMPLATE_KWARGS,
    MANIFEST_SCHEMA_VERSION,
    TAU2_COMMIT,
    TOOL_CALL_PARSER,
)

REPO = Path(__file__).resolve().parents[3]
FORMAL_MANIFEST = REPO / "results/gpu/grpo-4b-qwen3coder-r1/train_manifest.json"
RESTORE_MANIFEST = REPO / "results/gpu/restore-cp0015-r1/restore_manifest.json"

BASE_ALIAS = "final-frozen-r1"
RL_ALIAS = "final-rl-cp0015-r1"
SERVING_HOST = "127.0.0.1"
SERVING_PORT = 8100
CUDA_VISIBLE_DEVICES = "0"
SELECTED_CHECKPOINT = 15

ADAPTER_SHA256 = "1018931f9483c71ae20fbd59c76ab6a0c73137d4aefe9c8ad823175931b2c898"
CHAT_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
SYSTEM_PROMPT_SHA256 = "b76554cc96138af12ae5a6a053b213a08148a4edb6e2de9d19cd63eb7ac5ddac"
TOOLS_SHA256 = "2e0247eb453d015c5ef856e402776dec0e161230c273595f44107a2b4c93975a"
VLLM_BOOTSTRAP_SHA256 = (
    "64ceec18a55e3f19e80bd38a23ce2c3090f30fa994f479a3e386d64907198ab3"
)
VLLM_NINJA_SHA256 = "696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67"
VLLM_CUDART_SHA256 = "c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920"
VLLM_PACKAGE_VERSION = "0.23.0+cu129"
VLLM_API_VERSION = "0.23.0"
VLLM_TRANSFORMERS_VERSION = "5.12.1"
ART_VLLM_RUNTIME_VERSION = "0.1.0"
TRAINING_REPO_COMMIT = "e557bbffdee8e283f3a522e6a088ca74bf3ff907"

MAX_MODEL_LEN = 16_384
MAX_COMPLETION_TOKENS = 1_024
MAX_NUM_SEQS = 4
GPU_MEMORY_UTILIZATION = 0.68
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
FROZEN_RUNTIME_PACKAGES: Mapping[str, str] = MappingProxyType(
    {
        "art-vllm-runtime": ART_VLLM_RUNTIME_VERSION,
        "flashinfer-python": "0.6.12",
        "ninja": "1.13.0",
        "torch": "2.11.0+cu128",
        "transformers": VLLM_TRANSFORMERS_VERSION,
        "vllm": VLLM_PACKAGE_VERSION,
    }
)

FROZEN_ENGINE_ARGS: Mapping[str, object] = MappingProxyType(
    {
        "revision": BASE_MODEL_REVISION,
        "tokenizer_revision": BASE_MODEL_REVISION,
        "dtype": "bfloat16",
        "allowed_local_media_path": "/tmp",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "enable_sleep_mode": True,
        "enforce_eager": True,
        "max_logprobs": 1,
        "generation_config": "vllm",
        "max_loras": 2,
        "lora_target_modules": LORA_TARGET_MODULES,
    }
)
FROZEN_SERVER_ARGS: Mapping[str, object] = MappingProxyType(
    {
        "return_tokens_as_token_ids": True,
        "enable_auto_tool_choice": True,
        "tool_call_parser": TOOL_CALL_PARSER,
        "uvicorn_log_level": "warning",
    }
)

_TILELANG_ENV_KEYS = (
    "PYTHONPATH",
    "TVM_IMPORT_PYTHON_PATH",
    "TVM_LIBRARY_PATH",
    "TL_CUTLASS_PATH",
    "TL_TEMPLATE_PATH",
    "TL_COMPOSABLE_KERNEL_PATH",
)
_TILELANG_PATH_MARKERS = ("/site-packages/tilelang/", "\\site-packages\\tilelang\\")
_PRIVATE_KEY_SUFFIXES = (
    "api_key",
    "access_token",
    "auth_token",
    "credentials",
    "password",
    "private_key",
    "secret",
)

_HEALTH_TOOL = {
    "type": "function",
    "function": {
        "name": "health_probe",
        "description": "Report whether this local inference process is ready.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["ready"],
                }
            },
            "required": ["status"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class RuntimeLayout:
    """Files from the pinned ART inference environment used at launch."""

    server: Path
    python: Path
    ninja: Path
    cudart: Path
    bootstrap: Path
    flashinfer_workspace: Path


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _runtime_layout(
    repo_root: str | Path,
    runtime_server: str | Path | None = None,
) -> RuntimeLayout:
    root = _resolved(repo_root)
    server = _resolved(
        runtime_server
        or root / "third_party/ART/vllm_runtime/.venv/bin/art-vllm-runtime-server"
    )
    runtime_bin = server.parent
    runtime_venv = runtime_bin.parent
    return RuntimeLayout(
        server=server,
        python=runtime_bin / "python",
        ninja=runtime_bin / "ninja",
        cudart=(
            runtime_venv
            / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
        ),
        bootstrap=root / "src/service_agent/training/vllm_bootstrap/sitecustomize.py",
        flashinfer_workspace=(
            root / "third_party/ART/scratch/vllm_runtime_flashinfer"
        ),
    )


def _engine_args() -> dict[str, object]:
    payload = dict(FROZEN_ENGINE_ARGS)
    payload["lora_target_modules"] = list(LORA_TARGET_MODULES)
    return payload


def _server_args(adapter: str | Path) -> dict[str, object]:
    payload = dict(FROZEN_SERVER_ARGS)
    payload["lora_modules"] = [
        {
            "name": RL_ALIAS,
            "path": str(_resolved(adapter)),
            "base_model_name": BASE_ALIAS,
        }
    ]
    return payload


def _json_arg(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_vllm_command(
    snapshot: str | Path,
    adapter: str | Path,
    *,
    repo_root: str | Path = REPO,
    runtime_server: str | Path | None = None,
) -> list[str]:
    """Build the only permitted final vLLM command.

    Path relocation is allowed.  Model names, parser, dtype, context size,
    LoRA configuration, memory settings, host, port, and CUDA device are not
    function parameters and therefore cannot drift between factorial cells.
    """

    layout = _runtime_layout(repo_root, runtime_server)
    snapshot_path = _resolved(snapshot)
    return [
        str(layout.server),
        f"--model={snapshot_path}",
        f"--port={SERVING_PORT}",
        f"--host={SERVING_HOST}",
        f"--cuda-visible-devices={CUDA_VISIBLE_DEVICES}",
        f"--served-model-name={BASE_ALIAS}",
        "--rollout-weights-mode=lora",
        f"--engine-args-json={_json_arg(_engine_args())}",
        f"--server-args-json={_json_arg(_server_args(adapter))}",
    ]


def _drop_tilelang_paths(value: str | None) -> str | None:
    if value is None:
        return None
    entries = [
        item
        for item in value.split(os.pathsep)
        if item and not any(marker in item for marker in _TILELANG_PATH_MARKERS)
    ]
    return os.pathsep.join(entries) if entries else None


def _infer_hf_home(snapshot: Path) -> Path:
    if (
        snapshot.parent.name == "snapshots"
        and snapshot.parent.parent.name.startswith("models--")
    ):
        return snapshot.parent.parent.parent
    raise RuntimeError(
        "hf_home is required when the snapshot is not in a Hugging Face "
        "models--*/snapshots/<revision> cache"
    )


def build_vllm_environment(
    snapshot: str | Path,
    *,
    repo_root: str | Path = REPO,
    runtime_server: str | Path | None = None,
    hf_home: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the inherited environment with the pinned runtime overrides."""

    layout = _runtime_layout(repo_root, runtime_server)
    snapshot_path = _resolved(snapshot)
    env = dict(os.environ if environ is None else environ)
    cleaned_pythonpath = _drop_tilelang_paths(env.get("PYTHONPATH"))
    if cleaned_pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = cleaned_pythonpath
    # ART's managed runtime drops all TVM/TileLang control variables.  Keeping
    # a value merely because its text lacks "tilelang" can still redirect JIT
    # sources or load a second FFI library.
    for key in _TILELANG_ENV_KEYS:
        if key != "PYTHONPATH":
            env.pop(key, None)

    bootstrap_dir = str(layout.bootstrap.parent)
    python_paths = [
        item
        for item in env.get("PYTHONPATH", "").split(os.pathsep)
        if item and item != bootstrap_dir
    ]
    env["PYTHONPATH"] = os.pathsep.join([bootstrap_dir, *python_paths])

    runtime_bin = str(layout.server.parent)
    path_entries = [
        item
        for item in env.get("PATH", "").split(os.pathsep)
        if item and item != runtime_bin
    ]
    env["PATH"] = os.pathsep.join([runtime_bin, *path_entries])
    env.update(
        {
            "VLLM_CUDART_SO_PATH": str(layout.cudart),
            "FLASHINFER_WORKSPACE_BASE": str(layout.flashinfer_workspace),
            "HF_HOME": str(_resolved(hf_home) if hf_home else _infer_hf_home(snapshot_path)),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _safe_environment_overrides(
    snapshot: str | Path,
    *,
    repo_root: str | Path,
    runtime_server: str | Path | None,
    hf_home: str | Path | None,
) -> dict[str, str]:
    layout = _runtime_layout(repo_root, runtime_server)
    snapshot_path = _resolved(snapshot)
    return {
        "VLLM_CUDART_SO_PATH": str(layout.cudart),
        "PYTHONPATH_PREPEND": str(layout.bootstrap.parent),
        "PATH_PREPEND": str(layout.server.parent),
        "FLASHINFER_WORKSPACE_BASE": str(layout.flashinfer_workspace),
        "HF_HOME": str(_resolved(hf_home) if hf_home else _infer_hf_home(snapshot_path)),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_tree_digest(snapshot: Path) -> dict[str, int | str]:
    """Hash every regular snapshot file by path, size, and followed content."""

    files = sorted(
        (path for path in snapshot.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot).as_posix(),
    )
    _require(bool(files), f"model snapshot contains no files: {snapshot}")
    tree = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(snapshot).as_posix().encode()
        file_digest = hashlib.sha256()
        bytes_read = 0
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                file_digest.update(chunk)
                bytes_read += len(chunk)
        tree.update(len(relative).to_bytes(8, "big"))
        tree.update(relative)
        tree.update(bytes_read.to_bytes(8, "big"))
        tree.update(file_digest.digest())
        total_bytes += bytes_read
    return {
        "snapshot_tree_sha256": tree.hexdigest(),
        "snapshot_file_count": len(files),
        "snapshot_total_bytes": total_bytes,
    }


def _runtime_package_versions(layout: RuntimeLayout) -> dict[str, str]:
    """Read package metadata with the isolated runtime's own Python."""

    names = list(FROZEN_RUNTIME_PACKAGES)
    script = (
        "import json\n"
        "from importlib.metadata import version\n"
        f"print(json.dumps({{name: version(name) for name in {names!r}}}, "
        "sort_keys=True))\n"
    )
    completed = subprocess.run(
        [str(layout.python), "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ART vLLM runtime returned invalid package metadata") from exc
    _require(
        isinstance(payload, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in payload.items()),
        "ART vLLM runtime package metadata is invalid",
    )
    return payload


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{description} is missing: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _assert_no_private_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(normalized.endswith(suffix) for suffix in _PRIVATE_KEY_SUFFIXES):
                raise RuntimeError(f"serving manifest contains private field {path}.{key}")
            _assert_no_private_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_private_fields(item, f"{path}[{index}]")


def _validate_training_manifests(
    formal: dict[str, Any],
    restore: dict[str, Any],
) -> None:
    expected_formal = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "train",
        "status": "stopped_sparse_reward",
        "art_commit": ART_COMMIT,
        "tau2_commit": TAU2_COMMIT,
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "tool_call_parser": TOOL_CALL_PARSER,
        "test_split_locked": True,
    }
    for key, expected in expected_formal.items():
        _require(
            formal.get(key) == expected,
            f"formal manifest {key} drift: {formal.get(key)!r} != {expected!r}",
        )

    selected = formal.get("selected_checkpoint")
    _require(isinstance(selected, dict), "formal manifest selected checkpoint is missing")
    _require(
        selected.get("step") == SELECTED_CHECKPOINT,
        "formal manifest selected checkpoint is not 0015",
    )
    _require(
        Path(str(selected.get("checkpoint_path", ""))).name == "0015",
        "formal manifest selected checkpoint path is not checkpoint 0015",
    )

    runtime = formal.get("runtime")
    expected_runtime = {
        "base_model": BASE_MODEL_ID,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "lora_alpha": LORA_ALPHA,
        "lora_rank": LORA_RANK,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "rollout_concurrency": MAX_NUM_SEQS,
        "seed": 42,
    }
    _require(isinstance(runtime, dict), "formal runtime contract is missing")
    for key, expected in expected_runtime.items():
        _require(
            runtime.get(key) == expected,
            f"formal runtime {key} drift: {runtime.get(key)!r} != {expected!r}",
        )

    semantic = formal.get("semantic_input_hashes")
    _require(
        semantic
        == {
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "tokenizer_chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "tools_sha256": TOOLS_SHA256,
        },
        "formal semantic input hashes drifted",
    )

    vllm_runtime = formal.get("system", {}).get("vllm_runtime")
    _require(isinstance(vllm_runtime, dict), "formal vLLM runtime provenance is missing")
    packages = vllm_runtime.get("packages")
    _require(isinstance(packages, dict), "formal vLLM package versions are missing")
    for key, expected in FROZEN_RUNTIME_PACKAGES.items():
        _require(
            packages.get(key) == expected,
            f"formal vLLM package {key} drift: {packages.get(key)!r} != {expected!r}",
        )
    bootstrap = vllm_runtime.get("bootstrap")
    cudart = vllm_runtime.get("cudart")
    _require(isinstance(bootstrap, dict), "formal vLLM bootstrap provenance is missing")
    _require(isinstance(cudart, dict), "formal vLLM cudart provenance is missing")
    _require(
        bootstrap.get("sha256") == VLLM_BOOTSTRAP_SHA256,
        "formal vLLM bootstrap hash drifted",
    )
    _require(
        bootstrap.get("ninja_sha256") == VLLM_NINJA_SHA256,
        "formal vLLM ninja hash drifted",
    )
    _require(
        cudart.get("sha256") == VLLM_CUDART_SHA256
        and cudart.get("cuda_device_reset") is True,
        "formal vLLM CUDA runtime provenance drifted",
    )

    expected_restore = {
        "schema_version": 1,
        "status": "passed",
        "adapter_checkpoint": SELECTED_CHECKPOINT,
        "adapter_sha256": ADAPTER_SHA256,
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_snapshot_explicit": True,
        "dtype": "torch.bfloat16",
        "new_tokens_generated": 1,
        "test_split_accessed": False,
    }
    for key, expected in expected_restore.items():
        _require(
            restore.get(key) == expected,
            f"restore manifest {key} drift: {restore.get(key)!r} != {expected!r}",
        )


def validate_serving_inputs(
    snapshot: str | Path,
    adapter: str | Path,
    *,
    repo_root: str | Path = REPO,
    runtime_server: str | Path | None = None,
    formal_manifest: str | Path | None = None,
    restore_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Validate model, adapter, and runtime bits against committed evidence."""

    root = _resolved(repo_root)
    snapshot_path = _resolved(snapshot)
    adapter_path = _resolved(adapter)
    layout = _runtime_layout(root, runtime_server)
    formal_path = _resolved(formal_manifest or root / FORMAL_MANIFEST.relative_to(REPO))
    restore_path = _resolved(restore_manifest or root / RESTORE_MANIFEST.relative_to(REPO))
    formal = _read_json(formal_path, "formal GRPO manifest")
    restore = _read_json(restore_path, "selected-adapter restore manifest")
    _validate_training_manifests(formal, restore)

    _require(snapshot_path.is_dir(), f"pinned model snapshot is missing: {snapshot_path}")
    _require(
        snapshot_path.name == BASE_MODEL_REVISION,
        f"model snapshot revision drift: {snapshot_path.name}",
    )
    _require(adapter_path.is_dir(), f"selected adapter is missing: {adapter_path}")
    _require(layout.server.is_file(), f"ART vLLM runtime server is missing: {layout.server}")
    _require(
        os.access(layout.server, os.X_OK),
        f"ART vLLM runtime server is not executable: {layout.server}",
    )
    for description, path in (
        ("ART vLLM runtime Python", layout.python),
        ("ART vLLM runtime ninja", layout.ninja),
        ("ART vLLM CUDA runtime", layout.cudart),
        ("project vLLM bootstrap", layout.bootstrap),
    ):
        _require(path.is_file(), f"{description} is missing: {path}")
    for description, path in (
        ("ART vLLM runtime Python", layout.python),
        ("ART vLLM runtime ninja", layout.ninja),
    ):
        _require(os.access(path, os.X_OK), f"{description} is not executable: {path}")
    _require(
        layout.flashinfer_workspace.is_dir(),
        f"ART FlashInfer workspace is missing: {layout.flashinfer_workspace}",
    )

    adapter_weights = adapter_path / "adapter_model.safetensors"
    adapter_config_path = adapter_path / "adapter_config.json"
    template_path = snapshot_path / "chat_template.jinja"
    for description, path in (
        ("selected adapter weights", adapter_weights),
        ("selected adapter config", adapter_config_path),
        ("pinned tokenizer chat template", template_path),
    ):
        _require(path.is_file(), f"{description} is missing: {path}")

    hashes = {
        "adapter_sha256": _sha256_file(adapter_weights),
        "tokenizer_chat_template_sha256": _sha256_file(template_path),
        "vllm_bootstrap_sha256": _sha256_file(layout.bootstrap),
        "vllm_ninja_sha256": _sha256_file(layout.ninja),
        "vllm_cudart_sha256": _sha256_file(layout.cudart),
        **_snapshot_tree_digest(snapshot_path),
    }
    expected_hashes = {
        "adapter_sha256": ADAPTER_SHA256,
        "tokenizer_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "vllm_bootstrap_sha256": VLLM_BOOTSTRAP_SHA256,
        "vllm_ninja_sha256": VLLM_NINJA_SHA256,
        "vllm_cudart_sha256": VLLM_CUDART_SHA256,
    }
    for key, expected in expected_hashes.items():
        _require(hashes.get(key) == expected, f"final serving file hash drifted: {key}")
    actual_runtime_packages = _runtime_package_versions(layout)
    _require(
        actual_runtime_packages == dict(FROZEN_RUNTIME_PACKAGES),
        "installed ART vLLM runtime package versions drifted",
    )

    adapter_config = _read_json(adapter_config_path, "selected adapter config")
    _require(adapter_config.get("peft_type") == "LORA", "adapter is not a LoRA")
    _require(adapter_config.get("task_type") == "CAUSAL_LM", "adapter task type drifted")
    _require(adapter_config.get("r") == LORA_RANK, "adapter rank drifted")
    _require(adapter_config.get("lora_alpha") == LORA_ALPHA, "adapter alpha drifted")
    _require(adapter_config.get("lora_dropout") == 0.0, "adapter dropout drifted")
    _require(adapter_config.get("inference_mode") is True, "adapter is not inference-only")
    _require(
        set(adapter_config.get("target_modules", [])) == set(LORA_TARGET_MODULES),
        "adapter target modules drifted",
    )

    repo_commit = formal.get("repo_commit")
    _require(
        repo_commit == TRAINING_REPO_COMMIT,
        "formal training repo commit drifted",
    )
    return {
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_alias": BASE_ALIAS,
        "rl_model_alias": RL_ALIAS,
        "base_snapshot": str(snapshot_path),
        "adapter_path": str(adapter_path),
        "adapter_checkpoint": SELECTED_CHECKPOINT,
        **hashes,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "tool_call_parser": TOOL_CALL_PARSER,
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        "semantic_input_hashes": deepcopy(formal["semantic_input_hashes"]),
        "training_repo_commit": repo_commit,
        "art_commit": ART_COMMIT,
        "tau2_commit": TAU2_COMMIT,
        "runtime_packages": actual_runtime_packages,
        "serving_repo_root": str(root),
        "runtime_server": str(layout.server),
        "runtime_python": str(layout.python),
        "test_split_locked_during_training": True,
        "restore_test_split_accessed": False,
    }


def build_serving_manifest(
    snapshot: str | Path,
    adapter: str | Path,
    *,
    repo_root: str | Path = REPO,
    runtime_server: str | Path | None = None,
    hf_home: str | Path | None = None,
    formal_manifest: str | Path | None = None,
    restore_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Return the normalized, public launch contract for result provenance."""

    evidence = validate_serving_inputs(
        snapshot,
        adapter,
        repo_root=repo_root,
        runtime_server=runtime_server,
        formal_manifest=formal_manifest,
        restore_manifest=restore_manifest,
    )
    payload = {
        "schema_version": 1,
        "status": "prepared",
        "api_base": f"http://{SERVING_HOST}:{SERVING_PORT}/v1",
        **evidence,
        "engine_args": _engine_args(),
        "server_args": _server_args(adapter),
        "command": build_vllm_command(
            snapshot,
            adapter,
            repo_root=repo_root,
            runtime_server=runtime_server,
        ),
        "environment_overrides": _safe_environment_overrides(
            snapshot,
            repo_root=repo_root,
            runtime_server=runtime_server,
            hf_home=hf_home,
        ),
    }
    _assert_no_private_fields(payload)
    return payload


def launch_serving(
    snapshot: str | Path,
    adapter: str | Path,
    *,
    repo_root: str | Path = REPO,
    runtime_server: str | Path | None = None,
    hf_home: str | Path | None = None,
    formal_manifest: str | Path | None = None,
    restore_manifest: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> NoReturn:
    """Validate every frozen input, then replace this process with vLLM."""

    # This call performs the expensive adapter/runtime hashes and verifies both
    # committed evidence manifests before any GPU process is created.
    build_serving_manifest(
        snapshot,
        adapter,
        repo_root=repo_root,
        runtime_server=runtime_server,
        hf_home=hf_home,
        formal_manifest=formal_manifest,
        restore_manifest=restore_manifest,
    )
    command = build_vllm_command(
        snapshot,
        adapter,
        repo_root=repo_root,
        runtime_server=runtime_server,
    )
    environment = build_vllm_environment(
        snapshot,
        repo_root=repo_root,
        runtime_server=runtime_server,
        hf_home=hf_home,
        environ=environ,
    )
    os.execvpe(command[0], command, environment)
    raise RuntimeError("os.execvpe unexpectedly returned")


def _verify_manifest_files(payload: dict[str, Any]) -> None:
    snapshot = _resolved(payload["base_snapshot"])
    adapter = _resolved(payload["adapter_path"])
    repo_root = _resolved(payload["serving_repo_root"])
    runtime_server = _resolved(payload["runtime_server"])
    layout = _runtime_layout(repo_root, runtime_server)
    paths = {
        "adapter_sha256": adapter / "adapter_model.safetensors",
        "tokenizer_chat_template_sha256": snapshot / "chat_template.jinja",
        "vllm_bootstrap_sha256": layout.bootstrap,
        "vllm_ninja_sha256": layout.ninja,
        "vllm_cudart_sha256": layout.cudart,
    }
    for field, path in paths.items():
        _require(path.is_file(), f"recorded serving file is missing: {path}")
        _require(
            _sha256_file(path) == payload.get(field),
            f"recorded serving file drifted: {field}",
        )
    tree = _snapshot_tree_digest(snapshot)
    for field, value in tree.items():
        _require(
            payload.get(field) == value,
            f"recorded model snapshot tree drifted: {field}",
        )
    _require(
        _runtime_package_versions(layout) == payload.get("runtime_packages"),
        "recorded ART vLLM runtime packages drifted",
    )


def _validate_serving_manifest_payload(
    payload: dict[str, Any],
    *,
    allow_prepared: bool,
    verify_files: bool,
) -> dict[str, Any]:
    _assert_no_private_fields(payload)
    _require(payload.get("schema_version") == 1, "serving manifest schema drifted")
    permitted_statuses = {"prepared", "passed"} if allow_prepared else {"passed"}
    _require(payload.get("status") in permitted_statuses, "serving manifest is not passed")
    expected = {
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_alias": BASE_ALIAS,
        "rl_model_alias": RL_ALIAS,
        "adapter_checkpoint": SELECTED_CHECKPOINT,
        "adapter_sha256": ADAPTER_SHA256,
        "tokenizer_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "vllm_bootstrap_sha256": VLLM_BOOTSTRAP_SHA256,
        "vllm_ninja_sha256": VLLM_NINJA_SHA256,
        "vllm_cudart_sha256": VLLM_CUDART_SHA256,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "tool_call_parser": TOOL_CALL_PARSER,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "semantic_input_hashes": {
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "tokenizer_chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "tools_sha256": TOOLS_SHA256,
        },
        "training_repo_commit": TRAINING_REPO_COMMIT,
        "art_commit": ART_COMMIT,
        "tau2_commit": TAU2_COMMIT,
        "runtime_packages": dict(FROZEN_RUNTIME_PACKAGES),
        "engine_args": _engine_args(),
    }
    for key, value in expected.items():
        _require(
            payload.get(key) == value,
            f"serving manifest {key} drift: {payload.get(key)!r} != {value!r}",
        )
    _require(
        isinstance(payload.get("snapshot_tree_sha256"), str)
        and len(payload["snapshot_tree_sha256"]) == 64,
        "serving manifest snapshot tree hash is invalid",
    )
    _require(
        _positive_int(payload.get("snapshot_file_count")),
        "serving manifest snapshot file count is invalid",
    )
    _require(
        isinstance(payload.get("snapshot_total_bytes"), int)
        and not isinstance(payload["snapshot_total_bytes"], bool)
        and payload["snapshot_total_bytes"] >= 0,
        "serving manifest snapshot byte count is invalid",
    )
    adapter_path = payload.get("adapter_path")
    snapshot_path = payload.get("base_snapshot")
    runtime_server = payload.get("runtime_server")
    serving_repo_root = payload.get("serving_repo_root")
    _require(
        isinstance(adapter_path, str) and bool(adapter_path),
        "serving manifest adapter path is missing",
    )
    _require(
        isinstance(snapshot_path, str) and bool(snapshot_path),
        "serving manifest base snapshot is missing",
    )
    _require(
        isinstance(runtime_server, str) and bool(runtime_server),
        "serving manifest runtime server is missing",
    )
    _require(
        isinstance(serving_repo_root, str) and bool(serving_repo_root),
        "serving manifest repo root is missing",
    )
    _require(
        payload.get("server_args") == _server_args(adapter_path),
        "serving manifest static LoRA mapping or server args drifted",
    )
    _require(
        payload.get("api_base") == f"http://{SERVING_HOST}:{SERVING_PORT}/v1",
        "serving manifest api_base drifted",
    )
    _require(
        payload.get("command")
        == build_vllm_command(
            snapshot_path,
            adapter_path,
            repo_root=serving_repo_root,
            runtime_server=runtime_server,
        ),
        "serving manifest command drifted",
    )
    overrides = payload.get("environment_overrides")
    _require(
        isinstance(overrides, dict) and isinstance(overrides.get("HF_HOME"), str),
        "serving manifest environment overrides are missing",
    )
    _require(
        overrides
        == _safe_environment_overrides(
            snapshot_path,
            repo_root=serving_repo_root,
            runtime_server=runtime_server,
            hf_home=overrides["HF_HOME"],
        ),
        "serving manifest environment overrides drifted",
    )
    if payload.get("status") == "passed":
        probe = payload.get("probe")
        _require(isinstance(probe, dict), "passed serving manifest has no probe")
        _validate_probe_manifest(
            probe,
            snapshot=_resolved(snapshot_path),
            adapter=_resolved(adapter_path),
            expected_api_base=payload["api_base"],
        )
    if verify_files:
        _verify_manifest_files(payload)
    return payload


def validate_manifest_for_final_runner(
    path: str | Path,
    *,
    allow_prepared: bool = False,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Load a persisted serving manifest and reject tunable-field drift.

    The default is deliberately terminal: a prepared launch contract cannot
    authorize the final runner.  Pre-launch tooling may opt into reading it
    with ``allow_prepared=True`` before the server exists.
    """

    payload = _read_json(_resolved(path), "final serving manifest")
    return _validate_serving_manifest_payload(
        payload,
        allow_prepared=allow_prepared,
        verify_files=verify_files,
    )


def _local_server_root(api_base: str) -> str:
    parsed = urlparse(api_base)
    _require(parsed.scheme == "http", "final serving probe requires plain localhost HTTP")
    _require(
        parsed.hostname in {"127.0.0.1", "localhost", "::1"},
        "final serving probe refuses a non-loopback server",
    )
    _require(parsed.username is None and parsed.password is None, "api_base contains credentials")
    _require(parsed.query == "" and parsed.fragment == "", "api_base contains query or fragment")
    path = parsed.path.rstrip("/")
    _require(path in {"", "/v1"}, "api_base path must be empty or /v1")
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"http://{host}{port}"


def _http_request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, bytes]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")[-500:]
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _json_body(body: bytes, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description} returned invalid JSON") from exc
    _require(isinstance(payload, dict), f"{description} must return a JSON object")
    return payload


def _probe_request(alias: str) -> dict[str, Any]:
    return {
        "model": alias,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a local serving health probe. Call the supplied "
                    "health_probe function exactly once."
                ),
            },
            {
                "role": "user",
                "content": "Call health_probe now with status ready.",
            },
        ],
        "tools": [deepcopy(_HEALTH_TOOL)],
        "tool_choice": "auto",
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
    }


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_probe_response(alias: str, payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("model") == alias, f"{alias} probe response model drifted")
    choices = payload.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1, f"{alias} probe choices drifted")
    choice = choices[0]
    _require(isinstance(choice, dict), f"{alias} probe choice is invalid")
    _require(
        choice.get("finish_reason") == "tool_calls",
        f"{alias} auto-tool probe did not finish with tool_calls",
    )
    message = choice.get("message")
    _require(isinstance(message, dict), f"{alias} probe message is invalid")
    _require(
        message.get("reasoning_content") in {None, ""},
        f"{alias} probe leaked reasoning_content",
    )
    content = message.get("content")
    _require(
        content is None or isinstance(content, str),
        f"{alias} probe message content has an invalid type",
    )
    lowered = (content or "").lower()
    _require(
        "<think>" not in lowered and "</think>" not in lowered,
        f"{alias} probe returned thinking tags",
    )
    calls = message.get("tool_calls")
    _require(
        isinstance(calls, list) and len(calls) == 1,
        f"{alias} probe must return exactly one tool call",
    )
    call = calls[0]
    _require(isinstance(call, dict), f"{alias} probe tool call is invalid")
    _require(call.get("type") == "function", f"{alias} probe tool call type drifted")
    _require(
        isinstance(call.get("id"), str) and bool(call["id"]),
        f"{alias} probe tool call has no id",
    )
    function = call.get("function")
    _require(isinstance(function, dict), f"{alias} probe function is invalid")
    _require(
        function.get("name") == "health_probe",
        f"{alias} probe function name drifted",
    )
    arguments = function.get("arguments")
    _require(isinstance(arguments, str), f"{alias} probe arguments are not JSON text")
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{alias} probe arguments are invalid JSON") from exc
    _require(
        decoded == {"status": "ready"},
        f"{alias} probe arguments drifted: {decoded!r}",
    )
    usage = payload.get("usage")
    _require(isinstance(usage, dict), f"{alias} probe usage is missing")
    _require(
        _positive_int(usage.get("prompt_tokens"))
        and _positive_int(usage.get("completion_tokens")),
        f"{alias} probe token usage is invalid",
    )
    return {
        "status": "passed",
        "response_model": alias,
        "finish_reason": "tool_calls",
        "tool_call_count": 1,
        "tool_name": "health_probe",
        "arguments": {"status": "ready"},
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "reasoning_content_absent": True,
        "thinking_tags_absent": True,
    }


def probe_serving(
    api_base: str,
    snapshot: str | Path,
    adapter: str | Path,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Strictly probe an already-running local dual-alias server.

    The two generation calls use a synthetic health function and contain no
    tau2 policy, task, observation, label, or scenario identifier.
    """

    _require(timeout > 0, "serving probe timeout must be positive")
    server_root = _local_server_root(api_base)
    snapshot_path = _resolved(snapshot)
    adapter_path = _resolved(adapter)

    status, _ = _http_request("GET", f"{server_root}/health", timeout=timeout)
    _require(status == 200, "vLLM /health did not return HTTP 200")

    status, body = _http_request("GET", f"{server_root}/version", timeout=timeout)
    _require(status == 200, "vLLM /version did not return HTTP 200")
    version = _json_body(body, "vLLM /version")
    _require(
        isinstance(version.get("version"), str)
        and version["version"].startswith(VLLM_API_VERSION),
        f"vLLM API version drift: {version.get('version')!r}",
    )

    status, body = _http_request(
        "GET",
        f"{server_root}/art/capabilities",
        timeout=timeout,
    )
    _require(status == 200, "ART /art/capabilities did not return HTTP 200")
    capabilities = _json_body(body, "ART /art/capabilities")
    required_capabilities = {
        "runtime": "art_vllm",
        "protocol_version": 1,
        "inplace_lora_load": True,
        "policy_token_spans": True,
    }
    for key, expected in required_capabilities.items():
        _require(
            capabilities.get(key) == expected,
            f"ART serving capability {key} drifted",
        )

    status, body = _http_request("GET", f"{server_root}/v1/models", timeout=timeout)
    _require(status == 200, "vLLM /v1/models did not return HTTP 200")
    models = _json_body(body, "vLLM /v1/models")
    cards = models.get("data")
    _require(isinstance(cards, list), "vLLM model list is invalid")
    _require(
        all(isinstance(card, dict) and isinstance(card.get("id"), str) for card in cards),
        "vLLM model cards are invalid",
    )
    by_id = {card["id"]: card for card in cards}
    _require(
        set(by_id) == {BASE_ALIAS, RL_ALIAS},
        f"vLLM model aliases drifted: {sorted(by_id)}",
    )
    base_card = by_id[BASE_ALIAS]
    rl_card = by_id[RL_ALIAS]
    _require(
        base_card.get("root") == str(snapshot_path)
        and base_card.get("parent") is None
        and base_card.get("max_model_len") == MAX_MODEL_LEN,
        "vLLM frozen-base model card drifted",
    )
    _require(
        rl_card.get("root") == str(adapter_path)
        and rl_card.get("parent") == BASE_ALIAS,
        "vLLM RL adapter model card drifted",
    )

    probe_results: dict[str, Any] = {}
    for alias in (BASE_ALIAS, RL_ALIAS):
        status, body = _http_request(
            "POST",
            f"{server_root}/v1/chat/completions",
            payload=_probe_request(alias),
            timeout=timeout,
        )
        _require(status == 200, f"{alias} health probe did not return HTTP 200")
        response = _json_body(body, f"{alias} health probe")
        probe_results[alias] = _validate_probe_response(alias, response)

    payload = {
        "schema_version": 1,
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "api_base": f"{server_root}/v1",
        "vllm_api_version": version["version"],
        "capabilities": required_capabilities,
        "model_cards": {
            BASE_ALIAS: {
                "root": base_card["root"],
                "parent": base_card.get("parent"),
                "max_model_len": base_card["max_model_len"],
            },
            RL_ALIAS: {
                "root": rl_card["root"],
                "parent": rl_card["parent"],
            },
        },
        "probes": probe_results,
        "tool_call_parser": TOOL_CALL_PARSER,
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        "benchmark_data_accessed": False,
    }
    _assert_no_private_fields(payload)
    return payload


def _validate_probe_manifest(
    probe: dict[str, Any],
    *,
    snapshot: Path,
    adapter: Path,
    expected_api_base: str,
) -> None:
    _assert_no_private_fields(probe)
    expected = {
        "schema_version": 1,
        "status": "passed",
        "api_base": expected_api_base,
        "tool_call_parser": TOOL_CALL_PARSER,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "benchmark_data_accessed": False,
    }
    for key, value in expected.items():
        _require(
            probe.get(key) == value,
            f"serving probe {key} drift: {probe.get(key)!r} != {value!r}",
        )
    checked_at = probe.get("checked_at")
    _require(isinstance(checked_at, str), "serving probe checked_at is missing")
    try:
        timestamp = datetime.fromisoformat(checked_at)
    except ValueError as exc:
        raise RuntimeError("serving probe checked_at is invalid") from exc
    _require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "serving probe checked_at has no timezone",
    )
    version = probe.get("vllm_api_version")
    _require(
        isinstance(version, str) and version.startswith(VLLM_API_VERSION),
        "serving probe vLLM version drifted",
    )
    _require(
        probe.get("capabilities")
        == {
            "runtime": "art_vllm",
            "protocol_version": 1,
            "inplace_lora_load": True,
            "policy_token_spans": True,
        },
        "serving probe capabilities drifted",
    )
    _require(
        probe.get("model_cards")
        == {
            BASE_ALIAS: {
                "root": str(snapshot),
                "parent": None,
                "max_model_len": MAX_MODEL_LEN,
            },
            RL_ALIAS: {
                "root": str(adapter),
                "parent": BASE_ALIAS,
            },
        },
        "serving probe model cards drifted",
    )
    probes = probe.get("probes")
    _require(
        isinstance(probes, dict) and set(probes) == {BASE_ALIAS, RL_ALIAS},
        "serving probe alias results drifted",
    )
    for alias in (BASE_ALIAS, RL_ALIAS):
        result = probes[alias]
        _require(isinstance(result, dict), f"serving probe result for {alias} is invalid")
        expected_result = {
            "status": "passed",
            "response_model": alias,
            "finish_reason": "tool_calls",
            "tool_call_count": 1,
            "tool_name": "health_probe",
            "arguments": {"status": "ready"},
            "reasoning_content_absent": True,
            "thinking_tags_absent": True,
        }
        for key, value in expected_result.items():
            _require(
                result.get(key) == value,
                f"serving probe {alias} {key} drifted",
            )
        _require(
            _positive_int(result.get("prompt_tokens"))
            and _positive_int(result.get("completion_tokens")),
            f"serving probe {alias} token counts are invalid",
        )


def finalize_serving_manifest(
    prepared: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a passed live probe to its exact prepared launch contract."""

    prepared_payload = deepcopy(dict(prepared))
    probe_payload = deepcopy(dict(probe))
    _require(
        prepared_payload.get("status") == "prepared",
        "only a prepared serving manifest can be finalized",
    )
    _validate_serving_manifest_payload(
        prepared_payload,
        allow_prepared=True,
        verify_files=False,
    )
    snapshot = _resolved(prepared_payload["base_snapshot"])
    adapter = _resolved(prepared_payload["adapter_path"])
    _validate_probe_manifest(
        probe_payload,
        snapshot=snapshot,
        adapter=adapter,
        expected_api_base=prepared_payload["api_base"],
    )
    finalized = deepcopy(prepared_payload)
    finalized.update(
        {
            "status": "passed",
            "checked_at": probe_payload["checked_at"],
            "probe": probe_payload,
        }
    )
    return _validate_serving_manifest_payload(
        finalized,
        allow_prepared=False,
        verify_files=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("manifest", "launch", "probe", "finalize"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--runtime-server", type=Path)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--formal-manifest", type=Path)
    parser.add_argument("--restore-manifest", type=Path)
    parser.add_argument(
        "--api-base",
        default=f"http://{SERVING_HOST}:{SERVING_PORT}/v1",
    )
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--probe-manifest", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "manifest":
        if args.snapshot is None or args.adapter is None:
            raise SystemExit("manifest requires --snapshot and --adapter")
        payload = build_serving_manifest(
            args.snapshot,
            args.adapter,
            repo_root=args.repo_root,
            runtime_server=args.runtime_server,
            hf_home=args.hf_home,
            formal_manifest=args.formal_manifest,
            restore_manifest=args.restore_manifest,
        )
    elif args.action == "launch":
        if args.snapshot is None or args.adapter is None:
            raise SystemExit("launch requires --snapshot and --adapter")
        launch_serving(
            args.snapshot,
            args.adapter,
            repo_root=args.repo_root,
            runtime_server=args.runtime_server,
            hf_home=args.hf_home,
            formal_manifest=args.formal_manifest,
            restore_manifest=args.restore_manifest,
        )
    elif args.action == "probe":
        if args.snapshot is None or args.adapter is None:
            raise SystemExit("probe requires --snapshot and --adapter")
        payload = probe_serving(args.api_base, args.snapshot, args.adapter)
    else:
        if args.prepared_manifest is None or args.probe_manifest is None:
            raise SystemExit("finalize requires --prepared-manifest and --probe-manifest")
        payload = finalize_serving_manifest(
            _read_json(_resolved(args.prepared_manifest), "prepared serving manifest"),
            _read_json(_resolved(args.probe_manifest), "serving probe manifest"),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
