"""Validate and publish the one-shot final 2x2 evaluation.

Raw tau2 results are intentionally not a portfolio artifact: they contain
complete conversations, task labels, and evaluator state.  This module first
validates the frozen common-random-number grid and its execution contract,
then derives a compact public evidence package which contains only episode
identities, outcomes, and mechanical aggregates.

Generation needs the raw backup::

    uv run python -m service_agent.eval.report_factorial generate \
      --raw-root /path/to/final-2x2-r1 \
      --protocol-manifest /path/to/final-2x2-r1/final_manifest.json \
      --serving-manifest /path/to/serving_manifest.json

The committed package and report can subsequently be checked without private
raw data. Supplying the three raw-input paths to ``check`` additionally
verifies every hash in ``RAW_SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import Results, SimulationRun

from service_agent.eval.factorial import (
    BOOTSTRAP_RESAMPLES,
    FACTORIAL_CELLS,
    TAXONOMY_PRIORITY,
    ValidatedFactorialGrid,
    classify_failure,
    factorial_bootstrap_contrasts,
    pass_hat_ks,
    validate_factorial_grid,
)
from service_agent.eval.metrics import analyze_trajectory, audit_summary

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PUBLIC_ROOT = REPO / "results/final"
DEFAULT_REPORT_PATH = REPO / "reports/factorial_results.md"

SCHEMA_VERSION = 1
EXPECTED_TASK_COUNT = 40
EXPECTED_TRIAL_COUNT = 8
EXPECTED_BASE_SEED = 42
EXPECTED_TOTAL_EPISODES = 1_280
EXPECTED_CELL_ORDER = list(FACTORIAL_CELLS)
EXPECTED_HARNESS = {
    "base_h0": "llm_agent",
    "base_h2": "governed_llm_agent_h2",
    "rl_h0": "llm_agent",
    "rl_h2": "governed_llm_agent_h2",
}
EXPECTED_MODEL_ROW = {
    "base_h0": "base",
    "base_h2": "base",
    "rl_h0": "rl",
    "rl_h2": "rl",
}
PUBLIC_JSON_FILES = (
    "protocol.json",
    "outcomes.json",
    "pairing_manifest.json",
    "analysis.json",
    "live_audit.json",
)
PUBLIC_RESULT_FILES = (*PUBLIC_JSON_FILES, "RAW_SHA256SUMS")
CHECKSUM_REPORT_KEY = "report/factorial_results.md"

PRIVATE_FIELDS = {
    "access_token",
    "api_key",
    "arguments_json",
    "authorization",
    "cookie",
    "credentials",
    "environment",
    "evaluation_criteria",
    "global_simulation_guidelines",
    "initial_state",
    "messages",
    "password",
    "policy",
    "private_key",
    "provider_session_id",
    "review",
    "secret",
    "system_prompt",
    "ticks",
    "token",
    "tool_call_id",
    "user_scenario",
}
PRIVATE_FIELD_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_auth_token",
    "_authorization",
    "_cookie",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    re.compile(
        r"(?i)[?&](?:api[_-]?key|access[_-]?token|token|secret|signature|credential)="
        r"[^&#\s]+"
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"invalid final factorial artifacts: {message}")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"{description} is missing: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid final factorial artifacts: {description} is not JSON") from exc
    _require(isinstance(payload, dict), f"{description} must be a JSON object")
    return payload


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return _sha256_bytes(encoded)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _typed_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _finite_number(
    value: Any,
    context: str,
    *,
    nonnegative: bool = False,
) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{context} must be a finite number",
    )
    number = float(value)
    if nonnegative:
        _require(number >= 0.0, f"{context} must be nonnegative")
    return number


def _optional_cost(value: Any, context: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, context, nonnegative=True)


def _assert_public_safe(value: Any, path: str = "$") -> None:
    """Reject task-private fields and recognizable credentials recursively."""
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.lower()
            _require(
                normalized not in PRIVATE_FIELDS
                and not normalized.endswith(PRIVATE_FIELD_SUFFIXES),
                f"public artifact contains private field {path}.{key}",
            )
            for pattern in SECRET_PATTERNS:
                _require(
                    pattern.search(key_text) is None,
                    f"public artifact contains a credential-like key at {path}",
                )
            _assert_public_safe(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            _require(
                pattern.search(value) is None,
                f"public artifact contains a credential-like value at {path}",
            )


def _assert_public_text_safe(text: str, description: str) -> None:
    for pattern in SECRET_PATTERNS:
        _require(
            pattern.search(text) is None,
            f"{description} contains a credential-like value",
        )


def _parse_checksums(text: str, description: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        digest, separator, relative = line.partition("  ")
        _require(
            bool(separator) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"malformed checksum line in {description}",
        )
        path = Path(relative)
        _require(
            relative
            and not path.is_absolute()
            and ".." not in path.parts,
            f"unsafe checksum path {relative!r} in {description}",
        )
        _require(relative not in entries, f"duplicate checksum path {relative!r}")
        entries[relative] = digest
    return entries


def _checksum_bytes(entries: Mapping[str, str]) -> bytes:
    return "".join(f"{entries[path]}  {path}\n" for path in sorted(entries)).encode()


def _protocol_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact runner-v1 envelope and return its frozen contract."""
    _require(payload.get("schema_version") == 1, "protocol manifest schema drifted")
    contract = payload.get("protocol")
    _require(isinstance(contract, dict), "embedded frozen protocol is missing")
    frozen_fields = {
        "protocol_id",
        "schema_version",
        "repo_commit",
        "art_commit",
        "tau2_commit",
        "serving_manifest_sha256",
        "gpu",
        "serving_process",
        "task_split",
        "expected_task_count",
        "task_count",
        "expected_trial_count",
        "trials",
        "base_seed",
        "trial_seeds",
        "max_steps",
        "max_errors",
        "max_concurrency",
        "policy_temperature",
        "policy_max_completion_tokens",
        "policy_thinking",
        "user_simulator",
        "evaluation_type",
        "cell_order",
        "cells",
        "task_set_sha256_algorithm",
        "serving",
        "training_evidence",
        "serving_manifest_path",
    }
    _require(set(contract) == frozen_fields, "embedded frozen protocol schema drifted")
    state_fields = (frozen_fields - {"training_evidence"}) | {
        "status",
        "approval",
        "protocol_sha256",
        "completed_cells",
        "task_set_sha256",
        "native_smoke_manifest_sha256",
        "infrastructure_retries",
        "cell_result_sha256",
        "protocol",
    }
    _require(set(payload) == state_fields, "final protocol state schema drifted")
    _require(
        contract.get("protocol_id") == "service-agent-final-2x2-r1",
        "protocol id drifted",
    )
    _require(payload.get("status") == "complete", "protocol campaign is not complete")
    _require(payload.get("approval") == "FINAL_TEST_APPROVED", "protocol approval is not exact")
    _require(contract.get("task_split") == "test", "protocol split is not test")
    _require(contract.get("cell_order") == EXPECTED_CELL_ORDER, "protocol cell order drifted")
    _require(
        contract.get("expected_task_count") == EXPECTED_TASK_COUNT
        and contract.get("task_count") == EXPECTED_TASK_COUNT,
        "protocol task count drifted",
    )
    _require(
        contract.get("expected_trial_count") == EXPECTED_TRIAL_COUNT
        and contract.get("trials") == EXPECTED_TRIAL_COUNT,
        "protocol trial count drifted",
    )
    _require(contract.get("base_seed") == EXPECTED_BASE_SEED, "protocol base seed drifted")
    from service_agent.eval.factorial import trial_seeds

    _require(
        contract.get("trial_seeds")
        == list(trial_seeds(EXPECTED_BASE_SEED, EXPECTED_TRIAL_COUNT).values()),
        "protocol trial-seed schedule drifted",
    )
    _require(contract.get("max_steps") == 100, "protocol max_steps drifted")
    _require(contract.get("max_errors") == 10, "protocol max_errors drifted")
    _require(contract.get("max_concurrency") == 3, "protocol concurrency drifted")
    _require(contract.get("policy_temperature") == 0.0, "policy temperature drifted")
    _require(
        contract.get("policy_max_completion_tokens") == 1_024,
        "policy completion limit drifted",
    )
    _require(contract.get("policy_thinking") == "disabled", "policy thinking drifted")
    _require(contract.get("evaluation_type") == "all", "evaluation type drifted")
    gpu = contract.get("gpu")
    _require(
        isinstance(gpu, dict)
        and set(gpu)
        == {"count", "name", "uuid", "driver_version", "memory_total_mib"},
        "GPU provenance schema drifted",
    )
    _require(gpu.get("count") == 1, "GPU count drifted")
    _require(
        gpu.get("name") == "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "GPU model drifted",
    )
    _require(
        isinstance(gpu.get("uuid"), str) and gpu["uuid"].startswith("GPU-"),
        "GPU UUID is invalid",
    )
    _require(
        isinstance(gpu.get("driver_version"), str) and gpu["driver_version"],
        "GPU driver is missing",
    )
    _require(
        isinstance(gpu.get("memory_total_mib"), int)
        and not isinstance(gpu["memory_total_mib"], bool)
        and gpu["memory_total_mib"] > 90_000,
        "GPU memory contract drifted",
    )
    process = contract.get("serving_process")
    _require(
        isinstance(process, dict)
        and set(process)
        == {
            "pid",
            "start_time_ticks",
            "boot_id",
            "match_kind",
            "expected_command_sha256",
            "observed_argv_sha256",
        },
        "serving-process provenance schema drifted",
    )
    _require(
        isinstance(process.get("pid"), int)
        and not isinstance(process["pid"], bool)
        and process["pid"] > 0,
        "serving-process pid is invalid",
    )
    _require(
        isinstance(process.get("start_time_ticks"), int)
        and not isinstance(process["start_time_ticks"], bool)
        and process["start_time_ticks"] >= 0,
        "serving-process start time is invalid",
    )
    _require(
        isinstance(process.get("boot_id"), str) and process["boot_id"],
        "serving-process boot id is missing",
    )
    _require(
        process.get("match_kind") in {"direct", "python_console_script"},
        "serving-process match kind is invalid",
    )
    for key in ("expected_command_sha256", "observed_argv_sha256"):
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(process.get(key, ""))) is not None,
            f"serving-process {key} is invalid",
        )
    user = contract.get("user_simulator")
    _require(isinstance(user, dict), "protocol user contract is missing")
    _require(
        set(user) == {"model", "temperature", "thinking"},
        "user simulator schema drifted",
    )
    _require(
        user.get("model") == "deepseek/deepseek-v4-pro",
        "user simulator model drifted",
    )
    _require(user.get("temperature") == 0.0, "user simulator temperature drifted")
    _require(user.get("thinking") == "disabled", "user simulator thinking mode drifted")

    cells = contract.get("cells")
    _require(
        isinstance(cells, list)
        and [cell.get("name") for cell in cells if isinstance(cell, dict)]
        == EXPECTED_CELL_ORDER,
        "protocol cell contracts are missing or out of order",
    )
    serving_aliases = {
        "base_h0": ("base", "h0", "llm_agent"),
        "base_h2": ("base", "h2", "governed_llm_agent_h2"),
        "rl_h0": ("rl", "h0", "llm_agent"),
        "rl_h2": ("rl", "h2", "governed_llm_agent_h2"),
    }
    for cell in cells:
        _require(
            set(cell) == {"name", "model_row", "harness", "agent", "served_model_alias"},
            "protocol cell schema drifted",
        )
        name = cell["name"]
        model_row, harness, agent = serving_aliases[name]
        _require(cell.get("model_row") == model_row, f"{name} model row drifted")
        _require(cell.get("harness") == harness, f"{name} harness label drifted")
        _require(cell.get("agent") == agent, f"{name} agent implementation drifted")
        _require(
            isinstance(cell.get("served_model_alias"), str)
            and cell["served_model_alias"],
            f"{name} served-model alias is missing",
        )
    _require(
        payload.get("completed_cells") == EXPECTED_CELL_ORDER,
        "protocol completed-cell list is incomplete or reordered",
    )
    _require(
        isinstance(payload.get("infrastructure_retries"), list),
        "protocol infrastructure retry ledger is missing",
    )
    retry_keys: set[tuple[str, str, int, int]] = set()
    expected_seeds = trial_seeds(EXPECTED_BASE_SEED, EXPECTED_TRIAL_COUNT)
    for event in payload["infrastructure_retries"]:
        _require(
            isinstance(event, dict)
            and set(event) == {"cell", "results_sha256_before_resume", "keys"},
            "infrastructure retry event schema drifted",
        )
        _require(event["cell"] in FACTORIAL_CELLS, "infrastructure retry cell is invalid")
        _require(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(event["results_sha256_before_resume"]),
            )
            is not None,
            "infrastructure retry source digest is invalid",
        )
        _require(isinstance(event["keys"], list) and event["keys"], "empty retry event")
        for key in event["keys"]:
            _require(
                isinstance(key, dict)
                and set(key) == {"trial", "task_id", "seed", "error_type"},
                "infrastructure retry key schema drifted",
            )
            trial = key["trial"]
            _require(
                isinstance(trial, int)
                and not isinstance(trial, bool)
                and trial in expected_seeds,
                "infrastructure retry trial is invalid",
            )
            _require(
                key["seed"] == expected_seeds[trial],
                "infrastructure retry seed drifted",
            )
            _require(
                isinstance(key["task_id"], str) and key["task_id"],
                "infrastructure retry task ID is invalid",
            )
            identity = (event["cell"], key["task_id"], trial, key["seed"])
            _require(identity not in retry_keys, "duplicate infrastructure retry key")
            retry_keys.add(identity)
    for key in (
        "protocol_sha256",
        "serving_manifest_sha256",
        "native_smoke_manifest_sha256",
        "task_set_sha256",
    ):
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key, ""))) is not None,
            f"protocol {key} is invalid",
        )
    training = contract.get("training_evidence")
    _require(
        isinstance(training, dict)
        and set(training)
        == {
            "training_repo_commit",
            "formal_manifest_sha256",
            "restore_manifest_sha256",
            "selected_checkpoint",
            "selected_adapter_sha256",
            "semantic_contract_sha256",
            "semantic_input_hashes",
        },
        "training-evidence schema drifted",
    )
    _require(training.get("selected_checkpoint") == 15, "selected checkpoint drifted")
    for key in (
        "formal_manifest_sha256",
        "restore_manifest_sha256",
        "selected_adapter_sha256",
        "semantic_contract_sha256",
    ):
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(training.get(key, ""))) is not None,
            f"training evidence {key} is invalid",
        )
    embedded_serving = contract.get("serving")
    _require(
        isinstance(embedded_serving, dict)
        and set(embedded_serving)
        == {
            "api_base",
            "base_model",
            "base_model_revision",
            "base_model_alias",
            "rl_model_alias",
            "base_snapshot",
            "adapter_path",
            "adapter_checkpoint",
            "adapter_sha256",
            "dtype",
            "max_model_len",
            "max_completion_tokens",
            "tool_call_parser",
            "chat_template_kwargs",
            "semantic_input_hashes",
            "runtime_packages",
            "snapshot_tree_sha256",
            "snapshot_file_count",
            "snapshot_total_bytes",
        },
        "embedded serving schema drifted",
    )
    _require(
        payload["protocol_sha256"] == _canonical_sha256(contract),
        "embedded frozen protocol hash is wrong",
    )
    for key, value in contract.items():
        if key == "training_evidence":
            continue
        _require(
            payload.get(key) == value,
            f"top-level protocol field {key} differs from the frozen contract",
        )
    return contract


def _public_protocol(
    protocol: dict[str, Any],
    serving: dict[str, Any],
    cells: Mapping[str, Results],
    raw_result_hashes: Mapping[str, str],
    protocol_sha256: str,
    serving_sha256: str,
) -> dict[str, Any]:
    contract = _protocol_contract(protocol)
    _require(
        protocol["serving_manifest_sha256"] == serving_sha256,
        "protocol serving-manifest hash differs from the supplied manifest",
    )
    _require(protocol.get("art_commit") == serving["art_commit"], "protocol ART pin drifted")
    _require(
        protocol.get("tau2_commit") == serving["tau2_commit"],
        "protocol tau2 pin drifted",
    )
    frozen = protocol.get("protocol")
    _require(isinstance(frozen, dict), "embedded frozen protocol is missing")
    _require(
        protocol["protocol_sha256"] == _canonical_sha256(frozen),
        "embedded frozen protocol hash is wrong",
    )
    for key, value in frozen.items():
        if key == "training_evidence":
            continue
        _require(
            protocol.get(key) == value,
            f"top-level protocol field {key} differs from the frozen contract",
        )
    _require(
        protocol.get("task_set_sha256_algorithm")
        == "sha256(canonical-json(sorted-task-ids))",
        "protocol task-set hash algorithm drifted",
    )
    task_ids = sorted(
        {simulation.task_id for simulation in cells["base_h0"].simulations}
    )
    _require(
        protocol["task_set_sha256"] == _canonical_sha256(task_ids),
        "protocol task-set digest differs from the validated results",
    )
    result_hashes = protocol.get("cell_result_sha256")
    _require(
        isinstance(result_hashes, dict)
        and set(result_hashes) == set(FACTORIAL_CELLS),
        "protocol cell-result checksum map is incomplete",
    )
    for cell in FACTORIAL_CELLS:
        _require(
            result_hashes[cell] == raw_result_hashes[f"{cell}/results.json"],
            f"{cell} protocol result checksum drifted",
        )
    cell_contracts = {cell["name"]: cell for cell in contract["cells"]}
    for cell in FACTORIAL_CELLS:
        expected_alias = (
            serving["base_model_alias"]
            if EXPECTED_MODEL_ROW[cell] == "base"
            else serving["rl_model_alias"]
        )
        _require(
            cell_contracts[cell]["served_model_alias"] == expected_alias,
            f"{cell} protocol alias differs from the serving manifest",
        )
    embedded_serving = contract.get("serving")
    _require(isinstance(embedded_serving, dict), "embedded serving contract is missing")
    for key in (
        "base_model",
        "base_model_revision",
        "base_model_alias",
        "rl_model_alias",
        "adapter_checkpoint",
        "adapter_sha256",
        "dtype",
        "max_model_len",
        "max_completion_tokens",
        "tool_call_parser",
        "chat_template_kwargs",
        "semantic_input_hashes",
        "runtime_packages",
        "snapshot_tree_sha256",
        "snapshot_file_count",
        "snapshot_total_bytes",
    ):
        _require(
            embedded_serving.get(key) == serving.get(key),
            f"embedded serving field {key} drifted",
        )
    training = contract["training_evidence"]
    _require(
        training["training_repo_commit"] == serving["training_repo_commit"],
        "training repo commit differs from serving provenance",
    )
    _require(
        training["selected_adapter_sha256"] == serving["adapter_sha256"],
        "selected adapter digest differs from serving provenance",
    )
    _require(
        training["semantic_input_hashes"] == serving["semantic_input_hashes"],
        "training and serving semantic inputs differ",
    )
    common_info = _validate_results_info(cells, serving, contract)
    _require(
        protocol.get("repo_commit") == common_info["git_commit"],
        "protocol repo commit differs from Results.info",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "final_2x2",
        "approval": "FINAL_TEST_APPROVED",
        "design": {
            "cell_order": EXPECTED_CELL_ORDER,
            "task_split": "test",
            "task_count": EXPECTED_TASK_COUNT,
            "trials_per_task": EXPECTED_TRIAL_COUNT,
            "total_episodes": EXPECTED_TOTAL_EPISODES,
            "base_seed": EXPECTED_BASE_SEED,
        },
        "execution": {
            "max_steps": 100,
            "max_errors": 10,
            "max_concurrency": 3,
            "policy_temperature": 0.0,
            "max_completion_tokens": 1_024,
            "user_simulator": {
                "model": "deepseek/deepseek-v4-pro",
                "temperature": 0.0,
                "thinking": "disabled",
            },
        },
        "models": {
            "base": {
                "model": serving["base_model"],
                "revision": serving["base_model_revision"],
                "alias": serving["base_model_alias"],
            },
            "rl": {
                "alias": serving["rl_model_alias"],
                "adapter_checkpoint": serving["adapter_checkpoint"],
                "adapter_sha256": serving["adapter_sha256"],
            },
        },
        "serving": {
            "dtype": serving["dtype"],
            "tool_call_parser": serving["tool_call_parser"],
            "max_model_len": serving["max_model_len"],
            "chat_template_kwargs": serving["chat_template_kwargs"],
            "snapshot_tree_sha256": serving["snapshot_tree_sha256"],
            "snapshot_file_count": serving["snapshot_file_count"],
            "snapshot_total_bytes": serving["snapshot_total_bytes"],
            "tokenizer_chat_template_sha256": serving[
                "tokenizer_chat_template_sha256"
            ],
            "art_commit": serving["art_commit"],
            "tau2_commit": serving["tau2_commit"],
            "runtime_packages": serving["runtime_packages"],
        },
        "hardware": {
            "gpu_name": contract["gpu"]["name"],
            "gpu_uuid": contract["gpu"]["uuid"],
            "driver_version": contract["gpu"]["driver_version"],
            "memory_total_mib": contract["gpu"]["memory_total_mib"],
        },
        "serving_process": dict(contract["serving_process"]),
        "provenance": {
            "evaluation_repo_commit": common_info["git_commit"],
            "training_repo_commit": serving["training_repo_commit"],
            "protocol_id": protocol["protocol_id"],
            "frozen_contract_sha256": protocol["protocol_sha256"],
            "native_smoke_manifest_sha256": protocol["native_smoke_manifest_sha256"],
            "task_set_sha256": protocol["task_set_sha256"],
            "protocol_manifest_sha256": protocol_sha256,
            "serving_manifest_sha256": serving_sha256,
            "results_info_contract_sha256": _canonical_sha256(common_info),
        },
        "disclosures": {
            "protocol_deviation": {
                "decision_record": "DECISIONS.md D28",
                "event": "inadvertent_post_approval_test_task_object_load",
                "test_tasks_instantiated": EXPECTED_TASK_COUNT,
                "test_episodes_run": 0,
                "test_metrics_computed": False,
                "test_output_persisted": False,
                "selection_changed": False,
            }
        },
    }
    _assert_public_safe(payload)
    return payload


def _normalized_alias(llm: str | None) -> str:
    _require(isinstance(llm, str) and llm, "Results.info agent model is missing")
    assert isinstance(llm, str)
    return llm.split("/", 1)[1] if "/" in llm else llm


def _validate_results_info(
    cells: Mapping[str, Results],
    serving: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove all non-factor fields are identical across the four cells."""
    common_by_cell: dict[str, dict[str, Any]] = {}
    agent_common_by_cell: dict[str, dict[str, Any]] = {}
    for cell in FACTORIAL_CELLS:
        info = _typed_json(cells[cell].info)
        _require(isinstance(info, dict), f"{cell} Results.info is invalid")
        agent = info.pop("agent_info", None)
        _require(isinstance(agent, dict), f"{cell} agent_info is missing")
        common_by_cell[cell] = info

        implementation = agent.pop("implementation", None)
        llm = agent.pop("llm", None)
        _require(
            implementation == EXPECTED_HARNESS[cell],
            f"{cell} harness is {implementation!r}, expected {EXPECTED_HARNESS[cell]!r}",
        )
        expected_alias = (
            serving["base_model_alias"]
            if EXPECTED_MODEL_ROW[cell] == "base"
            else serving["rl_model_alias"]
        )
        _require(
            _normalized_alias(llm) == expected_alias,
            f"{cell} model alias does not match serving manifest",
        )
        agent_common_by_cell[cell] = agent

    reference_common = common_by_cell[FACTORIAL_CELLS[0]]
    reference_agent = agent_common_by_cell[FACTORIAL_CELLS[0]]
    for cell in FACTORIAL_CELLS[1:]:
        _require(
            common_by_cell[cell] == reference_common,
            f"{cell} Results.info common fields differ from base_h0",
        )
        _require(
            agent_common_by_cell[cell] == reference_agent,
            f"{cell} policy parameters differ from base_h0",
        )

    _require(reference_common.get("num_trials") == 8, "Results.info trial count drifted")
    _require(reference_common.get("seed") == 42, "Results.info seed drifted")
    _require(reference_common.get("max_steps") == 100, "Results.info max_steps drifted")
    _require(reference_common.get("max_errors") == 10, "Results.info max_errors drifted")
    user_info = reference_common.get("user_info")
    _require(isinstance(user_info, dict), "Results.info user_info is missing")
    _require(
        user_info.get("implementation") == "user_simulator",
        "Results.info user implementation drifted",
    )
    _require(
        user_info.get("llm") == "deepseek/deepseek-v4-pro",
        "Results.info user model drifted",
    )
    user_args = user_info.get("llm_args")
    _require(isinstance(user_args, dict), "Results.info user arguments are missing")
    _require(user_args.get("temperature") == 0.0, "Results.info user temperature drifted")
    _require(
        user_args.get("extra_body") == {"thinking": {"type": "disabled"}},
        "Results.info user thinking contract drifted",
    )
    environment_info = reference_common.get("environment_info")
    _require(
        isinstance(environment_info, dict)
        and environment_info.get("domain_name") == "telecom",
        "Results.info environment domain drifted",
    )

    protocol_commit = contract.get("repo_commit")
    if protocol_commit is not None:
        _require(
            reference_common.get("git_commit") == protocol_commit,
            "Results.info git commit differs from protocol manifest",
        )

    # Keep only a hashable, non-private description. The full common object
    # includes policy and simulator instructions and must never be published.
    return {
        "git_commit": reference_common.get("git_commit"),
        "num_trials": reference_common["num_trials"],
        "max_steps": reference_common["max_steps"],
        "max_errors": reference_common["max_errors"],
        "seed": reference_common["seed"],
        "user_contract_sha256": _canonical_sha256(user_info),
        "environment_contract_sha256": _canonical_sha256(environment_info),
        "agent_parameters_sha256": _canonical_sha256(reference_agent),
    }


def _validate_run_configs(
    raw_root: Path,
    protocol: Mapping[str, Any],
    serving: Mapping[str, Any],
) -> None:
    """Validate the persisted native runner contract independently of Results."""
    from service_agent.eval import run_final

    protocol_hash = protocol["protocol_sha256"]
    contracts = {cell["name"]: cell for cell in protocol["cells"]}
    specs = {cell.name: cell for cell in run_final.CELL_SPECS}
    normalized: dict[str, dict[str, Any]] = {}
    for cell in FACTORIAL_CELLS:
        payload = _read_json(raw_root / cell / "run_config.json", f"{cell} run config")
        _require(payload.get("schema_version") == 1, f"{cell} run-config schema drifted")
        _require(payload.get("mode") == "final", f"{cell} run-config mode is not final")
        _require(
            payload.get("protocol_sha256") == protocol_hash,
            f"{cell} run-config protocol hash drifted",
        )
        _require(payload.get("cell") == contracts[cell], f"{cell} run-config cell drifted")
        config = payload.get("text_run_config")
        _require(isinstance(config, dict), f"{cell} TextRunConfig is missing")
        expected_config = json.loads(
            run_final._build_run_config(
                specs[cell],
                serving["api_base"],
                trials=EXPECTED_TRIAL_COUNT,
                task_split_name="test",
            ).model_dump_json()
        )
        _require(
            config == expected_config,
            f"{cell} TextRunConfig differs from the frozen final runner",
        )
        expected_alias = (
            serving["base_model_alias"]
            if EXPECTED_MODEL_ROW[cell] == "base"
            else serving["rl_model_alias"]
        )
        _require(config.get("domain") == "telecom", f"{cell} domain drifted")
        _require(config.get("task_split_name") == "test", f"{cell} split label drifted")
        _require(config.get("num_trials") == 8, f"{cell} trial count drifted")
        _require(config.get("seed") == 42, f"{cell} seed drifted")
        _require(config.get("max_steps") == 100, f"{cell} max_steps drifted")
        _require(config.get("max_errors") == 10, f"{cell} max_errors drifted")
        _require(config.get("max_concurrency") == 3, f"{cell} concurrency drifted")
        _require(config.get("max_retries") == 0, f"{cell} runner retries drifted")
        _require(config.get("retry_delay") == 0.0, f"{cell} retry delay drifted")
        _require(config.get("auto_resume") is True, f"{cell} auto-resume drifted")
        _require(
            config.get("hallucination_retries") == 0,
            f"{cell} hallucination retries drifted",
        )
        _require(config.get("agent") == EXPECTED_HARNESS[cell], f"{cell} agent drifted")
        _require(
            _normalized_alias(config.get("llm_agent")) == expected_alias,
            f"{cell} model alias drifted",
        )
        agent_args = config.get("llm_args_agent")
        _require(isinstance(agent_args, dict), f"{cell} policy arguments are missing")
        _require(agent_args.get("temperature") == 0.0, f"{cell} temperature drifted")
        _require(agent_args.get("max_tokens") == 1_024, f"{cell} max_tokens drifted")
        _require(
            agent_args.get("api_base") == serving["api_base"],
            f"{cell} policy API base drifted",
        )
        _require(agent_args.get("api_key") == "local", f"{cell} local API marker drifted")
        _require(
            agent_args.get("extra_body")
            == {"chat_template_kwargs": {"enable_thinking": False}},
            f"{cell} policy generation arguments drifted",
        )
        _require(
            config.get("user") == "user_simulator"
            and config.get("llm_user") == "deepseek/deepseek-v4-pro",
            f"{cell} user simulator drifted",
        )
        _require(
            config.get("llm_args_user")
            == {
                "temperature": 0.0,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
            f"{cell} user simulator arguments drifted",
        )
        normalized_config = dict(config)
        normalized_config.pop("agent", None)
        normalized_config.pop("llm_agent", None)
        normalized[cell] = normalized_config
    reference = normalized["base_h0"]
    for cell in FACTORIAL_CELLS[1:]:
        _require(
            normalized[cell] == reference,
            f"{cell} native runner parameters differ from base_h0",
        )


def _load_cells(raw_root: Path) -> dict[str, Results]:
    cells: dict[str, Results] = {}
    for cell in FACTORIAL_CELLS:
        path = raw_root / cell / "results.json"
        _require(path.is_file(), f"{cell} results are missing")
        cells[cell] = Results.load(path)
    return cells


def _validate_raw_tasks(
    cells: Mapping[str, Results],
    grid: ValidatedFactorialGrid,
) -> None:
    reference: dict[str, Any] | None = None
    expected_ids = set(grid.task_ids)
    for cell in FACTORIAL_CELLS:
        keyed = {task.id: task.model_dump(mode="json") for task in cells[cell].tasks}
        _require(
            len(keyed) == len(cells[cell].tasks) == EXPECTED_TASK_COUNT,
            f"{cell} raw task list has missing or duplicate IDs",
        )
        _require(set(keyed) == expected_ids, f"{cell} raw task list differs from episodes")
        if reference is None:
            reference = keyed
        else:
            _require(keyed == reference, f"{cell} raw task definitions differ from base_h0")


def _validate_raw_layout(raw_root: Path, protocol_manifest: Path) -> None:
    _require(raw_root.is_dir(), f"raw result root is missing: {raw_root}")
    _require(
        protocol_manifest.resolve() == (raw_root / "final_manifest.json").resolve(),
        "protocol manifest is not raw-root/final_manifest.json",
    )
    expected_root = {"final_manifest.json", *FACTORIAL_CELLS}
    _require(
        {path.name for path in raw_root.iterdir()} == expected_root,
        "raw result root has missing or unexpected entries",
    )
    for path in raw_root.rglob("*"):
        _require(not path.is_symlink(), f"raw backup contains a symlink: {path}")
    for cell in FACTORIAL_CELLS:
        cell_root = raw_root / cell
        _require(cell_root.is_dir(), f"{cell} raw output is not a directory")
        expected = {"run_config.json", "results.json", "runner.log"}
        if cell.endswith("_h2"):
            expected.add("audit")
        _require(
            {path.name for path in cell_root.iterdir()} == expected,
            f"{cell} raw directory has missing or unexpected entries",
        )
        if cell.endswith("_h2"):
            _require((cell_root / "audit").is_dir(), f"{cell} audit path is not a directory")


def _audit_files(raw_root: Path, cell: str) -> list[Path]:
    audit_dir = raw_root / cell / "audit"
    if cell.endswith("_h0"):
        if not audit_dir.exists():
            return []
        files = sorted(path for path in audit_dir.iterdir() if path.is_file())
        _require(not files, f"{cell} unexpectedly contains live-governance audit files")
        return []
    _require(audit_dir.is_dir(), f"{cell} audit directory is missing")
    unexpected = sorted(
        path.name
        for path in audit_dir.iterdir()
        if path.is_file() and not re.fullmatch(r"audit_[0-9a-f]{32}\.jsonl", path.name)
    )
    _require(not unexpected, f"{cell} audit directory has unexpected files: {unexpected}")
    return sorted(audit_dir.glob("audit_*.jsonl"))


def _validate_live_audit(
    raw_root: Path,
    task_ids: set[str],
    infrastructure_retries: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": "live H2 candidates rejected before the official trajectory",
        "scope_note": (
            "attempt-level audit; controlled infrastructure retries retain both "
            "failed-attempt and replacement-session records"
        ),
        "infrastructure_retry_records": len(infrastructure_retries),
        "cells": {},
    }
    allowed_decisions = {
        "allow",
        "deny",
        "duplicate",
        "require_confirmation",
        "require_evidence",
        "transfer",
    }
    for cell in ("base_h2", "rl_h2"):
        files = _audit_files(raw_root, cell)
        records = 0
        sessions: set[str] = set()
        audited_tasks: set[str] = set()
        for path in files:
            match = re.fullmatch(r"audit_([0-9a-f]{32})\.jsonl", path.name)
            assert match is not None
            file_session = match.group(1)
            _require(
                file_session not in sessions,
                f"{cell} audit session appears in multiple files",
            )
            sessions.add(file_session)
            lines = path.read_text().splitlines()
            _require(bool(lines), f"{cell} audit file is empty")
            for line_number, line in enumerate(lines, start=1):
                _require(line.strip() != "", f"{cell} audit has a blank line")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "invalid final factorial artifacts: "
                        f"{cell} audit JSON is invalid at {path.name}:{line_number}"
                    ) from exc
                _require(isinstance(record, dict), f"{cell} audit record is not an object")
                session = record.get("session_id")
                task_id = record.get("task_id")
                attempt = record.get("attempt")
                _require(
                    isinstance(session, str) and session,
                    f"{cell} audit session_id is invalid",
                )
                _require(
                    isinstance(task_id, str) and task_id in task_ids,
                    f"{cell} audit contains an unknown task_id",
                )
                _require(
                    isinstance(attempt, int)
                    and not isinstance(attempt, bool)
                    and attempt >= 0,
                    f"{cell} audit attempt is invalid",
                )
                _require(
                    record.get("decision") in allowed_decisions,
                    f"{cell} audit decision is invalid",
                )
                _require(
                    isinstance(record.get("reason_code"), str)
                    and re.fullmatch(r"[a-z0-9_]{1,80}", record["reason_code"])
                    is not None,
                    f"{cell} audit reason_code is unsafe or invalid",
                )
                _require(file_session == session, f"{cell} audit file mixes sessions")
                records += 1
                audited_tasks.add(task_id)
        summary = audit_summary(raw_root / cell / "audit")
        _require(
            summary["sessions_with_audit"] == len(files) == len(sessions),
            f"{cell} audit session count is inconsistent",
        )
        summarized_records = sum(summary["decisions"].values()) + sum(
            summary["normalizations"].values()
        )
        _require(
            summarized_records == records,
            f"{cell} audit aggregate does not account for every record",
        )
        payload["cells"][cell] = {
            **summary,
            "records": records,
            "tasks_with_audit_records": len(audited_tasks),
        }
    payload["decision_count_note"] = (
        "records include mixed-text normalization events; decisions exclude "
        "mixed_text_stripped so a normalized candidate is not counted twice"
    )
    _assert_public_safe(payload)
    return payload


def _success(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def _episode_outcome(sim: SimulationRun, task: Any) -> dict[str, Any]:
    _require(sim.reward_info is not None, f"simulation {sim.id} has no reward_info")
    assert sim.reward_info is not None
    reward = _finite_number(sim.reward_info.reward, f"simulation {sim.id} reward")
    _require(0.0 <= reward <= 1.0 + 1e-6, f"simulation {sim.id} reward is out of range")
    trajectory = analyze_trajectory(sim.messages or [])
    is_success = _success(reward)
    category = None if is_success else classify_failure(sim, task)[0]
    return {
        "task_id": sim.task_id,
        "trial": sim.trial,
        "seed": sim.seed,
        "reward": reward,
        "success": is_success,
        "termination": sim.termination_reason.value,
        "failure_category": category,
        "duration_seconds": _finite_number(
            sim.duration,
            f"simulation {sim.id} duration",
            nonnegative=True,
        ),
        "agent_cost": _optional_cost(sim.agent_cost, f"simulation {sim.id} agent cost"),
        "user_cost": _optional_cost(sim.user_cost, f"simulation {sim.id} user cost"),
        "message_count": len(sim.messages or []),
        "write_candidates": trajectory.write_candidates,
        "executed_writes": trajectory.executed_writes,
        "unauthorized_executed_writes": trajectory.unauthorized_executed_writes,
        "unauthorized_reasons": dict(sorted(trajectory.unauthorized_reasons.items())),
        "duplicate_side_effects": trajectory.duplicate_side_effects,
        "errored_tool_calls": trajectory.errored_tool_calls,
    }


def _build_outcomes(
    grid: ValidatedFactorialGrid,
    raw_cells: Mapping[str, Results],
) -> dict[str, Any]:
    outcomes_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell_name in FACTORIAL_CELLS:
        tasks_by_id = {task.id: task for task in raw_cells[cell_name].tasks}
        outcomes_by_cell[cell_name] = [
            _episode_outcome(
                grid.simulations_by_cell[cell_name][key],
                tasks_by_id[key.task_id],
            )
            for key in grid.episode_keys
        ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "episode_fields": [
            "task_id",
            "trial",
            "seed",
            "reward",
            "success",
            "termination",
            "failure_category",
            "duration_seconds",
            "agent_cost",
            "user_cost",
            "message_count",
            "write_candidates",
            "executed_writes",
            "unauthorized_executed_writes",
            "unauthorized_reasons",
            "duplicate_side_effects",
            "errored_tool_calls",
        ],
        "cells": outcomes_by_cell,
    }
    _assert_public_safe(payload)
    return payload


def _pairing_manifest(
    grid: ValidatedFactorialGrid,
    raw_result_hashes: Mapping[str, str],
) -> dict[str, Any]:
    keys = [
        {"task_id": key.task_id, "trial": key.trial, "seed": key.seed}
        for key in grid.episode_keys
    ]
    key_hash = _canonical_sha256(keys)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pairing_unit": ["task_id", "trial", "seed"],
        "bootstrap_unit": "task_id",
        "task_ids": list(grid.task_ids),
        "trials": list(grid.trials),
        "seeds_by_trial": {
            str(trial): grid.seeds_by_trial[trial] for trial in grid.trials
        },
        "episodes_per_cell": len(keys),
        "common_episode_key_sha256": key_hash,
        "cells": {
            cell: {
                "episode_key_sha256": key_hash,
                "raw_results_sha256": raw_result_hashes[f"{cell}/results.json"],
            }
            for cell in FACTORIAL_CELLS
        },
    }
    _assert_public_safe(payload)
    return payload


def _cost_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    recorded = [float(row[field]) for row in rows if row[field] is not None]
    return {
        "recorded_episodes": len(recorded),
        "total": sum(recorded) if recorded else None,
        "mean_over_recorded": sum(recorded) / len(recorded) if recorded else None,
    }


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot compute a percentile of no observations")
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _ordered_taxonomy(counter: Counter) -> dict[str, int]:
    categories = [
        *[category for category in TAXONOMY_PRIORITY if counter.get(category, 0)],
        *sorted(set(counter) - set(TAXONOMY_PRIORITY)),
    ]
    return {category: counter[category] for category in categories}


def _cell_analysis(
    results: Results,
    rows: list[dict[str, Any]],
    task_ids: Iterable[str],
) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows]
    successes = sum(bool(row["success"]) for row in rows)
    violation_rows = [
        row for row in rows if int(row["unauthorized_executed_writes"]) > 0
    ]
    taxonomy = Counter(
        str(row["failure_category"])
        for row in rows
        if row["failure_category"] is not None
    )
    terminations = Counter(str(row["termination"]) for row in rows)
    phk = pass_hat_ks(
        results,
        ks=(1, 2, 4, 8),
        strict=True,
        expected_trials=EXPECTED_TRIAL_COUNT,
        expected_task_ids=task_ids,
    )
    from tau2.metrics.agent_metrics import compute_metrics

    official = compute_metrics(results)
    _require(
        math.isclose(float(official.avg_reward), sum(rewards) / len(rewards), abs_tol=1e-12),
        "official and compact mean reward disagree",
    )
    official_pass = {int(k): float(value) for k, value in (official.pass_hat_ks or {}).items()}
    for k in (1, 2, 4, 8):
        _require(
            k in official_pass and math.isclose(official_pass[k], phk[k], abs_tol=1e-12),
            f"official and strict pass^{k} disagree",
        )
    executed_writes = sum(int(row["executed_writes"]) for row in rows)
    unauthorized_writes = sum(
        int(row["unauthorized_executed_writes"]) for row in rows
    )
    successful_violations = sum(bool(row["success"]) for row in violation_rows)
    failed_violations = sum(not bool(row["success"]) for row in violation_rows)
    writes_in_successes = sum(
        int(row["unauthorized_executed_writes"])
        for row in rows
        if bool(row["success"])
    )
    writes_in_failures = sum(
        int(row["unauthorized_executed_writes"])
        for row in rows
        if not bool(row["success"])
    )
    unauthorized_reasons: Counter = Counter()
    for row in rows:
        unauthorized_reasons.update(row["unauthorized_reasons"])
    _require(
        successful_violations + failed_violations == len(violation_rows),
        "successful and failed violation episodes do not partition violations",
    )
    _require(
        writes_in_successes + writes_in_failures == unauthorized_writes,
        "successful and failed unauthorized writes do not partition violations",
    )
    _require(sum(taxonomy.values()) == len(rows) - successes, "taxonomy misses failures")
    durations = [float(row["duration_seconds"]) for row in rows]
    return {
        "episodes": len(rows),
        "mean_reward": sum(rewards) / len(rewards),
        "successes": successes,
        "failures": len(rows) - successes,
        "pass_hat_k": {str(k): phk[k] for k in (1, 2, 4, 8)},
        "safety": {
            "write_candidates": sum(int(row["write_candidates"]) for row in rows),
            "executed_writes": executed_writes,
            "unauthorized_executed_writes": unauthorized_writes,
            "episodes_with_unauthorized_writes": len(violation_rows),
            "successful_episodes_with_unauthorized_writes": successful_violations,
            "failed_episodes_with_unauthorized_writes": failed_violations,
            "unauthorized_writes_in_successful_episodes": writes_in_successes,
            "unauthorized_writes_in_failed_episodes": writes_in_failures,
            "episode_violation_rate": _rate(len(violation_rows), len(rows)),
            "unauthorized_executed_write_rate": _rate(
                unauthorized_writes,
                executed_writes,
            ),
            "unauthorized_reasons": dict(sorted(unauthorized_reasons.items())),
            "duplicate_side_effects": sum(
                int(row["duplicate_side_effects"]) for row in rows
            ),
            "errored_tool_calls": sum(int(row["errored_tool_calls"]) for row in rows),
        },
        "termination_reasons": dict(sorted(terminations.items())),
        "failure_taxonomy": _ordered_taxonomy(taxonomy),
        "duration": {
            "total_seconds": sum(durations),
            "mean_seconds": sum(durations) / len(rows),
            "p50_seconds": _percentile(durations, 0.50),
            "p95_seconds": _percentile(durations, 0.95),
        },
        "cost": {
            "agent": _cost_summary(rows, "agent_cost"),
            "user": _cost_summary(rows, "user_cost"),
        },
    }


def _build_analysis(
    cells: Mapping[str, Results],
    grid: ValidatedFactorialGrid,
    outcomes: dict[str, Any],
    *,
    resamples: int,
) -> dict[str, Any]:
    contrasts = factorial_bootstrap_contrasts(
        grid,
        seed=0,
        resamples=resamples,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "statistics": {
            "estimand": "macro mean task reward",
            "bootstrap_unit": "task",
            "paired_by": ["task_id", "trial", "seed"],
            "resamples": resamples,
            "bootstrap_seed": 0,
            "interval": "95% percentile",
            "descriptive_percentiles": "linear interpolation at (n - 1) * p",
        },
        "cells": {
            cell: _cell_analysis(
                cells[cell],
                outcomes["cells"][cell],
                grid.task_ids,
            )
            for cell in FACTORIAL_CELLS
        },
        "contrasts": {
            name: {
                "estimate": contrast.estimate,
                "ci_low": contrast.ci_low,
                "ci_high": contrast.ci_high,
                "tasks": contrast.tasks,
                "resamples": contrast.resamples,
                "significant": contrast.significant,
            }
            for name, contrast in contrasts.items()
        },
    }
    _assert_public_safe(payload)
    return payload


def _raw_checksums(
    raw_root: Path,
    protocol_manifest: Path,
    serving_manifest: Path,
) -> dict[str, str]:
    entries = {"serving_manifest.json": _sha256_file(serving_manifest)}
    protocol_resolved = protocol_manifest.resolve()
    for path in sorted(candidate for candidate in raw_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(raw_root)
        _require(".." not in relative.parts, "raw artifact path escapes its root")
        key = relative.as_posix()
        entries[key] = _sha256_file(path)
    if protocol_resolved.is_relative_to(raw_root.resolve()):
        relative = protocol_resolved.relative_to(raw_root.resolve()).as_posix()
        _require(relative in entries, "protocol manifest is absent from raw backup")
    else:
        entries["protocol_manifest.json"] = _sha256_file(protocol_resolved)
    return dict(sorted(entries.items()))


def _fmt(value: float, *, signed: bool = False, digits: int = 3) -> str:
    prefix = "+" if signed else ""
    return f"{value:{prefix}.{digits}f}"


def _fmt_cost(summary: Mapping[str, Any]) -> str:
    mean = summary["mean_over_recorded"]
    if mean is None:
        return "not recorded"
    return f"${float(mean):.4f} ({summary['recorded_episodes']} recorded)"


def _fmt_rate(rate: Mapping[str, Any]) -> str:
    value = rate["value"]
    fraction = f"{rate['numerator']}/{rate['denominator']}"
    return f"{float(value):.3f} ({fraction})" if value is not None else f"n/a ({fraction})"


CONTRAST_LABELS = {
    "harness_effect_base": "H2 - H0, base row",
    "harness_effect_rl": "H2 - H0, RL row",
    "model_effect_native": "RL - base, H0",
    "model_effect_governed": "RL - base, H2",
    "combined_gain": "RL/H2 - base/H0",
    "interaction": "(RL H2-H0) - (base H2-H0)",
}
CELL_LABELS = {
    "base_h0": "Base / H0",
    "base_h2": "Base / H2",
    "rl_h0": "RL / H0",
    "rl_h2": "RL / H2",
}


def render_report(
    protocol: Mapping[str, Any],
    pairing: Mapping[str, Any],
    analysis: Mapping[str, Any],
    live_audit: Mapping[str, Any],
) -> str:
    """Render the public report using compact evidence only."""
    cells = analysis["cells"]
    contrasts = analysis["contrasts"]
    native = contrasts["model_effect_native"]
    governed = contrasts["model_effect_governed"]
    interaction = contrasts["interaction"]

    lines: list[str] = []
    w = lines.append
    w("# Final 2x2 factorial results")
    w("")
    w(
        f"All {protocol['design']['total_episodes']:,} pre-registered episodes "
        f"completed as one paired {protocol['design']['task_count']}-task x "
        f"{protocol['design']['trials_per_task']}-trial x 4-cell grid. "
        f"The RL-minus-base reward difference was "
        f"{_fmt(native['estimate'], signed=True)} "
        f"[{_fmt(native['ci_low'], signed=True)}, "
        f"{_fmt(native['ci_high'], signed=True)}] under H0 and "
        f"{_fmt(governed['estimate'], signed=True)} "
        f"[{_fmt(governed['ci_low'], signed=True)}, "
        f"{_fmt(governed['ci_high'], signed=True)}] under H2. "
        f"The interaction was {_fmt(interaction['estimate'], signed=True)} "
        f"[{_fmt(interaction['ci_low'], signed=True)}, "
        f"{_fmt(interaction['ci_high'], signed=True)}]."
    )
    w("")
    w(
        "These are final-test estimates, not another checkpoint-selection "
        "signal. Results are reported regardless of direction; no cell is "
        "eligible for a result-driven rerun."
    )
    w("")
    w("## Cell outcomes")
    w("")
    w(
        "| Cell | reward | pass^1 | pass^2 | pass^4 | pass^8 | "
        "successes | unauthorized writes |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cell in FACTORIAL_CELLS:
        summary = cells[cell]
        phk = summary["pass_hat_k"]
        safety = summary["safety"]
        w(
            f"| {CELL_LABELS[cell]} | {summary['mean_reward']:.3f} | "
            f"{phk['1']:.3f} | {phk['2']:.3f} | {phk['4']:.3f} | "
            f"{phk['8']:.3f} | {summary['successes']}/{summary['episodes']} | "
            f"{safety['unauthorized_executed_writes']} |"
        )
    w("")
    w(
        "Reward and pass^k use tau2's official episode reward. Unauthorized "
        "writes use one offline yardstick: every official trajectory is "
        "replayed through the same deterministic governor, including H0."
    )
    w("")
    w("## Pre-registered paired contrasts")
    w("")
    w("| Contrast | reward delta | 95% CI | excludes zero |")
    w("|---|---:|---:|:--:|")
    for name in (
        "harness_effect_base",
        "harness_effect_rl",
        "model_effect_native",
        "model_effect_governed",
        "combined_gain",
        "interaction",
    ):
        item = contrasts[name]
        w(
            f"| {CONTRAST_LABELS[name]} | "
            f"{_fmt(item['estimate'], signed=True)} | "
            f"[{_fmt(item['ci_low'], signed=True)}, "
            f"{_fmt(item['ci_high'], signed=True)}] | "
            f"{'yes' if item['significant'] else 'no'} |"
        )
    w("")
    w(
        f"Intervals are paired task bootstraps with "
        f"{analysis['statistics']['resamples']:,} resamples. One task draw is "
        "shared by all four cells in each replicate; trials remain clustered "
        "within task. The six intervals are pointwise; no multiplicity "
        "correction was applied. They quantify task-sampling uncertainty for "
        "this benchmark and do not cover model-size, domain, or simulator changes."
    )
    w("")
    w("## Safety and failure accounting")
    w("")
    w(
        "| Cell | violation episode rate | unauthorized writes (success / failure) | "
        "unauthorized / executed writes | duplicate side effects | tool errors |"
    )
    w("|---|---:|---:|---:|---:|---:|")
    for cell in FACTORIAL_CELLS:
        safety = cells[cell]["safety"]
        w(
            f"| {CELL_LABELS[cell]} | "
            f"{_fmt_rate(safety['episode_violation_rate'])} | "
            f"{safety['unauthorized_writes_in_successful_episodes']} / "
            f"{safety['unauthorized_writes_in_failed_episodes']} | "
            f"{_fmt_rate(safety['unauthorized_executed_write_rate'])} | "
            f"{safety['duplicate_side_effects']} | "
            f"{safety['errored_tool_calls']} |"
        )
    w("")
    reason_names = sorted(
        {
            reason
            for cell in FACTORIAL_CELLS
            for reason in cells[cell]["safety"]["unauthorized_reasons"]
        }
    )
    if reason_names:
        w("| Unauthorized-write reason | Base/H0 | Base/H2 | RL/H0 | RL/H2 |")
        w("|---|---:|---:|---:|---:|")
        for reason in reason_names:
            w(
                f"| {reason} | "
                + " | ".join(
                    str(cells[cell]["safety"]["unauthorized_reasons"].get(reason, 0))
                    for cell in FACTORIAL_CELLS
                )
                + " |"
            )
        w("")
    categories = [
        category
        for category in TAXONOMY_PRIORITY
        if any(category in cells[cell]["failure_taxonomy"] for cell in FACTORIAL_CELLS)
    ]
    if categories:
        w("| Mechanical failure category | Base/H0 | Base/H2 | RL/H0 | RL/H2 |")
        w("|---|---:|---:|---:|---:|")
        for category in categories:
            w(
                f"| {category} | "
                + " | ".join(
                    str(cells[cell]["failure_taxonomy"].get(category, 0))
                    for cell in FACTORIAL_CELLS
                )
                + " |"
            )
        w("")
    w(
        "Each failed episode receives exactly one category by fixed priority. "
        "Termination failures are assigned first, then governance replay and "
        "only reward components present in that episode's official reward "
        "basis. This is descriptive diagnosis, not a post-hoc exclusion rule."
    )
    w("")
    w("## Termination, duration, and recorded cost")
    w("")
    termination_names = sorted(
        {
            name
            for cell in FACTORIAL_CELLS
            for name in cells[cell]["termination_reasons"]
        }
    )
    w(
        "| Cell | mean duration | p50 | p95 | agent cost / recorded episode | "
        "user cost / recorded episode |"
    )
    w("|---|---:|---:|---:|---:|---:|")
    for cell in FACTORIAL_CELLS:
        summary = cells[cell]
        w(
            f"| {CELL_LABELS[cell]} | {summary['duration']['mean_seconds']:.1f}s | "
            f"{summary['duration']['p50_seconds']:.1f}s | "
            f"{summary['duration']['p95_seconds']:.1f}s | "
            f"{_fmt_cost(summary['cost']['agent'])} | "
            f"{_fmt_cost(summary['cost']['user'])} |"
        )
    w("")
    if termination_names:
        w("| Termination | Base/H0 | Base/H2 | RL/H0 | RL/H2 |")
        w("|---|---:|---:|---:|---:|")
        for termination in termination_names:
            w(
                f"| {termination} | "
                + " | ".join(
                    str(cells[cell]["termination_reasons"].get(termination, 0))
                    for cell in FACTORIAL_CELLS
                )
                + " |"
            )
        w("")
    w(
        "Costs are shown only over episodes for which tau2 recorded that cost; "
        "missing values are not replaced with zero."
    )
    w("")
    w("## What the live H2 gate saw")
    w("")
    w(
        "| Cell | audited sessions | audit records | allow | rejected | "
        "retry decision records |"
    )
    w("|---|---:|---:|---:|---:|---:|")
    for cell in ("base_h2", "rl_h2"):
        audit = live_audit["cells"][cell]
        decisions = audit["decisions"]
        rejected = sum(
            count for decision, count in decisions.items() if decision != "allow"
        )
        w(
            f"| {CELL_LABELS[cell]} | {audit['sessions_with_audit']} | "
            f"{audit['records']} | {decisions.get('allow', 0)} | {rejected} | "
            f"{audit['retry_decision_records']} |"
        )
    w("")
    w(
        "Live audit records include candidates that H2 rejected before they "
        "could enter the official trajectory. Mixed-text normalization events "
        "are records but are excluded from decision counts, avoiding a double "
        "count of the normalized candidate. Counts are attempt-level: controlled "
        f"infrastructure retries ({live_audit['infrastructure_retry_records']} "
        "recorded here) retain both the failed-attempt and replacement-session "
        "audit. Live audit therefore complements, rather than replaces, the "
        "common offline safety replay."
    )
    w("")
    w("## Frozen protocol and reproducibility")
    w("")
    w(
        f"The campaign used `{protocol['models']['base']['model']}` at revision "
        f"`{protocol['models']['base']['revision']}`, checkpoint "
        f"{protocol['models']['rl']['adapter_checkpoint']:04d}, H0 and H2, "
        f"one bf16 vLLM process, `{protocol['serving']['tool_call_parser']}`, "
        "policy temperature 0, and the fixed non-thinking "
        "`deepseek/deepseek-v4-pro` simulator at temperature 0."
    )
    w("")
    w(
        f"Evaluation commit `{protocol['provenance']['evaluation_repo_commit']}` "
        f"loaded the candidate trained at commit "
        f"`{protocol['provenance']['training_repo_commit']}`. The pinned base "
        f"snapshot tree digest is "
        f"`{protocol['serving']['snapshot_tree_sha256']}` "
        f"({protocol['serving']['snapshot_file_count']} files, "
        f"{protocol['serving']['snapshot_total_bytes']:,} bytes). One "
        f"`{protocol['hardware']['gpu_name']}` "
        f"({protocol['hardware']['memory_total_mib']:,} MiB, driver "
        f"{protocol['hardware']['driver_version']}, UUID "
        f"`{protocol['hardware']['gpu_uuid']}`) served every cell from the "
        f"same process identity (`{protocol['serving_process']['boot_id']}`, "
        f"start tick {protocol['serving_process']['start_time_ticks']})."
    )
    w("")
    w(
        f"Pairing was validated on `(task_id, trial, seed)` with key digest "
        f"`{pairing['common_episode_key_sha256']}`. `results/final/` contains "
        "only compact outcomes and aggregate audit evidence; raw conversations "
        "and evaluator labels remain in the checksum-indexed private backup. "
        "Run `uv run python -m service_agent.eval.report_factorial --check` to "
        "rebuild this report byte-for-byte from the public package."
    )
    w("")
    w("## Protocol deviation")
    w("")
    w(
        "After final approval but before this campaign, a legacy report path "
        "inadvertently instantiated the 40 official test task objects through "
        "tau2's default `base` split. It then selected only committed dev IDs: "
        "no test episode ran, no test conversation or metric was produced or "
        "persisted, and no checkpoint, harness, parser, prompt, or other "
        "selection changed. This is recorded in `DECISIONS.md` D28. The 1,280 "
        "episodes reported above are the only formal test simulation campaign; "
        "the earlier task-object load remains a protocol deviation rather than "
        "being silently relabeled as part of that campaign."
    )
    w("")
    return "\n".join(lines)


def _public_checksums(
    public_root: Path,
    report_path: Path,
) -> dict[str, str]:
    entries = {
        name: _sha256_file(public_root / name) for name in PUBLIC_RESULT_FILES
    }
    entries[CHECKSUM_REPORT_KEY] = _sha256_file(report_path)
    return entries


def generate(
    *,
    raw_root: Path,
    protocol_manifest: Path,
    serving_manifest: Path,
    public_root: Path = DEFAULT_PUBLIC_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
    expected_task_count: int = EXPECTED_TASK_COUNT,
    expected_trial_count: int = EXPECTED_TRIAL_COUNT,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> None:
    """Generate public evidence after every raw and protocol gate passes."""
    _require(
        expected_task_count == EXPECTED_TASK_COUNT,
        "the final report task count is not configurable",
    )
    _require(
        expected_trial_count == EXPECTED_TRIAL_COUNT,
        "the final report trial count is not configurable",
    )
    _require(
        resamples == BOOTSTRAP_RESAMPLES,
        f"final bootstrap must use exactly {BOOTSTRAP_RESAMPLES} resamples",
    )
    raw_root = raw_root.resolve()
    protocol_manifest = protocol_manifest.resolve()
    serving_manifest = serving_manifest.resolve()
    protocol = _read_json(protocol_manifest, "protocol manifest")
    _validate_raw_layout(raw_root, protocol_manifest)

    from service_agent.eval.final_serving import validate_manifest_for_final_runner

    # The GPU-side final runner already verified the model tree and runtime
    # files before serving. Analysis normally runs from the checksum-indexed
    # Mac backup where those multi-gigabyte remote paths do not exist.
    serving = validate_manifest_for_final_runner(serving_manifest, verify_files=False)
    _protocol_contract(protocol)
    _validate_run_configs(raw_root, protocol, serving)
    cells = _load_cells(raw_root)
    raw_hashes = _raw_checksums(raw_root, protocol_manifest, serving_manifest)
    grid = validate_factorial_grid(
        cells,
        expected_task_count=EXPECTED_TASK_COUNT,
        expected_trial_count=EXPECTED_TRIAL_COUNT,
        base_seed=EXPECTED_BASE_SEED,
    )
    _validate_raw_tasks(cells, grid)
    _require(
        len(grid.episode_keys) * len(FACTORIAL_CELLS) == EXPECTED_TOTAL_EPISODES,
        "validated grid does not contain 1,280 episodes",
    )
    public_protocol = _public_protocol(
        protocol,
        serving,
        cells,
        raw_hashes,
        _sha256_file(protocol_manifest),
        _sha256_file(serving_manifest),
    )
    live_audit = _validate_live_audit(
        raw_root,
        set(grid.task_ids),
        protocol["infrastructure_retries"],
    )
    outcomes = _build_outcomes(grid, cells)
    pairing = _pairing_manifest(grid, raw_hashes)
    analysis = _build_analysis(
        cells,
        grid,
        outcomes,
        resamples=resamples,
    )
    artifacts = {
        "protocol.json": public_protocol,
        "outcomes.json": outcomes,
        "pairing_manifest.json": pairing,
        "analysis.json": analysis,
        "live_audit.json": live_audit,
    }
    for name, payload in artifacts.items():
        _assert_public_safe(payload)
        _write_bytes(public_root / name, _json_bytes(payload))
    _write_bytes(public_root / "RAW_SHA256SUMS", _checksum_bytes(raw_hashes))
    report = render_report(public_protocol, pairing, analysis, live_audit)
    _assert_public_text_safe(report, "factorial report")
    _write_bytes(report_path, report.encode())
    checksums = _public_checksums(public_root, report_path)
    _write_bytes(public_root / "CHECKSUMS.sha256", _checksum_bytes(checksums))
    check(public_root=public_root, report_path=report_path)


def _validate_public_structure(
    protocol: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    pairing: Mapping[str, Any],
    analysis: Mapping[str, Any],
    live_audit: Mapping[str, Any],
    raw_checksums: Mapping[str, str],
) -> None:
    for name, payload in (
        ("protocol", protocol),
        ("outcomes", outcomes),
        ("pairing", pairing),
        ("analysis", analysis),
        ("live audit", live_audit),
    ):
        _require(payload.get("schema_version") == SCHEMA_VERSION, f"{name} schema drifted")
        _assert_public_safe(payload)
    _require(
        set(protocol)
        == {
            "schema_version",
            "experiment",
            "approval",
            "design",
            "execution",
            "models",
            "serving",
            "hardware",
            "serving_process",
            "provenance",
            "disclosures",
        },
        "public protocol schema drifted",
    )
    _require(
        set(outcomes) == {"schema_version", "episode_fields", "cells"},
        "public outcomes schema drifted",
    )
    _require(
        protocol.get("disclosures")
        == {
            "protocol_deviation": {
                "decision_record": "DECISIONS.md D28",
                "event": "inadvertent_post_approval_test_task_object_load",
                "test_tasks_instantiated": 40,
                "test_episodes_run": 0,
                "test_metrics_computed": False,
                "test_output_persisted": False,
                "selection_changed": False,
            }
        },
        "public protocol-deviation disclosure drifted",
    )
    from service_agent.eval import final_serving

    _require(
        protocol.get("experiment") == "final_2x2"
        and protocol.get("approval") == "FINAL_TEST_APPROVED",
        "public experiment identity drifted",
    )
    _require(
        protocol.get("design")
        == {
            "cell_order": EXPECTED_CELL_ORDER,
            "task_split": "test",
            "task_count": 40,
            "trials_per_task": 8,
            "total_episodes": 1_280,
            "base_seed": 42,
        },
        "public factorial design drifted",
    )
    _require(
        protocol.get("execution")
        == {
            "max_steps": 100,
            "max_errors": 10,
            "max_concurrency": 3,
            "policy_temperature": 0.0,
            "max_completion_tokens": 1_024,
            "user_simulator": {
                "model": "deepseek/deepseek-v4-pro",
                "temperature": 0.0,
                "thinking": "disabled",
            },
        },
        "public execution contract drifted",
    )
    _require(
        protocol.get("models")
        == {
            "base": {
                "model": final_serving.BASE_MODEL_ID,
                "revision": final_serving.BASE_MODEL_REVISION,
                "alias": final_serving.BASE_ALIAS,
            },
            "rl": {
                "alias": final_serving.RL_ALIAS,
                "adapter_checkpoint": final_serving.SELECTED_CHECKPOINT,
                "adapter_sha256": final_serving.ADAPTER_SHA256,
            },
        },
        "public model contract drifted",
    )
    serving = protocol.get("serving")
    _require(
        isinstance(serving, dict)
        and set(serving)
        == {
            "dtype",
            "tool_call_parser",
            "max_model_len",
            "chat_template_kwargs",
            "snapshot_tree_sha256",
            "snapshot_file_count",
            "snapshot_total_bytes",
            "tokenizer_chat_template_sha256",
            "art_commit",
            "tau2_commit",
            "runtime_packages",
        },
        "public serving schema drifted",
    )
    _require(
        serving["dtype"] == "bfloat16"
        and serving["tool_call_parser"] == final_serving.TOOL_CALL_PARSER
        and serving["max_model_len"] == final_serving.MAX_MODEL_LEN
        and serving["chat_template_kwargs"] == final_serving.CHAT_TEMPLATE_KWARGS
        and serving["tokenizer_chat_template_sha256"]
        == final_serving.CHAT_TEMPLATE_SHA256
        and serving["art_commit"] == final_serving.ART_COMMIT
        and serving["tau2_commit"] == final_serving.TAU2_COMMIT
        and serving["runtime_packages"] == dict(final_serving.FROZEN_RUNTIME_PACKAGES),
        "public serving contract drifted",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{64}", str(serving["snapshot_tree_sha256"])) is not None
        and isinstance(serving["snapshot_file_count"], int)
        and not isinstance(serving["snapshot_file_count"], bool)
        and serving["snapshot_file_count"] > 0
        and isinstance(serving["snapshot_total_bytes"], int)
        and not isinstance(serving["snapshot_total_bytes"], bool)
        and serving["snapshot_total_bytes"] >= 0,
        "public snapshot provenance is invalid",
    )
    hardware = protocol.get("hardware")
    _require(
        isinstance(hardware, dict)
        and set(hardware)
        == {"gpu_name", "gpu_uuid", "driver_version", "memory_total_mib"}
        and hardware["gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        and isinstance(hardware["gpu_uuid"], str)
        and hardware["gpu_uuid"].startswith("GPU-")
        and isinstance(hardware["driver_version"], str)
        and hardware["driver_version"]
        and isinstance(hardware["memory_total_mib"], int)
        and not isinstance(hardware["memory_total_mib"], bool)
        and hardware["memory_total_mib"] > 90_000,
        "public GPU provenance drifted",
    )
    process = protocol.get("serving_process")
    _require(
        isinstance(process, dict)
        and set(process)
        == {
            "pid",
            "start_time_ticks",
            "boot_id",
            "match_kind",
            "expected_command_sha256",
            "observed_argv_sha256",
        }
        and isinstance(process["pid"], int)
        and not isinstance(process["pid"], bool)
        and process["pid"] > 0
        and isinstance(process["start_time_ticks"], int)
        and not isinstance(process["start_time_ticks"], bool)
        and process["start_time_ticks"] >= 0
        and isinstance(process["boot_id"], str)
        and process["boot_id"]
        and process["match_kind"] in {"direct", "python_console_script"}
        and all(
            re.fullmatch(r"[0-9a-f]{64}", str(process[key])) is not None
            for key in ("expected_command_sha256", "observed_argv_sha256")
        ),
        "public serving-process provenance drifted",
    )
    provenance = protocol.get("provenance")
    _require(
        isinstance(provenance, dict)
        and set(provenance)
        == {
            "evaluation_repo_commit",
            "training_repo_commit",
            "protocol_id",
            "frozen_contract_sha256",
            "native_smoke_manifest_sha256",
            "task_set_sha256",
            "protocol_manifest_sha256",
            "serving_manifest_sha256",
            "results_info_contract_sha256",
        },
        "public provenance schema drifted",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", str(provenance["evaluation_repo_commit"]))
        is not None
        and provenance["training_repo_commit"] == final_serving.TRAINING_REPO_COMMIT
        and provenance["protocol_id"] == "service-agent-final-2x2-r1"
        and all(
            re.fullmatch(r"[0-9a-f]{64}", str(provenance[key])) is not None
            for key in (
                "frozen_contract_sha256",
                "native_smoke_manifest_sha256",
                "task_set_sha256",
                "protocol_manifest_sha256",
                "serving_manifest_sha256",
                "results_info_contract_sha256",
            )
        ),
        "public provenance values are invalid",
    )
    _require(
        outcomes.get("episode_fields")
        == [
            "task_id",
            "trial",
            "seed",
            "reward",
            "success",
            "termination",
            "failure_category",
            "duration_seconds",
            "agent_cost",
            "user_cost",
            "message_count",
            "write_candidates",
            "executed_writes",
            "unauthorized_executed_writes",
            "unauthorized_reasons",
            "duplicate_side_effects",
            "errored_tool_calls",
        ],
        "public outcome field contract drifted",
    )
    _require(
        set(pairing)
        == {
            "schema_version",
            "pairing_unit",
            "bootstrap_unit",
            "task_ids",
            "trials",
            "seeds_by_trial",
            "episodes_per_cell",
            "common_episode_key_sha256",
            "cells",
        },
        "public pairing schema drifted",
    )
    _require(
        set(analysis) == {"schema_version", "statistics", "cells", "contrasts"},
        "public analysis schema drifted",
    )
    _require(
        analysis.get("statistics")
        == {
            "estimand": "macro mean task reward",
            "bootstrap_unit": "task",
            "paired_by": ["task_id", "trial", "seed"],
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": 0,
            "interval": "95% percentile",
            "descriptive_percentiles": "linear interpolation at (n - 1) * p",
        },
        "public statistical contract drifted",
    )
    _require(
        set(analysis.get("contrasts", {}))
        == {
            "harness_effect_base",
            "harness_effect_rl",
            "model_effect_native",
            "model_effect_governed",
            "combined_gain",
            "interaction",
        },
        "public contrast set drifted",
    )
    _require(
        set(live_audit)
        == {
            "schema_version",
            "scope",
            "scope_note",
            "infrastructure_retry_records",
            "cells",
            "decision_count_note",
        },
        "public live-audit schema drifted",
    )
    _require(
        isinstance(live_audit["infrastructure_retry_records"], int)
        and not isinstance(live_audit["infrastructure_retry_records"], bool)
        and live_audit["infrastructure_retry_records"] >= 0,
        "public live-audit infrastructure retry count is invalid",
    )
    design = protocol.get("design")
    _require(isinstance(design, dict), "public protocol design is missing")
    _require(design.get("cell_order") == EXPECTED_CELL_ORDER, "public cell order drifted")
    _require(design.get("task_count") == 40, "public task count drifted")
    _require(design.get("trials_per_task") == 8, "public trial count drifted")
    _require(design.get("total_episodes") == 1_280, "public episode count drifted")
    _require(
        set(outcomes.get("cells", {})) == set(FACTORIAL_CELLS),
        "public outcomes cells drifted",
    )
    _require(
        set(analysis.get("cells", {})) == set(FACTORIAL_CELLS),
        "public analysis cells drifted",
    )
    _require(
        set(live_audit.get("cells", {})) == {"base_h2", "rl_h2"},
        "public live audit cells drifted",
    )
    for cell, summary in live_audit["cells"].items():
        _require(
            set(summary)
            == {
                "sessions_with_audit",
                "decisions",
                "normalizations",
                "rejection_reasons",
                "retry_decision_records",
                "records",
                "tasks_with_audit_records",
            },
            f"{cell} public live-audit summary schema drifted",
        )
        for field in (
            "sessions_with_audit",
            "retry_decision_records",
            "records",
            "tasks_with_audit_records",
        ):
            _require(
                isinstance(summary[field], int)
                and not isinstance(summary[field], bool)
                and summary[field] >= 0,
                f"{cell} public live-audit {field} is invalid",
            )
        _require(
            summary["tasks_with_audit_records"] <= 40,
            f"{cell} public live-audit task coverage exceeds the final grid",
        )
        _require(
            set(summary["normalizations"]) <= {"mixed_text_stripped"},
            f"{cell} public live-audit normalization is unknown",
        )
        _require(
            all(
                re.fullmatch(r"[a-z0-9_]{1,80}", str(reason)) is not None
                for reason in summary["rejection_reasons"]
            ),
            f"{cell} public live-audit rejection reason is invalid",
        )
        _require(
            set(summary["decisions"])
            <= {
                "allow",
                "deny",
                "duplicate",
                "require_confirmation",
                "require_evidence",
                "transfer",
            },
            f"{cell} public live-audit decision is unknown",
        )
        _require(
            all(
                isinstance(count, int)
                and not isinstance(count, bool)
                and count >= 0
                for counts in (
                    summary["decisions"],
                    summary["normalizations"],
                    summary["rejection_reasons"],
                )
                for count in counts.values()
            ),
            f"{cell} public live-audit count is invalid",
        )
        _require(
            summary["records"]
            == sum(summary["decisions"].values())
            + sum(summary["normalizations"].values()),
            f"{cell} public live-audit record total is inconsistent",
        )
        _require(
            sum(summary["rejection_reasons"].values())
            == sum(
                count
                for decision, count in summary["decisions"].items()
                if decision != "allow"
            ),
            f"{cell} public live-audit rejection total is inconsistent",
        )
        _require(
            summary["retry_decision_records"] <= sum(summary["decisions"].values()),
            f"{cell} public live-audit retry-record total is inconsistent",
        )
    required_raw = {
        "final_manifest.json",
        "serving_manifest.json",
        *(
            f"{cell}/{name}"
            for cell in FACTORIAL_CELLS
            for name in ("results.json", "run_config.json", "runner.log")
        ),
    }
    _require(required_raw <= set(raw_checksums), "raw checksum index is incomplete")
    extra_raw = set(raw_checksums) - required_raw
    _require(
        all(
            re.fullmatch(
                r"(?:base_h2|rl_h2)/audit/audit_[0-9a-f]{32}\.jsonl",
                path,
            )
            for path in extra_raw
        ),
        "raw checksum index contains an unexpected path",
    )
    provenance = protocol.get("provenance")
    _require(isinstance(provenance, dict), "public provenance is missing")
    _require(
        provenance.get("protocol_manifest_sha256")
        == raw_checksums["final_manifest.json"],
        "public protocol-manifest digest differs from RAW_SHA256SUMS",
    )
    _require(
        provenance.get("serving_manifest_sha256")
        == raw_checksums["serving_manifest.json"],
        "public serving-manifest digest differs from RAW_SHA256SUMS",
    )

    task_ids = pairing.get("task_ids")
    trials = pairing.get("trials")
    seeds = pairing.get("seeds_by_trial")
    _require(
        isinstance(task_ids, list) and len(task_ids) == 40 and len(set(task_ids)) == 40,
        "public pairing task set is invalid",
    )
    _require(trials == list(range(8)), "public pairing trials drifted")
    from service_agent.eval.factorial import trial_seeds

    expected_seeds = {
        str(trial): seed for trial, seed in trial_seeds(42, 8).items()
    }
    _require(seeds == expected_seeds, "public pairing seed schedule drifted")
    expected_keys = [
        {"task_id": task_id, "trial": trial, "seed": expected_seeds[str(trial)]}
        for task_id in sorted(task_ids)
        for trial in range(8)
    ]
    key_hash = _canonical_sha256(expected_keys)
    _require(
        pairing.get("common_episode_key_sha256") == key_hash,
        "public pairing key digest is wrong",
    )
    _require(
        provenance.get("task_set_sha256") == _canonical_sha256(sorted(task_ids)),
        "public task-set digest differs from pairing task IDs",
    )

    for cell in FACTORIAL_CELLS:
        rows = outcomes["cells"][cell]
        _require(isinstance(rows, list) and len(rows) == 320, f"{cell} outcomes are incomplete")
        actual_keys = [
            {
                "task_id": row.get("task_id"),
                "trial": row.get("trial"),
                "seed": row.get("seed"),
            }
            for row in rows
        ]
        _require(actual_keys == expected_keys, f"{cell} public outcomes are not paired")
        _require(
            pairing["cells"][cell]["episode_key_sha256"] == key_hash,
            f"{cell} pairing digest drifted",
        )
        _require(
            pairing["cells"][cell]["raw_results_sha256"]
            == raw_checksums[f"{cell}/results.json"],
            f"{cell} pairing raw-result digest drifted",
        )
        for row in rows:
            _require(
                set(row) == set(outcomes["episode_fields"]),
                f"{cell} outcome fields drifted",
            )
            reward = _finite_number(row["reward"], f"{cell} public reward")
            _require(0.0 <= reward <= 1.0 + 1e-6, f"{cell} public reward is invalid")
            _require(
                row["success"] is _success(reward),
                f"{cell} public success flag is inconsistent",
            )
            category = row["failure_category"]
            _require(
                (category is None and row["success"])
                or (
                    not row["success"]
                    and isinstance(category, str)
                    and category in TAXONOMY_PRIORITY
                ),
                f"{cell} public failure category is invalid",
            )
            for field in (
                "message_count",
                "write_candidates",
                "executed_writes",
                "unauthorized_executed_writes",
                "duplicate_side_effects",
                "errored_tool_calls",
            ):
                _require(
                    isinstance(row[field], int)
                    and not isinstance(row[field], bool)
                    and row[field] >= 0,
                    f"{cell} public {field} is invalid",
                )
            _require(
                isinstance(row["unauthorized_reasons"], dict)
                and all(
                    re.fullmatch(r"[a-z0-9_]{1,80}", str(reason)) is not None
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                    for reason, count in row["unauthorized_reasons"].items()
                )
                and sum(row["unauthorized_reasons"].values())
                == row["unauthorized_executed_writes"],
                f"{cell} public unauthorized-reason counts are invalid",
            )
            _finite_number(
                row["duration_seconds"],
                f"{cell} public duration",
                nonnegative=True,
            )
            _optional_cost(row["agent_cost"], f"{cell} public agent cost")
            _optional_cost(row["user_cost"], f"{cell} public user cost")

        expected_summary = _compact_cell_analysis(rows)
        _require(
            analysis["cells"][cell] == expected_summary,
            f"{cell} public analysis is not derivable from compact outcomes",
        )

    expected_contrasts = _compact_bootstrap(
        outcomes["cells"],
        resamples=analysis["statistics"]["resamples"],
        seed=analysis["statistics"]["bootstrap_seed"],
    )
    _require(
        analysis["contrasts"] == expected_contrasts,
        "public contrasts are not derivable from compact outcomes",
    )


def _pass_hat_from_rows(rows: list[dict[str, Any]], k: int) -> float:
    by_task: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(bool(row["success"]))
    values = []
    for successes in by_task.values():
        n = len(successes)
        c = sum(successes)
        values.append(math.comb(c, k) / math.comb(n, k) if c >= k else 0.0)
    return sum(values) / len(values)


def _compact_cell_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    violation_rows = [
        row for row in rows if int(row["unauthorized_executed_writes"]) > 0
    ]
    terminations = Counter(str(row["termination"]) for row in rows)
    taxonomy = Counter(
        str(row["failure_category"])
        for row in rows
        if row["failure_category"] is not None
    )
    duration_total = sum(float(row["duration_seconds"]) for row in rows)
    durations = [float(row["duration_seconds"]) for row in rows]
    executed_writes = sum(int(row["executed_writes"]) for row in rows)
    unauthorized_writes = sum(
        int(row["unauthorized_executed_writes"]) for row in rows
    )
    successful_violations = sum(bool(row["success"]) for row in violation_rows)
    failed_violations = sum(not bool(row["success"]) for row in violation_rows)
    writes_in_successes = sum(
        int(row["unauthorized_executed_writes"])
        for row in rows
        if bool(row["success"])
    )
    writes_in_failures = sum(
        int(row["unauthorized_executed_writes"])
        for row in rows
        if not bool(row["success"])
    )
    unauthorized_reasons: Counter = Counter()
    for row in rows:
        unauthorized_reasons.update(row["unauthorized_reasons"])
    _require(
        successful_violations + failed_violations == len(violation_rows),
        "compact violation partition is inconsistent",
    )
    _require(
        writes_in_successes + writes_in_failures == unauthorized_writes,
        "compact unauthorized-write partition is inconsistent",
    )
    _require(sum(taxonomy.values()) == len(rows) - successes, "compact taxonomy misses failures")
    return {
        "episodes": len(rows),
        "mean_reward": sum(float(row["reward"]) for row in rows) / len(rows),
        "successes": successes,
        "failures": len(rows) - successes,
        "pass_hat_k": {
            str(k): _pass_hat_from_rows(rows, k) for k in (1, 2, 4, 8)
        },
        "safety": {
            "write_candidates": sum(int(row["write_candidates"]) for row in rows),
            "executed_writes": executed_writes,
            "unauthorized_executed_writes": unauthorized_writes,
            "episodes_with_unauthorized_writes": len(violation_rows),
            "successful_episodes_with_unauthorized_writes": successful_violations,
            "failed_episodes_with_unauthorized_writes": failed_violations,
            "unauthorized_writes_in_successful_episodes": writes_in_successes,
            "unauthorized_writes_in_failed_episodes": writes_in_failures,
            "episode_violation_rate": _rate(len(violation_rows), len(rows)),
            "unauthorized_executed_write_rate": _rate(
                unauthorized_writes,
                executed_writes,
            ),
            "unauthorized_reasons": dict(sorted(unauthorized_reasons.items())),
            "duplicate_side_effects": sum(
                int(row["duplicate_side_effects"]) for row in rows
            ),
            "errored_tool_calls": sum(int(row["errored_tool_calls"]) for row in rows),
        },
        "termination_reasons": dict(sorted(terminations.items())),
        "failure_taxonomy": _ordered_taxonomy(taxonomy),
        "duration": {
            "total_seconds": duration_total,
            "mean_seconds": duration_total / len(rows),
            "p50_seconds": _percentile(durations, 0.50),
            "p95_seconds": _percentile(durations, 0.95),
        },
        "cost": {
            "agent": _cost_summary(rows, "agent_cost"),
            "user": _cost_summary(rows, "user_cost"),
        },
    }


def _compact_bootstrap(
    cells: Mapping[str, list[dict[str, Any]]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Rebuild strict contrasts from compact outcomes for public-only checks."""
    from tau2.data_model.simulation import (
        AgentInfo,
        Info,
        RewardInfo,
        SimulationRun,
        TerminationReason,
        UserInfo,
    )
    from tau2.environment.environment import EnvironmentInfo

    results: dict[str, Results] = {}
    for cell in FACTORIAL_CELLS:
        rows = cells[cell]
        simulations = [
            SimulationRun(
                id=f"public-{cell}-{index}",
                task_id=row["task_id"],
                timestamp="public",
                start_time="public",
                end_time="public",
                duration=row["duration_seconds"],
                termination_reason=TerminationReason(row["termination"]),
                agent_cost=row["agent_cost"],
                user_cost=row["user_cost"],
                reward_info=RewardInfo(
                    reward=row["reward"],
                    reward_basis=[],
                    reward_breakdown={},
                ),
                messages=[],
                trial=row["trial"],
                seed=row["seed"],
            )
            for index, row in enumerate(rows)
        ]
        results[cell] = Results(
            timestamp="public",
            info=Info(
                git_commit="public",
                num_trials=8,
                max_steps=100,
                max_errors=10,
                user_info=UserInfo(implementation="user_simulator"),
                agent_info=AgentInfo(implementation="public"),
                environment_info=EnvironmentInfo(
                    domain_name="telecom",
                    policy="not published",
                ),
                seed=42,
            ),
            tasks=[],
            simulations=simulations,
        )
    grid = validate_factorial_grid(results)
    contrasts = factorial_bootstrap_contrasts(grid, seed=seed, resamples=resamples)
    return {
        name: {
            "estimate": contrast.estimate,
            "ci_low": contrast.ci_low,
            "ci_high": contrast.ci_high,
            "tasks": contrast.tasks,
            "resamples": contrast.resamples,
            "significant": contrast.significant,
        }
        for name, contrast in contrasts.items()
    }


def verify_raw_checksums(
    *,
    public_root: Path,
    raw_root: Path,
    protocol_manifest: Path,
    serving_manifest: Path,
) -> None:
    expected = _parse_checksums(
        (public_root / "RAW_SHA256SUMS").read_text(),
        "RAW_SHA256SUMS",
    )
    actual = _raw_checksums(
        raw_root.resolve(),
        protocol_manifest.resolve(),
        serving_manifest.resolve(),
    )
    _require(expected == actual, "raw input checksum index differs from the backup")


def check(
    *,
    public_root: Path = DEFAULT_PUBLIC_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
    raw_root: Path | None = None,
    protocol_manifest: Path | None = None,
    serving_manifest: Path | None = None,
) -> None:
    """Verify the public package and rebuild the report byte-for-byte."""
    public_root = public_root.resolve()
    report_path = report_path.resolve()
    expected_names = {*PUBLIC_RESULT_FILES, "CHECKSUMS.sha256"}
    _require(public_root.is_dir() and not public_root.is_symlink(), "public root is invalid")
    entries = list(public_root.iterdir())
    actual_names = {path.name for path in entries}
    _require(actual_names == expected_names, "public result directory has missing or extra files")
    _require(
        all(path.is_file() and not path.is_symlink() for path in entries),
        "public result directory contains a non-regular file",
    )
    _require(
        report_path.is_file() and not report_path.is_symlink(),
        "factorial report is missing or not a regular file",
    )

    checksum_path = public_root / "CHECKSUMS.sha256"
    expected_checksums = _parse_checksums(checksum_path.read_text(), "CHECKSUMS.sha256")
    _require(
        set(expected_checksums) == {*PUBLIC_RESULT_FILES, CHECKSUM_REPORT_KEY},
        "public checksum index has missing or extra paths",
    )
    actual_checksums = _public_checksums(public_root, report_path)
    _require(expected_checksums == actual_checksums, "public artifact checksum mismatch")

    loaded = {
        name: _read_json(public_root / name, name)
        for name in PUBLIC_JSON_FILES
    }
    raw_checksum_text = (public_root / "RAW_SHA256SUMS").read_text()
    _assert_public_text_safe(raw_checksum_text, "RAW_SHA256SUMS")
    raw_checksums = _parse_checksums(raw_checksum_text, "RAW_SHA256SUMS")
    _assert_public_text_safe(
        (public_root / "CHECKSUMS.sha256").read_text(),
        "CHECKSUMS.sha256",
    )
    _validate_public_structure(
        loaded["protocol.json"],
        loaded["outcomes.json"],
        loaded["pairing_manifest.json"],
        loaded["analysis.json"],
        loaded["live_audit.json"],
        raw_checksums,
    )
    rebuilt = render_report(
        loaded["protocol.json"],
        loaded["pairing_manifest.json"],
        loaded["analysis.json"],
        loaded["live_audit.json"],
    )
    _assert_public_text_safe(rebuilt, "rebuilt factorial report")
    _require(report_path.read_text() == rebuilt, "factorial report is stale")

    raw_args = (raw_root, protocol_manifest, serving_manifest)
    _require(
        all(value is None for value in raw_args)
        or all(value is not None for value in raw_args),
        "raw verification needs raw root and both manifests together",
    )
    if raw_root is not None:
        assert protocol_manifest is not None and serving_manifest is not None
        verify_raw_checksums(
            public_root=public_root,
            raw_root=raw_root,
            protocol_manifest=protocol_manifest,
            serving_manifest=serving_manifest,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser(
        "generate",
        help="validate the private raw backup and generate public artifacts",
    )
    generate_parser.add_argument("--raw-root", type=Path, required=True)
    generate_parser.add_argument("--protocol-manifest", type=Path, required=True)
    generate_parser.add_argument("--serving-manifest", type=Path, required=True)
    generate_parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    generate_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)

    check_parser = subparsers.add_parser(
        "check",
        help="verify and byte-rebuild the public artifacts",
    )
    check_parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    check_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    check_parser.add_argument("--raw-root", type=Path)
    check_parser.add_argument("--protocol-manifest", type=Path)
    check_parser.add_argument("--serving-manifest", type=Path)
    return parser


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--check":
        parser = argparse.ArgumentParser(description="Check the public final report")
        parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
        parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
        parser.add_argument("--raw-root", type=Path)
        parser.add_argument("--protocol-manifest", type=Path)
        parser.add_argument("--serving-manifest", type=Path)
        args = parser.parse_args(sys.argv[2:])
        check(
            public_root=args.public_root,
            report_path=args.report,
            raw_root=args.raw_root,
            protocol_manifest=args.protocol_manifest,
            serving_manifest=args.serving_manifest,
        )
        return
    args = _parser().parse_args()
    if args.command == "generate":
        generate(
            raw_root=args.raw_root,
            protocol_manifest=args.protocol_manifest,
            serving_manifest=args.serving_manifest,
            public_root=args.public_root,
            report_path=args.report,
        )
    else:
        check(
            public_root=args.public_root,
            report_path=args.report,
            raw_root=args.raw_root,
            protocol_manifest=args.protocol_manifest,
            serving_manifest=args.serving_manifest,
        )


if __name__ == "__main__":
    main()
