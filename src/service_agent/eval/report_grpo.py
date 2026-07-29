"""Generate the official GPU-training report from committed manifests.

The formal process ended with a protocol stop rather than a completed
60-position schedule.  Reporting that distinction from prose alone is too easy
to get wrong, especially because ART advances a checkpoint even when no reward
group has variance.  This module validates the three phase manifests, rebuilds
all cumulative progress from the per-position records, and only then renders
``reports/grpo_training.md``.

Run from the repository root:

    uv run python -m service_agent.eval.report_grpo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[3]
GPU_RESULTS = REPO / "results/gpu"
PREFLIGHT_PATH = GPU_RESULTS / "preflight-qwen3coder-r1/preflight_manifest.json"
SMOKE_PATH = GPU_RESULTS / "smoke-qwen3coder-r1/smoke_manifest.json"
FORMAL_PATH = GPU_RESULTS / "grpo-4b-qwen3coder-r1/train_manifest.json"
CHECKSUMS_PATH = GPU_RESULTS / "CHECKSUMS.sha256"
BACKUP_CHECKSUMS_PATH = GPU_RESULTS / "BACKUP_SHA256SUMS"
COUNTER_AUDIT_PATH = GPU_RESULTS / "WANDB_COUNTER_AUDIT.json"
RESTORE_MANIFEST_PATH = GPU_RESULTS / "restore-cp0015-r1/restore_manifest.json"

EXPECTED_MODEL = "Qwen/Qwen3.5-4B"
EXPECTED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_SNAPSHOT = (
    "/root/autodl-tmp/cache/huggingface/models--Qwen--Qwen3.5-4B/"
    f"snapshots/{EXPECTED_REVISION}"
)
EXPECTED_REPO = "e557bbffdee8e283f3a522e6a088ca74bf3ff907"
EXPECTED_ART = "828b839b1139ac780725f0a22a9bde70a82b4878"
EXPECTED_TAU2 = "2822d9030b621e6f13a190fb14fa08cf1c9c4ca4"
EXPECTED_PARSER = "qwen3_coder"
EXPECTED_SEMANTIC_CONTRACT = "91fa4cb5c06414976cf029003ad621b36becfe154ee86201c726f331ec9d6fb6"
EXPECTED_SELECTED_STEP = 15
EXPECTED_SELECTED_REWARD = 0.925
EXPECTED_FINAL_STEP = 24
EXPECTED_TRAINING = {
    "group_size": 4,
    "groups_per_step": 2,
    "kl_penalty_coef": 0.0,
    "learning_rate": 5e-6,
    "logprob_calculation_chunk_size": 512,
    "loss_fn": "ppo",
    "max_turns": 30,
    "steps": 60,
    "val_every": 5,
    "val_trials": 2,
}
EXPECTED_RUNTIME = {
    "base_model": EXPECTED_MODEL,
    "gpu_memory_utilization": 0.68,
    "lora_alpha": 32,
    "lora_rank": 16,
    "max_completion_tokens": 1024,
    "max_model_len": 16384,
    "project": "service-agent",
    "rollout_concurrency": 4,
    "seed": 42,
}
EXPECTED_USER_SIMULATOR = {
    "model": "deepseek/deepseek-v4-pro",
    "temperature": 0.0,
    "thinking": "disabled",
}
EXPECTED_RUN_NAMES = {
    "preflight": "preflight-qwen3coder-r1",
    "smoke": "smoke-qwen3coder-r1",
    "train": "grpo-4b-qwen3coder-r1",
}
SPARSE_REASON = "at most one mixed-reward group in the last 10 checkpoint/rollout steps"
PRIVATE_FIELDS = {
    "access_token",
    "api_key",
    "authorization",
    "cookie",
    "environment",
    "initial_state",
    "messages",
    "password",
    "private_key",
    "secret",
    "token",
    "user_scenario",
    "evaluation_criteria",
    "credentials",
}
PRIVATE_VALUE_PATTERNS = (
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
SELECTED_ADAPTER_PATH = (
    "art/grpo-4b-qwen3coder-r1/service-agent/models/grpo-4b-qwen3coder-r1/"
    "checkpoints/0015/adapter_model.safetensors"
)
LATEST_ADAPTER_PATH = (
    "art/grpo-4b-qwen3coder-r1/service-agent/models/grpo-4b-qwen3coder-r1/"
    "checkpoints/0024/adapter_model.safetensors"
)
FORMAL_HISTORY_PATH = (
    "art/grpo-4b-qwen3coder-r1/service-agent/models/grpo-4b-qwen3coder-r1/"
    "history.jsonl"
)
FORMAL_STATE_PATH = (
    "art/grpo-4b-qwen3coder-r1/service-agent/models/grpo-4b-qwen3coder-r1/"
    "state.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.relative_to(REPO)} must contain a JSON object")
    return payload


def _read_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, separator, file_path = line.partition("  ")
        _require(
            bool(separator) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"malformed checksum line in {path.relative_to(REPO)}",
        )
        _require(file_path not in entries, f"duplicate checksum path {file_path}")
        _require(not Path(file_path).is_absolute(), f"absolute checksum path {file_path}")
        _require(".." not in Path(file_path).parts, f"escaping checksum path {file_path}")
        entries[file_path] = digest
    return entries


def validate_artifact_checksums() -> dict[str, str]:
    """Verify committed evidence and the index of the ignored 3.3 GB backup."""
    committed = _read_checksums(CHECKSUMS_PATH)
    files = sorted(
        path
        for path in GPU_RESULTS.rglob("*")
        if path.is_file() and path != CHECKSUMS_PATH
    )
    expected_paths = {str(path.relative_to(REPO)) for path in files}
    _require(set(committed) == expected_paths, "committed GPU checksum file is stale")
    for path in files:
        relative = str(path.relative_to(REPO))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(digest == committed[relative], f"checksum mismatch for {relative}")

    backup = _read_checksums(BACKUP_CHECKSUMS_PATH)
    _require(len(backup) == 292, "backup index does not contain 292 source files")
    manifest_pairs = (
        ("runs/preflight-qwen3coder-r1/preflight_manifest.json", PREFLIGHT_PATH),
        ("runs/smoke-qwen3coder-r1/smoke_manifest.json", SMOKE_PATH),
        ("runs/grpo-4b-qwen3coder-r1/train_manifest.json", FORMAL_PATH),
    )
    for backup_path, committed_path in manifest_pairs:
        digest = hashlib.sha256(committed_path.read_bytes()).hexdigest()
        _require(backup.get(backup_path) == digest, f"backup mismatch for {backup_path}")
    _require(SELECTED_ADAPTER_PATH in backup, "selected adapter is absent from backup index")
    _require(LATEST_ADAPTER_PATH in backup, "latest adapter is absent from backup index")
    return backup


def validate_backup_files(backup_root: Path) -> None:
    """Read and hash every indexed source file in an available local backup."""
    backup = _read_checksums(BACKUP_CHECKSUMS_PATH)
    root = backup_root.resolve()
    _require(root.is_dir(), f"backup root does not exist: {root}")
    for relative, expected in backup.items():
        path = root / relative
        _require(path.is_file(), f"indexed backup file is missing: {relative}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        _require(actual == expected, f"backup checksum mismatch for {relative}")


def load_counter_audit() -> dict[str, Any]:
    """Load the committed aggregate that explains ART's cumulative W&B counters."""
    return _read_json(COUNTER_AUDIT_PATH)


def load_restore_manifest() -> dict[str, Any]:
    """Load the post-backup selected-adapter recovery proof."""
    return _read_json(RESTORE_MANIFEST_PATH)


def load_manifests() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the byte-identical manifest copies committed under ``results/gpu``."""
    return _read_json(PREFLIGHT_PATH), _read_json(SMOKE_PATH), _read_json(FORMAL_PATH)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"invalid committed GPU artifacts: {message}")


def _nonnegative_int(value: Any, context: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{context} must be a nonnegative integer",
    )
    return value


def _integral_metric(value: Any, context: str) -> int:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value).is_integer()
        and float(value) >= 0,
        f"{context} must be a finite nonnegative integer-valued metric",
    )
    return int(value)


def _bounded_reward(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0,
        f"{context} must be finite and within [0, 1]",
    )
    return float(value)


def _finite_number(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{context} must be a finite number",
    )
    return float(value)


def _typed_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _typed_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _typed_json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _validate_wandb_url(manifest: dict[str, Any], phase: str) -> None:
    value = manifest.get("wandb_url")
    _require(isinstance(value, str), f"{phase} W&B URL is missing")
    parsed = urlsplit(value)
    _require(
        parsed.scheme == "https"
        and parsed.hostname == "wandb.ai"
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment,
        f"{phase} W&B URL is not a clean wandb.ai HTTPS URL",
    )
    _require(
        parsed.path
        == f"/lqj-physics-nudt/service-agent/runs/{EXPECTED_RUN_NAMES[phase]}",
        f"{phase} W&B URL path drifted",
    )


def validate_counter_audit(
    audit: dict[str, Any],
    backup: dict[str, str],
    formal: dict[str, Any],
) -> None:
    """Bind the W&B double-count disclosure to raw-file hashes and the manifest."""
    _validate_public_payload(audit)
    _require(audit.get("schema_version") == 1, "W&B counter audit schema drifted")
    sources = audit.get("sources", {})
    expected_sources = {
        "history": FORMAL_HISTORY_PATH,
        "state": FORMAL_STATE_PATH,
    }
    for label, expected_path in expected_sources.items():
        source = sources.get(label, {})
        _require(source.get("path") == expected_path, f"W&B {label} source path drifted")
        _require(
            source.get("sha256") == backup.get(expected_path),
            f"W&B {label} source hash disagrees with backup index",
        )

    progress = formal["progress"]
    manifest_totals = {
        "gradient_steps": progress["gradient_steps"],
        "groups_submitted": progress["groups_submitted"],
        "groups_trainable": progress["trainable_groups"],
    }
    _require(
        _typed_json_equal(audit.get("manifest_totals"), manifest_totals),
        "W&B audit manifest totals disagree",
    )
    backend = audit.get("backend_record_totals")
    rollout = audit.get("rollout_record_totals")
    cumulative = audit.get("art_cumulative_state")
    _require(
        _typed_json_equal(backend, manifest_totals),
        "W&B backend totals disagree with manifest",
    )
    _require(
        _typed_json_equal(
            rollout,
            {
                "gradient_steps": 0,
                "groups_submitted": manifest_totals["groups_submitted"],
                "groups_trainable": manifest_totals["groups_trainable"],
            },
        ),
        "W&B rollout-record totals disagree",
    )
    _require(
        _typed_json_equal(
            cumulative,
            {
                "gradient_steps": backend["gradient_steps"],
                "groups_submitted": backend["groups_submitted"] + rollout["groups_submitted"],
                "groups_trainable": backend["groups_trainable"] + rollout["groups_trainable"],
            },
        ),
        "W&B cumulative-state arithmetic disagrees",
    )
    counts = audit.get("record_counts")
    expected_counts = {
        "backend_train": len(formal["train_steps"]),
        "dev": len(formal["dev_evaluations"]),
        "rollout": len(formal["train_steps"]),
        "total": 2 * len(formal["train_steps"]) + len(formal["dev_evaluations"]),
    }
    _require(
        _typed_json_equal(counts, expected_counts),
        "W&B history record counts disagree",
    )
    expected_sequences = {
        "backend_train": [step["checkpoint_step"] for step in formal["train_steps"]],
        "dev": [item["step"] for item in formal["dev_evaluations"]],
        "rollout": [step["checkpoint_step"] for step in formal["train_steps"]],
    }
    _require(
        _typed_json_equal(audit.get("record_sequences"), expected_sequences),
        "W&B history record sequences disagree",
    )
    expected_backend_counters = {
        "gradient_steps": [step["gradient_steps"] for step in formal["train_steps"]],
        "groups_submitted": [step["groups_submitted"] for step in formal["train_steps"]],
        "groups_trainable": [step["trainable_groups"] for step in formal["train_steps"]],
    }
    expected_rollout_counters = {
        "groups_submitted": [step["groups_submitted"] for step in formal["train_steps"]],
        "groups_trainable": [step["trainable_groups"] for step in formal["train_steps"]],
    }
    _require(
        _typed_json_equal(
            audit.get("backend_step_counters"),
            expected_backend_counters,
        ),
        "W&B backend per-step counters disagree",
    )
    _require(
        _typed_json_equal(
            audit.get("rollout_step_counters"),
            expected_rollout_counters,
        ),
        "W&B rollout per-step counters disagree",
    )
    _require(
        _typed_json_equal(
            audit.get("dev_record_rewards"),
            [item["avg_reward"] for item in formal["dev_evaluations"]],
        ),
        "W&B dev-record rewards disagree",
    )


def _history_counter_aggregate(history_path: Path, state_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in history_path.read_text().splitlines()
        if line.strip()
    ]
    rollout_rows: list[dict[str, Any]] = []
    backend_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for row in rows:
        predicates = (
            "train/reward" in row and "data/step_num_gradient_steps" not in row,
            "data/step_num_gradient_steps" in row,
            "dev/dev/reward" in row,
        )
        _require(
            sum(predicates) == 1,
            "W&B history record must belong to exactly one record class",
        )
        if predicates[0]:
            rollout_rows.append(row)
        elif predicates[1]:
            backend_rows.append(row)
        else:
            dev_rows.append(row)
    for label, records in (
        ("rollout", rollout_rows),
        ("backend", backend_rows),
        ("dev", dev_rows),
    ):
        for row in records:
            step = _integral_metric(row.get("step"), f"W&B {label} record step")
            training_step = _integral_metric(
                row.get("training_step"),
                f"W&B {label} record training_step",
            )
            _require(step == training_step, f"W&B {label} step fields disagree")

    def totals(records: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "gradient_steps": sum(
                _integral_metric(
                    row.get("data/step_num_gradient_steps", 0),
                    "W&B history gradient_steps",
                )
                for row in records
            ),
            "groups_submitted": sum(
                _integral_metric(
                    row.get("data/step_num_groups_submitted", 0),
                    "W&B history groups_submitted",
                )
                for row in records
            ),
            "groups_trainable": sum(
                _integral_metric(
                    row.get("data/step_num_groups_trainable", 0),
                    "W&B history groups_trainable",
                )
                for row in records
            ),
        }

    state = _read_json(state_path)
    cumulative = state.get("_metrics_builder_state", {}).get("cum_state", {})
    return {
        "art_cumulative_state": {
            "gradient_steps": _integral_metric(
                cumulative.get("data/cum/num_gradient_steps"),
                "W&B state gradient_steps",
            ),
            "groups_submitted": _integral_metric(
                cumulative.get("data/cum/num_groups_submitted"),
                "W&B state groups_submitted",
            ),
            "groups_trainable": _integral_metric(
                cumulative.get("data/cum/num_groups_trainable"),
                "W&B state groups_trainable",
            ),
        },
        "backend_record_totals": totals(backend_rows),
        "backend_step_counters": {
            "gradient_steps": [
                _integral_metric(
                    row.get("data/step_num_gradient_steps"),
                    "W&B backend per-step gradient_steps",
                )
                for row in backend_rows
            ],
            "groups_submitted": [
                _integral_metric(
                    row.get("data/step_num_groups_submitted"),
                    "W&B backend per-step groups_submitted",
                )
                for row in backend_rows
            ],
            "groups_trainable": [
                _integral_metric(
                    row.get("data/step_num_groups_trainable"),
                    "W&B backend per-step groups_trainable",
                )
                for row in backend_rows
            ],
        },
        "dev_record_rewards": [
            _bounded_reward(row.get("dev/dev/reward"), "W&B dev-record reward")
            for row in dev_rows
        ],
        "record_counts": {
            "backend_train": len(backend_rows),
            "dev": len(dev_rows),
            "rollout": len(rollout_rows),
            "total": len(rows),
        },
        "record_sequences": {
            "backend_train": [
                _integral_metric(row.get("training_step"), "W&B backend sequence")
                for row in backend_rows
            ],
            "dev": [
                _integral_metric(row.get("training_step"), "W&B dev sequence")
                for row in dev_rows
            ],
            "rollout": [
                _integral_metric(row.get("training_step"), "W&B rollout sequence")
                for row in rollout_rows
            ],
        },
        "rollout_step_counters": {
            "groups_submitted": [
                _integral_metric(
                    row.get("data/step_num_groups_submitted"),
                    "W&B rollout per-step groups_submitted",
                )
                for row in rollout_rows
            ],
            "groups_trainable": [
                _integral_metric(
                    row.get("data/step_num_groups_trainable"),
                    "W&B rollout per-step groups_trainable",
                )
                for row in rollout_rows
            ],
        },
        "rollout_record_totals": totals(rollout_rows),
    }


def validate_counter_audit_sources(backup_root: Path, audit: dict[str, Any]) -> None:
    """Recompute the committed W&B audit when the ignored raw backup is present."""
    root = backup_root.resolve()
    actual = _history_counter_aggregate(
        root / FORMAL_HISTORY_PATH,
        root / FORMAL_STATE_PATH,
    )
    for field, value in actual.items():
        _require(
            _typed_json_equal(audit.get(field), value),
            f"W&B raw-source aggregate {field} disagrees",
        )


def validate_restore_manifest(restore: dict[str, Any], backup: dict[str, str]) -> None:
    """Require evidence that the selected adapter loads from an explicit pinned base."""
    _validate_public_payload(restore)
    expected = {
        "adapter_checkpoint": EXPECTED_SELECTED_STEP,
        "adapter_relocated": True,
        "adapter_sha256": backup[SELECTED_ADAPTER_PATH],
        "base_model": EXPECTED_MODEL,
        "base_model_revision": EXPECTED_REVISION,
        "base_snapshot_explicit": True,
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "loaded_adapter_names": ["default"],
        "model_class": "Qwen3_5ForCausalLM",
        "new_tokens_generated": 1,
        "schema_version": 1,
        "status": "passed",
        "test_split_accessed": False,
    }
    _require(
        set(restore) == set(expected) | {"checked_at"},
        "restore proof schema contains missing or extra fields",
    )
    for field, value in expected.items():
        _require(
            _typed_json_equal(restore.get(field), value),
            f"restore proof {field} disagrees",
        )
    checked_at = restore.get("checked_at")
    parsed: datetime | None = None
    if isinstance(checked_at, str):
        try:
            parsed = datetime.fromisoformat(checked_at)
        except ValueError:
            pass
    _require(
        parsed is not None
        and parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        "restore proof timestamp is not a valid UTC time",
    )


def _progress(steps: list[dict[str, Any]]) -> dict[str, int]:
    trainable = sum(bool(step["gradient_work_performed"]) for step in steps)
    return {
        "checkpoint_steps_completed": len(steps),
        "final_checkpoint_step": steps[-1]["checkpoint_step"] if steps else 0,
        "gradient_steps": sum(int(step["gradient_steps"]) for step in steps),
        "groups_submitted": sum(int(step["groups_submitted"]) for step in steps),
        "skipped_checkpoint_steps": len(steps) - trainable,
        "trainable_checkpoint_steps": trainable,
        "trainable_groups": sum(int(step["trainable_groups"]) for step in steps),
    }


def _without_run_name(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("run_name", None)
    return value


def _validate_public_payload(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            private = normalized in PRIVATE_FIELDS or any(
                normalized.endswith(f"_{field}") for field in PRIVATE_FIELDS
            )
            _require(not private, f"private field at {path}.{key}")
            _validate_public_payload(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_public_payload(child, f"{path}[{index}]")
    elif isinstance(value, str):
        for pattern in PRIVATE_VALUE_PATTERNS:
            _require(pattern.search(value) is None, f"credential-shaped value at {path}")


def _validate_group_stats(
    stats: dict[str, Any],
    *,
    expected_groups: int,
    group_size: int,
    context: str,
) -> None:
    required = {
        "all_one",
        "all_zero",
        "constant_other",
        "groups",
        "mixed",
        "reward_mean",
        "rollouts",
    }
    _require(required <= set(stats), f"{context} group stats are incomplete")
    _require(expected_groups > 0, f"{context} expected group count must be positive")
    _require(group_size > 0, f"{context} group size must be positive")
    counts = {
        field: _nonnegative_int(stats[field], f"{context} {field}")
        for field in ("all_one", "all_zero", "constant_other", "groups", "mixed", "rollouts")
    }
    _bounded_reward(stats["reward_mean"], f"{context} reward_mean")
    _require(counts["groups"] == expected_groups, f"{context} group count disagrees")
    _require(
        counts["rollouts"] == expected_groups * group_size,
        f"{context} rollout count disagrees",
    )
    partition = sum(counts[field] for field in ("all_one", "all_zero", "constant_other", "mixed"))
    _require(partition == expected_groups, f"{context} group partition disagrees")


def _validate_art_metrics(
    metrics: dict[str, Any],
    *,
    groups: int,
    trainable_groups: int,
    gradient_steps: int,
    rollouts: int,
    context: str,
) -> None:
    expected = {
        "data/step_num_gradient_steps": gradient_steps,
        "data/step_num_groups_submitted": groups,
        "data/step_num_groups_trainable": trainable_groups,
        "data/step_num_scenarios": groups,
        "data/step_num_trajectories": rollouts,
    }
    for field, value in expected.items():
        actual = _integral_metric(metrics.get(field), f"{context} metric {field}")
        _require(actual == value, f"{context} metric {field} disagrees")


def validate_manifests(
    preflight: dict[str, Any],
    smoke: dict[str, Any],
    formal: dict[str, Any],
) -> None:
    """Fail if the committed evidence does not support the rendered claims."""
    phases = ((preflight, "preflight"), (smoke, "smoke"), (formal, "train"))
    for manifest, phase in phases:
        _validate_public_payload(manifest)
        _require(manifest.get("schema_version") == 3, f"{phase} schema is not 3")
        _require(manifest.get("phase") == phase, f"expected phase {phase}")
        _require(
            manifest.get("run_name") == EXPECTED_RUN_NAMES[phase],
            f"{phase} run name drift",
        )
        _validate_wandb_url(manifest, phase)
        _require(manifest.get("base_model") == EXPECTED_MODEL, f"{phase} model drift")
        _require(
            manifest.get("base_model_revision") == EXPECTED_REVISION,
            f"{phase} model revision drift",
        )
        _require(manifest.get("art_commit") == EXPECTED_ART, f"{phase} ART drift")
        _require(manifest.get("tau2_commit") == EXPECTED_TAU2, f"{phase} tau2 drift")
        _require(manifest.get("repo_commit") == EXPECTED_REPO, f"{phase} repository drift")
        _require(manifest.get("tool_call_parser") == EXPECTED_PARSER, f"{phase} parser drift")
        _require(
            manifest.get("semantic_contract_sha256") == EXPECTED_SEMANTIC_CONTRACT,
            f"{phase} semantic contract drift",
        )
        _require(manifest.get("test_split_locked") is True, f"{phase} test split was unlocked")
        _require(
            _typed_json_equal(manifest.get("training"), EXPECTED_TRAINING),
            f"{phase} training contract drift",
        )
        runtime = manifest.get("runtime", {})
        _require(
            _typed_json_equal(_without_run_name(runtime), EXPECTED_RUNTIME),
            f"{phase} runtime contract drift",
        )
        _require(
            runtime.get("run_name") == EXPECTED_RUN_NAMES[phase],
            f"{phase} runtime run name drift",
        )
        _require(
            _typed_json_equal(manifest.get("user_simulator"), EXPECTED_USER_SIMULATOR),
            f"{phase} user-simulator contract drift",
        )
        system = manifest.get("system", {})
        _require(system.get("cuda_available") is True, f"{phase} CUDA was unavailable")
        _require(system.get("bf16_supported") is True, f"{phase} bf16 was unavailable")
        _require(
            system.get("gpu") == "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            f"{phase} GPU drift",
        )
        snapshot = manifest.get("model_snapshot")
        _require(
            snapshot == EXPECTED_SNAPSHOT,
            f"{phase} model snapshot path drift",
        )

    shared_fields = (
        "model_snapshot",
        "repo_commit",
        "semantic_contract_sha256",
        "semantic_input_hashes",
        "token_budget",
        "training",
        "user_simulator",
        "system",
    )
    for field in shared_fields:
        _require(
            preflight.get(field) == smoke.get(field) == formal.get(field),
            f"{field} differs across phases",
        )
    _require(
        _without_run_name(preflight["runtime"])
        == _without_run_name(smoke["runtime"])
        == _without_run_name(formal["runtime"]),
        "runtime differs across phases beyond run_name",
    )

    _require(preflight.get("status") == "passed", "preflight did not pass")
    _require(
        _nonnegative_int(preflight.get("initial_step"), "preflight initial_step") == 0,
        "preflight did not start at step 0",
    )
    _require(
        _nonnegative_int(preflight.get("final_step"), "preflight final_step") == 0,
        "preflight performed an update",
    )
    logprob = preflight.get("logprob_gate", {})
    _require(logprob.get("status") == "passed", "logprob gate did not pass")
    _require(logprob.get("prompt_token_ids_exact") is True, "prompt token IDs differ")
    thresholds = logprob.get("thresholds", {})
    _require(
        _typed_json_equal(
            thresholds,
            {
            "abs_ratio_mean_minus_one": 0.02,
            "clip_fraction": 0.02,
            "clip_window": [0.8, 1.2],
            },
        ),
        "logprob thresholds drifted",
    )
    numeric_logprob = (
        "clip_fraction_before_first_update",
        "mean_abs_logprob_delta",
        "ratio_mean",
        "ratio_median",
        "ratio_p95",
        "ratio_p99",
    )
    logprob_values = {
        field: _finite_number(logprob.get(field), f"logprob {field}")
        for field in numeric_logprob
    }
    _require(
        _nonnegative_int(logprob.get("prompts"), "logprob prompts") > 0,
        "logprob gate has no prompts",
    )
    _require(
        _nonnegative_int(logprob.get("tokens"), "logprob tokens") > 0,
        "logprob gate has no completion tokens",
    )
    _require(
        all(
            logprob_values[field] > 0
            for field in ("ratio_mean", "ratio_median", "ratio_p95", "ratio_p99")
        ),
        "logprob importance ratios must be positive",
    )
    _require(
        logprob_values["mean_abs_logprob_delta"] >= 0,
        "mean absolute logprob delta is negative",
    )
    _require(
        abs(logprob_values["ratio_mean"] - 1.0)
        <= float(thresholds["abs_ratio_mean_minus_one"]),
        "mean importance ratio exceeds its threshold",
    )
    clip_fraction = logprob_values["clip_fraction_before_first_update"]
    _require(0.0 <= clip_fraction <= 1.0, "clip fraction is outside [0, 1]")
    _require(
        clip_fraction <= float(thresholds["clip_fraction"]),
        "clip fraction exceeds its threshold",
    )
    rollout_only = preflight.get("rollout_only", {})
    _require(rollout_only.get("status") == "passed", "rollout-only gate did not pass")
    _require(rollout_only.get("strict_replay") is True, "preflight strict replay failed")
    _require(
        rollout_only.get("reward_finalized_once") is True,
        "preflight reward was not finalized exactly once",
    )
    preflight_stats = rollout_only.get("stats", {})
    _validate_group_stats(
        preflight_stats,
        expected_groups=EXPECTED_TRAINING["groups_per_step"],
        group_size=int(preflight["training"]["group_size"]),
        context="preflight rollout",
    )

    _require(smoke.get("status") == "passed", "smoke did not pass")
    _require(
        _nonnegative_int(smoke.get("initial_step"), "smoke initial_step") == 0
        and _nonnegative_int(smoke.get("final_step"), "smoke final_step") == 1,
        "smoke step drift",
    )
    _require(smoke.get("strict_replay") is True, "smoke strict replay failed")
    _require(smoke.get("gradient_work_performed") is True, "smoke did no gradient work")
    smoke_groups = _nonnegative_int(smoke.get("trainable_groups"), "smoke trainable_groups")
    smoke_gradients = _nonnegative_int(smoke.get("gradient_steps"), "smoke gradient_steps")
    _require(smoke_groups > 0, "smoke had no trainable group")
    _require(smoke_gradients > 0, "smoke had no gradient steps")
    sampling = smoke.get("sampling", {})
    _require(
        sampling.get("strategy") == "contiguous_formal_prefix",
        "smoke was not a contiguous formal prefix",
    )
    _require(
        _typed_json_equal(sampling.get("formal_checkpoint_steps"), [0, 1]),
        "smoke checkpoint prefix drift",
    )
    smoke_formal_steps = len(sampling["formal_checkpoint_steps"])
    expected_smoke_groups = smoke_formal_steps * EXPECTED_TRAINING["groups_per_step"]
    _require(
        _typed_json_equal(
            sampling.get("groups_per_formal_checkpoint_step"),
            EXPECTED_TRAINING["groups_per_step"],
        ),
        "smoke groups-per-formal-step drift",
    )
    _require(
        _typed_json_equal(sampling.get("groups_submitted"), expected_smoke_groups),
        "smoke submitted-group contract drift",
    )
    _require(
        _typed_json_equal(
            sampling.get("formal_slots"),
            list(range(expected_smoke_groups)),
        ),
        "smoke slot prefix drift",
    )
    _require(
        _typed_json_equal(sampling.get("group_size"), EXPECTED_TRAINING["group_size"]),
        "smoke group-size contract drift",
    )
    _require(
        _typed_json_equal(sampling.get("policy_seed_base"), EXPECTED_RUNTIME["seed"]),
        "smoke seed base drift",
    )
    expected_smoke_rollouts = expected_smoke_groups * EXPECTED_TRAINING["group_size"]
    _require(
        _typed_json_equal(
            sampling.get("policy_seeds"),
            list(
                range(
                    EXPECTED_RUNTIME["seed"],
                    EXPECTED_RUNTIME["seed"] + expected_smoke_rollouts,
                )
            ),
        ),
        "smoke seed mapping drift",
    )
    _require(
        isinstance(smoke.get("scenario_ids"), list)
        and len(smoke["scenario_ids"]) == expected_smoke_groups,
        "smoke scenario count disagrees",
    )
    smoke_stats = smoke.get("stats", {})
    _validate_group_stats(
        smoke_stats,
        expected_groups=expected_smoke_groups,
        group_size=EXPECTED_TRAINING["group_size"],
        context="smoke",
    )
    _require(smoke_groups == smoke_stats["mixed"], "smoke trainable groups disagree")
    smoke_metrics = smoke.get("metrics", {})
    _validate_art_metrics(
        smoke_metrics,
        groups=expected_smoke_groups,
        trainable_groups=smoke_groups,
        gradient_steps=smoke_gradients,
        rollouts=int(smoke_stats["rollouts"]),
        context="smoke",
    )
    _require(
        smoke.get("checkpoint_path")
        == f"{str(smoke['lineage_path']).rstrip('/')}/checkpoints/0001",
        "smoke checkpoint path disagrees",
    )

    _require(formal.get("status") == "stopped_sparse_reward", "unexpected formal status")
    _require(formal.get("reason") == SPARSE_REASON, "unexpected formal stop reason")
    steps = formal.get("train_steps")
    _require(isinstance(steps, list) and bool(steps), "formal train_steps missing")
    _require(len(steps) == EXPECTED_FINAL_STEP, "unexpected formal terminal step")
    for index, step in enumerate(steps, start=1):
        _require(
            _nonnegative_int(
                step.get("checkpoint_step"),
                f"checkpoint {index} checkpoint_step",
            )
            == index,
            f"checkpoint sequence breaks at {index}",
        )
        _require(
            _nonnegative_int(step.get("rollout_step"), f"checkpoint {index} rollout_step")
            == index - 1,
            f"rollout sequence breaks at {index}",
        )
        gradient_steps = _nonnegative_int(
            step.get("gradient_steps"),
            f"checkpoint {index} gradient_steps",
        )
        trainable_groups = _nonnegative_int(
            step.get("trainable_groups"),
            f"checkpoint {index} trainable_groups",
        )
        gradient_work = step.get("gradient_work_performed")
        _require(isinstance(gradient_work, bool), f"gradient-work marker is not boolean at {index}")
        groups_submitted = _nonnegative_int(
            step.get("groups_submitted"),
            f"checkpoint {index} groups_submitted",
        )
        _require(
            groups_submitted == EXPECTED_TRAINING["groups_per_step"],
            f"submitted groups disagree with training contract at checkpoint {index}",
        )
        stats = step.get("stats", {})
        _require(
            gradient_work is (gradient_steps > 0),
            f"gradient-work marker disagrees at checkpoint {index}",
        )
        _validate_group_stats(
            stats,
            expected_groups=groups_submitted,
            group_size=int(formal["training"]["group_size"]),
            context=f"formal checkpoint {index}",
        )
        _require(
            (trainable_groups > 0) is (gradient_steps > 0),
            f"trainable-group count disagrees at checkpoint {index}",
        )
        _require(
            trainable_groups == stats["mixed"],
            f"trainable groups do not equal mixed groups at checkpoint {index}",
        )
        _validate_art_metrics(
            step.get("metrics", {}),
            groups=groups_submitted,
            trainable_groups=trainable_groups,
            gradient_steps=gradient_steps,
            rollouts=int(stats["rollouts"]),
            context=f"formal checkpoint {index}",
        )
        _require(
            step.get("checkpoint_path")
            == f"{str(formal['lineage_path']).rstrip('/')}/checkpoints/{index:04d}",
            f"checkpoint path disagrees at {index}",
        )

    rebuilt = _progress(steps)
    _require(
        _typed_json_equal(formal.get("progress"), rebuilt),
        "formal cumulative progress does not rebuild",
    )
    _require(
        _nonnegative_int(formal.get("last_completed_step"), "formal last_completed_step")
        == len(steps),
        "formal last step disagrees",
    )
    _require(
        formal.get("latest_checkpoint_path")
        == f"{str(formal['lineage_path']).rstrip('/')}/checkpoints/{len(steps):04d}",
        "latest checkpoint path disagrees",
    )

    dev = formal.get("dev_evaluations")
    _require(isinstance(dev, list) and bool(dev), "scheduled dev evaluations missing")
    expected_dev_steps = list(
        range(
            int(formal["training"]["val_every"]),
            len(steps) + 1,
            int(formal["training"]["val_every"]),
        )
    )
    _require(
        [int(item["step"]) for item in dev] == expected_dev_steps,
        "scheduled dev evaluation steps disagree",
    )
    for item in dev:
        step_number = _nonnegative_int(item.get("step"), "dev checkpoint step")
        stats = item.get("stats", {})
        _validate_group_stats(
            stats,
            expected_groups=20,
            group_size=int(formal["training"]["val_trials"]),
            context=f"dev checkpoint {step_number}",
        )
        _require(item.get("rollouts") == stats.get("rollouts"), "dev rollout count disagrees")
        avg_reward = _bounded_reward(item.get("avg_reward"), f"dev checkpoint {step_number} reward")
        _require(avg_reward == stats.get("reward_mean"), "dev reward mean disagrees")
    expected_selected = min(dev, key=lambda item: (-float(item["avg_reward"]), int(item["step"])))
    selected = formal.get("selected_checkpoint", {})
    for field in ("step", "avg_reward", "rollouts", "stats"):
        _require(
            _typed_json_equal(selected.get(field), expected_selected.get(field)),
            f"selected {field} disagrees",
        )
    selected_step = _nonnegative_int(selected.get("step"), "selected checkpoint step")
    _require(selected_step == EXPECTED_SELECTED_STEP, "unexpected official selected checkpoint")
    _require(
        _typed_json_equal(selected.get("avg_reward"), EXPECTED_SELECTED_REWARD),
        "unexpected official selected reward",
    )
    _require(
        selected.get("selection_rule") == "highest frozen-dev average reward; ties choose earliest",
        "selected checkpoint rule disagrees",
    )
    _require(
        selected.get("checkpoint_path")
        == f"{str(formal['lineage_path']).rstrip('/')}/checkpoints/{selected_step:04d}",
        "selected checkpoint path disagrees",
    )
    _require(
        _typed_json_equal(
            selected.get("training_progress"),
            _progress(steps[:selected_step]),
        ),
        "selected-checkpoint progress does not rebuild",
    )

    sparse_window = [int(step["stats"]["mixed"]) for step in steps[-10:]]
    _require(len(sparse_window) == 10, "formal stopped before a complete sparse window")
    _require(sum(sparse_window) <= 1, "last ten positions do not satisfy the sparse stop")
    prior_windows = [
        [int(step["stats"]["mixed"]) for step in steps[end - 10 : end]]
        for end in range(10, len(steps))
    ]
    _require(
        all(sum(window) > 1 for window in prior_windows),
        "formal continued after an earlier sparse-reward trigger",
    )


def _exit_code(run_name: str) -> int:
    path = GPU_RESULTS / run_name / "process.exit"
    return int(path.read_text().strip())


def build() -> str:
    """Validate the evidence and render the training report."""
    backup = validate_artifact_checksums()
    preflight, smoke, formal = load_manifests()
    validate_manifests(preflight, smoke, formal)
    counter_audit = load_counter_audit()
    validate_counter_audit(counter_audit, backup, formal)
    restore = load_restore_manifest()
    validate_restore_manifest(restore, backup)
    _require(_exit_code("preflight-qwen3coder-r1") == 0, "preflight process exit was nonzero")
    _require(_exit_code("smoke-qwen3coder-r1") == 0, "smoke process exit was nonzero")
    _require(_exit_code("grpo-4b-qwen3coder-r1") == 1, "formal protocol exit was not 1")

    progress = formal["progress"]
    selected = formal["selected_checkpoint"]
    steps = formal["train_steps"]
    sparse_window = [int(step["stats"]["mixed"]) for step in steps[-10:]]
    gradient_steps = [step for step in steps if step["gradient_work_performed"]]
    logprob = preflight["logprob_gate"]
    rollout_only = preflight["rollout_only"]
    smoke_trainable = int(smoke["trainable_groups"])
    smoke_group_label = "group" if smoke_trainable == 1 else "groups"

    lines: list[str] = []
    w = lines.append
    w("# Official GRPO training outcome")
    w("")
    w(
        "Generated by `uv run python -m service_agent.eval.report_grpo` from the "
        "three manifest-v3 artifacts under `results/gpu/`. The generator "
        "rebuilds cumulative progress from every formal train-step record and "
        "refuses inconsistent phase contracts, split locks, checkpoint "
        "selection, sparse-stop evidence, W&B counter reconciliation, or "
        "selected-adapter recovery evidence."
    )
    w("")
    w("## Outcome")
    w("")
    w(
        f"The formal lineage requested {formal['training']['steps']} rollout/checkpoint "
        f"positions and completed {progress['checkpoint_steps_completed']}. Across "
        f"those positions the formal manifest reconstructs "
        f"{progress['trainable_groups']} trainable "
        f"groups and {progress['gradient_steps']} gradient steps; "
        f"{progress['skipped_checkpoint_steps']} positions had no within-group "
        "reward variance and therefore performed no gradient work. The run then "
        "entered terminal `stopped_sparse_reward` under the predeclared ten-position "
        "rule. This was a controlled protocol stop, not a completed 60-position run "
        "and not an infrastructure crash."
    )
    w("")
    w(
        f"Among the scheduled frozen-dev evaluations completed before the stop, "
        f"checkpoint `{selected['step']:04d}` was selected at mean reward "
        f"{selected['avg_reward']:.3f}. Checkpoint "
        f"`{progress['final_checkpoint_step']:04d}` is only the latest terminal "
        f"lineage position. The {selected['avg_reward']:.3f} value is "
        "checkpoint-selection telemetry, not a "
        f"final 2x2 result; {selected['avg_reward']:.3f} is not comparable to the "
        "MLX base-dev ablation."
    )
    w("")
    w("| Phase | Manifest status | Process exit | What the gate established |")
    w("|---|---|---:|---|")
    w(
        "| preflight | `passed` | 0 | exact prompt tokens, bounded logprob drift, "
        "strict replay, reward once, zero updates |"
    )
    w(
        f"| smoke | `passed` | 0 | {smoke_trainable} trainable {smoke_group_label}, "
        f"{smoke['gradient_steps']} gradient steps, checkpoint 0000→0001 |"
    )
    w(
        f"| formal | `stopped_sparse_reward` | 1 | terminal protocol stop after "
        f"{progress['checkpoint_steps_completed']} positions; selected checkpoint "
        f"{selected['step']:04d} |"
    )
    w("")
    w(
        "The formal driver writes the terminal manifest atomically and then raises a "
        "protocol error, so process exit 1 is expected for this manifest status. A "
        "nonzero exit without the matching terminal manifest would not be accepted."
    )
    w("")
    w("## Preflight and smoke evidence")
    w("")
    w("| Check | Recorded value |")
    w("|---|---:|")
    w(f"| exact prompt-token match | `{str(logprob['prompt_token_ids_exact']).lower()}` |")
    w(f"| logprob prompts / completion tokens | {logprob['prompts']} / {logprob['tokens']} |")
    w(f"| mean importance ratio | {logprob['ratio_mean']:.6f} |")
    w(f"| p95 / p99 importance ratio | {logprob['ratio_p95']:.6f} / {logprob['ratio_p99']:.6f} |")
    w(f"| pre-update clip fraction | {logprob['clip_fraction_before_first_update']:.6f} |")
    w(f"| rollout-only groups / rollouts | {rollout_only['stats']['groups']} / {rollout_only['stats']['rollouts']} |")
    w(f"| smoke groups / rollouts | {smoke['stats']['groups']} / {smoke['stats']['rollouts']} |")
    w(f"| smoke reward mean | {smoke['stats']['reward_mean']:.4f} |")
    w(f"| smoke trainable groups / gradient steps | {smoke['trainable_groups']} / {smoke['gradient_steps']} |")
    w("")
    w(
        "Preflight reward and smoke reward are gate samples, not benchmark estimates. "
        "Their purpose is to prove token/logprob compatibility, trajectory replay, "
        "reward delivery, and real learner work before formal training."
    )
    w("")
    w("## Formal gradient work")
    w("")
    w("| Checkpoint position | Rollout reward mean | Mixed groups | ART gradient steps |")
    w("|---:|---:|---:|---:|")
    for step in gradient_steps:
        w(
            f"| {step['checkpoint_step']:04d} | {step['stats']['reward_mean']:.3f} | "
            f"{step['stats']['mixed']} | {step['gradient_steps']} |"
        )
    w(
        f"| **total** |  | **{progress['trainable_groups']} trainable groups** | "
        f"**{progress['gradient_steps']}** |"
    )
    w("")
    w(
        "Checkpoint positions are not optimizer updates. ART also advanced through "
        f"{progress['skipped_checkpoint_steps']} positions by copying the current "
        "adapter when both submitted groups were constant-reward."
    )
    w("")
    w("## W&B cumulative-counter reconciliation")
    w("")
    w(
        f"The formal ART history contains {counter_audit['record_counts']['total']} "
        "records: "
        f"{counter_audit['record_counts']['rollout']} rollout records, "
        f"{counter_audit['record_counts']['backend_train']} backend-train records, "
        f"and {counter_audit['record_counts']['dev']} dev records. Each formal "
        "position's submitted/trainable group counters appear once on its rollout "
        "record and again on its backend record. ART's cumulative W&B state therefore "
        f"shows {counter_audit['art_cumulative_state']['groups_submitted']} submitted "
        f"and {counter_audit['art_cumulative_state']['groups_trainable']} trainable "
        "groups, exactly twice the unique-position totals."
    )
    w("")
    w(
        f"The authoritative formal counts are "
        f"{counter_audit['manifest_totals']['groups_submitted']} submitted groups, "
        f"{counter_audit['manifest_totals']['groups_trainable']} trainable groups, "
        f"and {counter_audit['manifest_totals']['gradient_steps']} gradient steps, "
        "rebuilt from the 24 manifest train-step records and matched by the 24 "
        "backend records. Gradient steps occur only on backend records, so W&B also "
        f"shows {counter_audit['art_cumulative_state']['gradient_steps']}; they are "
        "not doubled. This is a logging-aggregation artifact, not a second training "
        "pass."
    )
    w("")
    w("## Frozen-dev checkpoint selection")
    w("")
    w("| Checkpoint | Rollouts | Mean reward | Mixed groups |")
    w("|---:|---:|---:|---:|")
    for item in formal["dev_evaluations"]:
        w(
            f"| {item['step']:04d} | {item['rollouts']} | {item['avg_reward']:.3f} | "
            f"{item['stats']['mixed']} |"
        )
    w("")
    w(
        "The selection rule was fixed before training: highest frozen-dev mean "
        "reward, with ties going to the earliest checkpoint. Through checkpoint "
        f"{selected['step']:04d}, the lineage had completed "
        f"{selected['training_progress']['gradient_steps']} ART gradient steps "
        f"across {selected['training_progress']['trainable_groups']} trainable groups."
    )
    w("")
    w("## Sparse-reward stop")
    w("")
    w(
        "Mixed-group counts over the final ten rollout/checkpoint positions "
        f"({len(steps) - 9:04d}–{len(steps):04d}) were "
        f"`{', '.join(map(str, sparse_window))}`. Their sum is "
        f"{sum(sparse_window)}, meeting the fail-closed rule of at most one. "
        "The exact formal lineage is terminal and cannot be resumed. SFT, a new "
        "reward design, or a different sampling schedule would be a new protocol "
        "decision rather than continuation of this run."
    )
    w("")
    w("## Adapter recovery proof")
    w("")
    w(
        "The 3.3 GB backup contains the ART lineages and LoRA checkpoints, not the "
        "bf16 base-model weights, so it is recoverable but not a standalone offline "
        "model bundle. Recovery must first obtain the exact pinned "
        f"`{restore['base_model']}` revision `{restore['base_model_revision']}`, "
        "load that snapshot explicitly, and then attach the selected adapter. The "
        "absolute source-machine path stored in `adapter_config.json` is not the "
        "recovery contract."
    )
    w("")
    w(
        f"That procedure was exercised on the RTX PRO 6000 with checkpoint "
        f"{restore['adapter_checkpoint']:04d} copied to a separate restore directory. "
        f"The explicit bf16 base plus adapter `{restore['adapter_sha256']}` loaded as "
        f"`{restore['model_class']}` with adapter name "
        f"`{restore['loaded_adapter_names'][0]}` and generated "
        f"{restore['new_tokens_generated']} non-benchmark token. No evaluation split "
        "was accessed."
    )
    w("")
    w("## Provenance")
    w("")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| superproject | `{formal['repo_commit']}` |")
    w(f"| ART | `{formal['art_commit']}` |")
    w(f"| tau2 fork | `{formal['tau2_commit']}` |")
    w(f"| model | `{formal['base_model']}` @ `{formal['base_model_revision']}` |")
    w(f"| tool parser | `{formal['tool_call_parser']}` |")
    w(f"| semantic contract | `{formal['semantic_contract_sha256']}` |")
    w(f"| GPU | {formal['system']['gpu']} |")
    w(f"| indexed backup files | {len(backup)} |")
    w(f"| backup index SHA-256 | `{hashlib.sha256(BACKUP_CHECKSUMS_PATH.read_bytes()).hexdigest()}` |")
    w(f"| selected adapter SHA-256 | `{backup[SELECTED_ADAPTER_PATH]}` |")
    w(f"| latest adapter SHA-256 | `{backup[LATEST_ADAPTER_PATH]}` |")
    w(f"| selected-adapter recovery | `{restore['status']}` at `{restore['checked_at']}` |")
    w(f"| preflight W&B | [{preflight['run_name']}]({preflight['wandb_url']}) |")
    w(f"| smoke W&B | [{smoke['run_name']}]({smoke['wandb_url']}) |")
    w(f"| formal W&B | [{formal['run_name']}]({formal['wandb_url']}) |")
    w("")
    w(
        "All three manifests record `test_split_locked=true`. No test, full, or "
        "base-split result is used in this report."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the committed report matches the manifests without writing",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        help="optionally hash every source file in the ignored local backup",
    )
    args = parser.parse_args()
    output = REPO / "reports/grpo_training.md"
    rendered = build()
    if args.backup_root is not None:
        validate_backup_files(args.backup_root)
        validate_counter_audit_sources(args.backup_root, load_counter_audit())
        print(f"verified {args.backup_root}")
    if args.check:
        _require(output.exists(), "reports/grpo_training.md is missing")
        _require(output.read_text() == rendered, "reports/grpo_training.md is stale")
        print("verified reports/grpo_training.md")
    else:
        output.write_text(rendered)
        print("wrote reports/grpo_training.md")


if __name__ == "__main__":
    main()
