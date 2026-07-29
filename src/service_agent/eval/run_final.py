"""Approval-gated native runner for the one final 2x2 evaluation.

This module deliberately does not extend ``run_ablation`` with a test option.
The final evaluation is a separate protocol with no tunable experiment
arguments: four fixed cells, one fixed serving process, one fixed simulator,
and one exact approval value.  Its three actions are:

``--preflight``
    Validate the committed training evidence, repository, model artifacts, and
    live dual-alias serving contract.  It never loads a benchmark task.

``--smoke``
    Run the exact four-cell policy configuration over three frozen dev tasks,
    one trial each.  This is an operational gate, not authorization and not a
    source of final numbers.

no action flag
    Require ``FINAL_TEST_APPROVAL=FINAL_TEST_APPROVED`` before the official
    test loader is called, validate the preflight and smoke gates, then run the
    four final cells in their frozen order through tau2's native runner.

Raw results stay under the caller-provided output directory.  The runner never
prints intermediate rewards and never computes a report; result validation and
reporting are separate, post-run steps.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from service_agent.training.contracts import (
    ART_COMMIT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHAT_TEMPLATE_KWARGS,
    TAU2_COMMIT,
    TOOL_CALL_PARSER,
)

REPO = Path(__file__).resolve().parents[3]

PROTOCOL_ID = "service-agent-final-2x2-r1"
PROTOCOL_SCHEMA_VERSION = 1
APPROVAL_ENV = "FINAL_TEST_APPROVAL"
APPROVAL_VALUE = "FINAL_TEST_APPROVED"

EXPECTED_TASK_COUNT = 40
EXPECTED_TRIAL_COUNT = 8
BASE_SEED = 42
MAX_STEPS = 100
MAX_ERRORS = 10
MAX_CONCURRENCY = 3
MAX_COMPLETION_TOKENS = 1_024
POLICY_TEMPERATURE = 0.0
USER_MODEL = "deepseek/deepseek-v4-pro"
USER_TEMPERATURE = 0.0
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

BASE_ALIAS = "final-frozen-r1"
RL_ALIAS = "final-rl-cp0015-r1"
SELECTED_CHECKPOINT = 15
ADAPTER_SHA256 = "1018931f9483c71ae20fbd59c76ab6a0c73137d4aefe9c8ad823175931b2c898"
SEMANTIC_CONTRACT_SHA256 = (
    "91fa4cb5c06414976cf029003ad621b36becfe154ee86201c726f331ec9d6fb6"
)

FINAL_MANIFEST = "final_manifest.json"
SMOKE_MANIFEST = "smoke_manifest.json"
CELL_CONFIG = "run_config.json"
CELL_RESULTS = "results.json"
CELL_LOG = "runner.log"


@dataclass(frozen=True)
class CellSpec:
    """One frozen cell in the model x harness factorial."""

    name: str
    model_row: str
    harness: str
    agent: str
    served_model_alias: str


CELL_SPECS = (
    CellSpec("base_h0", "base", "h0", "llm_agent", BASE_ALIAS),
    CellSpec("base_h2", "base", "h2", "governed_llm_agent_h2", BASE_ALIAS),
    CellSpec("rl_h0", "rl", "h0", "llm_agent", RL_ALIAS),
    CellSpec("rl_h2", "rl", "h2", "governed_llm_agent_h2", RL_ALIAS),
)
CELL_ORDER = tuple(cell.name for cell in CELL_SPECS)


def trial_seeds(count: int = EXPECTED_TRIAL_COUNT) -> list[int]:
    """Reproduce tau2 ``run_tasks`` trial seeds without changing global RNG."""

    rng = random.Random(BASE_SEED)
    return [rng.randint(0, 1_000_000) for _ in range(count)]


FINAL_TRIAL_SEEDS = tuple(trial_seeds())


@dataclass(frozen=True)
class CliArgs:
    out: Path
    serving_manifest: Path
    base_snapshot: Path
    adapter: Path
    smoke_manifest: Path | None
    preflight: bool
    smoke: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--preflight",
        action="store_true",
        help="validate frozen inputs and serving without loading benchmark tasks",
    )
    action.add_argument(
        "--smoke",
        action="store_true",
        help="run the four frozen cells over the fixed three-task dev smoke",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--serving-manifest", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=None,
        help="required only for the approved final run",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    args = _parser().parse_args(argv)
    if (args.preflight or args.smoke) and args.smoke_manifest is not None:
        _parser().error("--smoke-manifest is only valid for the approved final run")
    if not args.preflight and not args.smoke and args.smoke_manifest is None:
        _parser().error("--smoke-manifest is required for the approved final run")
    return CliArgs(
        out=args.out,
        serving_manifest=args.serving_manifest,
        base_snapshot=args.base_snapshot,
        adapter=args.adapter,
        smoke_manifest=args.smoke_manifest,
        preflight=args.preflight,
        smoke=args.smoke,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"{description} is missing: {path}")
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), f"{description} is not a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repo_state(repo: Path = REPO) -> dict[str, str]:
    """Require committed first-party code and the two exact upstream pins."""

    _require(
        Path.cwd().resolve() == repo.resolve(),
        "run the final protocol from the repository root so Results.info records its commit",
    )
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all")
    _require(not dirty, "repository is dirty; final protocol requires a committed tree")
    repo_commit = _git(repo, "rev-parse", "HEAD")
    art_commit = _git(repo / "third_party/ART", "rev-parse", "HEAD")
    tau2_commit = _git(repo / "third_party/tau2-bench", "rev-parse", "HEAD")
    _require(art_commit == ART_COMMIT, f"ART commit drift: {art_commit}")
    _require(tau2_commit == TAU2_COMMIT, f"tau2 commit drift: {tau2_commit}")
    _require(
        not _git(repo / "third_party/ART", "status", "--porcelain"),
        "ART submodule is dirty",
    )
    _require(
        not _git(repo / "third_party/tau2-bench", "status", "--porcelain"),
        "tau2 submodule is dirty",
    )
    return {
        "repo_commit": repo_commit,
        "art_commit": art_commit,
        "tau2_commit": tau2_commit,
    }


def _gpu_provenance() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    _require(len(rows) == 1, f"final protocol requires exactly one GPU; found {len(rows)}")
    parts = [part.strip() for part in rows[0].split(",", maxsplit=3)]
    _require(len(parts) == 4, "could not parse nvidia-smi GPU provenance")
    name, uuid, driver, memory_mib_text = parts
    _require(name == EXPECTED_GPU_NAME, f"final GPU drift: {name!r}")
    _require(uuid.startswith("GPU-"), "final GPU UUID is invalid")
    _require(bool(driver), "final GPU driver version is missing")
    try:
        memory_mib = int(memory_mib_text)
    except ValueError as exc:
        raise RuntimeError("final GPU memory is not an integer MiB value") from exc
    _require(memory_mib > 90_000, "final GPU memory is below the 96 GB hardware contract")
    return {
        "count": 1,
        "name": name,
        "uuid": uuid,
        "driver_version": driver,
        "memory_total_mib": memory_mib,
    }


def _serving_process_provenance(
    command: list[str],
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Bind preflight, smoke, final cells, and resume to one server process."""

    _require(command and all(isinstance(arg, str) and arg for arg in command), "bad server argv")
    _require(proc_root.is_dir(), "final serving process validation requires Linux /proc")
    matches: list[tuple[int, Path, list[str], str]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline_path = entry / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode(errors="surrogateescape") for part in raw.split(b"\0") if part]
        match_kind: str | None = None
        if argv == command:
            match_kind = "direct"
        elif (
            len(argv) == len(command) + 1
            and argv[1] == command[0]
            and argv[2:] == command[1:]
        ):
            # Linux binfmt_script inserts the Python interpreter before an
            # installed console-script path in /proc/<pid>/cmdline.
            match_kind = "python_console_script"
        if match_kind is not None:
            matches.append((int(entry.name), entry, argv, match_kind))
    _require(
        len(matches) == 1,
        f"expected one exact final serving process; found {len(matches)}",
    )
    pid, process_dir, observed_argv, match_kind = matches[0]
    try:
        stat = (process_dir / "stat").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError) as exc:
        raise RuntimeError("final serving process exited during validation") from exc
    close = stat.rfind(")")
    _require(close > 0, "could not parse serving /proc stat")
    fields_after_comm = stat[close + 2 :].split()
    _require(len(fields_after_comm) > 19, "serving /proc stat is incomplete")
    try:
        start_time_ticks = int(fields_after_comm[19])
    except ValueError as exc:
        raise RuntimeError("serving /proc start time is invalid") from exc
    boot_id = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    _require(bool(boot_id), "Linux boot id is missing")
    return {
        "pid": pid,
        "start_time_ticks": start_time_ticks,
        "boot_id": boot_id,
        "match_kind": match_kind,
        "expected_command_sha256": _canonical_sha256(command),
        "observed_argv_sha256": _canonical_sha256(observed_argv),
    }


def _validate_training_evidence(repo: Path = REPO) -> dict[str, Any]:
    """Reuse the committed GRPO evidence validator before final serving."""

    from service_agent.eval.report_grpo import (
        load_manifests,
        load_restore_manifest,
        validate_artifact_checksums,
        validate_manifests,
        validate_restore_manifest,
    )

    checksums = validate_artifact_checksums()
    preflight, smoke, formal = load_manifests()
    validate_manifests(preflight, smoke, formal)
    restore = load_restore_manifest()
    validate_restore_manifest(restore, checksums)
    selected = formal["selected_checkpoint"]
    _require(selected.get("step") == SELECTED_CHECKPOINT, "selected checkpoint drifted")
    _require(
        formal.get("semantic_contract_sha256") == SEMANTIC_CONTRACT_SHA256,
        "formal semantic contract drifted",
    )
    return {
        "training_repo_commit": formal["repo_commit"],
        "formal_manifest_sha256": _sha256_file(
            repo / "results/gpu/grpo-4b-qwen3coder-r1/train_manifest.json"
        ),
        "restore_manifest_sha256": _sha256_file(
            repo / "results/gpu/restore-cp0015-r1/restore_manifest.json"
        ),
        "selected_checkpoint": SELECTED_CHECKPOINT,
        "selected_adapter_sha256": restore["adapter_sha256"],
        "semantic_contract_sha256": formal["semantic_contract_sha256"],
        "semantic_input_hashes": deepcopy(formal["semantic_input_hashes"]),
    }


def _validate_serving(
    manifest_path: Path,
    snapshot: Path,
    adapter: Path,
) -> dict[str, Any]:
    """Validate the persisted contract and re-probe the live local server."""

    from service_agent.eval import final_serving

    _require(
        final_serving.BASE_ALIAS == BASE_ALIAS and final_serving.RL_ALIAS == RL_ALIAS,
        "runner and serving model aliases disagree",
    )
    payload = final_serving.validate_manifest_for_final_runner(manifest_path)
    _require(
        Path(payload["base_snapshot"]).resolve() == snapshot.resolve(),
        "serving manifest base snapshot differs from the runner input",
    )
    _require(
        Path(payload["adapter_path"]).resolve() == adapter.resolve(),
        "serving manifest adapter differs from the runner input",
    )
    _require(payload.get("base_model") == BASE_MODEL_ID, "serving base model drifted")
    _require(
        payload.get("base_model_revision") == BASE_MODEL_REVISION,
        "serving base revision drifted",
    )
    _require(payload.get("adapter_sha256") == ADAPTER_SHA256, "serving adapter hash drifted")
    _require(payload.get("dtype") == "bfloat16", "serving dtype is not bfloat16")
    _require(payload.get("tool_call_parser") == TOOL_CALL_PARSER, "serving parser drifted")
    _require(
        payload.get("chat_template_kwargs") == CHAT_TEMPLATE_KWARGS,
        "serving chat-template kwargs drifted",
    )
    _require(
        payload.get("max_completion_tokens") == MAX_COMPLETION_TOKENS,
        "serving completion limit drifted",
    )
    _require(payload.get("art_commit") == ART_COMMIT, "serving ART commit drifted")
    _require(payload.get("tau2_commit") == TAU2_COMMIT, "serving tau2 commit drifted")
    live_probe = final_serving.probe_serving(
        payload["api_base"],
        snapshot,
        adapter,
    )
    _require(live_probe.get("status") == "passed", "live serving probe did not pass")
    _require(
        live_probe.get("benchmark_data_accessed") is False,
        "serving probe claims benchmark access",
    )
    return payload


def _ensure_outside_repo(path: Path, repo: Path = REPO) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError:
        return resolved
    raise RuntimeError("final raw output must be outside the Git repository")


def _protocol_public_fields(
    repo_state: dict[str, str],
    serving: dict[str, Any],
    serving_manifest_sha256: str,
    gpu: dict[str, Any],
    serving_process: dict[str, Any],
) -> dict[str, Any]:
    cells = [asdict(cell) for cell in CELL_SPECS]
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "repo_commit": repo_state["repo_commit"],
        "art_commit": repo_state["art_commit"],
        "tau2_commit": repo_state["tau2_commit"],
        "serving_manifest_sha256": serving_manifest_sha256,
        "gpu": gpu,
        "serving_process": serving_process,
        "task_split": "test",
        "expected_task_count": EXPECTED_TASK_COUNT,
        "task_count": EXPECTED_TASK_COUNT,
        "expected_trial_count": EXPECTED_TRIAL_COUNT,
        "trials": EXPECTED_TRIAL_COUNT,
        "base_seed": BASE_SEED,
        "trial_seeds": list(FINAL_TRIAL_SEEDS),
        "max_steps": MAX_STEPS,
        "max_errors": MAX_ERRORS,
        "max_concurrency": MAX_CONCURRENCY,
        "policy_temperature": POLICY_TEMPERATURE,
        "policy_max_completion_tokens": MAX_COMPLETION_TOKENS,
        "policy_thinking": "disabled",
        "user_simulator": {
            "model": USER_MODEL,
            "temperature": USER_TEMPERATURE,
            "thinking": "disabled",
        },
        "evaluation_type": "all",
        "cell_order": list(CELL_ORDER),
        "cells": cells,
        "task_set_sha256_algorithm": "sha256(canonical-json(sorted-task-ids))",
        "serving": {
            "api_base": serving["api_base"],
            "base_model": serving["base_model"],
            "base_model_revision": serving["base_model_revision"],
            "base_model_alias": serving["base_model_alias"],
            "rl_model_alias": serving["rl_model_alias"],
            "base_snapshot": serving["base_snapshot"],
            "adapter_path": serving["adapter_path"],
            "adapter_checkpoint": serving["adapter_checkpoint"],
            "adapter_sha256": serving["adapter_sha256"],
            "dtype": serving["dtype"],
            "max_model_len": serving["max_model_len"],
            "max_completion_tokens": serving["max_completion_tokens"],
            "tool_call_parser": serving["tool_call_parser"],
            "chat_template_kwargs": serving["chat_template_kwargs"],
            "semantic_input_hashes": serving["semantic_input_hashes"],
            "runtime_packages": serving["runtime_packages"],
            "snapshot_tree_sha256": serving.get("snapshot_tree_sha256"),
            "snapshot_file_count": serving.get("snapshot_file_count"),
            "snapshot_total_bytes": serving.get("snapshot_total_bytes"),
        },
    }


def _prepare_protocol(args: CliArgs) -> tuple[dict[str, Any], dict[str, Any]]:
    _ensure_outside_repo(args.out)
    repo_state = _repo_state()
    training = _validate_training_evidence()
    serving = _validate_serving(
        args.serving_manifest.resolve(),
        args.base_snapshot.resolve(),
        args.adapter.resolve(),
    )
    serving_sha = _sha256_file(args.serving_manifest.resolve())
    public = _protocol_public_fields(
        repo_state,
        serving,
        serving_sha,
        _gpu_provenance(),
        _serving_process_provenance(serving["command"]),
    )
    frozen = {
        **public,
        "training_evidence": training,
        "serving_manifest_path": str(args.serving_manifest.resolve()),
    }
    return frozen, serving


def _new_state(frozen: dict[str, Any], *, status: str, approval: str) -> dict[str, Any]:
    protocol_hash = _canonical_sha256(frozen)
    return {
        **{key: deepcopy(value) for key, value in frozen.items() if key != "training_evidence"},
        "status": status,
        "approval": approval,
        "protocol_sha256": protocol_hash,
        "completed_cells": [],
        "task_set_sha256": None,
        "native_smoke_manifest_sha256": None,
        "infrastructure_retries": [],
        "cell_result_sha256": {},
        "protocol": deepcopy(frozen),
    }


def _assert_same_protocol(state: dict[str, Any], frozen: dict[str, Any]) -> None:
    _require(
        state.get("protocol_id") == PROTOCOL_ID,
        "output belongs to a different final protocol",
    )
    _require(
        state.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "final manifest schema drifted",
    )
    expected_hash = _canonical_sha256(frozen)
    _require(state.get("protocol_sha256") == expected_hash, "final protocol hash drifted")
    _require(state.get("protocol") == frozen, "embedded final protocol drifted")
    _require(state.get("cell_order") == list(CELL_ORDER), "cell order drifted")


def _root_entries(out: Path) -> set[str]:
    return {path.name for path in out.iterdir()} if out.exists() else set()


def _validate_root_layout(out: Path, manifest_name: str) -> None:
    allowed = {manifest_name, *CELL_ORDER}
    paths = list(out.iterdir()) if out.exists() else []
    _require(
        not any(path.is_symlink() for path in paths),
        "final output root contains a symlink",
    )
    unexpected = {path.name for path in paths} - allowed
    _require(not unexpected, f"unexpected files in output root: {sorted(unexpected)}")


_AUDIT_NAME = re.compile(r"audit_[0-9a-f]{32}\.jsonl")


def _validate_cell_layout(cell_dir: Path, *, complete: bool = False) -> None:
    if not cell_dir.exists():
        return
    _require(
        cell_dir.is_dir() and not cell_dir.is_symlink(),
        f"cell output is not a real directory: {cell_dir}",
    )
    is_h2 = cell_dir.name.endswith("_h2")
    required = {CELL_CONFIG, CELL_RESULTS, CELL_LOG}
    allowed = set(required)
    if is_h2:
        required.add("audit")
        allowed.add("audit")
    paths = list(cell_dir.iterdir())
    _require(
        not any(path.is_symlink() for path in paths),
        f"cell output contains a symlink: {cell_dir}",
    )
    entries = {path.name for path in paths}
    unexpected = entries - allowed
    _require(not unexpected, f"unexpected files in {cell_dir}: {sorted(unexpected)}")
    if complete:
        missing = required - entries
        _require(not missing, f"missing files in {cell_dir}: {sorted(missing)}")
    audit = cell_dir / "audit"
    if audit.exists():
        _require(audit.is_dir(), f"audit path is not a directory: {audit}")
        audit_paths = list(audit.iterdir())
        _require(
            not any(path.is_symlink() for path in audit_paths),
            f"audit output contains a symlink: {audit}",
        )
        bad = [path.name for path in audit_paths if not _AUDIT_NAME.fullmatch(path.name)]
        _require(not bad, f"unexpected audit files in {audit}: {sorted(bad)}")


def _write_or_validate_cell_config(
    cell_dir: Path,
    *,
    cell: CellSpec,
    config_payload: dict[str, Any],
    protocol_sha256: str,
    mode: str,
) -> None:
    payload = {
        "schema_version": 1,
        "mode": mode,
        "protocol_sha256": protocol_sha256,
        "cell": asdict(cell),
        "text_run_config": config_payload,
    }
    path = cell_dir / CELL_CONFIG
    if path.exists():
        _require(_read_json(path, "cell run config") == payload, f"{cell.name} config drifted")
    else:
        _write_json_atomic(path, payload)


def _termination_is_infrastructure(value: Any) -> bool:
    return str(value).lower().split(".")[-1] == "infrastructure_error"


def _simulation_key(simulation: Any) -> tuple[int, str, int]:
    trial = simulation.trial
    seed = simulation.seed
    task_id = simulation.task_id
    _require(isinstance(trial, int), "simulation trial is missing")
    _require(isinstance(seed, int), "simulation seed is missing")
    _require(isinstance(task_id, str) and task_id, "simulation task id is missing")
    return trial, task_id, seed


def _validate_existing_results(
    path: Path,
    *,
    tasks: list[Any],
    trials: int,
    config: Any | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate every existing key before allowing tau2 auto-resume."""

    if not path.exists():
        return False, []
    from tau2.data_model.simulation import Results

    results = Results.load(path)
    if config is not None:
        from tau2.runner.helpers import get_info
        from tau2.utils.pydantic_utils import get_pydantic_hash

        expected_info = get_info(config)
        exclude_fields = {"environment_info": {"policy"}}
        _require(
            get_pydantic_hash(results.info, exclude=exclude_fields)
            == get_pydantic_hash(expected_info, exclude=exclude_fields),
            "resume Results.info differs from the frozen native run config",
        )
    task_ids = {str(task.id) for task in tasks}
    _require(
        {str(task.id) for task in results.tasks} == task_ids,
        "resume task set differs from the frozen task set",
    )
    seeds = trial_seeds(trials)
    allowed = {
        (trial, task_id, seeds[trial])
        for trial in range(trials)
        for task_id in task_ids
    }
    seen: set[tuple[int, str, int]] = set()
    infrastructure: list[dict[str, Any]] = []
    for simulation in results.simulations:
        key = _simulation_key(simulation)
        _require(key in allowed, f"unexpected simulation key in resume file: {key}")
        _require(key not in seen, f"duplicate simulation key in resume file: {key}")
        seen.add(key)
        if _termination_is_infrastructure(simulation.termination_reason):
            info = simulation.info if isinstance(simulation.info, dict) else {}
            infrastructure.append(
                {
                    "trial": key[0],
                    "task_id": key[1],
                    "seed": key[2],
                    "error_type": info.get("error_type"),
                }
            )
            continue
        _require(simulation.reward_info is not None, f"used simulation {key} has no reward")
        _require(
            isinstance(simulation.messages, list) and bool(simulation.messages),
            f"used simulation {key} has no messages",
        )
        reward = simulation.reward_info.reward
        _require(
            isinstance(reward, (int, float))
            and not isinstance(reward, bool)
            and math.isfinite(float(reward)),
            f"used simulation {key} has a non-finite reward",
        )
        _require(
            -1e-9 <= float(reward) <= 1.0 + 1e-9,
            f"used simulation {key} reward is outside [0, 1]",
        )
    complete = seen == allowed and not infrastructure
    return complete, infrastructure


def _record_infrastructure_retry(
    state: dict[str, Any],
    *,
    cell: str,
    results_path: Path,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return
    event = {
        "cell": cell,
        "results_sha256_before_resume": _sha256_file(results_path),
        "keys": sorted(records, key=lambda item: (item["trial"], item["task_id"])),
    }
    if event not in state["infrastructure_retries"]:
        state["infrastructure_retries"].append(event)


def _agent_llm_args(api_base: str) -> dict[str, Any]:
    return {
        "temperature": POLICY_TEMPERATURE,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "api_base": api_base,
        "api_key": "local",
        "extra_body": {"chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)},
    }


def _user_llm_args() -> dict[str, Any]:
    return {
        "temperature": USER_TEMPERATURE,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def _build_run_config(
    cell: CellSpec,
    api_base: str,
    *,
    trials: int,
    task_split_name: str,
):
    from tau2.data_model.simulation import TextRunConfig

    return TextRunConfig(
        domain="telecom",
        task_set_name="telecom",
        task_split_name=task_split_name,
        agent=cell.agent,
        llm_agent=f"openai/{cell.served_model_alias}",
        llm_args_agent=_agent_llm_args(api_base),
        user="user_simulator",
        llm_user=USER_MODEL,
        llm_args_user=_user_llm_args(),
        num_trials=trials,
        seed=BASE_SEED,
        max_steps=MAX_STEPS,
        max_errors=MAX_ERRORS,
        max_concurrency=MAX_CONCURRENCY,
        max_retries=0,
        retry_delay=0.0,
        auto_resume=True,
        auto_review=False,
        hallucination_retries=0,
        verbose_logs=False,
        log_level="ERROR",
    )


@contextlib.contextmanager
def _quiet_native_output(log_path: Path):
    """Route native progress, including interim reward averages, to a raw log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            yield


def _run_cell(
    cell: CellSpec,
    *,
    tasks: list[Any],
    trials: int,
    out: Path,
    api_base: str,
    protocol_sha256: str,
    mode: str,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks

    from service_agent.eval.registration import register_governed_agents, set_audit_dir

    cell_dir = out / cell.name
    cell_dir.mkdir(parents=True, exist_ok=True)
    _validate_cell_layout(cell_dir)
    task_split_name = "test" if mode == "final" else "frozen_dev_smoke3"
    config = _build_run_config(
        cell,
        api_base,
        trials=trials,
        task_split_name=task_split_name,
    )
    _write_or_validate_cell_config(
        cell_dir,
        cell=cell,
        config_payload=json.loads(config.model_dump_json()),
        protocol_sha256=protocol_sha256,
        mode=mode,
    )
    results_path = cell_dir / CELL_RESULTS
    complete, infrastructure = _validate_existing_results(
        results_path,
        tasks=tasks,
        trials=trials,
        config=config,
    )
    if complete:
        return
    _record_infrastructure_retry(
        state,
        cell=cell.name,
        results_path=results_path,
        records=infrastructure,
    )
    _write_json_atomic(state_path, state)

    register_governed_agents()
    audit_dir = cell_dir / "audit"
    if cell.harness == "h2":
        audit_dir.mkdir(exist_ok=True)
        set_audit_dir(audit_dir)
    else:
        set_audit_dir(None)
    try:
        with _quiet_native_output(cell_dir / CELL_LOG):
            run_tasks(
                config,
                tasks,
                save_path=results_path,
                save_dir=cell_dir,
                evaluation_type=EvaluationType.ALL,
                console_display=False,
            )
    finally:
        set_audit_dir(None)

    complete, infrastructure = _validate_existing_results(
        results_path,
        tasks=tasks,
        trials=trials,
        config=config,
    )
    _require(not infrastructure, f"{cell.name} ended with infrastructure failures")
    _require(complete, f"{cell.name} did not produce the complete fixed Cartesian run")


def _task_set_sha256(tasks: list[Any]) -> str:
    return _canonical_sha256(sorted(str(task.id) for task in tasks))


def _record_completed_cell(
    state: dict[str, Any],
    *,
    cell: str,
    out: Path,
    state_path: Path,
) -> None:
    results_path = out / cell / CELL_RESULTS
    _require(results_path.is_file(), f"{cell} has no persisted results")
    result_sha = _sha256_file(results_path)
    recorded = state["cell_result_sha256"].get(cell)
    if recorded is not None:
        _require(recorded == result_sha, f"{cell} results changed after completion")
    else:
        state["cell_result_sha256"][cell] = result_sha
    if cell not in state["completed_cells"]:
        state["completed_cells"].append(cell)
    _write_json_atomic(state_path, state)


def _validate_completed_artifacts(state: dict[str, Any], out: Path) -> None:
    _require(state.get("completed_cells") == list(CELL_ORDER), "completed cell order drifted")
    hashes = state.get("cell_result_sha256")
    _require(isinstance(hashes, dict), "cell result hashes are missing")
    _require(set(hashes) == set(CELL_ORDER), "cell result hash set is incomplete")
    for cell in CELL_ORDER:
        _validate_cell_layout(out / cell, complete=True)
        path = out / cell / CELL_RESULTS
        _require(path.is_file(), f"{cell} results disappeared after completion")
        _require(_sha256_file(path) == hashes[cell], f"{cell} results hash drifted")


def _load_smoke_tasks() -> list[Any]:
    """Load only upstream train tasks, then select the frozen three-task dev smoke."""

    from tau2.registry import registry

    from service_agent.eval.run_ablation import smoke3_ids
    from service_agent.splits import load_frozen_dev_ids

    ids = smoke3_ids(load_frozen_dev_ids())
    wanted = set(ids)
    train_tasks = registry.get_tasks_loader("telecom")(task_split_name="train")
    tasks = [task for task in train_tasks if task.id in wanted]
    _require(len(tasks) == 3 and {task.id for task in tasks} == wanted, "dev smoke tasks drifted")
    return sorted(tasks, key=lambda task: task.id)


def _load_official_test_tasks() -> list[Any]:
    """The only production code path that loads the official test split."""

    from tau2.registry import registry

    tasks = registry.get_tasks_loader("telecom")(task_split_name="test")
    _require(len(tasks) == EXPECTED_TASK_COUNT, "official test task count drifted")
    _require(len({task.id for task in tasks}) == len(tasks), "official test task IDs duplicate")
    return sorted(tasks, key=lambda task: task.id)


def _require_final_approval(environ: dict[str, str] | os._Environ[str] = os.environ) -> None:
    """Fail before the final test loader unless the exact environment value exists."""

    if environ.get(APPROVAL_ENV) != APPROVAL_VALUE:
        raise PermissionError(
            f"set {APPROVAL_ENV} to the exact approval value before the final run"
        )


def _load_runtime_environment(
    environ: dict[str, str] | os._Environ[str] = os.environ,
) -> None:
    """Load the ignored secret file, then record only that the required key exists."""

    from dotenv import load_dotenv

    load_dotenv()
    _require(bool(environ.get("DEEPSEEK_API_KEY")), "DEEPSEEK_API_KEY is not configured")


def _run_preflight(args: CliArgs, frozen: dict[str, Any]) -> dict[str, Any]:
    out = _ensure_outside_repo(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _validate_root_layout(out, FINAL_MANIFEST)
    manifest_path = out / FINAL_MANIFEST
    if manifest_path.exists():
        state = _read_json(manifest_path, "final preflight manifest")
        _assert_same_protocol(state, frozen)
        _require(
            state.get("status") == "preflight_passed",
            "preflight cannot overwrite a final run that already started",
        )
        _require(not state.get("completed_cells"), "preflight output already has cell results")
        return state
    _require(not _root_entries(out), "preflight output must be empty")
    state = _new_state(frozen, status="preflight_passed", approval="not_consumed")
    _write_json_atomic(manifest_path, state)
    return state


def _run_smoke(
    args: CliArgs,
    frozen: dict[str, Any],
    serving: dict[str, Any],
    *,
    load_tasks_fn: Callable[[], list[Any]],
    run_cell_fn: Callable[..., None],
) -> dict[str, Any]:
    out = _ensure_outside_repo(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _validate_root_layout(out, SMOKE_MANIFEST)
    manifest_path = out / SMOKE_MANIFEST
    if manifest_path.exists():
        state = _read_json(manifest_path, "native smoke manifest")
        _assert_same_protocol(state, frozen)
        _require(
            state.get("status") in {"smoke_running", "smoke_complete"},
            "smoke manifest status is invalid",
        )
    else:
        _require(not _root_entries(out), "smoke output must be empty")
        state = _new_state(frozen, status="smoke_running", approval="not_consumed")
        state["task_split"] = "frozen_dev_smoke3"
        state["task_count"] = 3
        state["trials"] = 1
        state["trial_seeds"] = trial_seeds(1)
        _write_json_atomic(manifest_path, state)

    tasks = load_tasks_fn()
    digest = _task_set_sha256(tasks)
    if state["task_set_sha256"] is None:
        state["task_set_sha256"] = digest
    _require(state["task_set_sha256"] == digest, "smoke task set drifted")
    _write_json_atomic(manifest_path, state)

    for cell in CELL_SPECS:
        run_cell_fn(
            cell,
            tasks=tasks,
            trials=1,
            out=out,
            api_base=serving["api_base"],
            protocol_sha256=state["protocol_sha256"],
            mode="smoke",
            state=state,
            state_path=manifest_path,
        )
        _record_completed_cell(
            state,
            cell=cell.name,
            out=out,
            state_path=manifest_path,
        )
    _require(state["completed_cells"] == list(CELL_ORDER), "smoke cell order drifted")
    _validate_completed_artifacts(state, out)
    state["status"] = "smoke_complete"
    _write_json_atomic(manifest_path, state)
    return state


def _validate_smoke_gate(path: Path, frozen: dict[str, Any]) -> str:
    smoke = _read_json(path.resolve(), "native smoke manifest")
    _assert_same_protocol(smoke, frozen)
    _require(smoke.get("status") == "smoke_complete", "native smoke did not complete")
    _require(smoke.get("approval") == "not_consumed", "smoke consumed final approval")
    _require(smoke.get("task_split") == "frozen_dev_smoke3", "smoke split drifted")
    _require(smoke.get("task_count") == 3, "smoke task count drifted")
    _require(smoke.get("trials") == 1, "smoke trial count drifted")
    _require(smoke.get("trial_seeds") == trial_seeds(1), "smoke trial seed drifted")
    _require(smoke.get("completed_cells") == list(CELL_ORDER), "smoke cells incomplete")
    _validate_completed_artifacts(smoke, path.resolve().parent)
    return _sha256_file(path.resolve())


def _run_final(
    args: CliArgs,
    frozen: dict[str, Any],
    serving: dict[str, Any],
    *,
    load_tasks_fn: Callable[[], list[Any]],
    run_cell_fn: Callable[..., None],
) -> dict[str, Any]:
    _require(args.smoke_manifest is not None, "approved run requires a smoke manifest")
    smoke_sha = _validate_smoke_gate(args.smoke_manifest, frozen)
    out = _ensure_outside_repo(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _validate_root_layout(out, FINAL_MANIFEST)
    manifest_path = out / FINAL_MANIFEST
    _require(manifest_path.exists(), "approved final run requires a passed preflight manifest")
    state = _read_json(manifest_path, "final manifest")
    _assert_same_protocol(state, frozen)
    _require(
        state.get("status") in {"preflight_passed", "running", "complete"},
        "final manifest status is invalid",
    )
    if state["native_smoke_manifest_sha256"] is None:
        state["native_smoke_manifest_sha256"] = smoke_sha
    _require(
        state["native_smoke_manifest_sha256"] == smoke_sha,
        "native smoke manifest changed after final authorization",
    )
    state["approval"] = APPROVAL_VALUE
    if state["status"] == "complete":
        _validate_completed_artifacts(state, out)
        _write_json_atomic(manifest_path, state)
        return state
    state["status"] = "running"
    _write_json_atomic(manifest_path, state)

    # This is intentionally the first and only official-test loader call.
    tasks = load_tasks_fn()
    digest = _task_set_sha256(tasks)
    if state["task_set_sha256"] is None:
        state["task_set_sha256"] = digest
    _require(state["task_set_sha256"] == digest, "official test task set drifted")
    _write_json_atomic(manifest_path, state)

    for cell in CELL_SPECS:
        run_cell_fn(
            cell,
            tasks=tasks,
            trials=EXPECTED_TRIAL_COUNT,
            out=out,
            api_base=serving["api_base"],
            protocol_sha256=state["protocol_sha256"],
            mode="final",
            state=state,
            state_path=manifest_path,
        )
        _record_completed_cell(
            state,
            cell=cell.name,
            out=out,
            state_path=manifest_path,
        )
    _require(state["completed_cells"] == list(CELL_ORDER), "final cell order drifted")
    _validate_completed_artifacts(state, out)
    state["status"] = "complete"
    _write_json_atomic(manifest_path, state)
    return state


def execute(
    args: CliArgs,
    *,
    prepare_protocol_fn: Callable[
        [CliArgs], tuple[dict[str, Any], dict[str, Any]]
    ] | None = None,
    load_smoke_tasks_fn: Callable[[], list[Any]] | None = None,
    load_final_tasks_fn: Callable[[], list[Any]] | None = None,
    run_cell_fn: Callable[..., None] | None = None,
    environ: dict[str, str] | os._Environ[str] = os.environ,
) -> dict[str, Any]:
    """Execute one fixed action; injection points exist only for no-data tests."""

    prepare = prepare_protocol_fn or _prepare_protocol
    run_cell = run_cell_fn or _run_cell
    if not args.preflight and not args.smoke:
        # Keep this above preparation so no refactor can accidentally make a
        # loader or another benchmark-bearing check precede authorization.
        _require_final_approval(environ)
    _load_runtime_environment(environ)
    frozen, serving = prepare(args)
    if args.preflight:
        return _run_preflight(args, frozen)
    if args.smoke:
        return _run_smoke(
            args,
            frozen,
            serving,
            load_tasks_fn=load_smoke_tasks_fn or _load_smoke_tasks,
            run_cell_fn=run_cell,
        )
    return _run_final(
        args,
        frozen,
        serving,
        load_tasks_fn=load_final_tasks_fn or _load_official_test_tasks,
        run_cell_fn=run_cell,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    state = execute(args)
    # No reward or per-cell metric is emitted before post-run reporting.
    print(
        json.dumps(
            {
                "protocol_id": state["protocol_id"],
                "status": state["status"],
                "out": str(args.out.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
