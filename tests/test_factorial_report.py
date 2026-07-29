"""Public final-report generation on a fully synthetic four-cell campaign."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo

from service_agent.eval import run_final
from service_agent.eval.factorial import FACTORIAL_CELLS, trial_seeds
from service_agent.eval.final_serving import (
    ADAPTER_SHA256,
    BASE_ALIAS,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHAT_TEMPLATE_KWARGS,
    CHAT_TEMPLATE_SHA256,
    FROZEN_RUNTIME_PACKAGES,
    MAX_COMPLETION_TOKENS,
    MAX_MODEL_LEN,
    RL_ALIAS,
    SELECTED_CHECKPOINT,
    SERVING_HOST,
    SERVING_PORT,
    SYSTEM_PROMPT_SHA256,
    TOOL_CALL_PARSER,
    TOOLS_SHA256,
    TRAINING_REPO_COMMIT,
    VLLM_BOOTSTRAP_SHA256,
    VLLM_CUDART_SHA256,
    VLLM_NINJA_SHA256,
    _engine_args,
    _safe_environment_overrides,
    _server_args,
    build_vllm_command,
)
from service_agent.eval.report_factorial import check, generate, main
from service_agent.training.contracts import ART_COMMIT, TAU2_COMMIT

TASK_COUNT = 40
TRIAL_COUNT = 8
REPO_COMMIT = "a" * 40


def _serving_manifest(adapter_path: str) -> dict:
    snapshot = (
        "/synthetic/cache/models--Qwen--Qwen3.5-4B/snapshots/"
        f"{BASE_MODEL_REVISION}"
    )
    repo_root = "/synthetic/repo"
    runtime_server = "/synthetic/runtime/bin/art-vllm-runtime-server"
    api_base = f"http://{SERVING_HOST}:{SERVING_PORT}/v1"
    probe_result = {
        "status": "passed",
        "finish_reason": "tool_calls",
        "tool_call_count": 1,
        "tool_name": "health_probe",
        "arguments": {"status": "ready"},
        "reasoning_content_absent": True,
        "thinking_tags_absent": True,
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "api_base": api_base,
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_alias": BASE_ALIAS,
        "rl_model_alias": RL_ALIAS,
        "base_snapshot": snapshot,
        "adapter_path": adapter_path,
        "adapter_checkpoint": SELECTED_CHECKPOINT,
        "adapter_sha256": ADAPTER_SHA256,
        "tokenizer_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "vllm_bootstrap_sha256": VLLM_BOOTSTRAP_SHA256,
        "vllm_ninja_sha256": VLLM_NINJA_SHA256,
        "vllm_cudart_sha256": VLLM_CUDART_SHA256,
        "snapshot_tree_sha256": "4" * 64,
        "snapshot_file_count": 10,
        "snapshot_total_bytes": 1_000,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "tool_call_parser": TOOL_CALL_PARSER,
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        "semantic_input_hashes": {
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "tokenizer_chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "tools_sha256": TOOLS_SHA256,
        },
        "training_repo_commit": TRAINING_REPO_COMMIT,
        "art_commit": ART_COMMIT,
        "tau2_commit": TAU2_COMMIT,
        "runtime_packages": dict(FROZEN_RUNTIME_PACKAGES),
        "serving_repo_root": repo_root,
        "runtime_server": runtime_server,
        "engine_args": _engine_args(),
        "server_args": _server_args(adapter_path),
        "command": build_vllm_command(
            snapshot,
            adapter_path,
            repo_root=repo_root,
            runtime_server=runtime_server,
        ),
        "environment_overrides": _safe_environment_overrides(
            snapshot,
            repo_root=repo_root,
            runtime_server=runtime_server,
            hf_home="/synthetic/cache",
        ),
        "probe": {
            "schema_version": 1,
            "status": "passed",
            "checked_at": "2026-07-29T00:00:00+00:00",
            "api_base": api_base,
            "vllm_api_version": "0.23.0",
            "capabilities": {
                "runtime": "art_vllm",
                "protocol_version": 1,
                "inplace_lora_load": True,
                "policy_token_spans": True,
            },
            "model_cards": {
                BASE_ALIAS: {
                    "root": snapshot,
                    "parent": None,
                    "max_model_len": MAX_MODEL_LEN,
                },
                RL_ALIAS: {
                    "root": adapter_path,
                    "parent": BASE_ALIAS,
                },
            },
            "probes": {
                BASE_ALIAS: {**probe_result, "response_model": BASE_ALIAS},
                RL_ALIAS: {**probe_result, "response_model": RL_ALIAS},
            },
            "tool_call_parser": TOOL_CALL_PARSER,
            "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
            "benchmark_data_accessed": False,
        },
    }


def _info(cell: str) -> Info:
    alias = BASE_ALIAS if cell.startswith("base_") else RL_ALIAS
    implementation = "llm_agent" if cell.endswith("_h0") else "governed_llm_agent_h2"
    return Info(
        git_commit=REPO_COMMIT,
        num_trials=TRIAL_COUNT,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(
            implementation="user_simulator",
            llm="deepseek/deepseek-v4-pro",
            llm_args={
                "temperature": 0.0,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
        ),
        agent_info=AgentInfo(
            implementation=implementation,
            llm=f"openai/{alias}",
            llm_args={
                "temperature": 0.0,
                "max_tokens": 1_024,
                "api_base": f"http://{SERVING_HOST}:{SERVING_PORT}/v1",
                "api_key": "local",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        ),
        environment_info=EnvironmentInfo(
            domain_name="telecom",
            policy="synthetic fixed policy",
        ),
        seed=42,
    )


def _reward(cell: str, task: int, trial: int) -> float:
    # Non-constant task effects make the paired bootstrap exercise real draws.
    base = ((task * 3 + trial) % 8) / 7
    shifts = {
        "base_h0": 0.00,
        "base_h2": 0.05,
        "rl_h0": 0.10,
        "rl_h2": 0.20,
    }
    return min(1.0, base + shifts[cell])


def _cold_write_messages() -> list:
    return [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="cold-write",
                    name="resume_line",
                    arguments={"customer_id": "C1", "line_id": "L1"},
                )
            ],
        ),
        ToolMessage(
            id="cold-write",
            role="tool",
            requestor="assistant",
            content='{"message":"resumed","line":{"line_id":"L1"}}',
        ),
    ]


def _results(cell: str) -> Results:
    seeds = trial_seeds(42, TRIAL_COUNT)
    simulations = []
    for task in range(TASK_COUNT):
        for trial in range(TRIAL_COUNT):
            reward = _reward(cell, task, trial)
            messages = (
                _cold_write_messages()
                if cell == "base_h0" and task == 0 and trial == 7
                else []
            )
            simulations.append(
                SimulationRun(
                    id=f"{cell}-{task}-{trial}",
                    task_id=f"task-{task:02d}",
                    timestamp="2026-07-29T00:00:00Z",
                    start_time="2026-07-29T00:00:00Z",
                    end_time="2026-07-29T00:00:01Z",
                    duration=float(1 + task % 3),
                    termination_reason=TerminationReason.AGENT_STOP,
                    agent_cost=None if task == 0 and trial == 0 else 0.0,
                    user_cost=0.01,
                    reward_info=RewardInfo(
                        reward=reward,
                        reward_basis=[],
                        reward_breakdown={},
                    ),
                    messages=messages,
                    trial=trial,
                    seed=seeds[trial],
                )
            )
    return Results(
        timestamp="2026-07-29T00:00:00Z",
        info=_info(cell),
        tasks=[
            Task(
                id=f"task-{task:02d}",
                user_scenario=UserScenario(instructions="private synthetic label"),
                evaluation_criteria=EvaluationCriteria(reward_basis=[]),
            )
            for task in range(TASK_COUNT)
        ],
        simulations=simulations,
    )


def _audit_record(cell: str) -> dict:
    rejected = cell == "rl_h2"
    return {
        "session_id": "b" * 32 if rejected else "a" * 32,
        "task_id": "task-00",
        "attempt": 1 if rejected else 0,
        "tool_name": "resume_line",
        "arguments_json": '{"customer_id":"C1","line_id":"L1"}',
        "decision": "require_evidence" if rejected else "allow",
        "reason_code": "line_not_read" if rejected else "preconditions_met",
        "policy_ref": "main_policy.md:1",
        "tool_call_id": "call-1",
    }


def _write_campaign(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    serving_path = root / "serving_manifest.json"
    serving = _serving_manifest("/synthetic/checkpoints/0015")
    serving_path.write_text(json.dumps(serving, indent=2) + "\n")
    serving_digest = hashlib.sha256(serving_path.read_bytes()).hexdigest()
    serving_process = {
        "pid": 123,
        "start_time_ticks": 456,
        "boot_id": "synthetic-boot",
        "match_kind": "direct",
        "expected_command_sha256": "8" * 64,
        "observed_argv_sha256": "9" * 64,
    }
    public_protocol = run_final._protocol_public_fields(
        {
            "repo_commit": REPO_COMMIT,
            "art_commit": ART_COMMIT,
            "tau2_commit": TAU2_COMMIT,
        },
        serving,
        serving_digest,
        {
            "count": 1,
            "name": run_final.EXPECTED_GPU_NAME,
            "uuid": "GPU-synthetic",
            "driver_version": "synthetic",
            "memory_total_mib": 97_887,
        },
        serving_process,
    )
    frozen = {
        **public_protocol,
        "training_evidence": {
            "training_repo_commit": TRAINING_REPO_COMMIT,
            "formal_manifest_sha256": "5" * 64,
            "restore_manifest_sha256": "6" * 64,
            "selected_checkpoint": 15,
            "selected_adapter_sha256": ADAPTER_SHA256,
            "semantic_contract_sha256": "7" * 64,
            "semantic_input_hashes": serving["semantic_input_hashes"],
        },
        "serving_manifest_path": str(serving_path),
    }
    protocol_path = root / "raw/final_manifest.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)

    protocol = run_final._new_state(
        frozen,
        status="complete",
        approval="FINAL_TEST_APPROVED",
    )
    protocol_digest = protocol["protocol_sha256"]
    cell_specs = {cell.name: cell for cell in run_final.CELL_SPECS}
    for cell in FACTORIAL_CELLS:
        cell_root = root / "raw" / cell
        _results(cell).save(cell_root / "results.json")
        text_config = run_final._build_run_config(
            cell_specs[cell],
            f"http://{SERVING_HOST}:{SERVING_PORT}/v1",
            trials=TRIAL_COUNT,
            task_split_name="test",
        )
        (cell_root / "run_config.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": "final",
                    "protocol_sha256": protocol_digest,
                    "cell": public_protocol["cells"][
                        list(FACTORIAL_CELLS).index(cell)
                    ],
                    "text_run_config": json.loads(text_config.model_dump_json()),
                },
                indent=2,
            )
            + "\n"
        )
        (cell_root / "runner.log").write_text(f"{cell} completed\n")
        if cell.endswith("_h2"):
            audit_dir = cell_root / "audit"
            audit_dir.mkdir(exist_ok=True)
            session = "b" * 32 if cell == "rl_h2" else "a" * 32
            (audit_dir / f"audit_{session}.jsonl").write_text(
                json.dumps(_audit_record(cell)) + "\n"
            )
    task_ids = [f"task-{task:02d}" for task in range(TASK_COUNT)]
    protocol.update(
        {
            "completed_cells": list(FACTORIAL_CELLS),
            "task_set_sha256": run_final._canonical_sha256(sorted(task_ids)),
            "native_smoke_manifest_sha256": "2" * 64,
            "infrastructure_retries": [],
            "cell_result_sha256": {
                cell: hashlib.sha256(
                    (root / "raw" / cell / "results.json").read_bytes()
                ).hexdigest()
                for cell in FACTORIAL_CELLS
            },
        },
    )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol_path, serving_path


def _generate_synthetic(root: Path) -> tuple[Path, Path, Path, Path]:
    protocol, serving = _write_campaign(root)
    public = root / "public"
    report = root / "factorial_results.md"
    generate(
        raw_root=root / "raw",
        protocol_manifest=protocol,
        serving_manifest=serving,
        public_root=public,
        report_path=report,
    )
    return protocol, serving, public, report


def _rewrite_public_checksum(public: Path, relative: str, target: Path) -> None:
    checksum_path = public / "CHECKSUMS.sha256"
    entries = {}
    for line in checksum_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    entries[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    checksum_path.write_text(
        "".join(f"{entries[name]}  {name}\n" for name in sorted(entries))
    )


def test_final_report_generates_compact_reproducible_public_evidence(
    tmp_path: Path,
    monkeypatch,
):
    protocol, serving, public, report = _generate_synthetic(tmp_path)

    check(public_root=public, report_path=report)
    check(
        public_root=public,
        report_path=report,
        raw_root=tmp_path / "raw",
        protocol_manifest=protocol,
        serving_manifest=serving,
    )
    assert {path.name for path in public.iterdir()} == {
        "protocol.json",
        "outcomes.json",
        "pairing_manifest.json",
        "analysis.json",
        "live_audit.json",
        "RAW_SHA256SUMS",
        "CHECKSUMS.sha256",
    }
    outcomes = json.loads((public / "outcomes.json").read_text())
    assert len(outcomes["cells"]["base_h0"]) == 320
    assert outcomes["cells"]["base_h0"][7]["unauthorized_executed_writes"] == 1
    serialized = json.dumps(outcomes)
    assert "messages" not in serialized
    assert "evaluation_criteria" not in serialized
    assert "synthetic fixed policy" not in serialized
    assert "api_key" not in serialized
    analysis = json.loads((public / "analysis.json").read_text())
    assert analysis["statistics"]["resamples"] == 10_000
    assert set(analysis["contrasts"]) == {
        "harness_effect_base",
        "harness_effect_rl",
        "model_effect_native",
        "model_effect_governed",
        "combined_gain",
        "interaction",
    }
    live_audit = json.loads((public / "live_audit.json").read_text())
    assert live_audit["cells"]["base_h2"]["sessions_with_audit"] == 1
    assert live_audit["cells"]["base_h2"]["records"] == 1
    assert "# Final 2x2 factorial results" in report.read_text()
    assert "## Protocol deviation" in report.read_text()
    assert "no test episode ran" in report.read_text()
    raw_index = (public / "RAW_SHA256SUMS").read_text()
    assert "base_h0/runner.log" in raw_index
    assert f"base_h2/audit/audit_{'a' * 32}.jsonl" in raw_index

    before = {
        path.name: path.read_bytes() for path in public.iterdir() if path.is_file()
    }
    before_report = report.read_bytes()
    generate(
        raw_root=tmp_path / "raw",
        protocol_manifest=protocol,
        serving_manifest=serving,
        public_root=public,
        report_path=report,
    )
    assert before == {
        path.name: path.read_bytes() for path in public.iterdir() if path.is_file()
    }
    assert report.read_bytes() == before_report
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_factorial",
            "--check",
            "--public-root",
            str(public),
            "--report",
            str(report),
        ],
    )
    main()


def test_final_report_rejects_missing_cell_and_mismatched_info(tmp_path: Path):
    protocol, serving = _write_campaign(tmp_path)
    with pytest.raises(RuntimeError, match="exactly 10000 resamples"):
        generate(
            raw_root=tmp_path / "raw",
            protocol_manifest=protocol,
            serving_manifest=serving,
            public_root=tmp_path / "public",
            report_path=tmp_path / "report.md",
            resamples=1,
        )
    (tmp_path / "raw/rl_h2/results.json").unlink()
    with pytest.raises(RuntimeError, match="rl_h2 raw directory"):
        generate(
            raw_root=tmp_path / "raw",
            protocol_manifest=protocol,
            serving_manifest=serving,
            public_root=tmp_path / "public",
            report_path=tmp_path / "report.md",
        )

    _results("rl_h2").save(tmp_path / "raw/rl_h2/results.json")
    results = Results.load(tmp_path / "raw/rl_h2/results.json")
    results.info.user_info.llm_args["temperature"] = 0.5
    results.save(tmp_path / "raw/rl_h2/results.json")
    manifest = json.loads(protocol.read_text())
    manifest["cell_result_sha256"]["rl_h2"] = hashlib.sha256(
        (tmp_path / "raw/rl_h2/results.json").read_bytes()
    ).hexdigest()
    protocol.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="common fields differ"):
        generate(
            raw_root=tmp_path / "raw",
            protocol_manifest=protocol,
            serving_manifest=serving,
            public_root=tmp_path / "public",
            report_path=tmp_path / "report.md",
        )


def test_public_check_rejects_checksum_private_field_and_stale_report(tmp_path: Path):
    _, _, public, report = _generate_synthetic(tmp_path)
    outcomes_path = public / "outcomes.json"
    outcomes_path.write_text(outcomes_path.read_text() + " ")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        check(public_root=public, report_path=report)

    # Restore a valid package, then synchronize its checksum so the semantic
    # private-field gate—not the outer hash—has to reject the mutation.
    _, _, public, report = _generate_synthetic(tmp_path)
    outcomes = json.loads(outcomes_path.read_text())
    outcomes["sk-ABCDEFGHIJKLMNOPQRSTUVWX"] = "leaked-key"
    outcomes_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    _rewrite_public_checksum(public, "outcomes.json", outcomes_path)
    with pytest.raises(RuntimeError, match="credential-like key"):
        check(public_root=public, report_path=report)

    _, _, public, report = _generate_synthetic(tmp_path)
    outcomes = json.loads(outcomes_path.read_text())
    outcomes["messages"] = ["private"]
    outcomes_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    _rewrite_public_checksum(public, "outcomes.json", outcomes_path)
    with pytest.raises(RuntimeError, match="private field"):
        check(public_root=public, report_path=report)

    _, _, public, report = _generate_synthetic(tmp_path)
    report.write_text(report.read_text() + "\nstale\n")
    _rewrite_public_checksum(public, "report/factorial_results.md", report)
    with pytest.raises(RuntimeError, match="report is stale"):
        check(public_root=public, report_path=report)


def test_optional_raw_verification_detects_backup_drift(tmp_path: Path):
    protocol, serving, public, report = _generate_synthetic(tmp_path)
    (tmp_path / "raw/base_h0/runner.log").write_text("mutated after publication\n")

    with pytest.raises(RuntimeError, match="raw input checksum index differs"):
        check(
            public_root=public,
            report_path=report,
            raw_root=tmp_path / "raw",
            protocol_manifest=protocol,
            serving_manifest=serving,
        )


def test_report_rejects_extra_public_tree_audit_secret_and_uniform_config_drift(
    tmp_path: Path,
):
    extra_root = tmp_path / "extra"
    _, _, public, report = _generate_synthetic(extra_root)
    leaked_dir = public / "private-dump"
    leaked_dir.mkdir()
    (leaked_dir / "secret.txt").write_text("must not be ignored\n")
    with pytest.raises(RuntimeError, match="missing or extra"):
        check(public_root=public, report_path=report)

    audit_root = tmp_path / "audit-secret"
    protocol, serving = _write_campaign(audit_root)
    audit_path = audit_root / "raw/base_h2/audit" / f"audit_{'a' * 32}.jsonl"
    record = json.loads(audit_path.read_text())
    record["reason_code"] = "sk-ABCDEFGHIJKLMNOPQRSTUVWX"
    audit_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(RuntimeError, match="reason_code is unsafe"):
        generate(
            raw_root=audit_root / "raw",
            protocol_manifest=protocol,
            serving_manifest=serving,
            public_root=audit_root / "public",
            report_path=audit_root / "report.md",
        )

    empty_audit_root = tmp_path / "empty-audit"
    protocol, serving = _write_campaign(empty_audit_root)
    (
        empty_audit_root
        / "raw/base_h2/audit"
        / f"audit_{'c' * 32}.jsonl"
    ).write_text("")
    with pytest.raises(RuntimeError, match="audit file is empty"):
        generate(
            raw_root=empty_audit_root / "raw",
            protocol_manifest=protocol,
            serving_manifest=serving,
            public_root=empty_audit_root / "public",
            report_path=empty_audit_root / "report.md",
        )

    config_root = tmp_path / "config-drift"
    protocol, serving = _write_campaign(config_root)
    for cell in FACTORIAL_CELLS:
        path = config_root / "raw" / cell / "run_config.json"
        payload = json.loads(path.read_text())
        payload["text_run_config"]["task_set_name"] = "wrong-set"
        payload["text_run_config"]["auto_review"] = True
        path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="TextRunConfig differs"):
        generate(
            raw_root=config_root / "raw",
            protocol_manifest=protocol,
            serving_manifest=serving,
            public_root=config_root / "public",
            report_path=config_root / "report.md",
        )
