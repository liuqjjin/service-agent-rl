"""No-data tests for the approval-gated native final runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from service_agent.eval import run_final


class DummyTask:
    def __init__(self, task_id: str):
        self.id = task_id

    def model_dump(self, mode: str = "json"):
        assert mode == "json"
        return {"id": self.id}


def _args(
    tmp_path: Path,
    *,
    mode: str,
    out_name: str,
    smoke_manifest: Path | None = None,
) -> run_final.CliArgs:
    return run_final.CliArgs(
        out=tmp_path / out_name,
        serving_manifest=tmp_path / "serving.json",
        base_snapshot=tmp_path / run_final.BASE_MODEL_REVISION,
        adapter=tmp_path / "0015",
        smoke_manifest=smoke_manifest,
        preflight=mode == "preflight",
        smoke=mode == "smoke",
    )


def _frozen(repo_commit: str = "eval-commit") -> dict:
    cells = [run_final.asdict(cell) for cell in run_final.CELL_SPECS]
    return {
        "protocol_id": run_final.PROTOCOL_ID,
        "schema_version": run_final.PROTOCOL_SCHEMA_VERSION,
        "repo_commit": repo_commit,
        "art_commit": run_final.ART_COMMIT,
        "tau2_commit": run_final.TAU2_COMMIT,
        "serving_manifest_sha256": "a" * 64,
        "task_split": "test",
        "expected_task_count": run_final.EXPECTED_TASK_COUNT,
        "task_count": run_final.EXPECTED_TASK_COUNT,
        "expected_trial_count": run_final.EXPECTED_TRIAL_COUNT,
        "trials": run_final.EXPECTED_TRIAL_COUNT,
        "base_seed": run_final.BASE_SEED,
        "trial_seeds": list(run_final.FINAL_TRIAL_SEEDS),
        "max_steps": run_final.MAX_STEPS,
        "max_errors": run_final.MAX_ERRORS,
        "max_concurrency": run_final.MAX_CONCURRENCY,
        "policy_temperature": run_final.POLICY_TEMPERATURE,
        "policy_max_completion_tokens": run_final.MAX_COMPLETION_TOKENS,
        "policy_thinking": "disabled",
        "user_simulator": {
            "model": run_final.USER_MODEL,
            "temperature": run_final.USER_TEMPERATURE,
            "thinking": "disabled",
        },
        "evaluation_type": "all",
        "cell_order": list(run_final.CELL_ORDER),
        "cells": cells,
        "task_set_sha256_algorithm": "sha256(canonical-json(sorted-task-ids))",
        "gpu": {
            "count": 1,
            "name": run_final.EXPECTED_GPU_NAME,
            "uuid": "GPU-test",
            "driver_version": "test",
            "memory_total_mib": 97_887,
        },
        "serving_process": {
            "pid": 123,
            "start_time_ticks": 777,
            "boot_id": "boot-test",
            "match_kind": "python_console_script",
            "expected_command_sha256": "b" * 64,
            "observed_argv_sha256": "c" * 64,
        },
        "serving": {
            "api_base": "http://127.0.0.1:8100/v1",
            "base_model_alias": run_final.BASE_ALIAS,
            "rl_model_alias": run_final.RL_ALIAS,
        },
        "training_evidence": {"selected_checkpoint": run_final.SELECTED_CHECKPOINT},
    }


def _prepare(frozen: dict):
    def prepare(_args):
        return frozen, {"api_base": "http://127.0.0.1:8100/v1"}

    return prepare


def _environment(*, approved: bool = False) -> dict[str, str]:
    values = {"DEEPSEEK_API_KEY": "present-but-never-persisted"}
    if approved:
        values[run_final.APPROVAL_ENV] = run_final.APPROVAL_VALUE
    return values


def _fake_cell_runner(calls: list[tuple[str, int, str]]):
    def run_cell(cell, **kwargs):
        calls.append((cell.name, kwargs["trials"], kwargs["mode"]))
        cell_dir = kwargs["out"] / cell.name
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / run_final.CELL_CONFIG).write_text("{}\n")
        (cell_dir / run_final.CELL_RESULTS).write_text(
            json.dumps({"cell": cell.name, "mode": kwargs["mode"]}) + "\n"
        )
        (cell_dir / run_final.CELL_LOG).write_text("")
        if cell.harness == "h2":
            (cell_dir / "audit").mkdir(exist_ok=True)

    return run_cell


def test_frozen_cell_mapping_and_trial_seeds():
    assert run_final.CELL_ORDER == ("base_h0", "base_h2", "rl_h0", "rl_h2")
    assert [
        (cell.model_row, cell.harness, cell.agent, cell.served_model_alias)
        for cell in run_final.CELL_SPECS
    ] == [
        ("base", "h0", "llm_agent", run_final.BASE_ALIAS),
        ("base", "h2", "governed_llm_agent_h2", run_final.BASE_ALIAS),
        ("rl", "h0", "llm_agent", run_final.RL_ALIAS),
        ("rl", "h2", "governed_llm_agent_h2", run_final.RL_ALIAS),
    ]
    assert run_final.FINAL_TRIAL_SEEDS == (
        670487,
        116739,
        26225,
        777572,
        288389,
        256787,
        234053,
        146316,
    )


def test_cli_has_no_experiment_tuning_options():
    options = {
        option
        for action in run_final._parser()._actions
        for option in action.option_strings
    }
    prohibited = {
        "--split",
        "--tasks",
        "--arm",
        "--trials",
        "--seed",
        "--temperature",
        "--model",
        "--max-steps",
        "--max-concurrency",
    }
    assert options.isdisjoint(prohibited)
    with pytest.raises(SystemExit):
        run_final.parse_args(
            [
                "--out",
                "/tmp/out",
                "--serving-manifest",
                "/tmp/serving",
                "--base-snapshot",
                "/tmp/base",
                "--adapter",
                "/tmp/adapter",
                "--split",
                "test",
            ]
        )


@pytest.mark.parametrize("value", [None, "", "FINAL_TEST_APPROVE", " final_TEST_APPROVED"])
def test_exact_approval_precedes_preparation_and_test_loader(tmp_path: Path, value: str | None):
    calls: list[str] = []
    args = _args(
        tmp_path,
        mode="final",
        out_name="final",
        smoke_manifest=tmp_path / "smoke.json",
    )
    environ = _environment()
    if value is not None:
        environ[run_final.APPROVAL_ENV] = value

    with pytest.raises(PermissionError):
        run_final.execute(
            args,
            prepare_protocol_fn=lambda _: calls.append("prepare"),
            load_final_tasks_fn=lambda: calls.append("load-test"),
            run_cell_fn=lambda *args, **kwargs: calls.append("run"),
            environ=environ,
        )

    assert calls == []


def test_preflight_never_loads_any_tasks(tmp_path: Path):
    args = _args(tmp_path, mode="preflight", out_name="preflight")
    frozen = _frozen()

    state = run_final.execute(
        args,
        prepare_protocol_fn=_prepare(frozen),
        load_smoke_tasks_fn=lambda: pytest.fail("preflight loaded dev tasks"),
        load_final_tasks_fn=lambda: pytest.fail("preflight loaded test tasks"),
        run_cell_fn=lambda *args, **kwargs: pytest.fail("preflight ran a cell"),
        environ=_environment(),
    )

    assert state["status"] == "preflight_passed"
    assert state["approval"] == "not_consumed"
    assert state["completed_cells"] == []
    assert (args.out / run_final.FINAL_MANIFEST).is_file()


def test_preflight_requires_empty_or_exact_matching_output(tmp_path: Path):
    args = _args(tmp_path, mode="preflight", out_name="preflight")
    args.out.mkdir()
    (args.out / "unexpected.txt").write_text("not protocol state")
    with pytest.raises(RuntimeError, match="unexpected files"):
        run_final.execute(
            args,
            prepare_protocol_fn=_prepare(_frozen()),
            environ=_environment(),
        )

    (args.out / "unexpected.txt").unlink()
    run_final.execute(
        args,
        prepare_protocol_fn=_prepare(_frozen()),
        environ=_environment(),
    )
    with pytest.raises(RuntimeError, match="protocol hash drifted"):
        run_final.execute(
            args,
            prepare_protocol_fn=_prepare(_frozen(repo_commit="changed")),
            environ=_environment(),
        )


def test_smoke_is_four_fixed_cells_on_one_dev_trial(tmp_path: Path):
    args = _args(tmp_path, mode="smoke", out_name="smoke")
    calls: list[tuple[str, int, str]] = []
    load_count = 0

    def load_tasks():
        nonlocal load_count
        load_count += 1
        return [DummyTask("dev-a"), DummyTask("dev-b"), DummyTask("dev-c")]

    state = run_final.execute(
        args,
        prepare_protocol_fn=_prepare(_frozen()),
        load_smoke_tasks_fn=load_tasks,
        run_cell_fn=_fake_cell_runner(calls),
        environ=_environment(),
    )

    assert load_count == 1
    assert calls == [(cell, 1, "smoke") for cell in run_final.CELL_ORDER]
    assert state["status"] == "smoke_complete"
    assert state["approval"] == "not_consumed"
    assert state["task_split"] == "frozen_dev_smoke3"
    assert state["task_count"] == 3
    assert state["trials"] == 1
    assert state["trial_seeds"] == [670487]
    assert set(state["cell_result_sha256"]) == set(run_final.CELL_ORDER)


def test_approved_final_loads_once_and_runs_frozen_order(tmp_path: Path):
    frozen = _frozen()
    preflight_args = _args(tmp_path, mode="preflight", out_name="final")
    run_final.execute(
        preflight_args,
        prepare_protocol_fn=_prepare(frozen),
        environ=_environment(),
    )

    smoke_args = _args(tmp_path, mode="smoke", out_name="smoke")
    run_final.execute(
        smoke_args,
        prepare_protocol_fn=_prepare(frozen),
        load_smoke_tasks_fn=lambda: [
            DummyTask("dev-a"),
            DummyTask("dev-b"),
            DummyTask("dev-c"),
        ],
        run_cell_fn=_fake_cell_runner([]),
        environ=_environment(),
    )
    smoke_manifest = smoke_args.out / run_final.SMOKE_MANIFEST

    final_args = replace(
        preflight_args,
        preflight=False,
        smoke_manifest=smoke_manifest,
    )
    calls: list[tuple[str, int, str]] = []
    load_count = 0

    def load_final():
        nonlocal load_count
        load_count += 1
        return [DummyTask(f"official-{index:02d}") for index in range(40)]

    state = run_final.execute(
        final_args,
        prepare_protocol_fn=_prepare(frozen),
        load_final_tasks_fn=load_final,
        run_cell_fn=_fake_cell_runner(calls),
        environ=_environment(approved=True),
    )

    assert load_count == 1
    assert calls == [(cell, 8, "final") for cell in run_final.CELL_ORDER]
    assert state["status"] == "complete"
    assert state["approval"] == run_final.APPROVAL_VALUE
    assert state["completed_cells"] == list(run_final.CELL_ORDER)
    assert set(state["cell_result_sha256"]) == set(run_final.CELL_ORDER)
    assert state["native_smoke_manifest_sha256"] == run_final._sha256_file(smoke_manifest)
    serialized = json.dumps(state)
    assert "present-but-never-persisted" not in serialized

    # A completed invocation validates persisted hashes and returns without
    # loading the official split a second time.
    state_again = run_final.execute(
        final_args,
        prepare_protocol_fn=_prepare(frozen),
        load_final_tasks_fn=lambda: pytest.fail("completed final run reloaded test"),
        run_cell_fn=lambda *args, **kwargs: pytest.fail("completed final reran a cell"),
        environ=_environment(approved=True),
    )
    assert state_again["status"] == "complete"

    unexpected = final_args.out / "base_h0/unexpected.txt"
    unexpected.write_text("drift")
    with pytest.raises(RuntimeError, match="unexpected files"):
        run_final.execute(
            final_args,
            prepare_protocol_fn=_prepare(frozen),
            load_final_tasks_fn=lambda: pytest.fail("layout check reloaded test"),
            environ=_environment(approved=True),
        )
    unexpected.unlink()

    results_path = final_args.out / "base_h0" / run_final.CELL_RESULTS
    results_path.write_text('{"tampered":true}\n')
    with pytest.raises(RuntimeError, match="results hash drifted"):
        run_final.execute(
            final_args,
            prepare_protocol_fn=_prepare(frozen),
            load_final_tasks_fn=lambda: pytest.fail("hash check reloaded test"),
            environ=_environment(approved=True),
        )


def test_native_config_freezes_every_cell_input():
    for mode, split, trials in (
        ("smoke", "frozen_dev_smoke3", 1),
        ("final", "test", 8),
    ):
        for cell in run_final.CELL_SPECS:
            config = run_final._build_run_config(
                cell,
                "http://127.0.0.1:8100/v1",
                trials=trials,
                task_split_name=split,
            )
            assert config.task_set_name == "telecom"
            assert config.task_split_name == split
            assert config.agent == cell.agent
            assert config.llm_agent == f"openai/{cell.served_model_alias}"
            assert config.num_trials == trials
            assert config.seed == 42
            assert config.effective_max_steps == 100
            assert config.max_errors == 10
            assert config.max_concurrency == 3
            assert config.max_retries == 0
            assert config.llm_args_agent == run_final._agent_llm_args(
                "http://127.0.0.1:8100/v1"
            )
            assert config.llm_args_user == run_final._user_llm_args()
            assert mode in {"smoke", "final"}


def test_resume_validation_rejects_bad_used_records_and_tracks_infrastructure(
    tmp_path: Path, monkeypatch
):
    from tau2.data_model.simulation import Results

    path = tmp_path / "results.json"
    path.write_text("{}")
    tasks = [DummyTask("t1")]
    valid = SimpleNamespace(
        trial=0,
        task_id="t1",
        seed=670487,
        termination_reason="user_stop",
        reward_info=SimpleNamespace(reward=1.0),
        messages=[SimpleNamespace(role="assistant")],
        info={},
    )
    payload = SimpleNamespace(tasks=tasks, simulations=[valid])
    monkeypatch.setattr(Results, "load", lambda _path: payload)
    complete, infrastructure = run_final._validate_existing_results(
        path,
        tasks=tasks,
        trials=1,
    )
    assert complete
    assert infrastructure == []

    valid.reward_info = None
    with pytest.raises(RuntimeError, match="has no reward"):
        run_final._validate_existing_results(path, tasks=tasks, trials=1)

    valid.reward_info = SimpleNamespace(reward=1.0)
    valid.messages = []
    with pytest.raises(RuntimeError, match="has no messages"):
        run_final._validate_existing_results(path, tasks=tasks, trials=1)

    valid.termination_reason = "infrastructure_error"
    valid.info = {"error_type": "ConnectionError"}
    complete, infrastructure = run_final._validate_existing_results(
        path,
        tasks=tasks,
        trials=1,
    )
    assert not complete
    assert infrastructure == [
        {
            "trial": 0,
            "task_id": "t1",
            "seed": 670487,
            "error_type": "ConnectionError",
        }
    ]


def test_task_digest_is_sorted_ids_only():
    left = [DummyTask("b"), DummyTask("a")]
    right = [DummyTask("a"), DummyTask("b")]
    assert run_final._task_set_sha256(left) == run_final._task_set_sha256(right)
    assert run_final._task_set_sha256(left) == run_final._canonical_sha256(["a", "b"])


def test_smoke_loader_requests_only_upstream_train(monkeypatch):
    from tau2.registry import registry

    from service_agent.splits import load_frozen_dev_ids

    splits: list[str] = []
    tasks = [DummyTask(task_id) for task_id in load_frozen_dev_ids()]

    def task_loader(*, task_split_name):
        splits.append(task_split_name)
        return tasks

    monkeypatch.setattr(registry, "get_tasks_loader", lambda domain: task_loader)
    selected = run_final._load_smoke_tasks()

    assert splits == ["train"]
    assert len(selected) == 3


def test_gpu_provenance_requires_one_exact_pro_6000(monkeypatch):
    completed = SimpleNamespace(
        stdout=(
            "NVIDIA RTX PRO 6000 Blackwell Server Edition, "
            "GPU-1234, 590.48.01, 97887\n"
        )
    )
    monkeypatch.setattr(run_final.subprocess, "run", lambda *args, **kwargs: completed)
    assert run_final._gpu_provenance() == {
        "count": 1,
        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "uuid": "GPU-1234",
        "driver_version": "590.48.01",
        "memory_total_mib": 97887,
    }

    completed.stdout = "NVIDIA RTX 4090, GPU-1234, 590.48.01, 24564\n"
    with pytest.raises(RuntimeError, match="GPU drift"):
        run_final._gpu_provenance()


def test_serving_process_provenance_accepts_console_script_and_rejects_duplicates(
    tmp_path: Path,
):
    proc_root = tmp_path / "proc"
    boot_id_path = proc_root / "sys/kernel/random/boot_id"
    boot_id_path.parent.mkdir(parents=True)
    boot_id_path.write_text("boot-123\n")
    command = [str(tmp_path / "runtime-server"), "--model=/snapshot", "--port=8100"]
    observed = [str(tmp_path / "python"), *command]

    def write_process(pid: int):
        process = proc_root / str(pid)
        process.mkdir()
        (process / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in observed) + b"\0")
        fields = ["S", *(["0"] * 18), "777"]
        (process / "stat").write_text(f"{pid} (runtime server) {' '.join(fields)}\n")

    write_process(123)
    evidence = run_final._serving_process_provenance(command, proc_root=proc_root)
    assert evidence == {
        "pid": 123,
        "start_time_ticks": 777,
        "boot_id": "boot-123",
        "match_kind": "python_console_script",
        "expected_command_sha256": run_final._canonical_sha256(command),
        "observed_argv_sha256": run_final._canonical_sha256(observed),
    }

    write_process(456)
    with pytest.raises(RuntimeError, match="expected one exact final serving process; found 2"):
        run_final._serving_process_provenance(command, proc_root=proc_root)
