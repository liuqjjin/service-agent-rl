"""The published GPU report must rebuild from the committed raw manifests."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from service_agent.eval.report_grpo import (
    REPO,
    _history_counter_aggregate,
    build,
    load_counter_audit,
    load_manifests,
    load_restore_manifest,
    validate_artifact_checksums,
    validate_counter_audit,
    validate_manifests,
    validate_restore_manifest,
)


def test_committed_grpo_report_is_current(tmp_path):
    report = Path(REPO / "reports/grpo_training.md")
    assert report.read_text() == build()

    backup = validate_artifact_checksums()
    _, _, formal = load_manifests()
    audit = load_counter_audit()
    validate_counter_audit(audit, backup, formal)
    doubled_again = deepcopy(audit)
    doubled_again["art_cumulative_state"]["groups_submitted"] *= 2
    with pytest.raises(RuntimeError, match="cumulative-state arithmetic"):
        validate_counter_audit(doubled_again, backup, formal)
    duplicate_step = deepcopy(audit)
    duplicate_step["record_sequences"]["rollout"][0] = 2
    with pytest.raises(RuntimeError, match="record sequences"):
        validate_counter_audit(duplicate_step, backup, formal)

    restore = load_restore_manifest()
    validate_restore_manifest(restore, backup)
    wrong_adapter = deepcopy(restore)
    wrong_adapter["adapter_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="restore proof adapter_sha256"):
        validate_restore_manifest(wrong_adapter, backup)
    wrong_gpu = deepcopy(restore)
    wrong_gpu["gpu"] = "NVIDIA A100"
    with pytest.raises(RuntimeError, match="restore proof gpu"):
        validate_restore_manifest(wrong_gpu, backup)
    invalid_time = deepcopy(restore)
    invalid_time["checked_at"] = "not-a-real-time+00:00"
    with pytest.raises(RuntimeError, match="valid UTC time"):
        validate_restore_manifest(invalid_time, backup)
    extra_field = deepcopy(restore)
    extra_field["claimed_gpu"] = "NVIDIA A100"
    with pytest.raises(RuntimeError, match="missing or extra fields"):
        validate_restore_manifest(extra_field, backup)

    overlapping_history = tmp_path / "history.jsonl"
    overlapping_history.write_text(
        json.dumps(
            {
                "data/step_num_gradient_steps": 0,
                "dev/dev/reward": 1.0,
                "step": 1,
                "training_step": 1,
            }
        )
        + "\n"
    )
    dummy_state = tmp_path / "state.json"
    dummy_state.write_text("{}\n")
    with pytest.raises(RuntimeError, match="exactly one record class"):
        _history_counter_aggregate(overlapping_history, dummy_state)


def test_grpo_manifest_progress_rebuilds():
    preflight, smoke, formal = load_manifests()
    validate_manifests(preflight, smoke, formal)

    broken = deepcopy(formal)
    broken["progress"]["gradient_steps"] += 1
    with pytest.raises(RuntimeError, match="cumulative progress"):
        validate_manifests(preflight, smoke, broken)

    metric_drift = deepcopy(formal)
    metric_drift["train_steps"][1]["metrics"]["data/step_num_gradient_steps"] = 999.0
    with pytest.raises(RuntimeError, match="metric data/step_num_gradient_steps"):
        validate_manifests(preflight, smoke, metric_drift)

    variance_drift = deepcopy(formal)
    variance_drift["train_steps"][1]["stats"]["mixed"] = 0
    variance_drift["train_steps"][1]["stats"]["all_one"] = 2
    with pytest.raises(RuntimeError, match="trainable groups do not equal mixed groups"):
        validate_manifests(preflight, smoke, variance_drift)

    negative_count = deepcopy(formal)
    negative_count["train_steps"][0]["stats"]["mixed"] = -1
    negative_count["train_steps"][0]["stats"]["all_one"] = 3
    with pytest.raises(RuntimeError, match="must be a nonnegative integer"):
        validate_manifests(preflight, smoke, negative_count)

    nan_reward = deepcopy(smoke)
    nan_reward["stats"]["reward_mean"] = float("nan")
    with pytest.raises(RuntimeError, match=r"finite and within \[0, 1\]"):
        validate_manifests(preflight, nan_reward, formal)

    invalid_dev_reward = deepcopy(formal)
    invalid_dev_reward["dev_evaluations"][0]["avg_reward"] = -5.0
    invalid_dev_reward["dev_evaluations"][0]["stats"]["reward_mean"] = -5.0
    with pytest.raises(RuntimeError, match=r"finite and within \[0, 1\]"):
        validate_manifests(preflight, smoke, invalid_dev_reward)

    bad_logprob = deepcopy(preflight)
    bad_logprob["logprob_gate"]["ratio_mean"] = 1.5
    with pytest.raises(RuntimeError, match="mean importance ratio exceeds"):
        validate_manifests(bad_logprob, smoke, formal)


def test_grpo_manifest_rejects_test_unlock_and_selection_drift():
    preflight, smoke, formal = load_manifests()

    unlocked = deepcopy(smoke)
    unlocked["test_split_locked"] = False
    with pytest.raises(RuntimeError, match="test split was unlocked"):
        validate_manifests(preflight, unlocked, formal)

    for field in ("api_key", "wandb_api_key", "hf_token"):
        leaked = deepcopy(preflight)
        leaked[field] = "must-never-be-committed"
        with pytest.raises(RuntimeError, match="private field"):
            validate_manifests(leaked, smoke, formal)

    query_secret = deepcopy(preflight)
    query_secret["wandb_url"] += "?api_key=opaque-secret-value"
    with pytest.raises(RuntimeError, match="credential-shaped value|clean wandb.ai"):
        validate_manifests(query_secret, smoke, formal)

    training_drift = [deepcopy(value) for value in (preflight, smoke, formal)]
    for manifest in training_drift:
        manifest["training"]["steps"] = 600
    with pytest.raises(RuntimeError, match="training contract drift"):
        validate_manifests(*training_drift)

    numeric_type_drift = [deepcopy(value) for value in (preflight, smoke, formal)]
    for manifest in numeric_type_drift:
        manifest["training"]["steps"] = 60.0
        manifest["runtime"]["seed"] = 42.0
    with pytest.raises(RuntimeError, match="training contract drift|runtime contract drift"):
        validate_manifests(*numeric_type_drift)

    snapshot_drift = [deepcopy(value) for value in (preflight, smoke, formal)]
    for manifest in snapshot_drift:
        manifest["model_snapshot"] = f"/tmp/{manifest['base_model_revision']}"
    with pytest.raises(RuntimeError, match="model snapshot path drift"):
        validate_manifests(*snapshot_drift)

    sampling_drift = deepcopy(smoke)
    sampling_drift["sampling"]["groups_per_formal_checkpoint_step"] = 1
    with pytest.raises(RuntimeError, match="groups-per-formal-step"):
        validate_manifests(preflight, sampling_drift, formal)

    formal_group_drift = deepcopy(formal)
    formal_group_drift["train_steps"][0]["groups_submitted"] = 1
    with pytest.raises(RuntimeError, match="submitted groups disagree with training contract"):
        validate_manifests(preflight, smoke, formal_group_drift)

    misselected = deepcopy(formal)
    misselected["selected_checkpoint"]["step"] = 20
    with pytest.raises(RuntimeError, match="selected step"):
        validate_manifests(preflight, smoke, misselected)

    unscheduled = deepcopy(formal)
    unscheduled["dev_evaluations"].append(
        {
            "step": 999,
            "avg_reward": 1.0,
            "rollouts": 40,
            "stats": {
                "all_one": 20,
                "all_zero": 0,
                "constant_other": 0,
                "groups": 20,
                "mixed": 0,
                "reward_mean": 1.0,
                "rollouts": 40,
            },
        }
    )
    with pytest.raises(RuntimeError, match="scheduled dev evaluation steps"):
        validate_manifests(preflight, smoke, unscheduled)

    fake_path = deepcopy(formal)
    fake_path["selected_checkpoint"]["checkpoint_path"] = "/fake/checkpoints/0015"
    with pytest.raises(RuntimeError, match="selected checkpoint path"):
        validate_manifests(preflight, smoke, fake_path)

    wrong_final_step = deepcopy(formal)
    wrong_final_step["train_steps"].pop()
    with pytest.raises(RuntimeError, match="unexpected formal terminal step"):
        validate_manifests(preflight, smoke, wrong_final_step)

    earlier_sparse = deepcopy(formal)
    step_seven = earlier_sparse["train_steps"][6]
    step_seven["gradient_steps"] = 0
    step_seven["gradient_work_performed"] = False
    step_seven["trainable_groups"] = 0
    step_seven["stats"]["all_one"] = 2
    step_seven["stats"]["mixed"] = 0
    step_seven["metrics"]["data/step_num_gradient_steps"] = 0.0
    step_seven["metrics"]["data/step_num_groups_trainable"] = 0.0
    earlier_sparse["progress"].update(
        {
            "gradient_steps": 349,
            "skipped_checkpoint_steps": 20,
            "trainable_checkpoint_steps": 4,
            "trainable_groups": 4,
        }
    )
    earlier_sparse["selected_checkpoint"]["training_progress"].update(
        {
            "gradient_steps": 253,
            "skipped_checkpoint_steps": 12,
            "trainable_checkpoint_steps": 3,
            "trainable_groups": 3,
        }
    )
    with pytest.raises(RuntimeError, match="continued after an earlier sparse"):
        validate_manifests(preflight, smoke, earlier_sparse)
