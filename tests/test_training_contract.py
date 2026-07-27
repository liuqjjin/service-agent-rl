"""GPU-training contracts that must fail locally before money is spent."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from service_agent.training.art_tau_train import group_stats
from service_agent.training.contracts import (
    ART_COMMIT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHAT_TEMPLATE_KWARGS,
    MANIFEST_SCHEMA_VERSION,
    TAU2_COMMIT,
    RuntimeConfig,
    assert_pinned_art_api,
    build_internal_model_config,
    build_trainable_model_kwargs,
    validate_matching_protocol,
    validate_preflight_gate,
    validate_resume_contract,
)
from service_agent.training.tau_rollout import (
    MultipleToolCallsError,
    require_single_tool_call,
)

ROOT = Path(__file__).resolve().parents[1]
ART_ROOT = ROOT / "third_party/ART"


def test_trainable_model_kwargs_match_pinned_art_signature():
    config = RuntimeConfig(run_name="preflight-r1")
    kwargs = build_trainable_model_kwargs(config)

    assert kwargs["run_name"] == "preflight-r1"
    assert kwargs["name"] == "preflight-r1"
    assert kwargs["base_model"] == BASE_MODEL_ID

    model_source = (ART_ROOT / "src/art/model.py").read_text()
    tree = ast.parse(model_source)
    trainable = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrainableModel"
    )
    init = next(
        node
        for node in trainable.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    accepted = {arg.arg for arg in init.args.kwonlyargs}
    assert set(kwargs) <= accepted
    assert_pinned_art_api(ART_ROOT)


def test_runtime_pins_weights_template_and_48gb_limits():
    runtime = RuntimeConfig(
        run_name="preflight-r1",
        max_model_len=16_384,
        max_completion_tokens=1_024,
        rollout_concurrency=4,
        gpu_memory_utilization=0.68,
    )
    config = build_internal_model_config(runtime)

    assert BASE_MODEL_REVISION == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert config["init_args"]["revision"] == BASE_MODEL_REVISION
    assert config["engine_args"]["revision"] == BASE_MODEL_REVISION
    assert config["engine_args"]["tokenizer_revision"] == BASE_MODEL_REVISION
    assert config["init_args"]["load_in_4bit"] is False
    assert config["init_args"]["load_in_16bit"] is True
    assert config["engine_args"]["dtype"] == "bfloat16"
    assert config["engine_args"]["max_model_len"] == 16_384
    assert config["engine_args"]["max_num_seqs"] == 4
    assert config["engine_args"]["gpu_memory_utilization"] == pytest.approx(0.68)
    assert config["chat_template_kwargs"] == CHAT_TEMPLATE_KWARGS == {
        "enable_thinking": False
    }


def test_multiple_tool_calls_fail_before_environment_step():
    choice = SimpleNamespace(
        message=SimpleNamespace(
            tool_calls=[
                SimpleNamespace(id="a"),
                SimpleNamespace(id="b"),
            ]
        )
    )
    with pytest.raises(MultipleToolCallsError):
        require_single_tool_call(choice)


def test_formal_update_requires_complete_matching_preflight_gate():
    expected = "semantic-contract-sha256"
    valid = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "preflight",
        "status": "passed",
        "art_commit": ART_COMMIT,
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "semantic_contract_sha256": expected,
        "initial_step": 0,
        "final_step": 0,
        "logprob_gate": {"status": "passed", "prompt_token_ids_exact": True},
        "rollout_only": {
            "status": "passed",
            "test_split_locked": True,
            "strict_replay": True,
            "reward_finalized_once": True,
            "multi_tool_calls": 0,
        },
    }
    validate_preflight_gate(valid, expected)

    for mutation in (
        {"status": "failed"},
        {"final_step": 1},
        {"semantic_contract_sha256": "different"},
        {"logprob_gate": {"status": "failed", "prompt_token_ids_exact": False}},
        {
            "rollout_only": {
                "status": "passed",
                "test_split_locked": False,
                "strict_replay": True,
                "multi_tool_calls": 0,
            }
        },
    ):
        invalid = {**valid, **mutation}
        with pytest.raises(RuntimeError):
            validate_preflight_gate(invalid, expected)


def _manifest(run_name: str = "run-r1") -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "train",
        "run_name": run_name,
        "repo_commit": "repo-commit",
        "art_commit": ART_COMMIT,
        "tau2_commit": TAU2_COMMIT,
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "semantic_contract_sha256": "contract",
        "runtime": {
            "run_name": run_name,
            "project": "service-agent",
            "max_model_len": 16_384,
            "max_completion_tokens": 1_024,
            "rollout_concurrency": 4,
            "gpu_memory_utilization": 0.68,
        },
        "training": {
            "group_size": 4,
            "groups_per_step": 2,
            "max_turns": 30,
            "steps": 60,
            "learning_rate": 5e-6,
            "kl_penalty_coef": 0.0,
            "loss_fn": "ppo",
            "val_every": 5,
            "val_trials": 2,
        },
        "token_budget": {"required_prompt_capacity": 14_900},
        "system": {
            "python": "3.12.11",
            "torch": "2.11.0",
            "cuda_runtime": "13.0",
            "gpu": "NVIDIA GeForce RTX 4090",
            "bf16_supported": True,
            "packages": {"openpipe-art": "0.5.18", "vllm": "0.17.0"},
        },
        "user_simulator": {
            "model": "deepseek/deepseek-v4-pro",
            "temperature": 0.0,
            "thinking": "disabled",
        },
    }


def test_phase_gates_require_the_same_complete_protocol():
    preflight = _manifest("preflight-r1")
    current = _manifest("smoke-r1")
    validate_matching_protocol(preflight, current, "preflight")

    drifted = _manifest("smoke-r1")
    drifted["runtime"]["max_model_len"] = 8_192
    with pytest.raises(RuntimeError, match="runtime"):
        validate_matching_protocol(preflight, drifted, "preflight")

    drifted = _manifest("smoke-r1")
    drifted["repo_commit"] = "different"
    with pytest.raises(RuntimeError, match="repo_commit"):
        validate_matching_protocol(preflight, drifted, "preflight")


def test_resume_requires_the_same_run_and_protocol():
    previous = _manifest()
    current = _manifest()
    validate_resume_contract(previous, current)

    changed = _manifest()
    changed["training"]["learning_rate"] = 1e-5
    with pytest.raises(RuntimeError, match="training"):
        validate_resume_contract(previous, changed)

    renamed = _manifest("another-run")
    with pytest.raises(RuntimeError, match="run_name"):
        validate_resume_contract(previous, renamed)


def test_group_stats_uses_within_group_reward_variance():
    def group(*rewards: float) -> SimpleNamespace:
        return SimpleNamespace(
            trajectories=[SimpleNamespace(reward=reward) for reward in rewards]
        )

    stats = group_stats(
        [
            group(0.0, 0.0),
            group(1.0, 1.0),
            group(0.5, 0.5),
            group(0.0, 0.5),
        ]
    )

    assert stats["mixed"] == 1
    assert stats["all_zero"] == 1
    assert stats["all_one"] == 1
    assert stats["constant_other"] == 1
