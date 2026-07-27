"""GPU-training contracts that must fail locally before money is spent."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from service_agent.training.art_tau_train import (
    _configure_vllm_runtime_bootstrap,
    _runtime_system_info,
    group_stats,
)
from service_agent.training.contracts import (
    ART_COMMIT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHAT_TEMPLATE_KWARGS,
    MANIFEST_SCHEMA_VERSION,
    TAU2_COMMIT,
    RuntimeConfig,
    apply_chat_template_token_ids,
    assert_pinned_art_api,
    build_internal_model_config,
    build_trainable_model_kwargs,
    semantic_input_hashes,
    validate_matching_protocol,
    validate_preflight_gate,
    validate_resume_contract,
)
from service_agent.training.tau_rollout import (
    MultipleToolCallsError,
    require_single_tool_call,
)
from service_agent.training.token_budget import _assistant_message

ROOT = Path(__file__).resolve().parents[1]
ART_ROOT = ROOT / "third_party/ART"
VLLM_BOOTSTRAP = (
    ROOT / "src/service_agent/training/vllm_bootstrap/sitecustomize.py"
)


def test_chat_template_ids_extract_transformers_5_batch_encoding():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Hello"}]
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["enable_thinking"] is False
            return {"input_ids": [101, 102, 103], "attention_mask": [1, 1, 1]}

    assert apply_chat_template_token_ids(
        Tokenizer(),
        [{"role": "user", "content": "Hello"}],
        tools=[],
    ) == [101, 102, 103]


def test_dev_token_budget_uses_qwen_tool_argument_mappings():
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "name": "lookup_account",
                "arguments": {"customer_id": "C1001"},
            }
        ],
    }

    visible = _assistant_message(message)

    assert visible["tool_calls"][0]["function"]["arguments"] == {
        "customer_id": "C1001"
    }


def test_semantic_input_hashes_record_each_exact_model_visible_surface():
    prompt = "system prompt\n"
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    template = "{{ messages }}"

    assert semantic_input_hashes(
        system_prompt=prompt,
        tools=tools,
        tokenizer_chat_template=template,
    ) == {
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "tools_sha256": hashlib.sha256(
            json.dumps(
                tools,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest(),
        "tokenizer_chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }


def test_runtime_system_info_reads_the_isolated_art_vllm_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runtime_python = tmp_path / "python"
    runtime_python.touch()
    expected = {
        "python": "3.12.3",
        "packages": {
            "art-vllm-runtime": "0.1.0",
            "flashinfer-python": "0.6.12",
            "ninja": "1.13.0",
            "torch": "2.11.0+cu128",
            "torchaudio": "2.11.0+cu128",
            "torchvision": "0.26.0+cu128",
            "transformers": "5.12.1",
            "vllm": "0.23.0+cu129",
        },
        "cudart": {
            "path": "/runtime/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
            "sha256": "cudart-sha256",
            "cuda_device_reset": True,
        },
    }

    def fake_run(command, **kwargs):
        assert command[0] == str(runtime_python)
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return SimpleNamespace(stdout=json.dumps(expected))

    monkeypatch.setattr("service_agent.training.art_tau_train.subprocess.run", fake_run)

    assert _runtime_system_info(runtime_python) == expected


def test_vllm_bootstrap_prefers_verified_cudart_over_tilelang_stub(
    monkeypatch: pytest.MonkeyPatch,
):
    vllm = ModuleType("vllm")
    vllm.__path__ = []  # type: ignore[attr-defined]
    utils = ModuleType("vllm.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    system_utils = ModuleType("vllm.utils.system_utils")

    def find_loaded_library(name: str) -> str | None:
        if name == "libcudart":
            return "/runtime/site-packages/tilelang/lib/libcudart_stub.so"
        return f"/runtime/{name}.so"

    system_utils.find_loaded_library = find_loaded_library  # type: ignore[attr-defined]
    utils.system_utils = system_utils  # type: ignore[attr-defined]
    vllm.utils = utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.system_utils", system_utils)
    monkeypatch.setenv(
        "VLLM_CUDART_SO_PATH",
        "/runtime/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
    )

    runpy.run_path(str(VLLM_BOOTSTRAP), run_name="__vllm_bootstrap_test__")

    assert system_utils.find_loaded_library("libcudart") == (
        "/runtime/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
    )
    assert system_utils.find_loaded_library("cumem_allocator") == (
        "/runtime/cumem_allocator.so"
    )


def test_vllm_bootstrap_is_silent_when_vllm_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.delitem(sys.modules, "vllm.utils", raising=False)
    monkeypatch.delitem(sys.modules, "vllm.utils.system_utils", raising=False)
    monkeypatch.setenv("VLLM_CUDART_SO_PATH", "/runtime/libcudart.so.12")

    runpy.run_path(str(VLLM_BOOTSTRAP), run_name="__no_vllm_bootstrap_test__")


def test_vllm_bootstrap_is_probed_and_recorded_before_gpu_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runtime_python = tmp_path / ".venv/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.touch()
    ninja = runtime_python.parent / "ninja"
    ninja.write_bytes(b"verified ninja")
    ninja.chmod(0o755)
    cudart = tmp_path / "site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
    cudart.parent.mkdir(parents=True)
    cudart.write_bytes(b"verified cudart")
    cudart_sha = hashlib.sha256(cudart.read_bytes()).hexdigest()
    runtime_info = {
        "python": "3.12.3",
        "packages": {"ninja": "1.13.0", "vllm": "0.23.0+cu129"},
        "cudart": {
            "path": str(cudart),
            "sha256": cudart_sha,
            "cuda_device_reset": True,
        },
    }
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("VLLM_CUDART_SO_PATH", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")

    def fake_run(command, **kwargs):
        assert command[0] == str(runtime_python)
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["env"]["VLLM_CUDART_SO_PATH"] == str(cudart)
        assert str(VLLM_BOOTSTRAP.parent) in kwargs["env"]["PYTHONPATH"].split(
            os.pathsep
        )
        assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(
            runtime_python.parent
        )
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "selected_cudart": str(cudart),
                    "cuda_device_reset": True,
                    "ninja_path": str(ninja),
                    "ninja_version": "1.13.0.git.kitware.jobserver-pipe-1",
                }
            )
        )

    monkeypatch.setattr("service_agent.training.art_tau_train.subprocess.run", fake_run)

    recorded = _configure_vllm_runtime_bootstrap(runtime_python, runtime_info)

    assert recorded["cudart"] == runtime_info["cudart"]
    assert recorded["bootstrap"]["probe"] == "passed"
    assert recorded["bootstrap"]["path"] == str(VLLM_BOOTSTRAP)
    assert recorded["bootstrap"]["sha256"] == hashlib.sha256(
        VLLM_BOOTSTRAP.read_bytes()
    ).hexdigest()
    assert recorded["bootstrap"]["ninja_path"] == str(ninja)
    assert recorded["bootstrap"]["ninja_sha256"] == hashlib.sha256(
        ninja.read_bytes()
    ).hexdigest()
    assert recorded["bootstrap"]["ninja_binary_version"] == (
        "1.13.0.git.kitware.jobserver-pipe-1"
    )
    assert recorded["bootstrap"]["ninja_distribution_version"] == "1.13.0"
    assert os.environ["VLLM_CUDART_SO_PATH"] == str(cudart)
    assert str(VLLM_BOOTSTRAP.parent) in os.environ["PYTHONPATH"].split(os.pathsep)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(runtime_python.parent)


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


def test_trainable_model_can_use_the_verified_local_snapshot():
    config = RuntimeConfig(run_name="preflight-r1")
    snapshot = (
        "/cache/models--Qwen--Qwen3.5-4B/snapshots/"
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )

    kwargs = build_trainable_model_kwargs(config, model_source=snapshot)

    assert kwargs["base_model"] == snapshot


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
        "semantic_input_hashes": {
            "system_prompt_sha256": "prompt",
            "tools_sha256": "tools",
            "tokenizer_chat_template_sha256": "template",
        },
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
            "packages": {"openpipe-art": "0.5.18", "transformers": "5.2.0"},
            "vllm_runtime": {
                "python": "3.12.3",
                "packages": {
                    "art-vllm-runtime": "0.1.0",
                    "torch": "2.11.0+cu128",
                    "transformers": "5.12.1",
                    "vllm": "0.23.0+cu129",
                },
            },
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

    drifted = _manifest("smoke-r1")
    drifted["system"]["vllm_runtime"]["packages"]["vllm"] = "different"
    with pytest.raises(RuntimeError, match="system"):
        validate_matching_protocol(preflight, drifted, "preflight")

    drifted = _manifest("smoke-r1")
    drifted["semantic_input_hashes"]["tools_sha256"] = "different"
    with pytest.raises(RuntimeError, match="semantic_input_hashes"):
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
