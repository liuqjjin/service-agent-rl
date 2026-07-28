"""GPU-training contracts that must fail locally before money is spent."""

from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import service_agent.training.art_tau_train as training_driver
import service_agent.training.contracts as training_contracts
from service_agent.training.art_tau_train import (
    GRADIENT_STEPS_METRIC,
    TRAINABLE_GROUPS_METRIC,
    _configure_vllm_runtime_bootstrap,
    _formal_seed_base,
    _gather_groups,
    _register_model,
    _run_smoke,
    _runtime_system_info,
    _scenarios_for_formal_step,
    _smoke_sampling_contract,
    _smoke_scenarios,
    _training_progress,
    _training_work_counts,
    _validate_smoke_gate,
    group_stats,
)
from service_agent.training.contracts import (
    ART_COMMIT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHAT_TEMPLATE_KWARGS,
    MANIFEST_SCHEMA_VERSION,
    TAU2_COMMIT,
    TOOL_CALL_PARSER,
    RuntimeConfig,
    apply_chat_template_token_ids,
    assert_pinned_art_api,
    build_internal_model_config,
    build_trainable_model_kwargs,
    semantic_contract_sha256,
    semantic_input_hashes,
    validate_matching_protocol,
    validate_preflight_gate,
    validate_resume_contract,
)
from service_agent.training.logprob_check import ProbeRecord, evaluate_probe_records
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


def _training_args(tmp_path: Path | None = None, **overrides) -> SimpleNamespace:
    root = tmp_path or Path("/tmp/service-agent-test")
    values = {
        "phase": "smoke",
        "run_name": "smoke-r1",
        "project": "service-agent",
        "shim_url": "http://127.0.0.1:8000",
        "user_model": "deepseek/deepseek-v4-pro",
        "art_path": root / "art",
        "out": root / "out",
        "hf_cache": root / "cache",
        "preflight_manifest": root / "preflight.json",
        "smoke_manifest": root / "smoke.json",
        "group_size": 4,
        "groups_per_step": 2,
        "max_turns": 30,
        "max_completion_tokens": 1_024,
        "max_model_len": 16_384,
        "rollout_concurrency": 4,
        "gpu_memory_utilization": 0.68,
        "logprob_calculation_chunk_size": 512,
        "steps": 60,
        "learning_rate": 5e-6,
        "kl_penalty_coef": 0.0,
        "loss_fn": "ppo",
        "val_every": 5,
        "val_trials": 2,
        "seed": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def test_qwen_tool_parser_is_frozen_and_changes_the_semantic_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = {
        "system_prompt": "system",
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tokenizer_chat_template": "{{ messages }}",
    }

    assert TOOL_CALL_PARSER == "qwen3_coder"
    qwen_contract = semantic_contract_sha256(**inputs)
    monkeypatch.setattr(training_contracts, "TOOL_CALL_PARSER", "hermes")

    assert semantic_contract_sha256(**inputs) != qwen_contract


def test_model_registration_overrides_art_with_the_frozen_qwen_parser(tmp_path: Path):
    registered: dict = {}
    wandb_config: dict = {}

    class Backend:
        def __init__(self, *, path: str):
            self.path = path

    class Model:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def update_wandb_config(self, config):
            wandb_config.update(config)

        async def register(self, backend, **kwargs):
            registered["backend"] = backend
            registered.update(kwargs)

    art = SimpleNamespace(TrainableModel=Model)
    args = _training_args(tmp_path)
    backend, model = asyncio.run(
        _register_model(
            art,
            Backend,
            args,
            snapshot=tmp_path / BASE_MODEL_REVISION,
        )
    )

    assert backend is registered["backend"]
    assert model.kwargs["base_model"] == str(tmp_path / BASE_MODEL_REVISION)
    assert registered["_openai_client_config"]["server_args"]["tool_call_parser"] == (
        "qwen3_coder"
    )
    assert wandb_config["protocol"]["tool_call_parser"] == "qwen3_coder"


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


def test_logprob_reference_uses_only_the_verified_local_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    snapshot = tmp_path / BASE_MODEL_REVISION
    snapshot.mkdir()
    record = ProbeRecord(
        messages=[{"role": "user", "content": "hello"}],
        prompt_token_ids=[101, 102],
        completion_token_ids=[103],
        rollout_logprobs=[-0.5],
    )
    calls: list[tuple[str, Path, dict]] = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == record.messages
            return record.prompt_token_ids

    class Model:
        def eval(self):
            return self

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("tokenizer", Path(source), kwargs))
            return Tokenizer()

    class ModelLoader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("model", Path(source), kwargs))
            return Model()

    torch = ModuleType("torch")
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        empty_cache=lambda: None,
    )
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = TokenizerLoader  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = ModelLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "service_agent.training.logprob_check._local_logprobs",
        lambda model, prompt_ids, completion_ids: [-0.5],
    )

    result = evaluate_probe_records(
        [record],
        tools=[],
        model_source=snapshot,
    )

    assert result["status"] == "passed"
    assert [call[:2] for call in calls] == [
        ("tokenizer", snapshot),
        ("model", snapshot),
    ]
    assert calls[0][2] == {"local_files_only": True}
    assert calls[1][2]["local_files_only"] is True
    assert "revision" not in calls[0][2]
    assert "revision" not in calls[1][2]

    wrong = tmp_path / "wrong-revision"
    wrong.mkdir()
    with pytest.raises(RuntimeError, match="verified model snapshot"):
        evaluate_probe_records([record], tools=[], model_source=wrong)


def test_art_mask_adapter_preserves_transformers_5_argument_order():
    calls: list[tuple] = []
    normalized_position_ids = object()

    class ThreeDimensionalPositionIds:
        shape = (1, 2, 3)

        def __getitem__(self, index):
            assert index == 0
            return normalized_position_ids

    def transformers_target(
        config,
        inputs_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids,
        layer_idx,
    ):
        calls.append(
            (
                config,
                inputs_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
                layer_idx,
            )
        )
        return "mask-result"

    def art_old_signature(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids,
        layer_idx,
        encoder_hidden_states=None,
    ):
        raise AssertionError("the incompatible ART wrapper must be replaced")

    masking_utils = SimpleNamespace(
        _preprocess_mask_arguments=art_old_signature,
    )
    art_patches = SimpleNamespace(
        _preprocess_mask_arguments=transformers_target,
        _patched_preprocess_mask_arguments=art_old_signature,
    )

    provenance = training_driver._install_transformers_mask_compat(
        masking_utils=masking_utils,
        art_patches=art_patches,
    )

    values = tuple(object() for _ in range(5))
    config, inputs_embeds, attention_mask, cache_position, past_key_values = values
    result = masking_utils._preprocess_mask_arguments(
        config,
        inputs_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        ThreeDimensionalPositionIds(),
        7,
    )

    assert result == "mask-result"
    assert calls == [
        (
            config,
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            normalized_position_ids,
            7,
        )
    ]
    assert provenance["status"] == "installed"
    assert provenance["transformers_parameter_order"] == [
        "config",
        "inputs_embeds",
        "attention_mask",
        "cache_position",
        "past_key_values",
        "position_ids",
        "layer_idx",
    ]
    assert provenance["art_parameter_order"] == [
        "config",
        "inputs_embeds",
        "attention_mask",
        "past_key_values",
        "position_ids",
        "layer_idx",
        "encoder_hidden_states",
    ]

    masking_utils._preprocess_mask_arguments = lambda: None
    with pytest.raises(RuntimeError, match="ART Transformers mask patch"):
        training_driver._install_transformers_mask_compat(
            masking_utils=masking_utils,
            art_patches=art_patches,
        )


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
        "tool_call_parser": TOOL_CALL_PARSER,
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
        "lineage_path": f"/art/service-agent/models/{run_name}",
        "repo_commit": "repo-commit",
        "art_commit": ART_COMMIT,
        "tau2_commit": TAU2_COMMIT,
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "tool_call_parser": TOOL_CALL_PARSER,
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
            "logprob_calculation_chunk_size": 512,
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
            "transformers_mask_compat": {
                "status": "installed",
                "target": "transformers.masking_utils._preprocess_mask_arguments",
                "transformers_parameter_order": list(
                    training_driver.TRANSFORMERS_MASK_PARAMETER_ORDER
                ),
                "art_parameter_order": list(
                    training_driver.ART_MASK_PARAMETER_ORDER
                ),
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
    drifted["system"]["transformers_mask_compat"]["status"] = "different"
    with pytest.raises(RuntimeError, match="system"):
        validate_matching_protocol(preflight, drifted, "preflight")

    drifted = _manifest("smoke-r1")
    drifted["semantic_input_hashes"]["tools_sha256"] = "different"
    with pytest.raises(RuntimeError, match="semantic_input_hashes"):
        validate_matching_protocol(preflight, drifted, "preflight")

    drifted = _manifest("smoke-r1")
    drifted["tool_call_parser"] = "hermes"
    with pytest.raises(RuntimeError, match="tool_call_parser"):
        validate_matching_protocol(preflight, drifted, "preflight")

    drifted = _manifest("smoke-r1")
    drifted["training"]["logprob_calculation_chunk_size"] = 1_024
    with pytest.raises(RuntimeError, match="training"):
        validate_matching_protocol(preflight, drifted, "preflight")


def test_every_backend_training_call_uses_the_manifested_logprob_chunk_size():
    source = (ROOT / "src/service_agent/training/art_tau_train.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "backend"
        and node.func.attr == "train"
    ]

    assert len(calls) == 2
    for call in calls:
        keyword = next(
            (
                item
                for item in call.keywords
                if item.arg == "logprob_calculation_chunk_size"
            ),
            None,
        )
        assert keyword is not None
        assert isinstance(keyword.value, ast.Attribute)
        assert isinstance(keyword.value.value, ast.Name)
        assert keyword.value.value.id == "args"
        assert keyword.value.attr == "logprob_calculation_chunk_size"


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

    moved = _manifest()
    moved["lineage_path"] = "/different/art/service-agent/models/run-r1"
    with pytest.raises(RuntimeError, match="lineage_path"):
        validate_resume_contract(previous, moved)

    for terminal_status in ("passed", "stopped_sparse_reward"):
        terminal = _manifest()
        terminal["status"] = terminal_status
        with pytest.raises(RuntimeError, match="terminal"):
            validate_resume_contract(terminal, current)


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


def test_smoke_is_the_contiguous_prefix_of_the_formal_schedule():
    scenarios = list(range(54))
    args = _training_args()

    assert _scenarios_for_formal_step(
        scenarios,
        step=0,
        groups_per_step=2,
        seed=42,
    ) == [39, 21]
    assert _scenarios_for_formal_step(
        scenarios,
        step=1,
        groups_per_step=2,
        seed=42,
    ) == [44, 30]
    assert _smoke_scenarios(scenarios, args) == [39, 21, 44, 30]
    assert _formal_seed_base(args, 0) == 42
    assert _formal_seed_base(args, 1) == 50
    assert _smoke_sampling_contract(args) == {
        "strategy": "contiguous_formal_prefix",
        "formal_checkpoint_steps": [0, 1],
        "formal_slots": [0, 1, 2, 3],
        "groups_per_formal_checkpoint_step": 2,
        "groups_submitted": 4,
        "group_size": 4,
        "policy_seed_base": 42,
        "policy_seeds": list(range(42, 58)),
    }


def test_gather_groups_maps_formal_policy_and_user_seeds_exactly(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict] = []

    async def fake_rollout(scenario, model, **kwargs):
        calls.append({"scenario": scenario, **kwargs})
        return SimpleNamespace(
            reward=1.0,
            metadata={"scenario_id": scenario},
            metrics={
                "multi_tool_calls": 0.0,
                "strict_replay": 1.0,
                "reward_finalized_once": 1.0,
                "terminated": 1.0,
            },
        )

    class TrajectoryGroup:
        def __init__(self, trajectories):
            self.trajectories = trajectories

    async def gather_trajectory_groups(groups):
        for group in groups:
            group.trajectories = await asyncio.gather(*group.trajectories)
        return groups

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(training_driver, "rollout", fake_rollout)
    art = SimpleNamespace(
        TrajectoryGroup=TrajectoryGroup,
        gather_trajectory_groups=gather_trajectory_groups,
    )

    groups = asyncio.run(
        _gather_groups(
            art,
            scenarios=["slot-0", "slot-1", "slot-2", "slot-3"],
            model=object(),
            client=object(),
            args=_training_args(),
            trials=4,
            seed_base=42,
        )
    )

    assert len(groups) == 4
    by_seed = {call["policy_seed"]: call for call in calls}
    assert sorted(by_seed) == list(range(42, 58))
    for slot, expected_seeds in enumerate(
        (range(42, 46), range(46, 50), range(50, 54), range(54, 58))
    ):
        for seed in expected_seeds:
            assert by_seed[seed]["scenario"] == f"slot-{slot}"
            assert by_seed[seed]["user_chat_completion_kwargs"]["seed"] == seed


def test_smoke_gate_requires_sampling_variance_and_gradient_work():
    args = _training_args()
    sampling = _smoke_sampling_contract(args)
    valid = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "smoke",
        "status": "passed",
        "tool_call_parser": TOOL_CALL_PARSER,
        "semantic_contract_sha256": "contract",
        "sampling": sampling,
        "initial_step": 0,
        "final_step": 1,
        "checkpoint_path": "/art/checkpoints/0001",
        "strict_replay": True,
        "stats": {"groups": 4, "rollouts": 16, "mixed": 1},
        "trainable_groups": 1,
        "gradient_steps": 88,
        "gradient_work_performed": True,
        "wandb_url": "https://wandb.example/run",
    }
    _validate_smoke_gate(valid, "contract", sampling)

    mutations = (
        ("sampling", {"formal_slots": [2, 3]}),
        ("stats", {"groups": 4, "rollouts": 16, "mixed": 0}),
        ("stats", {"groups": 3, "rollouts": 12, "mixed": 1}),
        ("trainable_groups", 0),
        ("gradient_steps", 0),
        ("tool_call_parser", "hermes"),
    )
    for key, value in mutations:
        invalid = copy.deepcopy(valid)
        invalid[key] = value
        with pytest.raises(RuntimeError, match="smoke gate is not valid"):
            _validate_smoke_gate(invalid, "contract", sampling)


def test_formal_progress_separates_checkpoint_positions_from_gradient_work():
    records = [
        {
            "checkpoint_step": 1,
            "groups_submitted": 2,
            "trainable_groups": 0,
            "gradient_steps": 0,
            "gradient_work_performed": False,
        },
        {
            "checkpoint_step": 2,
            "groups_submitted": 2,
            "trainable_groups": 1,
            "gradient_steps": 88,
            "gradient_work_performed": True,
        },
        {
            "checkpoint_step": 3,
            "groups_submitted": 2,
            "trainable_groups": 2,
            "gradient_steps": 41,
            "gradient_work_performed": True,
        },
    ]

    assert _training_progress(records) == {
        "checkpoint_steps_completed": 3,
        "trainable_checkpoint_steps": 2,
        "skipped_checkpoint_steps": 1,
        "gradient_steps": 129,
        "groups_submitted": 6,
        "trainable_groups": 3,
        "final_checkpoint_step": 3,
    }
    assert _training_progress(records, through_checkpoint_step=2) == {
        "checkpoint_steps_completed": 2,
        "trainable_checkpoint_steps": 1,
        "skipped_checkpoint_steps": 1,
        "gradient_steps": 88,
        "groups_submitted": 4,
        "trainable_groups": 1,
        "final_checkpoint_step": 2,
    }


def test_training_work_metrics_must_agree_with_observed_variance():
    stats = {"mixed": 1}
    metrics = {
        TRAINABLE_GROUPS_METRIC: 1.0,
        GRADIENT_STEPS_METRIC: 88.0,
    }

    assert _training_work_counts(stats, metrics) == (1, 88)

    with pytest.raises(RuntimeError, match="trainable-group count"):
        _training_work_counts(
            stats,
            {
                TRAINABLE_GROUPS_METRIC: 0.0,
                GRADIENT_STEPS_METRIC: 0.0,
            },
        )
    with pytest.raises(RuntimeError, match="gradient-work metrics"):
        _training_work_counts(
            stats,
            {
                TRAINABLE_GROUPS_METRIC: 1.0,
                GRADIENT_STEPS_METRIC: 0.0,
            },
        )


def test_smoke_refuses_constant_rewards_before_backend_train(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _training_args(tmp_path)
    manifest = {
        "model_snapshot": str(tmp_path / BASE_MODEL_REVISION),
        "semantic_contract_sha256": "contract",
    }
    scenarios = [
        SimpleNamespace(task=SimpleNamespace(id=f"task-{index}")) for index in range(54)
    ]
    groups = [
        SimpleNamespace(
            trajectories=[SimpleNamespace(reward=1.0) for _ in range(args.group_size)]
        )
        for _ in range(4)
    ]
    train_calls = 0

    class Client:
        async def close(self):
            return None

    class Backend:
        async def train(self, *args, **kwargs):
            nonlocal train_calls
            train_calls += 1

        async def close(self):
            return None

    class Model:
        _wandb_run = SimpleNamespace(url="https://wandb.example/run", finish=lambda: None)

        async def get_step(self):
            return 0

    async def load_scenarios(tau_bench, client):
        return scenarios, []

    async def register_model(art, local_backend, run_args, *, snapshot):
        return Backend(), Model()

    async def gather_groups(art, **kwargs):
        return groups

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(training_driver, "_read_json", lambda *args: {})
    monkeypatch.setattr(training_driver, "validate_preflight_gate", lambda *args: None)
    monkeypatch.setattr(training_driver, "validate_matching_protocol", lambda *args: None)
    monkeypatch.setattr(training_driver, "_load_scenarios", load_scenarios)
    monkeypatch.setattr(training_driver, "_register_model", register_model)
    monkeypatch.setattr(training_driver, "_gather_groups", gather_groups)
    tau_bench = SimpleNamespace(TauBenchClient=lambda **kwargs: Client())

    with pytest.raises(RuntimeError, match="no within-group reward variance"):
        asyncio.run(
            _run_smoke(
                SimpleNamespace(),
                tau_bench,
                Backend,
                args,
                manifest,
                tmp_path / "smoke_manifest.json",
            )
        )

    assert train_calls == 0


def test_smoke_records_positive_gradient_work_in_its_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _training_args(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "0001"
    checkpoint.mkdir(parents=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "smoke",
        "status": "running",
        "tool_call_parser": TOOL_CALL_PARSER,
        "model_snapshot": str(tmp_path / BASE_MODEL_REVISION),
        "semantic_contract_sha256": "contract",
    }
    scenarios = [
        SimpleNamespace(task=SimpleNamespace(id=f"task-{index}")) for index in range(54)
    ]
    reward_rows = (
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
    )
    groups = [
        SimpleNamespace(
            trajectories=[SimpleNamespace(reward=reward) for reward in rewards]
        )
        for rewards in reward_rows
    ]
    train_calls = 0

    class Client:
        async def close(self):
            return None

    class Backend:
        async def train(self, model, submitted, **kwargs):
            nonlocal train_calls
            train_calls += 1
            assert submitted is groups
            return SimpleNamespace(
                step=1,
                checkpoint_path=str(checkpoint),
                metrics={
                    TRAINABLE_GROUPS_METRIC: 1.0,
                    GRADIENT_STEPS_METRIC: 88.0,
                },
            )

        async def close(self):
            return None

    class Model:
        _wandb_run = SimpleNamespace(url="https://wandb.example/run", finish=lambda: None)

        async def get_step(self):
            return 0

        async def log(self, *args, **kwargs):
            return None

    async def load_scenarios(tau_bench, client):
        return scenarios, []

    async def register_model(art, local_backend, run_args, *, snapshot):
        return Backend(), Model()

    async def gather_groups(art, **kwargs):
        assert kwargs["scenarios"] == _smoke_scenarios(scenarios, args)
        assert kwargs["trials"] == 4
        assert kwargs["seed_base"] == 42
        return groups

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(training_driver, "_read_json", lambda *args: {})
    monkeypatch.setattr(training_driver, "validate_preflight_gate", lambda *args: None)
    monkeypatch.setattr(training_driver, "validate_matching_protocol", lambda *args: None)
    monkeypatch.setattr(training_driver, "_load_scenarios", load_scenarios)
    monkeypatch.setattr(training_driver, "_register_model", register_model)
    monkeypatch.setattr(training_driver, "_gather_groups", gather_groups)
    tau_bench = SimpleNamespace(TauBenchClient=lambda **kwargs: Client())

    asyncio.run(
        _run_smoke(
            SimpleNamespace(),
            tau_bench,
            Backend,
            args,
            manifest,
            tmp_path / "smoke_manifest.json",
        )
    )

    assert train_calls == 1
    assert manifest["trainable_groups"] == 1
    assert manifest["gradient_steps"] == 88
    assert manifest["gradient_work_performed"] is True
    _validate_smoke_gate(
        manifest,
        "contract",
        _smoke_sampling_contract(args),
    )
