"""The final vLLM process must expose one frozen base and one static LoRA."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from service_agent.eval import final_serving
from service_agent.eval.final_serving import (
    ADAPTER_SHA256,
    BASE_ALIAS,
    BASE_MODEL_REVISION,
    CHAT_TEMPLATE_SHA256,
    FROZEN_ENGINE_ARGS,
    RL_ALIAS,
    VLLM_BOOTSTRAP_SHA256,
    VLLM_CUDART_SHA256,
    VLLM_NINJA_SHA256,
    build_serving_manifest,
    build_vllm_command,
    build_vllm_environment,
    finalize_serving_manifest,
    launch_serving,
    probe_serving,
    validate_manifest_for_final_runner,
    validate_serving_inputs,
)


def _arg(command: list[str], name: str) -> str:
    prefix = f"--{name}="
    return next(value.removeprefix(prefix) for value in command if value.startswith(prefix))


def test_vllm_command_and_environment_are_frozen(tmp_path):
    repo = tmp_path / "repo"
    runtime = repo / "third_party/ART/vllm_runtime/.venv/bin/art-vllm-runtime-server"
    snapshot = (
        tmp_path
        / "hf/models--Qwen--Qwen3.5-4B/snapshots"
        / BASE_MODEL_REVISION
    )
    adapter = tmp_path / "art/checkpoints/0015"

    command = build_vllm_command(
        snapshot,
        adapter,
        repo_root=repo,
        runtime_server=runtime,
    )
    assert command[0] == str(runtime.resolve())
    assert _arg(command, "model") == str(snapshot.resolve())
    assert _arg(command, "host") == "127.0.0.1"
    assert _arg(command, "port") == "8100"
    assert _arg(command, "cuda-visible-devices") == "0"
    assert _arg(command, "served-model-name") == BASE_ALIAS
    assert "--rollout-weights-mode=lora" in command
    assert not any(value.startswith("--lora-path") for value in command)

    engine = json.loads(_arg(command, "engine-args-json"))
    assert engine == {
        "allowed_local_media_path": "/tmp",
        "dtype": "bfloat16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enable_sleep_mode": True,
        "enforce_eager": True,
        "generation_config": "vllm",
        "gpu_memory_utilization": 0.68,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "max_logprobs": 1,
        "max_loras": 2,
        "max_model_len": 16384,
        "max_num_seqs": 4,
        "revision": BASE_MODEL_REVISION,
        "tokenizer_revision": BASE_MODEL_REVISION,
    }
    server = json.loads(_arg(command, "server-args-json"))
    assert server == {
        "enable_auto_tool_choice": True,
        "lora_modules": [
            {
                "base_model_name": BASE_ALIAS,
                "name": RL_ALIAS,
                "path": str(adapter.resolve()),
            }
        ],
        "return_tokens_as_token_ids": True,
        "tool_call_parser": "qwen3_coder",
        "uvicorn_log_level": "warning",
    }

    unsafe_tilelang = tmp_path / "site-packages/tilelang/runtime"
    env = build_vllm_environment(
        snapshot,
        repo_root=repo,
        runtime_server=runtime,
        hf_home=tmp_path / "hf",
        environ={
            "PATH": "/usr/bin",
            "PYTHONPATH": f"{unsafe_tilelang}:/safe/python",
            "TL_TEMPLATE_PATH": "/unsafe/non-tilelang-template-root",
            "TVM_LIBRARY_PATH": "/unsafe/non-tilelang-library",
            "UNRELATED": "kept",
        },
    )
    runtime_bin = str(runtime.resolve().parent)
    bootstrap_dir = str(
        (repo / "src/service_agent/training/vllm_bootstrap").resolve()
    )
    assert env["PATH"].split(":")[0] == runtime_bin
    assert env["PYTHONPATH"].split(":") == [bootstrap_dir, "/safe/python"]
    assert "TL_TEMPLATE_PATH" not in env
    assert "TVM_LIBRARY_PATH" not in env
    assert env["UNRELATED"] == "kept"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["VLLM_CUDART_SO_PATH"].endswith(
        "nvidia/cuda_runtime/lib/libcudart.so.12"
    )

    with pytest.raises(TypeError):
        FROZEN_ENGINE_ARGS["dtype"] = "float16"  # type: ignore[index]


def _fake_runtime_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    runtime = repo / "third_party/ART/vllm_runtime/.venv/bin/art-vllm-runtime-server"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    (runtime.parent / "python").write_text("")
    (runtime.parent / "ninja").write_text("")
    (runtime.parent / "python").chmod(0o755)
    (runtime.parent / "ninja").chmod(0o755)
    cudart = (
        runtime.parent.parent
        / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
    )
    cudart.parent.mkdir(parents=True)
    cudart.write_text("")
    bootstrap = repo / "src/service_agent/training/vllm_bootstrap/sitecustomize.py"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text("")
    (repo / "third_party/ART/scratch/vllm_runtime_flashinfer").mkdir(parents=True)

    snapshot = (
        tmp_path
        / "hf/models--Qwen--Qwen3.5-4B/snapshots"
        / BASE_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "chat_template.jinja").write_text("template")
    adapter = tmp_path / "art/checkpoints/0015"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "inference_mode": True,
                "target_modules": [
                    "down_proj",
                    "v_proj",
                    "up_proj",
                    "o_proj",
                    "q_proj",
                    "gate_proj",
                    "k_proj",
                ],
            }
        )
    )
    return repo, snapshot, adapter


def test_serving_inputs_bind_committed_manifests_and_file_hashes(tmp_path, monkeypatch):
    repo, snapshot, adapter = _fake_runtime_tree(tmp_path)
    runtime = repo / "third_party/ART/vllm_runtime/.venv/bin/art-vllm-runtime-server"

    expected_by_name = {
        "adapter_model.safetensors": ADAPTER_SHA256,
        "chat_template.jinja": CHAT_TEMPLATE_SHA256,
        "sitecustomize.py": VLLM_BOOTSTRAP_SHA256,
        "ninja": VLLM_NINJA_SHA256,
        "libcudart.so.12": VLLM_CUDART_SHA256,
    }
    monkeypatch.setattr(
        final_serving,
        "_sha256_file",
        lambda path: expected_by_name[path.name],
    )
    monkeypatch.setattr(
        final_serving,
        "_runtime_package_versions",
        lambda layout: dict(final_serving.FROZEN_RUNTIME_PACKAGES),
    )

    evidence = validate_serving_inputs(
        snapshot,
        adapter,
        repo_root=repo,
        runtime_server=runtime,
        formal_manifest=final_serving.FORMAL_MANIFEST,
        restore_manifest=final_serving.RESTORE_MANIFEST,
    )
    assert evidence["base_model_alias"] == BASE_ALIAS
    assert evidence["rl_model_alias"] == RL_ALIAS
    assert evidence["adapter_sha256"] == ADAPTER_SHA256
    assert evidence["tool_call_parser"] == "qwen3_coder"
    assert evidence["chat_template_kwargs"] == {"enable_thinking": False}

    manifest = build_serving_manifest(
        snapshot,
        adapter,
        repo_root=repo,
        runtime_server=runtime,
        hf_home=tmp_path / "hf",
        formal_manifest=final_serving.FORMAL_MANIFEST,
        restore_manifest=final_serving.RESTORE_MANIFEST,
    )
    assert manifest["status"] == "prepared"
    assert manifest["server_args"]["lora_modules"][0]["path"] == str(adapter.resolve())
    assert manifest["snapshot_file_count"] == 1
    assert manifest["snapshot_total_bytes"] == len("template")

    captured: dict = {}

    class ExecIntercept(RuntimeError):
        pass

    def fake_exec(file, command, environment):
        captured.update(file=file, command=command, environment=environment)
        raise ExecIntercept

    monkeypatch.setattr(final_serving.os, "execvpe", fake_exec)
    with pytest.raises(ExecIntercept):
        launch_serving(
            snapshot,
            adapter,
            repo_root=repo,
            runtime_server=runtime,
            hf_home=tmp_path / "hf",
            formal_manifest=final_serving.FORMAL_MANIFEST,
            restore_manifest=final_serving.RESTORE_MANIFEST,
            environ={"PATH": "/usr/bin"},
        )
    assert captured["file"] == str(runtime.resolve())
    assert captured["command"] == manifest["command"]
    assert captured["environment"]["HF_HUB_OFFLINE"] == "1"
    assert captured["environment"]["VLLM_CUDART_SO_PATH"].endswith("libcudart.so.12")

    path = tmp_path / "serving_manifest.json"
    path.write_text(json.dumps(manifest))
    assert validate_manifest_for_final_runner(path, allow_prepared=True) == manifest
    with pytest.raises(RuntimeError, match="not passed"):
        validate_manifest_for_final_runner(path)

    probe = {
        "schema_version": 1,
        "status": "passed",
        "checked_at": "2026-07-29T00:00:00+00:00",
        "api_base": manifest["api_base"],
        "vllm_api_version": "0.23.0",
        "capabilities": {
            "runtime": "art_vllm",
            "protocol_version": 1,
            "inplace_lora_load": True,
            "policy_token_spans": True,
        },
        "model_cards": {
            BASE_ALIAS: {
                "root": str(snapshot.resolve()),
                "parent": None,
                "max_model_len": 16384,
            },
            RL_ALIAS: {
                "root": str(adapter.resolve()),
                "parent": BASE_ALIAS,
            },
        },
        "probes": {
            alias: {
                "status": "passed",
                "response_model": alias,
                "finish_reason": "tool_calls",
                "tool_call_count": 1,
                "tool_name": "health_probe",
                "arguments": {"status": "ready"},
                "prompt_tokens": 20,
                "completion_tokens": 6,
                "reasoning_content_absent": True,
                "thinking_tags_absent": True,
            }
            for alias in (BASE_ALIAS, RL_ALIAS)
        },
        "tool_call_parser": "qwen3_coder",
        "chat_template_kwargs": {"enable_thinking": False},
        "benchmark_data_accessed": False,
    }
    finalized = finalize_serving_manifest(manifest, probe)
    assert finalized["status"] == "passed"
    assert finalized["probe"] == probe
    path.write_text(json.dumps(finalized))
    assert validate_manifest_for_final_runner(path) == finalized

    bad_probe = deepcopy(probe)
    bad_probe["probes"][RL_ALIAS]["finish_reason"] = "stop"
    with pytest.raises(RuntimeError, match="finish_reason drifted"):
        finalize_serving_manifest(manifest, bad_probe)

    drifted = deepcopy(manifest)
    drifted["engine_args"]["dtype"] = "float16"
    path.write_text(json.dumps(drifted))
    with pytest.raises(RuntimeError, match="engine_args drift"):
        validate_manifest_for_final_runner(path, allow_prepared=True)

    wrong_lora = deepcopy(manifest)
    wrong_lora["server_args"]["lora_modules"][0]["name"] = BASE_ALIAS
    path.write_text(json.dumps(wrong_lora))
    with pytest.raises(RuntimeError, match="static LoRA mapping"):
        validate_manifest_for_final_runner(path, allow_prepared=True)

    leaked = deepcopy(manifest)
    leaked["provider_api_key"] = "secret"
    path.write_text(json.dumps(leaked))
    with pytest.raises(RuntimeError, match="private field"):
        validate_manifest_for_final_runner(path, allow_prepared=True)

    path.write_text(json.dumps(finalized))
    (snapshot / "chat_template.jinja").write_text("template changed after preparation")
    with pytest.raises(RuntimeError, match="snapshot tree drifted"):
        validate_manifest_for_final_runner(path)


class _ProbeServer(ThreadingHTTPServer):
    snapshot: str
    adapter: str
    requests: list[dict]
    extra_model: bool = False
    leak_reasoning: bool = False


class _ProbeHandler(BaseHTTPRequestHandler):
    server: _ProbeServer

    def log_message(self, format, *args):  # noqa: A002
        return

    def _reply(self, payload: dict | None = None) -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._reply()
        elif self.path == "/version":
            self._reply({"version": "0.23.0"})
        elif self.path == "/art/capabilities":
            self._reply(
                {
                    "runtime": "art_vllm",
                    "protocol_version": 1,
                    "inplace_lora_load": True,
                    "policy_token_spans": True,
                    "binary_routed_experts": True,
                }
            )
        elif self.path == "/v1/models":
            cards = [
                {
                    "id": BASE_ALIAS,
                    "root": self.server.snapshot,
                    "parent": None,
                    "max_model_len": 16384,
                },
                {
                    "id": RL_ALIAS,
                    "root": self.server.adapter,
                    "parent": BASE_ALIAS,
                },
            ]
            if self.server.extra_model:
                cards.append({"id": "unexpected", "root": "/tmp", "parent": None})
            self._reply({"object": "list", "data": cards})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)
        alias = request["model"]
        self._reply(
            {
                "id": "chatcmpl-health",
                "model": alias,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "Checking readiness.",
                            "reasoning_content": (
                                "hidden" if self.server.leak_reasoning else None
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_health",
                                    "type": "function",
                                    "function": {
                                        "name": "health_probe",
                                        "arguments": '{"status":"ready"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 6,
                    "total_tokens": 26,
                },
            }
        )


@pytest.fixture
def probe_server(tmp_path):
    snapshot = (tmp_path / BASE_MODEL_REVISION).resolve()
    adapter = (tmp_path / "0015").resolve()
    server = _ProbeServer(("127.0.0.1", 0), _ProbeHandler)
    server.snapshot = str(snapshot)
    server.adapter = str(adapter)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, snapshot, adapter
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_probe_serving_checks_both_aliases_without_benchmark_data(probe_server):
    server, snapshot, adapter = probe_server
    api_base = f"http://127.0.0.1:{server.server_port}/v1"
    result = probe_serving(api_base, snapshot, adapter)
    assert result["status"] == "passed"
    assert result["benchmark_data_accessed"] is False
    assert set(result["probes"]) == {BASE_ALIAS, RL_ALIAS}
    assert len(server.requests) == 2
    assert [request["model"] for request in server.requests] == [BASE_ALIAS, RL_ALIAS]
    for request in server.requests:
        assert request["tool_choice"] == "auto"
        assert request["temperature"] == 0.0
        assert request["seed"] == 42
        assert request["max_tokens"] == 128
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        assert "extra_body" not in request
        assert request["messages"][0]["content"] == (
            "You are a local serving health probe. Call the supplied "
            "health_probe function exactly once."
        )
        assert request["tools"][0]["function"]["name"] == "health_probe"
        serialized = json.dumps(request)
        assert "scenario" not in serialized
        assert "task_id" not in serialized


def test_probe_serving_rejects_alias_and_reasoning_drift(probe_server):
    server, snapshot, adapter = probe_server
    api_base = f"http://127.0.0.1:{server.server_port}/v1"
    server.extra_model = True
    with pytest.raises(RuntimeError, match="model aliases drifted"):
        probe_serving(api_base, snapshot, adapter)

    server.extra_model = False
    server.leak_reasoning = True
    with pytest.raises(RuntimeError, match="reasoning_content"):
        probe_serving(api_base, snapshot, adapter)

    with pytest.raises(RuntimeError, match="non-loopback"):
        probe_serving("http://example.com/v1", snapshot, adapter)
