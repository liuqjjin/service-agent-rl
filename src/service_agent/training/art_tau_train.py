"""Preflight and run dual-control GRPO through the tau2 shim.

The three phases are deliberately separate lineages:

* preflight: register step 0, prove exact-token/logprob parity, then run
  train-core rollouts without an update;
* smoke: require the preflight artifact and perform exactly one update in a
  disposable ART directory;
* train: require both gates, start a fresh formal lineage, or resume that
  exact lineage from its durable manifest.

No phase can list or instantiate the official test split.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from service_agent.training.contracts import (
    ART_COMMIT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    MANIFEST_SCHEMA_VERSION,
    TAU2_COMMIT,
    TOOL_CALL_PARSER,
    RuntimeConfig,
    assert_pinned_art_api,
    build_trainable_model_kwargs,
    semantic_contract_sha256,
    validate_matching_protocol,
    validate_preflight_gate,
    validate_resume_contract,
)
from service_agent.training.logprob_check import (
    evaluate_probe_records,
    sample_probe_records,
)
from service_agent.training.model_snapshot import prepare_pinned_snapshot
from service_agent.training.tau_rollout import rollout
from service_agent.training.token_budget import (
    measure_dev_token_budget,
    validate_context_capacity,
)

DEFAULT_USER_MODEL = "deepseek/deepseek-v4-pro"
SPARSE_REWARD_WINDOW = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "smoke", "train"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default="service-agent")
    parser.add_argument("--shim-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    parser.add_argument("--art-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--groups-per-step", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-completion-tokens", type=int, default=1_024)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--rollout-concurrency", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.68)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.0)
    parser.add_argument("--loss-fn", choices=["cispo", "ppo"], default="ppo")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--val-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def assert_training_split_clean(scenario_ids: list[str]) -> None:
    """Trainer-side guard in addition to the shim's endpoint locks."""

    from service_agent.splits import load_frozen_dev_ids, load_split_ids, train_core_ids

    ids = set(scenario_ids)
    splits = load_split_ids()
    assert ids.isdisjoint(splits["test"]), "training scenarios intersect test"
    assert ids.isdisjoint(set(load_frozen_dev_ids())), "training scenarios intersect dev"
    assert ids <= set(train_core_ids()), "training scenarios outside train-core"


def assert_no_labels(scenarios: list[Any]) -> None:
    for scenario in scenarios:
        task = scenario.task.model_dump()
        for field in (
            "evaluation_criteria",
            "user_scenario",
            "initial_state",
            "ticket",
            "description",
        ):
            assert not task.get(field), f"scenario {task.get('id')} carries {field}"


def user_llm_args() -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    return {
        "temperature": 0.0,
        "extra_body": {"thinking": {"type": "disabled"}},
        "api_key": api_key,
    }


def group_stats(groups: list[Any]) -> dict[str, Any]:
    total = mixed = all_zero = all_one = constant_other = 0
    rewards: list[float] = []
    for group in groups:
        values = [float(trajectory.reward) for trajectory in group.trajectories]
        rewards.extend(values)
        total += 1
        if len(set(values)) > 1:
            mixed += 1
        elif values and values[0] <= 0.0:
            all_zero += 1
        elif values and values[0] >= 1.0:
            all_one += 1
        else:
            constant_other += 1
    return {
        "groups": total,
        "rollouts": len(rewards),
        "mixed": mixed,
        "all_zero": all_zero,
        "all_one": all_one,
        "constant_other": constant_other,
        "reward_mean": round(sum(rewards) / max(len(rewards), 1), 6),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path | None, description: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise RuntimeError(f"{description} is required")
    return json.loads(path.read_text())


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fetch_json(base_url: str, route: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{route}?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def _test_split_is_locked(base_url: str) -> bool:
    list_url = (
        f"{base_url.rstrip('/')}/scenarios?"
        + urlencode({"domain": "telecom", "split": "test"})
    )
    try:
        with urlopen(list_url, timeout=30):
            list_locked = False
    except HTTPError as exc:
        list_locked = exc.code == 403

    from service_agent.splits import load_split_ids

    body = json.dumps(
        {
            "domain": "telecom",
            "task_id": sorted(load_split_ids()["test"])[0],
            "user_llm": "locked/none",
        }
    ).encode()
    request = Request(
        f"{base_url.rstrip('/')}/environments",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30):
            create_locked = False
    except HTTPError as exc:
        create_locked = exc.code == 403
    return list_locked and create_locked


def _model_dir(args: argparse.Namespace) -> Path:
    return args.art_path / args.project / "models" / args.run_name


def _require_fresh_lineage(args: argparse.Namespace) -> None:
    path = _model_dir(args)
    if path.exists():
        raise RuntimeError(f"{args.phase} requires a fresh lineage; already exists: {path}")


def _runtime(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        run_name=args.run_name,
        project=args.project,
        max_model_len=args.max_model_len,
        max_completion_tokens=args.max_completion_tokens,
        rollout_concurrency=args.rollout_concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )


def _system_info() -> dict[str, Any]:
    import torch

    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    packages: dict[str, str | None] = {}
    for distribution in (
        "openpipe-art",
        "transformers",
        "unsloth",
        "vllm",
        "trl",
        "wandb",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "gpu": gpu.name if gpu is not None else None,
        "gpu_memory_bytes": gpu.total_memory if gpu is not None else None,
        "packages": packages,
    }


def _base_manifest(
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    *,
    semantic_hash: str,
    snapshot: Path,
    token_budget: dict[str, Any],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": args.phase,
        "status": "running",
        "run_name": args.run_name,
        "started_at": _now(),
        "repo_commit": _git_commit(repo_root),
        "art_commit": _git_commit(repo_root / "third_party/ART"),
        "tau2_commit": _git_commit(repo_root / "third_party/tau2-bench"),
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "model_snapshot": str(snapshot),
        "semantic_contract_sha256": semantic_hash,
        "runtime": asdict(runtime),
        "training": {
            "group_size": args.group_size,
            "groups_per_step": args.groups_per_step,
            "max_turns": args.max_turns,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "kl_penalty_coef": args.kl_penalty_coef,
            "loss_fn": args.loss_fn,
            "val_every": args.val_every,
            "val_trials": args.val_trials,
        },
        "token_budget": token_budget,
        "system": _system_info(),
        "user_simulator": {
            "model": args.user_model,
            "temperature": 0.0,
            "thinking": "disabled",
        },
        "test_split_locked": True,
        "events": [],
    }


def _wandb_url(model: Any) -> str | None:
    run = getattr(model, "_wandb_run", None)
    value = getattr(run, "url", None)
    return str(value) if value else None


def _finish_wandb(model: Any) -> None:
    run = getattr(model, "_wandb_run", None)
    finish = getattr(run, "finish", None)
    if callable(finish):
        finish()


def _checkpoint_path(args: argparse.Namespace, step: int) -> str:
    return str(_model_dir(args) / "checkpoints" / f"{step:04d}")


def _require_checkpoint(path: str | None, expected_step: int) -> str:
    if path is None:
        raise RuntimeError(f"ART returned no checkpoint for step {expected_step}")
    checkpoint = Path(path)
    if not checkpoint.is_dir() or checkpoint.name != f"{expected_step:04d}":
        raise RuntimeError(
            f"invalid checkpoint for step {expected_step}: {checkpoint}"
        )
    return str(checkpoint)


async def _register_model(art: Any, LocalBackend: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    runtime = _runtime(args)
    backend = LocalBackend(path=str(args.art_path))
    model = art.TrainableModel(**build_trainable_model_kwargs(runtime))
    model.update_wandb_config(
        {
            "protocol": {
                "base_model": BASE_MODEL_ID,
                "base_model_revision": BASE_MODEL_REVISION,
                "art_commit": ART_COMMIT,
                "phase": args.phase,
            },
            "runtime": asdict(runtime),
        }
    )
    await model.register(
        backend,
        _openai_client_config={
            "server_args": {
                "tool_call_parser": TOOL_CALL_PARSER,
                "uvicorn_log_level": "warning",
            }
        },
    )
    return backend, model


def _validate_official_groups(groups: list[Any]) -> None:
    invalid: list[str] = []
    for group in groups:
        for trajectory in group.trajectories:
            scenario_id = str(trajectory.metadata.get("scenario_id"))
            if trajectory.metrics.get("multi_tool_calls") != 0.0:
                invalid.append(f"{scenario_id}: multi-tool")
            if trajectory.metrics.get("strict_replay") != 1.0:
                invalid.append(f"{scenario_id}: no strict replay")
            if trajectory.metrics.get("reward_finalized_once") != 1.0:
                invalid.append(f"{scenario_id}: reward not finalized once")
            if trajectory.metrics.get("terminated") != 1.0:
                invalid.append(f"{scenario_id}: not terminated")
    if invalid:
        raise RuntimeError("invalid rollout group: " + "; ".join(invalid[:8]))


async def _gather_groups(
    art: Any,
    *,
    scenarios: list[Any],
    model: Any,
    client: Any,
    args: argparse.Namespace,
    trials: int,
    seed_base: int,
) -> list[Any]:
    semaphore = asyncio.Semaphore(args.rollout_concurrency)
    user_args = user_llm_args()

    async def one(scenario: Any, seed: int) -> Any:
        async with semaphore:
            return await rollout(
                scenario,
                model,
                client=client,
                max_turns=args.max_turns,
                max_completion_tokens=args.max_completion_tokens,
                max_model_len=args.max_model_len,
                temperature=1.0,
                policy_seed=seed,
                user_model_name=args.user_model,
                user_chat_completion_kwargs={**user_args, "seed": seed},
            )

    groups = await art.gather_trajectory_groups(
        [
            art.TrajectoryGroup(
                [
                    one(scenario, seed_base + group_index * trials + trial)
                    for trial in range(trials)
                ]
            )
            for group_index, scenario in enumerate(scenarios)
        ]
    )
    _validate_official_groups(groups)
    return groups


async def _load_scenarios(tau_bench: Any, client: Any) -> tuple[list[Any], list[Any]]:
    train = await tau_bench.get_scenarios(
        domain="telecom",
        split="train-core",
        client=client,
    )
    dev = await tau_bench.get_scenarios(domain="telecom", split="dev", client=client)
    assert_training_split_clean([scenario.task.id for scenario in train])
    assert_no_labels(train)
    assert_no_labels(dev)
    return train, dev


async def _run_preflight(
    art: Any,
    tau_bench: Any,
    LocalBackend: Any,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    contract: dict[str, Any],
) -> None:
    _require_fresh_lineage(args)
    client = tau_bench.TauBenchClient(base_url=args.shim_url)
    backend = model = None
    try:
        train, _ = await _load_scenarios(tau_bench, client)
        backend, model = await _register_model(art, LocalBackend, args)
        initial_step = await model.get_step()
        if initial_step != 0:
            raise RuntimeError(f"preflight must start at step 0, got {initial_step}")

        records = await sample_probe_records(
            openai_client=model.openai_client(),
            served_model=model.get_inference_name(),
            system_prompt=contract["system_prompt"],
            tools=contract["tools"],
            seed=args.seed,
        )
        await backend.close()
        backend = None
        _finish_wandb(model)

        logprob = evaluate_probe_records(records, tools=contract["tools"])
        manifest["logprob_gate"] = logprob
        manifest["initial_step"] = initial_step
        _write_json(manifest_path, manifest)
        if logprob["status"] != "passed":
            raise RuntimeError("step-0 logprob gate failed")

        # Reopen the same untouched step-0 checkpoint for the rollout-only gate.
        backend, model = await _register_model(art, LocalBackend, args)
        if await model.get_step() != 0:
            raise RuntimeError("preflight lineage changed during the logprob gate")
        groups = await _gather_groups(
            art,
            scenarios=train[:2],
            model=model,
            client=client,
            args=args,
            trials=args.group_size,
            seed_base=args.seed + 10_000,
        )
        stats = group_stats(groups)
        await model.log(groups, split="preflight", step=0)
        final_step = await model.get_step()
        if final_step != 0:
            raise RuntimeError("rollout-only preflight performed an update")
        manifest.update(
            {
                "status": "passed",
                "completed_at": _now(),
                "final_step": final_step,
                "rollout_only": {
                    "status": "passed",
                    "test_split_locked": True,
                    "strict_replay": True,
                    "reward_finalized_once": True,
                    "multi_tool_calls": 0,
                    "stats": stats,
                },
                "wandb_url": _wandb_url(model),
            }
        )
        _write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "failed_at": _now(), "error": str(exc)})
        _write_json(manifest_path, manifest)
        raise
    finally:
        if backend is not None:
            await backend.close()
        if model is not None:
            _finish_wandb(model)
        await client.close()


def _validate_smoke_gate(payload: dict[str, Any], expected_contract: str) -> None:
    checks = {
        "phase": payload.get("phase") == "smoke",
        "status": payload.get("status") == "passed",
        "semantic contract": payload.get("semantic_contract_sha256") == expected_contract,
        "fresh step": payload.get("initial_step") == 0,
        "one update": payload.get("final_step") == 1,
        "checkpoint": bool(payload.get("checkpoint_path")),
        "strict replay": payload.get("strict_replay") is True,
        "optimizer update": payload.get("optimizer_update") is True,
        "W&B run": bool(payload.get("wandb_url")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("smoke gate is not valid: " + ", ".join(failed))


async def _run_smoke(
    art: Any,
    tau_bench: Any,
    LocalBackend: Any,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    _require_fresh_lineage(args)
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is missing")
    preflight = _read_json(args.preflight_manifest, "preflight manifest")
    validate_preflight_gate(preflight, manifest["semantic_contract_sha256"])
    validate_matching_protocol(preflight, manifest, "preflight")
    client = tau_bench.TauBenchClient(base_url=args.shim_url)
    backend = model = None
    try:
        train, _ = await _load_scenarios(tau_bench, client)
        backend, model = await _register_model(art, LocalBackend, args)
        initial_step = await model.get_step()
        if initial_step != 0:
            raise RuntimeError(f"smoke must start at step 0, got {initial_step}")
        groups = await _gather_groups(
            art,
            scenarios=train[: args.groups_per_step],
            model=model,
            client=client,
            args=args,
            trials=args.group_size,
            seed_base=args.seed + 20_000,
        )
        stats = group_stats(groups)
        if stats["mixed"] < 1:
            raise RuntimeError(
                "smoke has no within-group reward variance; refusing a skipped update"
            )
        result = await backend.train(
            model,
            groups,
            learning_rate=args.learning_rate,
            loss_fn=args.loss_fn,
            kl_penalty_coef=args.kl_penalty_coef,
        )
        if result.step != 1:
            raise RuntimeError(f"smoke must make exactly one update, got step {result.step}")
        checkpoint_path = _require_checkpoint(result.checkpoint_path, result.step)
        if result.metrics.get("data/step_num_groups_trainable", 0.0) < 1.0:
            raise RuntimeError("ART reported no trainable group in the smoke update")
        wandb_url = _wandb_url(model)
        if not wandb_url:
            raise RuntimeError("smoke update has no W&B run URL")
        await model.log(groups, split="smoke", step=result.step)
        await model.log(split="smoke", step=result.step, metrics=result.metrics)
        manifest.update(
            {
                "status": "passed",
                "completed_at": _now(),
                "initial_step": initial_step,
                "final_step": result.step,
                "checkpoint_path": checkpoint_path,
                "strict_replay": True,
                "optimizer_update": True,
                "stats": stats,
                "wandb_url": wandb_url,
            }
        )
        _write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "failed_at": _now(), "error": str(exc)})
        _write_json(manifest_path, manifest)
        raise
    finally:
        if backend is not None:
            await backend.close()
        if model is not None:
            _finish_wandb(model)
        await client.close()


def _scenario_for_slot(scenarios: list[Any], slot: int, seed: int) -> Any:
    epoch, offset = divmod(slot, len(scenarios))
    order = list(range(len(scenarios)))
    random.Random(seed + epoch).shuffle(order)
    return scenarios[order[offset]]


async def _validate_dev(
    art: Any,
    *,
    dev: list[Any],
    model: Any,
    client: Any,
    args: argparse.Namespace,
    step: int,
) -> dict[str, Any]:
    groups = await _gather_groups(
        art,
        scenarios=dev,
        model=model,
        client=client,
        args=args,
        trials=args.val_trials,
        # Common random numbers: every checkpoint sees the same task/trial
        # seeds, so selection is not confounded by a different random draw.
        seed_base=args.seed + 1_000_000,
    )
    rewards = [float(t.reward) for group in groups for t in group.trajectories]
    await model.log(groups, split="dev", step=step)
    return {
        "step": step,
        "rollouts": len(rewards),
        "avg_reward": sum(rewards) / max(len(rewards), 1),
        "stats": group_stats(groups),
    }


async def _run_train(
    art: Any,
    tau_bench: Any,
    LocalBackend: Any,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    contract_hash = manifest["semantic_contract_sha256"]
    preflight = _read_json(args.preflight_manifest, "preflight manifest")
    smoke = _read_json(args.smoke_manifest, "smoke manifest")
    validate_preflight_gate(preflight, contract_hash)
    _validate_smoke_gate(smoke, contract_hash)
    validate_matching_protocol(preflight, manifest, "preflight")
    validate_matching_protocol(smoke, manifest, "smoke")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is missing")

    existing = manifest_path.exists()
    if existing:
        previous = json.loads(manifest_path.read_text())
        validate_resume_contract(previous, manifest)
        manifest = previous
        manifest["status"] = "running"
        manifest["resumed_at"] = _now()
    else:
        _require_fresh_lineage(args)
    _write_json(manifest_path, manifest)

    client = tau_bench.TauBenchClient(base_url=args.shim_url)
    backend = model = None
    try:
        train, dev = await _load_scenarios(tau_bench, client)
        backend, model = await _register_model(art, LocalBackend, args)
        if not _wandb_url(model):
            raise RuntimeError("formal run has no W&B run URL")
        start_step = await model.get_step()
        recorded_step = int(manifest.get("last_completed_step", start_step))
        if recorded_step != start_step:
            raise RuntimeError(
                f"checkpoint/manifest resume mismatch: {start_step} != {recorded_step}"
            )
        manifest.setdefault("initial_step", start_step)
        manifest.setdefault("train_steps", [])
        manifest.setdefault("dev_evaluations", [])
        recent_mixed = [
            int(item["stats"]["mixed"]) for item in manifest["train_steps"][-SPARSE_REWARD_WINDOW:]
        ]

        for step in range(start_step, args.steps):
            selected = [
                _scenario_for_slot(
                    train,
                    step * args.groups_per_step + index,
                    args.seed,
                )
                for index in range(args.groups_per_step)
            ]
            groups = await _gather_groups(
                art,
                scenarios=selected,
                model=model,
                client=client,
                args=args,
                trials=args.group_size,
                seed_base=args.seed + step * args.groups_per_step * args.group_size,
            )
            stats = group_stats(groups)
            recent_mixed.append(int(stats["mixed"]))
            recent_mixed = recent_mixed[-SPARSE_REWARD_WINDOW:]
            print(f"step {step} rollout: {json.dumps(stats, sort_keys=True)}", flush=True)

            result = await backend.train(
                model,
                groups,
                learning_rate=args.learning_rate,
                loss_fn=args.loss_fn,
                kl_penalty_coef=args.kl_penalty_coef,
            )
            if result.step != step + 1:
                raise RuntimeError(
                    f"ART step drift: expected {step + 1}, got {result.step}"
                )
            checkpoint_path = _require_checkpoint(result.checkpoint_path, result.step)
            await model.log(groups, split="train", step=result.step)
            await model.log(split="train", step=result.step, metrics=result.metrics)
            step_record = {
                "rollout_step": step,
                "checkpoint_step": result.step,
                "checkpoint_path": checkpoint_path,
                "stats": stats,
                "metrics": {key: float(value) for key, value in result.metrics.items()},
                "completed_at": _now(),
            }
            manifest["train_steps"].append(step_record)
            manifest["last_completed_step"] = result.step
            manifest["latest_checkpoint_path"] = checkpoint_path
            manifest["wandb_url"] = _wandb_url(model)

            if args.val_every and result.step % args.val_every == 0:
                dev_result = await _validate_dev(
                    art,
                    dev=dev,
                    model=model,
                    client=client,
                    args=args,
                    step=result.step,
                )
                manifest["dev_evaluations"].append(dev_result)
                best = max(
                    manifest["dev_evaluations"],
                    key=lambda item: (item["avg_reward"], -item["step"]),
                )
                manifest["selected_checkpoint"] = {
                    **best,
                    "selection_rule": "highest frozen-dev average reward; ties choose earliest",
                    "checkpoint_path": _checkpoint_path(args, int(best["step"])),
                }
                _require_checkpoint(
                    manifest["selected_checkpoint"]["checkpoint_path"],
                    int(best["step"]),
                )
                print(
                    f"step {result.step} dev: "
                    f"{dev_result['avg_reward']:.6f} over {dev_result['rollouts']}",
                    flush=True,
                )
            _write_json(manifest_path, manifest)

            if (
                len(recent_mixed) == SPARSE_REWARD_WINDOW
                and sum(recent_mixed) <= 1
            ):
                manifest.update(
                    {
                        "status": "stopped_sparse_reward",
                        "stopped_at": _now(),
                        "reason": (
                            f"at most one mixed-reward group in the last "
                            f"{SPARSE_REWARD_WINDOW} update steps"
                        ),
                    }
                )
                _write_json(manifest_path, manifest)
                raise RuntimeError(
                    "GRPO reward groups are too sparse; protocol decision required before SFT"
                )

        if not manifest["dev_evaluations"] or manifest["dev_evaluations"][-1][
            "step"
        ] != args.steps:
            dev_result = await _validate_dev(
                art,
                dev=dev,
                model=model,
                client=client,
                args=args,
                step=args.steps,
            )
            manifest["dev_evaluations"].append(dev_result)
            best = max(
                manifest["dev_evaluations"],
                key=lambda item: (item["avg_reward"], -item["step"]),
            )
            manifest["selected_checkpoint"] = {
                **best,
                "selection_rule": "highest frozen-dev average reward; ties choose earliest",
                "checkpoint_path": _checkpoint_path(args, int(best["step"])),
            }
            _require_checkpoint(
                manifest["selected_checkpoint"]["checkpoint_path"],
                int(best["step"]),
            )
        manifest.update(
            {
                "status": "passed",
                "completed_at": _now(),
                "final_step": await model.get_step(),
                "wandb_url": _wandb_url(model),
            }
        )
        _write_json(manifest_path, manifest)
    except BaseException as exc:
        if manifest.get("status") != "stopped_sparse_reward":
            manifest.update({"status": "failed", "failed_at": _now(), "error": str(exc)})
            _write_json(manifest_path, manifest)
        raise
    finally:
        if backend is not None:
            await backend.close()
        if model is not None:
            _finish_wandb(model)
        await client.close()


async def run(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    assert_pinned_art_api(repo_root / "third_party/ART")
    if _git_commit(repo_root / "third_party/tau2-bench") != TAU2_COMMIT:
        raise RuntimeError("tau2 commit drift")
    if args.user_model != DEFAULT_USER_MODEL:
        raise RuntimeError(
            f"formal protocol requires user simulator {DEFAULT_USER_MODEL}"
        )
    if os.environ.get("SHIM_ALLOW_EVAL_SPLITS"):
        raise RuntimeError("SHIM_ALLOW_EVAL_SPLITS must not be set during GPU work")
    if not _test_split_is_locked(args.shim_url):
        raise RuntimeError("official test split is not locked; refusing GPU work")
    contract = _fetch_json(
        args.shim_url,
        "/training-contract",
        {"domain": "telecom"},
    )

    os.environ["HF_HOME"] = str(args.hf_cache)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    snapshot = prepare_pinned_snapshot(args.hf_cache)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
    )
    if not isinstance(tokenizer.chat_template, str):
        raise RuntimeError("pinned tokenizer has no chat template")
    semantic_hash = semantic_contract_sha256(
        system_prompt=contract["system_prompt"],
        tools=contract["tools"],
        tokenizer_chat_template=tokenizer.chat_template,
    )
    token_budget = measure_dev_token_budget(
        repo_root=repo_root,
        tokenizer=tokenizer,
        system_prompt=contract["system_prompt"],
        tools=contract["tools"],
    )
    validate_context_capacity(
        token_budget,
        max_model_len=args.max_model_len,
        max_completion_tokens=args.max_completion_tokens,
    )
    del tokenizer

    runtime = _runtime(args)
    manifest_path = args.out / f"{args.phase}_manifest.json"
    manifest = _base_manifest(
        args,
        runtime,
        semantic_hash=semantic_hash,
        snapshot=snapshot,
        token_budget=token_budget,
    )
    if args.phase != "train":
        _write_json(manifest_path, manifest)

    import art
    from art import tau_bench
    from art.local import LocalBackend

    if args.phase == "preflight":
        await _run_preflight(
            art,
            tau_bench,
            LocalBackend,
            args,
            manifest,
            manifest_path,
            contract,
        )
    elif args.phase == "smoke":
        await _run_smoke(
            art,
            tau_bench,
            LocalBackend,
            args,
            manifest,
            manifest_path,
        )
    else:
        await _run_train(
            art,
            tau_bench,
            LocalBackend,
            args,
            manifest,
            manifest_path,
        )
    print(json.dumps(json.loads(manifest_path.read_text()), indent=2), flush=True)


def main() -> None:
    load_dotenv()
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
