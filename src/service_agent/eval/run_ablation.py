"""Run one ablation arm on the frozen dev set through tau2's native runner.

Usage (from the repo root, .env supplies API keys):

    uv run python -m service_agent.eval.run_ablation \
        --arm h1 --tasks smoke3 --trials 1 \
        --agent-llm openai/qwen-local --agent-api-base http://127.0.0.1:8080/v1 \
        --user-llm deepseek/deepseek-v4-pro \
        --out results/dev/h1_smoke

Every run directory is self-describing: run_config.json (exact parameters),
results.json (tau2's native format, checkpointed), audit/ (per-simulation
governance records for H1/H2), metrics.json (official + governance summary).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from dotenv import load_dotenv

ARM_AGENTS = {
    "h0": "llm_agent",
    "h1": "governed_llm_agent_h1",
    "h2": "governed_llm_agent_h2",
}


def smoke3_ids(dev_ids: list[str]) -> list[str]:
    """One dev task per issue family, deterministically the first of each."""
    picked: dict[str, str] = {}
    for tid in dev_ids:
        family = tid.split("]")[0].strip("[")
        picked.setdefault(family, tid)
    return sorted(picked.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARM_AGENTS), required=True)
    parser.add_argument("--tasks", choices=["smoke3", "dev", "train-core"], default="dev")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--agent-llm", required=True)
    parser.add_argument("--agent-api-base", default=None)
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument("--user-llm", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--user-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv()

    # Heavy imports after arg parsing so --help stays fast.
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.batch import run_tasks

    from service_agent.eval.metrics import audit_summary, summarize_results
    from service_agent.eval.registration import register_governed_agents, set_audit_dir
    from service_agent.splits import load_frozen_dev_ids, train_core_ids

    register_governed_agents()

    if args.tasks == "dev":
        task_ids = load_frozen_dev_ids()
    elif args.tasks == "smoke3":
        task_ids = smoke3_ids(load_frozen_dev_ids())
    else:
        task_ids = train_core_ids()

    all_tasks = registry.get_tasks_loader("telecom")()
    tasks = [t for t in all_tasks if t.id in set(task_ids)]
    assert len(tasks) == len(task_ids), "some frozen task IDs were not found upstream"

    agent_llm_args = {"temperature": args.agent_temperature}
    if args.agent_api_base:
        agent_llm_args["api_base"] = args.agent_api_base

    config = TextRunConfig(
        domain="telecom",
        agent=ARM_AGENTS[args.arm],
        llm_agent=args.agent_llm,
        llm_args_agent=agent_llm_args,
        user="user_simulator",
        llm_user=args.user_llm,
        llm_args_user={"temperature": args.user_temperature},
        num_trials=args.trials,
        seed=args.seed,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        auto_resume=True,
        log_level="ERROR",
    )

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    audit_dir = out / "audit"
    audit_dir.mkdir(exist_ok=True)
    set_audit_dir(audit_dir)

    (out / "run_config.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "task_ids": task_ids,
                "cli_args": {k: str(v) for k, v in vars(args).items()},
                "text_run_config": json.loads(config.model_dump_json()),
            },
            indent=2,
        )
        + "\n"
    )

    results = run_tasks(
        config,
        tasks,
        save_path=out / "results.json",
        save_dir=out,
        evaluation_type=EvaluationType.ALL,
    )

    metrics = summarize_results(results)
    payload = {
        "arm": args.arm,
        "metrics": dataclasses.asdict(metrics),
        "live_gate_audit": audit_summary(audit_dir) if args.arm != "h0" else None,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
