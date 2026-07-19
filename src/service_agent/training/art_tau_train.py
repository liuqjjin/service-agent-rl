"""Dual-control GRPO training on tau2 telecom through the shim.

Runs on the GPU box only (ART is not installed on the development machine;
see runbooks/autodl.md for the full launch sequence). The topology:

    this script (ART client + trainable model)
      -> shim (serve/tau2_shim.py) on localhost: tau2 env + user simulator + evaluator
      -> ART backend (vLLM + LoRA) serving and training the policy

Protocol guards run before any training step, in code rather than in
documentation: scenarios must come from train-core (the shim already 403s
test/full/base, and this script re-asserts against the frozen split files),
and scenario payloads must carry no evaluation labels. ART's own dev example
trains on split="test" -- that is exactly the mistake these asserts exist to
make impossible here.

Reward is the official tau2 evaluator's, returned by the shim exactly once
per episode. Training is in the native H0 environment: no governance in the
loop, so the 2x2's model effect stays attributable to training alone
(CLAUDE.md hard rule 4).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random

from dotenv import load_dotenv

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_USER_MODEL = "deepseek/deepseek-v4-pro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="service-agent-grpo")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--project", default="service-agent")
    parser.add_argument("--shim-url", default=os.environ.get("TAU_BENCH_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.0)
    parser.add_argument("--loss-fn", choices=["cispo", "ppo"], default="ppo")
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--val-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="2 tasks, 2 steps, tiny groups")
    return parser.parse_args()


def assert_training_split_clean(scenario_ids: list[str]) -> None:
    """Belt and braces on top of the shim's 403 guard: the scenario IDs the
    trainer is about to use must be exactly a subset of frozen train-core."""
    from service_agent.splits import load_frozen_dev_ids, load_split_ids, train_core_ids

    ids = set(scenario_ids)
    splits = load_split_ids()
    assert ids.isdisjoint(splits["test"]), "training scenarios intersect the official test split"
    assert ids.isdisjoint(set(load_frozen_dev_ids())), "training scenarios intersect the dev set"
    assert ids <= set(train_core_ids()), "training scenarios outside train-core"


def assert_no_labels(scenarios) -> None:
    for s in scenarios:
        task = s.task.model_dump() if hasattr(s.task, "model_dump") else dict(s.task)
        for label_field in ("evaluation_criteria", "user_scenario", "initial_state", "ticket", "description"):
            assert not task.get(label_field), (
                f"scenario {task.get('id')} carries {label_field}; the shim must redact labels"
            )


def user_llm_args() -> dict:
    # The fixed simulator (DECISIONS.md D1): non-thinking, temperature 0.
    # The shim forwards these to tau2's UserSimulator -> LiteLLM.
    args = {"temperature": 0.0, "extra_body": {"thinking": {"type": "disabled"}}}
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        args["api_key"] = api_key
    return args


async def run(args: argparse.Namespace) -> None:
    # ART imports live here: this module must stay importable on machines
    # without the training stack (tests import the assert helpers above).
    import art
    from art import tau_bench
    from art.local import LocalBackend

    random.seed(args.seed)

    client = tau_bench.TauBenchClient(base_url=args.shim_url)
    train_scenarios = await tau_bench.get_scenarios(
        domain="telecom", split="train-core", client=client
    )
    dev_scenarios = await tau_bench.get_scenarios(
        domain="telecom", split="dev", client=client
    )
    assert_training_split_clean([s.task.id for s in train_scenarios])
    assert_no_labels(train_scenarios)
    assert_no_labels(dev_scenarios)

    if args.smoke:
        train_scenarios = train_scenarios[:2]
        dev_scenarios = dev_scenarios[:2]
        args.steps = 2
        args.group_size = max(2, min(args.group_size, 4))
        args.groups_per_step = 1

    backend = LocalBackend()
    model = art.TrainableModel(
        name=args.model_name,
        project=args.project,
        base_model=args.base_model,
    )
    await model.register(backend)
    start_step = await model.get_step()
    print(f"registered {args.model_name} (base {args.base_model}), starting at step {start_step}")

    def rollout_kwargs() -> dict:
        return {
            "client": client,
            "max_turns": args.max_turns,
            "user_model_name": args.user_model,
            "user_chat_completion_kwargs": user_llm_args(),
        }

    order = list(range(len(train_scenarios)))
    for step in range(start_step, args.steps):
        # Deterministic task rotation given the seed: reshuffle each epoch.
        if (step * args.groups_per_step) % len(order) == 0:
            random.shuffle(order)
        picked = [
            train_scenarios[order[(step * args.groups_per_step + i) % len(order)]]
            for i in range(args.groups_per_step)
        ]

        groups = await art.gather_trajectory_groups(
            art.TrajectoryGroup(
                tau_bench.rollout(scenario, model, **rollout_kwargs())
                for _ in range(args.group_size)
            )
            for scenario in picked
        )

        # Group-quality accounting before the update: all-zero/all-one groups
        # contribute no gradient under GRPO; if mixed groups dry up, fix task
        # difficulty or warm-start, do not touch the learning rate (see the
        # training playbook in runbooks/autodl.md).
        stats = group_stats(groups)
        print(f"step {step}: {json.dumps(stats)}")

        result = await backend.train(
            model,
            groups,
            learning_rate=args.learning_rate,
            loss_fn=args.loss_fn,
            kl_penalty_coef=args.kl_penalty_coef,
        )
        await model.log(groups, split="train", step=result.step)
        await model.log(split="train", step=result.step, metrics=result.metrics)

        if args.val_every and (step + 1) % args.val_every == 0:
            val = await art.gather_trajectory_groups(
                art.TrajectoryGroup(
                    tau_bench.rollout(scenario, model, **rollout_kwargs())
                    for _ in range(args.val_trials)
                )
                for scenario in dev_scenarios
            )
            rewards = [t.reward for g in val for t in g.trajectories]
            print(
                f"step {step}: dev avg_reward="
                f"{sum(rewards) / max(len(rewards), 1):.3f} over {len(rewards)} rollouts"
            )
            await model.log(val, split="dev", step=result.step)


def group_stats(groups) -> dict:
    total = mixed = all_zero = all_one = 0
    rewards = []
    for group in groups:
        rs = [t.reward for t in group.trajectories]
        rewards.extend(rs)
        total += 1
        if all(r <= 0.0 for r in rs):
            all_zero += 1
        elif all(r >= 1.0 for r in rs):
            all_one += 1
        else:
            mixed += 1
    return {
        "groups": total,
        "mixed": mixed,
        "all_zero": all_zero,
        "all_one": all_one,
        "reward_mean": round(sum(rewards) / max(len(rewards), 1), 4),
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
