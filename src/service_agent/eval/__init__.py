"""Evaluation harness: ablation arms over tau2's native runner.

The native runner owns everything that must not be reimplemented -- per-trial
seeds, concurrency, retries, checkpointing, the evaluator. This package only
registers the governed agent arms, aggregates governance metrics, and pins
run configuration.
"""

from service_agent.eval.registration import register_governed_agents

__all__ = ["register_governed_agents"]
