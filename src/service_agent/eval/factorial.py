"""Arm comparison: paired bootstrap, factorial effects, failure taxonomy.

Statistics follow the protocol in reports/baseline_protocol.md: the task is
the statistical unit, trials are paired across arms by (task, trial) because
every arm runs the same frozen dev tasks with the same trial seeds, and
uncertainty comes from a bootstrap over tasks (resampling tasks, not trials,
respects the clustering: trials of one task share its difficulty).

The failure taxonomy is deliberately mechanical -- every label is computable
from the simulation record and the offline governance replay, no human
judgment involved, so the same classifier can run on every arm and every
model. Categories are assigned by first match in priority order; `notes`
carry the evidence. Task labels (expected actions) are used here for
*analysis* of finished runs, which is fine: the leak rules govern model
inputs, not offline reports.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from tau2.data_model.simulation import Results, SimulationRun

from service_agent.eval.metrics import analyze_trajectory

BOOTSTRAP_RESAMPLES = 10_000


# --- paired comparison --------------------------------------------------------


def rewards_by_task(results: Results) -> dict[str, list[float]]:
    per_task: dict[str, list[float]] = defaultdict(list)
    for sim in results.simulations:
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        per_task[sim.task_id].append(float(reward or 0.0))
    return dict(per_task)


@dataclass
class PairedComparison:
    mean_a: float
    mean_b: float
    diff: float  # b - a
    ci_low: float
    ci_high: float
    tasks: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def paired_bootstrap(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> PairedComparison:
    """Bootstrap the mean per-task reward difference (b - a) over shared tasks."""
    tasks = sorted(set(a) & set(b))
    assert tasks, "no shared tasks to compare"
    diffs = [
        sum(b[t]) / len(b[t]) - sum(a[t]) / len(a[t]) for t in tasks
    ]
    rng = random.Random(seed)
    n = len(tasks)
    stats = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n)
    stats.sort()
    mean_a = sum(sum(v) / len(v) for v in a.values()) / len(a)
    mean_b = sum(sum(v) / len(v) for v in b.values()) / len(b)
    return PairedComparison(
        mean_a=mean_a,
        mean_b=mean_b,
        diff=sum(diffs) / n,
        ci_low=stats[int(0.025 * resamples)],
        ci_high=stats[int(0.975 * resamples)],
        tasks=n,
    )


def factorial_effects(cells: dict[str, dict[str, list[float]]]) -> dict[str, float]:
    """The 2x2 arithmetic from the design docs. cells keys: h0, hbest, rl,
    hbest_rl -- each a rewards_by_task mapping."""

    def mean(cell: dict[str, list[float]]) -> float:
        return sum(sum(v) / len(v) for v in cell.values()) / len(cell)

    h0, hbest, rl, hbest_rl = (
        mean(cells["h0"]),
        mean(cells["hbest"]),
        mean(cells["rl"]),
        mean(cells["hbest_rl"]),
    )
    return {
        "harness_effect": hbest - h0,
        "model_effect": rl - h0,
        "combined_gain": hbest_rl - h0,
        "interaction": (hbest_rl - rl) - (hbest - h0),
    }


# --- failure taxonomy ---------------------------------------------------------

TAXONOMY_PRIORITY = [
    "infrastructure_error",
    "protocol_error",
    "max_steps_loop",
    "too_many_tool_errors",
    "policy_violation",
    "duplicate_side_effect",
    "missing_agent_action",
    "missing_user_action",
    "wrong_final_state",
]


def classify_failure(sim: SimulationRun, task) -> tuple[str, str]:
    """(category, evidence) for a failed simulation."""
    termination = str(sim.termination_reason)
    if "INFRASTRUCTURE" in termination.upper():
        return "infrastructure_error", termination
    if "AGENT_ERROR" in termination.upper() or "USER_ERROR" in termination.upper():
        return "protocol_error", termination
    if "MAX_STEPS" in termination.upper():
        return "max_steps_loop", f"{len(sim.messages)} messages"
    if "TOO_MANY_ERRORS" in termination.upper():
        return "too_many_tool_errors", termination

    analysis = analyze_trajectory(sim.messages)
    if analysis.unauthorized_executed_writes:
        return (
            "policy_violation",
            f"unauthorized writes: {dict(analysis.unauthorized_reasons)}",
        )
    if analysis.duplicate_side_effects:
        return "duplicate_side_effect", f"{analysis.duplicate_side_effects} duplicates"

    expected = (
        task.evaluation_criteria.actions if task.evaluation_criteria else None
    ) or []
    executed = {
        tc.name
        for m in sim.messages
        if getattr(m, "tool_calls", None)
        for tc in m.tool_calls
    }
    missing_agent = [
        a.name for a in expected if a.requestor == "assistant" and a.name not in executed
    ]
    missing_user = [
        a.name for a in expected if a.requestor == "user" and a.name not in executed
    ]
    if missing_agent:
        return "missing_agent_action", f"never called: {sorted(set(missing_agent))}"
    if missing_user:
        return "missing_user_action", f"user never did: {sorted(set(missing_user))}"
    return "wrong_final_state", "all expected actions present but end state wrong"


def taxonomy(results: Results, tasks_by_id: dict) -> dict[str, int]:
    counts: Counter = Counter()
    for sim in results.simulations:
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        if reward and reward >= 1.0:
            continue
        category, _ = classify_failure(sim, tasks_by_id[sim.task_id])
        counts[category] += 1
    return dict(counts)


def failure_examples(
    results: Results, tasks_by_id: dict, per_category: int = 2
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for sim in results.simulations:
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        if reward and reward >= 1.0:
            continue
        category, evidence = classify_failure(sim, tasks_by_id[sim.task_id])
        if len(out[category]) < per_category:
            out[category].append(
                {"task_id": sim.task_id, "evidence": evidence, "sim_id": sim.id}
            )
    return dict(out)


# --- pass^k across arms -------------------------------------------------------


def pass_hat_ks(results: Results, ks: Iterable[int] = (1, 2, 4)) -> dict[int, float]:
    from tau2.metrics.agent_metrics import pass_hat_k

    per_task: dict[str, list[bool]] = defaultdict(list)
    for sim in results.simulations:
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        per_task[sim.task_id].append(bool(reward and reward >= 1.0))
    out = {}
    for k in ks:
        vals = [
            pass_hat_k(len(trials), sum(trials), k)
            for trials in per_task.values()
            if len(trials) >= k
        ]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out
