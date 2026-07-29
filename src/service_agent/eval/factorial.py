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
carry the evidence. Finished-run evaluator checks may be used here for
offline analysis, but only when their component belongs to the task's actual
reward basis.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tau2.data_model.simulation import Results, SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType

from service_agent.eval.metrics import analyze_trajectory

BOOTSTRAP_RESAMPLES = 10_000
FACTORIAL_CELLS = ("base_h0", "base_h2", "rl_h0", "rl_h2")
CONTRAST_COEFFICIENTS = {
    "harness_effect_base": {"base_h0": -1.0, "base_h2": 1.0},
    "harness_effect_rl": {"rl_h0": -1.0, "rl_h2": 1.0},
    "model_effect_native": {"base_h0": -1.0, "rl_h0": 1.0},
    "model_effect_governed": {"base_h2": -1.0, "rl_h2": 1.0},
    "combined_gain": {"base_h0": -1.0, "rl_h2": 1.0},
    "interaction": {
        "base_h0": 1.0,
        "base_h2": -1.0,
        "rl_h0": -1.0,
        "rl_h2": 1.0,
    },
}


def _is_successful(reward: float) -> bool:
    """Match tau2.metrics.agent_metrics.is_successful without importing pandas."""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def _reward(sim: SimulationRun, *, strict: bool = False) -> float:
    if sim.reward_info is None:
        if strict:
            raise ValueError(f"simulation {sim.id} has no reward_info")
        return 0.0
    reward = float(sim.reward_info.reward)
    if strict and (not math.isfinite(reward) or not 0.0 <= reward <= 1.0 + 1e-6):
        raise ValueError(f"simulation {sim.id} has invalid reward {reward}")
    return reward


# --- final factorial grid -----------------------------------------------------


@dataclass(frozen=True, order=True)
class EpisodeKey:
    """The common-random-number identity of one final-evaluation episode."""

    task_id: str
    trial: int
    seed: int


@dataclass(frozen=True)
class ValidatedFactorialGrid:
    """A complete four-cell grid after all fail-closed checks have passed."""

    simulations_by_cell: dict[str, dict[EpisodeKey, SimulationRun]]
    task_ids: tuple[str, ...]
    trials: tuple[int, ...]
    seeds_by_trial: dict[int, int]

    @property
    def episode_keys(self) -> tuple[EpisodeKey, ...]:
        reference = self.simulations_by_cell[FACTORIAL_CELLS[0]]
        return tuple(sorted(reference))


def trial_seeds(base_seed: int, num_trials: int) -> dict[int, int]:
    """Reproduce tau2 runner.batch's per-trial seed schedule without global state."""
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")
    rng = random.Random(base_seed)
    return {trial: rng.randint(0, 1_000_000) for trial in range(num_trials)}


def validate_factorial_grid(
    cells: Mapping[str, Results],
    *,
    expected_task_count: int = 40,
    expected_trial_count: int = 8,
    base_seed: int = 42,
    expected_task_ids: Iterable[str] | None = None,
) -> ValidatedFactorialGrid:
    """Validate the exact 40x8x4 common-random-number result grid.

    Counts alone are insufficient: every cell must contain the same unique
    ``(task_id, trial, seed)`` keys.  Infrastructure errors are incomplete
    observations, not zero-reward scientific outcomes, and therefore fail the
    gate instead of being silently dropped by tau2's official metrics.
    """
    if expected_task_count <= 0:
        raise ValueError("expected_task_count must be positive")
    if expected_trial_count <= 0:
        raise ValueError("expected_trial_count must be positive")
    if set(cells) != set(FACTORIAL_CELLS):
        raise ValueError(
            f"factorial cells must be exactly {FACTORIAL_CELLS}; got {tuple(cells)}"
        )

    expected_ids = None
    if expected_task_ids is not None:
        expected_ids = set(expected_task_ids)
        if len(expected_ids) != expected_task_count:
            raise ValueError(
                "expected_task_ids does not contain expected_task_count unique tasks"
            )

    seeds = trial_seeds(base_seed, expected_trial_count)
    trials = tuple(range(expected_trial_count))
    simulations_by_cell: dict[str, dict[EpisodeKey, SimulationRun]] = {}
    reference_tasks: set[str] | None = expected_ids
    expected_simulations = expected_task_count * expected_trial_count

    for cell in FACTORIAL_CELLS:
        results = cells[cell]
        if results.info.num_trials != expected_trial_count:
            raise ValueError(
                f"{cell} info.num_trials={results.info.num_trials}, "
                f"expected {expected_trial_count}"
            )
        if results.info.seed != base_seed:
            raise ValueError(f"{cell} info.seed={results.info.seed}, expected {base_seed}")
        if len(results.simulations) != expected_simulations:
            raise ValueError(
                f"{cell} has {len(results.simulations)} simulations, "
                f"expected {expected_simulations}"
            )

        keyed: dict[EpisodeKey, SimulationRun] = {}
        simulation_ids: set[str] = set()
        for sim in results.simulations:
            if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
                raise ValueError(
                    f"{cell} contains infrastructure_error for task={sim.task_id}, "
                    f"trial={sim.trial}"
                )
            if sim.trial is None or isinstance(sim.trial, bool):
                raise ValueError(f"{cell} simulation {sim.id} has no integer trial")
            if sim.seed is None or isinstance(sim.seed, bool):
                raise ValueError(f"{cell} simulation {sim.id} has no integer seed")
            if sim.trial not in seeds:
                raise ValueError(f"{cell} simulation {sim.id} has unexpected trial {sim.trial}")
            if sim.seed != seeds[sim.trial]:
                raise ValueError(
                    f"{cell} task={sim.task_id}, trial={sim.trial} has seed={sim.seed}, "
                    f"expected {seeds[sim.trial]}"
                )
            if sim.id in simulation_ids:
                raise ValueError(f"{cell} has duplicate simulation id {sim.id}")
            simulation_ids.add(sim.id)
            key = EpisodeKey(sim.task_id, sim.trial, sim.seed)
            if key in keyed:
                raise ValueError(f"{cell} has duplicate episode key {key}")
            _reward(sim, strict=True)
            if sim.messages is None:
                raise ValueError(f"{cell} simulation {sim.id} has no official messages")
            keyed[key] = sim

        tasks = {key.task_id for key in keyed}
        if len(tasks) != expected_task_count:
            raise ValueError(
                f"{cell} has {len(tasks)} unique tasks, expected {expected_task_count}"
            )
        if reference_tasks is None:
            reference_tasks = tasks
        elif tasks != reference_tasks:
            raise ValueError(f"{cell} task set differs from the reference cell")

        expected_keys = {
            EpisodeKey(task_id, trial, seeds[trial])
            for task_id in reference_tasks
            for trial in trials
        }
        if set(keyed) != expected_keys:
            missing = len(expected_keys - set(keyed))
            extra = len(set(keyed) - expected_keys)
            raise ValueError(
                f"{cell} episode grid is incomplete: {missing} missing, {extra} extra"
            )
        simulations_by_cell[cell] = keyed

    reference_keys = set(simulations_by_cell[FACTORIAL_CELLS[0]])
    for cell in FACTORIAL_CELLS[1:]:
        if set(simulations_by_cell[cell]) != reference_keys:
            raise ValueError(f"{cell} episode keys differ from the reference cell")

    assert reference_tasks is not None
    return ValidatedFactorialGrid(
        simulations_by_cell=simulations_by_cell,
        task_ids=tuple(sorted(reference_tasks)),
        trials=trials,
        seeds_by_trial=seeds,
    )


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
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    tasks = sorted(set(a) & set(b))
    assert tasks, "no shared tasks to compare"
    if any(not a[t] or not b[t] for t in tasks):
        raise ValueError("shared tasks must have at least one reward in each arm")
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
    mean_a = sum(sum(a[t]) / len(a[t]) for t in tasks) / n
    mean_b = sum(sum(b[t]) / len(b[t]) for t in tasks) / n
    return PairedComparison(
        mean_a=mean_a,
        mean_b=mean_b,
        diff=sum(diffs) / n,
        ci_low=stats[int(0.025 * resamples)],
        ci_high=stats[min(int(0.975 * resamples), resamples - 1)],
        tasks=n,
    )


def _cell_mean(cell: Mapping[str, list[float]]) -> float:
    if not cell or any(not rewards for rewards in cell.values()):
        raise ValueError("factorial cells and per-task reward lists must be non-empty")
    return sum(sum(values) / len(values) for values in cell.values()) / len(cell)


def _contrasts_from_cell_means(means: Mapping[str, float]) -> dict[str, float]:
    return {
        name: sum(means[cell] * coefficient for cell, coefficient in coefficients.items())
        for name, coefficients in CONTRAST_COEFFICIENTS.items()
    }


def factorial_contrasts(
    cells: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, float]:
    """All six predeclared contrasts for the base/RL x native/governed design."""
    missing = set(FACTORIAL_CELLS) - set(cells)
    if missing:
        raise ValueError(f"missing factorial cells: {sorted(missing)}")
    means = {cell: _cell_mean(cells[cell]) for cell in FACTORIAL_CELLS}
    return _contrasts_from_cell_means(means)


def factorial_effects(cells: dict[str, dict[str, list[float]]]) -> dict[str, float]:
    """Backward-compatible four contrasts from the original design docs."""
    required = ("h0", "hbest", "rl", "hbest_rl")
    missing = set(required) - set(cells)
    if missing:
        raise ValueError(f"missing legacy factorial cells: {sorted(missing)}")
    h0, hbest, rl, hbest_rl = (_cell_mean(cells[cell]) for cell in required)
    return {
        "harness_effect": hbest - h0,
        "model_effect": rl - h0,
        "combined_gain": hbest_rl - h0,
        "interaction": (hbest_rl - rl) - (hbest - h0),
    }


@dataclass(frozen=True)
class BootstrapContrast:
    estimate: float
    ci_low: float
    ci_high: float
    tasks: int
    resamples: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def factorial_bootstrap_contrasts(
    cells: Mapping[str, Results] | ValidatedFactorialGrid,
    *,
    expected_task_count: int = 40,
    expected_trial_count: int = 8,
    base_seed: int = 42,
    expected_task_ids: Iterable[str] | None = None,
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, BootstrapContrast]:
    """Paired task bootstrap for all six contrasts using one draw per replicate."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    grid = (
        cells
        if isinstance(cells, ValidatedFactorialGrid)
        else validate_factorial_grid(
            cells,
            expected_task_count=expected_task_count,
            expected_trial_count=expected_trial_count,
            base_seed=base_seed,
            expected_task_ids=expected_task_ids,
        )
    )
    task_means: dict[str, dict[str, float]] = {}
    for cell in FACTORIAL_CELLS:
        keyed = grid.simulations_by_cell[cell]
        task_means[cell] = {
            task_id: sum(
                _reward(
                    keyed[
                        EpisodeKey(
                            task_id,
                            trial,
                            grid.seeds_by_trial[trial],
                        )
                    ],
                    strict=True,
                )
                for trial in grid.trials
            )
            / len(grid.trials)
            for task_id in grid.task_ids
        }

    observed_means = {
        cell: sum(task_means[cell].values()) / len(grid.task_ids)
        for cell in FACTORIAL_CELLS
    }
    observed = _contrasts_from_cell_means(observed_means)
    bootstrap_values = {name: [] for name in CONTRAST_COEFFICIENTS}
    rng = random.Random(seed)
    n = len(grid.task_ids)
    for _ in range(resamples):
        sample = [grid.task_ids[rng.randrange(n)] for _ in range(n)]
        sampled_cell_means = {
            cell: sum(task_means[cell][task_id] for task_id in sample) / n
            for cell in FACTORIAL_CELLS
        }
        sampled = _contrasts_from_cell_means(sampled_cell_means)
        for name, value in sampled.items():
            bootstrap_values[name].append(value)

    out: dict[str, BootstrapContrast] = {}
    for name, values in bootstrap_values.items():
        values.sort()
        out[name] = BootstrapContrast(
            estimate=observed[name],
            ci_low=values[int(0.025 * resamples)],
            ci_high=values[min(int(0.975 * resamples), resamples - 1)],
            tasks=n,
            resamples=resamples,
        )
    return out


# --- failure taxonomy ---------------------------------------------------------

TAXONOMY_PRIORITY = [
    "infrastructure_error",
    "timeout",
    "unexpected_error",
    "protocol_error",
    "context_window_exceeded",
    "max_steps_loop",
    "too_many_tool_errors",
    "policy_violation",
    "duplicate_side_effect",
    "missing_agent_action",
    "missing_user_action",
    "communication_missing",
    "nl_assertion_failure",
    "env_assertion_failed",
    "db_mismatch",
    "other_reward_failure",
]

TERMINATION_CATEGORY: dict[TerminationReason, str | None] = {
    TerminationReason.USER_STOP: None,
    TerminationReason.AGENT_STOP: None,
    TerminationReason.MAX_STEPS: "max_steps_loop",
    TerminationReason.TIMEOUT: "timeout",
    TerminationReason.TOO_MANY_ERRORS: "too_many_tool_errors",
    TerminationReason.AGENT_ERROR: "protocol_error",
    TerminationReason.USER_ERROR: "protocol_error",
    TerminationReason.INFRASTRUCTURE_ERROR: "infrastructure_error",
    TerminationReason.CONTEXT_WINDOW_EXCEEDED: "context_window_exceeded",
    TerminationReason.UNEXPECTED_ERROR: "unexpected_error",
}
assert set(TERMINATION_CATEGORY) == set(TerminationReason)


def _reward_basis(sim: SimulationRun, task) -> set[RewardType]:
    info = sim.reward_info
    raw_basis = info.reward_basis if info is not None else None
    if raw_basis is None and task is not None and task.evaluation_criteria is not None:
        raw_basis = task.evaluation_criteria.reward_basis
    basis: set[RewardType] = set()
    for value in raw_basis or []:
        try:
            basis.add(value if isinstance(value, RewardType) else RewardType(value))
        except ValueError:
            continue
    return basis


def _breakdown_failed(sim: SimulationRun, component: RewardType) -> bool:
    info = sim.reward_info
    if info is None or info.reward_breakdown is None:
        return False
    value = info.reward_breakdown.get(component)
    if value is None:
        value = info.reward_breakdown.get(component.value)  # type: ignore[arg-type]
    return value is not None and float(value) < 1 - 1e-6


def classify_failure(sim: SimulationRun, task=None) -> tuple[str, str]:
    """(category, evidence) for a failed simulation."""
    category = TERMINATION_CATEGORY[sim.termination_reason]
    if category == "max_steps_loop":
        return "max_steps_loop", f"{len(sim.messages or [])} messages"
    if category is not None:
        return category, sim.termination_reason.value

    analysis = analyze_trajectory(sim.messages or [])
    if analysis.unauthorized_executed_writes:
        return (
            "policy_violation",
            f"unauthorized writes: {dict(analysis.unauthorized_reasons)}",
        )
    if analysis.duplicate_side_effects:
        return "duplicate_side_effect", f"{analysis.duplicate_side_effects} duplicates"

    info = sim.reward_info
    if info is None:
        return "other_reward_failure", "reward_info missing"
    basis = _reward_basis(sim, task)

    if RewardType.ACTION in basis:
        failed_actions = [
            check.action for check in info.action_checks or [] if not check.action_match
        ]
        missing_agent = sorted(
            action.name for action in failed_actions if action.requestor == "assistant"
        )
        missing_user = sorted(
            action.name for action in failed_actions if action.requestor == "user"
        )
        if missing_agent:
            return "missing_agent_action", f"unmatched required actions: {missing_agent}"
        if missing_user:
            return "missing_user_action", f"unmatched required actions: {missing_user}"

    if RewardType.COMMUNICATE in basis:
        failed = [check for check in info.communicate_checks or [] if not check.met]
        if failed or _breakdown_failed(sim, RewardType.COMMUNICATE):
            return "communication_missing", f"{max(len(failed), 1)} required checks unmet"

    if RewardType.NL_ASSERTION in basis:
        failed = [check for check in info.nl_assertions or [] if not check.met]
        if failed or _breakdown_failed(sim, RewardType.NL_ASSERTION):
            return "nl_assertion_failure", f"{max(len(failed), 1)} assertions unmet"

    env_failed = (
        RewardType.ENV_ASSERTION in basis
        and (
            any(not check.met for check in info.env_assertions or [])
            or _breakdown_failed(sim, RewardType.ENV_ASSERTION)
        )
    )
    db_failed = (
        RewardType.DB in basis
        and (
            (info.db_check is not None and not info.db_check.db_match)
            or _breakdown_failed(sim, RewardType.DB)
        )
    )
    if env_failed:
        return "env_assertion_failed", "ENV_ASSERTION reward component failed"
    if db_failed:
        return "db_mismatch", "DB reward component failed"
    return "other_reward_failure", "no failed gating component was recorded"


def taxonomy(results: Results, tasks_by_id: Mapping | None = None) -> dict[str, int]:
    counts: Counter = Counter()
    failures = 0
    for sim in results.simulations:
        if _is_successful(_reward(sim)):
            continue
        failures += 1
        task = tasks_by_id.get(sim.task_id) if tasks_by_id is not None else None
        category, _ = classify_failure(sim, task)
        counts[category] += 1
    assert sum(counts.values()) == failures
    return dict(counts)


def failure_examples(
    results: Results, tasks_by_id: Mapping | None = None, per_category: int = 2
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for sim in results.simulations:
        if _is_successful(_reward(sim)):
            continue
        task = tasks_by_id.get(sim.task_id) if tasks_by_id is not None else None
        category, evidence = classify_failure(sim, task)
        if len(out[category]) < per_category:
            out[category].append(
                {"task_id": sim.task_id, "evidence": evidence, "sim_id": sim.id}
            )
    return dict(out)


# --- pass^k across arms -------------------------------------------------------


def pass_hat_ks(
    results: Results,
    ks: Iterable[int] = (1, 2, 4),
    *,
    strict: bool = False,
    expected_trials: int | None = None,
    expected_task_ids: Iterable[str] | None = None,
) -> dict[int, float]:
    """Macro task-level pass^k, with an optional fail-closed final mode."""
    from tau2.metrics.agent_metrics import pass_hat_k

    requested_ks = tuple(ks)
    if not requested_ks or any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in requested_ks
    ):
        raise ValueError("ks must contain positive integers")
    if strict and expected_trials is None:
        expected_trials = results.info.num_trials
    if expected_trials is not None and expected_trials <= 0:
        raise ValueError("expected_trials must be positive")

    per_task: dict[str, list[bool]] = defaultdict(list)
    trials_by_task: dict[str, set[int]] = defaultdict(set)
    for sim in results.simulations:
        if strict and sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
            raise ValueError("strict pass^k refuses infrastructure_error simulations")
        reward = _reward(sim, strict=strict)
        per_task[sim.task_id].append(_is_successful(reward))
        if strict:
            if sim.trial is None or isinstance(sim.trial, bool):
                raise ValueError(f"simulation {sim.id} has no integer trial")
            if sim.trial in trials_by_task[sim.task_id]:
                raise ValueError(
                    f"task {sim.task_id} has duplicate trial {sim.trial}"
                )
            trials_by_task[sim.task_id].add(sim.trial)

    if strict:
        assert expected_trials is not None
        expected_trial_ids = set(range(expected_trials))
        expected_ids = (
            set(expected_task_ids) if expected_task_ids is not None else set(per_task)
        )
        if set(per_task) != expected_ids:
            raise ValueError("strict pass^k task set differs from expected_task_ids")
        for task_id in expected_ids:
            if trials_by_task[task_id] != expected_trial_ids:
                raise ValueError(
                    f"task {task_id} trials={sorted(trials_by_task[task_id])}, "
                    f"expected {sorted(expected_trial_ids)}"
                )
        if any(k > expected_trials for k in requested_ks):
            raise ValueError("requested k exceeds expected_trials")

    out = {}
    for k in requested_ks:
        vals = [
            pass_hat_k(len(trials), sum(trials), k)
            for trials in per_task.values()
            if len(trials) >= k
        ]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out
