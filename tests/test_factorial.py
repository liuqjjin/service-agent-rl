"""Statistics and taxonomy mechanics on synthetic data."""

import math

import pytest
from tau2.data_model.simulation import (
    ActionCheck,
    AgentInfo,
    CommunicateCheck,
    DBCheck,
    Info,
    NLAssertionCheck,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import Action, RewardType
from tau2.environment.environment import EnvironmentInfo

from service_agent.eval.factorial import (
    FACTORIAL_CELLS,
    TERMINATION_CATEGORY,
    PairedComparison,
    classify_failure,
    factorial_bootstrap_contrasts,
    factorial_contrasts,
    factorial_effects,
    paired_bootstrap,
    pass_hat_ks,
    taxonomy,
    trial_seeds,
    validate_factorial_grid,
)


def make_info(num_trials: int, seed: int = 42) -> Info:
    return Info(
        git_commit="synthetic",
        num_trials=num_trials,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info=AgentInfo(implementation="llm_agent"),
        environment_info=EnvironmentInfo(
            domain_name="telecom",
            policy="synthetic policy",
        ),
        seed=seed,
    )


def make_simulation(
    *,
    sim_id: str,
    task_id: str,
    trial: int = 0,
    seed: int = 42,
    reward: float = 0.0,
    termination: TerminationReason = TerminationReason.AGENT_STOP,
    reward_info: RewardInfo | None = None,
) -> SimulationRun:
    if reward_info is None:
        reward_info = RewardInfo(
            reward=reward,
            reward_basis=[],
            reward_breakdown={},
        )
    return SimulationRun(
        id=sim_id,
        task_id=task_id,
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        duration=1.0,
        termination_reason=termination,
        reward_info=reward_info,
        messages=[],
        trial=trial,
        seed=seed,
    )


def make_results(
    cell: str,
    *,
    task_count: int,
    trial_count: int,
    reward: float,
    base_seed: int = 42,
) -> Results:
    seeds = trial_seeds(base_seed, trial_count)
    simulations = [
        make_simulation(
            sim_id=f"{cell}-{task_index}-{trial}",
            task_id=f"task-{task_index}",
            trial=trial,
            seed=seeds[trial],
            reward=reward,
        )
        for task_index in range(task_count)
        for trial in range(trial_count)
    ]
    return Results(
        info=make_info(trial_count, base_seed),
        tasks=[],
        simulations=simulations,
    )


def make_factorial_results(
    *, task_count: int = 2, trial_count: int = 3
) -> dict[str, Results]:
    rewards = {
        "base_h0": 0.1,
        "base_h2": 0.3,
        "rl_h0": 0.4,
        "rl_h2": 0.8,
    }
    return {
        cell: make_results(
            cell,
            task_count=task_count,
            trial_count=trial_count,
            reward=rewards[cell],
        )
        for cell in FACTORIAL_CELLS
    }


def test_paired_bootstrap_detects_clear_gap():
    a = {f"t{i}": [0.0, 0.0] for i in range(10)}
    b = {f"t{i}": [1.0, 1.0] for i in range(10)}
    cmp = paired_bootstrap(a, b, resamples=2000)
    assert isinstance(cmp, PairedComparison)
    assert cmp.diff == 1.0
    assert cmp.significant


def test_paired_bootstrap_flat_when_identical():
    a = {f"t{i}": [float(i % 2)] for i in range(10)}
    cmp = paired_bootstrap(a, a, resamples=2000)
    assert cmp.diff == 0.0
    assert not cmp.significant


def test_factorial_arithmetic():
    def cell(v):
        return {"t": [v]}

    effects = factorial_effects(
        {
            "h0": cell(0.4),
            "hbest": cell(0.6),
            "rl": cell(0.7),
            "hbest_rl": cell(0.8),
        }
    )
    assert round(effects["harness_effect"], 6) == 0.2
    assert round(effects["model_effect"], 6) == 0.3
    assert round(effects["combined_gain"], 6) == 0.4
    # (0.8 - 0.7) - (0.6 - 0.4) = -0.1: harness helps the RL model less.
    assert round(effects["interaction"], 6) == -0.1


def test_factorial_contrasts_include_both_rows_and_columns():
    def cell(value):
        return {"task": [value]}

    contrasts = factorial_contrasts(
        {
            "base_h0": cell(0.1),
            "base_h2": cell(0.3),
            "rl_h0": cell(0.4),
            "rl_h2": cell(0.8),
        }
    )

    assert contrasts == pytest.approx(
        {
            "harness_effect_base": 0.2,
            "harness_effect_rl": 0.4,
            "model_effect_native": 0.3,
            "model_effect_governed": 0.5,
            "combined_gain": 0.7,
            "interaction": 0.2,
        }
    )


def test_final_statistics_do_not_implicitly_alias_legacy_cell_names():
    legacy = {
        "h0": {"task": [0.1]},
        "hbest": {"task": [0.3]},
        "rl": {"task": [0.4]},
        "hbest_rl": {"task": [0.8]},
    }

    with pytest.raises(ValueError, match="missing factorial cells"):
        factorial_contrasts(legacy)


def test_final_grid_validates_exact_common_random_numbers():
    cells = make_factorial_results()
    grid = validate_factorial_grid(
        cells,
        expected_task_count=2,
        expected_trial_count=3,
    )

    assert grid.task_ids == ("task-0", "task-1")
    assert grid.trials == (0, 1, 2)
    assert grid.seeds_by_trial == {
        0: 670487,
        1: 116739,
        2: 26225,
    }
    assert len(grid.episode_keys) == 6


def test_final_grid_accepts_an_explicit_frozen_task_set():
    cells = make_factorial_results()
    validate_factorial_grid(
        cells,
        expected_task_count=2,
        expected_trial_count=3,
        expected_task_ids={"task-0", "task-1"},
    )
    with pytest.raises(ValueError, match="task set differs"):
        validate_factorial_grid(
            cells,
            expected_task_count=2,
            expected_trial_count=3,
            expected_task_ids={"task-0", "wrong-task"},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_seed", "has seed"),
        ("duplicate_key", "duplicate episode key"),
        ("different_task_set", "task set differs"),
        ("infrastructure", "infrastructure_error"),
        ("missing_episode", "simulations"),
    ],
)
def test_final_grid_fails_closed_on_incomplete_or_unpaired_cells(mutation, message):
    cells = make_factorial_results()
    bad = {name: results.model_copy(deep=True) for name, results in cells.items()}
    simulations = bad["rl_h2"].simulations

    if mutation == "wrong_seed":
        simulations[0].seed += 1
    elif mutation == "duplicate_key":
        simulations[1].task_id = simulations[0].task_id
        simulations[1].trial = simulations[0].trial
        simulations[1].seed = simulations[0].seed
    elif mutation == "different_task_set":
        for sim in simulations:
            if sim.task_id == "task-1":
                sim.task_id = "other-task"
    elif mutation == "infrastructure":
        simulations[0].termination_reason = TerminationReason.INFRASTRUCTURE_ERROR
    elif mutation == "missing_episode":
        simulations.pop()

    with pytest.raises(ValueError, match=message):
        validate_factorial_grid(
            bad,
            expected_task_count=2,
            expected_trial_count=3,
        )


def test_factorial_bootstrap_uses_one_paired_task_draw_for_all_contrasts():
    cells = make_factorial_results(task_count=4, trial_count=2)
    estimates = factorial_bootstrap_contrasts(
        cells,
        expected_task_count=4,
        expected_trial_count=2,
        resamples=500,
    )

    expected = {
        "harness_effect_base": 0.2,
        "harness_effect_rl": 0.4,
        "model_effect_native": 0.3,
        "model_effect_governed": 0.5,
        "combined_gain": 0.7,
        "interaction": 0.2,
    }
    assert set(estimates) == set(expected)
    for name, value in expected.items():
        assert estimates[name].estimate == pytest.approx(value)
        assert estimates[name].ci_low == pytest.approx(value)
        assert estimates[name].ci_high == pytest.approx(value)
        assert estimates[name].tasks == 4
        assert estimates[name].resamples == 500


def test_strict_pass_hat_k_matches_tau2_formula_and_success_tolerance():
    rewards = [1.0, 0.9999995, 1.0000005, 1.0, 0.0, 0.0, 0.0, 0.0]
    seeds = trial_seeds(42, len(rewards))
    results = Results(
        info=make_info(len(rewards)),
        tasks=[],
        simulations=[
            make_simulation(
                sim_id=f"pass-{trial}",
                task_id="task",
                trial=trial,
                seed=seeds[trial],
                reward=reward,
            )
            for trial, reward in enumerate(rewards)
        ],
    )

    metrics = pass_hat_ks(
        results,
        ks=(1, 2, 4, 8),
        strict=True,
        expected_trials=8,
        expected_task_ids={"task"},
    )

    assert metrics[1] == pytest.approx(4 / 8)
    assert metrics[2] == pytest.approx(math.comb(4, 2) / math.comb(8, 2))
    assert metrics[4] == pytest.approx(math.comb(4, 4) / math.comb(8, 4))
    assert metrics[8] == 0.0


@pytest.mark.parametrize("mutation", ["missing_trial", "duplicate_trial", "infrastructure"])
def test_strict_pass_hat_k_rejects_incomplete_trials(mutation):
    results = make_results("pass", task_count=1, trial_count=4, reward=1.0)
    if mutation == "missing_trial":
        results.simulations.pop()
    elif mutation == "duplicate_trial":
        results.simulations[-1].trial = results.simulations[-2].trial
    else:
        results.simulations[0].termination_reason = TerminationReason.INFRASTRUCTURE_ERROR

    with pytest.raises(ValueError):
        pass_hat_ks(results, ks=(1, 4), strict=True, expected_trials=4)


@pytest.mark.parametrize(
    ("termination", "category"),
    [
        (TerminationReason.INFRASTRUCTURE_ERROR, "infrastructure_error"),
        (TerminationReason.TIMEOUT, "timeout"),
        (TerminationReason.UNEXPECTED_ERROR, "unexpected_error"),
        (TerminationReason.AGENT_ERROR, "protocol_error"),
        (TerminationReason.USER_ERROR, "protocol_error"),
        (TerminationReason.CONTEXT_WINDOW_EXCEEDED, "context_window_exceeded"),
        (TerminationReason.MAX_STEPS, "max_steps_loop"),
        (TerminationReason.TOO_MANY_ERRORS, "too_many_tool_errors"),
        (TerminationReason.AGENT_STOP, "other_reward_failure"),
        (TerminationReason.USER_STOP, "other_reward_failure"),
    ],
)
def test_failure_taxonomy_maps_every_termination_reason(termination, category):
    sim = make_simulation(
        sim_id=termination.value,
        task_id="task",
        termination=termination,
    )

    assert classify_failure(sim)[0] == category


def test_termination_mapping_is_exhaustive():
    assert set(TERMINATION_CATEGORY) == set(TerminationReason)


def test_missing_action_requires_action_in_reward_basis():
    action = Action(
        action_id="required-refuel",
        requestor="assistant",
        name="refuel_data",
        arguments={"line_id": "L1", "gb_amount": 1},
    )
    failed_check = ActionCheck(
        action=action,
        action_match=False,
        action_reward=0.0,
    )
    action_failure = make_simulation(
        sim_id="action",
        task_id="task",
        reward_info=RewardInfo(
            reward=0.0,
            action_checks=[failed_check],
            reward_basis=[RewardType.ACTION],
            reward_breakdown={RewardType.ACTION: 0.0},
        ),
    )
    state_failure = make_simulation(
        sim_id="state",
        task_id="task",
        reward_info=RewardInfo(
            reward=0.0,
            action_checks=[failed_check],
            reward_basis=[RewardType.ENV_ASSERTION],
            reward_breakdown={RewardType.ENV_ASSERTION: 0.0},
        ),
    )

    assert classify_failure(action_failure)[0] == "missing_agent_action"
    assert classify_failure(state_failure)[0] == "env_assertion_failed"


def test_reward_basis_distinguishes_user_action_communication_nl_and_db():
    user_action = Action(
        action_id="required-toggle",
        requestor="user",
        name="toggle_data",
        arguments={"enabled": True},
    )
    simulations = [
        make_simulation(
            sim_id="user-action",
            task_id="task-user",
            reward_info=RewardInfo(
                reward=0.0,
                action_checks=[
                    ActionCheck(
                        action=user_action,
                        action_match=False,
                        action_reward=0.0,
                    )
                ],
                reward_basis=[RewardType.ACTION],
                reward_breakdown={RewardType.ACTION: 0.0},
            ),
        ),
        make_simulation(
            sim_id="communicate",
            task_id="task-communicate",
            reward_info=RewardInfo(
                reward=0.0,
                communicate_checks=[
                    CommunicateCheck(info="required", met=False, justification="missing")
                ],
                reward_basis=[RewardType.COMMUNICATE],
                reward_breakdown={RewardType.COMMUNICATE: 0.0},
            ),
        ),
        make_simulation(
            sim_id="nl",
            task_id="task-nl",
            reward_info=RewardInfo(
                reward=0.0,
                nl_assertions=[
                    NLAssertionCheck(
                        nl_assertion="required",
                        met=False,
                        justification="missing",
                    )
                ],
                reward_basis=[RewardType.NL_ASSERTION],
                reward_breakdown={RewardType.NL_ASSERTION: 0.0},
            ),
        ),
        make_simulation(
            sim_id="db",
            task_id="task-db",
            reward_info=RewardInfo(
                reward=0.0,
                db_check=DBCheck(db_match=False, db_reward=0.0),
                reward_basis=[RewardType.DB],
                reward_breakdown={RewardType.DB: 0.0},
            ),
        ),
    ]

    assert [classify_failure(sim)[0] for sim in simulations] == [
        "missing_user_action",
        "communication_missing",
        "nl_assertion_failure",
        "db_mismatch",
    ]


def test_environment_assertion_precedes_db_when_both_gating_components_fail():
    sim = make_simulation(
        sim_id="env-and-db",
        task_id="task",
        reward_info=RewardInfo(
            reward=0.0,
            db_check=DBCheck(db_match=False, db_reward=0.0),
            reward_basis=[RewardType.ENV_ASSERTION, RewardType.DB],
            reward_breakdown={
                RewardType.ENV_ASSERTION: 0.0,
                RewardType.DB: 0.0,
            },
        ),
    )

    assert classify_failure(sim)[0] == "env_assertion_failed"


def test_taxonomy_assigns_exactly_one_category_per_failed_episode():
    failures = [
        make_simulation(
            sim_id="timeout",
            task_id="task-timeout",
            termination=TerminationReason.TIMEOUT,
        ),
        make_simulation(
            sim_id="state",
            task_id="task-state",
            reward_info=RewardInfo(
                reward=0.0,
                reward_basis=[RewardType.DB],
                reward_breakdown={RewardType.DB: 0.0},
            ),
        ),
        make_simulation(
            sim_id="unknown",
            task_id="task-unknown",
        ),
    ]
    success = make_simulation(
        sim_id="success",
        task_id="task-success",
        reward=0.9999995,
    )
    results = Results(
        info=make_info(1),
        tasks=[],
        simulations=[*failures, success],
    )

    counts = taxonomy(results)

    assert counts == {
        "timeout": 1,
        "db_mismatch": 1,
        "other_reward_failure": 1,
    }
    assert sum(counts.values()) == len(failures)
