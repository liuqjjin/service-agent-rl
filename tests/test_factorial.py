"""Statistics and taxonomy mechanics on synthetic data."""

from service_agent.eval.factorial import (
    PairedComparison,
    factorial_effects,
    paired_bootstrap,
)


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
