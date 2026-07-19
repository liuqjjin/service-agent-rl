"""Guard: our tau2-bench submodule may differ from the upstream pin only in
the gym wrapper. Every result-bearing path (runner, orchestrator, environment,
evaluator, domains, data) must be byte-identical to cf71a80, so no reported
number can be an artifact of a local modification."""

import subprocess
from pathlib import Path

UPSTREAM_PIN = "cf71a8070269883e38a365ffa85f78f46844c1f4"
ALLOWED_CHANGES = {"src/tau2/gym/gym_agent.py"}
TAU2 = Path(__file__).resolve().parents[1] / "third_party/tau2-bench"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(TAU2), *args], check=True, capture_output=True, text=True
    ).stdout


def test_only_gym_wrapper_differs_from_pin():
    changed = set(_git("diff", "--name-only", UPSTREAM_PIN).splitlines())
    unexpected = changed - ALLOWED_CHANGES
    assert not unexpected, (
        f"tau2-bench differs from {UPSTREAM_PIN[:7]} outside the allowed gym fix: "
        f"{sorted(unexpected)}"
    )


def test_worktree_matches_checked_out_commit():
    # No uncommitted edits hiding in the submodule: what we run is what the
    # pinned fix-branch commit says.
    status = _git("status", "--porcelain").strip()
    assert status == "", f"tau2-bench submodule has uncommitted changes:\n{status}"


def test_fix_branch_is_based_on_pin():
    merge_base = _git("merge-base", "HEAD", UPSTREAM_PIN).strip()
    assert merge_base == UPSTREAM_PIN, (
        "the checked-out tau2-bench commit must descend from the upstream pin"
    )
