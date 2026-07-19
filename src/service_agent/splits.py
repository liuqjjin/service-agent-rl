"""Task-split protocol for the telecom domain.

tau2's own splits are not all safe to train on: `full` and `base` both contain
the official `test` tasks. This module is the single source of truth for which
task IDs may be used for what:

    train (74, upstream)  ->  train_core (54, RL training)  +  dev (20, all selection)
    test  (40, upstream)  ->  touched exactly once, for the final 2x2 table
    expanded_train        ->  full - test - dev, for optional training-data scaling

The dev set is not a random sample. Tasks are stratified by (issue family,
needs-agent-write, needs-user-action) with proportional allocation, and picked
within each stratum by systematic sampling over reference-action count, so the
short/long spectrum is covered deterministically -- no RNG seed to defend.

The selected dev IDs are frozen in data_protocol/telecom_dev_ids.json. Tests
assert the frozen file matches what this module regenerates, so any silent
drift (upstream data change, refactor) fails CI instead of corrupting the
protocol.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TELECOM_DATA_DIR = REPO_ROOT / "third_party/tau2-bench/data/tau2/domains/telecom"
DATA_PROTOCOL_DIR = REPO_ROOT / "data_protocol"
DEV_IDS_PATH = DATA_PROTOCOL_DIR / "telecom_dev_ids.json"

DEV_SIZE = 20

# Write tools per @is_tool(ToolType.WRITE) in
# third_party/tau2-bench/src/tau2/domains/telecom/tools.py
WRITE_TOOLS = frozenset(
    {
        "send_payment_request",
        "resume_line",
        "suspend_line",
        "refuel_data",
        "enable_roaming",
        "disable_roaming",
    }
)


def load_split_ids() -> dict[str, set[str]]:
    raw = json.loads((TELECOM_DATA_DIR / "split_tasks.json").read_text())
    return {name: set(map(str, ids)) for name, ids in raw.items()}


def load_raw_tasks() -> dict[str, dict[str, Any]]:
    tasks = json.loads((TELECOM_DATA_DIR / "tasks.json").read_text())
    return {t["id"]: t for t in tasks}


def assert_split_hygiene(splits: dict[str, set[str]] | None = None) -> None:
    """The invariants everything else relies on. Raises AssertionError."""
    s = splits or load_split_ids()
    assert s["train"].isdisjoint(s["test"]), "train and test overlap"
    assert s["test"] <= s["full"], "test is not a subset of full"
    assert s["test"] <= s["base"], "test is not a subset of base"
    assert s["train"] <= s["full"], "train is not a subset of full"


def expanded_train_ids(dev_ids: set[str] | None = None) -> set[str]:
    """The largest set safe to train on: full minus test minus dev."""
    s = load_split_ids()
    dev = dev_ids if dev_ids is not None else set(load_frozen_dev_ids())
    return s["full"] - s["test"] - dev


# --- stratification -----------------------------------------------------------


def task_features(task: dict[str, Any]) -> dict[str, Any]:
    ec = task.get("evaluation_criteria") or {}
    actions = ec.get("actions") or []
    return {
        "family": task["id"].split("]")[0].strip("["),
        "agent_write": any(
            a["requestor"] == "assistant" and a["name"] in WRITE_TOOLS for a in actions
        ),
        "user_action": any(a["requestor"] == "user" for a in actions),
        "num_actions": len(actions),
        "reward_basis": tuple(ec.get("reward_basis") or []),
    }


def _stratum(feats: dict[str, Any]) -> tuple[str, bool, bool]:
    return (feats["family"], feats["agent_write"], feats["user_action"])


def _proportional_allocation(counts: Counter, total: int) -> dict[Any, int]:
    """Largest-remainder allocation of `total` slots proportional to counts."""
    n = sum(counts.values())
    exact = {k: v * total / n for k, v in counts.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    remainders = sorted(exact.items(), key=lambda kv: (kv[1] - int(kv[1]), kv[0]), reverse=True)
    for k, _ in remainders[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def select_dev_ids() -> list[str]:
    """Deterministic stratified dev selection from the official train split."""
    splits = load_split_ids()
    tasks = load_raw_tasks()
    train = sorted(splits["train"])
    feats = {tid: task_features(tasks[tid]) for tid in train}

    strata: dict[tuple, list[str]] = {}
    for tid in train:
        strata.setdefault(_stratum(feats[tid]), []).append(tid)

    alloc = _proportional_allocation(
        Counter({k: len(v) for k, v in strata.items()}), DEV_SIZE
    )

    dev: list[str] = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda t: (feats[t]["num_actions"], t))
        k = alloc[stratum]
        # Systematic sampling over the action-count ordering: evenly spaced
        # picks cover short and long tasks without a random seed.
        picks = [members[(i * len(members)) // k + ((len(members) // k) // 2)] for i in range(k)]
        dev.extend(picks)
    return sorted(dev)


def load_frozen_dev_ids() -> list[str]:
    return json.loads(DEV_IDS_PATH.read_text())["dev"]


def train_core_ids() -> list[str]:
    return sorted(load_split_ids()["train"] - set(load_frozen_dev_ids()))


# --- fingerprints (near-duplicate disclosure) ---------------------------------


def task_fingerprint(task: dict[str, Any]) -> str:
    """Semantic fingerprint: two tasks sharing one are near-duplicates.

    ID disjointness does not imply semantic disjointness -- telecom tasks are
    generated from a small set of atomic faults. We report fingerprint overlap
    across splits rather than pretending it away.
    """
    ec = task.get("evaluation_criteria") or {}
    init = task.get("initial_state") or {}
    payload = {
        "family": task["id"].split("]")[0].strip("["),
        "actions": sorted(
            f"{a['requestor']}:{a['name']}" for a in (ec.get("actions") or [])
        ),
        "init_actions": sorted(
            f"{a['env_type']}:{a['func_name']}"
            for a in (init.get("initialization_actions") or [])
        ),
        "reward_basis": sorted(ec.get("reward_basis") or []),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fingerprint_overlap_report() -> dict[str, Any]:
    splits = load_split_ids()
    tasks = load_raw_tasks()
    dev = set(load_frozen_dev_ids())
    groups = {
        "train_core": splits["train"] - dev,
        "dev": dev,
        "test": splits["test"],
    }
    fps = {name: {task_fingerprint(tasks[t]) for t in ids} for name, ids in groups.items()}
    return {
        "sizes": {name: len(ids) for name, ids in groups.items()},
        "unique_fingerprints": {name: len(v) for name, v in fps.items()},
        "overlap": {
            "train_core∩dev": len(fps["train_core"] & fps["dev"]),
            "train_core∩test": len(fps["train_core"] & fps["test"]),
            "dev∩test": len(fps["dev"] & fps["test"]),
        },
    }


def freeze() -> None:
    """Write the frozen dev IDs and the split report. Run once, commit the output."""
    assert_split_hygiene()
    dev = select_dev_ids()
    DATA_PROTOCOL_DIR.mkdir(exist_ok=True)
    DEV_IDS_PATH.write_text(json.dumps({"dev": dev}, indent=2) + "\n")

    splits = load_split_ids()
    report = {
        "upstream_commit": "cf71a8070269883e38a365ffa85f78f46844c1f4",
        "split_sizes": {k: len(v) for k, v in sorted(splits.items())},
        "train∩test": len(splits["train"] & splits["test"]),
        "test⊂full": splits["test"] <= splits["full"],
        "test⊂base": splits["test"] <= splits["base"],
        "dev_size": len(dev),
        "train_core_size": len(splits["train"]) - len(dev),
        "expanded_train_size": len(splits["full"] - splits["test"] - set(dev)),
        "fingerprints": fingerprint_overlap_report(),
    }
    (DATA_PROTOCOL_DIR / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    freeze()
    print(json.dumps(json.loads((DATA_PROTOCOL_DIR / "split_report.json").read_text()), indent=2))
