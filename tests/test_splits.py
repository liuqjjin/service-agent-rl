"""Split protocol invariants. If any of these fail, stop: results built on top
of a broken split are unpublishable."""

from collections import Counter

from service_agent.splits import (
    DEV_SIZE,
    assert_split_hygiene,
    expanded_train_ids,
    load_frozen_dev_ids,
    load_split_ids,
    select_dev_ids,
    train_core_ids,
)


def test_upstream_split_hygiene():
    splits = load_split_ids()
    assert_split_hygiene(splits)
    assert len(splits["train"] & splits["test"]) == 0
    # full and base both contain the official test set: training on either
    # would be contamination. This is the fact the whole protocol exists for.
    assert splits["test"] <= splits["full"]
    assert splits["test"] <= splits["base"]


def test_frozen_dev_matches_regeneration():
    # The committed dev IDs must be exactly what the selection algorithm
    # produces from the pinned upstream data. Catches silent drift from
    # upstream data changes or refactors of the selection code.
    assert load_frozen_dev_ids() == select_dev_ids()


def test_dev_invariants():
    splits = load_split_ids()
    dev = set(load_frozen_dev_ids())
    core = set(train_core_ids())
    assert len(dev) == DEV_SIZE
    assert dev <= splits["train"]
    assert core == splits["train"] - dev
    assert core.isdisjoint(dev)
    assert len(core) == len(splits["train"]) - DEV_SIZE


def test_dev_stratification_covers_families():
    families = Counter(tid.split("]")[0].strip("[") for tid in load_frozen_dev_ids())
    assert families == {"mms_issue": 9, "mobile_data_issue": 7, "service_issue": 4}


def test_expanded_train_is_safe():
    splits = load_split_ids()
    expanded = expanded_train_ids()
    assert expanded.isdisjoint(splits["test"])
    assert expanded.isdisjoint(set(load_frozen_dev_ids()))
    assert expanded <= splits["full"]
