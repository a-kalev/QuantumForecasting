#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter, defaultdict

import pytest

from generate_ed_data import (
    build_seed_manifest,
    validate_seed_manifest,
)


def manifest_signature(manifest):
    return sorted(
        (
            entry["split"],
            entry["L"],
            entry["W"],
            entry["realization"],
            entry["seed"],
        )
        for entry in manifest
    )


def test_seed_manifest_has_expected_number_of_entries():
    sizes = [12, 14]
    disorders = [2.0, 3.0, 4.0, 5.0]
    split_counts = {
        "train": 20,
        "validation": 10,
        "test": 15,
    }

    manifest = build_seed_manifest(
        sizes=sizes,
        disorders=disorders,
        split_counts=split_counts,
        base_seed=100_000,
    )

    expected = (
        len(sizes)
        * len(disorders)
        * sum(split_counts.values())
    )

    assert len(manifest) == expected


def test_each_parameter_group_has_exact_split_counts():
    sizes = [12, 14]
    disorders = [2.0, 3.0, 4.0, 5.0]
    split_counts = {
        "train": 7,
        "validation": 5,
        "test": 9,
    }

    manifest = build_seed_manifest(
        sizes=sizes,
        disorders=disorders,
        split_counts=split_counts,
        base_seed=200_000,
    )

    counts = Counter(
        (entry["L"], entry["W"], entry["split"])
        for entry in manifest
    )

    for L in sizes:
        for W in disorders:
            for split, expected_count in split_counts.items():
                assert counts[(L, W, split)] == expected_count


def test_all_seeds_are_globally_unique():
    manifest = build_seed_manifest(
        sizes=[12, 14],
        disorders=[2.0, 3.0, 4.0, 5.0],
        split_counts={
            "train": 10,
            "validation": 5,
            "test": 10,
        },
        base_seed=300_000,
    )

    seeds = [entry["seed"] for entry in manifest]

    assert len(seeds) == len(set(seeds))


def test_train_validation_and_test_seed_sets_are_disjoint():
    manifest = build_seed_manifest(
        sizes=[12, 14],
        disorders=[2.0, 3.0, 4.0, 5.0],
        split_counts={
            "train": 12,
            "validation": 8,
            "test": 10,
        },
        base_seed=400_000,
    )

    seeds_by_split = defaultdict(set)

    for entry in manifest:
        seeds_by_split[entry["split"]].add(entry["seed"])

    assert seeds_by_split["train"].isdisjoint(
        seeds_by_split["validation"]
    )
    assert seeds_by_split["train"].isdisjoint(
        seeds_by_split["test"]
    )
    assert seeds_by_split["validation"].isdisjoint(
        seeds_by_split["test"]
    )


def test_seeds_are_not_reused_across_disorder_strengths():
    manifest = build_seed_manifest(
        sizes=[12],
        disorders=[2.0, 3.0, 4.0, 5.0],
        split_counts={
            "train": 6,
            "validation": 4,
            "test": 5,
        },
        base_seed=500_000,
    )

    seeds_by_disorder = defaultdict(set)

    for entry in manifest:
        seeds_by_disorder[entry["W"]].add(entry["seed"])

    disorders = sorted(seeds_by_disorder)

    for index, W_left in enumerate(disorders):
        for W_right in disorders[index + 1 :]:
            assert seeds_by_disorder[W_left].isdisjoint(
                seeds_by_disorder[W_right]
            )


def test_seeds_are_not_reused_across_system_sizes():
    manifest = build_seed_manifest(
        sizes=[10, 12, 14],
        disorders=[3.0],
        split_counts={
            "train": 6,
            "validation": 4,
            "test": 5,
        },
        base_seed=600_000,
    )

    seeds_by_size = defaultdict(set)

    for entry in manifest:
        seeds_by_size[entry["L"]].add(entry["seed"])

    sizes = sorted(seeds_by_size)

    for index, L_left in enumerate(sizes):
        for L_right in sizes[index + 1 :]:
            assert seeds_by_size[L_left].isdisjoint(
                seeds_by_size[L_right]
            )


def test_realization_indices_restart_within_each_group():
    split_counts = {
        "train": 4,
        "validation": 3,
        "test": 2,
    }

    manifest = build_seed_manifest(
        sizes=[12, 14],
        disorders=[2.0, 4.0],
        split_counts=split_counts,
        base_seed=700_000,
    )

    realizations_by_group = defaultdict(list)

    for entry in manifest:
        key = (
            entry["L"],
            entry["W"],
            entry["split"],
        )
        realizations_by_group[key].append(
            entry["realization"]
        )

    for (_, _, split), realizations in realizations_by_group.items():
        assert sorted(realizations) == list(
            range(split_counts[split])
        )


def test_manifest_contains_required_metadata():
    manifest = build_seed_manifest(
        sizes=[12],
        disorders=[3.0],
        split_counts={
            "train": 2,
            "validation": 1,
            "test": 1,
        },
        base_seed=800_000,
    )

    required_keys = {
        "split",
        "L",
        "W",
        "realization",
        "seed",
    }

    for entry in manifest:
        assert set(entry) == required_keys
        assert isinstance(entry["split"], str)
        assert isinstance(entry["L"], int)
        assert isinstance(entry["W"], float)
        assert isinstance(entry["realization"], int)
        assert isinstance(entry["seed"], int)


def test_manifest_is_deterministic():
    arguments = {
        "sizes": [12, 14],
        "disorders": [2.0, 3.0, 4.0, 5.0],
        "split_counts": {
            "train": 5,
            "validation": 3,
            "test": 4,
        },
        "base_seed": 900_000,
    }

    manifest_a = build_seed_manifest(**arguments)
    manifest_b = build_seed_manifest(**arguments)

    assert manifest_signature(manifest_a) == manifest_signature(
        manifest_b
    )


def test_manifest_is_independent_of_input_order():
    manifest_a = build_seed_manifest(
        sizes=[12, 14],
        disorders=[2.0, 3.0, 4.0],
        split_counts={
            "train": 5,
            "validation": 3,
            "test": 4,
        },
        base_seed=1_000_000,
    )

    manifest_b = build_seed_manifest(
        sizes=[14, 12],
        disorders=[4.0, 2.0, 3.0],
        split_counts={
            "test": 4,
            "train": 5,
            "validation": 3,
        },
        base_seed=1_000_000,
    )

    assert manifest_signature(manifest_a) == manifest_signature(
        manifest_b
    )


def test_changing_base_seed_changes_every_seed():
    manifest_a = build_seed_manifest(
        sizes=[12],
        disorders=[2.0, 4.0],
        split_counts={
            "train": 3,
            "validation": 2,
            "test": 2,
        },
        base_seed=1_100_000,
    )

    manifest_b = build_seed_manifest(
        sizes=[12],
        disorders=[2.0, 4.0],
        split_counts={
            "train": 3,
            "validation": 2,
            "test": 2,
        },
        base_seed=1_200_000,
    )

    mapping_a = {
        (
            entry["split"],
            entry["L"],
            entry["W"],
            entry["realization"],
        ): entry["seed"]
        for entry in manifest_a
    }

    mapping_b = {
        (
            entry["split"],
            entry["L"],
            entry["W"],
            entry["realization"],
        ): entry["seed"]
        for entry in manifest_b
    }

    assert mapping_a.keys() == mapping_b.keys()

    for key in mapping_a:
        assert mapping_a[key] != mapping_b[key]


def test_validate_seed_manifest_accepts_valid_manifest():
    manifest = build_seed_manifest(
        sizes=[12, 14],
        disorders=[2.0, 3.0],
        split_counts={
            "train": 4,
            "validation": 2,
            "test": 3,
        },
        base_seed=1_300_000,
    )

    validate_seed_manifest(manifest)


def test_validate_seed_manifest_rejects_duplicate_seed():
    manifest = build_seed_manifest(
        sizes=[12],
        disorders=[2.0],
        split_counts={
            "train": 2,
            "validation": 1,
            "test": 1,
        },
        base_seed=1_400_000,
    )

    manifest[1]["seed"] = manifest[0]["seed"]

    with pytest.raises(ValueError, match="duplicate seed"):
        validate_seed_manifest(manifest)


def test_validate_seed_manifest_rejects_duplicate_group_entry():
    manifest = build_seed_manifest(
        sizes=[12],
        disorders=[2.0],
        split_counts={
            "train": 2,
            "validation": 1,
            "test": 1,
        },
        base_seed=1_500_000,
    )

    duplicate = dict(manifest[0])
    duplicate["seed"] = max(
        entry["seed"] for entry in manifest
    ) + 1
    manifest.append(duplicate)

    with pytest.raises(
        ValueError,
        match="duplicate realization entry",
    ):
        validate_seed_manifest(manifest)


@pytest.mark.parametrize(
    "sizes, disorders, split_counts, base_seed",
    [
        ([], [2.0], {"train": 1}, 0),
        ([12], [], {"train": 1}, 0),
        ([12], [2.0], {}, 0),
        ([11], [2.0], {"train": 1}, 0),
        ([12], [-1.0], {"train": 1}, 0),
        ([12], [2.0], {"train": 0}, 0),
        ([12], [2.0], {"train": -1}, 0),
        ([12], [2.0], {"train": 1}, -1),
    ],
)
def test_invalid_manifest_requests_are_rejected(
    sizes,
    disorders,
    split_counts,
    base_seed,
):
    with pytest.raises(ValueError):
        build_seed_manifest(
            sizes=sizes,
            disorders=disorders,
            split_counts=split_counts,
            base_seed=base_seed,
        )