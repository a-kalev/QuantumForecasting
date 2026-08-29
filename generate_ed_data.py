#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate exact-diagonalization data for controlled STP validation.

Default dataset:
    L = 12, 14
    W = 2, 3, 4, 5
    train / validation / test = 30 / 10 / 10 realizations
    times = 0 ... 100 with dt = 0.2

Every realization has a globally unique disorder seed. Training,
validation, and test data are stored as separate episode files so an
interrupted calculation can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import numbers
import pickle
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.heisenberg_ed import generate_episode


DEFAULT_SIZES = (12, 14)
DEFAULT_DISORDERS = (2.0, 3.0, 4.0, 5.0)
DEFAULT_SPLIT_COUNTS = {
    "train": 30,
    "validation": 10,
    "test": 10,
}

DEFAULT_BASE_SEED = 2_000_000
DEFAULT_T_FINAL = 100.0
DEFAULT_DT = 0.2
DEFAULT_J = 1.0
DEFAULT_DELTA = 1.0
DEFAULT_OUTPUT = "ed_validation_v1"

_REQUIRED_MANIFEST_KEYS = {
    "split",
    "L",
    "W",
    "realization",
    "seed",
}


def _validate_split_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("split names must be nonempty strings.")

    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError(
            "split names may contain only letters, numbers, underscores, "
            "and hyphens."
        )


def build_seed_manifest(
    sizes: Sequence[int],
    disorders: Sequence[float],
    split_counts: Mapping[str, int],
    base_seed: int,
) -> list[dict]:
    """
    Build a deterministic manifest with globally unique seeds.

    The result is independent of the input ordering of sizes, disorders,
    and split_counts.
    """
    if not sizes:
        raise ValueError("sizes must not be empty.")

    if not disorders:
        raise ValueError("disorders must not be empty.")

    if not split_counts:
        raise ValueError("split_counts must not be empty.")

    if (
        not isinstance(base_seed, numbers.Integral)
        or isinstance(base_seed, bool)
        or base_seed < 0
    ):
        raise ValueError("base_seed must be a nonnegative integer.")

    normalized_sizes: list[int] = []
    for L in sizes:
        if (
            not isinstance(L, numbers.Integral)
            or isinstance(L, bool)
        ):
            raise ValueError("Every system size must be an integer.")

        L_int = int(L)

        if L_int < 2 or L_int % 2 != 0:
            raise ValueError(
                "Every system size must be an even integer >= 2."
            )

        normalized_sizes.append(L_int)

    normalized_disorders: list[float] = []
    for W in disorders:
        if (
            not isinstance(W, numbers.Real)
            or isinstance(W, bool)
            or not np.isfinite(W)
            or W < 0
        ):
            raise ValueError(
                "Every disorder strength must be finite and nonnegative."
            )

        normalized_disorders.append(float(W))

    normalized_counts: dict[str, int] = {}
    for split, count in split_counts.items():
        _validate_split_name(split)

        if (
            not isinstance(count, numbers.Integral)
            or isinstance(count, bool)
            or count <= 0
        ):
            raise ValueError(
                "Every split count must be a positive integer."
            )

        normalized_counts[str(split)] = int(count)

    if len(set(normalized_sizes)) != len(normalized_sizes):
        raise ValueError("sizes contains duplicate values.")

    if len(set(normalized_disorders)) != len(normalized_disorders):
        raise ValueError("disorders contains duplicate values.")

    manifest: list[dict] = []
    seed_offset = 0

    for L in sorted(normalized_sizes):
        for W in sorted(normalized_disorders):
            for split in sorted(normalized_counts):
                for realization in range(normalized_counts[split]):
                    manifest.append(
                        {
                            "split": split,
                            "L": int(L),
                            "W": float(W),
                            "realization": int(realization),
                            "seed": int(base_seed + seed_offset),
                        }
                    )
                    seed_offset += 1

    validate_seed_manifest(manifest)
    return manifest


def validate_seed_manifest(manifest: Sequence[dict]) -> None:
    """Validate manifest structure and seed independence."""
    if not isinstance(manifest, Sequence) or isinstance(
        manifest,
        (str, bytes),
    ):
        raise ValueError("manifest must be a sequence of dictionaries.")

    if len(manifest) == 0:
        raise ValueError("manifest must not be empty.")

    seen_seeds: set[int] = set()
    seen_entries: set[tuple[str, int, float, int]] = set()

    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError("Every manifest entry must be a dictionary.")

        if set(entry) != _REQUIRED_MANIFEST_KEYS:
            raise ValueError(
                "Every manifest entry must contain exactly: "
                "split, L, W, realization, seed."
            )

        split = entry["split"]
        L = entry["L"]
        W = entry["W"]
        realization = entry["realization"]
        seed = entry["seed"]

        _validate_split_name(split)

        if type(L) is not int or L < 2 or L % 2 != 0:
            raise ValueError(
                "Manifest L values must be even Python integers >= 2."
            )

        if type(W) is not float or not np.isfinite(W) or W < 0:
            raise ValueError(
                "Manifest W values must be finite nonnegative floats."
            )

        if type(realization) is not int or realization < 0:
            raise ValueError(
                "Manifest realization values must be nonnegative integers."
            )

        if type(seed) is not int or seed < 0:
            raise ValueError(
                "Manifest seed values must be nonnegative integers."
            )

        if seed in seen_seeds:
            raise ValueError(f"duplicate seed found: {seed}")

        entry_key = (
            split,
            L,
            W,
            realization,
        )

        if entry_key in seen_entries:
            raise ValueError(
                "duplicate realization entry found: "
                f"{entry_key}"
            )

        seen_seeds.add(seed)
        seen_entries.add(entry_key)


def make_time_grid(
    t_final: float,
    dt: float,
) -> np.ndarray:
    if not np.isfinite(t_final) or t_final <= 0:
        raise ValueError("t_final must be finite and positive.")

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")

    step_count_float = t_final / dt
    step_count = int(round(step_count_float))

    if not np.isclose(
        step_count_float,
        step_count,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("t_final must be an integer multiple of dt.")

    return np.linspace(
        0.0,
        float(t_final),
        step_count + 1,
        dtype=np.float64,
    )


def _float_tag(value: float) -> str:
    text = format(float(value), ".12g")
    return (
        text.replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def episode_filename(entry: Mapping[str, object]) -> str:
    return (
        f"L{entry['L']}"
        f"_W{_float_tag(float(entry['W']))}"
        f"_{entry['split']}"
        f"_{int(entry['realization']):04d}"
        f"_seed{entry['seed']}.pkl"
    )


def _atomic_write_json(
    value: object,
    path: Path,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    temporary_path.replace(path)


def _atomic_write_pickle(
    value: object,
    path: Path,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("wb") as handle:
        pickle.dump(
            value,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary_path.replace(path)


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _episode_matches_request(
    episode: dict,
    entry: Mapping[str, object],
    times: np.ndarray,
    J: float,
    delta: float,
) -> bool:
    required = {
        "L",
        "W",
        "seed",
        "J",
        "delta",
        "split",
        "realization",
        "fields",
        "times",
        "trajectory",
        "diagonal_ensemble",
        "channel_names",
    }

    if not required.issubset(episode):
        return False

    if episode["L"] != entry["L"]:
        return False

    if not np.isclose(episode["W"], entry["W"]):
        return False

    if episode["seed"] != entry["seed"]:
        return False

    if episode["split"] != entry["split"]:
        return False

    if episode["realization"] != entry["realization"]:
        return False

    if not np.isclose(episode["J"], J):
        return False

    if not np.isclose(episode["delta"], delta):
        return False

    stored_times = np.asarray(episode["times"])
    trajectory = np.asarray(episode["trajectory"])
    targets = np.asarray(episode["diagonal_ensemble"])
    fields = np.asarray(episode["fields"])

    if stored_times.shape != times.shape:
        return False

    if not np.allclose(
        stored_times,
        times,
        rtol=0.0,
        atol=1e-12,
    ):
        return False

    if fields.shape != (int(entry["L"]),):
        return False

    if trajectory.ndim != 2:
        return False

    if trajectory.shape[0] != times.size:
        return False

    if targets.ndim != 1:
        return False

    if trajectory.shape[1] != targets.size:
        return False

    if len(episode["channel_names"]) != targets.size:
        return False

    if not np.all(np.isfinite(fields)):
        return False

    if not np.all(np.isfinite(trajectory)):
        return False

    if not np.all(np.isfinite(targets)):
        return False

    return True


def generate_dataset(
    output_directory: str | Path,
    sizes: Sequence[int],
    disorders: Sequence[float],
    split_counts: Mapping[str, int],
    base_seed: int,
    t_final: float,
    dt: float,
    J: float = 1.0,
    delta: float = 1.0,
) -> None:
    if not np.isfinite(J):
        raise ValueError("J must be finite.")

    if not np.isfinite(delta):
        raise ValueError("delta must be finite.")

    output_path = Path(output_directory)
    episodes_path = output_path / "episodes"
    config_path = output_path / "config.json"
    manifest_path = output_path / "manifest.json"

    manifest = build_seed_manifest(
        sizes=sizes,
        disorders=disorders,
        split_counts=split_counts,
        base_seed=base_seed,
    )

    times = make_time_grid(
        t_final=t_final,
        dt=dt,
    )

    configuration = {
        "format_version": 1,
        "generator": "generate_ed_data.py",
        "model": "open_random_field_xxz",
        "sizes": sorted({int(L) for L in sizes}),
        "disorders": sorted({float(W) for W in disorders}),
        "split_counts": {
            split: int(split_counts[split])
            for split in sorted(split_counts)
        },
        "base_seed": int(base_seed),
        "J": float(J),
        "delta": float(delta),
        "t_initial": 0.0,
        "t_final": float(t_final),
        "dt": float(dt),
        "n_time_points": int(times.size),
        "initial_state": "up_down_up_down",
        "boundary_condition": "open",
        "disorder_distribution": "independent_uniform_minus_W_to_W",
        "spin_convention": "S_alpha = sigma_alpha / 2",
    }

    output_path.mkdir(parents=True, exist_ok=True)
    episodes_path.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        existing_configuration = _load_json(config_path)

        if existing_configuration != configuration:
            raise RuntimeError(
                "The output directory contains a different configuration. "
                "Use a new output directory."
            )
    else:
        _atomic_write_json(configuration, config_path)

    if manifest_path.exists():
        existing_manifest = _load_json(manifest_path)
        validate_seed_manifest(existing_manifest)

        if existing_manifest != manifest:
            raise RuntimeError(
                "The output directory contains a different seed manifest. "
                "Use a new output directory."
            )
    else:
        _atomic_write_json(manifest, manifest_path)

    total = len(manifest)

    for index, entry in enumerate(manifest, start=1):
        destination = episodes_path / episode_filename(entry)

        if destination.exists():
            with destination.open("rb") as handle:
                existing_episode = pickle.load(handle)

            if not _episode_matches_request(
                episode=existing_episode,
                entry=entry,
                times=times,
                J=J,
                delta=delta,
            ):
                raise RuntimeError(
                    f"Existing episode does not match the requested "
                    f"configuration: {destination}"
                )

            print(
                f"[{index:4d}/{total:4d}] existing "
                f"L={entry['L']} W={entry['W']} "
                f"{entry['split']} r={entry['realization']} "
                f"seed={entry['seed']}"
            )
            continue

        print(
            f"[{index:4d}/{total:4d}] generating "
            f"L={entry['L']} W={entry['W']} "
            f"{entry['split']} r={entry['realization']} "
            f"seed={entry['seed']}"
        )

        episode = generate_episode(
            L=int(entry["L"]),
            W=float(entry["W"]),
            seed=int(entry["seed"]),
            times=times,
            J=float(J),
            delta=float(delta),
        )

        episode["dataset_format_version"] = 1
        episode["split"] = str(entry["split"])
        episode["realization"] = int(entry["realization"])

        _atomic_write_pickle(
            episode,
            destination,
        )

    completion = {
        "complete": True,
        "episode_count": total,
    }
    _atomic_write_json(
        completion,
        output_path / "complete.json",
    )

    print(f"Completed {total} episodes in: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate exact-diagonalization trajectories and exact "
            "diagonal-ensemble targets."
        )
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
    )
    parser.add_argument(
        "--disorders",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISORDERS),
    )
    parser.add_argument(
        "--train",
        type=int,
        default=DEFAULT_SPLIT_COUNTS["train"],
    )
    parser.add_argument(
        "--validation",
        type=int,
        default=DEFAULT_SPLIT_COUNTS["validation"],
    )
    parser.add_argument(
        "--test",
        type=int,
        default=DEFAULT_SPLIT_COUNTS["test"],
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=DEFAULT_T_FINAL,
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
    )
    parser.add_argument(
        "--J",
        type=float,
        default=DEFAULT_J,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    generate_dataset(
        output_directory=arguments.output,
        sizes=arguments.sizes,
        disorders=arguments.disorders,
        split_counts={
            "train": arguments.train,
            "validation": arguments.validation,
            "test": arguments.test,
        },
        base_seed=arguments.base_seed,
        t_final=arguments.t_final,
        dt=arguments.dt,
        J=arguments.J,
        delta=arguments.delta,
    )


if __name__ == "__main__":
    main()