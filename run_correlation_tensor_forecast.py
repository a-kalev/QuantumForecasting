#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecast the future full two-point Pauli correlation tensor from its early-time
history in the existing exact-diagonalization dataset.

This is the direct quantum analogue of full-field Space-Time Projection (STP):
STP receives an early-time high-dimensional correlation field and forecasts the
future field, rather than predicting one scalar observable.

Reused project components
-------------------------
    - heisenberg_ed.HeisenbergModel
    - heisenberg_ed.build_hamiltonian
    - heisenberg_ed.neel_state
    - canonical_stp.CanonicalSTP
    - generate_ed_data.episode_filename
    - run_validation dataset validation/loading utilities

The existing episode files store local observables and disorder fields, but not
wavefunctions or the full correlation tensor. This script therefore rebuilds
exactly the stored Hamiltonian from each episode's disorder fields, regenerates
the state trajectory, computes the tensor on a configurable coarsened time
axis, and caches it.

Tensor definition
-----------------
For a,b in {x,y,z}, this script uses the real symmetrized connected covariance

    C_ij^{ab}(t) = Re <sigma_i^a sigma_j^b>
                   - <sigma_i^a><sigma_j^b>.

For i != j the Pauli operators commute, so this is the ordinary connected
correlator. For i == j it is the symmetrized on-site covariance. The stored
array order is

    tensor[time, i, j, alpha, beta],  alpha,beta = x,y,z.

The tensor is real and lies in [-1,1] up to numerical tolerance.

Experiments
-----------
within
    Train/validate/test at fixed (L,W) using the existing 30/10/10 splits.

transfer
    Train and validate on all other W values at fixed L; test on the held-out
    disorder using only its original test split.

Methods
-------
    stp             canonical STP
    stp_clipped     STP clipped to [-1,1]
    ridge           memory-efficient dual multioutput ridge regression
    ridge_clipped   ridge clipped to [-1,1]
    persistence     repeat the final observed tensor
    training_mean   fitting-set mean future tensor

Hyperparameters are selected on validation data only, using the longest
requested forecast lead. Final models are refitted on train+validation data.
Metrics are reported for the complete tensor and for off-site entries i != j,
which isolate the nontrivial correlation/light-cone field.

Outputs
-------
    config.json
    summary.csv
    comparisons.csv
    aggregate_summary.csv
    aggregate_comparisons.csv
    selected_hyperparameters.csv
    hyperparameter_scans.json
    tensor_cache/*.npz
    predictions/*.npz

Compatibility note
------------------
The local fixed-magnetization basis builder uses bin(...).count("1") instead of
int.bit_count(), matching the existing site/bit convention while remaining
compatible with older Python versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse.linalg import expm_multiply

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from src.canonical_stp import CanonicalSTP
    from src.heisenberg_ed import (
        HeisenbergModel,
        build_hamiltonian,
        neel_state,
    )
except ImportError:
    from canonical_stp import CanonicalSTP
    from heisenberg_ed import (
        HeisenbergModel,
        build_hamiltonian,
        neel_state,
    )

from generate_ed_data import episode_filename
from run_validation import (
    load_episode,
    safe_float_tag,
    save_json,
    validate_dataset_directory,
)


DEFAULT_DATASET = "ed_validation_v1"
DEFAULT_OUTPUT = "correlation_tensor_forecast_v1_r40"
DEFAULT_SIZES = (12, 14)
DEFAULT_EXPERIMENTS = ("within", "transfer")
DEFAULT_T_OBS = (10.0, 20.0)
DEFAULT_FORECAST_LEADS = (5.0, 10.0, 20.0, 40.0, 80.0)
DEFAULT_RANKS = tuple(range(1, 41))
DEFAULT_RIDGE_ALPHAS = (
    1e-8,
    1e-6,
    1e-4,
    1e-2,
    1.0,
    100.0,
    1e4,
    1e6,
)
DEFAULT_TENSOR_STRIDE = 5
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 2_026_0801
CACHE_FORMAT_VERSION = 1
TENSOR_KIND = "connected_symmetrized_pauli_covariance"
PAULI_LABELS = ("x", "y", "z")
PHYSICAL_MIN = -1.0
PHYSICAL_MAX = 1.0
METHODS = (
    "stp",
    "stp_clipped",
    "ridge",
    "ridge_clipped",
    "persistence",
    "training_mean",
)
COMPARISONS = (
    ("stp", "ridge"),
    ("stp", "persistence"),
    ("stp", "training_mean"),
    ("stp_clipped", "ridge_clipped"),
)
REGIONS = ("all", "offsite")


def write_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows were produced for {path.name}.")

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fixed_magnetization_basis_compatible(L: int, n_up: int) -> np.ndarray:
    """Match heisenberg_ed.fixed_magnetization_basis without bit_count()."""
    if L < 1 or not 0 <= n_up <= L:
        raise ValueError("Require L >= 1 and 0 <= n_up <= L.")

    expected_size = comb(L, n_up)
    basis = np.fromiter(
        (
            state
            for state in range(1 << L)
            if bin(state).count("1") == n_up
        ),
        dtype=np.int64,
        count=expected_size,
    )
    if basis.size != expected_size or np.any(np.diff(basis) <= 0):
        raise RuntimeError("Failed to construct the fixed-magnetization basis.")
    return basis


def expected_tensor_times(times: np.ndarray, stride: int) -> np.ndarray:
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("tensor-stride must be a positive integer.")
    indices = np.arange(0, times.size, stride, dtype=np.int64)
    if indices.size < 2:
        raise ValueError("tensor-stride leaves fewer than two time points.")
    sampled = np.asarray(times[indices], dtype=np.float64)
    if np.any(np.diff(sampled) <= 0):
        raise ValueError("Sampled tensor times are not strictly increasing.")
    return sampled


def tensor_cache_path(cache_directory: Path, entry: dict) -> Path:
    return cache_directory / Path(episode_filename(entry)).with_suffix(".npz")


def full_state_from_sector(
    sector_state: np.ndarray,
    basis: np.ndarray,
    L: int,
) -> np.ndarray:
    state = np.asarray(sector_state, dtype=np.complex128)
    if state.shape != (basis.size,):
        raise ValueError("Sector-state dimension does not match the basis.")
    full = np.zeros(1 << L, dtype=np.complex128)
    full[basis] = state
    return full


def pauli_covariance_tensor(full_state: np.ndarray, L: int) -> np.ndarray:
    """Return C[i,j,alpha,beta] for alpha,beta=(x,y,z)."""
    psi = np.asarray(full_state, dtype=np.complex128)
    dimension = 1 << L
    if psi.shape != (dimension,):
        raise ValueError(f"full_state must have shape ({dimension},).")

    norm_error = abs(float(np.vdot(psi, psi).real) - 1.0)
    if norm_error > 1e-9:
        raise ValueError(f"State is not normalized; error={norm_error:.3e}.")

    indices = np.arange(dimension, dtype=np.int64)
    applied = np.empty((dimension, 3 * L), dtype=np.complex128)

    for site in range(L):
        shift = L - 1 - site
        mask = 1 << shift
        bits = (indices >> shift) & 1
        z_eigenvalues = 2.0 * bits.astype(np.float64) - 1.0
        destinations = indices ^ mask

        x_column = 3 * site
        y_column = x_column + 1
        z_column = x_column + 2

        applied[destinations, x_column] = psi
        y_phase = np.where(bits == 0, -1j, 1j)
        applied[destinations, y_column] = y_phase * psi
        applied[:, z_column] = z_eigenvalues * psi

    one_point = np.conj(psi) @ applied
    one_point_imaginary_error = float(np.max(np.abs(one_point.imag)))
    if one_point_imaginary_error > 1e-9:
        raise RuntimeError(
            "Hermitian one-point expectations have a non-negligible "
            f"imaginary part: {one_point_imaginary_error:.3e}."
        )

    gram = applied.conj().T @ applied
    covariance = gram.real - np.outer(one_point.real, one_point.real)
    covariance = 0.5 * (covariance + covariance.T)

    tensor = covariance.reshape(L, 3, L, 3).transpose(0, 2, 1, 3)

    exchange_error = float(
        np.max(np.abs(tensor - tensor.transpose(1, 0, 3, 2)))
    )
    if exchange_error > 1e-9:
        raise RuntimeError(
            "Correlation-tensor exchange symmetry failed; "
            f"error={exchange_error:.3e}."
        )

    lower = float(np.min(tensor))
    upper = float(np.max(tensor))
    if lower < PHYSICAL_MIN - 1e-8 or upper > PHYSICAL_MAX + 1e-8:
        raise RuntimeError(
            "Exact covariance tensor violates [-1,1]; "
            f"range=[{lower:.12g},{upper:.12g}]."
        )

    return np.asarray(tensor, dtype=np.float64)


def load_tensor_cache(
    path: Path,
    entry: dict,
    expected_times: np.ndarray,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing tensor cache: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {
            "format_version",
            "tensor_kind",
            "L",
            "W",
            "seed",
            "times",
            "pauli_labels",
            "tensor",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Tensor cache is missing {sorted(missing)}: {path}")

        format_version = int(data["format_version"])
        tensor_kind = str(data["tensor_kind"])
        L = int(data["L"])
        W = float(data["W"])
        seed = int(data["seed"])
        times = np.asarray(data["times"], dtype=np.float64)
        labels = tuple(str(value) for value in data["pauli_labels"])
        tensor = np.asarray(data["tensor"], dtype=np.float64)

    if format_version != CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported tensor-cache version in {path}.")
    if tensor_kind != TENSOR_KIND:
        raise ValueError(f"Tensor-kind mismatch in {path}.")
    if L != int(entry["L"]):
        raise ValueError(f"Tensor-cache L mismatch: {path}")
    if not np.isclose(W, float(entry["W"])):
        raise ValueError(f"Tensor-cache W mismatch: {path}")
    if seed != int(entry["seed"]):
        raise ValueError(f"Tensor-cache seed mismatch: {path}")
    if not np.array_equal(times, expected_times):
        raise ValueError(f"Tensor-cache time-grid mismatch: {path}")
    if labels != PAULI_LABELS:
        raise ValueError(f"Tensor-cache Pauli-label mismatch: {path}")
    if tensor.shape != (times.size, L, L, 3, 3):
        raise ValueError(f"Tensor-cache shape mismatch: {path}")
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"Tensor cache contains non-finite values: {path}")
    if np.min(tensor) < PHYSICAL_MIN - 1e-6:
        raise ValueError(f"Tensor cache is below the physical bound: {path}")
    if np.max(tensor) > PHYSICAL_MAX + 1e-6:
        raise ValueError(f"Tensor cache is above the physical bound: {path}")

    return {"times": times, "tensor": tensor}


def compute_tensor_cache_for_entry(
    dataset_path: Path,
    cache_directory: Path,
    entry: dict,
    tensor_stride: int,
    overwrite: bool,
) -> Path:
    episode = load_episode(dataset_path, entry)
    source_times = np.asarray(episode["times"], dtype=np.float64)
    tensor_times = expected_tensor_times(source_times, tensor_stride)
    destination = tensor_cache_path(cache_directory, entry)

    if destination.exists() and not overwrite:
        load_tensor_cache(destination, entry, tensor_times)
        return destination

    L = int(episode["L"])
    fields = np.asarray(episode.get("fields"), dtype=np.float64)
    if fields.shape != (L,) or not np.all(np.isfinite(fields)):
        raise ValueError(
            f"Episode lacks valid stored fields: {episode_filename(entry)}"
        )

    J = float(episode.get("J", 1.0))
    delta = float(episode.get("delta", 1.0))
    n_up = int(episode.get("n_up", L // 2))

    model = HeisenbergModel(
        L=L,
        W=float(episode["W"]),
        seed=int(episode["seed"]),
        J=J,
        delta=delta,
        n_up=n_up,
    )
    basis = fixed_magnetization_basis_compatible(L=L, n_up=n_up)
    hamiltonian = build_hamiltonian(model, fields, basis)
    psi0 = neel_state(L=L, basis=basis, first_spin_up=True)

    generator = (-1j) * hamiltonian
    trace_generator = (-1j) * np.sum(hamiltonian.diagonal())
    states = expm_multiply(
        generator,
        psi0,
        start=float(tensor_times[0]),
        stop=float(tensor_times[-1]),
        num=tensor_times.size,
        endpoint=True,
        traceA=trace_generator,
    )

    norm_error = float(
        np.max(np.abs(np.sum(np.abs(states) ** 2, axis=1) - 1.0))
    )
    if norm_error > 1e-9:
        raise RuntimeError(
            f"State evolution lost normalization for {episode_filename(entry)}; "
            f"error={norm_error:.3e}."
        )

    tensor = np.empty((tensor_times.size, L, L, 3, 3), dtype=np.float32)
    for time_index, sector_state in enumerate(states):
        full_state = full_state_from_sector(sector_state, basis, L)
        tensor[time_index] = pauli_covariance_tensor(full_state, L).astype(
            np.float32
        )

    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.array(CACHE_FORMAT_VERSION, dtype=np.int64),
            tensor_kind=np.array(TENSOR_KIND),
            L=np.array(L, dtype=np.int64),
            W=np.array(float(episode["W"]), dtype=np.float64),
            seed=np.array(int(episode["seed"]), dtype=np.int64),
            J=np.array(J, dtype=np.float64),
            delta=np.array(delta, dtype=np.float64),
            n_up=np.array(n_up, dtype=np.int64),
            times=tensor_times,
            pauli_labels=np.asarray(PAULI_LABELS),
            tensor=tensor,
        )
    temporary.replace(destination)
    return destination


def build_tensor_cache(
    dataset_path: Path,
    cache_directory: Path,
    manifest: Sequence[dict],
    sizes: set[int],
    disorders: set[float] | None,
    tensor_stride: int,
    overwrite: bool,
) -> None:
    selected = [
        entry
        for entry in manifest
        if int(entry["L"]) in sizes
        and (
            disorders is None
            or float(entry["W"]) in disorders
        )
    ]
    if not selected:
        raise ValueError("No manifest entries match the requested cache scope.")

    for index, entry in enumerate(selected, start=1):
        path = compute_tensor_cache_for_entry(
            dataset_path=dataset_path,
            cache_directory=cache_directory,
            entry=entry,
            tensor_stride=tensor_stride,
            overwrite=overwrite,
        )
        print(
            f"[{index}/{len(selected)}] tensor cache ready: {path.name}",
            flush=True,
        )


def load_groups_with_tensor(
    dataset_path: Path,
    cache_directory: Path,
    manifest: Sequence[dict],
    sizes: set[int],
    disorders: set[float] | None,
    tensor_stride: int,
) -> dict[tuple[int, float, str], dict]:
    entries_by_group: dict[tuple[int, float, str], list[dict]] = defaultdict(list)
    for entry in manifest:
        L = int(entry["L"])
        W = float(entry["W"])
        if L not in sizes:
            continue
        if disorders is not None and W not in disorders:
            continue
        entries_by_group[(L, W, str(entry["split"]))].append(entry)

    groups: dict[tuple[int, float, str], dict] = {}
    for key in sorted(entries_by_group):
        L, W, split = key
        entries = sorted(
            entries_by_group[key], key=lambda item: int(item["realization"])
        )
        episodes = [load_episode(dataset_path, entry) for entry in entries]
        source_times = np.asarray(episodes[0]["times"], dtype=np.float64)
        tensor_times = expected_tensor_times(source_times, tensor_stride)

        tensors = []
        seeds = []
        for entry, episode in zip(entries, episodes):
            if not np.array_equal(
                np.asarray(episode["times"], dtype=np.float64), source_times
            ):
                raise ValueError(f"Source time-grid mismatch for {key}.")
            cached = load_tensor_cache(
                tensor_cache_path(cache_directory, entry),
                entry=entry,
                expected_times=tensor_times,
            )
            tensors.append(cached["tensor"])
            seeds.append(int(entry["seed"]))

        tensor_array = np.stack(tensors, axis=0)
        seed_array = np.asarray(seeds, dtype=np.int64)
        if seed_array.size != np.unique(seed_array).size:
            raise ValueError(f"Repeated seeds in group {key}.")

        groups[key] = {
            "L": L,
            "W": W,
            "split": split,
            "tensor_times": tensor_times,
            "tensors": tensor_array,
            "seeds": seed_array,
        }

    all_seeds = np.concatenate([group["seeds"] for group in groups.values()])
    if all_seeds.size != np.unique(all_seeds).size:
        raise ValueError("Loaded groups contain reused disorder seeds.")
    return groups


def combine_groups(
    groups: dict[tuple[int, float, str], dict],
    L: int,
    disorders: Sequence[float],
    split: str,
) -> dict:
    selected = []
    for W in sorted(float(value) for value in disorders):
        key = (int(L), W, split)
        if key not in groups:
            raise KeyError(f"Dataset group not found: {key}")
        selected.append(groups[key])

    reference = selected[0]
    for group in selected[1:]:
        if not np.array_equal(group["tensor_times"], reference["tensor_times"]):
            raise ValueError("Cannot combine groups with different tensor times.")
        if group["tensors"].shape[2:] != reference["tensors"].shape[2:]:
            raise ValueError("Cannot combine groups with different tensor shapes.")

    tensors = np.concatenate([group["tensors"] for group in selected], axis=0)
    seeds = np.concatenate([group["seeds"] for group in selected])
    labels = np.concatenate(
        [
            np.full(group["seeds"].size, group["W"], dtype=np.float64)
            for group in selected
        ]
    )
    if seeds.size != np.unique(seeds).size:
        raise ValueError("Repeated seeds while combining groups.")

    order = np.argsort(seeds)
    return {
        "L": int(L),
        "disorders": [float(value) for value in disorders],
        "split": split,
        "tensor_times": reference["tensor_times"],
        "tensors": tensors[order],
        "seeds": seeds[order],
        "disorder_labels": labels[order],
    }


def exact_split_index(times: np.ndarray, t_obs: float) -> int:
    index = int(np.searchsorted(times, t_obs, side="right"))
    if index < 2:
        raise ValueError("t_obs must contain at least two tensor time points.")
    if index >= times.size:
        raise ValueError("t_obs must precede the final tensor time.")
    if not np.isclose(times[index - 1], t_obs, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"t_obs={t_obs} is not present on the tensor time grid."
        )
    return index


def forecast_time_mask(
    tensor_times: np.ndarray,
    t_obs: float,
    maximum_lead: float,
) -> np.ndarray:
    mask = (tensor_times > t_obs + 1e-12) & (
        tensor_times <= t_obs + maximum_lead + 1e-12
    )
    if not np.any(mask):
        raise ValueError(
            f"No target times exist after t_obs={t_obs} within lead={maximum_lead}."
        )
    return mask


def flatten_tensor_episodes(tensors: np.ndarray) -> np.ndarray:
    values = np.asarray(tensors, dtype=np.float64)
    if values.ndim != 6 or values.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected tensors with shape (episodes,times,L,L,3,3)."
        )
    return values.reshape(values.shape[0], values.shape[1], -1)


def fit_stp_path(
    train_hindcasts: np.ndarray,
    train_targets: np.ndarray,
    ranks: Sequence[int],
) -> tuple[CanonicalSTP, list[int]]:
    candidates = sorted({int(rank) for rank in ranks if int(rank) >= 1})
    if not candidates:
        raise ValueError("No positive STP ranks were supplied.")

    last_error: Exception | None = None
    for maximum_rank in reversed(candidates):
        try:
            model = CanonicalSTP(rank=maximum_rank).fit(
                hindcast_training=train_hindcasts,
                forecast_training=train_targets,
            )
            return model, [rank for rank in candidates if rank <= maximum_rank]
        except ValueError as exc:
            last_error = exc
    raise RuntimeError("No valid STP rank was available.") from last_error


def predict_stp_at_rank(
    model: CanonicalSTP,
    hindcasts: np.ndarray,
    rank: int,
) -> np.ndarray:
    observed = np.asarray(hindcasts, dtype=np.float64)
    if observed.shape[1:] != model.hindcast_shape:
        raise ValueError("Hindcast shape does not match the fitted STP path.")
    if rank < 1 or rank > model.rank:
        raise ValueError("Requested rank is outside the fitted STP path.")

    centered = (observed - model.hindcast_mean).reshape(
        observed.shape[0], -1
    ).T
    hindcast_modes = model.hindcast_modes[:, :rank]
    forecast_modes = model.forecast_modes[:, :rank]
    coefficients = hindcast_modes.conj().T @ centered
    prediction = (forecast_modes @ coefficients).T.reshape(
        (observed.shape[0],) + model.forecast_shape
    )
    return np.asarray(prediction + model.forecast_mean, dtype=np.float64)


class DualRidgePath:
    """Multioutput ridge without materializing a feature-by-target matrix."""

    def __init__(self, hindcasts: np.ndarray, targets: np.ndarray) -> None:
        X_values = np.asarray(hindcasts, dtype=np.float64)
        Y_values = np.asarray(targets, dtype=np.float64)
        if X_values.ndim != 3:
            raise ValueError("Ridge hindcasts must be three-dimensional.")
        if Y_values.shape[0] != X_values.shape[0]:
            raise ValueError("Ridge episode counts do not match.")

        self.hindcast_shape = X_values.shape[1:]
        self.target_shape = Y_values.shape[1:]
        X = X_values.reshape(X_values.shape[0], -1)
        Y = Y_values.reshape(Y_values.shape[0], -1)

        self.feature_mean = np.mean(X, axis=0)
        X_centered = X - self.feature_mean
        self.feature_scale = np.std(X_centered, axis=0, ddof=0)
        self.feature_scale[self.feature_scale < 1e-12] = 1.0
        self.X_scaled = X_centered / self.feature_scale
        self.target_mean = np.mean(Y, axis=0)
        self.Y_centered = Y - self.target_mean
        self.gram = self.X_scaled @ self.X_scaled.T

    def predict(self, hindcasts: np.ndarray, alpha: float) -> np.ndarray:
        values = np.asarray(hindcasts, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != self.hindcast_shape:
            raise ValueError("Invalid hindcasts for ridge prediction.")

        regularized = self.gram + float(alpha) * np.eye(self.gram.shape[0])
        dual = np.linalg.solve(regularized, self.Y_centered)

        X_test = values.reshape(values.shape[0], -1)
        X_test = (X_test - self.feature_mean) / self.feature_scale
        test_train_kernel = X_test @ self.X_scaled.T
        prediction = test_train_kernel @ dual + self.target_mean
        return prediction.reshape((values.shape[0],) + self.target_shape)


def select_hyperparameters(
    train_hindcasts: np.ndarray,
    train_targets: np.ndarray,
    validation_hindcasts: np.ndarray,
    validation_targets: np.ndarray,
    ranks: Sequence[int],
    ridge_alphas: Sequence[float],
) -> tuple[int, float, list[dict], list[dict]]:
    stp_path, valid_ranks = fit_stp_path(
        train_hindcasts=train_hindcasts,
        train_targets=train_targets,
        ranks=ranks,
    )
    rank_scan = []
    for rank in valid_ranks:
        prediction = predict_stp_at_rank(stp_path, validation_hindcasts, rank)
        rank_scan.append(
            {
                "rank": int(rank),
                "validation_rmse": float(
                    np.sqrt(np.mean((prediction - validation_targets) ** 2))
                ),
            }
        )

    alpha_candidates = sorted(
        {
            float(alpha)
            for alpha in ridge_alphas
            if np.isfinite(alpha) and float(alpha) >= 0.0
        }
    )
    if not alpha_candidates:
        raise ValueError("No valid ridge alphas were supplied.")

    ridge_path = DualRidgePath(train_hindcasts, train_targets)
    alpha_scan = []
    for alpha in alpha_candidates:
        prediction = ridge_path.predict(validation_hindcasts, alpha)
        alpha_scan.append(
            {
                "alpha": float(alpha),
                "validation_rmse": float(
                    np.sqrt(np.mean((prediction - validation_targets) ** 2))
                ),
            }
        )

    best_rank = min(
        rank_scan, key=lambda row: (row["validation_rmse"], row["rank"])
    )["rank"]
    best_alpha = min(
        alpha_scan, key=lambda row: (row["validation_rmse"], row["alpha"])
    )["alpha"]
    return int(best_rank), float(best_alpha), rank_scan, alpha_scan


def region_mask(L: int, region: str) -> np.ndarray:
    mask = np.ones((L, L, 3, 3), dtype=bool)
    if region == "offsite":
        site_mask = ~np.eye(L, dtype=bool)
        mask &= site_mask[:, :, None, None]
    elif region != "all":
        raise ValueError(f"Unknown tensor region: {region}")
    return mask.reshape(-1)


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    feature_mask: np.ndarray,
    training_mean_prediction: np.ndarray,
) -> dict:
    truth_values = np.asarray(truth, dtype=np.float64)[:, :, feature_mask]
    prediction_values = np.asarray(prediction, dtype=np.float64)[
        :, :, feature_mask
    ]
    mean_values = np.asarray(training_mean_prediction, dtype=np.float64)[
        :, :, feature_mask
    ]
    if truth_values.shape != prediction_values.shape:
        raise ValueError("Truth and prediction shapes do not match.")

    errors = prediction_values - truth_values
    baseline_errors = mean_values - truth_values
    mse = float(np.mean(errors**2))
    baseline_mse = float(np.mean(baseline_errors**2))
    skill = None if baseline_mse <= 0.0 else float(1.0 - mse / baseline_mse)

    ensemble_mean_error = np.mean(prediction_values, axis=0) - np.mean(
        truth_values, axis=0
    )
    if truth_values.shape[0] > 1:
        truth_variance = np.var(truth_values, axis=0, ddof=1)
        prediction_variance = np.var(prediction_values, axis=0, ddof=1)
        denominator = float(np.mean(truth_variance))
        variance_ratio = (
            float(np.mean(prediction_variance) / denominator)
            if denominator > 0.0
            else None
        )
    else:
        variance_ratio = None

    out_of_bounds = (prediction_values < PHYSICAL_MIN) | (
        prediction_values > PHYSICAL_MAX
    )
    per_episode_rmse = np.sqrt(np.mean(errors**2, axis=(1, 2)))
    return {
        "n_episodes": int(truth_values.shape[0]),
        "n_forecast_times": int(truth_values.shape[1]),
        "n_tensor_features": int(truth_values.shape[2]),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "mean_episode_rmse": float(np.mean(per_episode_rmse)),
        "median_episode_rmse": float(np.median(per_episode_rmse)),
        "ensemble_mean_rmse": float(
            np.sqrt(np.mean(ensemble_mean_error**2))
        ),
        "boundary_rmse": float(np.sqrt(np.mean(errors[:, 0, :] ** 2))),
        "variance_ratio": variance_ratio,
        "skill_vs_training_mean": skill,
        "out_of_bounds_count": int(np.count_nonzero(out_of_bounds)),
        "out_of_bounds_fraction": float(np.mean(out_of_bounds)),
    }


def paired_bootstrap_rmse_difference(
    truth: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    feature_mask: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    truth_values = truth[:, :, feature_mask]
    a_values = prediction_a[:, :, feature_mask]
    b_values = prediction_b[:, :, feature_mask]
    squared_a = np.sum((a_values - truth_values) ** 2, axis=(1, 2))
    squared_b = np.sum((b_values - truth_values) ** 2, axis=(1, 2))
    denominator = truth_values.shape[0] * truth_values.shape[1] * truth_values.shape[2]
    point = float(
        np.sqrt(np.sum(squared_a) / denominator)
        - np.sqrt(np.sum(squared_b) / denominator)
    )

    if n_bootstrap <= 0:
        return {
            "rmse_difference": point,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_samples": 0,
        }

    n_episodes = truth_values.shape[0]
    indices = rng.integers(0, n_episodes, size=(n_bootstrap, n_episodes))
    bootstrap_denominator = n_episodes * truth_values.shape[1] * truth_values.shape[2]
    rmse_a = np.sqrt(
        np.sum(squared_a[indices], axis=1) / bootstrap_denominator
    )
    rmse_b = np.sqrt(
        np.sum(squared_b[indices], axis=1) / bootstrap_denominator
    )
    differences = rmse_a - rmse_b
    return {
        "rmse_difference": point,
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "bootstrap_samples": int(n_bootstrap),
    }


def fit_and_predict(
    fitting: dict,
    test: dict,
    t_obs: float,
    maximum_lead: float,
    rank: int,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    times = np.asarray(fitting["tensor_times"], dtype=np.float64)
    split_index = exact_split_index(times, t_obs)
    target_mask = forecast_time_mask(times, t_obs, maximum_lead)
    target_times = times[target_mask]

    fitting_flat = flatten_tensor_episodes(fitting["tensors"])
    test_flat = flatten_tensor_episodes(test["tensors"])
    train_hindcasts = fitting_flat[:, :split_index, :]
    test_hindcasts = test_flat[:, :split_index, :]
    train_targets = fitting_flat[:, target_mask, :]
    truth = test_flat[:, target_mask, :]

    stp = CanonicalSTP(rank=rank).fit(train_hindcasts, train_targets)
    stp_prediction = stp.predict_many(test_hindcasts)["forecast"]

    ridge = DualRidgePath(train_hindcasts, train_targets)
    ridge_prediction = ridge.predict(test_hindcasts, ridge_alpha)

    persistence = np.broadcast_to(
        test_hindcasts[:, -1:, :], truth.shape
    ).copy()
    training_mean = np.broadcast_to(
        np.mean(train_targets, axis=0), truth.shape
    ).copy()

    predictions = {
        "stp": np.asarray(stp_prediction, dtype=np.float64),
        "stp_clipped": np.clip(stp_prediction, PHYSICAL_MIN, PHYSICAL_MAX),
        "ridge": np.asarray(ridge_prediction, dtype=np.float64),
        "ridge_clipped": np.clip(
            ridge_prediction, PHYSICAL_MIN, PHYSICAL_MAX
        ),
        "persistence": persistence,
        "training_mean": training_mean,
    }
    return target_times, truth, predictions


def run_case(
    experiment: str,
    L: int,
    test_W: float,
    train_disorders: Sequence[float],
    train: dict,
    validation: dict,
    test: dict,
    t_obs: float,
    forecast_leads: Sequence[float],
    ranks: Sequence[int],
    ridge_alphas: Sequence[float],
    predictions_directory: Path,
    bootstrap_samples: int,
    bootstrap_rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict, dict, dict]:
    if not set(train["seeds"]).isdisjoint(validation["seeds"]):
        raise ValueError("Train and validation seeds overlap.")
    if not set(train["seeds"]).isdisjoint(test["seeds"]):
        raise ValueError("Train and test seeds overlap.")
    if not set(validation["seeds"]).isdisjoint(test["seeds"]):
        raise ValueError("Validation and test seeds overlap.")

    leads = sorted({float(value) for value in forecast_leads if value > 0.0})
    maximum_lead = min(
        max(leads), float(test["tensor_times"][-1] - t_obs)
    )
    valid_leads = [lead for lead in leads if lead <= maximum_lead + 1e-12]
    if not valid_leads:
        raise ValueError("No requested forecast lead is available.")

    times = np.asarray(train["tensor_times"], dtype=np.float64)
    split_index = exact_split_index(times, t_obs)
    target_mask = forecast_time_mask(times, t_obs, maximum_lead)

    train_flat = flatten_tensor_episodes(train["tensors"])
    validation_flat = flatten_tensor_episodes(validation["tensors"])
    train_hindcasts = train_flat[:, :split_index, :]
    validation_hindcasts = validation_flat[:, :split_index, :]
    train_targets = train_flat[:, target_mask, :]
    validation_targets = validation_flat[:, target_mask, :]

    best_rank, best_alpha, rank_scan, alpha_scan = select_hyperparameters(
        train_hindcasts=train_hindcasts,
        train_targets=train_targets,
        validation_hindcasts=validation_hindcasts,
        validation_targets=validation_targets,
        ranks=ranks,
        ridge_alphas=ridge_alphas,
    )

    fitting = {
        **train,
        "tensors": np.concatenate(
            [train["tensors"], validation["tensors"]], axis=0
        ),
        "seeds": np.concatenate([train["seeds"], validation["seeds"]]),
    }
    target_times, truth, predictions = fit_and_predict(
        fitting=fitting,
        test=test,
        t_obs=t_obs,
        maximum_lead=maximum_lead,
        rank=best_rank,
        ridge_alpha=best_alpha,
    )

    predictions_directory.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{experiment}_L{L}_W{safe_float_tag(test_W)}_"
        f"tobs{safe_float_tag(t_obs)}"
    )
    np.savez_compressed(
        predictions_directory / f"{tag}.npz",
        experiment=np.array(experiment),
        tensor_kind=np.array(TENSOR_KIND),
        pauli_labels=np.asarray(PAULI_LABELS),
        L=np.array(L, dtype=np.int64),
        test_W=np.array(test_W, dtype=np.float64),
        train_disorders=np.asarray(train_disorders, dtype=np.float64),
        t_obs=np.array(t_obs, dtype=np.float64),
        test_seeds=np.asarray(test["seeds"], dtype=np.int64),
        target_times=target_times,
        tensor_shape=np.asarray((L, L, 3, 3), dtype=np.int64),
        truth=truth.astype(np.float32),
        **{
            method: prediction.astype(np.float32)
            for method, prediction in predictions.items()
        },
    )

    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    region_masks = {region: region_mask(L, region) for region in REGIONS}

    for lead in valid_leads:
        lead_mask = target_times <= t_obs + lead + 1e-12
        if not np.any(lead_mask):
            continue
        lead_truth = truth[:, lead_mask, :]
        lead_predictions = {
            method: prediction[:, lead_mask, :]
            for method, prediction in predictions.items()
        }

        for region in REGIONS:
            feature_mask = region_masks[region]
            for method in METHODS:
                row = {
                    "experiment": experiment,
                    "L": L,
                    "test_W": test_W,
                    "train_disorders": ";".join(
                        format(value, ".12g") for value in train_disorders
                    ),
                    "t_obs": t_obs,
                    "forecast_lead": lead,
                    "tensor_region": region,
                    "method": method,
                    "tensor_kind": TENSOR_KIND,
                }
                row.update(
                    metrics(
                        truth=lead_truth,
                        prediction=lead_predictions[method],
                        feature_mask=feature_mask,
                        training_mean_prediction=lead_predictions[
                            "training_mean"
                        ],
                    )
                )
                summary_rows.append(row)

            for method_a, method_b in COMPARISONS:
                result = paired_bootstrap_rmse_difference(
                    truth=lead_truth,
                    prediction_a=lead_predictions[method_a],
                    prediction_b=lead_predictions[method_b],
                    feature_mask=feature_mask,
                    n_bootstrap=bootstrap_samples,
                    rng=bootstrap_rng,
                )
                comparison_rows.append(
                    {
                        "experiment": experiment,
                        "L": L,
                        "test_W": test_W,
                        "train_disorders": ";".join(
                            format(value, ".12g")
                            for value in train_disorders
                        ),
                        "t_obs": t_obs,
                        "forecast_lead": lead,
                        "tensor_region": region,
                        "method_a": method_a,
                        "method_b": method_b,
                        **result,
                    }
                )

    rank_values = sorted({int(value) for value in ranks if int(value) >= 1})
    alpha_values = sorted(
        {
            float(value)
            for value in ridge_alphas
            if np.isfinite(value) and float(value) >= 0.0
        }
    )
    hyperparameter_row = {
        "experiment": experiment,
        "L": L,
        "test_W": test_W,
        "train_disorders": ";".join(
            format(value, ".12g") for value in train_disorders
        ),
        "t_obs": t_obs,
        "maximum_forecast_lead": maximum_lead,
        "n_input_times": split_index,
        "n_input_tensor_features": int(L * L * 9),
        "stp_rank": best_rank,
        "stp_rank_at_grid_max": best_rank == max(rank_values),
        "ridge_alpha": best_alpha,
        "ridge_alpha_at_grid_min": best_alpha == min(alpha_values),
        "ridge_alpha_at_grid_max": best_alpha == max(alpha_values),
    }
    scan = {
        "case": hyperparameter_row,
        "stp_rank_scan": rank_scan,
        "ridge_alpha_scan": alpha_scan,
    }
    aggregate_payload = {
        "experiment": experiment,
        "L": L,
        "test_W": test_W,
        "t_obs": t_obs,
        "target_times": target_times,
        "truth": truth,
        "predictions": predictions,
    }
    return (
        summary_rows,
        comparison_rows,
        hyperparameter_row,
        scan,
        aggregate_payload,
    )


def aggregate_case_metrics(
    cases: Sequence[dict],
    lead: float,
    region: str,
    method: str,
) -> dict | None:
    if not cases:
        return None
    L = int(cases[0]["L"])
    feature_mask = region_mask(L, region)

    squared_sum = 0.0
    absolute_sum = 0.0
    error_sum = 0.0
    count = 0
    episode_rmses = []
    oob_count = 0
    training_mean_squared_sum = 0.0

    for case in cases:
        mask = case["target_times"] <= case["t_obs"] + lead + 1e-12
        if not np.any(mask):
            continue
        truth = case["truth"][:, mask, :][:, :, feature_mask]
        prediction = case["predictions"][method][:, mask, :][
            :, :, feature_mask
        ]
        training_mean = case["predictions"]["training_mean"][:, mask, :][
            :, :, feature_mask
        ]
        errors = prediction - truth
        squared_sum += float(np.sum(errors**2))
        absolute_sum += float(np.sum(np.abs(errors)))
        error_sum += float(np.sum(errors))
        count += int(errors.size)
        episode_rmses.extend(
            np.sqrt(np.mean(errors**2, axis=(1, 2))).tolist()
        )
        oob_count += int(
            np.count_nonzero(
                (prediction < PHYSICAL_MIN) | (prediction > PHYSICAL_MAX)
            )
        )
        training_mean_squared_sum += float(
            np.sum((training_mean - truth) ** 2)
        )

    if count == 0:
        return None
    mse = squared_sum / count
    baseline_mse = training_mean_squared_sum / count
    return {
        "n_cases": len(cases),
        "n_values": count,
        "rmse": float(np.sqrt(mse)),
        "mae": float(absolute_sum / count),
        "bias": float(error_sum / count),
        "mean_episode_rmse": float(np.mean(episode_rmses)),
        "median_episode_rmse": float(np.median(episode_rmses)),
        "skill_vs_training_mean": (
            None if baseline_mse <= 0.0 else float(1.0 - mse / baseline_mse)
        ),
        "out_of_bounds_count": oob_count,
        "out_of_bounds_fraction": float(oob_count / count),
    }


def aggregate_bootstrap_difference(
    cases: Sequence[dict],
    lead: float,
    region: str,
    method_a: str,
    method_b: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict | None:
    if not cases:
        return None
    L = int(cases[0]["L"])
    feature_mask = region_mask(L, region)

    case_data = []
    total_count = 0
    total_a = 0.0
    total_b = 0.0
    for case in cases:
        mask = case["target_times"] <= case["t_obs"] + lead + 1e-12
        if not np.any(mask):
            continue
        truth = case["truth"][:, mask, :][:, :, feature_mask]
        prediction_a = case["predictions"][method_a][:, mask, :][
            :, :, feature_mask
        ]
        prediction_b = case["predictions"][method_b][:, mask, :][
            :, :, feature_mask
        ]
        squared_a = np.sum((prediction_a - truth) ** 2, axis=(1, 2))
        squared_b = np.sum((prediction_b - truth) ** 2, axis=(1, 2))
        values_per_episode = truth.shape[1] * truth.shape[2]
        case_data.append((squared_a, squared_b, values_per_episode))
        total_a += float(np.sum(squared_a))
        total_b += float(np.sum(squared_b))
        total_count += int(truth.size)

    if total_count == 0:
        return None
    point = float(np.sqrt(total_a / total_count) - np.sqrt(total_b / total_count))
    if n_bootstrap <= 0:
        return {
            "rmse_difference": point,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_samples": 0,
        }

    differences = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_index in range(n_bootstrap):
        sum_a = 0.0
        sum_b = 0.0
        count = 0
        for squared_a, squared_b, values_per_episode in case_data:
            indices = rng.integers(0, squared_a.size, size=squared_a.size)
            sum_a += float(np.sum(squared_a[indices]))
            sum_b += float(np.sum(squared_b[indices]))
            count += int(indices.size * values_per_episode)
        differences[bootstrap_index] = np.sqrt(sum_a / count) - np.sqrt(
            sum_b / count
        )

    return {
        "rmse_difference": point,
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "bootstrap_samples": int(n_bootstrap),
    }


def aggregate_results(
    cases: Sequence[dict],
    forecast_leads: Sequence[float],
    bootstrap_samples: int,
    bootstrap_rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    grouped: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    for case in cases:
        grouped[(case["experiment"], case["L"], case["t_obs"])].append(case)

    for (experiment, L, t_obs), selected in sorted(grouped.items()):
        for lead in sorted({float(value) for value in forecast_leads}):
            available = [
                case
                for case in selected
                if np.any(
                    case["target_times"] <= case["t_obs"] + lead + 1e-12
                )
            ]
            if not available:
                continue
            for region in REGIONS:
                for method in METHODS:
                    result = aggregate_case_metrics(
                        available, lead, region, method
                    )
                    if result is None:
                        continue
                    summary_rows.append(
                        {
                            "experiment": experiment,
                            "L": L,
                            "t_obs": t_obs,
                            "forecast_lead": lead,
                            "tensor_region": region,
                            "method": method,
                            "tensor_kind": TENSOR_KIND,
                            **result,
                        }
                    )

                for method_a, method_b in COMPARISONS:
                    result = aggregate_bootstrap_difference(
                        available,
                        lead,
                        region,
                        method_a,
                        method_b,
                        bootstrap_samples,
                        bootstrap_rng,
                    )
                    if result is None:
                        continue
                    comparison_rows.append(
                        {
                            "experiment": experiment,
                            "L": L,
                            "t_obs": t_obs,
                            "forecast_lead": lead,
                            "tensor_region": region,
                            "method_a": method_a,
                            "method_b": method_b,
                            "n_cases": len(available),
                            **result,
                        }
                    )
    return summary_rows, comparison_rows


def run_self_test() -> None:
    basis = fixed_magnetization_basis_compatible(L=2, n_up=1)
    expected_basis = np.array([1, 2], dtype=np.int64)
    if not np.array_equal(basis, expected_basis):
        raise AssertionError("Fixed-magnetization basis self-test failed.")

    product = np.zeros(4, dtype=np.complex128)
    product[1] = 1.0  # |down,up> in the existing MSB/site convention.
    product_tensor = pauli_covariance_tensor(product, L=2)
    if not np.isclose(product_tensor[0, 0, 0, 0], 1.0, atol=1e-12):
        raise AssertionError("Product-state on-site x variance failed.")
    if not np.isclose(product_tensor[0, 1, 2, 2], 0.0, atol=1e-12):
        raise AssertionError("Product-state connected zz correlation failed.")

    # Project convention: ordered one-site basis is [down, up], so
    # sigma_z=diag(-1,+1) and sigma_y=[[0,i],[-i,0]].
    theta = 0.37
    one_site = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
    one_site_tensor = pauli_covariance_tensor(one_site, L=1)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.array([[0.0, 1j], [-1j, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    paulis = (sigma_x, sigma_y, sigma_z)
    means = np.array([np.vdot(one_site, op @ one_site).real for op in paulis])
    for alpha in range(3):
        for beta in range(3):
            symmetrized = 0.5 * (
                paulis[alpha] @ paulis[beta]
                + paulis[beta] @ paulis[alpha]
            )
            expected_value = (
                np.vdot(one_site, symmetrized @ one_site).real
                - means[alpha] * means[beta]
            )
            if not np.isclose(
                one_site_tensor[0, 0, alpha, beta],
                expected_value,
                atol=1e-12,
            ):
                raise AssertionError(
                    "One-site Pauli-convention self-test failed."
                )

    bell = np.zeros(4, dtype=np.complex128)
    bell[1] = 1.0 / np.sqrt(2.0)
    bell[2] = 1.0 / np.sqrt(2.0)
    bell_tensor = pauli_covariance_tensor(bell, L=2)
    expected = {
        (0, 0): 1.0,
        (1, 1): 1.0,
        (2, 2): -1.0,
    }
    for (alpha, beta), value in expected.items():
        observed = bell_tensor[0, 1, alpha, beta]
        if not np.isclose(observed, value, atol=1e-12):
            raise AssertionError(
                "Bell-state correlation self-test failed for "
                f"({alpha},{beta}): {observed} != {value}."
            )

    rng = np.random.default_rng(7)
    hindcasts = rng.normal(size=(6, 3, 5))
    targets = rng.normal(size=(6, 2, 7))
    test = rng.normal(size=(2, 3, 5))
    alpha = 0.3
    dual_path = DualRidgePath(hindcasts, targets)
    dual_prediction = dual_path.predict(test, alpha)

    X = hindcasts.reshape(6, -1)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    X_scale = np.std(X_centered, axis=0, ddof=0)
    X_scale[X_scale < 1e-12] = 1.0
    X_scaled = X_centered / X_scale
    Y = targets.reshape(6, -1)
    Y_mean = np.mean(Y, axis=0)
    Y_centered = Y - Y_mean
    weights = X_scaled.T @ np.linalg.solve(
        X_scaled @ X_scaled.T + alpha * np.eye(6), Y_centered
    )
    test_scaled = (test.reshape(2, -1) - X_mean) / X_scale
    direct_prediction = (test_scaled @ weights + Y_mean).reshape(2, 2, 7)
    if not np.allclose(dual_prediction, direct_prediction, atol=1e-12):
        raise AssertionError("Dual-ridge equivalence self-test failed.")

    print(
        "Self-test passed: basis convention, product/Bell tensors, "
        "and dual-ridge equivalence."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--disorders", nargs="+", type=float)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("within", "transfer"),
        default=list(DEFAULT_EXPERIMENTS),
    )
    parser.add_argument(
        "--t-obs", nargs="+", type=float, default=list(DEFAULT_T_OBS)
    )
    parser.add_argument(
        "--forecast-leads",
        nargs="+",
        type=float,
        default=list(DEFAULT_FORECAST_LEADS),
    )
    parser.add_argument(
        "--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS)
    )
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_RIDGE_ALPHAS),
    )
    parser.add_argument(
        "--tensor-stride", type=int, default=DEFAULT_TENSOR_STRIDE
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.self_test:
        run_self_test()
        return
    if arguments.cache_only and arguments.benchmark_only:
        raise ValueError("cache-only and benchmark-only are mutually exclusive.")
    if arguments.tensor_stride < 1:
        raise ValueError("tensor-stride must be positive.")
    if arguments.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be nonnegative.")

    dataset_path = Path(arguments.dataset)
    output_path = Path(arguments.output)
    cache_directory = output_path / "tensor_cache"
    predictions_directory = output_path / "predictions"
    output_path.mkdir(parents=True, exist_ok=True)

    source_configuration, manifest = validate_dataset_directory(dataset_path)
    available_sizes = {int(entry["L"]) for entry in manifest}
    sizes = set(int(value) for value in arguments.sizes).intersection(
        available_sizes
    )
    if not sizes:
        raise ValueError("No requested system sizes are present in the dataset.")

    disorders = (
        None
        if arguments.disorders is None
        else set(float(value) for value in arguments.disorders)
    )

    if not arguments.benchmark_only:
        build_tensor_cache(
            dataset_path=dataset_path,
            cache_directory=cache_directory,
            manifest=manifest,
            sizes=sizes,
            disorders=disorders,
            tensor_stride=arguments.tensor_stride,
            overwrite=arguments.overwrite_cache,
        )
    if arguments.cache_only:
        return

    groups = load_groups_with_tensor(
        dataset_path=dataset_path,
        cache_directory=cache_directory,
        manifest=manifest,
        sizes=sizes,
        disorders=disorders,
        tensor_stride=arguments.tensor_stride,
    )

    disorders_by_size: dict[int, list[float]] = {}
    for L in sorted(sizes):
        values = sorted({W for (group_L, W, _) in groups if group_L == L})
        if values:
            disorders_by_size[L] = values
    if not disorders_by_size:
        raise ValueError("No complete requested groups were loaded.")

    config = {
        "script": "run_correlation_tensor_forecast.py",
        "dataset": str(dataset_path),
        "source_dataset_configuration": source_configuration,
        "scope": "full_spatiotemporal_pauli_correlation_tensor_forecasting",
        "tensor_definition": (
            "C_ij_ab = Re<sigma_i^a sigma_j^b> "
            "- <sigma_i^a><sigma_j^b>"
        ),
        "tensor_kind": TENSOR_KIND,
        "tensor_axis_order": ["time", "i", "j", "alpha", "beta"],
        "pauli_labels": list(PAULI_LABELS),
        "physical_interval": [PHYSICAL_MIN, PHYSICAL_MAX],
        "sizes": sorted(sizes),
        "disorders": (
            None if disorders is None else sorted(disorders)
        ),
        "experiments": list(arguments.experiments),
        "t_obs": [float(value) for value in arguments.t_obs],
        "forecast_leads": [
            float(value) for value in arguments.forecast_leads
        ],
        "ranks": [int(value) for value in arguments.ranks],
        "ridge_alphas": [
            float(value) for value in arguments.ridge_alphas
        ],
        "tensor_stride": int(arguments.tensor_stride),
        "bootstrap_samples": int(arguments.bootstrap_samples),
        "bootstrap_seed": int(arguments.bootstrap_seed),
        "hyperparameter_selection": (
            "validation RMSE on the longest requested forecast lead"
        ),
        "ridge_implementation": (
            "dual prediction; no feature-by-target weight matrix"
        ),
    }
    save_json(config, output_path / "config.json")

    bootstrap_rng = np.random.default_rng(arguments.bootstrap_seed)
    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    hyperparameter_rows: list[dict] = []
    scans: list[dict] = []
    aggregate_payloads: list[dict] = []

    for L, available_disorders in sorted(disorders_by_size.items()):
        for test_W in available_disorders:
            for experiment in arguments.experiments:
                if experiment == "within":
                    train_disorders = [test_W]
                elif experiment == "transfer":
                    train_disorders = [
                        W for W in available_disorders if W != test_W
                    ]
                    if not train_disorders:
                        continue
                else:
                    raise ValueError(f"Unknown experiment: {experiment}")

                train = combine_groups(
                    groups, L, train_disorders, split="train"
                )
                validation = combine_groups(
                    groups, L, train_disorders, split="validation"
                )
                test = combine_groups(groups, L, [test_W], split="test")

                for t_obs in sorted({float(value) for value in arguments.t_obs}):
                    print(
                        f"Running {experiment}: L={L}, test W={test_W}, "
                        f"train W={train_disorders}, t_obs={t_obs}",
                        flush=True,
                    )
                    (
                        case_summary,
                        case_comparisons,
                        hyperparameter_row,
                        scan,
                        aggregate_payload,
                    ) = run_case(
                        experiment=experiment,
                        L=L,
                        test_W=test_W,
                        train_disorders=train_disorders,
                        train=train,
                        validation=validation,
                        test=test,
                        t_obs=t_obs,
                        forecast_leads=arguments.forecast_leads,
                        ranks=arguments.ranks,
                        ridge_alphas=arguments.ridge_alphas,
                        predictions_directory=predictions_directory,
                        bootstrap_samples=arguments.bootstrap_samples,
                        bootstrap_rng=bootstrap_rng,
                    )
                    summary_rows.extend(case_summary)
                    comparison_rows.extend(case_comparisons)
                    hyperparameter_rows.append(hyperparameter_row)
                    scans.append(scan)
                    aggregate_payloads.append(aggregate_payload)

    if not summary_rows:
        raise RuntimeError("No benchmark rows were produced.")

    aggregate_summary, aggregate_comparisons = aggregate_results(
        cases=aggregate_payloads,
        forecast_leads=arguments.forecast_leads,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_rng=bootstrap_rng,
    )

    write_csv(summary_rows, output_path / "summary.csv")
    write_csv(comparison_rows, output_path / "comparisons.csv")
    write_csv(
        hyperparameter_rows, output_path / "selected_hyperparameters.csv"
    )
    write_csv(aggregate_summary, output_path / "aggregate_summary.csv")
    write_csv(
        aggregate_comparisons, output_path / "aggregate_comparisons.csv"
    )
    save_json(scans, output_path / "hyperparameter_scans.json")

    print(f"Benchmark complete: {output_path}", flush=True)


if __name__ == "__main__":
    main()
