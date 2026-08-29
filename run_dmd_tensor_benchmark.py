#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark reduced-rank Dynamic Mode Decomposition (DMD) against the existing
STP and ridge forecasts of the full connected Pauli-correlation tensor.

This script is deliberately post hoc and surgical:
    - it reuses the existing tensor_cache/*.npz files;
    - it reuses the exact train/validation/test split logic;
    - it reads the already-saved STP/ridge prediction files;
    - it does not regenerate ED trajectories or alter previous results.

DMD definition
--------------
For each training ensemble, snapshot pairs from complete training episodes are
assembled as Y approximately A X. A reduced exact-DMD model is obtained from a
rank-r SVD of X. For a held-out episode, modal amplitudes are fitted by least
squares to the complete observed hindcast, not only its final snapshot. Future
states are then propagated with the learned DMD eigenvalues.

The DMD rank and eigenvalue treatment are selected using validation RMSE on the
same longest forecast lead used by the original tensor benchmark. Two standard
choices are tested:
    none       : unmodified exact-DMD eigenvalues
    unit_disk  : eigenvalues with modulus greater than one projected to |lambda|=1

Outputs are written to <results>/dmd_benchmark/ and include per-case and
aggregate metrics, paired bootstrap comparisons, selected hyperparameters,
hyperparameter scans, and DMD predictions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_RESULTS = "correlation_tensor_forecast_v1_r40"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 2_026_0802
DEFAULT_STABILIZATIONS = ("none", "unit_disk")
PHYSICAL_MIN = -1.0
PHYSICAL_MAX = 1.0
METHODS = (
    "stp",
    "ridge",
    "dmd",
    "stp_clipped",
    "ridge_clipped",
    "dmd_clipped",
    "persistence",
    "training_mean",
)
COMPARISONS = (
    ("dmd", "stp"),
    ("dmd", "ridge"),
    ("dmd_clipped", "stp_clipped"),
    ("dmd_clipped", "ridge_clipped"),
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


def save_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_tensor_script(explicit: str | None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Tensor benchmark script not found: {path}")
        return path.resolve()

    candidates = (
        Path("run_correlation_tensor_forecast.py"),
        Path("run_correlation_tensor_forecast(1).py"),
        Path(__file__).resolve().parent / "run_correlation_tensor_forecast.py",
        Path(__file__).resolve().parent
        / "run_correlation_tensor_forecast(1).py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate run_correlation_tensor_forecast.py. "
        "Pass it explicitly with --tensor-script."
    )


def load_tensor_module(path: Path):
    module_directory = str(path.parent)
    if module_directory not in sys.path:
        sys.path.insert(0, module_directory)
    specification = importlib.util.spec_from_file_location(
        "correlation_tensor_benchmark_module", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load module specification for {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def resolve_dataset_path(configured: str, results_path: Path) -> Path:
    candidates = (
        Path(configured),
        results_path.parent / configured,
        Path.cwd() / configured,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Dataset from results config could not be found: {configured}"
    )


def stabilize_eigenvalues(
    eigenvalues: np.ndarray, stabilization: str
) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=np.complex128).copy()
    if stabilization == "none":
        return values
    if stabilization == "unit_disk":
        magnitudes = np.abs(values)
        outside = magnitudes > 1.0
        values[outside] /= magnitudes[outside]
        return values
    raise ValueError(f"Unknown DMD stabilization: {stabilization}")


class ExactDMDPath:
    """One SVD of the training snapshot matrix, reusable across ranks."""

    def __init__(self, episodes: np.ndarray, maximum_rank: int) -> None:
        values = np.asarray(episodes, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError(
                "DMD episodes must have shape (episodes, times, features)."
            )
        if values.shape[0] < 2 or values.shape[1] < 2:
            raise ValueError("DMD requires at least two episodes and two times.")
        if not np.all(np.isfinite(values)):
            raise ValueError("DMD training episodes contain non-finite values.")
        if not isinstance(maximum_rank, int) or maximum_rank < 1:
            raise ValueError("maximum_rank must be a positive integer.")

        self.n_features = int(values.shape[2])
        X = values[:, :-1, :].reshape(-1, self.n_features).T
        Y = values[:, 1:, :].reshape(-1, self.n_features).T
        self.Y = Y

        U, singular_values, Vh = np.linalg.svd(X, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] <= 0.0:
            raise ValueError("DMD snapshot matrix has zero numerical rank.")

        tolerance = (
            np.finfo(np.float64).eps
            * max(X.shape)
            * float(singular_values[0])
        )
        numerical_rank = int(np.count_nonzero(singular_values > tolerance))
        retained = min(maximum_rank, numerical_rank)
        if retained < 1:
            raise ValueError("No valid DMD rank is available.")

        self.U = U[:, :retained]
        self.singular_values = singular_values[:retained]
        self.V = Vh.conj().T[:, :retained]
        self.numerical_rank = numerical_rank
        self.maximum_rank = retained

    def model(self, rank: int, stabilization: str) -> "ExactDMDModel":
        if not isinstance(rank, int) or not 1 <= rank <= self.maximum_rank:
            raise ValueError("Requested DMD rank is unavailable.")

        U = self.U[:, :rank]
        singular_values = self.singular_values[:rank]
        V = self.V[:, :rank]
        reduced_operator = (
            U.conj().T @ self.Y @ V / singular_values[None, :]
        )
        eigenvalues, eigenvectors = np.linalg.eig(reduced_operator)
        modes = (
            self.Y @ V / singular_values[None, :]
        ) @ eigenvectors
        eigenvalues = stabilize_eigenvalues(eigenvalues, stabilization)
        return ExactDMDModel(
            modes=modes,
            eigenvalues=eigenvalues,
            rank=rank,
            stabilization=stabilization,
        )


class ExactDMDModel:
    def __init__(
        self,
        modes: np.ndarray,
        eigenvalues: np.ndarray,
        rank: int,
        stabilization: str,
    ) -> None:
        self.modes = np.asarray(modes, dtype=np.complex128)
        self.eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
        self.rank = int(rank)
        self.stabilization = str(stabilization)
        if self.modes.shape[1] != self.eigenvalues.size:
            raise ValueError("DMD modes and eigenvalues are inconsistent.")

    def predict(
        self,
        hindcasts: np.ndarray,
        target_indices: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        observed = np.asarray(hindcasts, dtype=np.float64)
        indices = np.asarray(target_indices, dtype=np.int64)
        if observed.ndim != 3:
            raise ValueError("DMD hindcasts must be three-dimensional.")
        if observed.shape[2] != self.modes.shape[0]:
            raise ValueError("DMD hindcast feature dimension does not match.")
        if indices.ndim != 1 or indices.size < 1 or np.any(indices < 0):
            raise ValueError("target_indices must be nonnegative and nonempty.")
        if not np.all(np.isfinite(observed)):
            raise ValueError("DMD hindcasts contain non-finite values.")

        n_observed = observed.shape[1]
        observed_powers = np.arange(n_observed, dtype=np.int64)
        with np.errstate(over="ignore", invalid="ignore"):
            temporal_observed = self.eigenvalues[None, :] ** observed_powers[:, None]
            temporal_future = self.eigenvalues[None, :] ** indices[:, None]
        if not np.all(np.isfinite(temporal_observed)) or not np.all(
            np.isfinite(temporal_future)
        ):
            raise FloatingPointError("DMD eigenvalue powers became non-finite.")

        design = np.concatenate(
            [
                self.modes * temporal_observed[time_index][None, :]
                for time_index in range(n_observed)
            ],
            axis=0,
        )
        observations = observed.reshape(observed.shape[0], -1).T
        amplitudes, _, _, _ = np.linalg.lstsq(
            design, observations, rcond=None
        )

        predictions = np.empty(
            (observed.shape[0], indices.size, observed.shape[2]),
            dtype=np.complex128,
        )
        for output_index, powers in enumerate(temporal_future):
            predictions[:, output_index, :] = (
                self.modes @ (powers[:, None] * amplitudes)
            ).T

        maximum_imaginary = float(np.max(np.abs(predictions.imag)))
        real_predictions = np.asarray(predictions.real, dtype=np.float64)
        if not np.all(np.isfinite(real_predictions)):
            raise FloatingPointError("DMD predictions became non-finite.")
        return real_predictions, maximum_imaginary


def select_dmd_hyperparameters(
    train_episodes: np.ndarray,
    validation_hindcasts: np.ndarray,
    validation_targets: np.ndarray,
    target_indices: np.ndarray,
    ranks: Sequence[int],
    stabilizations: Sequence[str],
) -> tuple[int, str, list[dict], ExactDMDPath]:
    rank_candidates = sorted({int(value) for value in ranks if int(value) >= 1})
    if not rank_candidates:
        raise ValueError("No positive DMD ranks were supplied.")
    stabilization_candidates = []
    for value in stabilizations:
        text = str(value)
        if text not in DEFAULT_STABILIZATIONS:
            raise ValueError(f"Unknown DMD stabilization: {text}")
        if text not in stabilization_candidates:
            stabilization_candidates.append(text)
    if not stabilization_candidates:
        raise ValueError("No DMD stabilization choices were supplied.")

    path = ExactDMDPath(train_episodes, maximum_rank=max(rank_candidates))
    valid_ranks = [rank for rank in rank_candidates if rank <= path.maximum_rank]
    scan: list[dict] = []

    for rank in valid_ranks:
        for stabilization in stabilization_candidates:
            row = {
                "rank": int(rank),
                "stabilization": stabilization,
                "validation_rmse": None,
                "maximum_imaginary_part": None,
                "valid": False,
            }
            try:
                model = path.model(rank, stabilization)
                prediction, maximum_imaginary = model.predict(
                    validation_hindcasts, target_indices
                )
                row["validation_rmse"] = float(
                    np.sqrt(np.mean((prediction - validation_targets) ** 2))
                )
                row["maximum_imaginary_part"] = maximum_imaginary
                row["valid"] = True
            except (FloatingPointError, np.linalg.LinAlgError):
                pass
            scan.append(row)

    valid_rows = [row for row in scan if row["valid"]]
    if not valid_rows:
        raise RuntimeError("No finite DMD validation forecast was produced.")
    stabilization_order = {
        name: index for index, name in enumerate(stabilization_candidates)
    }
    best = min(
        valid_rows,
        key=lambda row: (
            row["validation_rmse"],
            row["rank"],
            stabilization_order[row["stabilization"]],
        ),
    )
    return int(best["rank"]), str(best["stabilization"]), scan, path


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
    denominator = truth_values.size
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
    bootstrap_denominator = (
        n_episodes * truth_values.shape[1] * truth_values.shape[2]
    )
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


def aggregate_case_metrics(
    cases: Sequence[dict],
    lead: float,
    region: str,
    method: str,
    tensor_module,
) -> dict | None:
    if not cases:
        return None
    L = int(cases[0]["L"])
    feature_mask = tensor_module.region_mask(L, region)
    squared_sum = 0.0
    absolute_sum = 0.0
    error_sum = 0.0
    count = 0
    episode_rmses: list[float] = []
    oob_count = 0
    baseline_squared_sum = 0.0

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
        baseline_squared_sum += float(np.sum((training_mean - truth) ** 2))

    if count == 0:
        return None
    mse = squared_sum / count
    baseline_mse = baseline_squared_sum / count
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
    tensor_module,
) -> dict | None:
    if not cases:
        return None
    L = int(cases[0]["L"])
    feature_mask = tensor_module.region_mask(L, region)
    case_data = []
    total_a = 0.0
    total_b = 0.0
    total_count = 0

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
    point = float(
        np.sqrt(total_a / total_count) - np.sqrt(total_b / total_count)
    )
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
    rng: np.random.Generator,
    tensor_module,
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
                        available, lead, region, method, tensor_module
                    )
                    if result is not None:
                        summary_rows.append(
                            {
                                "experiment": experiment,
                                "L": L,
                                "t_obs": t_obs,
                                "forecast_lead": lead,
                                "tensor_region": region,
                                "method": method,
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
                        rng,
                        tensor_module,
                    )
                    if result is not None:
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


def scalar_from_npz(archive: np.lib.npyio.NpzFile, key: str):
    value = archive[key]
    return value.item() if np.asarray(value).shape == () else value


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    eigenvalues = np.array([0.92, 0.8])
    modes = np.array(
        [[1.0, 0.2], [0.3, 1.0], [0.4, -0.5]], dtype=np.float64
    )
    amplitudes = rng.normal(size=(6, 2))
    times = np.arange(10)
    episodes = np.empty((amplitudes.shape[0], times.size, modes.shape[0]))
    for episode_index, amplitude in enumerate(amplitudes):
        episodes[episode_index] = np.stack(
            [modes @ (eigenvalues**time * amplitude) for time in times]
        )

    path = ExactDMDPath(episodes[:4], maximum_rank=2)
    model = path.model(rank=2, stabilization="none")
    prediction, maximum_imaginary = model.predict(
        episodes[4:, :4, :], np.arange(4, 10)
    )
    if not np.allclose(prediction, episodes[4:, 4:, :], atol=1e-9):
        raise AssertionError("Exact DMD linear-system forecast self-test failed.")
    if maximum_imaginary > 1e-9:
        raise AssertionError("Unexpected imaginary residual in DMD self-test.")

    stabilized = stabilize_eigenvalues(
        np.array([1.2 + 0.0j, 0.8 + 0.1j]), "unit_disk"
    )
    if np.max(np.abs(stabilized)) > 1.0 + 1e-12:
        raise AssertionError("Unit-disk stabilization self-test failed.")
    print("Self-test passed: exact linear forecast and unit-disk stabilization.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--tensor-script")
    parser.add_argument("--output")
    parser.add_argument("--dmd-ranks", nargs="+", type=int)
    parser.add_argument(
        "--stabilizations",
        nargs="+",
        choices=DEFAULT_STABILIZATIONS,
        default=list(DEFAULT_STABILIZATIONS),
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.self_test:
        run_self_test()
        return
    if arguments.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be nonnegative.")

    results_path = Path(arguments.results).resolve()
    if not results_path.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    config_path = results_path / "config.json"
    predictions_path = results_path / "predictions"
    cache_path = results_path / "tensor_cache"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing results config: {config_path}")
    if not predictions_path.is_dir():
        raise FileNotFoundError(f"Missing predictions directory: {predictions_path}")
    if not cache_path.is_dir():
        raise FileNotFoundError(f"Missing tensor cache: {cache_path}")

    tensor_script = find_tensor_script(arguments.tensor_script)
    tensor_module = load_tensor_module(tensor_script)
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("Results config.json must contain a dictionary.")

    dataset_path = resolve_dataset_path(str(config["dataset"]), results_path)
    tensor_stride = int(config["tensor_stride"])
    configured_ranks = [int(value) for value in config.get("ranks", [])]
    #dmd_ranks = (
    #    configured_ranks if arguments.dmd_ranks is None else arguments.dmd_ranks
    #)
    dmd_ranks = list(range(1, 81))
    dmd_ranks = sorted({rank for rank in dmd_ranks if rank >= 1})
    if not dmd_ranks:
        raise ValueError("No positive DMD ranks are available.")

    output_path = (
        Path(arguments.output).resolve()
        if arguments.output is not None
        else results_path / "dmd_benchmark"
    )
    dmd_predictions_path = output_path / "predictions"
    output_path.mkdir(parents=True, exist_ok=True)
    dmd_predictions_path.mkdir(parents=True, exist_ok=True)

    source_configuration, manifest = tensor_module.validate_dataset_directory(
        dataset_path
    )
    sizes = {int(value) for value in config["sizes"]}
    configured_disorders = config.get("disorders")
    disorders = (
        None
        if configured_disorders is None
        else {float(value) for value in configured_disorders}
    )
    groups = tensor_module.load_groups_with_tensor(
        dataset_path=dataset_path,
        cache_directory=cache_path,
        manifest=manifest,
        sizes=sizes,
        disorders=disorders,
        tensor_stride=tensor_stride,
    )

    prediction_files = sorted(predictions_path.glob("*.npz"))
    if not prediction_files:
        raise FileNotFoundError("No existing tensor prediction NPZ files found.")

    rng = np.random.default_rng(arguments.bootstrap_seed)
    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    hyperparameter_rows: list[dict] = []
    scans: list[dict] = []
    aggregate_cases: list[dict] = []
    all_forecast_leads = [float(value) for value in config["forecast_leads"]]

    for prediction_file in prediction_files:
        with np.load(prediction_file, allow_pickle=False) as archive:
            experiment = str(scalar_from_npz(archive, "experiment"))
            L = int(scalar_from_npz(archive, "L"))
            test_W = float(scalar_from_npz(archive, "test_W"))
            t_obs = float(scalar_from_npz(archive, "t_obs"))
            train_disorders = [
                float(value) for value in archive["train_disorders"].tolist()
            ]
            test_seeds = np.asarray(archive["test_seeds"], dtype=np.int64)
            target_times = np.asarray(archive["target_times"], dtype=np.float64)
            truth = np.asarray(archive["truth"], dtype=np.float64)
            existing_predictions = {
                method: np.asarray(archive[method], dtype=np.float64)
                for method in (
                    "stp",
                    "ridge",
                    "stp_clipped",
                    "ridge_clipped",
                    "persistence",
                    "training_mean",
                )
            }

        print(
            f"DMD: {experiment}, L={L}, test W={test_W}, "
            f"train W={train_disorders}, t_obs={t_obs}",
            flush=True,
        )

        train = tensor_module.combine_groups(
            groups, L, train_disorders, split="train"
        )
        validation = tensor_module.combine_groups(
            groups, L, train_disorders, split="validation"
        )
        test = tensor_module.combine_groups(
            groups, L, [test_W], split="test"
        )
        if not np.array_equal(test["seeds"], test_seeds):
            raise ValueError(
                f"Test seed order does not match existing predictions: {prediction_file}"
            )

        tensor_times = np.asarray(train["tensor_times"], dtype=np.float64)
        split_index = tensor_module.exact_split_index(tensor_times, t_obs)
        last_target_index = int(
            np.flatnonzero(np.isclose(tensor_times, target_times[-1]))[0]
        )
        target_indices = np.array(
            [
                int(np.flatnonzero(np.isclose(tensor_times, value))[0])
                for value in target_times
            ],
            dtype=np.int64,
        )
        if np.any(target_indices < split_index):
            raise ValueError("Existing target times overlap the DMD hindcast.")

        train_flat = tensor_module.flatten_tensor_episodes(train["tensors"])
        validation_flat = tensor_module.flatten_tensor_episodes(
            validation["tensors"]
        )
        test_flat = tensor_module.flatten_tensor_episodes(test["tensors"])
        train_for_dmd = train_flat[:, : last_target_index + 1, :]
        validation_hindcasts = validation_flat[:, :split_index, :]
        validation_targets = validation_flat[:, target_indices, :]

        best_rank, best_stabilization, scan, _ = select_dmd_hyperparameters(
            train_episodes=train_for_dmd,
            validation_hindcasts=validation_hindcasts,
            validation_targets=validation_targets,
            target_indices=target_indices,
            ranks=dmd_ranks,
            stabilizations=arguments.stabilizations,
        )

        fitting_flat = np.concatenate([train_flat, validation_flat], axis=0)
        final_path = ExactDMDPath(
            fitting_flat[:, : last_target_index + 1, :],
            maximum_rank=best_rank,
        )
        final_model = final_path.model(best_rank, best_stabilization)
        dmd_prediction, maximum_imaginary = final_model.predict(
            test_flat[:, :split_index, :], target_indices
        )
        dmd_clipped = np.clip(dmd_prediction, PHYSICAL_MIN, PHYSICAL_MAX)

        if dmd_prediction.shape != truth.shape:
            raise ValueError("DMD prediction shape does not match stored truth.")
        predictions = {
            **existing_predictions,
            "dmd": dmd_prediction,
            "dmd_clipped": dmd_clipped,
        }

        output_prediction_file = dmd_predictions_path / prediction_file.name
        np.savez_compressed(
            output_prediction_file,
            experiment=np.array(experiment),
            L=np.array(L, dtype=np.int64),
            test_W=np.array(test_W, dtype=np.float64),
            train_disorders=np.asarray(train_disorders, dtype=np.float64),
            t_obs=np.array(t_obs, dtype=np.float64),
            test_seeds=test_seeds,
            target_times=target_times,
            truth=truth.astype(np.float32),
            **{
                method: prediction.astype(np.float32)
                for method, prediction in predictions.items()
            },
        )

        available_leads = [
            lead
            for lead in all_forecast_leads
            if np.any(target_times <= t_obs + lead + 1e-12)
        ]
        for lead in available_leads:
            lead_mask = target_times <= t_obs + lead + 1e-12
            lead_truth = truth[:, lead_mask, :]
            lead_predictions = {
                method: prediction[:, lead_mask, :]
                for method, prediction in predictions.items()
            }
            for region in REGIONS:
                feature_mask = tensor_module.region_mask(L, region)
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
                    }
                    row.update(
                        tensor_module.metrics(
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
                            **paired_bootstrap_rmse_difference(
                                truth=lead_truth,
                                prediction_a=lead_predictions[method_a],
                                prediction_b=lead_predictions[method_b],
                                feature_mask=feature_mask,
                                n_bootstrap=arguments.bootstrap_samples,
                                rng=rng,
                            ),
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
            "maximum_forecast_lead": float(target_times[-1] - t_obs),
            "dmd_rank": best_rank,
            "dmd_stabilization": best_stabilization,
            "dmd_rank_at_grid_max": best_rank == max(dmd_ranks),
            "dmd_numerical_rank": final_path.numerical_rank,
            "maximum_imaginary_part": maximum_imaginary,
        }
        hyperparameter_rows.append(hyperparameter_row)
        scans.append({"case": hyperparameter_row, "dmd_scan": scan})
        aggregate_cases.append(
            {
                "experiment": experiment,
                "L": L,
                "test_W": test_W,
                "t_obs": t_obs,
                "target_times": target_times,
                "truth": truth,
                "predictions": predictions,
            }
        )

    aggregate_summary, aggregate_comparisons = aggregate_results(
        aggregate_cases,
        all_forecast_leads,
        arguments.bootstrap_samples,
        rng,
        tensor_module,
    )

    output_config = {
        "script": "run_dmd_tensor_benchmark.py",
        "source_results": str(results_path),
        "source_tensor_script": str(tensor_script),
        "dataset": str(dataset_path),
        "source_dataset_configuration": source_configuration,
        "dmd_definition": "reduced exact DMD with full-hindcast amplitude fit",
        "dmd_ranks": dmd_ranks,
        "dmd_stabilizations": list(arguments.stabilizations),
        "hyperparameter_selection": (
            "validation tensor RMSE on the longest stored forecast lead"
        ),
        "bootstrap_samples": int(arguments.bootstrap_samples),
        "bootstrap_seed": int(arguments.bootstrap_seed),
    }
    save_json(output_config, output_path / "config.json")
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
    print(f"DMD benchmark complete: {output_path}", flush=True)


if __name__ == "__main__":
    main()
