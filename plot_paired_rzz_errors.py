#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot paired per-realization errors in the forecasted correlation radius.

The script reads only prediction files written by
run_correlation_tensor_forecast.py:

    RESULTS/predictions/*.npz

For every test realization and every selected benchmark case, it computes

    x = MAE_t[R_baseline(t), R_exact(t)]
    y = MAE_t[R_method(t),   R_exact(t)]

for the requested connected Pauli component, using only forecast times stored
in each prediction file. Points below y=x favor the method.

The correlation radius is evaluated separately for each realization:

    G_ab(r,t) = sqrt(mean_{central references and equal-distance sites}
                         |C_ij^{ab}(t)|^2)

    R_ab(t)^2 = sum_{r>=1} r^2 G_ab(r,t)^2
                / sum_{r>=1} G_ab(r,t)^2.

A hierarchical paired bootstrap resamples benchmark cases and then test
realizations within each case. This avoids treating realizations from the same
case as fully independent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

TOLERANCE = 1e-10
DEFAULT_RESULTS = "correlation_tensor_forecast_v1_r40"
DEFAULT_COMPONENT = "zz"
DEFAULT_METHOD = "stp"
DEFAULT_BASELINE = "ridge"
VALID_COMPONENTS = tuple(a + b for a in "xyz" for b in "xyz")


def _scalar(value: np.ndarray):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected a scalar array.")
    result = array.reshape(()).item()
    if isinstance(result, bytes):
        return result.decode("utf-8")
    return result


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=TOLERANCE))


def component_indices(component: str) -> Tuple[int, int]:
    normalized = component.lower()
    if normalized not in VALID_COMPONENTS:
        raise ValueError(
            f"component must be one of {', '.join(VALID_COMPONENTS)}."
        )
    labels = {"x": 0, "y": 1, "z": 2}
    return labels[normalized[0]], labels[normalized[1]]


def central_reference_sites(L: int) -> Tuple[int, ...]:
    if L < 2:
        raise ValueError("L must be at least 2.")
    return tuple(sorted({L // 2 - 1, L // 2}))


def load_case(path: Path, method: str, baseline: str) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "experiment",
            "L",
            "test_W",
            "t_obs",
            "test_seeds",
            "target_times",
            "tensor_shape",
            "pauli_labels",
            "truth",
            method,
            baseline,
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")

        experiment = str(_scalar(data["experiment"]))
        L = int(_scalar(data["L"]))
        test_W = float(_scalar(data["test_W"]))
        t_obs = float(_scalar(data["t_obs"]))
        seeds = np.asarray(data["test_seeds"], dtype=np.int64)
        times = np.asarray(data["target_times"], dtype=np.float64)
        tensor_shape = tuple(int(value) for value in data["tensor_shape"])
        labels = tuple(str(value) for value in data["pauli_labels"])

        if tensor_shape != (L, L, 3, 3):
            raise ValueError(
                f"{path}: tensor_shape={tensor_shape}, expected {(L, L, 3, 3)}."
            )
        if labels != ("x", "y", "z"):
            raise ValueError(f"{path}: unexpected Pauli labels {labels}.")
        if seeds.ndim != 1 or seeds.size < 1:
            raise ValueError(f"{path}: test_seeds must be one-dimensional.")
        if times.ndim != 1 or times.size < 1:
            raise ValueError(f"{path}: target_times must be one-dimensional.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{path}: target_times are not strictly increasing.")
        if np.any(times <= t_obs + TOLERANCE):
            raise ValueError(f"{path}: target_times must lie after t_obs.")

        expected = (seeds.size, times.size, L * L * 9)
        arrays: Dict[str, np.ndarray] = {}
        for name in ("truth", method, baseline):
            values = np.asarray(data[name], dtype=np.float64)
            if values.shape != expected:
                raise ValueError(
                    f"{path}: {name} shape {values.shape}, expected {expected}."
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: {name} contains non-finite values.")
            arrays[name] = values.reshape(seeds.size, times.size, L, L, 3, 3)

    return {
        "path": path,
        "experiment": experiment,
        "L": L,
        "test_W": test_W,
        "t_obs": t_obs,
        "seeds": seeds,
        "times": times,
        "truth": arrays["truth"],
        "method": arrays[method],
        "baseline": arrays[baseline],
    }


def episode_distance_rms_map(
    tensors: np.ndarray,
    component: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return distances and G_ab[episode, time, distance]."""
    values = np.asarray(tensors, dtype=np.float64)
    if values.ndim != 6 or values.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected tensors with shape (episodes,times,L,L,3,3)."
        )

    L = values.shape[2]
    if values.shape[3] != L:
        raise ValueError("The spatial tensor dimensions must be square.")

    alpha, beta = component_indices(component)
    selected = values[..., alpha, beta]
    references = central_reference_sites(L)

    distances: List[int] = []
    maps: List[np.ndarray] = []
    for distance in range(L):
        samples = []
        for reference in references:
            for site in range(L):
                if abs(site - reference) == distance:
                    samples.append(selected[:, :, site, reference])
        if samples:
            stacked = np.stack(samples, axis=-1)
            maps.append(np.sqrt(np.mean(stacked**2, axis=-1)))
            distances.append(distance)

    if not maps:
        raise ValueError("No distance-resolved tensor samples were found.")

    return (
        np.asarray(distances, dtype=np.float64),
        np.stack(maps, axis=-1),
    )


def correlation_radius_per_episode(
    distances: np.ndarray,
    lightcone: np.ndarray,
) -> np.ndarray:
    """Return R_ab[episode, time]."""
    r = np.asarray(distances, dtype=np.float64)
    values = np.asarray(lightcone, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != r.size:
        raise ValueError(
            "lightcone must have shape (episodes, times, n_distances)."
        )

    offsite = r > 0.0
    if not np.any(offsite):
        raise ValueError("At least one off-site distance is required.")

    weights = values[:, :, offsite] ** 2
    numerator = np.sum(weights * (r[offsite][None, None, :] ** 2), axis=2)
    denominator = np.sum(weights, axis=2)

    radius = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 1e-30
    radius[valid] = np.sqrt(numerator[valid] / denominator[valid])
    return radius


def per_episode_mae(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("Radius arrays must have matching (episodes,time) shape.")

    result = np.full(truth.shape[0], np.nan, dtype=np.float64)
    for episode in range(truth.shape[0]):
        valid = np.isfinite(truth[episode]) & np.isfinite(prediction[episode])
        if np.any(valid):
            result[episode] = float(
                np.mean(np.abs(prediction[episode, valid] - truth[episode, valid]))
            )
    return result


def case_identifier(case: Dict[str, object]) -> str:
    return (
        f"{case['experiment']}_L{case['L']}_"
        f"W{format(float(case['test_W']), '.12g')}_"
        f"tobs{format(float(case['t_obs']), '.12g')}"
    )


def matches_filters(
    case: Dict[str, object],
    experiments: Optional[Sequence[str]],
    sizes: Optional[Sequence[int]],
    disorders: Optional[Sequence[float]],
    observation_times: Optional[Sequence[float]],
) -> bool:
    if experiments and str(case["experiment"]) not in set(experiments):
        return False
    if sizes and int(case["L"]) not in set(int(value) for value in sizes):
        return False
    if disorders and not any(
        _close(float(case["test_W"]), float(value)) for value in disorders
    ):
        return False
    if observation_times and not any(
        _close(float(case["t_obs"]), float(value)) for value in observation_times
    ):
        return False
    return True


def build_rows(
    prediction_paths: Iterable[Path],
    component: str,
    method: str,
    baseline: str,
    experiments: Optional[Sequence[str]],
    sizes: Optional[Sequence[int]],
    disorders: Optional[Sequence[float]],
    observation_times: Optional[Sequence[float]],
) -> List[dict]:
    rows: List[dict] = []

    for path in sorted(prediction_paths):
        case = load_case(path, method=method, baseline=baseline)
        if not matches_filters(
            case,
            experiments=experiments,
            sizes=sizes,
            disorders=disorders,
            observation_times=observation_times,
        ):
            continue

        distances, truth_map = episode_distance_rms_map(case["truth"], component)
        method_distances, method_map = episode_distance_rms_map(
            case["method"], component
        )
        baseline_distances, baseline_map = episode_distance_rms_map(
            case["baseline"], component
        )
        if not np.array_equal(distances, method_distances) or not np.array_equal(
            distances, baseline_distances
        ):
            raise RuntimeError(f"Distance-grid mismatch in {path}.")

        truth_radius = correlation_radius_per_episode(distances, truth_map)
        method_radius = correlation_radius_per_episode(distances, method_map)
        baseline_radius = correlation_radius_per_episode(distances, baseline_map)

        method_errors = per_episode_mae(truth_radius, method_radius)
        baseline_errors = per_episode_mae(truth_radius, baseline_radius)
        case_id = case_identifier(case)

        for index, seed in enumerate(np.asarray(case["seeds"], dtype=np.int64)):
            method_error = float(method_errors[index])
            baseline_error = float(baseline_errors[index])
            if not np.isfinite(method_error) or not np.isfinite(baseline_error):
                continue
            relative = (
                float((baseline_error - method_error) / baseline_error)
                if baseline_error > 0.0
                else None
            )
            rows.append(
                {
                    "case_id": case_id,
                    "prediction_file": path.name,
                    "experiment": str(case["experiment"]),
                    "L": int(case["L"]),
                    "test_W": float(case["test_W"]),
                    "t_obs": float(case["t_obs"]),
                    "seed": int(seed),
                    "n_forecast_times": int(np.asarray(case["times"]).size),
                    "method_radius_mae": method_error,
                    "baseline_radius_mae": baseline_error,
                    "method_minus_baseline_mae": method_error - baseline_error,
                    "relative_improvement": relative,
                }
            )

    if not rows:
        raise ValueError("No valid realizations matched the requested filters.")
    return rows


def hierarchical_bootstrap(
    rows: Sequence[dict],
    n_bootstrap: int,
    seed: int,
) -> dict:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    by_case: Dict[str, np.ndarray] = {}
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        values = np.asarray(
            [
                float(row["method_minus_baseline_mae"])
                for row in rows
                if row["case_id"] == case_id
            ],
            dtype=np.float64,
        )
        if values.size:
            by_case[case_id] = values

    case_ids = sorted(by_case)
    if not case_ids:
        raise ValueError("No cases were available for bootstrap analysis.")

    case_means = np.asarray(
        [float(np.mean(by_case[case_id])) for case_id in case_ids],
        dtype=np.float64,
    )
    point = float(np.mean(case_means))

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        selected_case_indices = rng.integers(0, len(case_ids), size=len(case_ids))
        selected_case_means = []
        for case_index in selected_case_indices:
            values = by_case[case_ids[int(case_index)]]
            episode_indices = rng.integers(0, values.size, size=values.size)
            selected_case_means.append(float(np.mean(values[episode_indices])))
        samples[draw] = float(np.mean(selected_case_means))

    return {
        "mean_case_balanced_difference": point,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_cases": int(len(case_ids)),
        "n_realizations": int(len(rows)),
        "bootstrap_draws": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }


def save_rows(rows: Sequence[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(destination)


def plot_scatter(
    rows: Sequence[dict],
    bootstrap: dict,
    component: str,
    method: str,
    baseline: str,
    destination: Path,
    dpi: int,
    log_scale: bool,
) -> None:
    x = np.asarray(
        [float(row["baseline_radius_mae"]) for row in rows], dtype=np.float64
    )
    y = np.asarray(
        [float(row["method_radius_mae"]) for row in rows], dtype=np.float64
    )
    disorders = np.asarray([float(row["test_W"]) for row in rows], dtype=np.float64)
    experiments = np.asarray([str(row["experiment"]) for row in rows])

    positive = np.concatenate([x[x > 0.0], y[y > 0.0]])
    if positive.size == 0:
        raise ValueError("All radius errors are zero; scatter limits are undefined.")

    upper = float(max(np.max(x), np.max(y)) * 1.06)
    lower = 0.0
    if log_scale:
        lower = float(np.min(positive) / 1.25)

    figure, axis = plt.subplots(figsize=(6.3, 5.4), constrained_layout=True)
    markers = {"within": "o", "transfer": "^"}
    disorder_min = float(np.min(disorders))
    disorder_max = float(np.max(disorders))
    if _close(disorder_min, disorder_max):
        disorder_max = disorder_min + 1.0
    norm = Normalize(vmin=disorder_min, vmax=disorder_max)
    cmap = plt.get_cmap("viridis")

    for experiment in sorted(set(experiments.tolist())):
        mask = experiments == experiment
        marker = markers.get(experiment, "s")
        axis.scatter(
            x[mask],
            y[mask],
            c=disorders[mask],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=36,
            alpha=0.78,
            edgecolors="none",
            label=experiment.capitalize(),
        )

    if log_scale:
        axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
    else:
        axis.plot([0.0, upper], [0.0, upper], linestyle="--", linewidth=1.2)
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(
        rf"{baseline.capitalize()} $\mathrm{{MAE}}_R[R_{{{component}}}(t)]$"
    )
    axis.set_ylabel(rf"{method.upper()} $\mathrm{{MAE}}_R[R_{{{component}}}(t)]$")
    axis.set_title("Paired realization-level correlation-radius error")
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left")

    if np.unique(disorders).size > 1:
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.02
        )
        colorbar.set_label("Test disorder $W$")

    fraction_better = float(np.mean(y < x))
    mean_difference = float(bootstrap["mean_case_balanced_difference"])
    ci_low = float(bootstrap["ci_low"])
    ci_high = float(bootstrap["ci_high"])
    annotation = (
        f"N={len(rows)} realizations, {bootstrap['n_cases']} cases\n"
        f"{method.upper()} better: {100.0 * fraction_better:.1f}%\n"
        f"case-balanced mean $\\Delta$MAE={mean_difference:+.3f}\n"
        f"95% hierarchical CI [{ci_low:+.3f}, {ci_high:+.3f}]"
    )
    axis.text(
        0.98,
        0.02,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired per-realization STP-versus-baseline errors for "
            "the correlation radius extracted from tensor forecasts."
        )
    )
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument(
        "--predictions-directory",
        default=None,
        help="Override RESULTS/predictions.",
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help="Default: RESULTS/physics_summary.",
    )
    parser.add_argument("--component", default=DEFAULT_COMPONENT)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--sizes", nargs="*", type=int, default=None)
    parser.add_argument("--disorders", nargs="*", type=float, default=None)
    parser.add_argument("--t-obs", nargs="*", type=float, default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--log-scale", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    component = arguments.component.lower()
    component_indices(component)

    results = Path(arguments.results)
    predictions_directory = (
        Path(arguments.predictions_directory)
        if arguments.predictions_directory is not None
        else results / "predictions"
    )
    output_directory = (
        Path(arguments.output_directory)
        if arguments.output_directory is not None
        else results / "physics_summary"
    )

    prediction_paths = sorted(predictions_directory.glob("*.npz"))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No prediction NPZ files were found in {predictions_directory}."
        )

    rows = build_rows(
        prediction_paths=prediction_paths,
        component=component,
        method=arguments.method,
        baseline=arguments.baseline,
        experiments=arguments.experiments,
        sizes=arguments.sizes,
        disorders=arguments.disorders,
        observation_times=arguments.t_obs,
    )
    bootstrap = hierarchical_bootstrap(
        rows=rows,
        n_bootstrap=arguments.bootstrap,
        seed=arguments.seed,
    )

    method_errors = np.asarray(
        [float(row["method_radius_mae"]) for row in rows], dtype=np.float64
    )
    baseline_errors = np.asarray(
        [float(row["baseline_radius_mae"]) for row in rows], dtype=np.float64
    )
    relative_values = np.asarray(
        [
            float(row["relative_improvement"])
            for row in rows
            if row["relative_improvement"] is not None
        ],
        dtype=np.float64,
    )

    summary = {
        "component": component,
        "method": arguments.method,
        "baseline": arguments.baseline,
        "n_realizations": int(len(rows)),
        "n_cases": int(bootstrap["n_cases"]),
        "fraction_method_better": float(np.mean(method_errors < baseline_errors)),
        "pooled_method_radius_mae": float(np.mean(method_errors)),
        "pooled_baseline_radius_mae": float(np.mean(baseline_errors)),
        "median_relative_improvement": (
            float(np.median(relative_values)) if relative_values.size else None
        ),
        **bootstrap,
    }

    tag = f"{component}_{arguments.method}_vs_{arguments.baseline}"
    scatter_path = output_directory / f"paired_radius_error_{tag}.png"
    rows_path = output_directory / f"paired_radius_error_{tag}.csv"
    summary_path = output_directory / f"paired_radius_error_{tag}.json"

    save_rows(rows, rows_path)
    save_json(summary, summary_path)
    plot_scatter(
        rows=rows,
        bootstrap=bootstrap,
        component=component,
        method=arguments.method,
        baseline=arguments.baseline,
        destination=scatter_path,
        dpi=arguments.dpi,
        log_scale=arguments.log_scale,
    )

    print(f"Saved: {scatter_path}")
    print(f"Saved: {rows_path}")
    print(f"Saved: {summary_path}")
    print(
        f"{arguments.method.upper()} better in "
        f"{100.0 * summary['fraction_method_better']:.1f}% of realizations."
    )
    print(
        "Case-balanced mean method-minus-baseline MAE: "
        f"{summary['mean_case_balanced_difference']:+.6f} "
        f"(95% CI {summary['ci_low']:+.6f}, {summary['ci_high']:+.6f})."
    )


if __name__ == "__main__":
    main()
