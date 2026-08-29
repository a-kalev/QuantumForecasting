#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:47:28 2026

@author: amirk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication figures for the correlation-tensor forecasting paper.

Reads existing results only. No fitting and no ED regeneration.

Outputs
-------
paper_figures/
    fig_benchmark_scatter.pdf
    fig_benchmark_scatter.png
    fig_tensor_forecast.pdf
    fig_tensor_forecast.png
    fig_tensor_forecast_metadata.txt

Expected directory structure
----------------------------
correlation_tensor_forecast_v1_r40/
    aggregate_summary.csv
    predictions/*.npz
    dmd_benchmark/aggregate_summary.csv
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path("correlation_tensor_forecast_v1_r40")
DMD_RESULTS = RESULTS / "dmd_benchmark"
OUTPUT = RESULTS / "paper_figures"

TENSOR_SUMMARY = RESULTS / "aggregate_summary.csv"
DMD_SUMMARY = DMD_RESULTS / "aggregate_summary.csv"
PREDICTIONS = RESULTS / "predictions"

KEYS = (
    "experiment",
    "L",
    "t_obs",
    "forecast_lead",
    "tensor_region",
)


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows in {path}")

    return rows


def key(row):
    return (
        str(row["experiment"]),
        int(row["L"]),
        float(row["t_obs"]),
        float(row["forecast_lead"]),
        str(row["tensor_region"]),
    )


def method_map(rows, method):
    selected = {}

    for row in rows:
        if row["method"] != method:
            continue

        k = key(row)
        if k in selected:
            raise ValueError(f"Duplicate {method} row for {k}")

        selected[k] = float(row["rmse"])

    return selected


def scalar(array):
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError("Expected scalar.")
    result = value.reshape(()).item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return result


# ----------------------------------------------------------------------
# Figure 1: all benchmark settings
# ----------------------------------------------------------------------

def benchmark_scatter():
    base_rows = read_csv(TENSOR_SUMMARY)
    dmd_rows = read_csv(DMD_SUMMARY)

    stp_base = method_map(base_rows, "stp")
    ridge = method_map(base_rows, "ridge")

    # Use STP values contained in the same DMD benchmark output for
    # the DMD comparison, guaranteeing identical aggregation.
    stp_dmd = method_map(dmd_rows, "stp")
    dmd = method_map(dmd_rows, "dmd")

    ridge_keys = sorted(set(stp_base) & set(ridge))
    dmd_keys = sorted(set(stp_dmd) & set(dmd))

    if len(ridge_keys) != 80:
        raise RuntimeError(
            f"Expected 80 STP-ridge settings, found {len(ridge_keys)}."
        )

    if len(dmd_keys) != 80:
        raise RuntimeError(
            f"Expected 80 STP-DMD settings, found {len(dmd_keys)}."
        )

    ridge_wins = sum(stp_base[k] < ridge[k] for k in ridge_keys)
    dmd_wins = sum(stp_dmd[k] < dmd[k] for k in dmd_keys)

    print(f"STP lower RMSE than ridge: {ridge_wins}/{len(ridge_keys)}")
    print(f"STP lower RMSE than DMD:   {dmd_wins}/{len(dmd_keys)}")

    figure, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.7), constrained_layout=True
    )

    comparisons = (
        (axes[0], ridge_keys, stp_base, ridge, "Ridge"),
        (axes[1], dmd_keys, stp_dmd, dmd, "DMD"),
    )

    marker_for_experiment = {
        "within": "o",
        "transfer": "^",
    }

    leads = sorted(
        {
            k[3]
            for k in ridge_keys + dmd_keys
        }
    )
    lead_to_number = {
        lead: index
        for index, lead in enumerate(leads)
    }

    all_values = []

    for axis, keys_now, stp_map, competitor_map, competitor_name in comparisons:
        for experiment in ("within", "transfer"):
            subset = [
                k for k in keys_now
                if k[0] == experiment
            ]

            x = np.asarray(
                [competitor_map[k] for k in subset],
                dtype=float,
            )
            y = np.asarray(
                [stp_map[k] for k in subset],
                dtype=float,
            )
            c = np.asarray(
                [lead_to_number[k[3]] for k in subset],
                dtype=float,
            )

            all_values.extend(x.tolist())
            all_values.extend(y.tolist())

            scatter = axis.scatter(
                x,
                y,
                c=c,
                cmap="viridis",
                marker=marker_for_experiment[experiment],
                s=35,
                alpha=0.82,
                label=experiment.capitalize(),
            )

        axis.set_xlabel(f"{competitor_name} RMSE")
        axis.set_ylabel("STP RMSE")
        axis.set_title(f"STP vs {competitor_name}")
        axis.grid(alpha=0.2)

    minimum = min(all_values)
    maximum = max(all_values)
    padding = 0.04 * (maximum - minimum)
    lower = minimum - padding
    upper = maximum + padding

    for axis in axes:
        axis.plot(
            [lower, upper],
            [lower, upper],
            "--",
            linewidth=1.0,
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")

    axes[0].legend(frameon=False)

    colorbar = figure.colorbar(
        scatter,
        ax=axes,
        fraction=0.035,
        pad=0.02,
        ticks=np.arange(len(leads)),
    )
    colorbar.ax.set_yticklabels(
        [f"{lead:g}" for lead in leads]
    )
    colorbar.set_label(r"Forecast lead $\Delta t_{\rm f}$")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        OUTPUT / "fig_benchmark_scatter.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT / "fig_benchmark_scatter.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.close(figure)


# ----------------------------------------------------------------------
# Figure 2: temporal STP-vs-ridge forecasting advantage
# ----------------------------------------------------------------------

EXPECTED_EXPERIMENTS = ("within", "transfer")
EXPECTED_SIZES = (12, 14)
EXPECTED_T_OBS = (10.0, 20.0)
EXPECTED_DISORDERS = (2.0, 3.0, 4.0, 5.0)
BENCHMARK_LEADS = (5.0, 10.0, 20.0, 40.0, 80.0)


def load_temporal_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "experiment",
            "L",
            "test_W",
            "t_obs",
            "test_seeds",
            "target_times",
            "tensor_shape",
            "truth",
            "stp",
            "ridge",
        }

        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{path} missing {sorted(missing)}"
            )

        experiment = str(
            scalar(data["experiment"])
        )
        L = int(
            scalar(data["L"])
        )
        W = float(
            scalar(data["test_W"])
        )
        t_obs = float(
            scalar(data["t_obs"])
        )

        seeds = np.asarray(
            data["test_seeds"],
            dtype=np.int64,
        )
        times = np.asarray(
            data["target_times"],
            dtype=np.float64,
        )

        tensor_shape = tuple(
            int(value)
            for value in data["tensor_shape"]
        )

        if tensor_shape != (L, L, 3, 3):
            raise ValueError(
                f"{path}: tensor_shape={tensor_shape}, "
                f"expected {(L, L, 3, 3)}."
            )

        if (
            times.ndim != 1
            or times.size < 1
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                f"{path}: invalid target_times."
            )

        expected_shape = (
            seeds.size,
            times.size,
            9 * L * L,
        )

        truth = np.asarray(
            data["truth"],
            dtype=np.float64,
        )
        stp = np.asarray(
            data["stp"],
            dtype=np.float64,
        )
        ridge = np.asarray(
            data["ridge"],
            dtype=np.float64,
        )

        for name, values in (
            ("truth", truth),
            ("stp", stp),
            ("ridge", ridge),
        ):
            if values.shape != expected_shape:
                raise ValueError(
                    f"{path}: {name} shape "
                    f"{values.shape}, expected "
                    f"{expected_shape}."
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{path}: {name} contains "
                    "non-finite values."
                )

    return {
        "path": path,
        "experiment": experiment,
        "L": L,
        "W": W,
        "t_obs": t_obs,
        "seeds": seeds,
        "times": times,
        "truth": truth,
        "stp": stp,
        "ridge": ridge,
    }


def temporal_region_mask(L, region):
    """
    Identical tensor-region convention to the benchmark:
        all     -> every C_ij^{alpha beta}
        offsite -> i != j only
    """
    mask = np.ones(
        (L, L, 3, 3),
        dtype=bool,
    )

    if region == "offsite":
        site_mask = ~np.eye(
            L,
            dtype=bool,
        )

        mask &= site_mask[
            :,
            :,
            None,
            None,
        ]

    elif region != "all":
        raise ValueError(
            f"Unknown tensor region: {region}"
        )

    return mask.reshape(-1)


def discover_temporal_cases():
    paths = sorted(
        PREDICTIONS.glob("*.npz")
    )

    if not paths:
        raise FileNotFoundError(
            f"No prediction files found in "
            f"{PREDICTIONS}."
        )

    cases = [
        load_temporal_prediction(path)
        for path in paths
    ]

    index = {}

    for case in cases:
        key_now = (
            case["experiment"],
            case["L"],
            case["t_obs"],
            case["W"],
        )

        if key_now in index:
            raise RuntimeError(
                "Duplicate prediction case: "
                f"{key_now}"
            )

        index[key_now] = case

    # Explicitly require the complete benchmark used
    # in the paper.
    expected_keys = {
        (
            experiment,
            L,
            t_obs,
            W,
        )
        for experiment in EXPECTED_EXPERIMENTS
        for L in EXPECTED_SIZES
        for t_obs in EXPECTED_T_OBS
        for W in EXPECTED_DISORDERS
    }

    missing = sorted(
        expected_keys.difference(index)
    )

    if missing:
        raise RuntimeError(
            "Missing benchmark prediction cases: "
            f"{missing}"
        )

    return index


def cumulative_rmse_curve(
    cases,
    region,
):
    """
    Pool the four disorder values and all independent
    test realizations.

    At each forecast lead tau, compute RMSE using every
    prediction from the first forecast time through tau.

    This is the same cumulative-in-lead definition used
    for the aggregate benchmark rows.
    """
    reference = cases[0]
    L = reference["L"]
    times = reference["times"]
    t_obs = reference["t_obs"]

    feature_mask = temporal_region_mask(
        L,
        region,
    )

    n_features = int(
        np.count_nonzero(feature_mask)
    )

    stp_sse_per_time = np.zeros(
        times.size,
        dtype=np.float64,
    )
    ridge_sse_per_time = np.zeros(
        times.size,
        dtype=np.float64,
    )
    values_per_time = np.zeros(
        times.size,
        dtype=np.float64,
    )

    seen_seeds = set()

    for case in cases:
        if case["L"] != L:
            raise ValueError(
                "Cannot aggregate different L."
            )

        if not np.isclose(
            case["t_obs"],
            t_obs,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Cannot aggregate different t_obs."
            )

        if not np.array_equal(
            case["times"],
            times,
        ):
            raise ValueError(
                "Target-time grids differ across "
                "disorder values."
            )

        for seed in case["seeds"]:
            seed_int = int(seed)

            if seed_int in seen_seeds:
                raise ValueError(
                    "Repeated test seed within "
                    "aggregated benchmark group."
                )

            seen_seeds.add(seed_int)

        truth = case["truth"][
            :,
            :,
            feature_mask,
        ]

        stp = case["stp"][
            :,
            :,
            feature_mask,
        ]

        ridge = case["ridge"][
            :,
            :,
            feature_mask,
        ]

        stp_error = stp - truth
        ridge_error = ridge - truth

        stp_sse_per_time += np.sum(
            stp_error ** 2,
            axis=(0, 2),
        )

        ridge_sse_per_time += np.sum(
            ridge_error ** 2,
            axis=(0, 2),
        )

        values_per_time += (
            truth.shape[0]
            * n_features
        )

    if np.any(values_per_time <= 0.0):
        raise RuntimeError(
            "Invalid aggregation denominator."
        )

    cumulative_values = np.cumsum(
        values_per_time
    )

    stp_cumulative_rmse = np.sqrt(
        np.cumsum(stp_sse_per_time)
        / cumulative_values
    )

    ridge_cumulative_rmse = np.sqrt(
        np.cumsum(ridge_sse_per_time)
        / cumulative_values
    )

    if np.any(
        ridge_cumulative_rmse <= 0.0
    ):
        raise RuntimeError(
            "Zero ridge RMSE encountered."
        )

    ratio = (
        stp_cumulative_rmse
        / ridge_cumulative_rmse
    )

    forecast_leads = (
        times - t_obs
    )

    return {
        "leads": forecast_leads,
        "stp_rmse": stp_cumulative_rmse,
        "ridge_rmse": ridge_cumulative_rmse,
        "ratio": ratio,
        "n_realizations": len(seen_seeds),
        "n_features": n_features,
    }


def aggregate_summary_lookup():
    rows = read_csv(
        TENSOR_SUMMARY
    )

    result = {}

    for row in rows:
        method = str(row["method"])

        if method not in (
            "stp",
            "ridge",
        ):
            continue

        lookup_key = (
            str(row["experiment"]),
            int(row["L"]),
            float(row["t_obs"]),
            float(row["forecast_lead"]),
            str(row["tensor_region"]),
            method,
        )

        if lookup_key in result:
            raise RuntimeError(
                "Duplicate aggregate-summary row: "
                f"{lookup_key}"
            )

        result[lookup_key] = float(
            row["rmse"]
        )

    return result


def validate_temporal_curve(
    curve,
    summary_lookup,
    experiment,
    L,
    t_obs,
    region,
):
    """
    Verify that the values reconstructed from the saved
    prediction NPZ files agree with aggregate_summary.csv
    at the five benchmark leads.
    """
    for lead in BENCHMARK_LEADS:
        matches = np.flatnonzero(
            np.isclose(
                curve["leads"],
                lead,
                rtol=0.0,
                atol=1e-12,
            )
        )

        if matches.size != 1:
            raise RuntimeError(
                f"Could not uniquely locate "
                f"forecast lead {lead:g}."
            )

        index = int(matches[0])

        for method, curve_name in (
            ("stp", "stp_rmse"),
            ("ridge", "ridge_rmse"),
        ):
            lookup_key = (
                experiment,
                L,
                t_obs,
                lead,
                region,
                method,
            )

            if lookup_key not in summary_lookup:
                raise RuntimeError(
                    "Missing aggregate-summary row: "
                    f"{lookup_key}"
                )

            expected = summary_lookup[
                lookup_key
            ]

            observed = float(
                curve[curve_name][index]
            )

            # Prediction NPZ files are stored as float32,
            # whereas aggregate_summary.csv was generated
            # from the original float64 predictions.
            if not np.isclose(
                observed,
                expected,
                rtol=5e-5,
                atol=5e-7,
            ):
                raise RuntimeError(
                    "Temporal figure does not reproduce "
                    "aggregate benchmark RMSE:\n"
                    f"  key={lookup_key}\n"
                    f"  NPZ={observed:.12g}\n"
                    f"  summary={expected:.12g}"
                )


def temporal_performance_figure():
    case_index = (
        discover_temporal_cases()
    )

    summary_lookup = (
        aggregate_summary_lookup()
    )

    # Rows:
    #   complete tensor
    #   off-site tensor
    #
    # Columns:
    #   within-disorder
    #   cross-disorder transfer
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.6, 6.4),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    regions = (
        ("all", "Complete tensor"),
        ("offsite", "Off-site tensor"),
    )

    experiment_titles = {
        "within": "Within disorder",
        "transfer": "Cross-disorder transfer",
    }

    #all_ratios = []
    output_rows = []

    for column, experiment in enumerate(
        EXPECTED_EXPERIMENTS
    ):
        for row_index, (
            region,
            region_label,
        ) in enumerate(regions):

            axis = axes[
                row_index,
                column,
            ]
            panel_ratios = []

            for L in EXPECTED_SIZES:
                for t_obs in EXPECTED_T_OBS:

                    cases = [
                        case_index[
                            (
                                experiment,
                                L,
                                t_obs,
                                W,
                            )
                        ]
                        for W in EXPECTED_DISORDERS
                    ]

                    curve = cumulative_rmse_curve(
                        cases=cases,
                        region=region,
                    )

                    validate_temporal_curve(
                        curve=curve,
                        summary_lookup=summary_lookup,
                        experiment=experiment,
                        L=L,
                        t_obs=t_obs,
                        region=region,
                    )

                    label = (
                        rf"$L={L}$, "
                        rf"$t_{{\rm obs}}={t_obs:g}$"
                    )

                    axis.plot(
                        curve["leads"],
                        curve["ratio"],
                        linewidth=1.8,
                        label=label,
                    )

                    panel_ratios.extend(
                        curve["ratio"].tolist()
                    )

                    for (
                        lead,
                        stp_rmse,
                        ridge_rmse,
                        ratio,
                    ) in zip(
                        curve["leads"],
                        curve["stp_rmse"],
                        curve["ridge_rmse"],
                        curve["ratio"],
                    ):
                        output_rows.append(
                            {
                                "experiment": experiment,
                                "tensor_region": region,
                                "L": L,
                                "t_obs": t_obs,
                                "forecast_lead": float(
                                    lead
                                ),
                                "stp_cumulative_rmse": float(
                                    stp_rmse
                                ),
                                "ridge_cumulative_rmse": float(
                                    ridge_rmse
                                ),
                                "stp_over_ridge": float(
                                    ratio
                                ),
                                "n_realizations": int(
                                    curve[
                                        "n_realizations"
                                    ]
                                ),
                                "n_tensor_features": int(
                                    curve[
                                        "n_features"
                                    ]
                                ),
                            }
                        )

            # axis.axhline(
            #     1.0,
            #     linestyle="--",
            #     linewidth=1.0,
            # )
            
            # Tight, panel-specific y range with a small visual margin.
            panel_min = float(np.min(panel_ratios))
            panel_max = float(np.max(panel_ratios))
            
            panel_span = panel_max - panel_min
            
            padding = max(
                0.08 * panel_span,
                0.002,
            )
            
            axis.set_ylim(
                panel_min - padding,
                panel_max + padding,
            )
            
            # Explicitly show y tick labels on every panel.
            axis.tick_params(
                axis="y",
                labelleft=True,
            )

            axis.grid(
                alpha=0.2,
            )

            if row_index == 0:
                axis.set_title(
                    experiment_titles[
                        experiment
                    ]
                )

            if column == 0:
                axis.set_ylabel(
                    (
                        r"$\mathrm{RMSE}_{\rm STP}"
                        r"/\mathrm{RMSE}_{\rm ridge}$"
                        "\n"
                        + region_label
                    )
                )

            if row_index == 1:
                axis.set_xlabel(
                    r"Forecast lead "
                    r"$\Delta t_{\rm f}$"
                )

    # # Common vertical range keeps all four panels
    # # directly comparable.
    # minimum_ratio = min(
    #     min(all_ratios),
    #     1.0,
    # )
    # maximum_ratio = max(
    #     max(all_ratios),
    #     1.0,
    # )

    # span = max(
    #     maximum_ratio - minimum_ratio,
    #     0.01,
    # )

    # padding = max(
    #     0.01,
    #     0.08 * span,
    # )

    # for axis in axes.flat:
    #     axis.set_ylim(
    #         minimum_ratio - padding,
    #         maximum_ratio + padding,
    #     )

    handles, labels = (
        axes[0, 0]
        .get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT
        / "fig_temporal_rmse_ratio.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT
        / "fig_temporal_rmse_ratio.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Save every plotted numerical value.
    csv_path = (
        OUTPUT
        / "fig_temporal_rmse_ratio.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        fieldnames = [
            "experiment",
            "tensor_region",
            "L",
            "t_obs",
            "forecast_lead",
            "stp_cumulative_rmse",
            "ridge_cumulative_rmse",
            "stp_over_ridge",
            "n_realizations",
            "n_tensor_features",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    print(
        "\nFigure 2 saved:"
    )
    print(
        OUTPUT
        / "fig_temporal_rmse_ratio.pdf"
    )
    print(
        OUTPUT
        / "fig_temporal_rmse_ratio.png"
    )
    print(
        OUTPUT
        / "fig_temporal_rmse_ratio.csv"
    )
    print(
        "Verified against aggregate_summary.csv "
        "at leads 5, 10, 20, 40, and 80."
    )
    
# ----------------------------------------------------------------------

def main():
    benchmark_scatter()
    temporal_performance_figure()

    print(f"\nFigures saved to: {OUTPUT}")


if __name__ == "__main__":
    main()