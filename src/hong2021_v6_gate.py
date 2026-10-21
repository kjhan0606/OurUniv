#!/usr/bin/env python
"""Combine V6 ensemble-field and paired grid-HOP statistical gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


HIGH_K_BANDS = ("3-6_h_mpc", "6-10.0531_h_mpc")
ENVIRONMENT_FIELDS = (
    "volume_fraction_rho_lt_0.1",
    "volume_fraction_rho_gt_10",
    "volume_fraction_rho_gt_100",
    "local_peak_count_rho_gt_10",
    "local_peak_count_rho_gt_100",
    "local_void_count_rho_lt_0.1",
)


def contains_one(interval: list[float]) -> bool:
    return interval[0] <= 1.0 <= interval[1]


def row_metrics(row: dict[str, Any], thresholds: list[float]) -> np.ndarray:
    masses = np.asarray(row["group_mass_m_sun_h"], dtype=np.float64)
    return np.asarray(
        [row["n_groups"]]
        + [np.count_nonzero(masses >= threshold) for threshold in thresholds]
        + [float(masses.max()) if len(masses) else 0.0],
        dtype=np.float64,
    )


def paired_hop_bootstrap(
    report: dict[str, Any], method: str, bootstrap: int, seed: int
) -> dict[str, Any]:
    rows = report["rows"]
    thresholds = [float(value) for value in report["mass_thresholds_m_sun_h"]]
    objects = sorted(
        row["object_index"] for row in rows if row["kind"] == "truth"
    )
    truth = np.asarray(
        [
            row_metrics(
                next(
                    row for row in rows
                    if row["kind"] == "truth" and row["object_index"] == obj
                ),
                thresholds,
            )
            for obj in objects
        ]
    )
    candidate = np.asarray(
        [
            np.mean(
                [
                    row_metrics(row, thresholds)
                    for row in rows
                    if row["kind"] == "generated"
                    and row["method"] == method
                    and row["object_index"] == obj
                ],
                axis=0,
            )
            for obj in objects
        ]
    )
    generator = np.random.default_rng(seed)
    index = generator.integers(0, len(objects), size=(bootstrap, len(objects)))
    names = ["all_groups"] + [
        f"mass_ge_{threshold:.0e}_m_sun_h" for threshold in thresholds
    ]
    metrics: dict[str, Any] = {}
    for column, name in enumerate(names):
        ratio = float(candidate[:, column].mean() / truth[:, column].mean())
        sampled = candidate[index, column].mean(axis=1) / truth[
            index, column
        ].mean(axis=1)
        interval = np.quantile(sampled, [0.025, 0.975]).tolist()
        metrics[name] = {
            "ratio_to_truth": ratio,
            "paired_environment_bootstrap_95": interval,
            "statistically_consistent_with_one": contains_one(interval),
        }
    point = float(np.median(candidate[:, -1]) / np.median(truth[:, -1]))
    sampled = np.median(candidate[index, -1], axis=1) / np.median(
        truth[index, -1], axis=1
    )
    interval = np.quantile(sampled, [0.025, 0.975]).tolist()
    metrics["maximum_group_mass_median"] = {
        "ratio_to_truth": point,
        "paired_environment_bootstrap_95": interval,
        "statistically_consistent_with_one": contains_one(interval),
    }
    return {
        "objects": len(objects),
        "members_per_object": report["members_per_environment"],
        "bootstrap_resamples": bootstrap,
        "seed": seed,
        "metrics": metrics,
        "pass": all(
            value["statistically_consistent_with_one"]
            for value in metrics.values()
        ),
    }


def field_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    power = metrics["fourier_log_density"]["generated_total_power_over_truth"]
    calibration = metrics["residual_calibration"]
    environment = metrics["environment"]
    two_point = metrics["two_point_cosmic_mean"]
    checks = {
        "high_k_total_power_within_10_percent": all(
            abs(power[band] - 1.0) <= 0.1 for band in HIGH_K_BANDS
        ),
        "residual_rms_within_10_percent": abs(
            calibration["generated_over_truth_rms"] - 1.0
        ) <= 0.1,
        "rank_histogram_tv_at_most_0.05": calibration[
            "rank_histogram_total_variation_from_uniform"
        ] <= 0.05,
        "finite_ensemble_coverage_within_0.03": max(
            abs(calibration["coverage_68_minus_finite_ensemble_expected"]),
            abs(calibration["coverage_95_minus_finite_ensemble_expected"]),
        ) <= 0.03,
        "density_pdf_improves_deterministic": metrics["density_pdf"][
            "log10_density_total_variation_generated_truth"
        ]
        < metrics["density_pdf"][
            "log10_density_total_variation_deterministic_truth"
        ],
        "two_point_improves_deterministic_all_scales": all(
            two_point["generated_vs_truth_ks"]["by_scale"][scale]["mean"]
            < two_point["deterministic_vs_truth_ks"]["by_scale"][scale]["mean"]
            for scale in two_point["generated_vs_truth_ks"]["by_scale"]
        ),
        "all_selected_peak_void_statistics_improve_deterministic": all(
            abs(environment["generated"][field]["mean"]
                - environment["truth"][field]["mean"])
            < abs(environment["deterministic"][field]["mean"]
                  - environment["truth"][field]["mean"])
            for field in ENVIRONMENT_FIELDS
        ),
        "exact_dc_projection": metrics["low_k_preservation"][
            "maximum_abs_cube_mean_generated_residual"
        ] <= 1.0e-7,
    }
    return {"checks": checks, "pass": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-metrics", type=Path, required=True)
    parser.add_argument("--hop", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    ensemble = json.loads(args.ensemble_metrics.read_text())
    hop = json.loads(args.hop.read_text())
    candidates: dict[str, Any] = {}
    for method, metrics in ensemble["candidates"].items():
        field = field_gate(metrics)
        hop_result = paired_hop_bootstrap(hop, method, args.bootstrap, args.seed)
        candidates[method] = {
            "field_gate": field,
            "grid_hop_gate": hop_result,
            "overall_pass": field["pass"] and hop_result["pass"],
        }
    accepted = [
        method for method, value in candidates.items() if value["overall_pass"]
    ]
    report = {
        "schema": "hong2021-v6-field-and-grid-hop-decision-v1",
        "ensemble_metrics": str(args.ensemble_metrics.resolve()),
        "hop_report": str(args.hop.resolve()),
        "candidates": candidates,
        "accepted_methods": accepted,
        "decision": (
            "Advance accepted methods to the independent-simulation/forward "
            "dynamics gate; grid-HOP is not a bound N-body halo catalog."
        ),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
