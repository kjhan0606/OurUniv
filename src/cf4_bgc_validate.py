#!/usr/bin/env python
"""Preregistered mock calibration and validation of the CF4 BGc transform."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cf4_bgc import bgc_transform  # noqa: E402

LN10_5 = np.log(10.0) / 5.0


def mock_once(
    cz: np.ndarray,
    sigln: np.ndarray,
    *,
    h0: float,
    true_sigma: float,
    window: int,
    seed: int,
    radial_bins: int,
    cz_min: float,
    cz_max: float,
) -> dict:
    rng = np.random.default_rng(seed)
    vtrue = rng.normal(0.0, true_sigma, cz.size)
    dtrue = (cz - vtrue) / h0
    epsilon = rng.normal(0.0, sigln)
    dobs = dtrue * np.exp(epsilon)
    raw = cz - h0 * dobs
    result = bgc_transform(
        cz,
        dobs,
        sigln,
        h0=h0,
        window=window,
        cz_min=cz_min,
        cz_max=cz_max,
    )
    use = result.corrected
    residual = result.velocity[use] - vtrue[use]
    z = residual / result.sigma_velocity[use]
    raw_residual = raw[use] - vtrue[use]
    cz_eval = cz[use]
    sigma_eval = result.sigma_velocity[use]
    edges = np.quantile(cz_eval, np.linspace(0.0, 1.0, radial_bins + 1))
    radial = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (cz_eval >= lo) & (
            cz_eval <= hi if i == radial_bins - 1 else cz_eval < hi
        )
        # Bias of the bin mean relative to the standard error expected from
        # independent measurement errors, expressed in sigma units.
        se = np.sqrt(np.sum(sigma_eval[mask] ** 2)) / mask.sum()
        radial.append(float(np.mean(residual[mask]) / se))
    return {
        "seed": int(seed),
        "n": int(use.sum()),
        "raw_mean_bias_km_s": float(np.mean(raw_residual)),
        "bgc_mean_bias_km_s": float(np.mean(residual)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "coverage_1sigma": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_2sigma": float(np.mean(np.abs(z) <= 2.0)),
        "max_abs_radial_bias_in_sigma": float(np.max(np.abs(radial))),
        "radial_bias_in_sigma": radial,
    }


def aggregate(rows: list[dict]) -> dict:
    keys = (
        "raw_mean_bias_km_s",
        "bgc_mean_bias_km_s",
        "z_mean",
        "z_std",
        "coverage_1sigma",
        "coverage_2sigma",
        "max_abs_radial_bias_in_sigma",
    )
    out = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std_across_seeds": float(np.std([row[key] for row in rows], ddof=1)),
        }
        for key in keys
    }
    all_radial = np.asarray([row["radial_bias_in_sigma"] for row in rows])
    out["ensemble_radial_bias_in_sigma"] = np.mean(all_radial, axis=0).tolist()
    out["ensemble_max_abs_radial_bias_in_sigma"] = float(
        np.max(np.abs(np.mean(all_radial, axis=0)))
    )
    out["pooled_z_mean"] = float(np.mean([row["z_mean"] for row in rows]))
    # All mocks have the same number of rows, so pooled moments are direct.
    second = np.mean(
        [row["z_std"] ** 2 + row["z_mean"] ** 2 for row in rows]
    )
    out["pooled_z_std"] = float(np.sqrt(max(second - out["pooled_z_mean"] ** 2, 0.0)))
    return out


def markdown(result: dict) -> str:
    val = result["validation"]["aggregate"]
    return "\n".join(
        [
            "# V3 BGc mock-validation result",
            "",
            f"- Verdict: **{result['verdict']}**",
            f"- Selected fixed-count redshift window: "
            f"`{result['selected_window']}` grouped CF4 rows",
            f"- Validation mocks: `{result['validation']['n_seeds']}` seeds, "
            f"`{result['n_catalog_rows']}` rows per seed",
            f"- Pooled normalized residual: "
            f"`{val['pooled_z_mean']:+.4f} +/- {val['pooled_z_std']:.4f}`",
            f"- Ensemble maximum radial-bin bias: "
            f"`{val['ensemble_max_abs_radial_bias_in_sigma']:.4f} sigma`",
            f"- Mean raw velocity bias: "
            f"`{val['raw_mean_bias_km_s']['mean']:+.1f} km/s`",
            f"- Mean BGc velocity bias: "
            f"`{val['bgc_mean_bias_km_s']['mean']:+.1f} km/s`",
            "",
            "This validates the one-point lognormal Gaussianization only in the",
            "preregistered `1500 <= cz <= 18000 km/s` range. It does not",
            "validate the Local Group/very-nearby likelihood below that range.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/v3_bgc_mock_validation_v1.json",
    )
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "data/cf4_clean.npz"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "recon/linear_cr/v3_bgc_mock_validation_v1",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not config.get("frozen_before_validation_run"):
        raise RuntimeError("BGc mock-validation configuration was not frozen")
    alg = config["algorithm"]
    mock = config["mock"]
    accept = config["acceptance"]
    with np.load(args.catalog) as cat:
        cz_all = cat["v3k"].astype(np.float64)
        sigln_all = np.maximum(cat["e_dm"].astype(np.float64) * LN10_5, 0.02)
    pool_min = mock.get("median_pool_cz_min_km_s", 500.0)
    pool_max = mock.get("median_pool_cz_max_km_s", 30000.0)
    keep = (
        np.isfinite(cz_all)
        & np.isfinite(sigln_all)
        & (sigln_all > 0)
        & (cz_all >= pool_min)
        & (cz_all <= pool_max)
    )
    cz = cz_all[keep]
    sigln = sigln_all[keep]

    development = {}
    for window in alg["candidate_redshift_neighbour_windows"]:
        rows = [
            mock_once(
                cz,
                sigln,
                h0=alg["h0_km_s_mpc"],
                true_sigma=mock["true_velocity_sigma_km_s"],
                window=window,
                seed=seed,
                radial_bins=mock["radial_bins"],
                cz_min=alg["cz_min_km_s"],
                cz_max=alg["cz_max_km_s"],
            )
            for seed in mock["development_seeds"]
        ]
        agg = aggregate(rows)
        score = (
            abs(agg["pooled_z_mean"])
            + abs(agg["pooled_z_std"] - 1.0)
            + agg["ensemble_max_abs_radial_bias_in_sigma"]
        )
        development[str(window)] = {
            "score": float(score),
            "aggregate": agg,
            "per_seed": rows,
        }
        print(f"[development] window={window} score={score:.5f}", flush=True)

    # The negative window term implements the preregistered larger-window tie
    # break without affecting non-tied floating-point scores.
    selected = min(
        alg["candidate_redshift_neighbour_windows"],
        key=lambda w: (development[str(w)]["score"], -w),
    )
    validation_rows = [
        mock_once(
            cz,
            sigln,
            h0=alg["h0_km_s_mpc"],
            true_sigma=mock["true_velocity_sigma_km_s"],
            window=selected,
            seed=seed,
            radial_bins=mock["radial_bins"],
            cz_min=alg["cz_min_km_s"],
            cz_max=alg["cz_max_km_s"],
        )
        for seed in mock["validation_seeds"]
    ]
    val = aggregate(validation_rows)
    seed_pass_fraction = float(
        np.mean(
            [
                abs(row["z_mean"])
                <= accept["validation_each_seed_abs_z_mean_max"]
                for row in validation_rows
            ]
        )
    )
    raw_abs = abs(val["raw_mean_bias_km_s"]["mean"])
    bgc_abs = abs(val["bgc_mean_bias_km_s"]["mean"])
    improvement = 1.0 - bgc_abs / max(raw_abs, 1e-30)
    gates = {
        "ensemble_z_mean": (
            abs(val["pooled_z_mean"])
            <= accept["validation_abs_ensemble_z_mean_max"]
        ),
        "ensemble_z_std": (
            accept["validation_ensemble_z_std_range"][0]
            <= val["pooled_z_std"]
            <= accept["validation_ensemble_z_std_range"][1]
        ),
        "radial_bias": (
            val["ensemble_max_abs_radial_bias_in_sigma"]
            <= accept["validation_max_abs_radial_bias_in_sigma"]
        ),
        "per_seed": seed_pass_fraction >= accept["required_seed_pass_fraction"],
        "bias_improvement": (
            improvement >= accept["raw_to_bgc_abs_mean_bias_improvement_min"]
        ),
    }
    result = {
        "schema": "cf4-v3-bgc-mock-validation-result-v1",
        "config": str(args.config),
        "catalog": str(args.catalog),
        "n_catalog_rows": int(cz.size),
        "selected_window": int(selected),
        "development": development,
        "validation": {
            "n_seeds": len(validation_rows),
            "aggregate": val,
            "per_seed": validation_rows,
            "seed_pass_fraction": seed_pass_fraction,
            "raw_to_bgc_abs_mean_bias_improvement": float(improvement),
        },
        "gates": gates,
        "verdict": "PASS_BGC_MOCK" if all(gates.values()) else "FAIL_BGC_MOCK",
        "scope": (
            "One-point BGc transform in the configured cz range only; "
            "nearby rows require a separate hierarchical likelihood."
        ),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "v3_bgc_mock_validation_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.outdir / "V3_BGC_MOCK_VALIDATION_REPORT.md").write_text(markdown(result))
    print(json.dumps({
        "selected_window": selected,
        "validation": val,
        "gates": gates,
        "verdict": result["verdict"],
    }, indent=2))
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
