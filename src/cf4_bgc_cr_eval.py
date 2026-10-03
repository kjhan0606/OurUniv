#!/usr/bin/env python
"""Evaluate the frozen N64 WF15 versus leakage-free BGc CR comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cf4_linear_cr import build_observer_density_probe, prepare_catalog  # noqa: E402


def load_model(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    args = argparse.Namespace(**manifest["configuration"])
    data = prepare_catalog(args)
    samples = []
    mean = None
    for output in manifest["outputs"]:
        with np.load(output) as item:
            if mean is None:
                mean = item["s_map"].astype(np.float32)
            samples.append(item["s_out"].astype(np.float32))
    return {
        "path": str(path.resolve()),
        "manifest": manifest,
        "args": args,
        "data": data,
        "mean": mean,
        "samples": samples,
    }


def in_range(value: float, bounds: list[float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def numerical_pass(model: dict, gates: dict) -> tuple[bool, dict]:
    manifest = model["manifest"]
    sample_residuals = [row["cg_rel"] for row in manifest["samples"]]
    values = {
        "adjoint_relative_error": manifest["adjoint_relative_error"],
        "mean_cg_relative_residual": manifest["mean_cg_relative_residual"],
        "max_sample_cg_relative_residual": max(sample_residuals),
    }
    passed = (
        values["adjoint_relative_error"] <= gates["adjoint_relative_error_max"]
        and values["mean_cg_relative_residual"] <= gates["cg_relative_residual_max"]
        and values["max_sample_cg_relative_residual"] <= gates["cg_relative_residual_max"]
    )
    return bool(passed), values


def calibration_pass(diag: dict, gates: dict) -> bool:
    return bool(
        abs(diag["z_mean"]) <= gates["each_model_abs_z_mean_max"]
        and in_range(diag["z_std"], gates["each_model_z_std_range"])
        and in_range(
            diag["coverage_1sigma"], gates["each_model_coverage_1sigma_range"]
        )
        and in_range(
            diag["coverage_2sigma"], gates["each_model_coverage_2sigma_range"]
        )
        and diag["delta_log_score"] > gates["each_model_delta_log_score_min"]
    )


def observer_density(model: dict, radius: float) -> dict:
    model["args"].observer_delta_radius = radius
    probe, _ = build_observer_density_probe(model["args"])
    values = np.asarray([float(probe(s)[0]) for s in model["samples"]])
    return {
        "gaussian_radius_mpc_h": radius,
        "posterior_mean": float(probe(model["mean"])[0]),
        "sample_values": values.tolist(),
        "sample_mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
    }


def render_markdown(result: dict) -> str:
    c = result["models"]["control"]
    b = result["models"]["bgc"]
    obs = result["observer_density"]
    return "\n".join(
        [
            "# V3 BGc N64 comparison",
            "",
            f"- Verdict: **{result['verdict']}**",
            f"- Common raw held-out rows: `{result['common_raw_holdout_n']}`",
            f"- Statistical BGc adoption: `{result['bgc_statistical_adoption']}`",
            f"- External Local-Sheet gate: `{result['external_local_sheet_gate_pass']}`",
            "",
            "| model | n | z mean | z std | cov68 | cov95 | delta log score | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            f"| WF15 control | {c['heldout']['n']} | {c['heldout']['z_mean']:+.3f} | "
            f"{c['heldout']['z_std']:.3f} | {c['heldout']['coverage_1sigma']:.3f} | "
            f"{c['heldout']['coverage_2sigma']:.3f} | {c['heldout']['delta_log_score']:+.1f} | "
            f"{c['all_gates_pass']} |",
            f"| BGc | {b['heldout']['n']} | {b['heldout']['z_mean']:+.3f} | "
            f"{b['heldout']['z_std']:.3f} | {b['heldout']['coverage_1sigma']:.3f} | "
            f"{b['heldout']['coverage_2sigma']:.3f} | {b['heldout']['delta_log_score']:+.1f} | "
            f"{b['all_gates_pass']} |",
            "",
            "Observer linear density (Gaussian R=5 Mpc/h):",
            "",
            f"- WF15 posterior mean: `{obs['control']['posterior_mean']:+.4f}`",
            f"- BGc posterior mean: `{obs['bgc']['posterior_mean']:+.4f}`",
            f"- BGc - WF15 paired sample mean: `{obs['paired_mean_difference']:+.4f}`",
            "",
            "Cross-estimator log densities are intentionally not subtracted: BGc and WF15",
            "define different transformed observables and measurement variances.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--bgc", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not (
        config.get("frozen_before_comparison")
        or config.get("frozen_before_validation_run")
    ):
        raise RuntimeError("BGc comparison or validation config was not frozen")

    control = load_model(args.control)
    bgc = load_model(args.bgc)
    gates = config["acceptance"]
    models = {}
    for name, model in (("control", control), ("bgc", bgc)):
        numerical, numerical_values = numerical_pass(model, gates)
        heldout = model["manifest"]["heldout"]
        calibrated = calibration_pass(heldout, gates)
        models[name] = {
            "manifest": model["path"],
            "numerical": numerical_values,
            "numerical_pass": numerical,
            "heldout": heldout,
            "calibration_pass": calibrated,
            "all_gates_pass": bool(numerical and calibrated),
        }

    c_raw = control["data"]["raw_idx"][control["data"]["holdout"]]
    b_raw = bgc["data"]["raw_idx"][bgc["data"]["holdout"]]
    common_n = int(np.intersect1d(c_raw, b_raw).size)
    common_pass = common_n >= gates["minimum_common_raw_holdout_rows"]

    sheet_gate = gates["external_local_sheet_gate"]
    c_obs = observer_density(control, sheet_gate["gaussian_radius_mpc_h"])
    b_obs = observer_density(bgc, sheet_gate["gaussian_radius_mpc_h"])
    paired = np.asarray(b_obs["sample_values"]) - np.asarray(c_obs["sample_values"])
    sheet_pass = b_obs["posterior_mean"] >= sheet_gate["bgc_posterior_mean_delta_min"]
    statistical = (
        models["control"]["all_gates_pass"]
        and models["bgc"]["all_gates_pass"]
        and common_pass
    )
    if statistical and sheet_pass:
        verdict = "AUTHORIZE_N192_BGC_THEN_P2"
    elif statistical:
        verdict = "ADOPT_BGC_DISTANT_DEVELOP_HIERARCHICAL_NEARBY"
    else:
        verdict = "DO_NOT_ADOPT_BGC_REAL_CATALOG_MODEL"

    result = {
        "schema": "cf4-v3-bgc-n64-comparison-result-v1",
        "config": str(args.config.resolve()),
        "models": models,
        "common_raw_holdout_n": common_n,
        "common_raw_holdout_gate_pass": bool(common_pass),
        "observer_density": {
            "control": c_obs,
            "bgc": b_obs,
            "paired_sample_difference": paired.tolist(),
            "paired_mean_difference": float(paired.mean()),
        },
        "bgc_statistical_adoption": bool(statistical),
        "external_local_sheet_gate_pass": bool(sheet_pass),
        "verdict": verdict,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    result_path = args.outdir / "v3_bgc_n64_comparison_result.json"
    report_path = args.outdir / "V3_BGC_N64_REPORT.md"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path.write_text(render_markdown(result))
    print(json.dumps(result, indent=2))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
