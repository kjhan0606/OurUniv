#!/usr/bin/env python
"""Evaluate the frozen single-model N192 BGc validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cf4_bgc_cr_eval import load_model, numerical_pass, observer_density


def in_range(value: float, bounds: list[float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not config.get("frozen_before_validation_run"):
        raise RuntimeError("N192 BGc validation config was not frozen")
    model = load_model(args.manifest)
    gates = config["acceptance"]
    numerical, numerical_values = numerical_pass(
        model,
        {
            "adjoint_relative_error_max": gates["adjoint_relative_error_max"],
            "cg_relative_residual_max": gates["cg_relative_residual_max"],
        },
    )
    diag = model["manifest"]["heldout"]
    calibrated = (
        abs(diag["z_mean"]) <= gates["abs_z_mean_max"]
        and in_range(diag["z_std"], gates["z_std_range"])
        and in_range(diag["coverage_1sigma"], gates["coverage_1sigma_range"])
        and in_range(diag["coverage_2sigma"], gates["coverage_2sigma_range"])
        and diag["delta_log_score"] > gates["delta_log_score_min"]
    )
    sheet = gates["external_local_sheet_gate"]
    obs = observer_density(model, sheet["gaussian_radius_mpc_h"])
    sheet_pass = obs["posterior_mean"] >= sheet["posterior_mean_delta_min"]
    passed = bool(numerical and calibrated and sheet_pass)
    result = {
        "schema": "cf4-v3-bgc-n192-validation-result-v1",
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "numerical": numerical_values,
        "numerical_pass": bool(numerical),
        "heldout": diag,
        "calibration_pass": bool(calibrated),
        "observer_density": obs,
        "external_local_sheet_gate_pass": bool(sheet_pass),
        "verdict": (
            "AUTHORIZE_N192_ALL_DATA_BGC_ENSEMBLE"
            if passed
            else "DO_NOT_GENERATE_N192_ALL_DATA_BGC_ENSEMBLE"
        ),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / "v3_bgc_n192_validation_result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
