#!/usr/bin/env python
"""Evaluate the preregistered N64 control/hybrid local-likelihood comparison.

The two models do not contain identical rows, so their aggregate predictive
scores are not directly comparable.  This evaluator uses the intersection of
their WF15 held-out raw catalog rows and separately reports the hybrid's
local-direct held-out calibration.  It also measures the unconstrained
Gaussian-smoothed linear density at the observer for matched CR seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cf4_linear_cr import (  # noqa: E402
    build_forward,
    build_observer_density_probe,
    posterior_predictive,
    prepare_catalog,
)


def load_model(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    namespace = argparse.Namespace(**manifest["configuration"])
    data = prepare_catalog(namespace)
    outputs = []
    sample_q = []
    mean_s = None
    for path in manifest["outputs"]:
        with np.load(path) as item:
            if mean_s is None:
                mean_s = item["s_map"].astype(np.float32)
            outputs.append(item["s_out"].astype(np.float32))
            sample_q.append(item["nuisance_q"].astype(np.float64))
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "args": namespace,
        "data": data,
        "mean_s": mean_s,
        "mean_q": np.asarray(manifest["mean_nuisance_q"], dtype=np.float64),
        "samples": outputs,
        "sample_q": sample_q,
    }


def common_wf_diagnostic(control: dict, hybrid: dict) -> tuple[dict, dict, int]:
    cdata, hdata = control["data"], hybrid["data"]
    craw = cdata["raw_idx"][
        cdata["holdout"] & (cdata["likelihood_kind"] == 0)
    ]
    hraw = hdata["raw_idx"][
        hdata["holdout"] & (hdata["likelihood_kind"] == 0)
    ]
    common = np.intersect1d(craw, hraw)
    results = []
    for model in (control, hybrid):
        data = model["data"]
        select = (
            data["holdout"]
            & (data["likelihood_kind"] == 0)
            & np.isin(data["raw_idx"], common)
        )
        A, _, _, _ = build_forward(
            data["pos"][select], data["rhat"][select], model["args"]
        )
        results.append(
            posterior_predictive(
                A,
                data,
                select,
                model["mean_s"],
                model["mean_q"],
                model["samples"],
                model["sample_q"],
            )
        )
    return results[0], results[1], int(len(common))


def observer_density(model: dict, radius: float) -> dict:
    model["args"].observer_delta_radius = radius
    probe, _ = build_observer_density_probe(model["args"])
    mean = float(probe(model["mean_s"])[0])
    samples = [float(probe(field)[0]) for field in model["samples"]]
    return {
        "gaussian_radius_mpc_h": radius,
        "posterior_mean": mean,
        "sample_values": samples,
        "sample_mean": float(np.mean(samples)),
        "sample_std": float(np.std(samples, ddof=1)),
    }


def in_range(value: float, bounds: list[float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def markdown(result: dict) -> str:
    c = result["control"]
    h = result["hybrid"]
    lc = result["hybrid_local_direct"]
    od = result["observer_density"]
    return "\n".join(
        [
            "# V2 local-likelihood N64 development result",
            "",
            f"- Verdict: **{result['verdict']}**",
            f"- Common WF15 holdout rows: {result['common_wf_holdout_n']}",
            f"- Common-WF log predictive change (hybrid-control): "
            f"`{result['common_wf_logp_change_hybrid_minus_control']:+.2f}`",
            f"- Hybrid local-direct holdout: `n={lc['n']}`, "
            f"`z={lc['z_mean']:+.3f} +/- {lc['z_std']:.3f}`, "
            f"`cov68={lc['coverage_1sigma']:.3f}`, "
            f"`cov95={lc['coverage_2sigma']:.3f}`",
            "",
            "| model | z mean | z std | cov68 | cov95 | common-WF logp |",
            "|---|---:|---:|---:|---:|---:|",
            f"| control | {c['z_mean']:+.3f} | {c['z_std']:.3f} | "
            f"{c['coverage_1sigma']:.3f} | {c['coverage_2sigma']:.3f} | "
            f"{c['log_predictive_density']:.1f} |",
            f"| hybrid | {h['z_mean']:+.3f} | {h['z_std']:.3f} | "
            f"{h['coverage_1sigma']:.3f} | {h['coverage_2sigma']:.3f} | "
            f"{h['log_predictive_density']:.1f} |",
            "",
            "Matched-seed observer linear density, Gaussian R=5 Mpc/h:",
            "",
            f"- control: `{od['control']['sample_values']}`",
            f"- hybrid: `{od['hybrid']['sample_values']}`",
            f"- hybrid-control: `{od['paired_difference']}`",
            "",
            "The aggregate delta-log scores in the sampler manifests are not",
            "compared because the two models contain different catalog rows.",
            "The direct local component is over-dispersed and biased on its own",
            "held-out rows, so this N64 model is not promoted to N192.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control",
        type=Path,
        default=ROOT / "recon/linear_cr/manifest_v2local_control_n64.json",
    )
    parser.add_argument(
        "--hybrid",
        type=Path,
        default=ROOT / "recon/linear_cr/manifest_v2local_hybrid15_n64.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/v2_local_likelihood_dev_v1.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "recon/linear_cr/v2_local_dev_v1",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not config.get("frozen_before_comparison"):
        raise RuntimeError("development comparison was not frozen")

    control = load_model(args.control)
    hybrid = load_model(args.hybrid)
    cdiag, hdiag, common_n = common_wf_diagnostic(control, hybrid)
    radius = config["diagnostics"][
        "observer_linear_density_gaussian_radius_mpc_h"
    ]
    cobs = observer_density(control, radius)
    hobs = observer_density(hybrid, radius)
    paired = (
        np.asarray(hobs["sample_values"]) - np.asarray(cobs["sample_values"])
    )
    local = hybrid["manifest"]["heldout_by_likelihood"]["local_direct"]
    accept = config["acceptance"]
    global_pass = all(
        (
            abs(hdiag["z_mean"]) <= accept["global_abs_z_mean_max"],
            in_range(hdiag["z_std"], accept["global_z_std_range"]),
            in_range(
                hdiag["coverage_1sigma"],
                accept["global_coverage_1sigma_range"],
            ),
            in_range(
                hdiag["coverage_2sigma"],
                accept["global_coverage_2sigma_range"],
            ),
        )
    )
    # With n>=30 the local component must at least satisfy the same broad
    # calibration envelope before it can be promoted to an expensive N192 run.
    local_calibrated = (
        local["n"] >= 30
        and abs(local["z_mean"]) <= accept["global_abs_z_mean_max"]
        and in_range(local["z_std"], accept["global_z_std_range"])
        and in_range(
            local["coverage_1sigma"], accept["global_coverage_1sigma_range"]
        )
        and in_range(
            local["coverage_2sigma"], accept["global_coverage_2sigma_range"]
        )
    )
    result = {
        "schema": "cf4-v2-local-likelihood-development-result-v1",
        "config": str(args.config.resolve()),
        "control_manifest": str(args.control.resolve()),
        "hybrid_manifest": str(args.hybrid.resolve()),
        "common_wf_holdout_n": common_n,
        "control": cdiag,
        "hybrid": hdiag,
        "common_wf_logp_change_hybrid_minus_control": (
            hdiag["log_predictive_density"] - cdiag["log_predictive_density"]
        ),
        "hybrid_local_direct": local,
        "observer_density": {
            "control": cobs,
            "hybrid": hobs,
            "paired_difference": paired.tolist(),
            "paired_mean_difference": float(paired.mean()),
        },
        "global_common_wf_pass": bool(global_pass),
        "local_direct_calibrated": bool(local_calibrated),
        "verdict": (
            "PROMOTE_TO_N192"
            if global_pass and local_calibrated
            else "DO_NOT_PROMOTE"
        ),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "v2_local_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.outdir / "V2_LOCAL_REPORT.md").write_text(markdown(result))
    print(json.dumps(result, indent=2))
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
