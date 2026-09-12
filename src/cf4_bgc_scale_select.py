#!/usr/bin/env python
"""Select the preregistered BGc error scale on the development split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not config.get("frozen_before_tuning"):
        raise RuntimeError("error-scale tuning was not frozen")

    candidates = set(config["candidate_error_scales"])
    rows = []
    for path in args.manifest:
        manifest = json.loads(path.read_text())
        scale = float(manifest["configuration"]["error_scale"])
        if scale not in candidates:
            raise RuntimeError(f"undeclared scale {scale} in {path}")
        diag = manifest["heldout"]
        residuals = [manifest["mean_cg_relative_residual"]] + [
            sample["cg_rel"] for sample in manifest["samples"]
        ]
        eligible = (
            manifest["adjoint_relative_error"] <= 5e-5
            and max(residuals) <= 1e-4
            and diag["delta_log_score"] > 0
        )
        objective = (
            abs(diag["z_mean"])
            + abs(diag["z_std"] - 1.0)
            + abs(diag["coverage_1sigma"] - 0.682689492)
            + abs(diag["coverage_2sigma"] - 0.954499736)
        )
        rows.append(
            {
                "error_scale": scale,
                "manifest": str(path.resolve()),
                "eligible": bool(eligible),
                "objective": float(objective),
                "heldout": diag,
            }
        )
    if {row["error_scale"] for row in rows} != candidates:
        raise RuntimeError("not every preregistered candidate has a manifest")
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError("no eligible error-scale candidate")
    selected = min(eligible_rows, key=lambda row: (row["objective"], -row["error_scale"]))
    result = {
        "schema": "cf4-v3-bgc-error-scale-selection-v1",
        "config": str(args.config.resolve()),
        "candidates": sorted(rows, key=lambda row: row["error_scale"]),
        "selected_error_scale": selected["error_scale"],
        "selected_objective": selected["objective"],
        "fresh_validation": config["fresh_validation_reserved_before_tuning"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
