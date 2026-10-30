#!/usr/bin/env python
"""Repair the one omitted evaluator metadata attribute in completed V31 ensembles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from hong2021_v31_copula import ENSEMBLE_SCHEMA, REGISTRY_SHA256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.paths:
        resolved = path.resolve()
        with h5py.File(resolved, "r+") as handle:
            if (
                handle.attrs.get("schema") != ENSEMBLE_SCHEMA
                or handle.attrs.get("v31_registry_sha256") != REGISTRY_SHA256
                or not bool(handle.attrs.get("complete", False))
                or tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64)
                or tuple(handle["truth"].shape) != (16, 1, 64, 64, 64)
            ):
                raise ValueError(f"not a complete frozen V31 ensemble: {resolved}")
            before = handle.attrs.get("diagnostic_k_h_mpc")
            if before is not None and float(before) != 1.0:
                raise ValueError(f"conflicting diagnostic k in {resolved}")
            sample_shape = list(handle["sample"].shape)
            truth_shape = list(handle["truth"].shape)
            handle.attrs["diagnostic_k_h_mpc"] = 1.0
            handle.flush()
            after = float(handle.attrs["diagnostic_k_h_mpc"])
        rows.append(
            {
                "path": str(resolved),
                "attribute_before": None if before is None else float(before),
                "attribute_after": after,
                "sample_shape_unchanged": sample_shape,
                "truth_shape_unchanged": truth_shape,
            }
        )
    report = {
        "schema": "hong2021-v31-evaluator-diagnostic-attribute-repair-v1",
        "scope": "metadata_only_no_dataset_write",
        "reason": "the frozen evaluator requires diagnostic_k_h_mpc=1.0",
        "ensembles": rows,
    }
    if args.report.exists():
        raise RuntimeError("V31 refuses to overwrite repair report")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
