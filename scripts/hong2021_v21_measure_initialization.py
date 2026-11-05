#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path

from hong2021_v18_init import measure_band_mode_variances, sha256_file


CACHE_SCHEMA = "hong2021-v21-conditional-affine-standardized-residual-cache-v1"
OUT = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v21/model")
SOURCES = {
    "TNG100": {"path": str(OUT / "tng100_train_conditional_affine.h5"), "objects": 432, "domain_attribute": "TNG100"},
    "SIMBA": {"path": str(OUT / "simba_train_conditional_affine.h5"), "objects": 202, "domain_attribute": "SIMBA"},
    "Swift": {"path": str(OUT / "swift_eagle_train_conditional_affine.h5"), "objects": 409, "domain_attribute": "Swift-EAGLE"},
}


def main() -> None:
    for row in SOURCES.values():
        row["sha256"] = sha256_file(row["path"])
    measured = measure_band_mode_variances(
        SOURCES,
        parseval_relative_tolerance=1.0e-12,
        maximum_absolute_ortho_dc=1.0e-9,
        allowed_schemas=(CACHE_SCHEMA,),
    )
    report = {
        "schema": "hong2021-v21-source-balanced-conditional-affine-band-variance-v1",
        "sources": SOURCES,
        **measured,
        "source_balanced_per_band_mode_variance": measured["source_balanced"],
    }
    destination = OUT / "initialization_variance_measurement.json"
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V21 measurement: {destination}")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, destination)
    print(json.dumps({"path": str(destination), "sha256": sha256_file(destination), **measured}, indent=2))


if __name__ == "__main__":
    main()
