#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path

from hong2021_v18_init import sha256_file
from hong2021_v21_conditional_affine import (
    PROFILE_SCHEMA, TRANSFORM_SCHEMA, fit_profile, fit_v21_transform,
)


BASE = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v14/model")
OUT = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v21/model")
PATHS = {
    "TNG100": BASE / "tng_train_standardized.h5",
    "SIMBA": BASE / "simba_train_standardized.h5",
    "Swift": BASE / "swift_eagle_train_standardized.h5",
}


def write_new(path: Path, value: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V21 fit artifact: {path}")
    partial.write_text(
        json.dumps(value, sort_keys=True, indent=2, default=lambda row: row.tolist()) + "\n"
    )
    os.replace(partial, path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = fit_profile(PATHS)
    if profile["schema"] != PROFILE_SCHEMA:
        raise RuntimeError("V21 profile schema mismatch")
    profile_path = OUT / "conditional_affine_profile.json"
    write_new(profile_path, profile)
    transform = fit_v21_transform(PATHS, profile)
    transform.update({
        "schema": TRANSFORM_SCHEMA,
        "fit_sources": ["TNG100 train", "SIMBA train", "Swift train"],
        "source_weights": [1/3, 1/3, 1/3],
        "train_only": True,
        "profile_sha256": sha256_file(profile_path),
    })
    transform_path = OUT / "gaussianization_v21.json"
    write_new(transform_path, transform)
    print(json.dumps({
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "transform": {"path": str(transform_path), "sha256": sha256_file(transform_path)},
        "mu": profile["mu"], "sigma": profile["sigma"],
    }, indent=2))


if __name__ == "__main__":
    main()
