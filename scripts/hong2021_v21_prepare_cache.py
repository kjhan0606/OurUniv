#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from hong2021_v21_conditional_affine import prepare_cache


BASE = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v14/model")
OUT = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v21/model")
JOBS = (
    ("TNG100", "train", BASE / "tng_train_standardized.h5", "tng100_train_conditional_affine.h5"),
    ("TNG100", "validation", BASE / "tng_validation_standardized.h5", "tng100_validation_conditional_affine.h5"),
    ("SIMBA", "train", BASE / "simba_train_standardized.h5", "simba_train_conditional_affine.h5"),
    ("SIMBA", "validation", BASE / "simba_validation_standardized.h5", "simba_validation_conditional_affine.h5"),
    ("Swift", "train", BASE / "swift_eagle_train_standardized.h5", "swift_eagle_train_conditional_affine.h5"),
    ("Swift", "validation", BASE / "swift_eagle_validation_standardized.h5", "swift_eagle_validation_conditional_affine.h5"),
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = json.loads((OUT / "conditional_affine_profile.json").read_text())
    transform = json.loads((OUT / "gaussianization_v21.json").read_text())
    for domain, split, source, name in JOBS:
        print(f"[v21-cache] {domain} {split}", flush=True)
        print(prepare_cache(source, OUT / name, profile, transform), flush=True)


if __name__ == "__main__":
    main()
