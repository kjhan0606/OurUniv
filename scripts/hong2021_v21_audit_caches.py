#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


OUT = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v21/model")
JOBS = (
    ("TNG100", "train", "tng100_train_conditional_affine.h5"),
    ("TNG100", "validation", "tng100_validation_conditional_affine.h5"),
    ("SIMBA", "train", "simba_train_conditional_affine.h5"),
    ("SIMBA", "validation", "simba_validation_conditional_affine.h5"),
    ("Swift", "train", "swift_eagle_train_conditional_affine.h5"),
    ("Swift", "validation", "swift_eagle_validation_conditional_affine.h5"),
)


def main() -> None:
    report = {}
    train_rms = []
    for domain, split, name in JOBS:
        path = OUT / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with h5py.File(path, "r") as handle:
            values = handle["standardized_residual"]
            square_sum = 0.0
            count = 0
            finite = True
            for row in values:
                array = np.asarray(row[0], dtype=np.float32)
                finite = finite and bool(np.isfinite(array).all())
                square_sum += float(np.square(array, dtype=np.float64).sum())
                count += array.size
            rms = math.sqrt(square_sum / count)
            if split == "train":
                train_rms.append(rms)
            report[f"{domain}_{split}"] = {
                "sha256": digest, "objects": len(values),
                "complete": bool(handle.attrs["complete"]), "finite": finite,
                "rms": rms,
                "max_dc": float(handle.attrs["maximum_postprojection_ortho_dc"]),
                "max_adjustment": float(handle.attrs["maximum_absolute_adjustment"]),
            }
    sigma = math.sqrt(sum(value * value for value in train_rms) / 3.0)
    report["sigma_data"] = sigma
    report["p_mean"] = math.log(0.6 * sigma)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
