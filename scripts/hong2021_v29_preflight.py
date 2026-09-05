#!/usr/bin/env python
"""Hard preflight for V29 direct physical residual transport."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from hong2021_v15_edm import git_state
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v29_physical import (
    PREFLIGHT_SCHEMA,
    REGISTRY_SHA256,
    centered_donor_residual,
    load_frozen_program,
    transport_residual,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V29 preflight requires a clean committed worktree")
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V29 preflight: {output}")
    registry, artifacts, v20 = load_frozen_program(args.registry.resolve(), repo)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    parent = Path(registry["frozen_v28_selections"]["tng"]["ensemble"])
    with h5py.File(parent, "r") as old:
        source = DOMAIN_ORDER[int(old["donor_source"][0, 0])]
        donor_index = int(old["donor_index"][0, 0])
        isometry = int(old["donor_isometry"][0, 0])
        query_index = int(old["source_index"][0])
    with h5py.File(experiment["data"][source]["train_data"]["path"], "r") as data, h5py.File(
        artifacts["caches"][f"{source}_train"]["path"], "r"
    ) as cache:
        residual = centered_donor_residual(
            np.asarray(data["target"][donor_index], dtype=np.float32),
            np.asarray(cache["conditional_mean"][donor_index], dtype=np.float32),
            float(cache["predicted_residual_dc"][donor_index]),
        )
    tng_cache_path = artifacts["caches"]["TNG100_validation"]["path"]
    with h5py.File(tng_cache_path, "r") as cache:
        baseline = np.asarray(cache["conditional_mean"][query_index], dtype=np.float32)
        baseline += np.float32(cache["predicted_residual_dc"][query_index])
    sample = transport_residual(residual, baseline, isometry)
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "code_commit": commit,
        "worktree_clean": clean,
        "registry": str(args.registry.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "full_pytest_required_by_launcher": True,
        "real_donor": {
            "source": source,
            "index": donor_index,
            "isometry": isometry,
            "maximum_absolute_centered_residual_dc": float(
                np.max(np.abs(residual.mean(axis=(-3, -2, -1))))
            ),
            "sample_finite": bool(np.isfinite(sample).all()),
            "sample_minimum_y": float(sample.min()),
            "sample_maximum_y": float(sample.max()),
        },
        "donor_reselection": False,
        "validation_truth_used_for_sampling": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    if (
        report["real_donor"]["maximum_absolute_centered_residual_dc"] > 1.0e-7
        or not report["real_donor"]["sample_finite"]
    ):
        raise RuntimeError("V29 real-data preflight failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
