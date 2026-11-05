#!/usr/bin/env python
"""Seal used and reserved EAGLE cubes before any further model development.

The first 16 EAGLE cubes have already been inspected by the frozen V6 gate.
This utility records those exact objects and cryptographic hashes of every
material input/output.  All remaining cubes are reserved for a one-time
confirmation test and must not be used for training, normalization, model
selection, or threshold tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA = "hong2021-eagle-confirmation-seal-v1"


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_seal(
    data_path: Path,
    representative_path: Path,
    ensemble_path: Path,
    metrics_path: Path,
    decision_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    representative = json.loads(representative_path.read_text())
    used = np.asarray(representative["indices"], dtype=np.int64)
    listed_ids = np.asarray(representative["galaxy_ids"], dtype=np.int64)
    if used.ndim != 1 or len(used) == 0 or len(np.unique(used)) != len(used):
        raise ValueError("representative indices must be a non-empty unique vector")

    with h5py.File(data_path, "r") as handle:
        galaxy_ids = np.asarray(handle["center_galaxy_id"], dtype=np.int64)
        positions = np.asarray(handle["center_position_mpc_h"], dtype=np.float64)
        schema = str(handle.attrs["schema"])
        independent = bool(handle.attrs["independent_test_only"])
    if not independent:
        raise ValueError("EAGLE data are not marked independent-test-only")
    if np.any(used < 0) or np.any(used >= len(galaxy_ids)):
        raise IndexError("representative index outside EAGLE data")
    if not np.array_equal(galaxy_ids[used], listed_ids):
        raise ValueError("representative GalaxyIDs do not match the prepared data")

    reserved = np.setdiff1d(
        np.arange(len(galaxy_ids), dtype=np.int64), used, assume_unique=True
    )
    files = {
        "prepared_data": file_record(data_path),
        "representative_selection": file_record(representative_path),
        "v6_ensemble": file_record(ensemble_path),
        "v6_metrics": file_record(metrics_path),
        "v6_decision": file_record(decision_path),
        "v6_checkpoint": file_record(checkpoint_path),
    }
    return {
        "schema": SCHEMA,
        "status": "sealed_before_v8_development",
        "prepared_data_schema": schema,
        "policy": {
            "used_v6_test": (
                "These 16 cubes are permanently classified as inspected and "
                "may be used only for diagnosis, never model selection."
            ),
            "reserved_confirmation": (
                "These 156 cubes may not be used for training, normalization, "
                "feature standardization, architecture/hyperparameter choice, "
                "checkpoint choice, or gate-threshold tuning."
            ),
            "opening_rule": (
                "Run one frozen confirmation only after the unchanged TNG "
                "and SIMBA development gates pass and the historically "
                "inspected SIMBA CV0-15 stress test also passes."
            ),
            "failure_rule": (
                "A failed confirmation remains a failure; do not iterate on "
                "EAGLE truth. Return to TNG and development-only simulations."
            ),
        },
        "counts": {
            "all": int(len(galaxy_ids)),
            "used_v6_test": int(len(used)),
            "reserved_confirmation": int(len(reserved)),
        },
        "used_v6_test": {
            "indices": used.tolist(),
            "galaxy_ids": galaxy_ids[used].tolist(),
            "positions_mpc_h": positions[used].tolist(),
        },
        "reserved_confirmation": {
            "indices": reserved.tolist(),
            "galaxy_ids": galaxy_ids[reserved].tolist(),
            "positions_mpc_h": positions[reserved].tolist(),
        },
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/gpfs/kjhan/EAGLE/RefL0100N1504")
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "derived/hong2021_v1/eagle_ref100_z0_test.h5",
    )
    parser.add_argument(
        "--representative",
        type=Path,
        default=root / "derived/hong2021_v1/representative16_indices.json",
    )
    parser.add_argument(
        "--ensemble",
        type=Path,
        default=root
        / "evaluation/hong2021_v6_edm/edm_representative16_ensemble16.h5",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=root / "evaluation/hong2021_v6_edm/ensemble_evaluation/metrics.json",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=root / "evaluation/hong2021_v6_edm/independent_gate_decision.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/gpfs/kjhan/IllustrisTNG/TNG100-1/training/"
            "tng100_v6_edm_laplacian_sigma2/minimum_validation.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/hong2021_eagle_confirmation_seal_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seal = build_seal(
        args.data,
        args.representative,
        args.ensemble,
        args.metrics,
        args.decision,
        args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": seal["counts"]}, indent=2))


if __name__ == "__main__":
    main()
