#!/usr/bin/env python3
"""Publish the Phase-A datum with globally integrated order-6 raw exposure.

The publisher revalidates the count arrays and row manifest preserved by the
V2 hard-gate failure, replaces only the under-resolved order-2 exposure with
the independently generated global order-6 exposure, and publishes no field
posterior or mock result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROGRAM_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-publisher-program-v3"
RESULT_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-result-v3"
MANIFEST_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-manifest-v3"
COMPLETE_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-complete-v3"
STATUS = "PASS_PHASE_A_ACTUAL_36635_COUNT_DATUM_ORDER6_RAW_EXPOSURE"
EXPECTED_FILES = {"datum.npz", "row_manifest.csv", "result.json", "manifest.json", "COMPLETE"}


class PublisherError(ValueError):
    """Fail-closed Phase-A publication error."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding["path"]))
    if not path.is_file():
        raise PublisherError(f"bound {label} is absent: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise PublisherError(f"bound {label} size changed")
    if sha256_file(path) != str(binding["sha256"]):
        raise PublisherError(f"bound {label} hash changed")
    return path


def load_program(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    program = json.loads(raw)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise PublisherError("unexpected Phase-A publisher program schema")
    authorization = program.get("authorization", {})
    if not authorization.get("Phase_A_datum_publication", False):
        raise PublisherError("Phase-A datum publication is not authorized")
    for forbidden in (
        "field_inference",
        "mock_seed_access",
        "Phase_B_or_later",
        "automatic_follow_on",
    ):
        if authorization.get(forbidden, True):
            raise PublisherError(f"program improperly authorizes {forbidden}")
    for label, binding in program["bindings"].items():
        _verify_binding(binding, label)
    return program, hashlib.sha256(raw).hexdigest()


def reconstruct_manifest_counts(
    path: str | Path, grid: int, expected_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    fields = [
        "recno",
        "population_index",
        "voxel_flat_index",
        "split",
        "redshift_space_radius_cMpc_h",
    ]
    recnos: set[int] = set()
    joint_all: list[int] = []
    joint_train: list[int] = []
    joint_holdout: list[int] = []
    voxel_count = grid**3
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise PublisherError("row manifest header changed")
        for row in reader:
            recno = int(row["recno"])
            population = int(row["population_index"])
            voxel = int(row["voxel_flat_index"])
            split = row["split"]
            radius = float(row["redshift_space_radius_cMpc_h"])
            if recno in recnos:
                raise PublisherError("row manifest recno is not unique")
            if population < 0 or population >= 6 or voxel < 0 or voxel >= voxel_count:
                raise PublisherError("row manifest population or voxel is invalid")
            if split not in {"train", "holdout"} or not np.isfinite(radius) or radius <= 0.0:
                raise PublisherError("row manifest split or radius is invalid")
            recnos.add(recno)
            joint = population * voxel_count + voxel
            joint_all.append(joint)
            (joint_holdout if split == "holdout" else joint_train).append(joint)
    if len(recnos) != expected_rows:
        raise PublisherError("row manifest retained count changed")

    def count(values: list[int]) -> np.ndarray:
        return np.bincount(values, minlength=6 * voxel_count).reshape((6, grid, grid, grid)).astype(np.int64)

    return count(joint_all), count(joint_train), count(joint_holdout), {
        "row_count": len(recnos),
        "unique_recno_count": len(recnos),
        "train_row_count": len(joint_train),
        "holdout_row_count": len(joint_holdout),
    }


def collect_publication(
    program: Mapping[str, Any], program_sha256: str, implementation_commit: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], bytes]:
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise PublisherError("implementation commit must be lowercase 40-hex")
    with np.load(program["bindings"]["failed_datum"]["path"], allow_pickle=False) as archive:
        counts_all = np.asarray(archive["counts_all"])
        counts_train = np.asarray(archive["counts_train"])
        counts_holdout = np.asarray(archive["counts_holdout"])
    with np.load(
        program["bindings"]["quadrature_arrays"]["path"], allow_pickle=False
    ) as archive:
        if set(archive.files) != {"raw_selection_order4", "raw_selection_order6"}:
            raise PublisherError("quadrature diagnostic NPZ key set changed")
        order4 = np.asarray(archive["raw_selection_order4"])
        exposure = np.asarray(archive["raw_selection_order6"])
    expected_shape = tuple(program["datum"]["required_shape"])
    arrays = {
        "counts_all": counts_all,
        "counts_train": counts_train,
        "counts_holdout": counts_holdout,
        "raw_selection_exposure": exposure,
    }
    manifest_all, manifest_train, manifest_holdout, row_summary = reconstruct_manifest_counts(
        program["bindings"]["failed_row_manifest"]["path"],
        int(program["datum"]["grid_N"]),
        int(program["frozen_subset"]["retained_count_exact"]),
    )
    expected_populations = np.asarray(
        program["frozen_subset"]["population_counts_exact"], dtype=np.int64
    )
    positive_count = counts_all > 0
    gates = {
        "all_arrays_shape_exact": all(value.shape == expected_shape for value in arrays.values()),
        "count_arrays_int64": all(
            arrays[name].dtype == np.int64
            for name in ("counts_all", "counts_train", "counts_holdout")
        ),
        "count_arrays_nonnegative": all(
            np.all(arrays[name] >= 0)
            for name in ("counts_all", "counts_train", "counts_holdout")
        ),
        "split_elementwise_exhaustive": np.array_equal(
            counts_all, counts_train + counts_holdout
        ),
        "population_counts_exact": np.array_equal(
            counts_all.reshape(6, -1).sum(axis=1), expected_populations
        ),
        "row_manifest_reconstructs_all_counts": np.array_equal(
            counts_all, manifest_all
        ),
        "row_manifest_reconstructs_train_counts": np.array_equal(
            counts_train, manifest_train
        ),
        "row_manifest_reconstructs_holdout_counts": np.array_equal(
            counts_holdout, manifest_holdout
        ),
        "order6_exposure_float64": exposure.dtype == np.float64,
        "order6_exposure_finite_unit_interval": bool(
            np.all(np.isfinite(exposure))
            and np.all(exposure >= 0.0)
            and np.all(exposure <= 1.0 + 2.0e-14)
        ),
        "no_positive_count_in_nonpositive_order6_exposure": not np.any(
            positive_count & (exposure <= 0.0)
        ),
        "order4_and_order6_global_arrays_distinct": not np.array_equal(order4, exposure),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise PublisherError(f"Phase-A publication gates failed: {failed}")
    quadrature_result = json.loads(
        Path(program["bindings"]["quadrature_result"]["path"]).read_bytes()
    )
    comparison = quadrature_result["comparisons"][
        "order4_reference_to_order6_candidate"
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "status": STATUS,
        "program_sha256": program_sha256,
        "implementation_commit": implementation_commit,
        "counts": {
            "retained_total": int(counts_all.sum()),
            "population_all": counts_all.reshape(6, -1).sum(axis=1).tolist(),
            "population_train": counts_train.reshape(6, -1).sum(axis=1).tolist(),
            "population_holdout": counts_holdout.reshape(6, -1).sum(axis=1).tolist(),
            "occupied_voxels_all": np.count_nonzero(
                counts_all, axis=(1, 2, 3)
            ).tolist(),
        },
        "row_manifest": row_summary,
        "selection": {
            "normalization": "raw_exposure_not_normalized_to_observed_counts",
            "quadrature_order_per_axis": 6,
            "quadrature_subpoints_per_voxel": 216,
            "minimum": float(np.min(exposure)),
            "maximum": float(np.max(exposure)),
            "support_sum": exposure.reshape(6, -1).sum(axis=1).tolist(),
            "positive_voxel_fraction": np.mean(
                exposure.reshape(6, -1) > 0.0, axis=1
            ).tolist(),
            "positive_count_nonpositive_exposure_count": int(
                np.count_nonzero(positive_count & (exposure <= 0.0))
            ),
            "order4_to_order6_relative_L1_by_population": comparison[
                "relative_L1_by_population"
            ],
            "order4_to_order6_relative_support_change_by_population": comparison[
                "relative_support_change_by_population"
            ],
            "final_selection_convergence_claim": False,
        },
        "gates": gates,
        "failed_gates": [],
        "source_failed_staging_preserved": True,
        "field_inference_executed": False,
        "mock_seed_accessed": False,
        "observational_resolution_claim_allowed": False,
        "automatic_follow_on_executed": False,
        "Phase_B_allowed_by_this_result": False,
    }
    rows = Path(program["bindings"]["failed_row_manifest"]["path"]).read_bytes()
    return result, arrays, rows


def publish_datum(
    program_path: str | Path, output: str | Path, implementation_commit: str
) -> dict[str, Any]:
    program, program_sha = load_program(program_path)
    target = Path(output)
    stage = target.with_name(f".{target.name}.v3.staging")
    if target.exists() or stage.exists():
        raise PublisherError("Phase-A output or V3 staging path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    result, arrays, rows = collect_publication(
        program, program_sha, implementation_commit
    )
    np.savez_compressed(stage / "datum.npz", **arrays)
    (stage / "row_manifest.csv").write_bytes(rows)
    (stage / "result.json").write_bytes(canonical_json_bytes(result))
    artifacts = ("datum.npz", "row_manifest.csv", "result.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "program_sha256": program_sha,
        "files": {
            name: {
                "bytes": (stage / name).stat().st_size,
                "sha256": sha256_file(stage / name),
            }
            for name in artifacts
        },
    }
    (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    complete = {
        "schema": COMPLETE_SCHEMA,
        "status": STATUS,
        "program_sha256": program_sha,
        "manifest_sha256": sha256_file(stage / "manifest.json"),
        "result_sha256": sha256_file(stage / "result.json"),
        "field_inference_executed": False,
        "automatic_follow_on_executed": False,
    }
    (stage / "COMPLETE").write_bytes(canonical_json_bytes(complete))
    if {path.name for path in stage.iterdir()} != EXPECTED_FILES:
        raise PublisherError("Phase-A publication file set is not exact")
    stage.rename(target)
    return result


def validate_datum(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise PublisherError("published Phase-A file set is not exact")
    result = json.loads((root / "result.json").read_bytes())
    manifest = json.loads((root / "manifest.json").read_bytes())
    complete = json.loads((root / "COMPLETE").read_bytes())
    if result.get("status") != STATUS or result.get("failed_gates"):
        raise PublisherError("published Phase-A result is not passing")
    for name in ("datum.npz", "row_manifest.csv", "result.json"):
        expected = {
            "bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        if manifest["files"].get(name) != expected:
            raise PublisherError(f"manifest does not bind {name}")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise PublisherError("COMPLETE does not bind manifest")
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise PublisherError("COMPLETE does not bind result")
    with np.load(root / "datum.npz", allow_pickle=False) as archive:
        if set(archive.files) != {
            "counts_all",
            "counts_train",
            "counts_holdout",
            "raw_selection_exposure",
        }:
            raise PublisherError("published Phase-A NPZ key set changed")
        counts = np.asarray(archive["counts_all"])
        train = np.asarray(archive["counts_train"])
        holdout = np.asarray(archive["counts_holdout"])
        exposure = np.asarray(archive["raw_selection_exposure"])
    if counts.shape != (6, 32, 32, 32) or exposure.shape != counts.shape:
        raise PublisherError("published Phase-A array shape changed")
    if counts.dtype != np.int64 or train.dtype != np.int64 or holdout.dtype != np.int64:
        raise PublisherError("published Phase-A count dtype changed")
    if not np.array_equal(counts, train + holdout):
        raise PublisherError("published Phase-A split changed")
    if np.any((counts > 0) & (exposure <= 0.0)):
        raise PublisherError("published Phase-A exposure gate changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-publication")
    run.add_argument("--program", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = sub.add_parser("validate-datum")
    validate.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run-publication":
        result = publish_datum(args.program, args.output, args.implementation_commit)
    else:
        result = validate_datum(args.directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
