#!/usr/bin/env python3
"""Four-field, no-PM technical pilot for the promoted LG high-k schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cf4_aggregate_evidence_oracle import points_from_geometry_key, target_vector
from cf4_lg_highk_conditional_field import conditional_field
from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    full_spectrum_from_rfft,
)


PILOT_INDICES = np.asarray([0, 64, 128, 192], dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _load_pilot_rows(schedule_path: Path) -> dict[str, np.ndarray]:
    required = {
        "schedule_index", "group_id", "parent_seed", "keys",
        "fine_field_seed", "likelihood_noise_seed", "posterior_weight",
    }
    with np.load(schedule_path, allow_pickle=False) as item:
        missing = required.difference(item.files)
        if missing:
            raise ValueError(f"schedule is missing arrays: {sorted(missing)}")
        if len(item["schedule_index"]) != 256 \
                or not np.array_equal(item["schedule_index"], np.arange(256)):
            raise ValueError("schedule identity changed")
        rows = {name: np.asarray(item[name])[PILOT_INDICES] for name in required}
    if not np.array_equal(rows["schedule_index"], PILOT_INDICES) \
            or not np.array_equal(rows["group_id"], np.arange(4)):
        raise ValueError("technical pilot no longer contains one row per bridge group")
    if not np.allclose(rows["posterior_weight"], 1.0 / 256.0):
        raise ValueError("technical pilot schedule weights changed")
    return rows


def _parent_entries(manifest_path: Path) -> dict[int, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete_exact_parent_response_atlas" \
            or manifest.get("parent_count") != 256:
        raise ValueError("parent response atlas manifest is not complete")
    entries = {int(row["seed"]): row for row in manifest["entries"]}
    if sorted(entries) != list(range(3193, 3449)):
        raise ValueError("parent manifest seed range changed")
    return entries


def run_technical_pilot(
    *,
    schedule_path: Path,
    schedule_sha256: str,
    filter_path: Path,
    filter_sha256: str,
    parent_manifest_path: Path,
    parent_manifest_sha256: str,
    output_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    if workers != 1:
        raise ValueError("the frozen technical pilot requires exactly one FFT worker")
    pins = {
        "schedule": (Path(schedule_path), schedule_sha256),
        "density_filter": (Path(filter_path), filter_sha256),
        "parent_manifest": (Path(parent_manifest_path), parent_manifest_sha256),
    }
    for label, (path, expected) in pins.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} SHA256 changed")
    rows = _load_pilot_rows(Path(schedule_path))
    entries = _parent_entries(Path(parent_manifest_path))
    filter_rfft = np.load(filter_path, allow_pickle=False)
    if filter_rfft.shape != (576, 576, 289) or filter_rfft.dtype != np.float32:
        raise ValueError("production density filter identity changed")
    filter_full = full_spectrum_from_rfft(filter_rfft)
    del filter_rfft
    point_sets = [
        points_from_geometry_key(key, fine_n=576) for key in rows["keys"]
    ]
    covariance, covariance_diagnostics = covariance_for_point_sets(
        filter_full, 192, point_sets
    )
    targets = target_vector(1.8, 1.5)
    cases, all_pass = [], True
    for local_index, schedule_index in enumerate(PILOT_INDICES):
        parent_seed = int(rows["parent_seed"][local_index])
        entry = entries[parent_seed]
        parent_path = Path(entry["parent_field"])
        if sha256_file(parent_path) != entry["parent_field_sha256"]:
            raise RuntimeError(f"parent {parent_seed} SHA256 changed")
        with np.load(parent_path, allow_pickle=False) as item:
            if int(item["sample_seed"]) != parent_seed:
                raise ValueError("parent internal seed changed")
            coarse = np.asarray(item["s_out"], dtype=np.float32)
        field, diagnostics = conditional_field(
            coarse,
            filter_full,
            point_sets[local_index],
            targets,
            0.25,
            fine_seed=int(rows["fine_field_seed"][local_index]),
            noise_seed=int(rows["likelihood_noise_seed"][local_index]),
            signal_covariance=covariance[local_index],
            float_dtype=np.float32,
            workers=workers,
        )
        gates = {
            "finite_field": bool(np.all(np.isfinite(field))),
            "coarse_roundtrip": bool(
                diagnostics["coarse_roundtrip_relative_RMS"] <= 2.0e-6
            ),
            "correction_in_null_space": bool(
                diagnostics["correction_restriction_relative_RMS"] <= 2.0e-6
            ),
            "response_identity": bool(
                diagnostics["maximum_response_identity_error"] <= 2.0e-5
            ),
            "null_power": bool(
                0.95 <= diagnostics["null_subspace_mean_square"] <= 1.05
            ),
            "global_mean": bool(abs(diagnostics["field_mean"]) <= 0.005),
            "peak_evidence_not_reapplied": bool(
                diagnostics["peak_evidence_reapplied"] is False
            ),
        }
        passed = all(gates.values())
        all_pass = all_pass and passed
        cases.append({
            "schedule_index": int(schedule_index),
            "group_id": int(rows["group_id"][local_index]),
            "parent_seed": parent_seed,
            "field_sha256": hashlib.sha256(memoryview(field).cast("B")).hexdigest(),
            "field_persisted": False,
            "diagnostics": diagnostics,
            "gates": gates,
            "passed": passed,
        })
        del field, coarse
    result = {
        "schema": "ouruniv-cf4-lg-highk-technical-pilot-v1",
        "status": (
            "complete_pass_four_field_technical_pilot"
            if all_pass else "complete_fail_four_field_technical_pilot"
        ),
        "field_count": 4,
        "no_PM_or_halo_finder_run": True,
        "no_field_persisted": True,
        "pins": {
            label: {"path": str(path), "sha256": digest}
            for label, (path, digest) in pins.items()
        },
        "covariance_diagnostics": covariance_diagnostics,
        "cases": cases,
        "decision": {
            "technical_pilot_pass": all_pass,
            "production_field_generation_authorized": False,
            "PM_authorized": False,
        },
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=False, exist_ok=False)
    _atomic_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--filter", type=Path, required=True)
    parser.add_argument("--filter-sha256", required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_technical_pilot(
        schedule_path=args.schedule,
        schedule_sha256=args.schedule_sha256,
        filter_path=args.filter,
        filter_sha256=args.filter_sha256,
        parent_manifest_path=args.parent_manifest,
        parent_manifest_sha256=args.parent_manifest_sha256,
        output_root=args.output_root,
    )
    print(json.dumps({
        "status": result["status"],
        "field_count": result["field_count"],
        "technical_pilot_pass": result["decision"]["technical_pilot_pass"],
    }, sort_keys=True))
    if not result["decision"]["technical_pilot_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
