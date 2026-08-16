#!/usr/bin/env python3
"""Build immutable exact parent-response atlas shards for CF4 SMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

from cf4_aggregate_evidence_oracle import (
    AtlasBounds,
    extract_response_atlas,
    parent_response_grid,
    response_atlas_bounds,
)
from cf4_peak_evidence_phase_cache import full_spectrum_from_rfft


_WORKER_FILTER: np.ndarray | None = None
_WORKER_BOUNDS: AtlasBounds | None = None
_WORKER_OUTPUT_DIRECTORY: Path | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atlas_parent_case(case: dict[str, Any]) -> dict[str, Any]:
    if (
        _WORKER_FILTER is None
        or _WORKER_BOUNDS is None
        or _WORKER_OUTPUT_DIRECTORY is None
    ):
        raise RuntimeError("atlas worker was not initialized")
    parent_path = Path(case["path"])
    actual_parent_sha = sha256_file(parent_path)
    if actual_parent_sha != case["sha256"]:
        raise RuntimeError(f"parent hash mismatch for seed {case['seed']}")
    with np.load(parent_path, allow_pickle=False) as item:
        if int(item["sample_seed"]) != int(case["seed"]):
            raise RuntimeError("parent internal seed mismatch")
        coarse = item["s_out"].astype(np.float32)
    expected_coarse_n = _WORKER_FILTER.shape[0] // 3
    if coarse.shape != (expected_coarse_n,) * 3 or not np.all(np.isfinite(coarse)):
        raise RuntimeError("parent coarse field shape or finite gate failed")
    response = parent_response_grid(coarse, _WORKER_FILTER)
    if not np.all(np.isfinite(response)):
        raise RuntimeError("parent exact response contains nonfinite values")
    atlas = extract_response_atlas(response, _WORKER_BOUNDS)
    del response
    if (
        atlas.dtype != np.float64
        or atlas.shape != _WORKER_BOUNDS.shape
        or not np.all(np.isfinite(atlas))
    ):
        raise RuntimeError("response atlas shape, dtype, or finite gate failed")
    output = _WORKER_OUTPUT_DIRECTORY / f"parent_response_s{case['seed']}.npy"
    atomic_npy(output, atlas)
    return {
        "seed": int(case["seed"]),
        "parent_field": str(parent_path),
        "parent_field_sha256": actual_parent_sha,
        "atlas": str(output),
        "atlas_sha256": sha256_file(output),
        "shape": list(atlas.shape),
        "dtype": str(atlas.dtype),
    }


def validate_program(program: dict[str, Any], program_path: Path) -> None:
    if program.get("status") != "frozen_before_response_atlas_construction":
        raise RuntimeError("response-atlas program is not frozen")
    storage = program["storage"]
    if Path(storage["program"]).resolve() != program_path.resolve():
        raise RuntimeError("response-atlas program path is not canonical")
    design_path = Path(program["design"]["path"])
    if not design_path.is_absolute():
        design_path = program_path.parents[1] / design_path
    if sha256_file(design_path.resolve()) != program["design"]["sha256"]:
        raise RuntimeError("SMC design hash mismatch")
    design = json.loads(design_path.read_text())
    if design.get("status") != "frozen_scientific_design_before_implementation":
        raise RuntimeError("SMC scientific design is not frozen")
    authorization = design["authorization"]
    if not authorization.get("response_atlas_construction_authorized", False):
        raise RuntimeError("SMC design did not authorize atlas construction")
    if authorization.get("production_execution_authorized") is not False:
        raise RuntimeError("SMC production authorization opened prematurely")
    fixed = design["fixed_inputs"]
    atlas_design = design["oracle_and_cache"]["parent_response_atlas"]
    atlas = program.get("atlas", {})
    exact_atlas = {
        "prior_mean_mpc_h": fixed["midpoint_prior_mean_mpc_h"],
        "prior_sigma_mpc_h": fixed["midpoint_prior_sigma_mpc_h"],
        "dx_mpc_h": fixed["dx_mpc_h"],
        "sigma_extent": 10.0,
        "padding_cells": atlas_design["point_padding_cells"],
        "shape": [101, 101, 101],
        "dtype": atlas_design["dtype"],
        "outside_atlas_policy": atlas_design["outside_atlas_policy"],
    }
    if atlas != exact_atlas:
        raise RuntimeError("response-atlas constants differ from the frozen design")
    parents = program.get("parents", {})
    if parents != {
        "seed_range_inclusive": fixed["parent_seed_range_inclusive"],
        "count": fixed["parent_count"],
    }:
        raise RuntimeError("response-atlas parent contract differs from the design")
    density_filter = program.get("density_filter", {})
    design_filter = fixed["density_filter"]
    if density_filter != {
        "path": design_filter["path"],
        "sha256": design_filter["sha256"],
        "shape": design_filter["shape"],
        "dtype": design_filter["dtype"],
    }:
        raise RuntimeError("density-filter contract differs from the design")
    reference = program.get("reference_calibration", {})
    design_reference = fixed["reference_calibration"]
    if reference != {
        "path": design_reference["path"],
        "sha256": design_reference["sha256"],
        "status": "complete_reference_calibration_parent3429_pass",
    }:
        raise RuntimeError("reference-calibration contract differs from the design")
    execution = program.get("execution", {})
    if execution != {
        "host": design["execution"]["host"],
        "worker_processes": design["execution"]["worker_processes"],
        "threads_per_worker": design["execution"]["threads_per_worker"],
        "process_table_polling": design["execution"]["process_table_polling"],
    }:
        raise RuntimeError("response-atlas execution contract differs from the design")
    required_sources = {
        "src/cf4_aggregate_evidence_oracle.py",
        "src/cf4_parent_response_atlas.py",
        "src/cf4_peak_evidence_phase_cache.py",
        "src/cf4_projection_contract.py",
        "config/cf4_aggregate_evidence_annealed_smc_design.json",
    }
    pinned = program.get("pinned_local_files", [])
    if {item.get("path") for item in pinned} != required_sources:
        raise RuntimeError("response-atlas pinned source set is incomplete")
    for item in program["pinned_local_files"]:
        path = program_path.parents[1] / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"local hash mismatch: {item['path']}")
    filter_path = Path(density_filter["path"])
    if sha256_file(filter_path) != density_filter["sha256"]:
        raise RuntimeError("density filter preflight hash mismatch")
    filter_value = np.load(filter_path, mmap_mode="r", allow_pickle=False)
    if (
        list(filter_value.shape) != density_filter["shape"]
        or str(filter_value.dtype) != density_filter["dtype"]
        or not np.all(np.isfinite(filter_value))
    ):
        raise RuntimeError("density filter preflight shape, dtype, or finite gate failed")


def run(program: dict[str, Any]) -> dict[str, Any]:
    calibration_path = Path(program["reference_calibration"]["path"])
    if sha256_file(calibration_path) != program["reference_calibration"]["sha256"]:
        raise RuntimeError("reference calibration hash mismatch")
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("status") != program["reference_calibration"]["status"]:
        raise RuntimeError("reference calibration status mismatch")
    parent_cases = calibration["reference_field_hashes"]
    expected = np.arange(
        program["parents"]["seed_range_inclusive"][0],
        program["parents"]["seed_range_inclusive"][1] + 1,
    )
    actual = np.asarray([case["seed"] for case in parent_cases])
    if not np.array_equal(actual, expected):
        raise RuntimeError("parent calibration does not contain the exact seed range")

    filter_path = Path(program["density_filter"]["path"])
    if sha256_file(filter_path) != program["density_filter"]["sha256"]:
        raise RuntimeError("density filter hash mismatch")
    filter_rfft = np.load(filter_path, allow_pickle=False)
    if (
        list(filter_rfft.shape) != program["density_filter"]["shape"]
        or str(filter_rfft.dtype) != program["density_filter"]["dtype"]
        or not np.all(np.isfinite(filter_rfft))
    ):
        raise RuntimeError("density filter shape, dtype, or finite gate failed")
    filter_full = full_spectrum_from_rfft(filter_rfft)
    atlas_spec = program["atlas"]
    bounds = response_atlas_bounds(
        atlas_spec["prior_mean_mpc_h"],
        atlas_spec["prior_sigma_mpc_h"],
        dx_mpc_h=atlas_spec["dx_mpc_h"],
        sigma_extent=atlas_spec["sigma_extent"],
        padding_cells=atlas_spec["padding_cells"],
    )
    if list(bounds.shape) != atlas_spec["shape"]:
        raise RuntimeError("frozen response-atlas shape mismatch")

    output_directory = Path(program["storage"]["directory"])
    manifest_path = Path(program["storage"]["manifest"])
    if output_directory.exists() or manifest_path.exists():
        raise FileExistsError("refusing to reuse response-atlas state")
    output_directory.mkdir(parents=True, exist_ok=False)

    global _WORKER_FILTER, _WORKER_BOUNDS, _WORKER_OUTPUT_DIRECTORY
    _WORKER_FILTER = filter_full
    _WORKER_BOUNDS = bounds
    _WORKER_OUTPUT_DIRECTORY = output_directory
    workers = int(program["execution"]["worker_processes"])
    context = mp.get_context("fork")
    entries = []
    with context.Pool(processes=workers) as pool:
        for index, entry in enumerate(
            pool.imap(atlas_parent_case, parent_cases, chunksize=1), 1
        ):
            entries.append(entry)
            print(
                f"[response-atlas] {index}/{len(parent_cases)} seed={entry['seed']}",
                flush=True,
            )
    _WORKER_FILTER = None
    _WORKER_BOUNDS = None
    _WORKER_OUTPUT_DIRECTORY = None

    if [entry["seed"] for entry in entries] != expected.tolist():
        raise RuntimeError("atlas worker results broke parent order")
    manifest = {
        "schema": "ouruniv-cf4-parent-response-atlas-manifest-v1",
        "status": "complete_exact_parent_response_atlas",
        "program": program["storage"]["program"],
        "design": program["design"],
        "density_filter": program["density_filter"],
        "reference_calibration": program["reference_calibration"],
        "bounds": {
            "relative_min": list(bounds.relative_min),
            "relative_max": list(bounds.relative_max),
            "padded_min": list(bounds.padded_min),
            "padded_max": list(bounds.padded_max),
            "shape": list(bounds.shape),
        },
        "parent_count": len(entries),
        "dtype": "float64",
        "entries": entries,
        "information_firewall": {
            "CF4_deviance_loaded": False,
            "production_SMC_particles_loaded": False,
            "old_adaptation_cache_imported": False,
        },
        "decision": {
            "atlas_construction_pass": True,
            "production_SMC_authorized": False,
            "conditional_field_bank_authorized": False,
            "PM_or_RAMSES_authorized": False,
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    args = parser.parse_args()
    program_path = args.program.resolve()
    program = json.loads(program_path.read_text())
    validate_program(program, program_path)
    run(program)


if __name__ == "__main__":
    main()
