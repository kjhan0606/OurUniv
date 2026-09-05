#!/usr/bin/env python
"""Freeze V50 open support from all and only the immutable train residuals."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v46_tail_occupancy_audit import EXPECTED_OBJECTS
from hong2021_v48_train import load_cache, load_program as load_v48_program


PROGRAM_SCHEMA = "hong2021-v50-train-only-bounded-support-selection-program-v1"
PROGRAM_SHA256 = "38cd50239958004935e6a028db4619b7b5a976c6e7afa3286bb959be5da6389e"
RESULT_SCHEMA = "hong2021-v50-train-only-bounded-support-selection-v1"
MINIMUM_MARGIN = 0.25
RANGE_MARGIN_FRACTION = 0.05
NATIVE_VOXELS = 64**3


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V50 support {label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_train_scan_or_V50_model_implementation"
    ):
        raise ValueError("V50 support program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v49_record"]).resolve(),
        parent["v49_record_sha256"],
        "V49 record",
    )
    if (
        record.get("status") != parent["required_status"]
        or record.get("selected_next_likelihood", {}).get("family")
        != parent["required_family"]
        or record.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V50 support V49 evidence or firewall differs")
    frozen = program["frozen_inputs"]
    v48_path = (repo / frozen["v48_program"]).resolve()
    if (
        sha256_file(v48_path) != frozen["v48_program_sha256"]
        or sha256_file(Path(frozen["conditioning_cache"]))
        != frozen["conditioning_cache_sha256"]
    ):
        raise ValueError("V50 support frozen input hash differs")
    _, v35, _ = load_v48_program(v48_path, repo)
    return program, v35


def select_support(minimum: float, maximum: float) -> dict[str, float]:
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError("V50 support extrema differ")
    value_range = maximum - minimum
    margin = max(MINIMUM_MARGIN, RANGE_MARGIN_FRACTION * value_range)
    return {
        "global_minimum": minimum,
        "global_maximum": maximum,
        "range": value_range,
        "symmetric_margin": margin,
        "lower_support": minimum - margin,
        "upper_support": maximum + margin,
    }


def scan(program_path: Path, repo: Path, output: Path) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V50 support scan requires a clean committed worktree")
    if output.exists():
        raise FileExistsError("V50 support scan refuses existing output")
    frozen = program["frozen_inputs"]
    prepared = load_cache(
        Path(frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    if not math.isfinite(target_mean) or not math.isfinite(target_std) or target_std <= 0.0:
        raise RuntimeError("V50 support normalization differs")
    domains: dict[str, Any] = {}
    global_minimum = math.inf
    global_maximum = -math.inf
    total_voxels = 0
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            objects = int(row["train_objects"])
            if objects != EXPECTED_OBJECTS[domain]:
                raise RuntimeError("V50 support train object count differs")
            domain_minimum = math.inf
            domain_maximum = -math.inf
            data, cache = _open_split(row, "train")
            try:
                for object_index in range(objects):
                    truth = np.asarray(data["target"][object_index, 0], dtype=np.float32)
                    backbone = _backbone(cache, object_index).astype(np.float32)
                    standardized = (
                        truth.astype(np.float64)
                        - backbone.astype(np.float64)
                        - target_mean
                    ) / target_std
                    if standardized.size != NATIVE_VOXELS or not np.isfinite(standardized).all():
                        raise RuntimeError("V50 support train residual differs")
                    domain_minimum = min(domain_minimum, float(standardized.min()))
                    domain_maximum = max(domain_maximum, float(standardized.max()))
                    if (object_index + 1) % 32 == 0 or object_index + 1 == objects:
                        print(
                            f"[v50-support] {domain} {object_index + 1}/{objects}",
                            flush=True,
                        )
            finally:
                data.close()
                cache.close()
            count = objects * NATIVE_VOXELS
            domains[domain] = {
                "train_objects": objects,
                "native_voxels": count,
                "minimum_standardized_residual": domain_minimum,
                "maximum_standardized_residual": domain_maximum,
            }
            total_voxels += count
            global_minimum = min(global_minimum, domain_minimum)
            global_maximum = max(global_maximum, domain_maximum)
    finally:
        prepared.close()
    if total_voxels != int(program["population"]["total_native_voxels"]):
        raise RuntimeError("V50 support total voxel count differs")
    support = select_support(global_minimum, global_maximum)
    if not (
        support["lower_support"] < global_minimum
        and global_maximum < support["upper_support"]
    ):
        raise RuntimeError("V50 support is not strictly interior")
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_train_only_open_support_selection",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "selection_code_commit": commit,
        "worktree_clean": clean,
        "normalization": {
            "target_mean": target_mean,
            "target_std": target_std,
        },
        "domains": domains,
        "total_native_voxels": total_voxels,
        "support": support,
        "all_train_values_strictly_interior": True,
        "model_implementation_performed": False,
        "fit_or_optimizer_performed": False,
        "validation_accessed": False,
        "development_arrays_accessed": False,
        "support_adjusted_after_scan": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scan(args.program, args.repo, args.out)


if __name__ == "__main__":
    main()
