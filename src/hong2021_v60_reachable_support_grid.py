#!/usr/bin/env python
"""Materialize the train-only reachable-support survival grid selected by V59."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v46_tail_occupancy_audit import EXPECTED_OBJECTS
from hong2021_v50_network import LOWER_SUPPORT, UPPER_SUPPORT
from hong2021_v56_train import load_cache, load_program as load_v56_program


PROGRAM_SHA256 = "88cff6f3d3c4295a83edf6d1e919fe8fefe434abbaa13a1a829b06ed117c54ab"
PROGRAM_SCHEMA = "hong2021-v60-reachable-support-grid-program-v1"
SCHEMA = "hong2021-v60-reachable-support-grid-v1"


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V60 {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_grid_implementation_or_materialization"
    ):
        raise ValueError("V60 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v59_record"]), parent["v59_record_sha256"], "V59 record"
    )
    audit_row = record.get("audit", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or audit_row.get("classification") != parent["required_classification"]
        or audit_row.get("next") != parent["required_next"]
        or firewall.get("development_accessed")
        is not parent["required_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or firewall.get("Astrid_accessed") is not False
        or firewall.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V60 parent conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key in (
        "v59_audit",
        "v56_program",
        "v56_grid",
        "v54_threshold_selection",
        "conditioning_cache",
        "support_selection",
    ):
        if sha256_file(_path(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V60 frozen input differs: {key}")
    audit = _verified_json(
        _path(repo, frozen["v59_audit"]), frozen["v59_audit_sha256"], "V59 audit"
    )
    grid = _verified_json(
        _path(repo, frozen["v56_grid"]), frozen["v56_grid_sha256"], "V56 grid"
    )
    if (
        canonical_digest(audit) != frozen["v59_audit_decision_digest_sha256"]
        or canonical_digest(grid) != frozen["v56_grid_decision_digest_sha256"]
        or audit.get("numerical_requirements_pass") is not True
        or audit.get("classification") != parent["required_classification"]
        or audit.get("development_accessed") is not False
        or audit.get("independent_gate_locked") is not True
        or grid.get("cells") != 16
        or grid.get("development_accessed") is not False
    ):
        raise ValueError("V60 V59 audit or V56 grid binding differs")
    support = program["reachable_support_definition"]
    if support["unchanged_open_standardized_residual_support"] != [
        LOWER_SUPPORT,
        UPPER_SUPPORT,
    ]:
        raise ValueError("V60 bounded support differs")
    return program, audit, grid


def extended_grid(
    lower: float,
    existing_thresholds: np.ndarray,
    step: float,
    reachable_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        not math.isfinite(lower)
        or not math.isfinite(step)
        or not math.isfinite(reachable_upper)
        or step <= 0.0
        or existing_thresholds.ndim != 1
        or len(existing_thresholds) == 0
        or not np.all(np.isfinite(existing_thresholds))
        or not np.all(np.diff(existing_thresholds) > 0.0)
        or not lower < existing_thresholds[0]
        or not existing_thresholds[-1] < reachable_upper
    ):
        raise ValueError("V60 grid extension input differs")
    thresholds = existing_thresholds.astype(np.float64).tolist()
    while thresholds[-1] + step < reachable_upper:
        thresholds.append(thresholds[-1] + step)
    if thresholds[-1] != reachable_upper:
        thresholds.append(reachable_upper)
    values = np.asarray(thresholds, dtype=np.float64)
    edges = np.concatenate(([lower], values))
    proxy = np.square(np.power(10.0, edges) - 1.0)
    weights = np.diff(proxy) / (proxy[-1] - proxy[0])
    if (
        not np.array_equal(values[: len(existing_thresholds)], existing_thresholds)
        or values[-1] != reachable_upper
        or not np.all(np.diff(values) > 0.0)
        or not np.all(np.isfinite(proxy))
        or not np.all(np.isfinite(weights))
        or not np.all(weights > 0.0)
        or abs(float(weights.sum(dtype=np.float64)) - 1.0) > 1.0e-12
    ):
        raise RuntimeError("V60 extended grid differs")
    return values, weights


def materialize(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, _, v56_grid = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V60 materialization requires clean Lageunha")
    frozen = program["frozen_inputs"]
    _, v35, _ = load_v56_program(_path(repo, frozen["v56_program"]), repo)
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    domains: dict[str, Any] = {}
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            objects = int(row["train_objects"])
            if objects != EXPECTED_OBJECTS[domain]:
                raise RuntimeError("V60 train object count differs")
            data, cache = _open_split(row, "train")
            maximum_backbone = -math.inf
            maximum_truth = -math.inf
            truth_exceedances = 0
            native_voxels = 0
            try:
                for index in range(objects):
                    backbone = _backbone(cache, index).astype(np.float64)
                    truth = np.asarray(data["target"][index, 0], dtype=np.float32).astype(
                        np.float64
                    )
                    maximum_backbone = max(maximum_backbone, float(backbone.max()))
                    log10rho = 4.5 * truth
                    maximum_truth = max(maximum_truth, float(log10rho.max()))
                    truth_exceedances += int(
                        np.count_nonzero(log10rho > float(v56_grid["upper_edge_log10rho"]))
                    )
                    native_voxels += int(backbone.size)
                    if (index + 1) % 32 == 0 or index + 1 == objects:
                        print(f"[v60-grid] {domain} {index + 1}/{objects}", flush=True)
            finally:
                data.close()
                cache.close()
            sealed = v56_grid["domains"][domain]
            if (
                native_voxels != int(sealed["native_voxels"])
                or maximum_truth != float(sealed["maximum_log10rho"])
                or truth_exceedances != 0
            ):
                raise RuntimeError("V60 native truth confirmation differs")
            maximum_base = maximum_backbone + target_mean
            reachable_y = maximum_base + target_std * UPPER_SUPPORT
            reachable_log10rho = 4.5 * reachable_y
            if not math.isfinite(reachable_log10rho):
                raise RuntimeError("V60 reachable support is nonfinite")
            domains[domain] = {
                "train_objects": objects,
                "native_voxels": native_voxels,
                "maximum_backbone": maximum_backbone,
                "maximum_backbone_plus_target_mean": maximum_base,
                "reachable_upper_physical_y_supremum": reachable_y,
                "reachable_upper_log10rho_supremum": reachable_log10rho,
                "maximum_truth_log10rho": maximum_truth,
                "strict_truth_exceedances_above_V56_final_threshold": truth_exceedances,
            }
    finally:
        prepared.close()
    global_upper = max(
        float(row["reachable_upper_log10rho_supremum"]) for row in domains.values()
    )
    rule = program["grid_extension"]
    existing = np.asarray(v56_grid["thresholds_log10rho"], dtype=np.float64)
    lower = float(rule["lower_anchor_log10rho"])
    step = float(rule["unchanged_log10rho_step"])
    if (
        lower != float(v56_grid["lower_edge_log10rho"])
        or existing[-1] != float(rule["existing_final_threshold_log10rho"])
        or not np.allclose(np.diff(np.concatenate(([lower], existing))), step, rtol=0.0, atol=2e-16)
    ):
        raise ValueError("V60 inherited V56 spacing differs")
    thresholds, weights = extended_grid(lower, existing, step, global_upper)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_reachable_support_grid",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "v59_audit_sha256": frozen["v59_audit_sha256"],
        "v56_grid_sha256": frozen["v56_grid_sha256"],
        "conditioning_cache_sha256": frozen["conditioning_cache_sha256"],
        "open_standardized_residual_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "target_mean": target_mean,
        "target_std": target_std,
        "domains": domains,
        "lower_edge_log10rho": lower,
        "V56_existing_cells": len(existing),
        "total_cells": len(thresholds),
        "appended_cells": len(thresholds) - len(existing),
        "unchanged_log10rho_step": step,
        "global_reachable_upper_log10rho_supremum": global_upper,
        "thresholds_log10rho": thresholds.tolist(),
        "physical_moment_weights": weights.tolist(),
        "physical_moment_weight_sum": float(weights.sum(dtype=np.float64)),
        "reference_probability": float(rule["reference_probability"]),
        "normalization": float(rule["normalization"]),
        "coefficient": float(rule["coefficient"]),
        "existing_thresholds_byte_equal": bool(
            np.array_equal(thresholds[: len(existing)], existing)
        ),
        "final_threshold_equals_global_reachable_upper": bool(
            thresholds[-1] == global_upper
        ),
        "training_or_refit_performed": False,
        "validation_accessed": False,
        "development_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V60 refuses an existing grid")
    result = materialize(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
