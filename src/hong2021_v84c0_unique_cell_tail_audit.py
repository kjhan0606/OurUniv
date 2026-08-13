#!/usr/bin/env python
"""V84C0 fit-only, unique-cell weighted tail-shape and split-validity audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v84b_contract import DOMAIN_ORDER, group_partition


PROGRAM_SCHEMA = "hong2021-v84c0-unique-cell-tail-shape-audit-program-v1"
PROGRAM_STATUS = "frozen_before_inner_development_payload_audit"
REPORT_SCHEMA = "hong2021-v84c0-unique-cell-tail-shape-audit-report-v1"
TNG_OUTER_SEED = 840384
TNG_OUTER_OBJECTS = 32
CAMELS_OUTER_SEEDS = {"SIMBA": 840484, "Swift": 840485}
CAMELS_OUTER_GROUPS = {"SIMBA": 2, "Swift": 3}
TNG_BLOCK_CELLS = 60
MINIMUM_TNG_BLOCK_UNIQUE_CELL_WEIGHT = 50_000.0
MINIMUM_TNG_BLOCK_TAIL_WEIGHT = 100.0
LOWER_THRESHOLD = -2.35
UPPER_THRESHOLD = 3.10
NEAR_SURVIVAL_FRACTION = 0.2
FAR_SURVIVAL_FRACTION = 0.02
CURVATURE_RATIO_DEVIATION = 0.15
MINIMUM_SIGN_CONSISTENCY = 0.70
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 840584


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def cube_cells(origin: np.ndarray, full_grid: int) -> np.ndarray:
    origin = np.asarray(origin, dtype=np.int64)
    if origin.shape != (3,):
        raise ValueError("V84C0 cube origin differs")
    axes = [(origin[axis] + np.arange(64)) % full_grid for axis in range(3)]
    return (
        axes[0][:, None, None] * full_grid**2
        + axes[1][None, :, None] * full_grid
        + axes[2][None, None, :]
    ).reshape(-1)


def _periodic_distance(query: np.ndarray, reference: np.ndarray, box: int) -> np.ndarray:
    delta = np.abs(query[:, None] - reference[None])
    delta = np.minimum(delta, box - delta)
    return np.sqrt(np.square(delta).sum(axis=-1))


def prospective_partition(v35: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Seal an inner audit set and a payload-unopened outer development set."""
    base = group_partition(v35)
    result: dict[str, dict[str, Any]] = {}
    tng_path = v35["development_domains"]["TNG100"]["train_data"]
    with h5py.File(tng_path, "r") as handle:
        origins = np.asarray(handle["cube_origin_cell"], dtype=np.int64) % 240
    base_fit = np.asarray(base["TNG100"]["fit"], dtype=np.int64)
    anchor = np.random.default_rng(TNG_OUTER_SEED).integers(0, 240, size=3)
    centers = (origins[base_fit] + 32) % 240
    distance = _periodic_distance(centers, anchor[None], 240)[:, 0]
    outer = base_fit[np.argsort(distance, kind="stable")[:TNG_OUTER_OBJECTS]]
    outer_mask = np.zeros(240**3, dtype=bool)
    for index in outer:
        outer_mask[cube_cells(origins[index], 240)] = True
    outer_set = set(map(int, outer))
    inner: list[int] = []
    embargo: list[int] = []
    for index in base_fit:
        if int(index) in outer_set:
            continue
        if outer_mask[cube_cells(origins[index], 240)].any():
            embargo.append(int(index))
        else:
            inner.append(int(index))
    inner_mask = np.zeros(240**3, dtype=bool)
    for index in inner:
        inner_mask[cube_cells(origins[index], 240)] = True
    intersection = int(np.count_nonzero(inner_mask & outer_mask))
    if intersection:
        raise RuntimeError("V84C0 TNG inner/outer target cell overlap")
    result["TNG100"] = {
        "inner": sorted(inner),
        "outer": sorted(map(int, outer)),
        "embargo": sorted(embargo),
        "anchor_cell": anchor.tolist(),
        "inner_unique_target_cells": int(inner_mask.sum()),
        "outer_unique_target_cells": int(outer_mask.sum()),
        "inner_outer_target_cell_intersection": intersection,
    }
    for domain in ("SIMBA", "Swift"):
        path = v35["development_domains"][domain]["train_data"]
        with h5py.File(path, "r") as handle:
            realization = np.asarray(handle["realization"], dtype=np.int64)
        base_fit = np.asarray(base[domain]["fit"], dtype=np.int64)
        available = np.unique(realization[base_fit])
        permutation = np.random.default_rng(CAMELS_OUTER_SEEDS[domain]).permutation(
            available
        )
        outer_groups = np.sort(permutation[: CAMELS_OUTER_GROUPS[domain]])
        inner_groups = np.sort(permutation[CAMELS_OUTER_GROUPS[domain] :])
        outer = base_fit[np.isin(realization[base_fit], outer_groups)]
        inner = base_fit[np.isin(realization[base_fit], inner_groups)]
        result[domain] = {
            "inner": sorted(map(int, inner)),
            "outer": sorted(map(int, outer)),
            "embargo": [],
            "inner_groups": sorted(map(int, inner_groups)),
            "outer_groups": sorted(map(int, outer_groups)),
            "group_intersection": [],
        }
    return result


def partition_digest(partition: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(partition, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _program_freeze_commit(program_path: Path, repo: Path) -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(program_path.resolve())],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_program(
    program_path: Path, repo: Path, commit: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    program = strict_json(program_path.resolve())
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("training_or_refit") is not False
        or program.get("statistically_independent") is not False
    ):
        raise ValueError("V84C0 program scope differs")
    for label, row in program["evidence"].items():
        path = Path(row["path"])
        path = path if path.is_absolute() else repo / path
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"V84C0 evidence differs: {label}")
    for label, row in program["implementation"].items():
        if sha256_file(repo / row["path"]) != row["sha256"]:
            raise ValueError(f"V84C0 implementation differs: {label}")
    freeze_commit = _program_freeze_commit(program_path, repo)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze_commit, commit],
        cwd=repo,
        check=False,
    ).returncode:
        raise ValueError("V84C0 execution does not descend from freeze")
    frozen = program["frozen_inputs"]
    v35_path = repo / frozen["v35_program"]
    if sha256_file(v35_path) != frozen["v35_program_sha256"]:
        raise ValueError("V84C0 V35 program differs")
    v35 = strict_json(v35_path)
    for domain in DOMAIN_ORDER:
        source = v35["development_domains"][domain]
        binding = frozen["training_domains"][domain]
        for kind in ("data", "cache"):
            key = f"train_{kind}"
            digest_key = f"train_{kind}_sha256"
            path = Path(binding[key])
            if (
                binding[key] != source[key]
                or binding[digest_key] != source[digest_key]
                or sha256_file(path) != binding[digest_key]
            ):
                raise ValueError(f"V84C0 {domain} train {kind} differs")
    numerics = program["fixed_numerics"]
    if (
        numerics.get("TNG_outer_seed") != TNG_OUTER_SEED
        or numerics.get("TNG_outer_objects") != TNG_OUTER_OBJECTS
        or numerics.get("CAMELS_outer_seeds") != CAMELS_OUTER_SEEDS
        or numerics.get("CAMELS_outer_group_counts") != CAMELS_OUTER_GROUPS
        or numerics.get("TNG_block_cells") != TNG_BLOCK_CELLS
        or numerics.get("lower_threshold") != LOWER_THRESHOLD
        or numerics.get("upper_threshold") != UPPER_THRESHOLD
        or numerics.get("near_survival_fraction") != NEAR_SURVIVAL_FRACTION
        or numerics.get("far_survival_fraction") != FAR_SURVIVAL_FRACTION
        or numerics.get("curvature_ratio_deviation") != CURVATURE_RATIO_DEVIATION
        or numerics.get("minimum_sign_consistency") != MINIMUM_SIGN_CONSISTENCY
        or numerics.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or numerics.get("bootstrap_seed") != BOOTSTRAP_SEED
    ):
        raise ValueError("V84C0 fixed numerics differ")
    partition = prospective_partition(v35)
    counts = {
        domain: {
            key: len(partition[domain][key]) for key in ("inner", "outer", "embargo")
        }
        for domain in DOMAIN_ORDER
    }
    if (
        partition_digest(partition) != program["partition"]["sha256"]
        or counts != program["partition"]["counts"]
    ):
        raise ValueError("V84C0 prospective partition differs")
    firewall = program["firewall"]
    if (
        firewall.get("outer_development_payload_access") != "forbidden"
        or firewall.get("V84B_burned_holdout_payload_access") != "forbidden"
        or firewall.get("validation_payload_access") != "forbidden"
        or firewall.get("independent_gate_locked") is not True
    ):
        raise ValueError("V84C0 firewall differs")
    return program, v35, partition


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(values) or len(values) != len(weights) or not 0.0 <= probability <= 1.0:
        raise ValueError("V84C0 weighted quantile input differs")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = probability * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    return float(sorted_values[index])


def tail_shape(
    excess: np.ndarray, weights: np.ndarray, total_weight: float
) -> dict[str, float]:
    tail_weight = float(np.asarray(weights, dtype=np.float64).sum())
    near_excess = weighted_quantile(excess, weights, 1.0 - NEAR_SURVIVAL_FRACTION)
    far_excess = weighted_quantile(excess, weights, 1.0 - FAR_SURVIVAL_FRACTION)
    near_scale = near_excess / math.log(1.0 / NEAR_SURVIVAL_FRACTION)
    far_scale = (far_excess - near_excess) / math.log(
        NEAR_SURVIVAL_FRACTION / FAR_SURVIVAL_FRACTION
    )
    return {
        "total_unique_cell_weight": total_weight,
        "tail_unique_cell_weight": tail_weight,
        "tail_mass": tail_weight / total_weight,
        "near_excess_quantile_at_tail_survival_0_2": near_excess,
        "far_excess_quantile_at_tail_survival_0_02": far_excess,
        "near_exponential_scale": near_scale,
        "far_exponential_scale": far_scale,
        "far_over_near_scale": far_scale / near_scale,
        "effective_far_tail_weight": tail_weight * FAR_SURVIVAL_FRACTION,
    }


def _summarize_units(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ratios = np.asarray([row["far_over_near_scale"] for row in rows], dtype=np.float64)
    generator = np.random.default_rng(seed)
    bootstrap = np.mean(
        ratios[generator.integers(0, len(ratios), size=(BOOTSTRAP_REPLICATES, len(ratios)))],
        axis=1,
    )
    return {
        "units": len(rows),
        "far_over_near_scale_by_unit": ratios.tolist(),
        "mean_far_over_near_scale": float(ratios.mean()),
        "median_far_over_near_scale": float(np.median(ratios)),
        "fraction_below_one": float(np.mean(ratios < 1.0)),
        "fraction_above_one": float(np.mean(ratios > 1.0)),
        "bootstrap_equal_unit_mean_90pct_interval": np.quantile(
            bootstrap, [0.05, 0.95]
        ).tolist(),
        "decreasing_far_slope_evidence": bool(
            np.median(ratios) < 1.0 - CURVATURE_RATIO_DEVIATION
            and np.mean(ratios < 1.0) >= MINIMUM_SIGN_CONSISTENCY
            and np.quantile(bootstrap, 0.95) < 1.0
        ),
        "increasing_far_slope_evidence": bool(
            np.median(ratios) > 1.0 + CURVATURE_RATIO_DEVIATION
            and np.mean(ratios > 1.0) >= MINIMUM_SIGN_CONSISTENCY
            and np.quantile(bootstrap, 0.05) > 1.0
        ),
    }


def _scan_unit(
    data: h5py.File,
    cache: h5py.File,
    indices: list[int],
    full_grid: int,
    target_mean: float,
    target_std: float,
    spatial_blocks: bool,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    origins = np.asarray(data["cube_origin_cell"], dtype=np.int64) % full_grid
    coverage = np.zeros(full_grid**3, dtype=np.uint16)
    cell_cache: dict[int, np.ndarray] = {}
    for index in indices:
        cells = cube_cells(origins[index], full_grid)
        cell_cache[index] = cells
        coverage[cells] += 1
    unique_cells = int(np.count_nonzero(coverage))
    occurrence_count = len(indices) * 64**3
    unit_total: dict[int, float] = {}
    tails: dict[str, dict[int, list[np.ndarray]]] = {
        "lower": {},
        "upper": {},
    }
    for index in indices:
        cells = cell_cache[index]
        weights = 1.0 / coverage[cells].astype(np.float64)
        if spatial_blocks:
            x = cells // full_grid**2
            remainder = cells % full_grid**2
            y = remainder // full_grid
            z = remainder % full_grid
            blocks_per_axis = full_grid // TNG_BLOCK_CELLS
            units = (
                (x // TNG_BLOCK_CELLS) * blocks_per_axis**2
                + (y // TNG_BLOCK_CELLS) * blocks_per_axis
                + z // TNG_BLOCK_CELLS
            ).astype(np.int16)
        else:
            units = np.zeros(len(cells), dtype=np.int16)
        truth = np.asarray(data["target"][index, 0], dtype=np.float32).reshape(-1)
        backbone = _backbone(cache, index).astype(np.float32).reshape(-1)
        residual = (truth - backbone - target_mean) / target_std
        for unit in np.unique(units):
            selected = units == unit
            unit_total[int(unit)] = unit_total.get(int(unit), 0.0) + float(
                weights[selected].sum()
            )
            values = residual[selected]
            selected_weights = weights[selected]
            lower = values < LOWER_THRESHOLD
            upper = values > UPPER_THRESHOLD
            if np.any(lower):
                row = tails["lower"].setdefault(int(unit), [[], []])
                row[0].append((LOWER_THRESHOLD - values[lower]).astype(np.float32))
                row[1].append(selected_weights[lower].astype(np.float32))
            if np.any(upper):
                row = tails["upper"].setdefault(int(unit), [[], []])
                row[0].append((values[upper] - UPPER_THRESHOLD).astype(np.float32))
                row[1].append(selected_weights[upper].astype(np.float32))
    rows: dict[int, dict[str, Any]] = {}
    for unit, total in unit_total.items():
        if spatial_blocks and total < MINIMUM_TNG_BLOCK_UNIQUE_CELL_WEIGHT:
            continue
        row: dict[str, Any] = {"unit": unit, "unique_cell_weight": total}
        valid = True
        for side in ("lower", "upper"):
            arrays = tails[side].get(unit)
            if arrays is None:
                valid = False
                break
            excess = np.concatenate(arrays[0])
            weights = np.concatenate(arrays[1])
            if spatial_blocks and float(weights.sum()) < MINIMUM_TNG_BLOCK_TAIL_WEIGHT:
                valid = False
                break
            row[side] = tail_shape(excess, weights, total)
        if valid:
            rows[unit] = row
    provenance = {
        "objects": len(indices),
        "voxel_occurrences": occurrence_count,
        "unique_physical_cells": unique_cells,
        "occurrences_per_unique_cell": occurrence_count / unique_cells,
        "inverse_multiplicity_weight_sum": float(sum(unit_total.values())),
        "included_units": len(rows),
    }
    return provenance, rows


def audit(program_path: Path, repo: Path, output_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V84C0 audit requires a clean frozen worktree")
    program, v35, partition = load_program(program_path, repo, commit)
    if output_path.resolve() != Path(program["output"]).resolve() or output_path.exists():
        raise FileExistsError("V84C0 output exists or differs")
    cache_path = Path(program["frozen_inputs"]["conditioning_cache"])
    if sha256_file(cache_path) != program["frozen_inputs"]["conditioning_cache_sha256"]:
        raise ValueError("V84C0 conditioning cache differs")
    with h5py.File(cache_path, "r") as prepared:
        target_mean = float(prepared["target_mean"][()])
        target_std = float(prepared["target_std"][()])
    domain_rows: dict[str, Any] = {}
    for domain_position, domain in enumerate(DOMAIN_ORDER):
        data, cache = _open_split(v35["development_domains"][domain], "train")
        try:
            indices = partition[domain]["inner"]
            if domain == "TNG100":
                provenance, units = _scan_unit(
                    data, cache, indices, 240, target_mean, target_std, True
                )
                unit_rows = list(units.values())
            else:
                realization = np.asarray(data["realization"], dtype=np.int64)
                provenance = {
                    "objects": len(indices),
                    "voxel_occurrences": 0,
                    "unique_physical_cells": 0,
                    "inverse_multiplicity_weight_sum": 0.0,
                    "units": 0,
                }
                unit_rows = []
                units = {}
                for group in partition[domain]["inner_groups"]:
                    selected = [index for index in indices if realization[index] == group]
                    group_provenance, group_rows = _scan_unit(
                        data, cache, selected, 80, target_mean, target_std, False
                    )
                    row = next(iter(group_rows.values()))
                    row["unit"] = int(group)
                    units[int(group)] = row
                    unit_rows.append(row)
                    for key in (
                        "objects",
                        "voxel_occurrences",
                        "unique_physical_cells",
                        "inverse_multiplicity_weight_sum",
                    ):
                        provenance[key] += group_provenance[key]
                    provenance["units"] += 1
                provenance["occurrences_per_unique_cell"] = (
                    provenance["voxel_occurrences"] / provenance["unique_physical_cells"]
                )
            domain_rows[domain] = {
                "provenance": provenance,
                "units": {str(key): value for key, value in sorted(units.items())},
                "lower_summary": _summarize_units(
                    [row["lower"] for row in unit_rows],
                    BOOTSTRAP_SEED + 2 * domain_position,
                ),
                "upper_summary": _summarize_units(
                    [row["upper"] for row in unit_rows],
                    BOOTSTRAP_SEED + 2 * domain_position + 1,
                ),
            }
            print(f"[v84c0] {domain} complete", flush=True)
        finally:
            data.close()
            cache.close()
    upper_decreasing = [
        domain_rows[domain]["upper_summary"]["decreasing_far_slope_evidence"]
        for domain in DOMAIN_ORDER
    ]
    upper_medians = [
        domain_rows[domain]["upper_summary"]["median_far_over_near_scale"]
        for domain in DOMAIN_ORDER
    ]
    lower_directions = [
        (
            domain_rows[domain]["lower_summary"]["decreasing_far_slope_evidence"],
            domain_rows[domain]["lower_summary"]["increasing_far_slope_evidence"],
        )
        for domain in DOMAIN_ORDER
    ]
    upper_two_slope_go = sum(upper_decreasing) >= 2 and max(upper_medians) < 1.05
    lower_consistent_go = all(row[0] for row in lower_directions) or all(
        row[1] for row in lower_directions
    )
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_inner_development_unique_cell_tail_shape_audit",
        "program": str(program_path.resolve()),
        "program_sha256": sha256_file(program_path),
        "program_freeze_commit": _program_freeze_commit(program_path, repo),
        "code_commit": commit,
        "host": socket.gethostname(),
        "partition_sha256": partition_digest(partition),
        "partition": {
            domain: {
                key: value
                for key, value in partition[domain].items()
                if key not in ("inner", "outer", "embargo")
            }
            | {
                "counts": {
                    key: len(partition[domain][key])
                    for key in ("inner", "outer", "embargo")
                }
            }
            for domain in DOMAIN_ORDER
        },
        "weighting": {
            "physical_cell": "each global target cell has total weight one across overlapping observer occurrences",
            "CAMELS_uncertainty_unit": "one realization, equal weight",
            "TNG100_uncertainty_unit": "nonoverlapping 60^3-cell global spatial block, equal weight after minimum coverage/tail requirements",
        },
        "tail_definition": {
            "lower_threshold": LOWER_THRESHOLD,
            "upper_threshold": UPPER_THRESHOLD,
            "near_tail_survival_fraction_within_threshold_exceedances": NEAR_SURVIVAL_FRACTION,
            "far_tail_survival_fraction_within_threshold_exceedances": FAR_SURVIVAL_FRACTION,
            "exponential_null": "far_over_near_scale equals one",
        },
        "domains": domain_rows,
        "decision": {
            "upper_two_slope_candidate_supported": upper_two_slope_go,
            "lower_two_slope_candidate_supported_with_cross_domain_consistency": lower_consistent_go,
            "V84C_training_go": upper_two_slope_go,
            "candidate_if_go": "one finite-moment upper two-slope tempered exponential versus the unchanged V84B exponential control; change lower tail only if cross-domain direction is consistent",
            "next": (
                "freeze_low_cost_nested_group_CV_V84C_candidate_and_control"
                if upper_two_slope_go
                else "stop_tail_model_expansion_and_return_to_CF4_low_k_plus_LCDM_high_k_forward_selection"
            ),
        },
        "gate_redesign": {
            "primary": "equal-group/block weighted NLL and 1e-3 tail calibration with uncertainty",
            "secondary": "1e-4 point estimate and group/block interval, never pooled-voxel hard pass",
            "TNG_outer_target_cell_intersection_required": 0,
            "pooled_voxel_only_pass_forbidden": True,
        },
        "training_or_refit_performed": False,
        "outer_development_payload_accessed": False,
        "V84B_burned_holdout_payload_accessed": False,
        "validation_payload_accessed": False,
        "consumed_development_payload_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.program, args.repo, args.out)


if __name__ == "__main__":
    main()
