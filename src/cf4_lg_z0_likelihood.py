#!/usr/bin/env python3
"""Prospective z=0 Local-Group likelihood on PM halo catalogues."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


VUNIT_HUBBLE_KMS_PER_MPC_H = 100.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def periodic_delta(a: np.ndarray, b: np.ndarray, box_size: float) -> np.ndarray:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (delta + box_size / 2.0) % box_size - box_size / 2.0


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return -math.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return logsumexp(values) - math.log(values.size) if values.size else -math.inf


def _normal_logpdf(value: np.ndarray, target: np.ndarray, sigma: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    residual = (value - target) / sigma
    return float(np.sum(-0.5 * residual**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)))


def enumerate_candidate_pairs(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    *,
    centre: np.ndarray,
    box_size: float,
    preselection: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enumerate a loose, target-independent support of unordered halo pairs."""
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64)
    mass_lo, mass_hi = map(float, preselection["member_mass_range_msun_h"])
    sep_lo, sep_hi = map(float, preselection["separation_range_mpc_h"])
    midpoint_max = float(preselection["midpoint_max_offset_mpc_h"])
    local_radius = np.linalg.norm(periodic_delta(pos, centre, box_size), axis=1)
    eligible = np.flatnonzero(
        (mass >= mass_lo)
        & (mass <= mass_hi)
        & (local_radius <= midpoint_max + sep_hi / 2.0)
    )
    if eligible.size < 2:
        return []

    tree = cKDTree(pos[eligible], boxsize=box_size)
    index_pairs = tree.query_pairs(sep_hi, output_type="ndarray")
    external_threshold = float(preselection["external_mass_threshold_msun_h"])
    massive = np.flatnonzero(mass >= external_threshold)
    rows: list[dict[str, Any]] = []
    for aa, bb in index_pairs:
        i, j = int(eligible[aa]), int(eligible[bb])
        separation_vector = periodic_delta(pos[i], pos[j], box_size)
        separation = float(np.linalg.norm(separation_vector))
        if separation < sep_lo:
            continue
        ratio = float(max(mass[i], mass[j]) / min(mass[i], mass[j]))
        if ratio > float(preselection["mass_ratio_max"]):
            continue
        midpoint = np.mod(pos[j] + 0.5 * separation_vector, box_size)
        midpoint_offset_vector = periodic_delta(midpoint, centre, box_size)
        midpoint_offset = float(np.linalg.norm(midpoint_offset_vector))
        if midpoint_offset > midpoint_max:
            continue

        external = [int(k) for k in massive if int(k) not in (i, j)]
        if external:
            isolation = float(np.min(np.linalg.norm(
                periodic_delta(pos[external], midpoint, box_size), axis=1)))
        else:
            isolation = box_size / 2.0
        radial_hat = separation_vector / separation
        relative_velocity = vel[i] - vel[j]
        peculiar_radial = float(np.dot(relative_velocity, radial_hat))
        tangential = float(np.linalg.norm(
            relative_velocity - peculiar_radial * radial_hat))
        rows.append({
            "halo_i": i,
            "halo_j": j,
            "masses_msun_h": sorted([float(mass[i]), float(mass[j])], reverse=True),
            "mass_ratio": ratio,
            "separation_mpc_h": separation,
            "midpoint_mpc_h": midpoint.tolist(),
            "midpoint_offset_vector_mpc_h": midpoint_offset_vector.tolist(),
            "midpoint_offset_mpc_h": midpoint_offset,
            "isolation_mpc_h": isolation,
            "peculiar_radial_velocity_km_s": peculiar_radial,
            "total_radial_velocity_km_s": (
                peculiar_radial + VUNIT_HUBBLE_KMS_PER_MPC_H * separation),
            "tangential_velocity_km_s": tangential,
        })
    return rows


def pair_log_likelihood(pair: dict[str, Any], likelihood: dict[str, Any]) -> tuple[float, dict[str, float]]:
    mass_cfg = likelihood["member_log10_mass"]
    mass_component = _normal_logpdf(
        np.log10(pair["masses_msun_h"]),
        np.log10(mass_cfg["target_msun_h"]),
        mass_cfg["sigma_dex"],
    )
    sep_cfg = likelihood["separation_mpc_h"]
    separation_component = _normal_logpdf(
        pair["separation_mpc_h"], sep_cfg["target"], sep_cfg["sigma"])
    midpoint_cfg = likelihood["midpoint_offset_vector_mpc_h"]
    midpoint_component = _normal_logpdf(
        pair["midpoint_offset_vector_mpc_h"],
        midpoint_cfg["target"],
        midpoint_cfg["sigma_each_axis"],
    )
    radial_cfg = likelihood["total_radial_velocity_km_s"]
    radial_component = _normal_logpdf(
        pair["total_radial_velocity_km_s"], radial_cfg["target"], radial_cfg["sigma"])
    tangential_cfg = likelihood["tangential_speed_km_s"]
    tangential_component = _normal_logpdf(
        pair["tangential_velocity_km_s"], tangential_cfg["target"], tangential_cfg["sigma"])
    isolation_cfg = likelihood["isolation"]
    z = (
        float(pair["isolation_mpc_h"])
        - float(isolation_cfg["half_probability_radius_mpc_h"])
    ) / float(isolation_cfg["width_mpc_h"])
    isolation_component = float(-np.logaddexp(0.0, -z))
    components = {
        "member_log10_mass": mass_component,
        "separation": separation_component,
        "midpoint": midpoint_component,
        "total_radial_velocity": radial_component,
        "tangential_speed": tangential_component,
        "isolation": isolation_component,
    }
    return float(sum(components.values())), components


def score_catalog(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    *,
    centre: np.ndarray,
    box_size: float,
    program: dict[str, Any],
) -> dict[str, Any]:
    pairs = enumerate_candidate_pairs(
        positions,
        velocities,
        masses,
        centre=centre,
        box_size=box_size,
        preselection=program["candidate_preselection"],
    )
    for pair in pairs:
        score, components = pair_log_likelihood(pair, program["z0_likelihood"])
        pair["log_likelihood"] = score
        pair["log_likelihood_components"] = components
    pairs.sort(key=lambda row: row["log_likelihood"], reverse=True)
    mixture = logmeanexp(np.asarray([row["log_likelihood"] for row in pairs]))
    return {
        "n_candidate_pairs": len(pairs),
        "log_likelihood": mixture,
        "best_pair": pairs[0] if pairs else None,
        "candidate_pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    program = json.loads(args.program.read_text())
    if program.get("status") != "frozen_before_v6_catalog_scoring":
        parser.error("program is not frozen before v6 catalogue scoring")

    repo = Path(__file__).resolve().parents[1]
    parent_spec = program["parent_result"]
    parent_path = repo / parent_spec["path"]
    if sha256_file(parent_path) != parent_spec["sha256"]:
        parser.error("v6 terminal result hash mismatch")
    parent = json.loads(parent_path.read_text())
    if parent.get("status") != parent_spec["required_status"]:
        parser.error("v6 terminal result status mismatch")

    development = program["development_input"]
    p2_dir = Path(development["p2_directory"])
    p2_path = p2_dir / "p2_screen_result.json"
    preview_path = p2_dir / "recentered_p1_preview.json"
    if sha256_file(p2_path) != development["p2_result_sha256"]:
        parser.error("v6 P2 result hash mismatch")
    if sha256_file(preview_path) != development["recentered_p1_preview_sha256"]:
        parser.error("v6 recentered P1 preview hash mismatch")
    p2 = json.loads(p2_path.read_text())
    hard_passes = {int(row[1]) for row in p2["passing_combinations"]}

    rows = []
    for result_row in p2["results"]:
        parent_seed = int(result_row["parent_seed"])
        seed = int(result_row["small_scale_seed"])
        catalog_path = p2_dir / f"halos_p{parent_seed}_s{seed}.npz"
        with np.load(catalog_path, allow_pickle=False) as data:
            box_size = float(data["box_size"])
            catalog = score_catalog(
                data["halo_pos"], data["halo_vel"], data["halo_mass"],
                centre=np.full(3, box_size / 2.0),
                box_size=box_size,
                program=program,
            )
        rows.append({
            "parent_seed": parent_seed,
            "small_scale_seed": seed,
            "catalog": str(catalog_path.resolve()),
            "catalog_sha256": sha256_file(catalog_path),
            "old_hard_screen_pass": seed in hard_passes,
            **catalog,
        })
    if len(rows) != int(development["expected_catalogs"]):
        parser.error("unexpected number of v6 development catalogues")

    finite_indices = [i for i, row in enumerate(rows) if np.isfinite(row["log_likelihood"])]
    weights = np.zeros(len(rows), dtype=np.float64)
    if finite_indices:
        finite_log = np.asarray([rows[i]["log_likelihood"] for i in finite_indices])
        finite_weights = np.exp(finite_log - logsumexp(finite_log))
        weights[finite_indices] = finite_weights
    for row, weight in zip(rows, weights):
        row["normalized_development_weight"] = float(weight)
    ess = float(1.0 / np.sum(weights**2)) if np.any(weights) else 0.0
    maximum_weight = float(weights.max(initial=0.0))
    ranked = sorted(rows, key=lambda row: row["normalized_development_weight"], reverse=True)
    gate = program["development_gate"]
    top_k = int(gate["top_k_for_hard_screen_overlap"])
    overlap = sum(bool(row["old_hard_screen_pass"]) for row in ranked[:top_k])
    checks = {
        "minimum_finite_catalogs": len(finite_indices) >= int(gate["minimum_finite_catalogs"]),
        "minimum_effective_sample_size": ess >= float(gate["minimum_effective_sample_size"]),
        "maximum_single_normalized_weight": maximum_weight <= float(gate["maximum_single_normalized_weight"]),
        "minimum_old_hard_screen_pairs_in_top_k": overlap >= int(gate["minimum_old_hard_screen_pairs_in_top_k"]),
    }
    passed = all(checks.values())
    report = {
        "schema": "ouruniv-lg-z0-forward-likelihood-v7-development-result-v1",
        "status": "complete_pass_authorize_fresh_v7" if passed else "complete_fail_fresh_v7_locked",
        "program": str(args.program.resolve()),
        "program_sha256": sha256_file(args.program),
        "v6_parent_result_sha256": sha256_file(parent_path),
        "n_catalogs": len(rows),
        "n_finite_catalogs": len(finite_indices),
        "effective_sample_size": ess,
        "maximum_single_normalized_weight": maximum_weight,
        "old_hard_screen_pairs_in_top_k": overlap,
        "checks": checks,
        "authorize_fresh_v7": passed,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "n_catalogs", "n_finite_catalogs",
        "effective_sample_size", "maximum_single_normalized_weight",
        "old_hard_screen_pairs_in_top_k", "checks", "authorize_fresh_v7",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
