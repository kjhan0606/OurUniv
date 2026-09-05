#!/usr/bin/env python
"""Paired high-resolution Local-Group screen for the P1 parent survivors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

RHO_CRIT = 2.775e11
VUNIT_KMS = 100.0


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict:
    """Load a frozen P2 config, optionally inheriting a SHA-pinned base."""
    raw = json.loads(path.read_text())
    if "extends" not in raw:
        return raw
    base_path = Path(raw["extends"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    expected = raw.get("extends_sha256")
    actual = file_hash(base_path)
    if expected != actual:
        raise RuntimeError(
            f"base config hash mismatch for {base_path}: {actual} != {expected}")
    base = load_config(base_path)
    return _deep_merge(base, raw)


def rank_score(pair: dict, ranking: dict) -> float:
    weights = ranking["weights"]
    target_mass = ranking["target_member_mass_msun_h"]
    score = weights["log_mass_each"] * (
        abs(np.log10(pair["m1_fof_msun_h"] / target_mass))
        + abs(np.log10(pair["m2_fof_msun_h"] / target_mass))
    )
    score += weights["separation_per_0p30_mpc_h"] * (
        abs(pair["separation_mpc_h"] - ranking["target_separation_mpc_h"]) / 0.30
    )
    score += weights["midpoint_per_2_mpc_h"] * pair["midpoint_offset_mpc_h"] / 2.0
    score += weights["radial_velocity_per_80_km_s"] * (
        abs(
            pair["total_radial_velocity_km_s"]
            - ranking["target_total_radial_velocity_km_s"]
        )
        / 80.0
    )
    score += weights["tangential_velocity_per_80_km_s"] * (
        abs(
            pair["tangential_velocity_km_s"]
            - ranking["target_tangential_velocity_km_s"]
        )
        / 80.0
    )
    if pair["m33_candidate"] is not None:
        score += weights["m33_candidate_bonus"]
    return float(score)


def find_pairs(halos: dict, centre: np.ndarray, screen: dict, m33_gate: dict):
    from scipy.spatial import cKDTree

    pos = halos["pos"].astype(np.float64, copy=False)
    mass = halos["mass"].astype(np.float64, copy=False)
    vel = halos["vel"].astype(np.float64, copy=False)
    radius = np.linalg.norm(pos - centre, axis=1)
    mass_lo, mass_hi = screen["pair_member_mass_range_msun_h"]
    eligible = np.flatnonzero(
        (mass >= mass_lo)
        & (mass <= mass_hi)
        & (radius <= screen["pair_midpoint_max_offset_mpc_h"] + 1.0)
    )
    if eligible.size < 2:
        return []
    pair_tree = cKDTree(pos[eligible])
    sep_lo, sep_hi = screen["pair_separation_range_mpc_h"]
    candidates = pair_tree.query_pairs(sep_hi, output_type="ndarray")
    massive = np.flatnonzero(mass >= screen["isolation_mass_threshold_msun_h"])
    massive_tree = cKDTree(pos[massive]) if massive.size else None
    m33_lo, m33_hi = m33_gate["mass_range_msun_h"]
    possible_m33 = np.flatnonzero((mass >= m33_lo) & (mass <= m33_hi))

    rows = []
    for aa, bb in candidates:
        i, j = int(eligible[aa]), int(eligible[bb])
        separation_vector = pos[i] - pos[j]
        separation = float(np.linalg.norm(separation_vector))
        if separation < sep_lo:
            continue
        ratio = float(max(mass[i], mass[j]) / min(mass[i], mass[j]))
        if ratio > screen["pair_mass_ratio_max"]:
            continue
        midpoint = 0.5 * (pos[i] + pos[j])
        midpoint_offset = float(np.linalg.norm(midpoint - centre))
        if midpoint_offset > screen["pair_midpoint_max_offset_mpc_h"]:
            continue
        if massive_tree is None:
            isolation = 99.0
        else:
            distances, near = massive_tree.query(midpoint, k=min(3, massive.size))
            distances = np.atleast_1d(distances)
            near = np.atleast_1d(near)
            external = [
                float(distance)
                for distance, local_index in zip(distances, near)
                if int(massive[int(local_index)]) not in (i, j)
            ]
            isolation = min(external) if external else 99.0
        if isolation < screen["isolation_radius_mpc_h"]:
            continue

        radial_hat = separation_vector / separation
        relative_velocity = vel[i] - vel[j]
        peculiar_radial = float(np.dot(relative_velocity, radial_hat))
        tangential = float(
            np.linalg.norm(relative_velocity - peculiar_radial * radial_hat)
        )
        total_radial = peculiar_radial + 100.0 * separation

        third = None
        if possible_m33.size:
            third_distance = np.minimum(
                np.linalg.norm(pos[possible_m33] - pos[i], axis=1),
                np.linalg.norm(pos[possible_m33] - pos[j], axis=1),
            )
            valid = (
                (possible_m33 != i)
                & (possible_m33 != j)
                & (mass[possible_m33] < min(mass[i], mass[j]))
                & (third_distance <= 0.60)
            )
            if np.any(valid):
                indices = np.flatnonzero(valid)
                selected = indices[np.argmin(third_distance[valid])]
                k = int(possible_m33[selected])
                host = i if np.linalg.norm(pos[k] - pos[i]) < np.linalg.norm(
                    pos[k] - pos[j]
                ) else j
                third = {
                    "halo_index": k,
                    "host_index": int(host),
                    "mass_fof_msun_h": float(mass[k]),
                    "host_separation_mpc_h": float(np.linalg.norm(pos[k] - pos[host])),
                }

        row = {
            "halo_i": i,
            "halo_j": j,
            "m1_fof_msun_h": float(mass[i]),
            "m2_fof_msun_h": float(mass[j]),
            "mass_ratio": ratio,
            "separation_mpc_h": separation,
            "midpoint_mpc_h": midpoint.tolist(),
            "midpoint_offset_mpc_h": midpoint_offset,
            "isolation_mpc_h": isolation,
            "peculiar_radial_velocity_km_s": peculiar_radial,
            "total_radial_velocity_km_s": total_radial,
            "tangential_velocity_km_s": tangential,
            "m33_candidate": third,
        }
        rows.append(row)
    return rows


def extract_central_particles(
    particles,
    centre: np.ndarray,
    half_width: float,
    *,
    velocity_unit: float,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy PM state to host and reconstruct only the central Eulerian cube.

    Materializing ``Particles.pos()`` for all 576^3 particles adds a large
    periodic-remainder executable on the GPU.  The P2 screen needs only the
    central 60 Mpc/h cube, so reconstruct positions in bounded host chunks
    from pmid and displacement while preserving the full N=576 PM evolution.
    """
    pmid = np.asarray(particles.pmid)
    disp = np.asarray(particles.disp, dtype=np.float32)
    vel = np.asarray(particles.vel, dtype=np.float32)
    cell_size = np.asarray(particles.conf.cell_size, dtype=np.float32)
    box_size = np.asarray(particles.conf.box_size, dtype=np.float32)
    selected_pos = []
    selected_vel = []
    for start in range(0, pmid.shape[0], chunk_size):
        stop = min(start + chunk_size, pmid.shape[0])
        pos = (
            pmid[start:stop].astype(np.float32) * cell_size
            + disp[start:stop]
        ) % box_size
        select = np.all(np.abs(pos - centre) <= half_width, axis=1)
        if np.any(select):
            selected_pos.append(pos[select])
            selected_vel.append(vel[start:stop][select] * velocity_unit)
    if not selected_pos:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    return np.concatenate(selected_pos), np.concatenate(selected_vel)


def extract_central_arrays(
    position_array,
    velocity_array,
    centre: np.ndarray,
    half_width: float,
    *,
    velocity_unit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy only final position/velocity outputs and select the central cube.

    This path pairs with ``make_forward(return_particle_arrays=True)`` and
    avoids exporting the full pmid+disp+vel particle pytree from XLA.
    """
    pos = np.asarray(position_array, dtype=np.float32)
    select = np.all(np.abs(pos - centre) <= half_width, axis=1)
    central_pos = pos[select].copy()
    del pos
    vel = np.asarray(velocity_array, dtype=np.float32)
    central_vel = (vel[select] * velocity_unit).astype(np.float32, copy=False)
    return central_pos, central_vel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p1-result",
        type=Path,
        default=ROOT / "recon/linear_cr/p1_parent_v1/p1_result.json",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/p2_lg_targets_v1.json"
    )
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "recon/linear_cr/p2_screen_v1"
    )
    parser.add_argument(
        "--only", nargs="*", default=[], help="optional parent:smallseed combinations"
    )
    parser.add_argument(
        "--proposal-manifest",
        type=Path,
        help=(
            "optional manifest of preconditioned N-mesh white fields; normally "
            "SHA-pinned as input.conditioned_proposal_manifest in the config"
        ),
    )
    parser.add_argument(
        "--conditioned-p1-result",
        type=Path,
        help="P1 result for parent-resolution projections of conditioned proposals",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    p1 = json.loads(args.p1_result.read_text())
    if not config.get("frozen_before_high_resolution_forwarding"):
        raise RuntimeError("P2 config is not frozen")
    if p1["passing_seeds"] != config["parent_seeds"]:
        raise RuntimeError(
            f"P1 survivors {p1['passing_seeds']} differ from P2 parents "
            f"{config['parent_seeds']}"
        )
    parent_files = {
        row["seed"]: Path(row["input"]) for row in p1["members"] if row["pass"]
    }
    manifest_value = config.get("input", {}).get("conditioned_proposal_manifest")
    manifest_path = args.proposal_manifest or (
        Path(manifest_value) if manifest_value is not None else None)
    proposal_files: dict[tuple[int, int], tuple[Path, str]] = {}
    proposal_manifest_hash = None
    if manifest_path is not None:
        proposal_manifest_hash = file_hash(manifest_path)
        expected_manifest_hash = config.get("input", {}).get(
            "conditioned_proposal_manifest_sha256")
        if expected_manifest_hash is not None and proposal_manifest_hash != expected_manifest_hash:
            raise RuntimeError(
                "conditioned proposal manifest hash mismatch: "
                f"{proposal_manifest_hash} != {expected_manifest_hash}")
        proposal_manifest = json.loads(manifest_path.read_text())
        if proposal_manifest.get("status") != "complete":
            raise RuntimeError("conditioned proposal manifest is not complete")
        for entry in proposal_manifest.get("entries", []):
            key = (int(entry["parent_seed"]), int(entry["proposal_seed"]))
            field_path = Path(entry["field"])
            if key in proposal_files:
                raise RuntimeError(f"duplicate conditioned proposal {key}")
            actual_field_hash = file_hash(field_path)
            if actual_field_hash != entry["field_sha256"]:
                raise RuntimeError(f"conditioned proposal hash mismatch for {field_path}")
            proposal_files[key] = (field_path, actual_field_hash)
        combinations = [
            (parent, small)
            for parent in config["parent_seeds"]
            for small in config["paired_small_scale_seeds"]
        ]
        missing = [pair for pair in combinations if pair not in proposal_files]
        if missing:
            raise RuntimeError(f"conditioned proposal manifest lacks {missing}")

        conditioned_p1_value = config.get("input", {}).get(
            "conditioned_p1_result")
        conditioned_p1_path = args.conditioned_p1_result or (
            Path(conditioned_p1_value) if conditioned_p1_value is not None else None)
        if conditioned_p1_path is None:
            raise RuntimeError(
                "conditioned proposals require a separately validated P1 result")
        conditioned_p1_hash = file_hash(conditioned_p1_path)
        expected_p1_hash = config.get("input", {}).get(
            "conditioned_p1_result_sha256")
        if expected_p1_hash is not None and conditioned_p1_hash != expected_p1_hash:
            raise RuntimeError(
                "conditioned P1 result hash mismatch: "
                f"{conditioned_p1_hash} != {expected_p1_hash}")
        conditioned_p1 = json.loads(conditioned_p1_path.read_text())
        if conditioned_p1.get("status") != "complete":
            raise RuntimeError("conditioned P1 result is not complete")
        required_proposal_seeds = sorted({small for _, small in combinations})
        if sorted(conditioned_p1.get("passing_seeds", [])) != required_proposal_seeds:
            raise RuntimeError(
                "P2 proposal seeds differ from the conditioned P1 survivors")
    else:
        conditioned_p1_path = None
        conditioned_p1_hash = None
        combinations = [
            (parent, small)
            for parent in config["parent_seeds"]
            for small in config["paired_small_scale_seeds"]
        ]
    if args.only:
        requested = {
            tuple(int(value) for value in item.split(":")) for item in args.only
        }
        combinations = [pair for pair in combinations if pair in requested]
    if not combinations:
        raise RuntimeError("no P2 combinations requested")

    import jax
    import jax.numpy as jnp
    from cf4_make_ic import embed_ic, embed_ic_projected
    from fof import fof
    from mock_pipeline import make_forward

    first_parent = parent_files[config["parent_seeds"][0]]
    with np.load(first_parent) as data:
        coarse_n = int(data["N"])
        box_size = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]),
            "Ob": float(data["Ob"]),
            "h": float(data["hh"]),
            "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    screen = config["screen"]
    fine_n = int(screen["mesh_size"])
    canonical_n = int(screen.get("canonical_mesh_size", fine_n))
    if canonical_n < fine_n:
        raise RuntimeError("canonical mesh cannot be smaller than the PM mesh")
    spacing = box_size / fine_n
    if not np.isclose(spacing, screen["particle_spacing_mpc_h"]):
        raise RuntimeError("P2 screen spacing differs from the frozen config")
    particle_mass = cosmology["Om"] * RHO_CRIT * spacing**3
    centre = np.full(3, box_size / 2.0)

    print(f"[P2] JAX {jax.__version__}; device={jax.devices()[0]}", flush=True)
    print(f"[P2] config SHA-256={file_hash(args.config)}", flush=True)
    print(
        f"[P2] {len(combinations)} paired forwards; {coarse_n}^3 -> "
        f"{canonical_n}^3 canonical -> {fine_n}^3 PM; "
        f"m_particle={particle_mass:.3e} Msun/h",
        flush=True,
    )
    _, _, forward = make_forward(
        fine_n, spacing, jnp.float32, return_dens=False, cosmology=cosmology,
        return_particle_arrays=True,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    cache = {}
    for number, (parent_seed, small_seed) in enumerate(combinations, 1):
        started = time.time()
        proposal_source = None
        proposal_source_hash = None
        if proposal_files:
            proposal_source, proposal_source_hash = proposal_files[
                (parent_seed, small_seed)]
            with np.load(proposal_source, allow_pickle=False) as data:
                initial = data["s_conditioned"].astype(np.float32)
            if initial.shape != (fine_n, fine_n, fine_n):
                raise RuntimeError(
                    f"conditioned proposal {proposal_source} has shape "
                    f"{initial.shape}, expected {(fine_n,) * 3}")
        else:
            if parent_seed not in cache:
                with np.load(parent_files[parent_seed]) as data:
                    cache[parent_seed] = data["s_out"].astype(np.float64)
            initial = embed_ic_projected(
                cache[parent_seed], canonical_n, fine_n, small_seed)
        final_pos, final_vel = forward(jnp.asarray(initial))
        initial = None
        half_width = screen["central_half_width_mpc_h"]
        central_pos, central_vel = extract_central_arrays(
            final_pos,
            final_vel,
            centre,
            half_width,
            velocity_unit=VUNIT_KMS,
        )
        final_pos = final_vel = None
        halos = fof(
            central_pos,
            central_vel,
            L=box_size,
            mean_sep=spacing,
            b=0.2,
            n_min=20,
            m_particle=particle_mass,
            periodic=False,
            verbose=False,
        )
        pairs = find_pairs(halos, centre, screen, config["m33_subpeak_gate"])
        for pair in pairs:
            pair["ranking_score"] = rank_score(pair, config["ranking"])
        pairs.sort(key=lambda row: row["ranking_score"])
        elapsed = time.time() - started
        row = {
            "parent_seed": parent_seed,
            "small_scale_seed": small_seed,
            "seconds": elapsed,
            "n_central_particles": int(central_pos.shape[0]),
            "n_halos": int(halos["mass"].size),
            "n_screen_pairs": len(pairs),
            "screen_pass": bool(pairs),
            "best_pair": pairs[0] if pairs else None,
            # Preserve every physically admissible pair.  The cheapest P1
            # environment recheck is observer-centred on the evolved pair,
            # so a realization must not be rejected merely because its
            # morphology-only rank put a different pair first.
            "screen_pairs": pairs,
            "conditioned_proposal": (
                str(proposal_source.resolve()) if proposal_source else None),
            "conditioned_proposal_sha256": proposal_source_hash,
        }
        results.append(row)
        combo = f"p{parent_seed}_s{small_seed}"
        np.savez(
            args.outdir / f"halos_{combo}.npz",
            halo_pos=halos["pos"].astype(np.float32),
            halo_vel=halos["vel"].astype(np.float32),
            halo_mass=halos["mass"].astype(np.float32),
            particle_mass=np.float64(particle_mass),
            box_size=np.float64(box_size),
        )
        (args.outdir / f"result_{combo}.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
        if pairs:
            best = pairs[0]
            summary = (
                f"{len(pairs)} pairs; best sep={best['separation_mpc_h']:.2f} "
                f"M=({best['m1_fof_msun_h']:.2e},{best['m2_fof_msun_h']:.2e}) "
                f"r={best['midpoint_offset_mpc_h']:.2f} "
                f"vr={best['total_radial_velocity_km_s']:+.0f} "
                f"vt={best['tangential_velocity_km_s']:.0f}"
            )
        else:
            summary = "no screen pair"
        print(
            f"[P2] {number:02d}/{len(combinations)} {combo}: {summary} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        del central_pos, central_vel, halos
        gc.collect()

    complete = {
        "schema": "cf4-p2-screen-result-v1",
        "status": "complete" if not args.only else "subset",
        "config": str(args.config.resolve()),
        "config_sha256": file_hash(args.config),
        "p1_result": str(args.p1_result.resolve()),
        "conditioned_proposal_manifest": (
            str(manifest_path.resolve()) if manifest_path else None),
        "conditioned_proposal_manifest_sha256": proposal_manifest_hash,
        "conditioned_p1_result": (
            str(conditioned_p1_path.resolve()) if conditioned_p1_path else None),
        "conditioned_p1_result_sha256": conditioned_p1_hash,
        "cosmology": cosmology,
        "canonical_mesh_size": canonical_n,
        "pm_mesh_size": fine_n,
        "particle_mass_msun_h": particle_mass,
        "results": results,
        "passing_combinations": [
            [row["parent_seed"], row["small_scale_seed"]]
            for row in results
            if row["screen_pass"]
        ],
    }
    (args.outdir / "p2_screen_result.json").write_text(
        json.dumps(complete, indent=2) + "\n"
    )
    print(
        f"[P2] passing combinations: {complete['passing_combinations']}", flush=True
    )


if __name__ == "__main__":
    main()
