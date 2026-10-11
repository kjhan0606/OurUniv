#!/usr/bin/env python3
"""Stream LG-conditioned high-k proposals through PM/FoF without field retention."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cf4_lg_peak_cr import (  # noqa: E402
    condition_translated_constraints,
    linear_density_filter,
    prepare_translated_conditioner,
    two_peak_points,
)
from cf4_make_ic import embed_ic  # noqa: E402
from cf4_p2_screen import (  # noqa: E402
    RHO_CRIT,
    VUNIT_KMS,
    extract_central_arrays,
    find_pairs,
    load_config,
    rank_score,
)
from fof import fof  # noqa: E402
from mock_pipeline import make_forward  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def radial_free_rfft_mask(n: int, box: float, frozen_k_max: float) -> np.ndarray:
    """Return the Hermitian-compatible rFFT mask for modes above a physical k cut."""
    spacing = box / n
    kxy = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=spacing)
    kmag2 = (
        kxy[:, None, None] ** 2
        + kxy[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    return kmag2 > float(frozen_k_max) ** 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    program = json.loads(args.config.read_text())
    parent_path = Path(program["parent_field"])
    if sha256_file(parent_path) != program["parent_field_sha256"]:
        raise RuntimeError("parent field SHA-256 mismatch")
    p2_path = Path(program["hard_p2_config"])
    if not p2_path.is_absolute():
        p2_path = ROOT / p2_path
    p2 = load_config(p2_path)
    output = Path(program["output"])
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    with np.load(parent_path, allow_pickle=False) as data:
        coarse = np.asarray(data["s_out"], dtype=np.float32)
        cosmology = {
            "Om": float(data["Om"]), "Ob": float(data["Ob"]),
            "h": float(data["hh"]), "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    n = int(program["mesh_size"])
    box = float(program["box_size_mpc_h"])
    spacing = box / n
    frozen_k_max = float(program["frozen_k_max_h_mpc"])
    peak = program["peak_constraints"]
    screen = p2["screen"]
    if n != int(screen["mesh_size"]) or not np.isclose(spacing, screen["particle_spacing_mpc_h"]):
        raise ValueError("conditioning and P2 grids differ")

    density_filter = linear_density_filter(
        n, box, float(peak["gaussian_radius_mpc_h"]), cosmology)
    free = radial_free_rfft_mask(n, box, frozen_k_max)
    prepared = prepare_translated_conditioner(density_filter, free, (n, n, n))
    separation_cells = int(round(float(peak["protohalo_separation_mpc_h"]) / spacing))
    shell_cells = int(round(float(peak["shell_radius_mpc_h"]) / spacing))
    prior = peak["protohalo_midpoint_prior"]
    prior_mean = np.asarray(prior["mean_mpc_h"], dtype=np.float64)
    prior_sigma = np.asarray(prior["sigma_mpc_h"], dtype=np.float64)
    bank = program["seed_bank"]
    count = int(bank["count"])

    _, _, forward = make_forward(
        n, spacing, jnp.float32, return_dens=False, cosmology=cosmology,
        return_particle_arrays=True,
    )
    centre = np.full(3, box / 2.0)
    particle_mass = cosmology["Om"] * RHO_CRIT * spacing**3
    retain_passing_field = bool(
        program.get("retention", {}).get("passing_conditioned_fields", False)
    )
    rows = []
    started_all = time.time()
    for index in range(count):
        started = time.time()
        field_seed = int(bank["field_seed_start"]) + index
        geometry_seed = int(bank["geometry_seed_start"]) + index
        noise_seed = int(bank["likelihood_noise_seed_start"]) + index
        midpoint_seed = int(bank["midpoint_seed_start"]) + index
        midpoint_draw = np.random.default_rng(midpoint_seed).normal(prior_mean, prior_sigma)
        midpoint = np.full(3, n // 2, dtype=np.int64) + np.rint(midpoint_draw / spacing).astype(np.int64)
        midpoint_realized = (midpoint - n // 2) * spacing
        axis = np.random.default_rng(geometry_seed).normal(size=3)
        points, kinds = two_peak_points(n, midpoint, axis, separation_cells, shell_cells)
        targets = np.where(
            kinds == 1,
            float(peak["centre_target_delta_linear"]),
            float(peak["six_shell_target_delta_linear"]),
        )
        base = embed_ic(coarse, n, field_seed)
        field, conditioning = condition_translated_constraints(
            base,
            density_filter,
            free,
            points,
            targets,
            float(peak["likelihood_sigma_delta"]),
            noise_seed,
            prepared=prepared,
        )
        field_hash = hashlib.sha256(memoryview(field).cast("B")).hexdigest()
        final_pos, final_vel = forward(jnp.asarray(field))
        del base
        if not retain_passing_field:
            del field
            field = None
        central_pos, central_vel = extract_central_arrays(
            final_pos,
            final_vel,
            centre,
            float(screen["central_half_width_mpc_h"]),
            velocity_unit=VUNIT_KMS,
        )
        del final_pos, final_vel
        halos = fof(
            central_pos,
            central_vel,
            L=box,
            mean_sep=spacing,
            b=0.2,
            n_min=20,
            m_particle=particle_mass,
            periodic=False,
            verbose=False,
        )
        pairs = find_pairs(halos, centre, screen, p2["m33_subpeak_gate"])
        for pair in pairs:
            pair["ranking_score"] = rank_score(pair, p2["ranking"])
        pairs.sort(key=lambda row: row["ranking_score"])
        row = {
            "index": index,
            "field_seed": field_seed,
            "geometry_seed": geometry_seed,
            "likelihood_noise_seed": noise_seed,
            "midpoint_seed": midpoint_seed,
            "protohalo_midpoint_offset_draw_mpc_h": midpoint_draw.tolist(),
            "protohalo_midpoint_offset_realized_mpc_h": midpoint_realized.tolist(),
            "axis": (axis / np.linalg.norm(axis)).tolist(),
            "field_sha256": field_hash,
            "field_persisted": False,
            "conditioning": conditioning,
            "n_central_particles": int(central_pos.shape[0]),
            "n_halos": int(halos["mass"].size),
            "n_screen_pairs": len(pairs),
            "screen_pass": bool(pairs),
            "best_pair": pairs[0] if pairs else None,
            "screen_pairs": pairs,
            "seconds": time.time() - started,
        }
        expected = program.get("expected_reproduction")
        if expected is not None:
            if count != 1:
                raise ValueError("expected_reproduction is valid only for a one-row run")
            observed = {
                "field_sha256": field_hash,
                "screen_pass": bool(pairs),
                "n_screen_pairs": len(pairs),
            }
            if observed != expected:
                raise RuntimeError(
                    f"selected realization did not reproduce: {observed} != {expected}"
                )
        if pairs:
            np.savez(
                output / f"passing_halos_{index:03d}.npz",
                halo_pos=np.asarray(halos["pos"], dtype=np.float32),
                halo_vel=np.asarray(halos["vel"], dtype=np.float32),
                halo_mass=np.asarray(halos["mass"], dtype=np.float32),
                particle_mass=np.float64(particle_mass),
                box_size=np.float64(box),
            )
            if retain_passing_field:
                field_path = output / f"passing_field_{index:03d}.npz"
                np.savez(
                    field_path,
                    s_conditioned=field,
                    N=np.int64(n),
                    spacing=np.float64(spacing),
                    L=np.float64(box),
                    hh=np.float64(cosmology["h"]),
                    Om=np.float64(cosmology["Om"]),
                    Ob=np.float64(cosmology["Ob"]),
                    A_s_1e9=np.float64(cosmology["A_s_1e9"]),
                    ns=np.float64(cosmology["ns"]),
                    parent_seed=np.int64(program["parent_seed"]),
                    field_seed=np.int64(field_seed),
                    geometry_seed=np.int64(geometry_seed),
                    likelihood_noise_seed=np.int64(noise_seed),
                    midpoint_seed=np.int64(midpoint_seed),
                )
                row["persisted_field"] = str(field_path.resolve())
                row["persisted_field_sha256"] = sha256_file(field_path)
        rows.append(row)
        print(
            f"[lg-cond] {index + 1:02d}/{count} seed={field_seed} "
            f"mid={midpoint_realized.tolist()} pairs={len(pairs)} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
        del field, central_pos, central_vel, halos
        gc.collect()

    passing = [row["index"] for row in rows if row["screen_pass"]]
    result = {
        "schema": "ouruniv-cf4-lg-highk-conditioning-stream-result-v1",
        "status": "complete_no_conditioned_fields_persisted",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "parent_field": str(parent_path.resolve()),
        "parent_field_sha256": program["parent_field_sha256"],
        "hard_p2_config": str(p2_path.resolve()),
        "hard_p2_config_sha256": sha256_file(p2_path),
        "grid": {"N": n, "spacing_mpc_h": spacing, "box_mpc_h": box},
        "frozen_k_max_h_mpc": frozen_k_max,
        "particle_mass_msun_h": particle_mass,
        "rows": rows,
        "passing_indices": passing,
        "pass_fraction": len(passing) / count,
        "seconds": time.time() - started_all,
        "decision": "CONDITIONED_LG_CANDIDATES_AVAILABLE" if passing else "NO_CONDITIONED_LG_CANDIDATE",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passing_indices": passing, "decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
