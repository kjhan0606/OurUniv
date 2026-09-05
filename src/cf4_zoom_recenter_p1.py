#!/usr/bin/env python3
"""Re-evaluate every frozen P1 gate about the final M200c LG midpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cf4_parent_p1 import file_hash, score_member  # noqa: E402
from cf4_zoom_z0_gate import min_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1/gate_result_v2.json"))
    parser.add_argument("--p1-result", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_p1_v3_inverse/"
        "p1_result.json"))
    parser.add_argument("--p1-config", type=Path,
                        default=ROOT / "config/p1_targets_v2_observer.json")
    parser.add_argument("--hop-catalog", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1/"
        "hop_catalog_exact.npz"))
    parser.add_argument("--out", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1/"
        "environment_result_v2.json"))
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text())
    p1_result = json.loads(args.p1_result.read_text())
    config = json.loads(args.p1_config.read_text())
    seed = int(gate["screen_pair"]["small_scale_seed"])
    rows = [row for row in p1_result["members"] if int(row["seed"]) == seed]
    if len(rows) != 1:
        parser.error(f"could not resolve one P1 input row for seed {seed}")
    parent_path = Path(rows[0]["input"])
    with np.load(parent_path, allow_pickle=False) as data:
        initial = data["s_out"].astype(np.float32)
        nmesh = int(data["N"])
        spacing = float(data["spacing"])
        box = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]),
            "Ob": float(data["Ob"]),
            "h": float(data["hh"]),
            "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    midpoint = np.asarray(gate["pair"]["midpoint_mpc_h"], np.float64)
    observer_offset = min_image(midpoint - box / 2.0, box)

    import jax.numpy as jnp
    from scipy.ndimage import gaussian_filter
    from mock_pipeline import make_forward

    started = time.time()
    _, _, forward = make_forward(
        nmesh, spacing, jnp.float32, return_dens=True, cosmology=cosmology)
    density, _ = forward(jnp.asarray(initial))
    density.block_until_ready()
    smoothed = gaussian_filter(
        np.asarray(density, np.float32),
        config["density_smoothing_mpc_h"] / spacing,
        mode="wrap",
    )
    delta = smoothed / np.mean(smoothed, dtype=np.float64) - 1.0
    metrics = score_member(
        delta, spacing, config, omega_m=cosmology["Om"],
        observer_offset=observer_offset)

    with np.load(args.hop_catalog, allow_pickle=False) as data:
        halo_mass = np.asarray(data["mass"], np.float64)
        halo_pos = np.asarray(data["pos"], np.float64)
    distance = np.linalg.norm(min_image(halo_pos - midpoint, box), axis=1)
    massive = np.flatnonzero((halo_mass >= 5e12) & (distance <= 8.0))
    massive_rows = [{
        "catalog_index": int(k),
        "mass_fof_msun_h": float(halo_mass[k]),
        "distance_mpc_h": float(distance[k]),
        "center_mpc_h": halo_pos[k].tolist(),
    } for k in massive[np.argsort(-halo_mass[massive])]]
    no_massive_halo = not massive_rows
    environment_pass = bool(metrics["pass"] and no_massive_halo)
    result = {
        "schema": "ouruniv-cf4-recentered-p1-v2",
        "status": "complete",
        "gate": str(args.gate.resolve()),
        "p1_result": str(args.p1_result.resolve()),
        "p1_config": str(args.p1_config.resolve()),
        "p1_config_sha256": file_hash(args.p1_config),
        "conditioned_parent": str(parent_path.resolve()),
        "conditioned_parent_sha256": file_hash(parent_path),
        "observer_midpoint_mpc_h": midpoint.tolist(),
        "observer_offset_from_box_center_mpc_h": observer_offset.tolist(),
        "seconds": time.time() - started,
        "p1_recentered": metrics,
        "no_hop_host_ge5e12_within_8_mpc_h": no_massive_halo,
        "massive_hop_hosts_within_8_mpc_h": massive_rows,
        "pass": environment_pass,
        "scope": (
            "Frozen P1 PM density gate recentered on the final M200c midpoint, "
            "plus the definitive nearby massive-host exclusion from z=0 HOP."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    gate["environment"] = {
        "status": "complete",
        "result": str(args.out.resolve()),
        "p1_recentered_pass": bool(metrics["pass"]),
        "no_massive_hop_host_within_8_mpc_h": no_massive_halo,
        "passed": environment_pass,
    }
    gate["verdict"]["environment_gate"] = environment_pass
    gate["verdict"]["overall"] = bool(
        gate["verdict"]["p2b_lg_gate"] and environment_pass)
    temporary = args.gate.with_suffix(args.gate.suffix + ".tmp")
    temporary.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.gate)
    print(f"[environment] P1={metrics['n_gates_passed']}/{len(metrics['gates'])} "
          f"massive8={len(massive_rows)} pass={environment_pass}", flush=True)
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
