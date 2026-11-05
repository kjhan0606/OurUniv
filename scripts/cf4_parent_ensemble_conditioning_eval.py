#!/usr/bin/env python3
"""Forward validation for the low-rank parent-conditioning proposals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cf4_parent_cluster_pm_fof_scan import local_fof, sg_xyz  # noqa: E402
from cf4_parent_p1 import score_member  # noqa: E402
from mock_pipeline import RHO_CRIT, make_forward  # noqa: E402


def shell_power_ratio(candidate: np.ndarray, reference: np.ndarray, box: float) -> dict:
    n = candidate.shape[0]
    ck = np.fft.rfftn(candidate, norm="ortho")
    rk = np.fft.rfftn(reference, norm="ortho")
    kxy = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box / n)
    kmag = np.sqrt(
        kxy[:, None, None] ** 2 + kxy[None, :, None] ** 2 + kz[None, None, :] ** 2
    )
    rows = []
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, np.inf)):
        keep = (kmag >= lo) & (kmag < hi)
        ratio = float(np.mean(np.abs(ck[keep]) ** 2) / np.mean(np.abs(rk[keep]) ** 2))
        rows.append({"k_lo": lo, "k_hi": None if np.isinf(hi) else hi, "power_ratio": ratio})
    return {"shells": rows, "maximum_abs_ratio_minus_one_above_0p6": max(
        abs(row["power_ratio"] - 1.0) for row in rows if row["k_lo"] >= 0.6
    )}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    proposals = json.loads(args.proposal_manifest.read_text())
    parent_manifest = json.loads(args.parent_manifest.read_text())
    config = json.loads(args.p1_config.read_text())
    if proposals["status"] != "complete_proposals_require_forward_validation":
        raise RuntimeError(f"proposal manifest is not executable: {proposals['status']}")
    source_by_seed = {}
    for path_text in parent_manifest["outputs"]:
        path = Path(path_text)
        with np.load(path, allow_pickle=False) as data:
            source_by_seed[int(data["sample_seed"])] = path

    first_path = Path(proposals["proposals"][0]["path"])
    with np.load(first_path, allow_pickle=False) as data:
        n = int(data["N"])
        spacing = float(data["spacing"])
        box = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]), "Ob": float(data["Ob"]),
            "h": float(data["hh"]), "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    particle_mass = cosmology["Om"] * RHO_CRIT * spacing**3
    _, _, forward = make_forward(n, spacing, jnp.float32, return_dens=True, cosmology=cosmology)
    observer = np.full(3, box / 2.0)
    targets = {}
    for name in ("Virgo", "Coma"):
        spec = config["clusters"][name]
        targets[name] = {
            "position": (observer + sg_xyz(
                spec["sgl_deg"], spec["sgb_deg"],
                spec["distance_mpc"] * config["cosmology_h"],
            )) % box,
            "search_radius": float(spec["search_radius_mpc_h"]),
        }

    rows = []
    for number, proposal_row in enumerate(proposals["proposals"], 1):
        path = Path(proposal_row["path"])
        with np.load(path, allow_pickle=False) as data:
            white = np.asarray(data["s_out"], dtype=np.float32)
        with np.load(source_by_seed[int(proposal_row["base_seed"])], allow_pickle=False) as data:
            reference = np.asarray(data["s_out"], dtype=np.float32)
        density, particles = forward(jnp.asarray(white))
        density.block_until_ready()
        positions = np.asarray(particles.pos(), dtype=np.float32)
        smoothed = gaussian_filter(
            np.asarray(density, dtype=np.float32),
            config["density_smoothing_mpc_h"] / spacing,
            mode="wrap",
        )
        delta = smoothed / np.mean(smoothed, dtype=np.float64) - 1.0
        p1 = score_member(delta, spacing, config, omega_m=cosmology["Om"])
        anchors = {
            name: local_fof(
                positions, spec["position"], spec["search_radius"], 10.0,
                box, spacing, particle_mass,
            )
            for name, spec in targets.items()
        }
        cluster_mass_pass = bool(
            anchors["Virgo"]["maximum_mass_msun_h"] is not None
            and anchors["Virgo"]["maximum_mass_msun_h"] >= 1.0e14
            and anchors["Coma"]["maximum_mass_msun_h"] is not None
            and anchors["Coma"]["maximum_mass_msun_h"] >= 5.0e14
        )
        power = shell_power_ratio(white, reference, box)
        power_pass = bool(power["maximum_abs_ratio_minus_one_above_0p6"] <= 1.0e-5)
        row = {
            **proposal_row,
            "p1": p1,
            "anchors": anchors,
            "cluster_mass_pass": cluster_mass_pass,
            "power": power,
            "high_k_power_preserved": power_pass,
            "pass": bool(p1["pass"] and cluster_mass_pass and power_pass),
        }
        rows.append(row)
        print(
            f"[proposal-eval] {number}/{len(proposals['proposals'])} "
            f"{proposal_row['label']} p1={p1['pass']} clusters={cluster_mass_pass} "
            f"highk={power_pass}",
            flush=True,
        )
        del density, particles, positions, smoothed, delta

    passing = [row["label"] for row in rows if row["pass"]]
    result = {
        "schema": "ouruniv-cf4-parent-ensemble-conditioning-eval-v1",
        "status": "complete_no_automatic_promotion",
        "proposal_manifest": str(args.proposal_manifest.resolve()),
        "parent_manifest": str(args.parent_manifest.resolve()),
        "p1_config": str(args.p1_config.resolve()),
        "rows": rows,
        "passing_proposals": passing,
        "decision": "PARENT_CANDIDATE_AVAILABLE" if passing else "CONDITIONING_PILOT_FAILED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passing": passing, "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
