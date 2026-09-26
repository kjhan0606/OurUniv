#!/usr/bin/env python3
"""Read-only observed-sky check of a diagnostic LG pair; no IC selection."""

import json
import re
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord


ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path("/gpfs/kjhan/CF4/diagnostics/cf4_lg_s40349_zoom_l19_z0_newgalfinder_local_audit_v1.json")


def sky_unit_sg(ra_deg, dec_deg):
    direction = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg,
                         distance=1 * u.kpc, frame="icrs")
    return direction.supergalactic.cartesian.xyz.value


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def main():
    audit = json.loads(AUDIT.read_text())
    contract = json.loads((ROOT / "config/cf4_lg_observation_contract_v1.json").read_text())
    p1 = json.loads((ROOT / "config/p1_targets_v2_observer.json").read_text())
    ic = json.loads((ROOT / "config/cf4_lg_s40349_zoom_ic_l12_decision_v2.json").read_text())
    namelist = (ROOT / "config/ramses_cf4_lg_b3292_s40349_zoom_l19_z0_v1.nml").read_text()
    match = re.search(r"(?m)^h0\s*=\s*([0-9.]+)", namelist)
    if p1["coordinate_frame"] != "de Vaucouleurs supergalactic Cartesian, observer at box centre" or not match:
        raise ValueError("unverified coordinate frame or h0")
    h = float(match.group(1))
    if not np.isclose(h, p1["cosmology_h"], rtol=0, atol=1e-12):
        raise ValueError("simulation and observation h differ")
    pair = audit["candidate_pair"]
    pos = np.asarray(pair["positions_mpc_h"], float)
    if not np.isclose(np.linalg.norm(pos[1] - pos[0]), pair["separation_mpc_h"], atol=1e-8):
        raise ValueError("pair coordinate/separation mismatch")
    target = contract["measurements"]["M31"]
    direction = sky_unit_sg(target["ra_deg"], target["dec_deg"])
    distance_mpc = 10 ** ((target["value"][0] - 10) / 5) / 1000
    box_size = (ic["checks"]["parent_grid"] *
                ic["checks"]["global_base_spacing_cMpc_h"])
    if not np.isclose(box_size, 384.0):
        raise ValueError("unexpected observer-centred box size")
    observer = np.full(3, box_size / 2)
    assignments = []
    for mw in (0, 1):
        relative = pos[1 - mw] - pos[mw]
        assignments.append({
            "MW_host": int(pair["halo_i"] if mw == 0 else pair["halo_j"]),
            "M31_host": int(pair["halo_j"] if mw == 0 else pair["halo_i"]),
            "M31_sky_separation_deg": angle_deg(relative, direction),
            "MW_offset_from_fixed_observer_mpc_h": float(np.linalg.norm(pos[mw] - observer)),
            "MW_M31_separation_physical_mpc": float(np.linalg.norm(relative) / h),
        })
    print(json.dumps({
        "status": "DIAGNOSTIC_NO_GO_NOT_A_LIKELIHOOD",
        "scope": "One frozen trace-only pair; no random-seed or role selection",
        "frame": p1["coordinate_frame"],
        "h": h,
        "observed_M31_distance_physical_mpc": distance_mpc,
        "observed_M31_direction_SG": direction.tolist(),
        "assignments": assignments,
        "limits": "MW centre versus solar position, halo COM versus galaxy centre, model discrepancy and covariance are not calibrated. No M33 identity is assigned.",
    }, indent=2))


if __name__ == "__main__":
    main()
