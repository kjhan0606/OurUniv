#!/usr/bin/env python3
"""Definitive HOP/M200c gate for the p3429/s5108 CF4 LG zoom pilot.

This replaces the legacy cr6/e19-specific verdict path.  It identifies the
descendant of the frozen P2 screen pair, measures direct spherical-overdensity
properties, checks low-resolution contamination, and searches a deliberately
unmerged HOP peak catalog for an M33 analogue.  The large-scale P1 environment
is reported as a separate, still-required recentered gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cf4_zoom_z0_gate import (  # noqa: E402
    MPC_CM,
    MSUN_G,
    catalog_from_hop_tags,
    collect_regions,
    min_image,
    particle_files,
    read_info,
    scan_mass_species,
    spherical_overdensity,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_extended_config(path: Path, seen: set[Path] | None = None) -> dict:
    """Resolve the frozen P2 config inheritance chain and verify every SHA."""
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise RuntimeError(f"cyclic P2 config inheritance at {path}")
    seen.add(path)
    row = json.loads(path.read_text())
    parent_name = row.get("extends")
    if not parent_name:
        return row
    parent = Path(parent_name)
    if not parent.is_absolute():
        parent = ROOT / parent
    expected = row.get("extends_sha256")
    actual = file_hash(parent)
    if expected and actual != expected:
        raise RuntimeError(
            f"P2 parent-config SHA mismatch for {parent}: {actual} != {expected}")
    return deep_merge(load_extended_config(parent, seen), row)


def load_screen_pair(result_path: Path, halo_path: Path, box: float) -> dict:
    result = json.loads(result_path.read_text())
    pair = result["best_pair"]
    with np.load(halo_path, allow_pickle=False) as data:
        positions = np.asarray(data["halo_pos"], np.float64)
        masses = np.asarray(data["halo_mass"], np.float64)
    i, j = int(pair["halo_i"]), int(pair["halo_j"])
    dr = min_image(positions[i] - positions[j], box)
    midpoint = np.mod(positions[j] + 0.5 * dr, box)
    recorded = np.asarray(pair["midpoint_mpc_h"], np.float64)
    if np.linalg.norm(min_image(midpoint - recorded, box)) > 1e-3:
        raise RuntimeError("P2 result midpoint disagrees with its halo catalog")
    return {
        "parent_seed": int(result["parent_seed"]),
        "small_scale_seed": int(result["small_scale_seed"]),
        "indices": [i, j],
        "centers_mpc_h": positions[[i, j]].tolist(),
        "masses_fof_msun_h": masses[[i, j]].tolist(),
        "midpoint_mpc_h": midpoint.tolist(),
        "separation_mpc_h": float(np.linalg.norm(dr)),
        "screen_result": str(result_path.resolve()),
        "screen_halos": str(halo_path.resolve()),
    }


def select_screen_descendant_pair(cat: dict, screen: dict, box: float) -> tuple[dict, list]:
    """Select a broad HOP host pair by continuity with the frozen P2 pair.

    These broad limits locate the intended object only.  They are not the
    definitive pass/fail cuts, which are applied to direct M200c profiles.
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(cat["pos"], np.float64)
    mass = np.asarray(cat["mass"], np.float64)
    screen_mid = np.asarray(screen["midpoint_mpc_h"], np.float64)
    near = np.linalg.norm(min_image(pos - screen_mid, box), axis=1) <= 5.0
    eligible = near & (mass >= 1e11) & (mass <= 1e13)
    indices = np.flatnonzero(eligible)
    if len(indices) < 2:
        raise RuntimeError("fewer than two broad-mass HOP hosts near the P2 midpoint")
    tree = cKDTree(pos[indices], boxsize=box)
    pairs = tree.query_pairs(1.5, output_type="ndarray")
    rows = []
    screen_centers = np.asarray(screen["centers_mpc_h"], np.float64)
    screen_sep = float(screen["separation_mpc_h"])
    for aa, bb in pairs:
        i, j = int(indices[aa]), int(indices[bb])
        dr = min_image(pos[i] - pos[j], box)
        sep = float(np.linalg.norm(dr))
        if sep < 0.2:
            continue
        midpoint = np.mod(pos[j] + 0.5 * dr, box)
        direct = (
            np.linalg.norm(min_image(pos[i] - screen_centers[0], box))
            + np.linalg.norm(min_image(pos[j] - screen_centers[1], box))
        )
        swapped = (
            np.linalg.norm(min_image(pos[i] - screen_centers[1], box))
            + np.linalg.norm(min_image(pos[j] - screen_centers[0], box))
        )
        continuity = min(direct, swapped)
        cost = (
            continuity
            + np.linalg.norm(min_image(midpoint - screen_mid, box))
            + abs(sep - screen_sep)
        )
        rows.append({
            "i": i,
            "j": j,
            "m1_fof_msun_h": float(mass[i]),
            "m2_fof_msun_h": float(mass[j]),
            "sep_fof_mpc_h": sep,
            "midpoint_fof_mpc_h": midpoint.tolist(),
            "screen_continuity_cost_mpc_h": float(cost),
            "screen_member_offset_sum_mpc_h": float(continuity),
            "screen_midpoint_offset_mpc_h": float(
                np.linalg.norm(min_image(midpoint - screen_mid, box))),
        })
    if not rows:
        raise RuntimeError("no distinct HOP host pair within 1.5 Mpc/h near P2 midpoint")
    rows.sort(key=lambda row: row["screen_continuity_cost_mpc_h"])
    return rows[0], rows


def nearby_third_halo(cat: dict, pair_indices: tuple[int, int], midpoint: np.ndarray,
                      radius: float, mass_limit: float, box: float) -> dict:
    dist = np.linalg.norm(min_image(cat["pos"] - midpoint, box), axis=1)
    keep = dist <= radius
    keep[list(pair_indices)] = False
    indices = np.flatnonzero(keep)
    if len(indices):
        order = indices[np.argsort(-cat["mass"][indices])]
        most = int(order[0])
        candidates = [{
            "catalog_index": int(k),
            "mass_fof_msun_h": float(cat["mass"][k]),
            "distance_from_midpoint_mpc_h": float(dist[k]),
            "center_mpc_h": np.asarray(cat["pos"][k], float).tolist(),
        } for k in order[:20]]
        maximum = float(cat["mass"][most])
    else:
        candidates = []
        maximum = 0.0
    return {
        "radius_mpc_h": float(radius),
        "mass_limit_msun_h": float(mass_limit),
        "maximum_third_mass_fof_msun_h": maximum,
        "passed": bool(maximum <= mass_limit),
        "candidates": candidates,
    }


def evaluate_m33_peaks(peaks: dict, m31_center: np.ndarray, m31_mass: float,
                       gate: dict, box: float) -> dict:
    mass = np.asarray(peaks["mass"], np.float64)
    dist = np.linalg.norm(min_image(peaks["pos"] - m31_center, box), axis=1)
    mlo, mhi = map(float, gate["mass_range_msun_h"])
    dlo, dhi = map(float, gate["m31_separation_range_mpc_h"])
    fraction = mass / m31_mass
    keep = ((mass >= mlo) & (mass <= mhi) & (dist >= dlo) & (dist <= dhi)
            & (fraction <= float(gate["maximum_mass_fraction_of_m31"])))
    indices = np.flatnonzero(keep)
    target_mass = 1e11
    target_distance = 0.20
    score = (np.abs(np.log10(mass[indices] / target_mass))
             + np.abs(dist[indices] - target_distance) / 0.15) if len(indices) else []
    if len(indices):
        indices = indices[np.argsort(score)]
    candidates = []
    for k in indices[:20]:
        row = {
            "catalog_index": int(k),
            "group_id": int(peaks["group_id"][k]),
            "mass_hop_peak_msun_h": float(mass[k]),
            "mass_fraction_of_m31": float(fraction[k]),
            "m31_separation_mpc_h": float(dist[k]),
            "center_mpc_h": np.asarray(peaks["pos"][k], float).tolist(),
            "npart": int(peaks["n"][k]),
        }
        if "contamination_fof" in peaks:
            row["contamination_hop_peak"] = float(peaks["contamination_fof"][k])
        candidates.append(row)
    return {
        "passed": bool(candidates),
        "status": "unmerged_hop_peak_test",
        "definition": gate,
        "candidate": candidates[0] if candidates else None,
        "candidates": candidates,
        "caveat": (
            "An unmerged HOP density basin is not a bound-subhalo catalog; "
            "the promoted L13/L21 run must repeat this test and may add PHEW."
        ),
    }


def evaluate_core_checks(profiles: list[dict], pair: dict, third: dict,
                         target: dict) -> dict:
    masses = [float(row["m200c_msun_h"]) for row in profiles]
    mlo, mhi = map(float, target["pair_member_mass_range_msun_h"])
    slo, shi = map(float, target["pair_separation_range_mpc_h"])
    vlo, vhi = map(float, target["total_radial_velocity_range_km_s"])
    ratio = max(masses) / min(masses)
    return {
        "mass_1": bool(mlo <= masses[0] <= mhi),
        "mass_2": bool(mlo <= masses[1] <= mhi),
        "mass_ratio": bool(ratio <= float(target["pair_mass_ratio_max"])),
        "separation": bool(slo <= pair["separation_mpc_h"] <= shi),
        "total_radial_velocity": bool(vlo <= pair["vtotal_kms"] <= vhi),
        "tangential_velocity": bool(
            pair["vtan_kms"] <= float(target["maximum_tangential_velocity_km_s"])),
        "midpoint": bool(
            pair["midpoint_offset_mpc_h"]
            <= float(target["pair_midpoint_max_offset_mpc_h"])),
        "massive_halo_isolation": bool(
            pair["massive_halo_isolation_mpc_h"]
            >= float(target["isolation_radius_mpc_h"])),
        "third_halo": bool(third["passed"]),
    }


def write_plot(path: Path, cat: dict, peaks: dict, pair: dict, profiles: list[dict],
               m33: dict, box: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    midpoint = np.asarray(pair["midpoint_mpc_h"])
    delta = min_image(cat["pos"] - midpoint, box)
    keep = ((np.abs(delta[:, 0]) < 5) & (np.abs(delta[:, 1]) < 5)
            & (cat["mass"] > 1e9))
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    sc = ax.scatter(delta[keep, 0], delta[keep, 1],
                    c=np.log10(cat["mass"][keep]), s=10, cmap="viridis",
                    vmin=9, vmax=13.5, alpha=0.75)
    for index, color in enumerate(("tab:red", "tab:orange")):
        center = np.asarray(profiles[index]["center_mpc_h"])
        d = min_image(center - midpoint, box)
        ax.scatter(d[0], d[1], marker="*", s=190, color=color,
                   edgecolor="black", label=f"LG member {index + 1}")
    if m33["candidate"] is not None:
        center = np.asarray(m33["candidate"]["center_mpc_h"])
        d = min_image(center - midpoint, box)
        ax.scatter(d[0], d[1], marker="^", s=120, color="cyan",
                   edgecolor="black", label="M33 HOP peak")
    ax.scatter(0, 0, marker="+", s=100, color="black", label="LG midpoint")
    ax.set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal",
           xlabel=r"$x-x_{LG}$ [$h^{-1}$ Mpc]",
           ylabel=r"$y-y_{LG}$ [$h^{-1}$ Mpc]",
           title="p3429/s5108 z=0 HOP/M200c gate")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}(M_{HOP}/[M_\odot/h])$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "/gpfs/kjhan/CF4/ramses/lg_p3429_s5108_l12_l19_z0_v1/output_00008"))
    parser.add_argument("--work", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1"))
    parser.add_argument("--hop-work", type=Path, default=None)
    parser.add_argument("--p2-config", type=Path,
                        default=ROOT / "config/p2_lg_targets_v11_bgc_inverse_peak.json")
    parser.add_argument("--p2-result", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_p2_v3_inverse/"
        "result_p3429_s5108.json"))
    parser.add_argument("--p2-halos", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_p2_v3_inverse/"
        "halos_p3429_s5108.npz"))
    parser.add_argument("--box", type=float, default=384.0)
    parser.add_argument("--reuse-catalog", action="store_true")
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    hop_work = args.hop_work or args.work / "hop_work"
    standard_tag = hop_work / "grp00010.tag"
    peak_tag = hop_work / "peaks00010.tag"
    if not standard_tag.is_file() or not peak_tag.is_file():
        parser.error(f"HOP products are incomplete under {hop_work}")

    config = load_extended_config(args.p2_config)
    target = config["definitive_zoom_gate"]
    m33_target = config["m33_subpeak_gate"]
    info = read_info(args.output)
    files = particle_files(args.output)
    if len(files) != 16:
        parser.error(f"expected 16 RAMSES particle files, found {len(files)}")
    mass_unit = info["unit_d"] * info["unit_l"] ** 3 / MSUN_G * (info["H0"] / 100.0)
    velocity_unit = info["unit_l"] / info["unit_t"] / 1e5
    ntotal, species = scan_mass_species(files, mass_unit)
    fine_mass_code = species[0]["mass_code"]
    fine_mass = species[0]["mass_msun_h"]
    print(f"[meta] a={info['aexp']:.9f} N={ntotal:,} mp={fine_mass:.6e}", flush=True)

    standard_path = args.work / "hop_catalog_exact.npz"
    peaks_path = args.work / "hop_peaks_unmerged_exact.npz"
    if args.reuse_catalog and standard_path.is_file() and peaks_path.is_file():
        with np.load(standard_path, allow_pickle=False) as data:
            cat = {key: data[key] for key in data.files}
        with np.load(peaks_path, allow_pickle=False) as data:
            peaks = {key: data[key] for key in data.files}
    else:
        cat = catalog_from_hop_tags(
            args.output, standard_tag, args.box, mass_unit, velocity_unit,
            fine_mass_code)
        np.savez(standard_path, **cat)
        peaks = catalog_from_hop_tags(
            args.output, peak_tag, args.box, mass_unit, velocity_unit,
            fine_mass_code)
        np.savez(peaks_path, **peaks)
    print(f"[catalog] hosts={len(cat['mass']):,} peaks={len(peaks['mass']):,}", flush=True)

    screen = load_screen_pair(args.p2_result, args.p2_halos, args.box)
    selected, pair_candidates = select_screen_descendant_pair(cat, screen, args.box)
    i, j = int(selected["i"]), int(selected["j"])
    screen_mid = np.asarray(screen["midpoint_mpc_h"])
    environment_spheres = [(screen_mid, 1.0)]
    chunks, env = collect_regions(
        files, [cat["pos"][i], cat["pos"][j]], [1.0, 1.0], args.box,
        velocity_unit, mass_unit, environment_spheres, fine_mass)
    profiles = [spherical_overdensity(
        chunks[k], cat["pos"][[i, j][k]], args.box, fine_mass) for k in range(2)]

    c1, c2 = (np.asarray(row["center_mpc_h"]) for row in profiles)
    dr = min_image(c1 - c2, args.box)
    separation = float(np.linalg.norm(dr))
    rhat = dr / separation
    v1, v2 = (np.asarray(row["velocity_kms"]) for row in profiles)
    dv = v1 - v2
    vrad_pec = float(np.dot(dv, rhat))
    e2 = info["omega_m"] / info["aexp"] ** 3 + info["omega_l"]
    hubble_term = 100.0 * info["aexp"] * math.sqrt(e2)
    vtotal = vrad_pec + hubble_term * separation
    vtan = float(np.linalg.norm(dv - vrad_pec * rhat))
    midpoint = np.mod(c2 + 0.5 * dr, args.box)
    box_center = np.full(3, args.box / 2.0)

    massive = cat["mass"] >= float(target["isolation_mass_threshold_msun_h"])
    massive[[i, j]] = False
    massive_distance = np.linalg.norm(min_image(cat["pos"][massive] - midpoint,
                                                args.box), axis=1)
    isolation = float(massive_distance.min()) if len(massive_distance) else 99.0
    pair = {
        **selected,
        "separation_mpc_h": separation,
        "midpoint_mpc_h": midpoint.tolist(),
        "midpoint_offset_mpc_h": float(
            np.linalg.norm(min_image(midpoint - box_center, args.box))),
        "screen_midpoint_offset_after_m200c_mpc_h": float(
            np.linalg.norm(min_image(midpoint - screen_mid, args.box))),
        "vrad_pec_kms": vrad_pec,
        "vtotal_kms": vtotal,
        "vtan_kms": vtan,
        "hubble_term_kms_per_mpc_h": hubble_term,
        "massive_halo_isolation_mpc_h": isolation,
        "m1_m200c_msun_h": profiles[0]["m200c_msun_h"],
        "m2_m200c_msun_h": profiles[1]["m200c_msun_h"],
        "m200c_mass_ratio": float(max(row["m200c_msun_h"] for row in profiles)
                                    / min(row["m200c_msun_h"] for row in profiles)),
    }
    third = nearby_third_halo(
        cat, (i, j), midpoint, 2.5,
        min(row["m200c_msun_h"] for row in profiles), args.box)
    m31_index = int(np.argmax([row["m200c_msun_h"] for row in profiles]))
    m33 = evaluate_m33_peaks(
        peaks, np.asarray(profiles[m31_index]["center_mpc_h"]),
        float(profiles[m31_index]["m200c_msun_h"]), m33_target, args.box)
    core_checks = evaluate_core_checks(profiles, pair, third, target)
    contamination_limit = float(target["maximum_contaminant_mass_fraction_within_r200c"])
    contamination_checks = {
        "halo_1": bool(profiles[0]["contaminant_mass_fraction_r200c"]
                       < contamination_limit),
        "halo_2": bool(profiles[1]["contaminant_mass_fraction_r200c"]
                       < contamination_limit),
    }
    phase_checks = {
        "screen_region_contains_finest_particles": bool(env[0]["fine_count"] > 0),
        "screen_seed_matches": bool(screen["parent_seed"] == 3429
                                    and screen["small_scale_seed"] == 5108),
    }
    p2b_pass = bool(all(core_checks.values()) and all(contamination_checks.values())
                    and m33["passed"] and all(phase_checks.values()))
    result = {
        "schema": "ouruniv-cf4-zoom-z0-gate-v2",
        "snapshot": str(args.output.resolve()),
        "metadata": {
            **info,
            "box_mpc_h": args.box,
            "npart_total": ntotal,
            "mass_species": species,
            "velocity_unit_kms": velocity_unit,
            "p2_config": str(args.p2_config.resolve()),
            "p2_config_sha256": file_hash(args.p2_config),
            "p2_resolved_schema": config["schema"],
            "p2_result_sha256": file_hash(args.p2_result),
            "hop_binary_sha256": file_hash(Path(
                "/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses/hop")),
        },
        "screen_pair": screen,
        "catalog": {
            "halo_finder": target["halo_finder"],
            "n_regrouped_hosts_ge20": int(len(cat["mass"])),
            "n_unmerged_peaks_ge20": int(len(peaks["mass"])),
        },
        "pair": pair,
        "halo_profiles": profiles,
        "third_halo": third,
        "m33": m33,
        "screen_region_r1": env[0],
        "checks": {
            "core": core_checks,
            "contamination": contamination_checks,
            "phase": phase_checks,
        },
        "environment": {
            "status": "pending_recentered_p1_gate",
            "required": bool(target["require_p1_environment_after_recentering"]),
            "reason": "The LG midpoint is only known after the direct M200c centers.",
        },
        "verdict": {
            "core_lg_gate": bool(all(core_checks.values())),
            "clean_zoom": bool(all(contamination_checks.values())),
            "m33_subpeak_gate": bool(m33["passed"]),
            "phase_preserved": bool(all(phase_checks.values())),
            "p2b_lg_gate": p2b_pass,
            "environment_gate": None,
            "overall": None,
        },
        "pair_candidates_by_screen_continuity": pair_candidates[:20],
    }
    result_path = args.work / "gate_result_v2.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_plot(args.work / "z0_gate_v2.png", cat, peaks, pair, profiles, m33, args.box)
    print("[verdict]", json.dumps(result["verdict"]), flush=True)
    print(f"[done] {result_path}", flush=True)


if __name__ == "__main__":
    main()
