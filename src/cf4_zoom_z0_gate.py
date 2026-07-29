#!/usr/bin/env python3
"""Post-process the completed CF4 cr6/e19 RAMSES zoom at z=0.

The production binary was built with patch/cuRamses/output_part.f90.  Its
particle layout is

    8 header records; x,y,z; vx,vy,vz; mass; int64 id; int32 level;
    int8 ptype; float64 potential

and is therefore not readable by yt's stock RAMSES particle handler.

This script:
  1. extracts the finest-mass particles;
  2. runs the existing MPI OPFoF halo finder at b=0.2;
  3. finds/ranks Local-Group pairs using the same cuts as cf4_lg_screen3.py;
  4. streams the multi-mass snapshot around the selected objects to measure
     M200c, velocities, and low-resolution contamination;
  5. writes a machine-readable gate result and a compact diagnostic figure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

import numpy as np


MPC_CM = 3.0856775814913673e24
MSUN_G = 1.98847e33
RHO_CRIT = 2.77536627e11  # (Msun/h)/(Mpc/h)^3
VIRGO_SG = (102.9, -2.3, 16.5)  # SGL, SGB, distance in physical Mpc
VOID_D = 17.2  # h^-1 Mpc, convention used by cf4_lg_screen3.py


def _record(fh, dtype=None):
    """Read or skip one little-endian Fortran sequential record."""
    raw = fh.read(4)
    if len(raw) != 4:
        raise EOFError
    nbyte = struct.unpack("<i", raw)[0]
    if dtype is None:
        fh.seek(nbyte, os.SEEK_CUR)
        out = None
    else:
        dt = np.dtype(dtype)
        if nbyte % dt.itemsize:
            raise ValueError(f"record length {nbyte} is not divisible by {dt}")
        out = np.fromfile(fh, dtype=dt, count=nbyte // dt.itemsize)
    end = struct.unpack("<i", fh.read(4))[0]
    if end != nbyte:
        raise ValueError(f"Fortran record markers disagree: {nbyte} != {end}")
    return out


def _skip_header(fh):
    ncpu = int(_record(fh, "<i4")[0])
    ndim = int(_record(fh, "<i4")[0])
    npart = int(_record(fh, "<i4")[0])
    for _ in range(5):
        _record(fh)
    if ndim != 3:
        raise ValueError(f"expected ndim=3, found {ndim}")
    return ncpu, npart


def read_info(output: Path):
    text = (output / f"info_{output.name[-5:]}.txt").read_text()

    def val(key):
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([0-9.Ee+-]+)", text, re.M)
        if not m:
            raise KeyError(key)
        return float(m.group(1))

    return {k: val(k) for k in
            ("aexp", "H0", "omega_m", "omega_l", "unit_l", "unit_d", "unit_t")}


def particle_files(output: Path):
    nout = output.name[-5:]
    files = sorted(output.glob(f"part_{nout}.out*"))
    if not files:
        raise FileNotFoundError(f"no particle files under {output}")
    return files


def scan_mass_species(files, mass_unit):
    hist = {}
    ntotal = 0
    for path in files:
        with path.open("rb") as fh:
            _, n = _skip_header(fh)
            ntotal += n
            for _ in range(6):
                _record(fh)
            mass = _record(fh, "<f8")
        u, c = np.unique(mass, return_counts=True)
        for mm, nn in zip(u, c):
            hist[float(mm)] = hist.get(float(mm), 0) + int(nn)
    species = [
        {"mass_code": m, "mass_msun_h": m * mass_unit, "count": n}
        for m, n in sorted(hist.items())
    ]
    return ntotal, species


def extract_finest(files, fine_mass_code, box_mpc_h, velocity_unit):
    """Load only the finest mass species into contiguous float32 arrays."""
    # The exact masses are powers of eight, so equality is intentional here.
    nfine = 0
    for path in files:
        with path.open("rb") as fh:
            _, _ = _skip_header(fh)
            for _ in range(6):
                _record(fh)
            mass = _record(fh, "<f8")
            nfine += int(np.count_nonzero(mass == fine_mass_code))

    pos = np.empty((nfine, 3), np.float32)
    vel = np.empty((nfine, 3), np.float32)
    cursor = 0
    t0 = time.time()
    for ifile, path in enumerate(files, 1):
        with path.open("rb") as fh:
            _, n = _skip_header(fh)
            xyz = [_record(fh, "<f8") for _ in range(3)]
            vvv = [_record(fh, "<f8") for _ in range(3)]
            mass = _record(fh, "<f8")
        keep = mass == fine_mass_code
        nk = int(keep.sum())
        sl = slice(cursor, cursor + nk)
        for j in range(3):
            pos[sl, j] = xyz[j][keep] * box_mpc_h
            vel[sl, j] = vvv[j][keep] * velocity_unit
        cursor += nk
        if ifile % 4 == 0:
            print(f"[extract] {ifile}/{len(files)} files, {cursor:,}/{nfine:,} fine "
                  f"particles ({time.time()-t0:.1f}s)", flush=True)
    if cursor != nfine:
        raise RuntimeError(f"fine particle count changed: {cursor} != {nfine}")
    return pos, vel


def catalog_from_hop_tags(output, tag_path, box, mass_unit, velocity_unit,
                          fine_mass_code):
    """Recompute HOP group properties without poshalo's coarse text rounding."""
    files = particle_files(output)
    with tag_path.open("rb") as fh:
        header = _record(fh, "<i4")
    ntags, ngroup = map(int, header)
    # The second Fortran record starts after the 16-byte first record and its
    # leading marker occupies bytes 16:20.
    tags = np.memmap(tag_path, dtype="<i4", mode="r", offset=20, shape=(ntags,))
    count = np.zeros(ngroup, np.int64)
    mass_sum = np.zeros(ngroup)
    fine_sum = np.zeros(ngroup)
    mom = np.zeros((ngroup, 3))
    circ_sin = np.zeros((ngroup, 3))
    circ_cos = np.zeros((ngroup, 3))
    cursor = 0
    for ifile, path in enumerate(files, 1):
        with path.open("rb") as fh:
            _, n = _skip_header(fh)
            xyz = np.stack([_record(fh, "<f8") for _ in range(3)], axis=1)
            vvv = np.stack([_record(fh, "<f8") for _ in range(3)], axis=1)
            mass_code = _record(fh, "<f8")
        group = np.asarray(tags[cursor:cursor + n])
        cursor += n
        keep = group >= 0
        gg = group[keep]
        mm = mass_code[keep]
        count += np.bincount(gg, minlength=ngroup)
        mass_sum += np.bincount(gg, weights=mm, minlength=ngroup)
        fine_sum += np.bincount(
            gg, weights=mm * (mm == fine_mass_code), minlength=ngroup)
        for axis in range(3):
            mom[:, axis] += np.bincount(
                gg, weights=mm * vvv[keep, axis], minlength=ngroup)
            angle = 2.0 * np.pi * xyz[keep, axis]
            circ_sin[:, axis] += np.bincount(
                gg, weights=mm * np.sin(angle), minlength=ngroup)
            circ_cos[:, axis] += np.bincount(
                gg, weights=mm * np.cos(angle), minlength=ngroup)
        if ifile % 4 == 0:
            print(f"[hop-cat] accumulated {ifile}/{len(files)} files", flush=True)
    if cursor != ntags:
        raise RuntimeError(f"HOP tag count mismatch: {cursor} != {ntags}")
    valid = (count >= 20) & (mass_sum > 0)
    gid = np.flatnonzero(valid)
    angle = np.mod(np.arctan2(circ_sin[valid], circ_cos[valid]), 2.0 * np.pi)
    pos = angle * (box / (2.0 * np.pi))
    cat = {
        "group_id": gid,
        "n": count[valid],
        "mass": mass_sum[valid] * mass_unit,
        "contamination_fof": 1.0 - fine_sum[valid] / mass_sum[valid],
        "pos": pos,
        "vel": mom[valid] / mass_sum[valid, None] * velocity_unit,
    }
    order = np.argsort(-cat["mass"])
    return {key: value[order] for key, value in cat.items()}


def run_hop_catalog(output, work, box, mass_unit, velocity_unit, fine_mass_code,
                    reuse=False):
    """Run the user's serial multi-mass HOP and parse poshalo's group catalog."""
    hop_dir = Path("/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses")
    hop_bin = hop_dir / "hop"
    regroup_bin = hop_dir / "regroup"
    poshalo_bin = hop_dir / "poshalo"
    for exe in (hop_bin, regroup_bin, poshalo_bin):
        if not (exe.is_file() and os.access(exe, os.X_OK)):
            raise FileNotFoundError(f"missing executable: {exe}")

    work.mkdir(parents=True, exist_ok=True)
    pos_path = work / "grp00010.pos"
    tag_path = work / "grp00010.tag"
    if not (reuse and tag_path.exists()):
        prefix = output / f"part_{output.name[-5:]}.out"
        commands = [
            ([str(hop_bin), "-in", str(prefix), "-p", "1.", "-o", "hop00010"],
             "hop.log"),
            ([str(regroup_bin), "-root", "hop00010", "-douter", "80.",
              "-dsaddle", "200.", "-dpeak", "240.", "-f77", "-o", "grp00010"],
             "regroup.log"),
            ([str(poshalo_bin), "-inp", str(output), "-pre", "grp00010",
              "-cut", f"{fine_mass_code * 1.01:.18e}"], "poshalo.log"),
        ]
        for command, logname in commands:
            print("[hop]", " ".join(command), flush=True)
            with (work / logname).open("w") as log:
                run = subprocess.run(command, cwd=work, stdout=log,
                                     stderr=subprocess.STDOUT, text=True)
            if run.returncode:
                tail = (work / logname).read_text(errors="replace")[-4000:]
                raise RuntimeError(f"{command[0]} failed ({run.returncode})\n{tail}")
    if not tag_path.exists():
        raise FileNotFoundError(tag_path)
    return catalog_from_hop_tags(
        output, tag_path, box, mass_unit, velocity_unit, fine_mass_code)


def sgdir(lon_deg, lat_deg):
    lon, lat = np.radians([lon_deg, lat_deg])
    return np.array([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)])


def min_image(dx, box):
    return dx - box * np.round(dx / box)


def pair_score(sep, m1, m2, rmid, vtotal, vtan, pair_to_void):
    """The screening score, retained only for ranking (not pass/fail)."""
    return (abs(np.log10(m1) - 12.1) + abs(np.log10(m2) - 12.1)
            + 0.7 * abs(sep - 0.57) / 0.50
            + 0.5 * abs(vtotal + 110.0) / 80.0
            + 0.25 * vtan / 60.0
            + 0.15 * rmid
            + 0.10 * abs(pair_to_void - VOID_D) / 5.0)


def find_pairs(cat, box, hubble_term, observer, void, rmax=8.0,
               max_contamination=None):
    from scipy.spatial import cKDTree

    mass = cat["mass"]
    pos = cat["pos"]
    vel = cat["vel"]
    robs = np.linalg.norm(min_image(pos - observer, box), axis=1)
    eligible = (mass > 5e11) & (mass < 4e12) & (robs < rmax)
    if max_contamination is not None and "contamination_fof" in cat:
        eligible &= cat["contamination_fof"] < max_contamination
    mw = np.flatnonzero(eligible)
    if len(mw) < 2:
        return []
    tree = cKDTree(pos[mw])
    local_pairs = tree.query_pairs(1.2, output_type="ndarray")
    big = pos[mass > 5e12]
    bigtree = cKDTree(big) if len(big) else None
    rows = []
    for aa, bb in local_pairs:
        i, j = int(mw[aa]), int(mw[bb])
        dr = min_image(pos[i] - pos[j], box)
        sep = float(np.linalg.norm(dr))
        if sep <= 0.3:
            continue
        mid = np.mod(pos[j] + 0.5 * dr, box)
        isolation = float(bigtree.query(mid)[0]) if bigtree is not None else 99.0
        if isolation < 3.0:
            continue
        rhat = dr / sep
        dv = vel[i] - vel[j]
        vrad_pec = float(np.dot(dv, rhat))
        vtotal = vrad_pec + hubble_term * sep
        vtan = float(np.linalg.norm(dv - vrad_pec * rhat))
        rmid = float(np.linalg.norm(min_image(mid - observer, box)))
        p2void = float(np.linalg.norm(min_image(mid - void, box)))
        rows.append({
            "i": i, "j": j, "sep_mpc_h": sep,
            "m1_fof_msun_h": float(mass[i]), "m2_fof_msun_h": float(mass[j]),
            "midpoint_mpc_h": mid.tolist(), "r_mid_mpc_h": rmid,
            "vrad_pec_kms": vrad_pec, "vtotal_kms": vtotal, "vtan_kms": vtan,
            "isolation_mpc_h": isolation, "pair_to_void_mpc_h": p2void,
            "score": pair_score(sep, mass[i], mass[j], rmid, vtotal, vtan, p2void),
        })
    rows.sort(key=lambda x: x["score"])
    return rows


def collect_regions(files, centers, radii, box, velocity_unit, mass_unit,
                    environment_spheres, fine_particle_mass):
    """Stream all mass species, collecting object profiles and environment sums."""
    chunks = [{"pos": [], "vel": [], "mass": []} for _ in centers]
    env = [{"mass": 0.0, "fine_mass": 0.0, "momentum": np.zeros(3),
            "count": 0, "fine_count": 0} for _ in environment_spheres]
    for ifile, path in enumerate(files, 1):
        with path.open("rb") as fh:
            _, _ = _skip_header(fh)
            xyz = np.stack([_record(fh, "<f8") for _ in range(3)], axis=1) * box
            vvv = np.stack([_record(fh, "<f8") for _ in range(3)], axis=1) * velocity_unit
            mass = _record(fh, "<f8") * mass_unit
        for k, (center, radius) in enumerate(zip(centers, radii)):
            dr = min_image(xyz - center, box)
            keep = np.einsum("ij,ij->i", dr, dr) < radius * radius
            if np.any(keep):
                chunks[k]["pos"].append(xyz[keep])
                chunks[k]["vel"].append(vvv[keep])
                chunks[k]["mass"].append(mass[keep])
        for k, (center, radius) in enumerate(environment_spheres):
            dr = min_image(xyz - center, box)
            keep = np.einsum("ij,ij->i", dr, dr) < radius * radius
            if np.any(keep):
                mm = mass[keep]
                env[k]["mass"] += float(mm.sum())
                isfine = mm < fine_particle_mass * 1.01
                env[k]["fine_mass"] += float(mm[isfine].sum())
                env[k]["momentum"] += (vvv[keep] * mm[:, None]).sum(axis=0)
                env[k]["count"] += int(keep.sum())
                env[k]["fine_count"] += int(isfine.sum())
        if ifile % 4 == 0:
            print(f"[profile] streamed {ifile}/{len(files)} files", flush=True)
    for chunk in chunks:
        for key in chunk:
            shape = (0, 3) if key in ("pos", "vel") else (0,)
            chunk[key] = np.concatenate(chunk[key]) if chunk[key] else np.empty(shape)
    for row in env:
        row["velocity_kms"] = (row["momentum"] / row["mass"]).tolist() if row["mass"] else [0.0] * 3
        row["contaminant_mass_fraction"] = (
            1.0 - row["fine_mass"] / row["mass"] if row["mass"] else None)
        row.pop("momentum")
    return chunks, env


def shrinking_center(pos, mass, initial, box, fine_particle_mass):
    center = np.asarray(initial, float).copy()
    radius = 0.30
    fine = mass < fine_particle_mass * 1.01
    for _ in range(12):
        dr = min_image(pos - center, box)
        keep = fine & (np.einsum("ij,ij->i", dr, dr) < radius * radius)
        if keep.sum() < 100:
            break
        center = np.mod(center + dr[keep].mean(axis=0), box)
        radius *= 0.75
        if radius < 0.025:
            break
    return center


def spherical_overdensity(chunk, initial, box, fine_particle_mass,
                          rho_crit=RHO_CRIT):
    pos, vel, mass = chunk["pos"], chunk["vel"], chunk["mass"]
    center = shrinking_center(pos, mass, initial, box, fine_particle_mass)
    dr = min_image(pos - center, box)
    rad = np.linalg.norm(dr, axis=1)
    order = np.argsort(rad)
    rr = rad[order]
    mm = mass[order]
    cum = np.cumsum(mm)
    with np.errstate(divide="ignore", invalid="ignore"):
        overdensity = cum / (4.0 * np.pi * rr ** 3 / 3.0) / rho_crit
    valid = np.flatnonzero((rr > 0) & (overdensity >= 200.0))
    if not len(valid):
        raise RuntimeError("could not find an enclosed 200 rho_crit region")
    k = int(valid[-1])
    r200 = float(rr[k])
    inside = rad <= r200
    m200 = float(mass[inside].sum())
    velocity = np.average(vel[inside], axis=0, weights=mass[inside])
    coarse = mass[inside] > fine_particle_mass * 1.01
    coarse_all = mass > fine_particle_mass * 1.01
    return {
        "center_mpc_h": center.tolist(),
        "r200c_mpc_h": r200,
        "m200c_msun_h": m200,
        "velocity_kms": velocity.tolist(),
        "n_inside": int(inside.sum()),
        "fine_particle_mass_msun_h": fine_particle_mass,
        "contaminant_count_r200c": int(coarse.sum()),
        "contaminant_mass_fraction_r200c": (
            float(mass[inside][coarse].sum() / m200) if np.any(coarse) else 0.0),
        "nearest_contaminant_mpc_h": (
            float(rad[coarse_all].min()) if np.any(coarse_all) else None),
    }


def write_plot(path, cat, pair, profiles, observer, virgo_target, virgo_index, box):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cat["pos"]
    m = cat["mass"]
    d = min_image(p - observer, box)
    keep = (np.abs(d[:, 0]) < 16) & (np.abs(d[:, 1]) < 16) & (m > 1e9)
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    sc = ax.scatter(d[keep, 0], d[keep, 1], c=np.log10(m[keep]), s=8,
                    cmap="viridis", vmin=9, vmax=14.8, alpha=0.75)
    ax.scatter(0, 0, marker="+", s=100, color="black", label="box centre")
    for k, color in enumerate(("tab:red", "tab:orange")):
        c = np.asarray(profiles[k]["center_mpc_h"])
        dc = min_image(c - observer, box)
        ax.scatter(dc[0], dc[1], marker="*", s=180, color=color, edgecolor="k",
                   label=f"diagnostic pair halo {k+1}")
    vt = min_image(virgo_target - observer, box)
    ax.scatter(vt[0], vt[1], marker="x", s=100, color="magenta", label="Virgo target")
    if virgo_index is not None:
        vh = min_image(cat["pos"][virgo_index] - observer, box)
        ax.scatter(vh[0], vh[1], marker="D", s=70, facecolor="none",
                   edgecolor="magenta", label="Virgo halo")
    ax.set(xlabel=r"$x-x_{\rm obs}$ [$h^{-1}$ Mpc]",
           ylabel=r"$y-y_{\rm obs}$ [$h^{-1}$ Mpc]",
           xlim=(-16, 16), ylim=(-16, 16),
           title="CF4 cr6/e19 zoom: z=0 HOP halos")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}(M_{\rm HOP}/[M_\odot/h])$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path(
        "/gpfs/kjhan/Hydro/CF4_LG/zoom_cr6_e19_L14_s20260717/output_00010"))
    ap.add_argument("--work", type=Path, default=Path("recon/zoom_z0_gate"))
    ap.add_argument("--box", type=float, default=384.0, help="comoving box [Mpc/h]")
    ap.add_argument("--screen", type=Path,
                    default=Path("recon/screen3b_halos_cr6_e19.npz"))
    ap.add_argument("--nrank", type=int, default=16)
    ap.add_argument("--nfile", type=int, default=32)
    ap.add_argument("--halo-finder", choices=("hop", "opfof"), default="hop")
    ap.add_argument("--reuse-catalog", action="store_true")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    info = read_info(args.output)
    files = particle_files(args.output)
    mass_unit = info["unit_d"] * info["unit_l"] ** 3 / MSUN_G * (info["H0"] / 100.0)
    velocity_unit = info["unit_l"] / info["unit_t"] / 1e5
    ntotal, species = scan_mass_species(files, mass_unit)
    fine_mass_code = species[0]["mass_code"]
    fine_mass = species[0]["mass_msun_h"]
    print(f"[meta] a={info['aexp']:.9f}, N={ntotal:,}, finest={species[0]['count']:,}, "
          f"mp={fine_mass:.6e} Msun/h, vunit={velocity_unit:.6f} km/s", flush=True)

    catalog_name = ("hop_catalog_exact.npz" if args.halo_finder == "hop"
                    else "opfof_catalog.npz")
    catalog_path = args.work / catalog_name
    if args.reuse_catalog and catalog_path.exists():
        z = np.load(catalog_path)
        cat = {k: z[k] for k in z.files}
        print(f"[halo] reused {catalog_path} ({len(cat['mass']):,} halos)", flush=True)
    elif args.halo_finder == "hop":
        hop_work = args.work / "hop_work"
        cat = run_hop_catalog(
            args.output, hop_work, args.box, mass_unit,
            velocity_unit, fine_mass_code,
            reuse=(args.reuse_catalog or (hop_work / "grp00010.tag").exists()))
        np.savez(catalog_path, **cat)
        print(f"[hop] {len(cat['mass']):,} regrouped halos", flush=True)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import opfof_io
        pos, vel = extract_finest(files, fine_mass_code, args.box, velocity_unit)
        opwork = args.work / "opfof_work"
        cat = opfof_io.fof_opfof(
            pos, vel, L=args.box, nx=2 ** 14, nstep=10,
            nfile=args.nfile, nid=args.nrank, outdir=str(opwork),
            m_particle=fine_mass, nmin=20, verbose=True)
        np.savez(catalog_path, n=cat["n"], mass=cat["mass"],
                 pos=cat["pos"], vel=cat["vel"])
        cat = {k: cat[k] for k in ("n", "mass", "pos", "vel")}
        del pos, vel

    observer = np.full(3, args.box / 2.0)
    virgo_target = observer + sgdir(*VIRGO_SG[:2]) * (VIRGO_SG[2] * info["H0"] / 100.0)
    void = observer + sgdir(78.0, 74.0) * VOID_D
    screen_pair = None
    if args.screen.exists():
        screen = np.load(args.screen)
        bi, bj = map(int, screen["best"][-2:])
        screen_pair = {
            "centers_mpc_h": [screen["halo_pos"][bi].astype(float).tolist(),
                              screen["halo_pos"][bj].astype(float).tolist()],
            "midpoint_mpc_h": (
                0.5 * (screen["halo_pos"][bi] + screen["halo_pos"][bj])
            ).astype(float).tolist(),
            "preliminary_metrics": screen["best"].astype(float).tolist(),
        }
    e2 = info["omega_m"] / info["aexp"] ** 3 + info["omega_l"]
    hubble_term = 100.0 * info["aexp"] * math.sqrt(e2)

    pairs = find_pairs(cat, args.box, hubble_term, observer, void, rmax=8.0,
                       max_contamination=0.01)
    search_radius = 8.0
    if not pairs:
        pairs = find_pairs(cat, args.box, hubble_term, observer, void, rmax=15.0,
                           max_contamination=0.01)
        search_radius = 15.0
    pair_clean_selection = True
    if not pairs:
        pairs = find_pairs(cat, args.box, hubble_term, observer, void, rmax=15.0)
        pair_clean_selection = False
    if not pairs:
        raise RuntimeError("no isolated 0.3--1.2 Mpc/h MW-mass pair within 15 Mpc/h")
    pair = pairs[0]
    print("[pair] best preliminary:", json.dumps(pair, indent=2), flush=True)

    dvir = np.linalg.norm(min_image(cat["pos"] - virgo_target, args.box), axis=1)
    near_virgo = np.flatnonzero(dvir < 4.0)
    virgo_index = (int(near_virgo[np.argmax(cat["mass"][near_virgo])])
                   if len(near_virgo) else None)

    i, j = pair["i"], pair["j"]
    object_centers = [cat["pos"][i], cat["pos"][j]]
    object_radii = [1.0, 1.0]
    if virgo_index is not None:
        object_centers.append(cat["pos"][virgo_index])
        object_radii.append(3.0)
    env_spheres = [
        (np.asarray(pair["midpoint_mpc_h"]), 2.5),
        ((cat["pos"][virgo_index] if virgo_index is not None else virgo_target), 3.0),
        (void, 8.0),
        (virgo_target, 8.0),
    ]
    if screen_pair is not None:
        env_spheres.append((np.asarray(screen_pair["midpoint_mpc_h"]), 1.0))
    chunks, env = collect_regions(files, object_centers, object_radii, args.box,
                                  velocity_unit, mass_unit, env_spheres, fine_mass)
    profiles = [spherical_overdensity(
                    chunks[k], object_centers[k], args.box, fine_mass)
                for k in range(2)]
    virgo_profile = (spherical_overdensity(
                        chunks[2], object_centers[2], args.box, fine_mass)
                     if virgo_index is not None else None)

    c1 = np.asarray(profiles[0]["center_mpc_h"])
    c2 = np.asarray(profiles[1]["center_mpc_h"])
    dr = min_image(c1 - c2, args.box)
    sep = float(np.linalg.norm(dr))
    rhat = dr / sep
    v1 = np.asarray(profiles[0]["velocity_kms"])
    v2 = np.asarray(profiles[1]["velocity_kms"])
    dv = v1 - v2
    vrad_pec = float(np.dot(dv, rhat))
    vtotal = vrad_pec + hubble_term * sep
    vtan = float(np.linalg.norm(dv - vrad_pec * rhat))
    midpoint = np.mod(c2 + 0.5 * dr, args.box)

    big = cat["pos"][cat["mass"] > 5e12]
    isolation = (float(np.linalg.norm(min_image(big - midpoint, args.box), axis=1).min())
                 if len(big) else 99.0)
    pair.update({
        "search_radius_mpc_h": search_radius,
        "selected_with_fof_contamination_below_1pct": pair_clean_selection,
        "sep_mpc_h": sep,
        "midpoint_mpc_h": midpoint.tolist(),
        "r_mid_mpc_h": float(np.linalg.norm(min_image(midpoint - observer, args.box))),
        "vrad_pec_kms": vrad_pec,
        "vtotal_kms": vtotal,
        "vtan_kms": vtan,
        "isolation_mpc_h": isolation,
        "m1_m200c_msun_h": profiles[0]["m200c_msun_h"],
        "m2_m200c_msun_h": profiles[1]["m200c_msun_h"],
    })

    lg_bulk = np.asarray(env[0]["velocity_kms"])
    virgo_bulk = np.asarray(env[1]["velocity_kms"])
    rhat_virgo = min_image(
        (cat["pos"][virgo_index] if virgo_index is not None else virgo_target) - midpoint,
        args.box)
    rhat_virgo /= np.linalg.norm(rhat_virgo)
    infall = float(np.dot(lg_bulk - virgo_bulk, rhat_virgo))
    void_expected = (info["omega_m"] * RHO_CRIT * 4.0 * np.pi * 8.0 ** 3 / 3.0)
    void_od = env[2]["mass"] / void_expected
    virgo_od = env[3]["mass"] / void_expected

    core_checks = {
        "mass_1": 5e11 <= profiles[0]["m200c_msun_h"] <= 4e12,
        "mass_2": 5e11 <= profiles[1]["m200c_msun_h"] <= 4e12,
        "separation": 0.3 <= sep <= 1.2,
        "approaching": vtotal < 0.0,
        "tangential": vtan < 100.0,
        "isolation": isolation >= 3.0,
    }
    contamination_checks = {
        "halo_1": profiles[0]["contaminant_mass_fraction_r200c"] < 0.01,
        "halo_2": profiles[1]["contaminant_mass_fraction_r200c"] < 0.01,
    }
    screen_group = None
    if screen_pair is not None:
        screen_mid = np.asarray(screen_pair["midpoint_mpc_h"])
        ds = np.linalg.norm(min_image(cat["pos"] - screen_mid, args.box), axis=1)
        kk = int(np.argmin(ds))
        screen_group = {
            "catalog_index": kk,
            "group_id": int(cat["group_id"][kk]) if "group_id" in cat else None,
            "distance_from_screen_midpoint_mpc_h": float(ds[kk]),
            "mass_fof_msun_h": float(cat["mass"][kk]),
            "npart": int(cat["n"][kk]),
            "contamination_fof": (
                float(cat["contamination_fof"][kk])
                if "contamination_fof" in cat else None),
            "center_mpc_h": cat["pos"][kk].tolist(),
        }
    phase_checks = {
        "pair_near_observer": pair["r_mid_mpc_h"] < 5.0,
        "virgo_present_near_target": virgo_index is not None,
        "screen_pair_region_contains_finest_particles": (
            env[4]["fine_count"] > 0 if screen_pair is not None else False),
    }
    result = {
        "snapshot": str(args.output),
        "metadata": {
            **info, "box_mpc_h": args.box, "npart_total": ntotal,
            "mass_species": species, "velocity_unit_kms": velocity_unit,
            "hubble_term_kms_per_mpc_h": hubble_term,
            "production_commit": "7b81f3193013ece48655aae0d1dece4748f4a9cf",
        },
        "catalog": {
            "halo_finder": args.halo_finder,
            "n_halos_ge20": int(len(cat["mass"])),
            "n_halos_ge5e11": int(np.count_nonzero(cat["mass"] >= 5e11)),
        },
        "pair": pair,
        "screen_pair": screen_pair,
        "screen_pair_region_r1": (env[4] if screen_pair is not None else None),
        "screen_pair_nearest_hop_group": screen_group,
        "m33": {
            "passed": False,
            "status": "not_resolved",
            "reason": ("The screen-selected MW/M31 region contains no level-14 "
                       "particles and HOP returns one fully contaminated coarse group; "
                       "a physical M33 satellite test is therefore undefined."),
        },
        "halo_profiles": profiles,
        "virgo": {
            "target_mpc_h": virgo_target.tolist(),
            "halo_index": virgo_index,
            "target_offset_mpc_h": (
                float(dvir[virgo_index]) if virgo_index is not None else None),
            "profile": virgo_profile,
            "infall_kms": infall,
            "overdensity_r8": virgo_od,
        },
        "void": {"target_mpc_h": void.tolist(), "overdensity_r8": void_od},
        "checks": {
            "core": core_checks,
            "contamination": contamination_checks,
            "phase": phase_checks,
        },
        "verdict": {
            "core_lg_gate": all(core_checks.values()),
            "clean_zoom": all(contamination_checks.values()),
            "phase_preserved": all(phase_checks.values()),
            "overall": (all(core_checks.values()) and all(contamination_checks.values())
                        and all(phase_checks.values())),
        },
        "top_pairs_preliminary": pairs[:20],
    }
    (args.work / "gate_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_plot(args.work / "z0_gate.png", cat, pair, profiles, observer,
               virgo_target, virgo_index, args.box)
    print("[verdict]", json.dumps(result["verdict"]), flush=True)
    print(f"[done] {args.work / 'gate_result.json'}", flush=True)


if __name__ == "__main__":
    main()
