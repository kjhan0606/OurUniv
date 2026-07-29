#!/usr/bin/env python
"""Local halo field around the observer, and what lowering the HOD threshold does.

Loads the forwarded particle dump (cf4_ic_test --dump-ptcl), extracts the central cube,
runs FoF, and asks: as we lower the mass threshold, do Local-Group-scale objects appear
near us? It shows the local halos exist down to the resolution floor but form a random
field -- no Milky Way / M31 / M33 at r<1 Mpc, because their scale is unconstrained.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import RHO_CRIT
from fof import fof
from hod import populate

CLUSTERS = {"Virgo": (102.9, -2.3, 16.0), "Coma": (89.0, 8.0, 90.0),
            "Centaurus": (156.0, -11.0, 45.0), "Norma/GA": (155.0, -6.0, 65.0),
            "Fornax": (236.0, -44.0, 19.0)}


def sg(sgl, sgb, d):
    l, b = np.radians(sgl), np.radians(sgb)
    return d * np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptcl", default="recon/cf4_ptcl768.npz")
    ap.add_argument("--half", type=float, default=35.0, help="central cube half-size [h^-1Mpc]")
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--logMmin", type=float, default=11.3, help="lowered HOD threshold")
    ap.add_argument("--out", default="recon/cf4_local_halos.png")
    args = ap.parse_args()

    z = np.load(args.ptcl)
    pos = z["pos"].astype(np.float64); vel = z["vel"].astype(np.float64)
    sp = float(z["spacing"]); L = float(z["L"]); h = float(z["h"]); c = L / 2.0
    m_p = args.Om * RHO_CRIT * sp ** 3
    print(f"[local] m_p={m_p:.2e} Msun/h  observer at {c:.0f}", flush=True)

    m = np.all(np.abs(pos - c) < args.half, axis=1)
    sp_pos, sp_vel = pos[m], vel[m]
    print(f"[local] central cube +-{args.half:.0f}: {sp_pos.shape[0]} particles", flush=True)
    halos = fof(sp_pos, sp_vel, L=L, mean_sep=sp, b=0.2, n_min=20,
                m_particle=m_p, periodic=False, verbose=True)
    hp = halos["pos"]; hm = halos["mass"]
    rh = np.linalg.norm(hp - c, axis=1)
    print(f"[local] FoF floor = 20 particles = {20*m_p:.2e} Msun/h (MW~1e12, M33~5e11)", flush=True)

    print("\n  halo counts near the observer by mass threshold:")
    print(f"  {'R[h^-1Mpc]':>10} {'>2e11':>8} {'>5e11':>8} {'>1e12':>8} {'>2e12':>8}")
    for R in (1, 2, 3, 5, 10, 20):
        sel = rh < R
        row = [np.sum(sel & (hm > t)) for t in (2e11, 5e11, 1e12, 2e12)]
        print(f"  {R:>10} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

    nearest = np.argsort(rh)[:6]
    print("\n  nearest halos to us (any mass >= floor):")
    for i in nearest:
        p = hp[i] - c
        print(f"    r={rh[i]:5.2f}  M={hm[i]:.2e}  SG=({p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f})")

    # HOD at the lowered threshold
    hod = dict(logMmin=args.logMmin, sigma_logM=0.2, logM0=args.logMmin - 0.5,
               logM1=args.logMmin + 1.0, alpha=1.0)
    gal = populate(halos, pos_all=sp_pos, vel_all=sp_vel, params=hod, mode="particles",
                   Om=args.Om, seed=7, verbose=True)

    # figure: central halo field, two thresholds side by side
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.3))
    for ax, thr, ttl in [(axes[0], 2e11, f"all FoF halos (>2e11, floor)  logMmin={args.logMmin}"),
                         (axes[1], 2e12, "default HOD threshold (>2e12)")]:
        sel = (hm > thr) & (np.abs(hp[:, 2] - c) < 12)
        p = hp[sel] - c
        s = 6 + 22 * (np.log10(hm[sel]) - np.log10(thr))
        scat = ax.scatter(p[:, 0], p[:, 1], s=s, c=np.log10(hm[sel]), cmap="viridis",
                          alpha=0.8, lw=0, vmin=11.3, vmax=14.5)
        ax.plot(0, 0, "+", color="red", ms=16, mew=2.5)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="red", ls="--", lw=1))
        for nm, (l, b, d) in CLUSTERS.items():
            x = sg(l, b, d) * h
            if abs(x[2]) < 15 and max(abs(x[0]), abs(x[1])) < 30:
                ax.plot(x[0], x[1], "k*", ms=12); ax.annotate(nm, (x[0], x[1]), fontsize=8)
        ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_aspect("equal")
        ax.set_title(f"{ttl}  [{sel.sum()} halos in slice]", fontsize=10)
        ax.set_xlabel("SGX [$h^{-1}$Mpc]"); ax.set_ylabel("SGY [$h^{-1}$Mpc]")
    fig.colorbar(scat, ax=axes, label="log$_{10}$ M [$M_\\odot/h$]", shrink=0.8)
    fig.suptitle("Local halo field around us (red +): lowering the threshold shows more "
                 "small halos, but no Milky Way / M31 / M33 at r<1 Mpc (dashed) -- the "
                 "sub-Mpc phases are random", fontsize=11)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"\n[local] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
