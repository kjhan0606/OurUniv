#!/usr/bin/env python
"""Size a zoom (high-resolution) region: trace a z=0 target back to its Lagrangian volume.

For a target object at z=0, select the particles within R_sel, map them back to their
INITIAL (Lagrangian) positions -- for pmwd these are the grid positions recovered from the
particle index -- and measure the Lagrangian region (convex hull) that a zoom must refine.
Then report how many high-resolution particles that region holds at a target particle mass.

This is the sizing step of a constrained zoom (CLUES/HESTIA/SIBELIUS style): the zoom region
is the Lagrangian volume of the target plus a buffer, NOT the small z=0 virial radius.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import RHO_CRIT
from fof import fof


def lagrangian_of(idx, N, sp):
    """Initial grid position of pmwd particles from their flat index (C-order (N,N,N))."""
    ix, iy, iz = np.unravel_index(idx, (N, N, N))
    return np.stack([ix, iy, iz], 1).astype(np.float64) * sp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptcl", default="recon/cf4_ptcl768.npz")
    ap.add_argument("--target-mass", type=float, default=5e12, help="LG-total-mass analog [Msun/h]")
    ap.add_argument("--rsel", type=float, default=5.0, help="z=0 selection radius [h^-1Mpc]")
    ap.add_argument("--pad", type=float, default=0.15, help="Lagrangian buffer fraction")
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--out", default="recon/cf4_zoom_region.png")
    args = ap.parse_args()

    z = np.load(args.ptcl)
    pos = z["pos"].astype(np.float32)
    N = int(z["N"]); sp = float(z["spacing"]); L = float(z["L"]); c = L / 2.0
    m_p = args.Om * RHO_CRIT * sp ** 3
    print(f"[zoom] {pos.shape[0]} ptcl, N={N}, sp={sp} h^-1Mpc, m_p={m_p:.2e} Msun/h", flush=True)

    # validate the index->Lagrangian ordering via the displacement field magnitude
    rng = np.random.default_rng(0)
    samp = rng.integers(0, N ** 3, 8000)
    d = pos[samp].astype(np.float64) - lagrangian_of(samp, N, sp)
    d -= L * np.round(d / L)
    disp = np.sqrt((d ** 2).sum(1).mean())
    print(f"[zoom] index->Lagrangian check: displacement rms = {disp:.2f} h^-1Mpc "
          f"({'OK, ~few Mpc' if disp < 30 else 'WRONG ordering (~157 if random)'})", flush=True)

    # pick a target halo of the requested mass, close to the observer (FoF the central cube)
    m = np.all(np.abs(pos - c) < 35.0, axis=1)
    subidx = np.flatnonzero(m)
    halos = fof(pos[subidx], L=L, mean_sep=sp, n_min=20, m_particle=m_p, periodic=False)
    hm = halos["mass"]; hp = halos["pos"]; rh = np.linalg.norm(hp - c, axis=1)
    score = np.abs(np.log10(hm) - np.log10(args.target_mass)) + 0.02 * rh
    j = int(np.argmin(score)); center = hp[j].astype(np.float32)
    print(f"[zoom] target halo M={hm[j]:.2e} Msun/h at r={rh[j]:.1f} "
          f"SG=({center[0]-c:.1f},{center[1]-c:.1f},{center[2]-c:.1f})", flush=True)

    # select z=0 particles within R_sel of the target (full box)
    dd = pos - center; dd -= L * np.round(dd / L)
    r2 = (dd.astype(np.float64) ** 2).sum(1)
    sel = np.flatnonzero(r2 < args.rsel ** 2)
    q = lagrangian_of(sel, N, sp)
    # unwrap the Lagrangian patch about its centroid
    qc = q.mean(0); q = q - qc; q -= L * np.round(q / L)
    print(f"[zoom] selected {sel.size} particles within {args.rsel} h^-1Mpc at z=0", flush=True)

    from scipy.spatial import ConvexHull
    hull = ConvexHull(q)
    V = hull.volume * (1 + args.pad) ** 3
    R_eff = (3 * V / (4 * np.pi)) ** (1.0 / 3.0)
    ext = (q.max(0) - q.min(0)) * (1 + args.pad)
    R_L_target = (3 * args.target_mass / (4 * np.pi * args.Om * RHO_CRIT)) ** (1.0 / 3.0)
    print(f"[zoom] target's own Lagrangian radius R_L(M) = {R_L_target:.2f} h^-1Mpc", flush=True)
    print(f"[zoom] zoom Lagrangian region (+{int(args.pad*100)}% pad): "
          f"bbox {ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f}, hull V={V:.0f} (h^-1Mpc)^3, "
          f"R_eff={R_eff:.2f} h^-1Mpc", flush=True)

    print("\n  high-res particles filling the zoom region at target m_p:")
    print(f"  {'m_p':>10} {'refine/dim':>11} {'N_hires':>12}")
    for mp_t in (1e8, 1e7, 1e6, 1e5):
        f = (m_p / mp_t) ** (1.0 / 3.0)                 # refinement per dimension
        n_h = V / (sp / f) ** 3                          # fill hull volume at fine spacing
        print(f"  {mp_t:>10.0e} {f:>11.0f} {n_h:>12.2e}")

    # figure: z=0 selection (left) vs traced-back Lagrangian patch (right)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.3))
    zsel = (pos[sel] - center); zsel -= L * np.round(zsel / L)
    ax[0].scatter(zsel[:, 0], zsel[:, 1], s=2, alpha=0.3, lw=0)
    ax[0].plot(0, 0, "r+", ms=14, mew=2)
    ax[0].set_title(f"z=0: target (M={hm[j]:.1e}) + {args.rsel:.0f} h$^{{-1}}$Mpc selection\n"
                    f"{sel.size} particles", fontsize=10)
    ax[0].set_xlabel("dSGX"); ax[0].set_ylabel("dSGY"); ax[0].set_aspect("equal")
    ax[1].scatter(q[:, 0], q[:, 1], s=2, alpha=0.3, lw=0, c="C1")
    ax[1].set_title(f"Lagrangian (initial) patch = zoom region\n"
                    f"R_eff={R_eff:.2f}, bbox {ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f} h$^{{-1}}$Mpc",
                    fontsize=10)
    ax[1].set_xlabel("dqx"); ax[1].set_ylabel("dqy"); ax[1].set_aspect("equal")
    fig.suptitle("Zoom-region sizing: trace the z=0 target back to its Lagrangian volume",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=115)
    print(f"\n[zoom] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
