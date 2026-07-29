#!/usr/bin/env python
"""Zoom trace-back for the CONFIRMED constrained Local Group (c48f3).

Identify the winning MW-M31 pair in the saved lg_search catalog, take its z=0 particles
within R_sel, trace them to Lagrangian positions (from the stored global grid index), and
size the zoom region + report the high-res particle budget to resolve the satellites/dwarfs.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIRGO_D = 16.5 * 0.746


def sgdir(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lg", default="recon/lg_search/lg_c48f3.npz")
    ap.add_argument("--rsel", type=float, default=5.0, help="z=0 selection radius around LG [h^-1Mpc]")
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--out", default="recon/cf4_zoom_lg.png")
    args = ap.parse_args()

    z = np.load(args.lg)
    hp = z["halo_pos"].astype(float); hm = z["halo_mass"].astype(float)
    hv = z["halo_vel"].astype(float)
    cen_gidx = z["cen_gidx"].astype(np.int64); cen_pos = z["cen_pos"].astype(float)
    sp = float(z["sp"]); L = float(z["L"]); m_p = float(z["m_p"]); c = L / 2.0
    N = args.Nfine
    vhat = sgdir(102.9, -2.3)

    # identify the winning MW-M31 pair by the same chi^2 used in selection
    rh = np.linalg.norm(hp - c, axis=1)
    mw = np.flatnonzero((hm > 5e11) & (hm < 4e12) & (rh < 20))
    big = hp[hm > 5e12]
    best = None
    for a in range(len(mw)):
        for b in range(a + 1, len(mw)):
            i, j = mw[a], mw[b]
            sep = np.linalg.norm(hp[i] - hp[j])
            if not (0.4 < sep < 0.9):
                continue
            mid = 0.5 * (hp[i] + hp[j])
            if len(big) and np.linalg.norm(big - mid, axis=1).min() < 3.0:
                continue
            off = mid - c
            lg_virgo = np.linalg.norm(vhat * VIRGO_D - off)
            vrel = np.dot(hv[i]-hv[j], (hp[i]-hp[j])/sep)
            vlg = (hm[i]*hv[i] + hm[j]*hv[j]) / (hm[i]+hm[j])
            infall = np.dot(vlg, vhat)
            chi2 = (((lg_virgo-VIRGO_D)/2.5)**2 + ((infall-200)/60)**2 + ((vrel+110)/40)**2
                    + ((np.log10(max(hm[i],hm[j]))-12.15)/0.2)**2
                    + ((np.log10(min(hm[i],hm[j]))-12.05)/0.2)**2 + ((sep-0.57)/0.12)**2)
            if best is None or chi2 < best[0]:
                best = (chi2, i, j, mid, sep, hm[i], hm[j], vrel, infall, lg_virgo)
    chi2, i, j, lg_mid, sep, Mi, Mj, vrel, infall, lg_virgo = best
    print(f"[zoom] confirmed LG (c48f3): MW-M31 sep={sep:.2f} M=({Mi:.1e},{Mj:.1e}) "
          f"chi2={chi2:.1f}", flush=True)
    print(f"[zoom]   approach={vrel:.0f} km/s  infall={infall:.0f} km/s  LG-Virgo={lg_virgo:.1f}  "
          f"offset={np.linalg.norm(lg_mid-c):.1f} h^-1Mpc", flush=True)
    # M33 candidate near the pair (sub-grid; report if present)
    m33 = np.flatnonzero((hm > 2e11) & (hm < 9e11))
    d33 = np.minimum(np.linalg.norm(hp[m33]-hp[i], axis=1), np.linalg.norm(hp[m33]-hp[j], axis=1))
    near = m33[(d33 < 0.8) & (m33 != i) & (m33 != j)]
    if len(near):
        k = near[np.argmin(d33[np.isin(m33, near)])]
        print(f"[zoom]   M33 candidate: M={hm[k]:.1e} at {min(np.linalg.norm(hp[k]-hp[i]),np.linalg.norm(hp[k]-hp[j])):.2f} h^-1Mpc (parent-marginal)", flush=True)
    else:
        print(f"[zoom]   no M33 candidate at parent resolution (expected -- zoom product)", flush=True)

    # select z=0 particles within R_sel of the LG centre
    dd = cen_pos - lg_mid
    sel = np.flatnonzero((dd**2).sum(1) < args.rsel**2)
    gidx = cen_gidx[sel]
    ix, iy, iz = np.unravel_index(gidx, (N, N, N))
    q = np.stack([ix, iy, iz], 1).astype(np.float64) * sp     # Lagrangian positions
    qc = q.mean(0); q = q - qc; q -= L * np.round(q / L)
    print(f"[zoom] {sel.size} z=0 particles within {args.rsel} h^-1Mpc of the LG", flush=True)

    from scipy.spatial import ConvexHull
    hull = ConvexHull(q); V = hull.volume * (1 + args.pad) ** 3
    R_eff = (3 * V / (4 * np.pi)) ** (1.0/3.0)
    ext = (q.max(0) - q.min(0)) * (1 + args.pad)
    print(f"[zoom] zoom Lagrangian region (+{int(args.pad*100)}% pad): R_eff={R_eff:.2f} h^-1Mpc, "
          f"bbox {ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f}, hull V={V:.0f} (h^-1Mpc)^3", flush=True)
    print(f"\n  high-res particles in the zoom region + what each resolves:")
    print(f"  {'m_p':>9} {'refine/dim':>11} {'N_hires':>11}  target")
    for mp_t, what in [(1e8, "LG large-scale"), (1e7, "MW/M31 halos"),
                       (1e6, "Magellanic + classical dwarfs"), (1e5, "+ ultra-faint dwarfs")]:
        f = (m_p / mp_t) ** (1.0/3.0)
        n_h = V / (sp / f) ** 3
        print(f"  {mp_t:>9.0e} {f:>11.0f} {n_h:>11.2e}  {what}")

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.3))
    zsel = cen_pos[sel] - lg_mid
    ax[0].scatter(zsel[:, 0], zsel[:, 1], s=2, alpha=0.3, lw=0)
    ax[0].plot((hp[i]-lg_mid)[0], (hp[i]-lg_mid)[1], "r*", ms=14, label=f"MW {Mi:.1e}")
    ax[0].plot((hp[j]-lg_mid)[0], (hp[j]-lg_mid)[1], "m*", ms=12, label=f"M31 {Mj:.1e}")
    ax[0].legend(fontsize=8); ax[0].set_aspect("equal")
    ax[0].set_title(f"z=0: confirmed LG (c48f3) + {args.rsel:.0f} h$^{{-1}}$Mpc, {sel.size} ptcl")
    ax[0].set_xlabel("dSGX"); ax[0].set_ylabel("dSGY")
    ax[1].scatter(q[:, 0], q[:, 1], s=2, alpha=0.3, lw=0, c="C1"); ax[1].set_aspect("equal")
    ax[1].set_title(f"Lagrangian zoom region: R_eff={R_eff:.2f} h$^{{-1}}$Mpc")
    ax[1].set_xlabel("dqx"); ax[1].set_ylabel("dqy")
    fig.suptitle("Constrained LG zoom sizing (c48f3): trace the LG back to its Lagrangian volume")
    fig.tight_layout(); fig.savefig(args.out, dpi=115)
    print(f"\n[zoom] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
