#!/usr/bin/env python
"""Selection-matched comparison of the closed-loop mock galaxies to CF4.

The raw mock-vs-CF4 galaxy density cross-correlation is dominated by the CF4 radial
selection function (a flux/distance-limited survey has far more galaxies nearby),
not by the density field, so it reads near zero even when the structure agrees. This
redoes the comparison two selection-independent ways on the SAVED mock catalog:

  (A) match the mock radial number profile to CF4's, then cross-correlate;
  (B) compare the ANGULAR overdensity in radial shells (divide each shell by its
      angular mean so the selection cancels).
"""
import os
import numpy as np


def radial_profile(r, edges):
    n, _ = np.histogram(r, bins=edges)
    vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return n / vol                                    # number density per shell


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", default="recon/cf4_ic_test/mock_catalog.npz")
    ap.add_argument("--real", default="data/cf4_clean.npz")
    ap.add_argument("--rmax", type=float, default=80.0)
    ap.add_argument("--smooth", type=float, default=5.0, help="Gaussian smoothing [h^-1Mpc]")
    ap.add_argument("--out", default="recon/cf4_ic_test/cf4_ic_diag.png")
    args = ap.parse_args()
    from scipy.ndimage import gaussian_filter

    zc = np.load(args.mock)
    gp = zc["pos"].astype(np.float64); L = float(zc["L"]); h = float(zc["h"])
    c = float(zc["observer"][0]) if "observer" in zc else L / 2.0
    zr = np.load(args.real)
    cf4 = zr["pos_dist"].astype(np.float64) * h + c
    rm = np.linalg.norm(gp - c, axis=1)
    rc = np.linalg.norm(cf4 - c, axis=1)
    print(f"[diag] mock {gp.shape[0]} gal, CF4 {cf4.shape[0]} gal; observer at {c:.0f}", flush=True)

    # (A) match the mock radial profile to CF4's, then grid + cross-correlate
    edges = np.linspace(0, args.rmax, 17)
    pr_c = radial_profile(rc[rc < args.rmax], edges)
    pr_m = radial_profile(rm[rm < args.rmax], edges)
    ratio = np.divide(pr_c, pr_m, out=np.zeros_like(pr_c), where=pr_m > 0)
    ratio /= ratio.max() + 1e-12                       # keep prob <= 1
    ib = np.clip(np.digitize(rm, edges) - 1, 0, len(edges) - 2)
    rng = np.random.default_rng(0)
    keep = (rm < args.rmax) & (rng.random(rm.shape[0]) < ratio[ib])
    gm = gp[keep]
    print(f"[diag] selection-matched mock: {gm.shape[0]} of {int((rm<args.rmax).sum())}", flush=True)

    Ng = 64; cell = 2 * args.rmax / Ng
    ax = (np.arange(Ng) + 0.5) * cell - args.rmax
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    rr = np.sqrt(X**2 + Y**2 + Z**2)
    msk = rr < args.rmax
    sg = args.smooth / cell

    def dens(p):
        q = p - c
        idx = np.floor((q + args.rmax) / cell).astype(int)
        ok = np.all((idx >= 0) & (idx < Ng), axis=1)
        idx = idx[ok]
        f = np.zeros((Ng, Ng, Ng), np.float32)
        np.add.at(f, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        f = gaussian_filter(f, sg)
        return f / f[msk].mean() - 1.0

    dmm = dens(gm); dcc = dens(cf4[rc < args.rmax])
    rA = np.corrcoef(dmm[msk], dcc[msk])[0, 1]
    print(f"[diag] (A) selection-matched density cross-corr (r<{args.rmax:.0f}, "
          f"{args.smooth:.0f} h^-1Mpc smooth) = {rA:.3f}", flush=True)

    # (B) angular overdensity per radial shell (selection cancels in each shell)
    def ang_over(p, r, r0, r1, nside=24):
        m = (r >= r0) & (r < r1)
        q = p[m] - c
        rr2 = np.linalg.norm(q, axis=1) + 1e-9
        th = np.arccos(q[:, 2] / rr2); ph = np.arctan2(q[:, 1], q[:, 0]) + np.pi
        hi = (np.clip((ph / (2 * np.pi) * 2 * nside).astype(int), 0, 2 * nside - 1),
              np.clip((th / np.pi * nside).astype(int), 0, nside - 1))
        g = np.zeros((2 * nside, nside)); np.add.at(g, hi, 1.0)
        g = gaussian_filter(g, 1.0)
        return g / (g.mean() + 1e-9) - 1.0

    shells = [(8, 20), (20, 40), (40, 60), (60, 80)]
    rBs = []
    for r0, r1 in shells:
        a = ang_over(gp, rm, r0, r1); b = ang_over(cf4, rc, r0, r1)
        rB = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        rBs.append(rB)
        print(f"[diag] (B) angular overdensity corr shell {r0}-{r1}: r = {rB:.3f}", flush=True)

    # figure: angular overdensity maps (mock vs CF4) per shell -- the selection cancels,
    # so this shows the actual structural agreement the raw density corr hid
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(shells), 2, figsize=(11, 3.0 * len(shells)))
    for row, (r0, r1) in enumerate(shells):
        a = ang_over(gp, rm, r0, r1); b = ang_over(cf4, rc, r0, r1)
        vlim = 2.0
        for col, (g, ttl) in enumerate([(a, "mock"), (b, "CF4")]):
            ax = axes[row, col]
            im = ax.imshow(g.T, origin="lower", extent=[-180, 180, 0, 180],
                           cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
            ax.set_title(f"{ttl}  {r0}-{r1} h$^{{-1}}$Mpc (r={rBs[row]:.2f})", fontsize=9)
            ax.set_xlabel("SGL [deg]"); ax.set_ylabel("SGB'")
    fig.suptitle("Angular overdensity in radial shells (selection cancels): mock vs CF4",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(args.out, dpi=110)
    print(f"[diag] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
