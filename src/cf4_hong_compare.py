#!/usr/bin/env python
"""Reproduce Hong et al. (2021, ApJ 913, 76) validation figures on OUR reconstruction.

Hong+2021 validate the galaxy->DM density reconstruction with:
  - Fig 4/6 : density SLICES (galaxy input | velocity input | truth | reconstruction),
              integrated over 4 h^-1Mpc thickness.
  - Fig 5   : JOINT PDF (2D histogram) of predicted vs true DM density (cell-by-cell).
  - Table 2 : two-point correlation function (2pCF) goodness-of-fit (KS statistic)
              in bins 0-1, 1-3, 3-10 h^-1Mpc  (TNG100: 0.263, 0.175, 0.130).

This makes the SAME panels from our test set so the two can be juxtaposed.
Caveats vs Hong+2021 (printed on the figure): they use IllustrisTNG (dense
galaxies), 0.3125 h^-1Mpc voxel, 20-40 Mpc sub-cubes; we use pmwd+FoF+HOD,
2 h^-1Mpc voxel, 128 Mpc periodic box (and a CF4-like sparse+noisy variant).
"""
import os
import argparse
import glob
import numpy as np
from cf4_verify import rk_persample


def twopcf(field, box, nbin=20):
    """Isotropic two-point correlation xi(r) of a gridded field via FFT."""
    N = field.shape[-1]
    d = field - field.mean()
    pk = np.abs(np.fft.rfftn(d)) ** 2
    xi = np.fft.irfftn(pk, s=(N, N, N)) / d.size          # autocorrelation
    dx = box / N
    ix = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    RX, RY, RZ = np.meshgrid(ix, ix, ix, indexing="ij")
    r = np.sqrt(RX ** 2 + RY ** 2 + RZ ** 2) * dx
    edges = np.linspace(0, box / 2, nbin + 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    xr = np.array([xi[(r >= edges[i]) & (r < edges[i + 1])].mean()
                   for i in range(nbin)])
    return rc, xr


def ks_stat(a, b):
    """KS distance between two 1D samples (CDF sup-norm)."""
    x = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), x, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), x, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction npz (true,pred)")
    ap.add_argument("--data", required=True, help="data dir with test shard (n_gal,vlos)")
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    box = args.N * args.spacing

    z = np.load(args.pred)
    true, pred = z["true"], z["pred"]              # (n,N,N,N) normalized asinh(delta_m)
    tf = sorted(glob.glob(os.path.join(args.data, "test", "shard_*.npz")))[0]
    zt = np.load(tf, allow_pickle=True)
    n_gal, vlos, dm = zt["n_gal"], zt["vlos"], zt["delta_m"]

    # cell-by-cell correlation (Hong Fig 5 metric)
    r_cell = float(np.corrcoef(true.ravel(), pred.ravel())[0, 1])
    kc, R = rk_persample(true, pred, box)
    lo = kc < 0.15; mid = (kc > 0.1) & (kc < 0.3)
    print(f"[hong-compare] cell-by-cell r = {r_cell:.3f}")
    print(f"[hong-compare] r(k) >40Mpc = {np.nanmean(R[:, lo]):.3f}, "
          f"~30Mpc = {np.nanmean(R[:, mid]):.3f}")

    # 2pCF true vs recon (avg over test) + KS in Hong bins (as accessible: 2-10, 10-30)
    rc, xt = twopcf(true[0], box); _, xp = twopcf(pred[0], box)
    for i in range(1, len(true)):
        xt += twopcf(true[i], box)[1]; xp += twopcf(pred[i], box)[1]
    xt /= len(true); xp /= len(true)
    # 2pCF agreement in Hong-like radial bins (ratio recon/true; 1.0 = perfect).
    # Our 2 Mpc/h voxel can't access <2 Mpc/h, so we report the accessible bins.
    def xi_ratio(rlo, rhi):
        m = (rc >= rlo) & (rc < rhi)
        return float(xp[m].sum() / (xt[m].sum() + 1e-12))
    b1, b2, b3 = xi_ratio(2, 6), xi_ratio(6, 12), xi_ratio(12, 30)
    print(f"[hong-compare] 2pCF ratio recon/true: 2-6Mpc={b1:.2f} 6-12Mpc={b2:.2f} "
          f"12-30Mpc={b3:.2f} (Hong: 2pCF KS smallest at large r)")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = max(1, int(round(4.0 / args.spacing)))    # 4 Mpc/h slab thickness
    zc = args.N // 2
    sl = lambda f: f[0, :, :, zc:zc + th].mean(-1)

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1])
    # Row 1: Hong Fig 4/6-style slices
    panels = [(np.log10(1 + sl(n_gal)), "galaxy input  log(1+n_gal)", "viridis"),
              (sl(vlos), "velocity input  v_los", "RdBu_r"),
              (sl(true), "TRUE  asinh(delta_m)", "magma"),
              (sl(pred), "RECONSTRUCTED", "magma")]
    for j, (im, ttl, cm) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        vlim = dict(vmin=-np.abs(im).max(), vmax=np.abs(im).max()) if cm == "RdBu_r" else {}
        ax.imshow(im.T, origin="lower", extent=[0, box, 0, box], cmap=cm, **vlim)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("Mpc/h")
        if j == 0:
            ax.set_ylabel(f"slice, {th*args.spacing:.0f} Mpc/h thick")

    # Row 2 panel A: joint PDF (Hong Fig 5)
    ax = fig.add_subplot(gs[1, 0])
    ax.hist2d(true.ravel(), pred.ravel(), bins=80, cmap="Blues",
              range=[[true.min(), true.max()], [true.min(), true.max()]])
    lim = [true.min(), true.max()]
    ax.plot(lim, lim, "r-", lw=1)
    ax.set_xlabel("true"); ax.set_ylabel("reconstructed")
    ax.set_title(f"joint PDF (Hong Fig 5)\ncell r = {r_cell:.3f}", fontsize=10)

    # Row 2 panel B: 2pCF true vs recon (Hong Table 2 spirit)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rc, xt, "k-", lw=2, label="true")
    ax.plot(rc, xp, "C0--", lw=2, label="reconstructed")
    ax.set_xlabel("r [Mpc/h]"); ax.set_ylabel(r"$\xi(r)$")
    ax.set_title("two-point corr. (Hong Table 2)", fontsize=10)
    ax.legend(fontsize=9); ax.set_xlim(0, box / 3)

    # Row 2 panel C: r(k) with band
    ax = fig.add_subplot(gs[1, 2])
    mu = np.nanmean(R, 0); sd = np.nanstd(R, 0)
    ax.semilogx(kc, mu, "C0-o", ms=3); ax.fill_between(kc, mu - sd, mu + sd, alpha=0.15)
    ax.axhline(1, ls=":", c="gray"); ax.axhline(0, ls=":", c="gray")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("r(k)"); ax.set_ylim(-0.1, 1.05)
    ax.set_title(f"cross-corr r(k)\ncell r={r_cell:.2f}", fontsize=10)

    # Row 2 panel D: metrics text / caveats
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    txt = ("Direct comparison to Hong+2021\n"
           "(ApJ 913,76; same 2-ch input:\n"
           " galaxy count + radial v_pec)\n\n"
           f"OURS ({args.tag or 'this run'}):\n"
           f"  cell r      = {r_cell:.3f}\n"
           f"  r(k)>40Mpc  = {np.nanmean(R[:,lo]):.3f}\n"
           f"  r(k)~30Mpc  = {np.nanmean(R[:,mid]):.3f}\n"
           f"  2pCF ratio recon/true:\n"
           f"    2-6 Mpc/h  = {b1:.2f}\n"
           f"    6-12 Mpc/h = {b2:.2f}\n"
           f"    12-30Mpc/h = {b3:.2f}\n\n"
           "HONG+2021 (TNG100) 2pCF KS:\n"
           "  0-1 / 1-3 / 3-10 Mpc/h\n"
           "  = 0.263 / 0.175 / 0.130\n"
           "  (smaller = better fit)\n\n"
           "Setup differs:\n"
           "  Hong: TNG(dense), 0.31 Mpc vox,\n"
           "        20-40 Mpc sub-cube\n"
           f"  ours: pmwd+FoF+HOD, {args.spacing:.0f} Mpc vox,\n"
           f"        {box:.0f} Mpc periodic box")
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle(f"Hong+2021-style validation — {args.tag}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", f"cf4_hong_compare_{args.tag or 'run'}.png")
    fig.savefig(out, dpi=120)
    print(f"[hong-compare] saved {out}")


if __name__ == "__main__":
    main()
