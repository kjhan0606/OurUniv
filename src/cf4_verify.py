#!/usr/bin/env python
"""Verify the amortized-net training: per-sample r(k) scatter + baselines.

Checks (CPU-only, from saved predictions):
  1. per-test-sample cross-correlation r(k) mean +/- std (is r(k) robust?)
  2. baseline r(k): raw galaxy-count field n_gal vs true asinh(delta_m)
     -- what "just use the galaxy density, no network" already achieves.
  3. velocity-only (nogal) vs the n_gal baseline -- the SCIENCE test: does the
     peculiar-velocity field carry density information the galaxy positions don't?
"""
import os
import glob
import numpy as np


def rk_persample(A, B, box):
    """r(k) per sample -> (kc, R[n,nk]).  A,B: (n,N,N,N)."""
    n, N = A.shape[0], A.shape[-1]
    dx = box / N
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx); kz = 2 * np.pi * np.fft.rfftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    kk = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    kf = 2 * np.pi / box; kny = np.pi / dx
    bins = np.arange(0.5 * kf, kny, kf); m = kk > 0
    idx = np.digitize(kk[m], bins)
    kc = 0.5 * (bins[:-1] + bins[1:])
    R = np.full((n, len(kc)), np.nan)
    for s in range(n):
        ta = np.fft.rfftn(A[s]); tb = np.fft.rfftn(B[s])
        cr = np.real(ta * np.conj(tb))[m]; pa = (np.abs(ta) ** 2)[m]; pb = (np.abs(tb) ** 2)[m]
        for j in range(1, len(bins)):
            sel = idx == j
            if sel.any():
                R[s, j - 1] = cr[sel].sum() / np.sqrt(pa[sel].sum() * pb[sel].sum() + 1e-30)
    return kc, R


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    box = 64 * 2.0  # N * spacing = 128 Mpc/h

    preds = {ab: np.load(os.path.join(HERE, f"cf4_cnn_pred_{ab}.npz"))
             for ab in ("full", "novel", "nogal")}
    true = preds["full"]["true"]                 # (32,N,N,N) normalized asinh(delta_m)

    # baseline: galaxy-count field from the test shard (same order/seeds)
    tf = sorted(glob.glob(os.path.join(ROOT, "data_train", "test", "shard_*.npz")))
    zt = np.load(tf[0], allow_pickle=True)
    n_gal = zt["n_gal"].astype(np.float32)       # (32,N,N,N)

    curves = {}
    kc, R = rk_persample(true, n_gal, box); curves["n_gal baseline"] = R
    for ab in ("full", "novel", "nogal"):
        _, R = rk_persample(true, preds[ab]["pred"], box)
        curves[ab] = R

    print(f"{'k[h/Mpc]':>9} " + " ".join(f"{k:>16}" for k in curves))
    for i in range(len(kc)):
        row = f"{kc[i]:9.4f} "
        for k in curves:
            mu = np.nanmean(curves[k][:, i]); sd = np.nanstd(curves[k][:, i])
            row += f" {mu:6.3f}+/-{sd:5.3f}"
        print(row)

    # headline numbers
    print("\n--- large-scale (k<0.15) mean r(k) ---")
    lo = kc < 0.15
    for k in curves:
        print(f"  {k:16s}: {np.nanmean(curves[k][:, lo]):.3f}")
    print("--- intermediate (0.1<k<0.3) mean r(k)  [where v-info peaks] ---")
    mid = (kc > 0.1) & (kc < 0.3)
    for k in curves:
        print(f"  {k:16s}: {np.nanmean(curves[k][:, mid]):.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {"n_gal baseline": ("k", "--"), "full": ("C0", "-"),
              "novel": ("C2", "-."), "nogal": ("C3", "-")}
    labels = {"n_gal baseline": "galaxy density only (no net)",
              "full": "CNN: galaxies+velocity", "novel": "CNN: density only",
              "nogal": "CNN: peculiar-velocity ONLY"}
    for k in curves:
        mu = np.nanmean(curves[k], 0); sd = np.nanstd(curves[k], 0)
        c, ls = styles[k]
        ax.plot(kc, mu, ls, color=c, label=labels[k], lw=2)
        ax.fill_between(kc, mu - sd, mu + sd, color=c, alpha=0.12)
    ax.axhline(0, ls=":", c="gray"); ax.axhline(1, ls=":", c="gray")
    ax.set_xscale("log"); ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("cross-corr r(k)")
    ax.set_ylim(-0.1, 1.05); ax.set_title("Density recovery vs scale (32 test, +/-1 sigma)")
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.5, 0.05, "scale [Mpc/h]", transform=ax.transAxes, ha="center", fontsize=8)
    sec = ax.secondary_xaxis("top", functions=(lambda k: 2 * np.pi / np.maximum(k, 1e-6),
                                               lambda L: 2 * np.pi / np.maximum(L, 1e-6)))
    sec.set_xlabel("wavelength 2pi/k [Mpc/h]")
    fig.tight_layout()
    out = os.path.join(ROOT, "recon", "cf4_verify_rk.png")
    fig.savefig(out, dpi=120)
    print(f"\n[fig] saved {out}")


if __name__ == "__main__":
    main()
