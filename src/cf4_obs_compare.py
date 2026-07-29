#!/usr/bin/env python
"""Compare the HR-constrained forward density with the OBSERVED CF4 galaxies.

Overlays the actual CF4 galaxy positions (converted to the box frame) on our forwarded
density slice, and shows a galaxy-count density panel beside it. This is a check against
OBSERVATION, not against ourselves.

Honest caveats printed by the script:
  - CF4 is a peculiar-velocity sample with a radial selection (density falls with distance);
    raw galaxy counts are selection-modulated, not the true density.
  - Our field is CONSTRAINED by these galaxies' velocities, so agreement at Virgo is partly
    circular. Agreement at structure NOT sitting on a strong tracer is the more independent test.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import make_forward, RHO_CRIT
from cf4_make_ic import embed_ic


def sgdir(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192_hr.npz")
    ap.add_argument("--cat", default="data/cf4_clean.npz")
    ap.add_argument("--key", default="s_out")
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--half", type=float, default=40.0)
    ap.add_argument("--slab", type=float, default=8.0, help="slice full-thickness [h^-1Mpc]")
    ap.add_argument("--smooth", type=float, default=0.8)
    ap.add_argument("--out", default="recon/cf4_obs_compare.png")
    args = ap.parse_args()
    import jax.numpy as jnp
    import time
    from scipy.ndimage import gaussian_filter

    z = np.load(args.recon)
    s = z[args.key].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    L = Nc * spc; c = L / 2.0; H = float(z["hh"]) if "hh" in z else 0.746
    sp = L / args.Nfine
    print(f"[cmp] {args.recon} key={args.key} L={L:.0f} h={H} half={args.half}", flush=True)

    # forward the HR realization
    s_fine = embed_ic(s, args.Nfine, 1)
    t0 = time.time()
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    print(f"[cmp] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)

    half = args.half; Ng = 200; cell = 2*half/Ng
    def slab_grid(p3):                       # count -> smoothed 1+delta slice
        m = np.all(np.abs(p3 - c) < half, axis=1); q = p3[m] - c
        idx = np.floor((q + half) / cell).astype(int)
        ok = np.all((idx >= 0) & (idx < Ng), axis=1); idx = idx[ok]
        f = np.zeros((Ng, Ng, Ng), np.float32)
        np.add.at(f, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        f = gaussian_filter(f, args.smooth/cell); d = f/f.mean() - 1.0
        th = max(1, int(round(args.slab/2/cell)))
        return d[:, :, Ng//2-th:Ng//2+th].mean(2)
    sl = slab_grid(pos)

    # --- observed CF4 galaxies -> box frame (h^-1Mpc), select the slab ---
    cz = np.load(args.cat)
    gpos_mpc = cz["pos_dist"].astype(float)          # supergalactic Cartesian, Mpc
    gdist = cz["dist"].astype(float)                 # Mpc
    gbox = c + gpos_mpc * H                           # box frame, h^-1Mpc
    # SELECTION WEIGHT: 1/nbar(r) removes the radial selection (CF4 piles up near the observer).
    seln_r = cz["seln_r"].astype(float); seln_nbar = cz["seln_nbar"].astype(float)
    nbar_g = np.interp(gdist, seln_r, seln_nbar, left=seln_nbar[0], right=seln_nbar[-1])
    wsel = 1.0 / np.clip(nbar_g, seln_nbar.max()*1e-3, None)
    gz = gbox[:, 2] - c
    ing = (np.abs(gz) < args.slab/2) & np.all(np.abs(gbox - c) < half, axis=1)
    gx = gbox[ing, 0] - c; gy = gbox[ing, 1] - c
    print(f"[cmp] CF4 galaxies in +-{half:.0f} box: {np.all(np.abs(gbox-c)<half,axis=1).sum()}; "
          f"in |SGZ|<{args.slab/2:.0f} slab: {ing.sum()}", flush=True)

    # selection-CORRECTED galaxy density in the slab (1/nbar weighted): real structure, not selection
    gg = np.zeros((Ng, Ng), np.float32)
    gi = np.floor((gbox[ing, :2] - c + half) / cell).astype(int)
    okg = np.all((gi >= 0) & (gi < Ng), axis=1)
    np.add.at(gg, (gi[okg, 0], gi[okg, 1]), wsel[ing][okg].astype(np.float32))
    gg = gaussian_filter(gg, 2.5/cell)               # coarser: sparse tracer

    virgo = (sgdir(102.9, -2.3) * (16.5 * H))[:2]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    ax = axs[0]
    im = ax.imshow(sl.T, origin="lower", extent=[-half, half, -half, half],
                   cmap="inferno", vmin=-0.8, vmax=np.percentile(sl, 99.5))
    ax.scatter(gx, gy, s=9, c="cyan", alpha=0.55, lw=0, label=f"CF4 gal (|SGZ|<{args.slab/2:.0f})")
    ax.plot(virgo[0], virgo[1], "o", ms=15, mfc="none", mec="lime", mew=2.5, label="Virgo (lit.)")
    ax.plot(0, 0, "w+", ms=13, mew=2)
    ax.set_title(f"HR forward density + observed CF4 galaxies\n(key={args.key}, N={args.Nfine})")
    ax.legend(loc="upper right", framealpha=0.35, labelcolor="white")
    ax.set_xlabel("SGX [$h^{-1}$Mpc]"); ax.set_ylabel("SGY [$h^{-1}$Mpc]")
    fig.colorbar(im, ax=ax, label=r"$\delta$ (reconstructed)", fraction=0.046)

    ax = axs[1]
    im2 = ax.imshow(gg.T, origin="lower", extent=[-half, half, -half, half],
                    cmap="viridis", vmax=np.percentile(gg, 99.5))
    ax.plot(virgo[0], virgo[1], "o", ms=15, mfc="none", mec="red", mew=2.5, label="Virgo (lit.)")
    ax.plot(0, 0, "w+", ms=13, mew=2)
    ax.set_title(f"Observed CF4 density (same slab, {ing.sum()} gal)\nSELECTION-CORRECTED (1/nbar weighted)")
    ax.legend(loc="upper right", framealpha=0.35)
    ax.set_xlabel("SGX [$h^{-1}$Mpc]"); ax.set_ylabel("SGY [$h^{-1}$Mpc]")
    fig.colorbar(im2, ax=ax, label="galaxy count (smoothed)", fraction=0.046)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"[cmp] saved {args.out}", flush=True)

    # --- quantitative: correlate reconstructed delta with galaxy count over the slab cells ---
    a = sl.ravel(); b = gg.ravel()
    good = b > 0
    r_all = np.corrcoef(a, b)[0, 1]
    r_pos = np.corrcoef(a[good], b[good])[0, 1]
    print(f"[cmp] pixel corr(recon delta, galaxy count): all={r_all:.3f}  where-galaxies={r_pos:.3f}", flush=True)
    # peak coincidence: is the reconstructed field overdense where galaxies pile up?
    hot = b > np.percentile(b[good], 90)
    print(f"[cmp] recon 1+delta at galaxy-rich cells (top10%): median={np.median(1+a[hot]):.2f} "
          f"vs field median={np.median(1+a):.2f}", flush=True)


if __name__ == "__main__":
    main()
