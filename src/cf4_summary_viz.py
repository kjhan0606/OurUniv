#!/usr/bin/env python
"""Capstone visualization: the CF4-constrained local universe (winner seed) with the LG,
Virgo, and Local Void marked. Two orthogonal supergalactic slices so all three appear."""
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
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--half", type=float, default=35.0)
    ap.add_argument("--smooth", type=float, default=0.9)
    ap.add_argument("--out", default="recon/cf4_local_universe_s22.png")
    args = ap.parse_args()
    import jax.numpy as jnp
    import time
    from scipy.ndimage import gaussian_filter

    z = np.load(args.recon)
    s = z["s_out"].astype(np.float64)
    Nc = int(z["N"]); spc = float(z["spacing"]); L = Nc*spc; c = L/2.0
    H = float(z["hh"]) if "hh" in z else 0.746
    sp = L/args.Nfine
    hz = np.load(f"recon/screen3_halos_s{args.seed}.npz")
    hp = hz["halo_pos"].astype(float); best = hz["best"]
    i, j = int(best[5]), int(best[6]); mw, m31 = hp[i], hp[j]
    virgo = c + sgdir(102.9, -2.3) * (16.5*H)
    void = c + sgdir(78.0, 74.0) * 17.2

    s_fine = embed_ic(s, args.Nfine, args.seed)
    t0 = time.time()
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    print(f"[viz] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)

    half = args.half; Ng = 260; cell = 2*half/Ng

    def slice_map(ax0, ax1, axp, cval, feats):
        m = np.all(np.abs(pos - c) < half, axis=1); q = pos[m] - c
        idx = np.floor((q[:, [ax0, ax1]] + half)/cell).astype(int)
        pz = q[:, axp]
        ok = np.all((idx >= 0) & (idx < Ng), axis=1) & (np.abs(pz) < 8.0)
        idx = idx[ok]
        f = np.zeros((Ng, Ng), np.float32); np.add.at(f, (idx[:, 0], idx[:, 1]), 1.0)
        f = gaussian_filter(f, args.smooth/cell); return f/f.mean() - 1.0

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(17, 8))
    lab = ["SGX", "SGY", "SGZ"]
    for panel, (a0, a1, ap_) in enumerate([(0, 1, 2), (1, 2, 0)]):
        d = slice_map(a0, a1, ap_, c, None)
        ax = axs[panel]
        im = ax.imshow(d.T, origin="lower", extent=[-half, half, -half, half],
                       cmap="inferno", vmin=-0.8, vmax=np.percentile(d, 99.5))
        ax.plot(0, 0, "w+", ms=15, mew=2.5, label="observer (us)")
        for p, mk, cl, nm in [(mw, "*", "cyan", "MW"), (m31, "*", "deepskyblue", "M31"),
                              (virgo, "o", "lime", "Virgo"), (void, "s", "white", "Local Void")]:
            pp = p - c
            ax.plot(pp[a0], pp[a1], mk, ms=13, mfc="none", mec=cl, mew=2.2)
            ax.annotate(nm, (pp[a0], pp[a1]), color=cl, fontsize=9,
                        xytext=(6, 6), textcoords="offset points")
        ax.set_xlabel(f"{lab[a0]}-c [$h^{{-1}}$Mpc]"); ax.set_ylabel(f"{lab[a1]}-c [$h^{{-1}}$Mpc]")
        ax.set_title(f"{lab[a0]}-{lab[a1]} plane (|{lab[ap_]}|<8)")
    fig.suptitle(f"CF4-constrained local universe (velocity-only, embed seed {args.seed})\n"
                 f"LG: MW-M31 sep 0.57 h$^{{-1}}$Mpc | Virgo 2.2e14 @ 12.4 | Local Void underdense +SGZ",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[viz] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
