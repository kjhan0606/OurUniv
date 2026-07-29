#!/usr/bin/env python
"""Baseline: does forwarding a FULL-amplitude LambdaCDM field produce a filamentary web?

Per fable: our web was missing because we forwarded the posterior MEAN (MAP, suppressed
variance), not a full-variance SAMPLE. This measures the baseline -- forward a pure unit-
variance white-noise IC (full P(k)) and render a fine-resolution density slice. If a web
appears, the diagnosis holds (a proper full-amplitude IC gives a web; our embed did not).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import make_forward, RHO_CRIT


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=576)
    ap.add_argument("--L", type=float, default=384.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smooth", type=float, default=0.8, help="display smoothing [h^-1Mpc]")
    ap.add_argument("--out", default="recon/cf4_web_baseline.png")
    args = ap.parse_args()
    import jax.numpy as jnp
    import time
    from scipy.ndimage import gaussian_filter

    N = args.N; L = args.L; sp = L / N; c = L / 2.0
    rng = np.random.default_rng(args.seed)
    s = rng.standard_normal((N, N, N)).astype(np.float32)   # FULL-amplitude unit-variance IC
    print(f"[web] pure LCDM IC N={N} sp={sp:.3f} std(s)={s.std():.3f} (full amplitude)", flush=True)
    t0 = time.time()
    conf, cosmo, fwd = make_forward(N, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s))
    pos = np.asarray(ptcl.pos(), np.float64)
    print(f"[web] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)

    # central +-40 Mpc density on a fine grid, thin slice
    half = 40.0; Ng = 200; cell = 2 * half / Ng
    m = np.all(np.abs(pos - c) < half, axis=1); q = pos[m] - c
    idx = np.floor((q + half) / cell).astype(int)
    ok = np.all((idx >= 0) & (idx < Ng), axis=1); idx = idx[ok]
    f = np.zeros((Ng, Ng, Ng), np.float32); np.add.at(f, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    f = gaussian_filter(f, args.smooth / cell); d = f / f.mean() - 1.0
    thick = max(1, int(round(4 / cell)))
    sl = d[:, :, Ng//2 - thick:Ng//2 + thick].mean(2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(sl.T, origin="lower", extent=[-half, half, -half, half],
                   cmap="inferno", vmin=-0.8, vmax=np.percentile(sl, 99.5))
    ax.set_xlabel("SGX [$h^{-1}$Mpc]"); ax.set_ylabel("SGY [$h^{-1}$Mpc]")
    ax.set_title(f"Baseline: forward of a FULL-amplitude LCDM IC (N={N}, {args.smooth:.1f} Mpc smooth)\n"
                 f"filamentary web = a proper IC works; our embed had none")
    fig.colorbar(im, label=r"$\delta$")
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    # quantify: power at large scales (should be full, unlike the embed)
    print(f"[web] density slice delta: std={sl.std():.2f} min={sl.min():.2f} max={sl.max():.1f}", flush=True)
    print(f"[web] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
