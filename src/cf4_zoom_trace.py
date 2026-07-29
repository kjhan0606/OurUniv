#!/usr/bin/env python
"""Trace the winning LG analog back to its Lagrangian region and size the zoom.

Given a winning embed seed (from cf4_lg_screen3) and its MW-M31 pair, re-forward that seed,
select z=0 particles within R_sel of the LG midpoint, map them to their INITIAL (Lagrangian)
grid positions (pmwd particle k starts at unravel_index(k, (N,N,N))*sp), and measure the
Lagrangian volume a zoom must refine + the high-res particle budget at target m_p.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import make_forward, RHO_CRIT, VUNIT_KMS
from cf4_make_ic import embed_ic


def lagrangian_of(idx, N, sp):
    ix, iy, iz = np.unravel_index(idx, (N, N, N))
    return np.stack([ix, iy, iz], 1).astype(np.float64) * sp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192_hr.npz")
    ap.add_argument("--key", default="s_out")
    ap.add_argument("--seed", type=int, required=True, help="winning embed seed")
    ap.add_argument("--halos", default=None, help="screen3_halos_s{seed}.npz (for the LG pair)")
    ap.add_argument("--rsel", type=float, default=5.0, help="z=0 selection radius around LG [h^-1Mpc]")
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import jax.numpy as jnp
    import time

    halos_f = args.halos or f"recon/screen3_halos_s{args.seed}.npz"
    hz = np.load(halos_f)
    hp = hz["halo_pos"].astype(float); best = hz["best"]
    L = float(hz["L"]); sp0 = float(hz["sp"]); c = L / 2.0
    i, j = int(best[5]), int(best[6])
    mid = 0.5 * (hp[i] + hp[j])
    print(f"[zoom] seed={args.seed} LG pair sep={best[0]:.2f} M=({best[1]:.1e},{best[2]:.1e}) "
          f"mid=({mid[0]:.1f},{mid[1]:.1f},{mid[2]:.1f}) r_off={best[3]:.1f}", flush=True)

    z = np.load(args.recon)
    s = z[args.key].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    N = args.Nfine; sp = L / N; m_p = args.Om * RHO_CRIT * sp ** 3
    s_fine = embed_ic(s, N, args.seed)
    t0 = time.time()
    conf, cosmo, fwd = make_forward(N, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(N, N, N)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)      # ordered by particle index
    print(f"[zoom] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s (m_p={m_p:.2e})", flush=True)

    # z=0 selection around the LG, plus everything that will fall into R_sel
    sel = np.flatnonzero(np.linalg.norm(pos - mid, axis=1) < args.rsel)
    lag = lagrangian_of(sel, N, sp)                       # initial positions
    print(f"[zoom] {len(sel)} particles within {args.rsel} h^-1Mpc of the LG at z=0", flush=True)

    # Lagrangian region geometry
    lo = lag.min(0); hi = lag.max(0); ext = hi - lo
    lcen = 0.5 * (lo + hi)
    r_lag = np.linalg.norm(lag - lcen, axis=1)
    R_eff = np.percentile(r_lag, 99.5)                    # robust Lagrangian radius
    bbox_vol = np.prod(ext)
    sphere_vol = 4/3 * np.pi * R_eff ** 3
    print(f"[zoom] LAGRANGIAN region: bbox extent = "
          f"({ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}) h^-1Mpc, "
          f"R_eff(99.5%)={R_eff:.1f} h^-1Mpc", flush=True)
    print(f"[zoom]   bbox volume={bbox_vol:.0f}, sphere volume={sphere_vol:.0f} (h^-1Mpc)^3", flush=True)

    # high-res particle budget at target m_p (Lagrangian volume * mean density / m_p)
    rho_m = args.Om * RHO_CRIT                            # Msun/h per (h^-1Mpc)^3
    print(f"[zoom] high-res particle budget (Lagrangian sphere R_eff, buffer x1.5 linear):", flush=True)
    Vbuf = 4/3 * np.pi * (1.5 * R_eff) ** 3
    for tmp in (1e6, 1e5, 1e4):
        Nhr = Vbuf * rho_m / tmp
        # equivalent uniform grid level over the whole box that would give this m_p
        lvl = np.log2((rho_m * L**3 / tmp) ** (1/3))
        print(f"   m_p={tmp:.0e} Msun/h: {Nhr:.2e} zoom particles (R_eff x1.5 sphere), "
              f"~equiv full-box grid 2^{lvl:.1f}", flush=True)

    out = args.out or f"recon/cf4_zoom_trace_s{args.seed}.png"
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    # present-day
    ax[0].scatter(pos[sel, 0]-c, pos[sel, 1]-c, s=1, alpha=0.2, color="steelblue")
    ax[0].plot(hp[i, 0]-c, hp[i, 1]-c, "r*", ms=13); ax[0].plot(hp[j, 0]-c, hp[j, 1]-c, "r*", ms=13)
    ax[0].plot(0, 0, "k+", ms=12, mew=2)
    ax[0].set_title(f"z=0: LG selection (R_sel={args.rsel}) seed {args.seed}")
    ax[0].set_xlabel("SGX-c"); ax[0].set_ylabel("SGY-c"); ax[0].set_aspect("equal")
    # Lagrangian
    ax[1].scatter(lag[:, 0]-lcen[0], lag[:, 1]-lcen[1], s=1, alpha=0.2, color="darkorange")
    th = np.linspace(0, 2*np.pi, 100)
    ax[1].plot(R_eff*np.cos(th), R_eff*np.sin(th), "k--", lw=1, label=f"R_eff={R_eff:.1f}")
    ax[1].set_title(f"Lagrangian region (zoom volume)\nextent {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f}")
    ax[1].set_xlabel("Lag X-cen"); ax[1].set_ylabel("Lag Y-cen"); ax[1].set_aspect("equal"); ax[1].legend()
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"[zoom] saved {out}", flush=True)

    np.savez(f"recon/cf4_zoom_trace_s{args.seed}.npz",
             seed=args.seed, mid=mid, rsel=args.rsel, lag_lo=lo, lag_hi=hi, lag_cen=lcen,
             R_eff=R_eff, ext=ext, n_sel=len(sel), m_p=m_p, L=L, N=N)
    print(f"[zoom] saved recon/cf4_zoom_trace_s{args.seed}.npz", flush=True)


if __name__ == "__main__":
    main()
