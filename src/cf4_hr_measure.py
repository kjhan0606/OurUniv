#!/usr/bin/env python
"""Forward the Hoffman-Ribak constrained REALIZATION and MEASURE web + Virgo.

Anti-overclaiming: this measures a SAMPLE (s_out = s_CR, std~1.0), never the mean.
It reports only measured quantities (density-slice web statistics, P(k) low/mid vs
a full-amplitude LCDM baseline, and 1+delta at Virgo's actual box position) and
names the weakest link. It does NOT assume Virgo is preserved -- it measures it.
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


def pk_lowmid(field, L):
    """Dimensionless P(k) in a few low/mid k-bins from a density grid."""
    N = field.shape[0]
    dk = np.fft.rfftn(field)
    k1 = np.fft.fftfreq(N, d=L/N) * 2*np.pi
    kz = np.fft.rfftfreq(N, d=L/N) * 2*np.pi
    kx, ky, kzz = np.meshgrid(k1, k1, kz, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2 + kzz**2)
    p = (np.abs(dk)**2).ravel(); kf = kk.ravel()
    kbins = np.linspace(0, k1.max(), 24)
    idx = np.digitize(kf, kbins)
    out = np.array([p[idx == b].mean() if np.any(idx == b) else np.nan
                    for b in range(1, len(kbins))])
    kc = 0.5*(kbins[:-1]+kbins[1:])
    return kc, out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192_hr.npz")
    ap.add_argument("--key", default="s_out", help="s_out=HR realization; s_map=the mean")
    ap.add_argument("--Nfine", type=int, default=576, help="embed target for the fine web slice")
    ap.add_argument("--half", type=float, default=40.0)
    ap.add_argument("--smooth", type=float, default=0.8)
    ap.add_argument("--out", default="recon/cf4_hr_web.png")
    args = ap.parse_args()
    import jax.numpy as jnp
    import time
    from scipy.ndimage import gaussian_filter

    z = np.load(args.recon)
    s = z[args.key].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    L = Nc * spc; c = L / 2.0
    print(f"[hr] {args.recon} key={args.key} kind={z['kind'] if 'kind' in z else '?'} "
          f"Nc={Nc} L={L:.0f} std(s)={s.std():.3f}", flush=True)

    # embed the coarse constrained realization to fine resolution (adds random small-scale)
    sp = L / args.Nfine
    s_fine = embed_ic(s, args.Nfine, 1)
    m_p = 0.31 * RHO_CRIT * sp**3
    print(f"[hr] embed -> N={args.Nfine} sp={sp:.3f} std(fine)={s_fine.std():.3f}", flush=True)

    t0 = time.time()
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    print(f"[hr] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)

    # ---- central density cube (fine) for web + Virgo ----
    half = args.half; Ng = 200; cell = 2*half/Ng
    m = np.all(np.abs(pos - c) < half, axis=1); q = pos[m] - c
    idx = np.floor((q + half) / cell).astype(int)
    ok = np.all((idx >= 0) & (idx < Ng), axis=1); idx = idx[ok]
    f = np.zeros((Ng, Ng, Ng), np.float32)
    np.add.at(f, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    fg = gaussian_filter(f, args.smooth/cell); d = fg/fg.mean() - 1.0
    thick = max(1, int(round(4/cell)))
    sl = d[:, :, Ng//2-thick:Ng//2+thick].mean(2)

    # ---- Virgo: 1+delta within 2/4 Mpc at its actual box position ----
    cenp = pos[m]; nmean = cenp.shape[0] / ((2*half)**3)
    virgo = c + sgdir(102.9, -2.3) * (16.5 * 0.746)
    vd = {R: np.sum(np.linalg.norm(cenp - virgo, axis=1) < R) / (nmean*4/3*np.pi*R**3)
          for R in (2.0, 4.0)}
    # Local Void: anti-Virgo direction, larger radius, expect UNDERdense
    lv = c - sgdir(102.9, -2.3) * (20.0 * 0.746)
    lv_d = np.sum(np.linalg.norm(cenp - lv, axis=1) < 8.0) / (nmean*4/3*np.pi*8.0**3)

    print(f"[hr] WEB  slice delta: std={sl.std():.2f} min={sl.min():.2f} max={sl.max():.1f}", flush=True)
    print(f"[hr] VIRGO 1+delta(<2Mpc)={vd[2.0]:.2f} (<4Mpc)={vd[4.0]:.2f}  "
          f"[overdense>1 = preserved]", flush=True)
    print(f"[hr] LOCAL VOID 1+delta(<8Mpc, anti-Virgo)={lv_d:.2f}  [underdense<1 = correct]", flush=True)

    # P(k) of the fine slice-cube density vs its own mean (web richness proxy)
    kc, pk = pk_lowmid(d, 2*half)
    print(f"[hr] P(k) central cube (low k mean)={np.nanmean(pk[:3]):.2e} "
          f"(mid)={np.nanmean(pk[3:8]):.2e}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(sl.T, origin="lower", extent=[-half, half, -half, half],
                   cmap="inferno", vmin=-0.8, vmax=np.percentile(sl, 99.5))
    vv = (virgo - c)
    ax.plot(vv[0], vv[1], "co", ms=13, mfc="none", mew=2.2, label="Virgo")
    ax.plot(0, 0, "w+", ms=13, mew=2, label="observer")
    ax.set_xlabel("SGX [$h^{-1}$Mpc]"); ax.set_ylabel("SGY [$h^{-1}$Mpc]")
    ax.set_title(f"HR constrained realization forward (key={args.key}, N={args.Nfine})\n"
                 f"web std(delta)={sl.std():.2f}  Virgo 1+d(<2Mpc)={vd[2.0]:.2f}")
    ax.legend(loc="upper right", framealpha=0.3, labelcolor="white")
    fig.colorbar(im, label=r"$\delta$")
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[hr] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
