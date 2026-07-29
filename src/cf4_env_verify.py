#!/usr/bin/env python
"""Verify the reconstructed large-scale environment: do known clusters land where they should?

Forwards the MAP reconstruction s_map (the realistic-amplitude field; the power-completed
s_out over-evacuates the centre) and checks whether the present-day density has an
overdensity at each catalogued local cluster, and an underdensity toward the Local Void.
Because the MAP amplitude is suppressed, we score by the density PERCENTILE within a shell
at the same radius (selection/amplitude independent), not the absolute delta.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import make_forward, RHO_CRIT

# supergalactic (SGL, SGB [deg], distance [Mpc]) of local structures
STRUCT = {
    "Virgo":      (102.9,  -2.3,  16.5),
    "Fornax":     (236.0, -44.0,  19.0),
    "Centaurus":  (156.0, -11.0,  45.0),
    "GreatAttr":  (155.0,  -6.0,  65.0),
    "Coma":       ( 89.0,   8.0,  90.0),
    "Perseus":    (340.0, -13.0,  73.0),
    "Hydra":      (139.0, -37.0,  53.0),
}
# a rough Local-Void direction (should be UNDERdense) -- opposite the Local-Group motion
VOID = ("LocalVoid", 15.0, 10.0, 25.0)


def sg(sgl, sgb, d):
    l, b = np.radians(sgl), np.radians(sgb)
    return d * np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192.npz")
    ap.add_argument("--field", default="s_map")
    ap.add_argument("--h", type=float, default=0.746)
    ap.add_argument("--smooth", type=float, default=4.0, help="density smoothing [h^-1Mpc]")
    ap.add_argument("--out", default="recon/cf4_env_verify.png")
    args = ap.parse_args()
    import jax.numpy as jnp
    from scipy.ndimage import gaussian_filter

    z = np.load(args.recon)
    s = z[args.field].astype(np.float32)
    N = int(z["N"]); sp = float(z["spacing"]); L = N * sp; c = L / 2.0
    print(f"[env] forwarding {args.field} {N}^3 @ {sp} h^-1Mpc", flush=True)
    conf, cosmo, fwd = make_forward(N, sp, jnp.float32)
    dens, _ = fwd(jnp.asarray(s.reshape(N, N, N))); dens.block_until_ready()
    d = gaussian_filter(np.asarray(dens, np.float64), args.smooth / sp)
    delta = d / d.mean() - 1.0

    ax = (np.arange(N) + 0.5) * sp - c
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)

    def score(name, sgl, sgb, dist):
        x = sg(sgl, sgb, dist) * args.h                  # box frame offset [h^-1Mpc]
        rc = np.linalg.norm(x)
        idx = np.round((x + c) / sp - 0.5).astype(int) % N
        dv = delta[idx[0], idx[1], idx[2]]
        shell = np.abs(R - rc) < 6.0                     # same-radius reference
        pct = 100.0 * (delta[shell] < dv).mean()
        return rc, dv, pct

    print(f"\n  {'structure':>10} {'r[h^-1Mpc]':>10} {'delta':>8} {'percentile':>11}")
    rows = []
    for nm, (sgl, sgb, dd) in STRUCT.items():
        rc, dv, pct = score(nm, sgl, sgb, dd)
        rows.append((nm, rc, dv, pct))
        flag = "OK" if pct > 75 else ("~" if pct > 55 else "LOW")
        print(f"  {nm:>10} {rc:>10.1f} {dv:>8.3f} {pct:>9.0f}%  {flag}")
    nm, sglv, sgbv, ddv = VOID
    rc, dv, pct = score(nm, sglv, sgbv, ddv)
    print(f"  {nm:>10} {rc:>10.1f} {dv:>8.3f} {pct:>9.0f}%  (expect LOW = underdense)")
    good = np.mean([r[3] > 70 for r in rows])
    print(f"\n[env] {int(good*len(rows))}/{len(rows)} clusters at >70th percentile "
          f"(overdense at their catalogued positions)", flush=True)

    # figure: two SG slices of the forwarded density with structures marked
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ci = N // 2; thick = int(round(6 / sp))
    sl_xy = delta[:, :, ci-thick:ci+thick].mean(2)
    sl_xz = delta[:, ci-thick:ci+thick, :].mean(1)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.2))
    ext = [-c, c, -c, c]; vlim = np.percentile(np.abs(delta), 99)
    for a, (slc, lab) in zip(axs, [(sl_xy, ("SGX", "SGY", 2)), (sl_xz, ("SGX", "SGZ", 1))]):
        im = a.imshow(slc.T, origin="lower", extent=ext, cmap="RdBu_r",
                      vmin=-vlim, vmax=vlim, aspect="equal")
        a.plot(0, 0, "+", color="lime", ms=14, mew=2.5)
        for nm, (sgl, sgb, dd) in STRUCT.items():
            x = sg(sgl, sgb, dd) * args.h
            ax3 = [x[0], x[1], x[2]]
            if abs(ax3[lab[2]]) < 30:
                a.plot(ax3[0], ax3[1] if lab[1] == "SGY" else x[2], "k*", ms=10)
                a.annotate(nm, (ax3[0], ax3[1] if lab[1] == "SGY" else x[2]), fontsize=8)
        a.set_xlim(-150, 150); a.set_ylim(-150, 150)
        a.set_xlabel(lab[0]); a.set_ylabel(lab[1])
        a.set_title(f"forwarded {args.field}: {lab[0]}-{lab[1]} slice")
    fig.colorbar(im, ax=axs, label=r"$\delta$ (smoothed)", shrink=0.8)
    fig.suptitle(f"Large-scale environment check: known clusters (stars) on the reconstructed "
                 f"present density ({args.field})", fontsize=12)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"[env] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
