#!/usr/bin/env python
"""Comprehensive linear-field environment screen of completion seeds (fable's full recipe).

For each completion seed, from the LINEAR field (free, no N-body) score the environment on:
  - Virgo geometry : the central density peak offset, and the resulting LG->Virgo distance
  - Virgocentric infall : the linear (Zel'dovich) velocity at the peak, projected toward Virgo
  - Local Void : underdensity on the ANTI-Virgo side at 10-30 h^-1Mpc
  - isolation : no cluster-scale peak inside r<8 (kept as a soft gate)
Soft-rank and keep a pool; the expensive FoF forwards + final chi^2 come later.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf4_explicit_map import power_complete
from mock_pipeline import make_forward, VUNIT_KMS

VIRGO_D = 16.5 * 0.746   # ~12.3 h^-1Mpc


def sgdir(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192.npz")
    ap.add_argument("--nseed", type=int, default=200)
    ap.add_argument("--smooth", type=float, default=4.0)
    ap.add_argument("--keep", type=int, default=12)
    ap.add_argument("--out", default="recon/env_screen.npz")
    args = ap.parse_args()
    import jax.numpy as jnp
    from pmwd import linear_modes, lpt
    from scipy.ndimage import gaussian_filter

    z = np.load(args.recon)
    s_map = z["s_map"].astype(np.float64); N = int(z["N"]); sp = float(z["spacing"]); c = N // 2
    conf, cosmo, _ = make_forward(N, sp, jnp.float32)
    vhat = sgdir(102.9, -2.3)
    axc = (np.arange(N) - c) * sp
    XX, YY, ZZ = np.meshgrid(axc, axc, axc, indexing="ij")
    r = np.sqrt(XX**2 + YY**2 + ZZ**2)
    sig = args.smooth / sp
    # anti-Virgo cone (Local Void should live here) at 10-30 Mpc
    with np.errstate(invalid="ignore"):
        cosang = -(XX*vhat[0] + YY*vhat[1] + ZZ*vhat[2]) / (r + 1e-9)
    void_reg = (r > 10) & (r < 30) & (cosang > 0.5)

    def env(s_out):
        sk = jnp.asarray(s_out.reshape(N, N, N).astype(np.float32))
        d = gaussian_filter(np.asarray(linear_modes(sk, cosmo, conf, a=1.0, real=True), np.float64), sig)
        d /= d.std()
        ptcl, _ = lpt(linear_modes(sk, cosmo, conf), cosmo, conf)
        v = np.asarray(ptcl.vel, np.float64).reshape(N, N, N, 3) * VUNIT_KMS
        v = np.stack([gaussian_filter(v[..., i], sig) for i in range(3)], -1)
        inner = np.where(r < 8, d, -1e9)
        pk = np.unravel_index(np.argmax(inner), inner.shape)
        off = np.array([XX[pk], YY[pk], ZZ[pk]]); r_peak = np.linalg.norm(off)
        lg_virgo = np.linalg.norm(vhat * VIRGO_D - off)
        infall = float(np.dot(v[pk], vhat))          # linear infall toward Virgo (km/s)
        void = float(d[void_reg].mean())              # anti-Virgo density (want < 0)
        return r_peak, lg_virgo, infall, void, float(inner[pk])

    rows = []
    for cseed in range(1, args.nseed + 1):
        rows.append((cseed,) + env(power_complete(s_map, N, cseed)))
    rows = np.array(rows)
    cs, r_peak, lg_virgo, infall, void, peak8 = rows.T
    # soft score (lower = better): Virgo distance ~12.3, high infall, anti-Virgo void, isolated
    score = (((lg_virgo - VIRGO_D) / 3.0) ** 2
             - infall / 60.0                          # reward toward-Virgo infall
             + np.maximum(void + 0.1, 0) * 3.0        # reward underdense anti-Virgo (Local Void)
             + np.where((peak8 > 0.5) & (peak8 < 3.0), 0.0, 5.0))  # soft isolation gate
    order = np.argsort(score)
    keep = [int(cs[i]) for i in order[:args.keep]]

    print(f"env screen ({args.nseed} completion seeds): targets LG-Virgo~{VIRGO_D:.1f}, "
          f"high infall, anti-Virgo void<0")
    print(f"  {'cseed':>6} {'score':>6} {'LG-Vir':>7} {'infall_lin':>10} {'void':>6} {'peak8':>6}")
    for i in order[:args.keep + 4]:
        s = "  <=keep" if int(cs[i]) in keep else ""
        print(f"  {int(cs[i]):>6} {score[i]:>6.2f} {lg_virgo[i]:>7.1f} {infall[i]:>10.0f} "
              f"{void[i]:>6.2f} {peak8[i]:>6.2f}{s}")
    print(f"\nKEEP pool: {keep}")
    np.savez(args.out, rows=rows, keep=np.array(keep))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
