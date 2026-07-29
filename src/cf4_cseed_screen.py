#!/usr/bin/env python
"""Stage 1: linear-field screen of completion seeds for a central LG-like environment.

Free (no N-body forward): for each completion seed, build s_out = power_complete(s_map, cseed),
take the LINEAR density delta_lin = linear_modes(s_out), smooth at 4 h^-1Mpc, and score the
central environment -- a modest overdensity at the observer with no rich peak inside 8 Mpc
(isolated, LG-like). Keep the best seeds to forward in Stage 2.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf4_explicit_map import power_complete
from mock_pipeline import make_forward


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192.npz")
    ap.add_argument("--nseed", type=int, default=150)
    ap.add_argument("--smooth", type=float, default=4.0, help="Gaussian smoothing [h^-1Mpc]")
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--out", default="recon/cseed_screen.npz")
    args = ap.parse_args()
    import jax.numpy as jnp
    from pmwd import linear_modes
    from scipy.ndimage import gaussian_filter

    z = np.load(args.recon)
    s_map = z["s_map"].astype(np.float64); N = int(z["N"]); sp = float(z["spacing"]); c = N // 2
    conf, cosmo, _ = make_forward(N, sp, jnp.float32)
    ax = (np.arange(N) - c) * sp
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij"); r = np.sqrt(X**2 + Y**2 + Z**2)
    sg = args.smooth / sp

    def lin_delta(s_out):
        lin = np.asarray(linear_modes(jnp.asarray(s_out.reshape(N, N, N).astype(np.float32)),
                                      cosmo, conf, a=1.0, real=True), np.float64)
        d = gaussian_filter(lin, sg)
        return d / d.std()                              # normalized linear delta (amplitude-free)

    # Virgo direction (for selecting peaks offset PERPENDICULAR to LG->Virgo, per fable)
    def sgdir(l, b):
        l, b = np.radians(l), np.radians(b)
        return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])
    vhat = sgdir(102.9, -2.3)
    axc = (np.arange(N) - c) * sp
    XX, YY, ZZ = np.meshgrid(axc, axc, axc, indexing="ij")
    rows = []
    for cseed in range(1, args.nseed + 1):
        d = lin_delta(power_complete(s_map, N, cseed))
        inner = np.where(r < 8, d, -1e9)                 # strongest peak within 8 Mpc
        pk = np.unravel_index(np.argmax(inner), inner.shape)
        off = np.array([XX[pk], YY[pk], ZZ[pk]])         # peak offset from observer
        r_peak = np.linalg.norm(off); peak8 = inner[pk]
        radial = abs(np.dot(off, vhat))                  # offset component toward Virgo (want small)
        rows.append((cseed, r_peak, radial, peak8))
    rows = np.array(rows)
    r_peak = rows[:, 1]; radial = rows[:, 2]; peak8 = rows[:, 3]
    # want: peak modestly off-centre but PERPENDICULAR to Virgo (small radial), isolated (peak8<3)
    score = np.where((peak8 > 0.5) & (peak8 < 3.0) & (r_peak < 9),
                     radial + 0.15 * r_peak, 1e9)        # lower = better
    order = np.argsort(score)
    keep = [int(rows[i, 0]) for i in order if score[i] < 1e8][:args.keep]

    print(f"screened {args.nseed} completion seeds (peak offset perpendicular to LG->Virgo):")
    print(f"  {'cseed':>6} {'r_peak':>7} {'radial(Virgo)':>13} {'peak<8':>7}")
    for i in order[:min(args.keep + 4, len(order))]:
        s = "  <= keep" if int(rows[i, 0]) in keep else ""
        print(f"  {int(rows[i,0]):>6} {rows[i,1]:>7.1f} {rows[i,2]:>13.1f} {rows[i,3]:>7.2f}{s}")
    print(f"\nKEEP for forward (Virgo-perpendicular peaks): {keep}")
    np.savez(args.out, rows=rows, keep=np.array(keep))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
