#!/usr/bin/env python
"""Export a recovered initial field s -> GRAFIC1 ICs for lagRAMSES.

Takes a whitened initial field s (a diffusion posterior *sample* -- a full-power,
data-consistent realization, NOT the smoothed posterior mean) on a full periodic
cubic box and writes RAMSES-ready GRAFIC1 files:

  ic_deltab           linear overdensity delta(a_start)      via pmwd linear_modes
  ic_velcx/y/z        peculiar velocity (km/s, proper)       via 2LPT (grafic_io)

delta(a_start) uses the correct LCDM transfer function (pmwd linear_modes at a_start,
real space). Velocities are 2LPT (Psi1 + 3/7 Psi2; user directive) built from that
same delta so the density and velocity ICs are mutually consistent. Growth rate
f1 = dlnD/dlna and H(a_start) come from the pmwd cosmology.

Usage:
  python cf4_export_grafic.py --s-npz recon/cf4_diff_n64.npz --key post_sample --obj 0 \
      --N 64 --spacing 2.0 --astart 0.02 --out recon/ic_grafic_obj0
  python cf4_export_grafic.py --s-npy s_field.npy --N 192 --spacing 2.0 --out ic_dir
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import numpy as np


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import grafic_io as G

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--s-npz", help="npz holding the s field (see --key/--obj)")
    src.add_argument("--s-npy", help="npy of a single (N,N,N) s field")
    ap.add_argument("--key", default="post_sample", help="npz key for the s field(s)")
    ap.add_argument("--obj", type=int, default=0, help="index if the npz key is stacked")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--spacing", type=float, required=True, help="Mpc/h per cell")
    # CF4 cosmology (MUST match cf4_explicit_map, which reconstructed s at these values).
    # pmwd SimpleLCDM defaults (Om 0.30, h 0.70, A_s 2.0e-9) are WRONG for this field.
    ap.add_argument("--Om", type=float, default=0.31, help="Omega_m (CF4: 0.31)")
    ap.add_argument("--h", type=float, default=0.746, help="little-h (CF4: 0.746)")
    ap.add_argument("--A-s-1e9", type=float, default=1.63, help="A_s in 1e-9 (CF4: 1.63)")
    ap.add_argument("--astart", type=float, default=0.02, help="IC start scale factor")
    ap.add_argument("--out", required=True, help="output GRAFIC directory")
    ap.add_argument("--offset-mpc", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    args = ap.parse_args()

    if args.s_npy:
        s = np.load(args.s_npy).astype(np.float32)
    else:
        with np.load(args.s_npz) as z:
            arr = z[args.key]
            s = (arr[args.obj] if arr.ndim == 4 else arr).astype(np.float32)
    N = args.N
    assert s.shape == (N, N, N), f"s shape {s.shape} != {(N,N,N)}"
    print(f"[export] s: shape={s.shape} mean={s.mean():.3f} std={s.std():.3f}")

    import jax, jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, growth
    conf = Configuration(ptcl_spacing=float(args.spacing), ptcl_grid_shape=(N,) * 3,
                         mesh_shape=1, float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=args.Om, h=args.h, A_s_1e9=args.A_s_1e9), conf)
    Om = float(cosmo.Omega_m); OL = 1.0 - Om
    h = float(cosmo.h)
    a = float(args.astart)

    # linear overdensity at a_start (real space), correct LCDM P(k)
    delta = np.array(linear_modes(jnp.asarray(s), cosmo, conf, a=a, real=True),
                     np.float64)
    delta -= delta.mean()

    # growth rate f1 and H(a) for the 2LPT velocity normalization
    D = float(growth(a, cosmo, conf, order=1, deriv=0))
    dD = float(growth(a, cosmo, conf, order=1, deriv=1))
    f1 = dD / D
    H0 = 100.0 * h                                         # km/s/Mpc
    H_a = H0 * np.sqrt(Om / a ** 3 + OL)                   # km/s/Mpc
    L_mpc_h = N * args.spacing
    L_mpc = L_mpc_h / h
    print(f"[export] cosmo Om={Om:.3f} h={h:.3f} | a_start={a} D={D:.4f} f1={f1:.3f} "
          f"H(a)={H_a:.1f} km/s/Mpc | box={L_mpc_h:.0f} Mpc/h ({L_mpc:.1f} Mpc) "
          f"dx={L_mpc/N:.3f} Mpc")
    print(f"[export] delta(a_start): std={delta.std():.3e} (expect ~D*sigma_lin)")

    vx, vy, vz = G.lpt2_velocity(delta, L_mpc, a, H_a, f1)
    print(f"[export] 2LPT velocity rms: vx={vx.std():.2f} vy={vy.std():.2f} "
          f"vz={vz.std():.2f} km/s")

    info = G.write_grafic_ic(args.out, delta.astype(np.float32), vx, vy, vz,
                             L_mpc_h=L_mpc_h, h=h, astart=a, omega_m=Om, omega_l=OL,
                             offset_mpc=tuple(args.offset_mpc))
    files = sorted(os.listdir(args.out))
    print(f"[export] wrote {files} -> {args.out}")
    print(f"[export] GRAFIC: N={info['N']} dx={info['dx_mpc']:.4f} Mpc "
          f"box={info['box_mpc']:.2f} Mpc h0={info['h0']:.2f} astart={info['astart']}")
    # roundtrip sanity
    d2, meta = G.read_grafic_field(os.path.join(args.out, "ic_deltab"))
    print(f"[export] readback ic_deltab: max|err|={np.abs(d2-delta.astype(np.float32)).max():.2e} "
          f"header dx={meta['dx']:.4f} astart={meta['astart']}")


if __name__ == "__main__":
    main()
