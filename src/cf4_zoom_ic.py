#!/usr/bin/env python
"""Nested mass-refined GRAFIC zoom IC for lagRAMSES (Scheme A), preserving cr6.e19.

P0 (plumbing validation): build a coarse full-box level + nested sub-box levels by
Fourier-projecting the running 1024^3 parent (ic_cr6_e19_1024/level_010), plus a geometric
Lagrangian ic_refmap sphere. No pmwd/transfer needed here:
  - hydro=off => ic_deltab is header-only for N-body (positions come from ic_velc*),
  - coarse+equal-or-lower resolution than the parent => retain the exact shared
    physical Fourier modes (no fresh small-scale power to synthesize).
P2 (science, levels finer than the parent) will instead synthesize fresh small-scale
power via embed_ic + a transfer function measured from the parent; that is a separate
code path added later.

lagRAMSES Scheme A layout produced here (see memory lagramses-zoom-ic-convention):
  level_00L/ic_deltab  ic_velcx ic_velcy ic_velcz  ic_refmap
levelmin level is full box (offset 0); finer levels are sub-boxes (nonzero xoff).
Particle mass is set per level automatically by RAMSES (mp = 0.5^(3*level)).
"""
import os
import sys
import argparse
import numpy as np
from scipy import fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grafic_io as G


def block_downsample(a, N):
    """Mean-pool (M,M,M) -> (N,N,N), M a multiple of N (mode-preserving enough for P0)."""
    M = a.shape[0]
    assert M % N == 0, (M, N)
    r = M // N
    return a.reshape(N, r, N, r, N, r).mean(axis=(1, 3, 5))


def fourier_resample_field(a, N, workers=-1):
    """Sample the same periodic physical field on a smaller Fourier grid.

    Unlike mean pooling, this preserves the amplitude and phase of every
    retained physical mode.  The ``(N/M)^3`` factor converts between the
    unnormalised FFT conventions for two sampled physical fields; it is not
    the ``(N/M)^1.5`` white-noise normalization used for random fields.
    """
    a = np.asarray(a)
    M = a.shape[0]
    if a.ndim != 3 or a.shape != (M, M, M):
        raise ValueError("input must be a cubic 3-D field")
    if N > M or N <= 0 or N % 2:
        raise ValueError("N must be a positive even integer no larger than input")
    if N == M:
        return a.astype(np.float32, copy=False)
    source = sfft.rfftn(a.astype(np.float32, copy=False), workers=workers)
    half = N // 2
    index = np.r_[0:half, M - half:M]
    target = source[np.ix_(index, index, np.arange(half + 1))].copy()
    target *= np.float32((float(N) / M) ** 3)
    return sfft.irfftn(target, s=(N, N, N), axes=(0, 1, 2), workers=workers).astype(np.float32)


def snap_subbox(N, Nmin, L_hmpc, center_hmpc, half_hmpc):
    """Integer cell window [i0,i1) on the N^3 level grid enclosing center +- half,
    snapped to the coarse (Nmin) grid so finer levels nest cleanly. Cubic, same each axis."""
    dx = L_hmpc / N
    stride = N // Nmin                                    # cells per coarse cell
    i0 = int(np.floor((center_hmpc - half_hmpc) / dx))
    i1 = int(np.ceil((center_hmpc + half_hmpc) / dx))
    i0 = max(0, (i0 // stride) * stride)                 # snap down to coarse grid
    i1 = min(N, ((i1 + stride - 1) // stride) * stride)  # snap up
    return i0, i1, dx


def build_refmap(i0, i1, dx, center_hmpc, R_hmpc):
    """1.0 where a cell's Lagrangian (grid) position is within R of center, else 0."""
    coord = (np.arange(i0, i1) + 0.5) * dx - center_hmpc
    X, Y, Z = np.meshgrid(coord, coord, coord, indexing="ij")
    return (X * X + Y * Y + Z * Z <= R_hmpc * R_hmpc).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", default="/gpfs/kjhan/Hydro/CF4_LG/ic_cr6_e19_1024/level_010")
    ap.add_argument("--out", default="/gpfs/kjhan/Hydro/CF4_LG/zoom_p0")
    ap.add_argument("--levelmin", type=int, default=7)
    ap.add_argument("--levelmax-ic", type=int, default=9, help="finest IC level (<= parent level 10)")
    ap.add_argument("--box-hmpc", type=float, default=384.0)
    ap.add_argument("--h", type=float, default=0.746)
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--OL", type=float, default=0.69)
    ap.add_argument("--astart", type=float, default=0.02)
    ap.add_argument("--center-hmpc", type=float, default=192.0, help="LG Lagrangian centre (box centre)")
    ap.add_argument("--mask-R-hmpc", type=float, default=6.0, help="high-res sphere radius")
    ap.add_argument("--subbox-half-hmpc", type=float, default=24.0, help="nested sub-box half-width")
    ap.add_argument("--write-deltab-zeros", action="store_true",
                    help="write ic_deltab as zeros (valid header; unused for hydro=off) to skip the parent delta read")
    args = ap.parse_args()

    Lc = args.levelmin
    Nmin = 2 ** Lc
    Lbox_mpc = args.box_hmpc / args.h
    h0 = 100.0 * args.h
    print(f"[zoom] parent={args.parent}")
    print(f"[zoom] levels {Lc}..{args.levelmax_ic}  box={args.box_hmpc} h^-1Mpc ({Lbox_mpc:.2f} Mpc)  "
          f"centre={args.center_hmpc}  mask R={args.mask_R_hmpc}  subbox half={args.subbox_half_hmpc}", flush=True)

    # ---- read parent velocity fields once (1024^3); delta optional ----
    print("[zoom] reading parent ic_velcx/y/z ...", flush=True)
    vx1, _ = G.read_grafic_field(os.path.join(args.parent, "ic_velcx"))
    vy1, _ = G.read_grafic_field(os.path.join(args.parent, "ic_velcy"))
    vz1, _ = G.read_grafic_field(os.path.join(args.parent, "ic_velcz"))
    Npar = vx1.shape[0]
    print(f"[zoom] parent N={Npar}  v_rms=({vx1.std():.1f},{vy1.std():.1f},{vz1.std():.1f}) km/s", flush=True)
    if args.write_deltab_zeros:
        d1 = None
    else:
        print("[zoom] reading parent ic_deltab ...", flush=True)
        d1, _ = G.read_grafic_field(os.path.join(args.parent, "ic_deltab"))

    for L in range(Lc, args.levelmax_ic + 1):
        N = 2 ** L
        assert N <= Npar, f"P0 only supports levels <= parent ({Npar}); level {L} => {N}"
        dxg = (Lbox_mpc) / N                              # GRAFIC comoving Mpc cell
        vx = fourier_resample_field(vx1, N)
        vy = fourier_resample_field(vy1, N)
        vz = fourier_resample_field(vz1, N)
        dd = fourier_resample_field(d1, N) if d1 is not None else None

        if L == Lc:
            i0, i1 = 0, N
        else:
            i0, i1, _ = snap_subbox(N, Nmin, args.box_hmpc, args.center_hmpc, args.subbox_half_hmpc)
        off_hmpc = i0 * (args.box_hmpc / N)
        off_mpc = off_hmpc / args.h
        sl = slice(i0, i1)
        vxs, vys, vzs = vx[sl, sl, sl], vy[sl, sl, sl], vz[sl, sl, sl]
        dds = (dd[sl, sl, sl] if dd is not None else np.zeros_like(vxs))
        refmap = build_refmap(i0, i1, args.box_hmpc / N, args.center_hmpc, args.mask_R_hmpc)

        outdir = os.path.join(args.out, f"level_{L:03d}")
        os.makedirs(outdir, exist_ok=True)
        off3 = (off_mpc, off_mpc, off_mpc)
        wargs = (dxg, off3, args.astart, args.Om, args.OL, h0)
        G.write_grafic_field(os.path.join(outdir, "ic_deltab"), dds, *wargs)
        G.write_grafic_field(os.path.join(outdir, "ic_velcx"), vxs, *wargs)
        G.write_grafic_field(os.path.join(outdir, "ic_velcy"), vys, *wargs)
        G.write_grafic_field(os.path.join(outdir, "ic_velcz"), vzs, *wargs)
        G.write_grafic_field(os.path.join(outdir, "ic_refmap"), refmap, *wargs)
        nref = int(refmap.sum())
        print(f"[zoom] level {L:2d}  N={N:4d}  sub=[{i0}:{i1}] ({i1-i0}^3)  off={off_mpc:.3f} Mpc  "
              f"dx={dxg:.4f} Mpc  refmap_cells={nref}  -> {outdir}", flush=True)

    # namelist snippet
    print("\n[zoom] &INIT_PARAMS  filetype='grafic'")
    for i, L in enumerate(range(Lc, args.levelmax_ic + 1)):
        print(f"[zoom]   initfile({L})='{os.path.join(args.out, f'level_{L:03d}')}'")
    print(f"[zoom] &AMR_PARAMS  levelmin={Lc}  levelmax>={args.levelmax_ic}")
    print(f"[zoom] &REFINE_PARAMS  ivar_refine=0  (ic_refmap present -> Scheme A)")


if __name__ == "__main__":
    main()
