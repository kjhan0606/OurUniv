#!/usr/bin/env python
"""Measure the effective transfer T_eff(k) = delta_hat / s_hat from the parent IC (pmwd-free).

The parent ic_deltab was made by delta = linear_modes(s) at a_start, an isotropic linear
filter of the white-noise field s. So per spherical shell:

    T_eff(k) = Sum Re(dhat . conj(shat)) / Sum |shat|^2      (least-squares transfer)
    r(k)     = Sum Re(dhat . conj(shat)) / sqrt(Sum|dhat|^2 . Sum|shat|^2)   (phase alignment)

If the model holds, r(k) ~ 1 across the band (delta is just s filtered). T_eff(k) then lets
us synthesize delta at ANY resolution from a mode-consistent white-noise field (embed_ic),
including levels FINER than the parent by extrapolating T_eff past the parent Nyquist with a
smooth (log-log quadratic) tail fit. This replaces the missing pmwd for the zoom.
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grafic_io as G


def block_down(a, N):
    M = a.shape[0]; r = M // N
    return a.reshape(N, r, N, r, N, r).mean(axis=(1, 3, 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deltab", default="/gpfs/kjhan/Hydro/CF4_LG/ic_cr6_e19_1024/level_010/ic_deltab")
    ap.add_argument("--sfield", default="/gpfs/kjhan/Hydro/CF4_LG/ic_src/s_cr6_e19_1024.npy")
    ap.add_argument("--skey", default="s_fine", help="array key when --sfield is NPZ")
    ap.add_argument("--box-hmpc", type=float, default=384.0)
    ap.add_argument("--N", type=int, default=0, help="block-downsample to N for a quick check (0=full)")
    ap.add_argument("--out", default="/gpfs/kjhan/Hydro/CF4_LG/tier1/transfer_cr6_e19.npz")
    args = ap.parse_args()

    print("[T] loading delta (grafic) ...", flush=True)
    d, meta = G.read_grafic_field(args.deltab)
    d = d.astype(np.float32)
    print(f"[T]   delta N={d.shape[0]} std={d.std():.4e}", flush=True)
    print("[T] loading s (npy) ...", flush=True)
    loaded = np.load(args.sfield)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            s = loaded[args.skey].astype(np.float32)
        finally:
            loaded.close()
    else:
        s = loaded.astype(np.float32)
    print(f"[T]   s N={s.shape[0]} std={s.std():.4e}", flush=True)
    assert d.shape == s.shape, (d.shape, s.shape)
    Nfull = d.shape[0]
    if args.N and args.N < Nfull:
        d = block_down(d.astype(np.float64), args.N).astype(np.float32)
        s = block_down(s.astype(np.float64), args.N).astype(np.float32)
    N = d.shape[0]
    print(f"[T] measuring at N={N} ...", flush=True)

    dk = np.fft.rfftn(d); sk = np.fft.rfftn(s)
    kx = np.fft.fftfreq(N) * N; kr = np.fft.rfftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kr, indexing="ij")
    kbin = np.round(np.sqrt(KX * KX + KY * KY + KZ * KZ)).astype(np.int32)
    nb = int(kbin.max()) + 1
    w = (dk * np.conj(sk)).real.ravel().astype(np.float64)
    cross = np.bincount(kbin.ravel(), weights=w, minlength=nb)
    ss = np.bincount(kbin.ravel(), weights=(np.abs(sk) ** 2).ravel().astype(np.float64), minlength=nb)
    dd = np.bincount(kbin.ravel(), weights=(np.abs(dk) ** 2).ravel().astype(np.float64), minlength=nb)
    cnt = np.bincount(kbin.ravel(), minlength=nb)

    kf = 2.0 * np.pi / args.box_hmpc                     # fundamental [h/Mpc]
    kphys = np.arange(nb) * kf
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(ss > 0, cross / ss, 0.0)
        r = np.where((ss > 0) & (dd > 0), cross / np.sqrt(dd * ss), 0.0)

    kNyq = (N // 2) * kf
    good = (cnt > 0) & (kphys > 0) & (T > 0)
    band = good & (kphys > 0.3 * kNyq) & (kphys <= kNyq)   # tail for the fit
    print(f"[T] kf={kf:.4f} h/Mpc  kNyq={kNyq:.3f} h/Mpc  nbins={nb}", flush=True)
    print(f"[T] phase alignment r(k): median={np.median(r[good]):.4f} "
          f"min={r[good].min():.3f} (expect ~1 if linear_modes model holds)", flush=True)
    # log-log quadratic tail fit for extrapolation beyond kNyq
    lx = np.log10(kphys[band]); ly = np.log10(T[band])
    coeff = np.polyfit(lx, ly, 2)
    print(f"[T] tail fit log10 T = {coeff[0]:.3f}(logk)^2 + {coeff[1]:.3f} logk + {coeff[2]:.3f} "
          f"over k=[{kphys[band].min():.2f},{kphys[band].max():.2f}] h/Mpc", flush=True)

    # sample the fit a bit beyond, as a sanity print
    for kk in (kNyq, 2 * kNyq, 4 * kNyq):
        lT = np.polyval(coeff, np.log10(kk))
        print(f"[T]   extrap T({kk:.2f} h/Mpc) = {10 ** lT:.3e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, kphys=kphys, T=T, r=r, cnt=cnt, kf=kf, kNyq=kNyq,
             N=N, box_hmpc=args.box_hmpc, tail_coeff=coeff)
    print(f"[T] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
