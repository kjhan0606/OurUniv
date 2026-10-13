#!/usr/bin/env python
"""Constrained fine-resolution initial conditions from the 2 h^-1Mpc CF4 reconstruction.

The initial field s is white noise, so a fine IC is trivial to build: keep the constrained
large-scale Fourier modes from the 2 h^-1Mpc reconstruction and fill the unconstrained
small-scale modes with fresh white noise. This is the standard IC refinement (MUSIC/genetIC).

  S_fine(k) = S_coarse(k) * (Nf/Nc)^1.5   for |k| < k_Nyquist,coarse   (constrained)
  S_fine(k) = white-noise                 for larger |k|              (unconstrained LCDM)

Result: a unit white-noise field on the fine grid whose large scales match the CF4-constrained
reconstruction. Same periodic cubic box (user constraint). Feeds the pmwd forward test and
GRAFIC output.
"""
import os
import argparse
import numpy as np


def embed_ic(s_coarse, Nf, seed):
    """Embed the coarse constrained modes into a fine white-noise field."""
    Nc = s_coarse.shape[0]
    assert Nf % Nc == 0 or Nf > Nc, "Nf should exceed Nc"
    rng = np.random.default_rng(seed)
    Xf = np.fft.rfftn(rng.standard_normal((Nf, Nf, Nf)))         # fine white noise (target small-k replaced)
    Sc = np.fft.rfftn(s_coarse.astype(np.float64))
    scale = (Nf / Nc) ** 1.5                                     # unit-variance normalization
    hc = Nc // 2
    # map coarse frequency index -> fine index (preserve physical k = 2pi n / L).
    # EXCLUDE the coarse Nyquist index hc on every axis: hc maps to +hc only (not the
    # self-conjugate -hc), which breaks Hermitian symmetry on that shell and lets irfftn
    # discard its imaginary part. Skipping it leaves that one ~coarse-Nyquist shell as fine
    # white noise (Hermitian-correct), a negligible loss (~4 h^-1Mpc, one shell).
    fi = np.array([i if i < hc else Nf - (Nc - i) for i in range(Nc)])
    keep = [i for i in range(Nc) if i != hc]
    for i in keep:
        for j in keep:
            Xf[fi[i], fi[j], 0:hc] = Sc[i, j, 0:hc] * scale     # k = 0..hc-1 (skip Nyquist)
    s_fine = np.fft.irfftn(Xf, s=(Nf, Nf, Nf), axes=(0, 1, 2))
    return s_fine.astype(np.float32)


def fourier_resample_white_field(s_in, Nout):
    """Fourier-truncate a canonical white field onto a smaller cubic grid.

    The physical Fourier mode numbers are preserved and the coefficients are
    renormalized so that a unit-variance white field remains unit variance on
    the output grid.  This is the required operation when a realization was
    screened at a non-power-of-two mesh (for example N576) but the executable
    RAMSES parent must be L9 (N512).  Recalling ``embed_ic`` with the same RNG
    seed at N512 would define a different realization.
    """
    s_in = np.asarray(s_in)
    if s_in.ndim != 3 or not (s_in.shape[0] == s_in.shape[1] == s_in.shape[2]):
        raise ValueError("s_in must be a cubic 3-D field")
    Nin = s_in.shape[0]
    if Nout > Nin or Nout <= 0 or Nout % 2:
        raise ValueError("Nout must be a positive even integer no larger than Nin")
    if Nout == Nin:
        return s_in.astype(np.float32, copy=True)
    Win = np.fft.rfftn(s_in.astype(np.float64, copy=False))
    half = Nout // 2
    index = np.r_[0:half, Nin - half:Nin]
    Wout = Win[np.ix_(index, index, np.arange(half + 1))].copy()
    Wout *= (float(Nout) / Nin) ** 1.5
    result = np.fft.irfftn(Wout, s=(Nout, Nout, Nout), axes=(0, 1, 2))
    return result.astype(np.float32)


def embed_ic_projected(s_coarse, Ncanonical, Nout, seed):
    """Define modes on a canonical mesh and project directly to ``Nout``.

    This is algebraically the same realization as ``embed_ic`` at
    ``Ncanonical`` followed by ``fourier_resample_white_field``, but avoids an
    unnecessary canonical inverse FFT and second canonical forward FFT.
    """
    if Nout > Ncanonical:
        raise ValueError("Nout cannot exceed Ncanonical")
    if Nout == Ncanonical:
        return embed_ic(s_coarse, Nout, seed)
    Nc = s_coarse.shape[0]
    rng = np.random.default_rng(seed)
    canonical_fft = np.fft.rfftn(
        rng.standard_normal((Ncanonical, Ncanonical, Ncanonical)))
    half_out = Nout // 2
    out_index = np.r_[0:half_out, Ncanonical - half_out:Ncanonical]
    projected = canonical_fft[
        np.ix_(out_index, out_index, np.arange(half_out + 1))].copy()
    projected *= (float(Nout) / Ncanonical) ** 1.5

    coarse_fft = np.fft.rfftn(np.asarray(s_coarse, dtype=np.float64))
    half_coarse = Nc // 2
    fine_index = np.array(
        [i if i < half_coarse else Nout - (Nc - i) for i in range(Nc)])
    keep = [i for i in range(Nc) if i != half_coarse]
    scale = (float(Nout) / Nc) ** 1.5
    for i in keep:
        for j in keep:
            projected[fine_index[i], fine_index[j], :half_coarse] = (
                coarse_fft[i, j, :half_coarse] * scale)
    result = np.fft.irfftn(
        projected, s=(Nout, Nout, Nout), axes=(0, 1, 2))
    return result.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "cf4_map_cf4_real192.npz"))
    ap.add_argument("--field", default="s_out")
    ap.add_argument("--Nfine", type=int, default=384, help="fine grid (L fixed by the recon box)")
    ap.add_argument("--canonical-N", type=int, default=None,
                    help="draw at this canonical mesh then Fourier-project to Nfine")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "cf4_ic_fine.npz"))
    args = ap.parse_args()

    z = np.load(args.recon)
    s_coarse = z[args.field].astype(np.float64)
    Nc = s_coarse.shape[0]; spc_c = float(z["spacing"]); L = Nc * spc_c
    spc_f = L / args.Nfine
    print(f"[ic] coarse {Nc}^3 @ {spc_c} h^-1Mpc (L={L:.0f}) -> fine {args.Nfine}^3 @ "
          f"{spc_f:.3f} h^-1Mpc", flush=True)

    canonical_n = args.Nfine if args.canonical_N is None else args.canonical_N
    s_fine = embed_ic(s_coarse, canonical_n, args.seed)
    if canonical_n != args.Nfine:
        s_fine = fourier_resample_white_field(s_fine, args.Nfine)
    print(f"[ic] s_fine mean={s_fine.mean():.4f} std={s_fine.std():.4f} (expect ~N(0,1))", flush=True)
    # verify the large scales match the coarse reconstruction (downsample the fine, cross-correlate)
    Xf = np.fft.rfftn(s_fine); hc = Nc // 2
    fi = np.array([i if i <= hc else args.Nfine - (Nc - i) for i in range(Nc)])
    Sc_back = np.zeros((Nc, Nc, hc + 1), complex)
    for i in range(Nc):
        for j in range(Nc):
            Sc_back[i, j] = Xf[fi[i], fi[j], 0:hc + 1]
    s_lowk = np.fft.irfftn(Sc_back / (args.Nfine / Nc) ** 1.5, s=(Nc, Nc, Nc), axes=(0, 1, 2))
    r = np.corrcoef(s_lowk.ravel(), s_coarse.ravel())[0, 1]
    print(f"[ic] large-scale recovery: corr(fine low-k, coarse) = {r:.4f} (should be ~1)", flush=True)

    np.savez(args.out, s_fine=s_fine, Nfine=args.Nfine, canonical_N=canonical_n,
             spacing=spc_f, L=L,
             h=float(z["hh"]) if "hh" in z else 0.746, Om=0.31, A_s_1e9=1.63)
    print(f"[ic] saved {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
