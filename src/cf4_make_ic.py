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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "cf4_map_cf4_real192.npz"))
    ap.add_argument("--field", default="s_out")
    ap.add_argument("--Nfine", type=int, default=384, help="fine grid (L fixed by the recon box)")
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

    s_fine = embed_ic(s_coarse, args.Nfine, args.seed)
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

    np.savez(args.out, s_fine=s_fine, Nfine=args.Nfine, spacing=spc_f, L=L,
             h=float(z["hh"]) if "hh" in z else 0.746, Om=0.31, A_s_1e9=1.63)
    print(f"[ic] saved {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
