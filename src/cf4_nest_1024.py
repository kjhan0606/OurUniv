#!/usr/bin/env python
"""Resolution-portable 1024^3 IC for cr6 e19 by NESTING the tested embed_ic (fable's fixed-phase,
done via nested embedding instead of a noise-generator rewrite).

s_1024 = embed_ic(F_576, 1024, seed2), where F_576 = embed_ic(s_cr6, 576, 19) is the exact 576^3
field the screen selected. embed_ic preserves the coarse field's Fourier modes at fixed physical
k, so ALL modes |k| < 576-Nyquist (which fully contain the MW-M31 pair, formed from lambda~2-4 Mpc
power well below the 576-Nyquist lambda~1.3 Mpc) are carried into 1024^3 unchanged; only 576-1024
Nyquist small-scale detail is fresh. The pair is preserved -> no re-screen. Verifies by
downsampling s_1024 back to 576^3 and cross-correlating with F_576 (must be ~1).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf4_make_ic import embed_ic


def downsample_modes(sf, Nc):
    """Truncate a fine field to |k|<Nc-Nyquist and inverse-transform to Nc^3 (mode-preserving)."""
    Nf = sf.shape[0]; hc = Nc // 2
    Xf = np.fft.rfftn(sf.astype(np.float64))
    fi = np.array([i if i <= hc else Nf - (Nc - i) for i in range(Nc)])
    Sc = np.zeros((Nc, Nc, hc + 1), complex)
    for i in range(Nc):
        for j in range(Nc):
            Sc[i, j] = Xf[fi[i], fi[j], 0:hc + 1]
    return np.fft.irfftn(Sc / (Nf / Nc) ** 1.5, s=(Nc, Nc, Nc), axes=(0, 1, 2))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--f576", default="recon/s_cr6_e19_576.npy", help="the selected 576^3 field")
    ap.add_argument("--Nf", type=int, default=1024)
    ap.add_argument("--seed2", type=int, default=2019, help="fresh seed for 576-1024 Nyquist detail")
    ap.add_argument("--out", default="recon/s_cr6_e19_1024.npy")
    args = ap.parse_args()

    F576 = np.load(args.f576).astype(np.float64)
    Nc = F576.shape[0]
    print(f"[nest] F_{Nc}: mean={F576.mean():.4f} std={F576.std():.4f}", flush=True)

    s1024 = embed_ic(F576, args.Nf, args.seed2)
    print(f"[nest] s_{args.Nf}: mean={s1024.mean():.4f} std={s1024.std():.4f} (expect ~N(0,1))", flush=True)

    # verify mode preservation: downsample s_1024 -> 576^3, cross-correlate with F_576
    back = downsample_modes(s1024, Nc)
    r = np.corrcoef(back.ravel(), F576.ravel())[0, 1]
    print(f"[nest] mode-preservation corr(downsample(s_{args.Nf}), F_{Nc}) = {r:.5f} (MUST be ~1)", flush=True)
    # also report per-region corr near the observer (the LG patch), the part that matters
    N = Nc; c = N // 2; w = N // 6
    sl = slice(c - w, c + w)
    rc = np.corrcoef(back[sl, sl, sl].ravel(), F576[sl, sl, sl].ravel())[0, 1]
    print(f"[nest] observer-patch corr = {rc:.5f}", flush=True)

    np.save(args.out, s1024.astype(np.float32))
    print(f"[nest] saved {args.out} ({os.path.getsize(args.out)/1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
