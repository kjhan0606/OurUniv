#!/usr/bin/env python
"""Training data for position-conditioned super-resolution from TNG300 galaxies.

The CF4 reconstruction constrains the field only to ~large scales. To get galaxy-scale
structure whose biased peaks match the observations we learn a super-resolution map from
IllustrisTNG. The model takes a coarse constrained density and the SPARSE observed galaxy
positions, then generates the full fine galaxy field. Training pairs come from TNG300.

For each sub-cube of the TNG300 box we store three fields on a fine grid:
  n_fine   full TNG galaxy number field (target; realistic bias and clustering)
  n_obs    a CF4-sparse subsample of the same galaxies (conditioning; the observed positions)
  d_coarse coarse density on a coarse grid (conditioning; the constrained large-scale field)

The model then learns  (d_coarse, n_obs) -> n_fine. At application time we feed the CF4
reconstruction as d_coarse and the real CF4 groups as n_obs, so the generated peaks are
pinned to the observed galaxies where we have data and TNG-statistical elsewhere.

TNG300-1: box 205 h^-1Mpc, h=0.6774, 122764 galaxies at z=0 (galaxy_099.sav).
Run:  python cf4_sr_gen.py --fine 0.5 --sub 64 --coarse-factor 4
"""
import os
import argparse
import json
import numpy as np

TNG_SAV = "/scratch/jhshin/02_illustris/08_illustrisTNG/TNG300-1/sav/galaxy_099.sav"
H_TNG = 0.6774
L_TNG = 205.0  # h^-1 Mpc


def ngp(pos, N, L):
    """Nearest-grid-point count field. pos in [0,L)."""
    idx = np.floor((pos % L) / L * N).astype(np.int64) % N
    field = np.zeros((N, N, N), np.float32)
    np.add.at(field, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return field


def density(pos, N, L, sm):
    """Smoothed galaxy overdensity delta = n/nbar - 1. A continuous field (unlike raw
    counts) that a diffusion model can represent. sm = Gaussian smoothing in cells."""
    from scipy.ndimage import gaussian_filter
    n = ngp(pos, N, L)
    if sm > 0:
        n = gaussian_filter(n, sm, mode="wrap")
    nbar = n.mean()
    return (n / max(nbar, 1e-8) - 1.0).astype(np.float32) if nbar > 0 else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sav", default=TNG_SAV)
    ap.add_argument("--fine", type=float, default=0.5, help="fine cell [h^-1 Mpc]")
    ap.add_argument("--sub", type=int, default=64, help="fine cells per sub-cube")
    ap.add_argument("--coarse-factor", type=int, default=4, help="fine/coarse cell ratio")
    ap.add_argument("--stride", type=int, default=64, help="fine cells between sub-cubes")
    ap.add_argument("--mstar-min", type=float, default=1e9, help="stellar-mass cut [Msun]")
    ap.add_argument("--p-obs", type=float, default=0.055,
                    help="keep fraction for the observed subsample (CF4 sparsity ~1/18)")
    ap.add_argument("--smooth-fine", type=float, default=1.5,
                    help="Gaussian smoothing of the fine target [cells] -> continuous field")
    ap.add_argument("--smooth-obs", type=float, default=2.0,
                    help="Gaussian smoothing of the observed channel [cells]")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data_sr"))
    args = ap.parse_args()

    from scipy.io import readsav
    d = readsav(args.sav)
    g = d["galaxy"]
    pos = np.array([p for p in g["POS"]], np.float64).reshape(len(g), 3) * H_TNG  # -> h^-1 Mpc
    mstar = np.array(g["MSTAR"], np.float64)
    keep = mstar > args.mstar_min
    pos = pos[keep] % L_TNG
    print(f"[sr-gen] TNG300 {keep.sum()}/{len(g)} galaxies (MSTAR>{args.mstar_min:.0e}) "
          f"in {L_TNG:.0f} h^-1Mpc; mean density {keep.sum()/L_TNG**3:.2e} (h/Mpc)^3", flush=True)

    fine = args.fine; sub = args.sub; cf = args.coarse_factor
    subL = sub * fine                                   # sub-cube size [h^-1 Mpc]
    Nbox = int(round(L_TNG / fine))                     # full fine grid
    csub = sub // cf                                    # coarse cells per sub-cube
    print(f"[sr-gen] fine={fine} h^-1Mpc sub={sub} ({subL:.0f} h^-1Mpc) coarse={cf*fine} "
          f"h^-1Mpc; full grid {Nbox}^3", flush=True)

    rng = np.random.default_rng(0)
    obsmask = rng.random(pos.shape[0]) < args.p_obs     # fixed observed subsample
    pos_obs = pos[obsmask]
    print(f"[sr-gen] observed subsample {pos_obs.shape[0]} ({args.p_obs:.3f}) "
          f"-> density {pos_obs.shape[0]/L_TNG**3:.2e} (CF4-like)", flush=True)

    offs = list(range(0, Nbox - sub + 1, args.stride))
    NF, NO, DC = [], [], []
    for ox in offs:
        for oy in offs:
            for oz in offs:
                lo = np.array([ox, oy, oz]) * fine
                # galaxies in this sub-cube (with periodic wrap of the box)
                rel = (pos - lo) % L_TNG
                m = np.all(rel < subL, axis=1)
                relo = (pos_obs - lo) % L_TNG
                mo = np.all(relo < subL, axis=1)
                if m.sum() < 5:
                    continue
                nf = density(rel[m], sub, subL, args.smooth_fine)   # continuous fine overdensity
                no = density(relo[mo], sub, subL, args.smooth_obs)  # smoothed observed overdensity
                dc = density(rel[m], csub, subL, 0.0)               # coarse overdensity
                NF.append(nf); NO.append(no); DC.append(dc)
    NF = np.array(NF); NO = np.array(NO); DC = np.array(DC)
    ntot = NF.shape[0]
    nval = int(args.val_frac * ntot)
    perm = rng.permutation(ntot)
    splits = {"val": perm[:nval], "train": perm[nval:]}
    meta = dict(fine=fine, sub=sub, coarse_factor=cf, subL=subL, L_TNG=L_TNG, h=H_TNG,
                mstar_min=args.mstar_min, p_obs=args.p_obs, n_subcubes=ntot)
    for split, idx in splits.items():
        outdir = os.path.join(args.out, split); os.makedirs(outdir, exist_ok=True)
        np.savez(os.path.join(outdir, "shard_000.npz"),
                 n_fine=NF[idx], n_obs=NO[idx], d_coarse=DC[idx], meta=json.dumps(meta))
        print(f"[sr-gen] {split}: {len(idx)} sub-cubes -> {outdir}", flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[sr-gen] done. {ntot} sub-cubes total. continuous overdensity fields: "
          f"fine std={NF.std():.2f} coarse std={DC.std():.2f} obs std={NO.std():.2f}", flush=True)


if __name__ == "__main__":
    main()
