#!/usr/bin/env python
"""Observation comparison, done right: angular per-shell maps + literature cluster peaks.

Two selection-robust tests of the HR-constrained forward against OBSERVATION:
  (1) Angular per-shell (Hong-style): within each distance shell, the radial selection is
      ~constant, so angular overdensity is selection-free. Compare recon vs CF4 galaxy sky
      maps shell by shell, and report the angular correlation.
  (2) Literature clusters: measure recon 1+delta in a 3 h^-1Mpc sphere at the (approximate)
      published supergalactic position of each known nearby cluster. Independent of the CF4
      galaxy field. Cluster coords are literature approximations (flagged).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import make_forward, RHO_CRIT
from cf4_make_ic import embed_ic

# approximate literature supergalactic (l,b,dist[Mpc]); nearest 3 most reliable/in-range
CLUSTERS = [
    ("Virgo",     102.9,  -2.3,  16.5),
    ("Fornax",    236.5, -45.7,  19.0),
    ("Centaurus", 156.0, -11.5,  45.0),
    ("Hydra",     139.5, -37.5,  50.0),
    ("Norma/GA",  149.5,  -7.2,  67.0),
    ("Perseus",   347.0, -13.0,  74.0),
]
SHELLS = [(5.0, 15.0), (15.0, 25.0), (25.0, 40.0)]   # h^-1Mpc


def sgvec(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def ang_map(dx, r, rlo, rhi, nlon, nlat, smooth):
    """Overdensity in (SGL, sinSGB) equal-area angular bins for particles in [rlo,rhi)."""
    from scipy.ndimage import gaussian_filter
    sel = (r >= rlo) & (r < rhi)
    d = dx[sel]; rr = r[sel]
    l = np.arctan2(d[:, 1], d[:, 0])                 # [-pi,pi]
    sb = d[:, 2] / rr                                # sin(b) in [-1,1]
    H, _, _ = np.histogram2d(l, sb, bins=[nlon, nlat],
                             range=[[-np.pi, np.pi], [-1, 1]])
    Hs = gaussian_filter(H, smooth, mode=["wrap", "nearest"])
    m = Hs.mean()
    return (Hs / m - 1.0) if m > 0 else Hs, sel.sum()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192_hr.npz")
    ap.add_argument("--cat", default="data/cf4_clean.npz")
    ap.add_argument("--key", default="s_out")
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--nlon", type=int, default=72)
    ap.add_argument("--nlat", type=int, default=36)
    ap.add_argument("--out", default="recon/cf4_obs_compare2.png")
    args = ap.parse_args()
    import jax.numpy as jnp
    import time

    z = np.load(args.recon)
    s = z[args.key].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    L = Nc * spc; c = L / 2.0; H = float(z["hh"]) if "hh" in z else 0.746
    sp = L / args.Nfine
    print(f"[obs2] {args.recon} key={args.key} L={L:.0f} h={H}", flush=True)

    s_fine = embed_ic(s, args.Nfine, 1)
    t0 = time.time()
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    print(f"[obs2] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)
    nbar = pos.shape[0] / L**3                        # global particle number density

    dxp = pos - c; rp = np.linalg.norm(dxp, axis=1)

    # ---- (2) literature cluster peaks (3 h^-1Mpc sphere) ----
    print("\n[obs2] LITERATURE CLUSTER PEAKS (recon 1+delta in 3 h^-1Mpc sphere):", flush=True)
    print(f"  {'cluster':>10} {'SGL':>6} {'SGB':>6} {'d[hMpc]':>7} {'1+delta':>8}  note", flush=True)
    Vsp = 4/3*np.pi*3.0**3
    cl_rows = []
    for name, l, b, dmpc in CLUSTERS:
        dh = dmpc * H
        p = c + sgvec(l, b) * dh
        nin = np.sum(np.linalg.norm(pos - p, axis=1) < 3.0)
        opd = nin / (nbar * Vsp)
        note = "reliable" if dh < 40 else "edge of good S/N"
        cl_rows.append((name, dh, opd))
        print(f"  {name:>10} {l:>6.1f} {b:>6.1f} {dh:>7.1f} {opd:>8.2f}  {note}", flush=True)

    # ---- (1) angular per-shell recon vs CF4 galaxies ----
    cz = np.load(args.cat)
    gbox = c + cz["pos_dist"].astype(float) * H
    dxg = gbox - c; rg = np.linalg.norm(dxg, axis=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 4.2*len(SHELLS)))
    print("\n[obs2] ANGULAR PER-SHELL (selection-free within a shell):", flush=True)
    lon_edges = np.linspace(-np.pi, np.pi, args.nlon+1)
    sb_edges = np.linspace(-1, 1, args.nlat+1)
    LON, SB = np.meshgrid(0.5*(lon_edges[:-1]+lon_edges[1:]),
                          np.arcsin(0.5*(sb_edges[:-1]+sb_edges[1:])), indexing="ij")
    for i, (r0, r1) in enumerate(SHELLS):
        mr, nr = ang_map(dxp, rp, r0, r1, args.nlon, args.nlat, smooth=1.0)
        mg, ng = ang_map(dxg, rg, r0, r1, args.nlon, args.nlat, smooth=1.5)
        good = mg != mg.min()                         # drop empty (ZoA/edge) cells
        r_ang = np.corrcoef(mr[good].ravel(), mg[good].ravel())[0, 1] if good.sum() > 10 else np.nan
        print(f"  shell {r0:.0f}-{r1:.0f} hMpc: recon {nr} ptcl, CF4 {ng} gal, "
              f"angular corr r={r_ang:.3f}", flush=True)
        for j, (mm, ttl, cmap) in enumerate([(mr, f"recon {r0:.0f}-{r1:.0f}", "inferno"),
                                             (mg, f"CF4 gal {r0:.0f}-{r1:.0f} (r={r_ang:.2f})", "viridis")]):
            ax = fig.add_subplot(len(SHELLS), 2, 2*i+j+1, projection="mollweide")
            vmax = np.percentile(mm, 98)
            ax.pcolormesh(LON, SB, mm, cmap=cmap, vmin=-0.8, vmax=max(vmax, 0.5), shading="auto")
            # mark clusters in this shell
            for name, cl, cb, dmpc in CLUSTERS:
                if r0 <= dmpc*H < r1:
                    ll = np.radians(cl); ll = (ll+np.pi) % (2*np.pi) - np.pi
                    ax.plot(ll, np.radians(cb), "o", ms=9, mfc="none",
                            mec="red", mew=2)
                    ax.text(ll, np.radians(cb)+0.12, name, color="red", fontsize=8, ha="center")
            ax.set_title(ttl, fontsize=11); ax.grid(True, alpha=0.3)
            ax.set_xticklabels([])
    fig.suptitle("Angular per-shell: HR reconstruction vs observed CF4 galaxies (supergalactic)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"\n[obs2] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
