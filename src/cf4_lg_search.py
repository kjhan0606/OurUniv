#!/usr/bin/env python
"""Search a completion+fine seed pair for a Milky-Way / M31 analog at the observer.

Pipeline for one (completion seed, fine seed):
  s_out  = power_complete(s_map, cseed)          # 2-10 Mpc environment (unconstrained)
  s_fine = embed(s_out, 768, fseed)              # sub-2 Mpc structure (the actual LG pair)
  forward (pmwd 768^3) -> particles
  FoF the central cube -> look for an MW-M31 pair:
     two halos ~1-2e12 Msun/h, 0.3-1.2 h^-1Mpc apart, near the observer, isolated
     (no >5e12 halo within 3 Mpc, matching the real LG's isolation from Virgo at 12 Mpc).
Saves the central particles (for the zoom trace-back) + a summary + a figure.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf4_explicit_map import power_complete
from cf4_make_ic import embed_ic
from mock_pipeline import make_forward, RHO_CRIT, VUNIT_KMS
from fof import fof


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192.npz")
    ap.add_argument("--cseed", type=int, default=0, help="completion seed (only for --field s_out)")
    ap.add_argument("--fseed", type=int, required=True, help="fine-embed (LG pair) seed")
    ap.add_argument("--field", choices=["s_out", "s_map"], default="s_out",
                    help="s_map: embed the MAP directly (preserves Virgo); s_out: power-complete first")
    ap.add_argument("--Nfine", type=int, default=768)
    ap.add_argument("--half", type=float, default=40.0, help="central cube half-size [h^-1Mpc]")
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--outdir", default="recon/lg_search")
    args = ap.parse_args()
    import jax.numpy as jnp
    import time

    z = np.load(args.recon)
    s_map = z["s_map"].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    L = Nc * spc; sp = L / args.Nfine; c = L / 2.0
    m_p = args.Om * RHO_CRIT * sp ** 3
    if args.field == "s_map":
        tag = f"smapf{args.fseed}"; coarse = s_map          # preserves Virgo; no completion
    else:
        tag = f"c{args.cseed}f{args.fseed}"; coarse = power_complete(s_map, Nc, args.cseed)
    print(f"[lg] {tag}: field={args.field} N={args.Nfine} sp={sp:.3f} m_p={m_p:.2e}", flush=True)

    t0 = time.time()
    s_fine = embed_ic(coarse, args.Nfine, args.fseed)
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    # extract in float32 (native) and upcast on the HOST -- forcing float64 on the GPU
    # doubles the array and blows up the mod-L intermediate (24 GiB OOM on a 48 GB card)
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    vel = np.asarray(ptcl.vel).astype(np.float64) * VUNIT_KMS
    print(f"[lg] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)

    m = np.all(np.abs(pos - c) < args.half, axis=1)
    gi = np.flatnonzero(m)                              # global indices of central particles
    halos = fof(pos[gi], vel[gi], L=L, mean_sep=sp, b=0.2, n_min=20,
                m_particle=m_p, periodic=False, verbose=True)
    hp = halos["pos"]; hm = halos["mass"]; hv = halos["vel"]
    rh = np.linalg.norm(hp - c, axis=1)

    # MEASURE whether Virgo is actually an overdensity at its position (not assumed)
    def sgdir(l, b):
        l, b = np.radians(l), np.radians(b)
        return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])
    virgo = c + sgdir(102.9, -2.3) * (16.5 * 0.746)
    cenp = pos[gi]; nmean = cenp.shape[0] / ((2*args.half)**3)
    virgo_dens = {R: np.sum(np.linalg.norm(cenp - virgo, axis=1) < R) / (nmean*4/3*np.pi*R**3)
                  for R in (2.0, 4.0)}
    dvir = np.linalg.norm(hp - virgo, axis=1)
    virgo_Mmax = hm[dvir < 3].max() if np.any(dvir < 3) else 0.0
    print(f"[lg] VIRGO check: 1+delta(<2Mpc)={virgo_dens[2.0]:.2f} (<4Mpc)={virgo_dens[4.0]:.2f}  "
          f"max halo within 3Mpc={virgo_Mmax:.1e} Msun/h", flush=True)

    # MW-M31 pair candidates
    mw = np.flatnonzero((hm > 5e11) & (hm < 4e12) & (rh < 20))
    big = hp[hm > 5e12]                                 # rich systems to avoid
    pairs = []
    for a in range(len(mw)):
        for b in range(a + 1, len(mw)):
            i, j = mw[a], mw[b]
            d = np.linalg.norm(hp[i] - hp[j])
            if not (0.3 < d < 1.2):
                continue
            mid = 0.5 * (hp[i] + hp[j])
            iso = True if not len(big) else (np.linalg.norm(big - mid, axis=1).min() > 3.0)
            if not iso:
                continue
            vrel = np.dot(hv[i] - hv[j], (hp[i] - hp[j]) / d)   # <0 = approaching
            pairs.append(dict(i=int(i), j=int(j), sep=d, Mi=hm[i], Mj=hm[j],
                              rmid=np.linalg.norm(mid - c), vrel=vrel))
    pairs.sort(key=lambda p: abs(np.log10(p["Mi"]) - 12.1) + abs(np.log10(p["Mj"]) - 12.1)
               + 0.5 * abs(p["sep"] - 0.6) + 0.02 * p["rmid"])
    print(f"[lg] {tag}: {len(mw)} MW-mass halos near centre, {len(pairs)} isolated pairs", flush=True)
    for p in pairs[:5]:
        print(f"[lg]   pair sep={p['sep']:.2f} M=({p['Mi']:.1e},{p['Mj']:.1e}) "
              f"r_mid={p['rmid']:.1f} vrel={p['vrel']:.0f} km/s "
              f"{'(approaching)' if p['vrel']<0 else ''}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, f"lg_{tag}.npz"),
             cseed=args.cseed, fseed=args.fseed, field=args.field, sp=sp, L=L, m_p=m_p,
             virgo_d2=virgo_dens[2.0], virgo_d4=virgo_dens[4.0], virgo_Mmax=virgo_Mmax,
             halo_pos=hp.astype(np.float32), halo_mass=hm.astype(np.float32),
             halo_vel=hv.astype(np.float32),
             cen_gidx=gi.astype(np.int64), cen_pos=pos[gi].astype(np.float32),
             n_pairs=len(pairs),
             best=(np.array([pairs[0][k] for k in ("sep", "Mi", "Mj", "rmid", "vrel")])
                   if pairs else np.zeros(5)))

    # figure: central halos, pair candidates circled
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    sl = np.abs(hp[:, 2] - c) < 8
    ax.scatter(hp[sl, 0]-c, hp[sl, 1]-c, s=4+18*(np.log10(hm[sl])-11.3),
               c=np.log10(hm[sl]), cmap="viridis", vmin=11.3, vmax=14.5, alpha=0.8, lw=0)
    ax.plot(0, 0, "r+", ms=14, mew=2)
    for p in pairs[:3]:
        for k in ("i", "j"):
            hh = hp[p[k]] - c
            ax.add_patch(plt.Circle((hh[0], hh[1]), 1.5, fill=False, color="red", lw=1.5))
    ax.set_xlim(-20, 20); ax.set_ylim(-20, 20); ax.set_aspect("equal")
    ax.set_title(f"{tag}: {len(pairs)} MW-M31 pair candidates (red circles)\n"
                 f"observer=red+, central density profile screened")
    ax.set_xlabel("SGX [$h^{-1}$Mpc]"); ax.set_ylabel("SGY [$h^{-1}$Mpc]")
    fig.savefig(os.path.join(args.outdir, f"lg_{tag}.png"), dpi=110, bbox_inches="tight")
    print(f"[lg] saved {args.outdir}/lg_{tag}.*", flush=True)


if __name__ == "__main__":
    main()
