#!/usr/bin/env python
"""Object-level validation: named-structure scorecard + constrained-vs-random discriminator.

Statistics (P(k), psi(r)) only check "is this LCDM?"; they do NOT check "is this OUR
universe" -- whether Virgo, Coma, the Great Attractor sit where observed. This does that:

  1. forward the reconstructed IC -> z=0 particles -> FoF halos (clusters)
  2. cross-match the most massive recon halo near each KNOWN local cluster (supergalactic
     position from the literature) -> position offset [Mpc/h] + halo mass
  3. constrained-vs-RANDOM discriminator: repeat for N_random random-phase LCDM IC (same
     box/cosmology). The reconstruction should match the named clusters FAR better than
     random -> the gap = the constraint information ("this universe" vs "a universe").

A match = a >Mthr halo within Rsearch of the cluster's SG position. Score = number matched
and median offset; reported for the reconstruction vs the random-phase baseline distribution.

Run on GPU: sbatch src/cf4_job.slurm src/cf4_scorecard.py --pred recon/cf4_map_cf4_real192.npz
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import numpy as np

# Known local clusters: name -> (SGL deg, SGB deg, dist Mpc, approx M [Msun/h])
CLUSTERS = {
    "Virgo":      (102.9,  -2.3,  16.0, 7e14),
    "Coma":       ( 89.0,   8.0,  90.0, 1.0e15),
    "Centaurus":  (156.0, -11.0,  45.0, 3e14),
    "Norma(GA)":  (155.0,  -6.0,  65.0, 5e14),
    "Perseus":    (340.0, -13.0,  73.0, 6e14),
    "Hydra":      (139.0,  26.0,  50.0, 2e14),
    "Fornax":     (236.0, -44.0,  19.0, 8e13),
}


def sg_to_xyz(sgl, sgb, d):
    l = np.radians(sgl); b = np.radians(sgb)
    return d * np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def score_field(halo_pos, halo_mass, c, h, L, Rsearch, Mthr):
    """Match recon halos to known clusters. Returns per-cluster records + summary."""
    recs = {}
    nmatch = 0; offs = []
    for name, (sgl, sgb, d, Mlit) in CLUSTERS.items():
        Xt = c + sg_to_xyz(sgl, sgb, d) * h                 # box frame [Mpc/h]
        dv = halo_pos - Xt; dv -= L * np.round(dv / L)
        rr = np.linalg.norm(dv, axis=1)
        near = rr < Rsearch
        if near.any():
            j = np.where(near)[0][np.argmax(halo_mass[near])]
            off = float(rr[j]); M = float(halo_mass[j])
            hit = M > Mthr
            recs[name] = (off, M, Mlit, hit)
            if hit:
                nmatch += 1; offs.append(off)
        else:
            recs[name] = (np.nan, 0.0, Mlit, False)
    return recs, nmatch, (np.median(offs) if offs else np.nan)


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mock_pipeline import RHO_CRIT
    from fof import fof

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--field", default="s_out")
    ap.add_argument("--real-npz", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "cf4_clean.npz"))
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--A-s-1e9", type=float, default=1.63)
    ap.add_argument("--Rsearch", type=float, default=12.0, help="match radius [Mpc/h]")
    ap.add_argument("--Mthr", type=float, default=1e14, help="halo-mass threshold for a hit")
    ap.add_argument("--n-random", type=int, default=5, help="random-phase baseline realizations")
    ap.add_argument("--dsmooth", type=float, default=5.0, help="density smoothing [Mpc/h]")
    ap.add_argument("--skip-fof", action="store_true", help="only the density-at-cluster test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P = np.load(args.pred); s_rec = P[args.field].astype(np.float64)
    N = int(P["N"]); sp = float(P["spacing"]); L = N * sp
    h = float(np.load(args.real_npz)["H0"]) / 100.0
    c = L / 2.0; m_p = args.Om * RHO_CRIT * sp ** 3
    print(f"[score] N={N} L={L:.0f} Mpc/h h={h:.3f} m_p={m_p:.2e} Rsearch={args.Rsearch} "
          f"Mthr={args.Mthr:.0e}", flush=True)

    import jax, jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody
    conf = Configuration(ptcl_spacing=float(sp), ptcl_grid_shape=(N,) * 3,
                         mesh_shape=1, float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=args.Om, h=h, A_s_1e9=args.A_s_1e9), conf)
    from mock_pipeline import VUNIT_KMS

    def forward_halos(s):
        lin = linear_modes(jnp.asarray(s.reshape(N, N, N)), cosmo, conf)
        ptcl, obs = lpt(lin, cosmo, conf); ptcl, obs = nbody(ptcl, obs, cosmo, conf)
        pos = np.asarray(ptcl.pos(), np.float64)
        vel = np.asarray(ptcl.vel, np.float64) * VUNIT_KMS
        hal = fof(pos, vel, L=L, b=0.2, n_min=20, m_particle=m_p, verbose=False)
        return hal["pos"], hal["mass"]

    # ---- density-at-cluster test (constrained large-scale field; the RIGHT test for
    # cluster POSITIONS -- s_map posterior-mean density, no realization noise) ----
    from pmwd import scatter
    kf = np.fft.fftfreq(N, d=sp) * 2 * np.pi; kr = np.fft.rfftfreq(N, d=sp) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    Wsm = np.exp(-0.5 * (KX**2 + KY**2 + KZ**2) * args.dsmooth**2)

    def density_at_clusters(s):
        lin = linear_modes(jnp.asarray(s.reshape(N, N, N)), cosmo, conf)
        ptcl, o = lpt(lin, cosmo, conf); ptcl, o = nbody(ptcl, o, cosmo, conf)
        d = np.asarray(scatter(ptcl, conf), np.float64) - 1.0
        d = np.fft.irfftn(np.fft.rfftn(d) * Wsm, s=(N, N, N), axes=(0, 1, 2))
        d = (d - d.mean()) / d.std()                        # normalize -> sigma units
        cell = L / N; vals = {}
        for name, (sgl, sgb, dist, Ml) in CLUSTERS.items():
            Xt = c + sg_to_xyz(sgl, sgb, dist) * h
            ix = (np.round(Xt / cell).astype(int)) % N
            vals[name] = float(d[ix[0], ix[1], ix[2]])
        return vals

    s_map = np.load(args.pred)["s_map"].astype(np.float64)
    dv_rec = density_at_clusters(s_map)
    dm_rec = np.mean(list(dv_rec.values()))
    print(f"\n[DENSITY-AT-CLUSTER — constrained s_map, {args.dsmooth:.0f} Mpc/h smooth, sigma units]")
    for name, v in dv_rec.items():
        print(f"  {name:12s} delta = {v:+.2f} sigma  {'(overdense OK)' if v > 0 else '(underdense!)'}")
    print(f"  => mean delta at clusters = {dm_rec:+.2f} sigma  "
          f"({sum(v>0 for v in dv_rec.values())}/{len(dv_rec)} overdense)")
    dm_rand = []
    for i in range(args.n_random):
        rng = np.random.default_rng(2000 + i)
        dm_rand.append(np.mean(list(density_at_clusters(rng.standard_normal(N ** 3)).values())))
    dm_rand = np.array(dm_rand)
    dexc = (dm_rec - dm_rand.mean()) / (dm_rand.std() + 1e-6)
    print(f"  random baseline mean delta = {dm_rand.mean():+.2f} +/- {dm_rand.std():.2f}"
          f"  => reconstruction {dexc:.1f}-sigma above random", flush=True)

    if args.skip_fof:
        print("[score] --skip-fof: density test only", flush=True); return
    # reconstruction
    hp, hm = forward_halos(s_rec)
    recs, nmatch, medoff = score_field(hp, hm, c, h, L, args.Rsearch, args.Mthr)
    print(f"\n[SCORECARD — reconstruction]  ({len(hm)} halos, {nmatch}/{len(CLUSTERS)} matched)")
    print(f"  {'cluster':12s} {'offset[Mpc/h]':>13s} {'M_recon':>10s} {'M_lit':>10s} {'hit':>4s}")
    for name, (off, M, Mlit, hit) in recs.items():
        print(f"  {name:12s} {off:13.1f} {M:10.2e} {Mlit:10.2e} {'YES' if hit else 'no':>4s}")
    print(f"  => matched {nmatch}/{len(CLUSTERS)}, median offset {medoff:.1f} Mpc/h")

    # constrained-vs-random baseline
    rmatch = []; roff = []
    for i in range(args.n_random):
        rng = np.random.default_rng(1000 + i)
        s_rand = rng.standard_normal(N ** 3)
        hp_r, hm_r = forward_halos(s_rand)
        _, nm, mo = score_field(hp_r, hm_r, c, h, L, args.Rsearch, args.Mthr)
        rmatch.append(nm); roff.append(mo)
        print(f"[random {i}] matched {nm}/{len(CLUSTERS)} median offset {mo:.1f}", flush=True)
    rmatch = np.array(rmatch)
    print(f"\n[DISCRIMINATOR — constrained vs random]")
    print(f"  reconstruction matched : {nmatch}/{len(CLUSTERS)}  (median off {medoff:.1f} Mpc/h)")
    print(f"  random-phase baseline  : {rmatch.mean():.1f} +/- {rmatch.std():.1f} "
          f"(range {rmatch.min()}-{rmatch.max()})")
    excess = (nmatch - rmatch.mean()) / (rmatch.std() + 1e-6)
    print(f"  => reconstruction is {excess:.1f}-sigma above random "
          f"({'CONSTRAINT INFORMATION confirmed' if excess > 2 else 'not clearly above random'})")

    outf = args.out or args.pred.replace(".npz", "_scorecard.npz")
    np.savez(outf, recs={k: v for k, v in recs.items()}, nmatch=nmatch, medoff=medoff,
             rand_match=rmatch, excess_sigma=excess, allow_pickle=True)
    print(f"[score] saved {outf}", flush=True)


if __name__ == "__main__":
    main()
