#!/usr/bin/env python
"""Screen (cr_seed x embed_seed) for an LG analog on the constrained field, corrected score.

Two random axes (per fable): cr_seed = the Hoffman-Ribak compensation draw (sets the CF4-
UNCONSTRAINED intermediate ~5-10 Mpc modes near the observer, which dominate pair yield);
embed_seed = the sub-coarse-Nyquist (<~4 Mpc) modes (the actual MW/M31 halos). One process per
(cr_seed, embed_seed) -- two 576^3 forwards in one process OOM the 32 GB Ada -- appending a TSV
row. --rank-only reads the TSV and ranks.

SCORE FIX (fable): the observed MW-M31 approach -110 km/s is the TOTAL physical radial velocity
= v_pec_radial + 100*sep (Hubble term, sep in h^-1Mpc). Kinematics are unreadable at 576^3 (pair
< PM force resolution), so approach/tangential are RECORDED and weakly weighted, judged later by
GOTPM, not trusted here. Also record pair->Virgo and pair->Void distances (offset direction
matters: a -SGZ offset degrades the void geometry).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIRGO_SG = (102.9, -2.3, 16.5)
VOID_D = 17.2
COLS = ["cseed", "eseed", "npair", "virgo_M", "infall", "void",
        "sep", "Mi", "Mj", "rmid", "vtot", "vtan", "p2vir", "p2void", "virgo_od", "obs_big"]


def sgdir(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def pair_score(sep, Mi, Mj, rmid, vtot, vtan, p2void):
    return (abs(np.log10(Mi) - 12.1) + abs(np.log10(Mj) - 12.1)
            + 0.7 * abs(sep - 0.57) / 0.50          # sep < PM force res at 576^3: loose, not sharp (fable)
            + 0.5 * abs(vtot + 110) / 80            # total radial vel vs observed -110 (weak)
            + 0.25 * vtan / 60                      # tangential vs Gaia <~60 (weak)
            + 0.15 * rmid
            + 0.10 * abs(p2void - VOID_D) / 5)      # keep the void geometry


def rank_only(tsv):
    rows = []
    with open(tsv) as f:
        for line in f:
            if line.startswith("cseed\t") or not line.strip():
                continue
            v = line.split()
            rows.append({c: (float(v[i]) if i < len(v) else float("nan")) for i, c in enumerate(COLS)})
    have = [r for r in rows if r["npair"] > 0 and not np.isnan(r["sep"])]
    for r in have:
        r["score"] = pair_score(r["sep"], r["Mi"], r["Mj"], r["rmid"], r["vtot"], r["vtan"], r["p2void"])
    have.sort(key=lambda r: r["score"])
    print(f"[rank] {len(rows)} (cr,embed) screened, {len(have)} produced an LG pair. "
          f"approaching (vtot<0): {sum(r['vtot']<0 for r in have)}. Ranked:")
    print(f"   {'cr':>3} {'emb':>3} {'score':>6} {'sep':>5} {'Mi':>8} {'Mj':>8} {'r_off':>6} "
          f"{'vtot':>6} {'vtan':>5} {'p2vir':>6} {'p2void':>6} {'VirgoM':>8}")
    for r in have[:15]:
        print(f"   {int(r['cseed']):>3} {int(r['eseed']):>3} {r['score']:>6.2f} {r['sep']:>5.2f} "
              f"{r['Mi']:>8.1e} {r['Mj']:>8.1e} {r['rmid']:>6.1f} {r['vtot']:>+6.0f} {r['vtan']:>5.0f} "
              f"{r['p2vir']:>6.1f} {r['p2void']:>6.1f} {r['virgo_M']:>8.1e}")
    if have:
        w = have[0]
        print(f"\n[rank] WINNER cr={int(w['cseed'])} embed={int(w['eseed'])}: sep={w['sep']:.2f} "
              f"M=({w['Mi']:.1e},{w['Mj']:.1e}) vtot={w['vtot']:+.0f}(t-110) vtan={w['vtan']:.0f} "
              f"r_off={w['rmid']:.1f} p2vir={w['p2vir']:.1f} p2void={w['p2void']:.1f}")
    # cr-seed spread of the (seed-robust-WITHIN-cr) environment
    import collections
    byc = collections.defaultdict(list)
    for r in rows:
        byc[int(r["cseed"])].append(r)
    print(f"\n[rank] environment spread ACROSS cr_seeds (the real robustness axis):")
    vm = [np.median([x["virgo_M"] for x in g]) for g in byc.values()]
    inf = [np.median([x["infall"] for x in g]) for g in byc.values()]
    vd = [np.median([x["void"] for x in g]) for g in byc.values()]
    print(f"   VirgoM median/cr: {np.min(vm):.1e}..{np.max(vm):.1e}; "
          f"infall: {np.min(inf):+.0f}..{np.max(inf):+.0f}; void: {np.min(vd):.2f}..{np.max(vd):.2f}")
    return have


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192_hr.npz")
    ap.add_argument("--key", default="s_out")
    ap.add_argument("--cseed", type=int, default=1, help="cr_seed label (which HR realization)")
    ap.add_argument("--seed", type=int, default=None, help="single embed seed to process")
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--half", type=float, default=30.0)
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--tsv", default="recon/screen3b_rows.tsv")
    ap.add_argument("--rank-only", action="store_true")
    args = ap.parse_args()

    if args.rank_only:
        rank_only(args.tsv); return

    from mock_pipeline import make_forward, RHO_CRIT, VUNIT_KMS
    from cf4_make_ic import embed_ic
    from fof import fof
    import jax.numpy as jnp
    import time

    z = np.load(args.recon)
    s = z[args.key].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    L = Nc * spc; c = L / 2.0; H = float(z["hh"]) if "hh" in z else 0.746
    sp = L / args.Nfine; m_p = args.Om * RHO_CRIT * sp ** 3
    virgo = c + sgdir(*VIRGO_SG[:2]) * (VIRGO_SG[2] * H)
    void = c + sgdir(78.0, 74.0) * VOID_D
    eseed = args.seed
    print(f"[scr] cr={args.cseed} embed={eseed} recon={os.path.basename(args.recon)} "
          f"N={args.Nfine} m_p={m_p:.2e}", flush=True)

    t0 = time.time()
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    s_fine = embed_ic(s, args.Nfine, eseed)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    vel = np.asarray(ptcl.vel).astype(np.float64) * VUNIT_KMS
    nbar = pos.shape[0] / L ** 3
    print(f"[scr] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)

    def bulk(center, R):
        sel = np.linalg.norm(pos - center, axis=1) < R
        return vel[sel].mean(0) if sel.sum() else np.zeros(3)
    v_LG = bulk(c, 2.5); v_Vir = bulk(virgo, 3.0)
    rhat = (virgo - c) / np.linalg.norm(virgo - c)
    infall = float(np.dot(v_LG - v_Vir, rhat))
    void_od = np.sum(np.linalg.norm(pos - void, axis=1) < 8.0) / (nbar*4/3*np.pi*8.0**3)
    # particle-based 1+delta at the Virgo position (fable: immune to FoF bridge/split, unlike virgo_M)
    virgo_od = np.sum(np.linalg.norm(pos - virgo, axis=1) < 8.0) / (nbar*4/3*np.pi*8.0**3)

    m = np.all(np.abs(pos - c) < args.half, axis=1); gi = np.flatnonzero(m)
    halos = fof(pos[gi], vel[gi], L=L, mean_sep=sp, b=0.2, n_min=20,
                m_particle=m_p, periodic=False, verbose=False)
    hp = halos["pos"]; hm = halos["mass"]; hv = halos["vel"]
    rh = np.linalg.norm(hp - c, axis=1)
    dvir = np.linalg.norm(hp - virgo, axis=1)
    virgo_M = float(hm[dvir < 4].max()) if np.any(dvir < 4) else 0.0

    mw = np.flatnonzero((hm > 5e11) & (hm < 4e12) & (rh < args.rmax))
    big = hp[hm > 5e12]
    # near-observer cleanliness (fable): distance from observer to nearest cluster-scale halo.
    # cr3/cr12 have a 1.5-3.1e13 group at ~2.7 Mpc that blocks a clean LG; cr6 has none within 9 Mpc.
    obs_big = float(np.linalg.norm(big - c, axis=1).min()) if len(big) else 99.0
    pairs = []
    for a in range(len(mw)):
        for b in range(a + 1, len(mw)):
            i, j = mw[a], mw[b]
            svec = hp[i] - hp[j]; d = np.linalg.norm(svec); rp = svec / d
            if not (0.3 < d < 1.2):
                continue
            mid = 0.5 * (hp[i] + hp[j])
            if len(big) and np.linalg.norm(big - mid, axis=1).min() < 3.0:
                continue
            vrelpec = hv[i] - hv[j]
            vrad_pec = float(np.dot(vrelpec, rp))
            vtot = vrad_pec + 100.0 * d                     # + Hubble term (total radial vel)
            vtan = float(np.linalg.norm(vrelpec - vrad_pec * rp))
            p2vir = float(np.linalg.norm(mid - virgo)); p2void = float(np.linalg.norm(mid - void))
            pairs.append((d, float(hm[i]), float(hm[j]), float(np.linalg.norm(mid-c)),
                          vtot, vtan, p2vir, p2void, int(i), int(j)))
    # pair tuple = (sep,Mi,Mj,rmid,vtot,vtan,p2vir,p2void,i,j); pair_score wants indices 0-5,7
    def score_of(p):
        return pair_score(p[0], p[1], p[2], p[3], p[4], p[5], p[7])
    best = min(pairs, key=score_of) if pairs else None
    if best:
        sep, Mi, Mj, rmid, vtot, vtan, p2vir, p2void = best[:8]
        print(f"[scr] cr{args.cseed} e{eseed}: {len(pairs)} pairs BEST sep={sep:.2f} "
              f"M=({Mi:.1e},{Mj:.1e}) r={rmid:.1f} vtot={vtot:+.0f} vtan={vtan:.0f} "
              f"p2vir={p2vir:.1f} p2void={p2void:.1f} score={score_of(best):.2f} | "
              f"VirgoM={virgo_M:.1e} vOD={virgo_od:.2f} infall={infall:+.0f} void={void_od:.2f} "
              f"obs_big={obs_big:.1f}", flush=True)
    else:
        sep = Mi = Mj = rmid = vtot = vtan = p2vir = p2void = float("nan")
        print(f"[scr] cr{args.cseed} e{eseed}: 0 pairs | VirgoM={virgo_M:.1e} vOD={virgo_od:.2f} "
              f"infall={infall:+.0f} void={void_od:.2f} obs_big={obs_big:.1f}", flush=True)

    new = not os.path.exists(args.tsv)
    with open(args.tsv, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(x) for x in
                [args.cseed, eseed, len(pairs), virgo_M, infall, void_od,
                 sep, Mi, Mj, rmid, vtot, vtan, p2vir, p2void, virgo_od, obs_big]) + "\n")
    if best:
        np.savez(f"recon/screen3b_halos_cr{args.cseed}_e{eseed}.npz",
                 cseed=args.cseed, eseed=eseed, halo_pos=hp.astype(np.float32),
                 halo_mass=hm.astype(np.float32), halo_vel=hv.astype(np.float32),
                 best=np.array(best), m_p=m_p, L=L, sp=sp, recon=args.recon)
    print(f"[scr] cr{args.cseed} e{eseed} appended to {args.tsv}", flush=True)


if __name__ == "__main__":
    main()
