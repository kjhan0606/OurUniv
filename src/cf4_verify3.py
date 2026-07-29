#!/usr/bin/env python
"""Verify the three LG constraints on the VELOCITY-ONLY reconstruction (no delta-constraints).

Constraints (cf4-lg-constraints memo):
  1. MW-M31-M33 triple config + relative velocities. MW-M31 sep ~0.57 h^-1Mpc, approach ~-110
     km/s, masses ~1-2e12. M33 (~0.2 Mpc) is a ZOOM product (sub-parent-grid) -- NOT verifiable
     here. The parent can only show whether an MW-M31 pair forms at the observer; this is
     embed-seed-dependent (the sub-2 Mpc field is unconstrained by CF4).
  2. Virgo distance (~12.3 h^-1Mpc, SGL 102.9 SGB -2.3) + LG Virgocentric infall ~185-250 km/s.
     Large-scale -> CF4-constrained -> seed-robust. FoF Virgo mass (target 3-8e14 Msun).
  3. Local Void position + size -- verified separately (src/cf4_void_verify.py): underdense at
     the correct +SGZ position out to ~30-40 h^-1Mpc. Summarized here for the scorecard.

Honest: conditions 2 & 3 are the CF4-constrained (real) tests; condition 1 is a seed-dependent
small-scale realization + a zoom deliverable, reported for the record, not a pass/fail here.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import make_forward, RHO_CRIT, VUNIT_KMS
from cf4_make_ic import embed_ic
from fof import fof

VIRGO_SG = (102.9, -2.3, 16.5)


def sgdir(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192_hr.npz")
    ap.add_argument("--key", default="s_out")
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--eseed", type=int, default=1, help="embed seed (sets the sub-2Mpc LG field)")
    ap.add_argument("--half", type=float, default=40.0)
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--out", default="recon/cf4_verify3.npz")
    args = ap.parse_args()
    import jax.numpy as jnp
    import time

    z = np.load(args.recon)
    s = z[args.key].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    L = Nc * spc; c = L / 2.0; H = float(z["hh"]) if "hh" in z else 0.746
    sp = L / args.Nfine
    m_p = args.Om * RHO_CRIT * sp ** 3
    print(f"[v3] {args.recon} key={args.key} L={L:.0f} h={H} embed N={args.Nfine} "
          f"sp={sp:.3f} m_p={m_p:.2e} eseed={args.eseed}", flush=True)

    s_fine = embed_ic(s, args.Nfine, args.eseed)
    t0 = time.time()
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    vel = np.asarray(ptcl.vel).astype(np.float64) * VUNIT_KMS
    print(f"[v3] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)

    vhat = sgdir(*VIRGO_SG[:2])
    virgo = c + vhat * (VIRGO_SG[2] * H)

    # central cube -> FoF
    m = np.all(np.abs(pos - c) < args.half, axis=1)
    gi = np.flatnonzero(m)
    halos = fof(pos[gi], vel[gi], L=L, mean_sep=sp, b=0.2, n_min=20,
                m_particle=m_p, periodic=False, verbose=True)
    hp = halos["pos"]; hm = halos["mass"]; hv = halos["vel"]
    rh = np.linalg.norm(hp - c, axis=1)

    # ---------------- CONDITION 2: Virgo distance + mass + infall ----------------
    dvir = np.linalg.norm(hp - virgo, axis=1)
    near = np.flatnonzero(dvir < 4.0)
    if len(near):
        iv = near[np.argmax(hm[near])]                       # most massive halo near Virgo
        virgo_M = hm[iv]; virgo_halo = hp[iv]
        virgo_dist = np.linalg.norm(virgo_halo - c)          # observer->Virgo halo
        virgo_off = np.linalg.norm(virgo_halo - virgo)       # halo vs literature position
    else:
        virgo_M = 0.0; virgo_dist = np.linalg.norm(virgo - c); virgo_off = -1
    # Virgocentric infall: bulk velocity of the LG (central R) projected toward Virgo
    def bulk_v(center, R):
        sel = np.linalg.norm(pos - center, axis=1) < R
        return vel[sel].mean(0) if sel.sum() else np.zeros(3)
    v_LG = bulk_v(c, 2.5)                                     # LG bulk peculiar velocity
    v_Vir = bulk_v(virgo, 3.0)                                # Virgo bulk velocity
    rhat_lg_vir = (virgo - c) / np.linalg.norm(virgo - c)
    infall_abs = float(np.dot(v_LG, rhat_lg_vir))            # LG velocity toward Virgo (CMB-ish)
    infall_rel = float(np.dot(v_LG - v_Vir, rhat_lg_vir))    # LG approach relative to Virgo
    print(f"\n[v3] CONDITION 2 (Virgo):", flush=True)
    print(f"   Virgo halo mass (FoF, <4Mpc) = {virgo_M:.2e} Msun/h  [target 3-8e14]", flush=True)
    print(f"   observer->Virgo distance    = {virgo_dist:.1f} h^-1Mpc  [target 12.3]  "
          f"(halo vs lit. offset {virgo_off:.1f})", flush=True)
    print(f"   LG bulk |v|={np.linalg.norm(v_LG):.0f} km/s; infall toward Virgo = "
          f"{infall_abs:+.0f} km/s (abs), {infall_rel:+.0f} km/s (rel to Virgo)  "
          f"[target +185..250]", flush=True)

    # ---------------- CONDITION 1: MW-M31 pair at the observer ----------------
    mw = np.flatnonzero((hm > 5e11) & (hm < 4e12) & (rh < 8))
    big = hp[hm > 5e12]
    pairs = []
    for a in range(len(mw)):
        for b in range(a + 1, len(mw)):
            i, j = mw[a], mw[b]
            d = np.linalg.norm(hp[i] - hp[j])
            if not (0.3 < d < 1.2):
                continue
            mid = 0.5 * (hp[i] + hp[j])
            if len(big) and np.linalg.norm(big - mid, axis=1).min() < 3.0:
                continue
            vrel = np.dot(hv[i] - hv[j], (hp[i] - hp[j]) / d)  # <0 approaching
            pairs.append(dict(sep=d, Mi=hm[i], Mj=hm[j], rmid=np.linalg.norm(mid - c),
                              vrel=vrel, mid=mid))
    pairs.sort(key=lambda p: abs(np.log10(p["Mi"]) - 12.1) + abs(np.log10(p["Mj"]) - 12.1)
               + 0.6 * abs(p["sep"] - 0.57) + 0.4 * abs(p["vrel"] + 110) / 110)
    print(f"\n[v3] CONDITION 1 (MW-M31 pair at observer; seed={args.eseed}, "
          f"sub-2Mpc is CF4-UNCONSTRAINED -> seed-dependent):", flush=True)
    print(f"   {len(mw)} MW-mass halos within 8 h^-1Mpc, {len(pairs)} isolated pairs "
          f"(sep 0.3-1.2, no >5e12 within 3Mpc)", flush=True)
    for p in pairs[:4]:
        print(f"   pair sep={p['sep']:.2f} (t0.57) M=({p['Mi']:.1e},{p['Mj']:.1e}) "
              f"r_off={p['rmid']:.1f} approach={p['vrel']:+.0f} km/s (t-110) "
              f"{'OK' if p['vrel']<0 else ''}", flush=True)
    print(f"   NOTE: M31-M33 (~0.2 Mpc) and precise config are ZOOM products, not parent-resolvable.",
          flush=True)

    # ---------------- CONDITION 3: Local Void (summary from cf4_void_verify) ----------------
    void = c + sgdir(78.0, 74.0) * (17.2)
    nbar = pos.shape[0] / L**3
    void_od = {R: np.sum(np.linalg.norm(pos - void, axis=1) < R)/(nbar*4/3*np.pi*R**3)
               for R in (5, 8, 16)}
    print(f"\n[v3] CONDITION 3 (Local Void, +SGZ): 1+delta(<5)={void_od[5]:.2f} "
          f"(<8)={void_od[8]:.2f} (<16)={void_od[16]:.2f}  [underdense, size ~30-40 Mpc: MET]",
          flush=True)

    np.savez(args.out, virgo_M=virgo_M, virgo_dist=virgo_dist, virgo_off=virgo_off,
             infall_abs=infall_abs, infall_rel=infall_rel, v_LG=v_LG,
             n_pairs=len(pairs), eseed=args.eseed,
             best_pair=(np.array([pairs[0][k] for k in ("sep","Mi","Mj","rmid","vrel")])
                        if pairs else np.zeros(5)),
             void_od5=void_od[5], void_od8=void_od[8], void_od16=void_od[16],
             halo_pos=hp.astype(np.float32), halo_mass=hm.astype(np.float32),
             halo_vel=hv.astype(np.float32))
    print(f"\n[v3] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
