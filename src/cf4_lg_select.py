#!/usr/bin/env python
"""Direction-aware selection of the constrained LG zoom target (per fable's criterion).

A ~5 Mpc offset of the LG from the box centre is acceptable IF it is perpendicular to the
real LG->Virgo direction, so the LG-Virgo distance (and the Local Void geometry) stays ~right.
For every MW-M31 pair candidate across all lg_search combos, compute the pair's offset from
the observer and the resulting LG->Virgo distance, and pick the pair that (i) keeps Virgo near
its true 12.3 h^-1Mpc, (ii) is a good LG pair (masses ~1e12, sep ~0.57, isolated, approaching).
"""
import os
import glob
import numpy as np

VIRGO_SG = (102.9, -2.3, 16.5)          # supergalactic (l,b,d[Mpc])
H = 0.746
VIRGO_D = VIRGO_SG[2] * H               # ~12.3 h^-1Mpc


def sg(l, b, d=1.0):
    l, b = np.radians(l), np.radians(b)
    return d * np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="recon/lg_search")
    ap.add_argument("--glob", default="lg_smapf*.npz",
                    help="combos to select over (default s_map field: Virgo actually preserved)")
    args = ap.parse_args()
    vhat = sg(*VIRGO_SG[:2])                 # unit vector to Virgo
    cand = []
    vd2_all = []
    for f in sorted(glob.glob(os.path.join(args.dir, args.glob))):
        z = np.load(f)
        hp = z["halo_pos"].astype(float); hm = z["halo_mass"].astype(float)
        hv = z["halo_vel"].astype(float); L = float(z["L"]); c = L / 2.0
        vd2 = float(z["virgo_d2"]) if "virgo_d2" in z else -1.0  # MEASURED Virgo 1+delta(<2Mpc)
        vd2_all.append(vd2)
        tag = os.path.basename(f)[3:-4]
        rh = np.linalg.norm(hp - c, axis=1)
        mw = np.flatnonzero((hm > 5e11) & (hm < 4e12) & (rh < 20))
        big = hp[hm > 5e12]
        for a in range(len(mw)):
            for b in range(a + 1, len(mw)):
                i, j = mw[a], mw[b]
                sep = np.linalg.norm(hp[i] - hp[j])
                if not (0.4 < sep < 0.9):
                    continue
                mid = 0.5 * (hp[i] + hp[j])
                if len(big) and np.linalg.norm(big - mid, axis=1).min() < 3.0:
                    continue
                off = mid - c                           # LG offset from observer
                roff = np.linalg.norm(off)
                lg_virgo = np.linalg.norm(vhat * VIRGO_D - off)   # LG->Virgo distance
                vrel = np.dot(hv[i]-hv[j], (hp[i]-hp[j])/sep)     # MW-M31 radial (approach<0)
                vlg = (hm[i]*hv[i] + hm[j]*hv[j]) / (hm[i]+hm[j]) # LG bulk (mass-weighted)
                v_infall = np.dot(vlg, vhat)             # LG velocity toward Virgo (infall>0)
                cand.append(dict(tag=tag, sep=sep, Mi=hm[i], Mj=hm[j], roff=roff,
                                 lg_virgo=lg_virgo, vrel=vrel, v_infall=v_infall,
                                 off=off, mid=mid))
    if not cand:
        print("no pair candidates"); return
    vd2 = np.array([v for v in vd2_all if v >= 0])
    if len(vd2):
        print(f"MEASURED Virgo 1+delta(<2Mpc) across combos: min={vd2.min():.1f} max={vd2.max():.1f} "
              f"(overdense -> LG->Virgo distances below are to the REAL Virgo cluster, not a proxy)\n")
    # SOFT chi^2 over all constraints, each term normalized by its realistic scatter (per fable).
    # Hard cuts were already applied (isolation: no >5e12 within 3 Mpc). Rank, don't reject.
    #   target (sigma):  LG-Virgo 12.3 (2.5)  infall +200 (60)  approach -110 (40)
    #                    MW mass 12.05 (0.2)  M31 mass 12.15 (0.2)  sep 0.57 (0.12)
    def chi2(k):
        return ((k["lg_virgo"] - VIRGO_D) / 2.5) ** 2 \
             + ((k["v_infall"] - 200.0) / 60.0) ** 2 \
             + ((k["vrel"] + 110.0) / 40.0) ** 2 \
             + ((np.log10(max(k["Mi"], k["Mj"])) - 12.15) / 0.2) ** 2 \
             + ((np.log10(min(k["Mi"], k["Mj"])) - 12.05) / 0.2) ** 2 \
             + ((k["sep"] - 0.57) / 0.12) ** 2
    for k in cand:
        k["chi2"] = chi2(k)
    cand.sort(key=lambda k: k["chi2"])
    print(f"targets(sigma): LG-Virgo={VIRGO_D:.1f}(2.5), infall+200(60), approach-110(40), "
          f"sep0.57(0.12), M~1e12(0.2 dex)\n")
    print(f"  {'combo':>8} {'chi2':>6} {'sep':>5} {'Mi':>8} {'Mj':>8} {'r_off':>6} "
          f"{'LG-Vir':>7} {'appr':>6} {'infall':>7}")
    for k in cand[:12]:
        print(f"  {k['tag']:>8} {k['chi2']:>6.1f} {k['sep']:>5.2f} {k['Mi']:>8.1e} {k['Mj']:>8.1e} "
              f"{k['roff']:>6.1f} {k['lg_virgo']:>7.1f} {k['vrel']:>6.0f} {k['v_infall']:>7.0f}")
    w = cand[0]
    print(f"\nWINNER: {w['tag']}  chi2={w['chi2']:.1f}  MW-M31 sep={w['sep']:.2f} "
          f"M=({w['Mi']:.1e},{w['Mj']:.1e}) offset={w['roff']:.1f}  "
          f"LG->Virgo={w['lg_virgo']:.1f}(t{VIRGO_D:.0f}) approach={w['vrel']:.0f}(t-110) "
          f"infall={w['v_infall']:.0f}(t+200)")
    print(f"  LG centre (box frame) = ({w['mid'][0]:.1f},{w['mid'][1]:.1f},{w['mid'][2]:.1f})")


if __name__ == "__main__":
    main()
