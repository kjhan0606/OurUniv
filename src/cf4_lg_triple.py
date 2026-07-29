#!/usr/bin/env python
"""Rank Local-Group TRIPLE candidates (MW + M31 + M33) from the lg_search halo catalogs.

The parent resolution cannot separate M31 and M33 (0.15 h^-1Mpc apart, sub-grid), so at
this level M33 shows only as a smaller companion near M31 -- the true M31-M33 split is a
zoom product. Here we score each (cseed,fseed) by its best triple: an MW-M31 pair
(two ~1e12, 0.4-0.9 h^-1Mpc, isolated) plus an M33 candidate (~3-8e11) near one member.
"""
import os
import glob
import numpy as np


def triples(hp, hm, c, rmax=25.0):
    rh = np.linalg.norm(hp - c, axis=1)
    mw = np.flatnonzero((hm > 5e11) & (hm < 4e12) & (rh < rmax))     # MW/M31 candidates
    m33 = np.flatnonzero((hm > 2e11) & (hm < 9e11) & (rh < rmax + 3))  # M33 candidates
    big = hp[hm > 5e12]
    out = []
    for a in range(len(mw)):
        for b in range(a + 1, len(mw)):
            i, j = mw[a], mw[b]
            sep = np.linalg.norm(hp[i] - hp[j])
            if not (0.4 < sep < 0.9):
                continue
            mid = 0.5 * (hp[i] + hp[j])
            if len(big) and np.linalg.norm(big - mid, axis=1).min() < 3.0:
                continue                                             # not isolated
            # M33 candidate: near either member (<0.6), lighter than both, not i/j
            d_i = np.linalg.norm(hp[m33] - hp[i], axis=1)
            d_j = np.linalg.norm(hp[m33] - hp[j], axis=1)
            near = m33[(np.minimum(d_i, d_j) < 0.6) & (m33 != i) & (m33 != j)]
            near = [k for k in near if hm[k] < min(hm[i], hm[j])]
            k3 = near[int(np.argmin([min(np.linalg.norm(hp[k]-hp[i]),
                                         np.linalg.norm(hp[k]-hp[j])) for k in near]))] if near else -1
            out.append(dict(i=int(i), j=int(j), k=int(k3), sep=sep,
                            Mi=float(hm[i]), Mj=float(hm[j]),
                            Mk=float(hm[k3]) if k3 >= 0 else 0.0,
                            rmid=float(np.linalg.norm(mid - c)), has_m33=k3 >= 0))
    # prefer: has M33, central (small rmid), MW/M31 masses ~1e12, sep ~0.57
    out.sort(key=lambda t: (0 if t["has_m33"] else 1,
                            abs(np.log10(t["Mi"]) - 12.1) + abs(np.log10(t["Mj"]) - 12.1)
                            + 0.5 * abs(t["sep"] - 0.57) + 0.05 * t["rmid"]))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="recon/lg_search")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "lg_c*f*.npz")))
    if not files:
        print("no lg_c*f*.npz yet"); return
    print(f"{'combo':>7} {'Ntrip':>6} {'m33?':>5} {'sep':>5} {'Mi':>8} {'Mj':>8} {'Mk(M33)':>9} {'rmid':>6}")
    best_overall = None
    for f in files:
        z = np.load(f)
        hp = z["halo_pos"].astype(float); hm = z["halo_mass"].astype(float)
        L = float(z["L"]); c = L / 2.0
        tag = os.path.basename(f)[3:-4]
        ts = triples(hp, hm, c)
        n_m33 = sum(t["has_m33"] for t in ts)
        if ts:
            t = ts[0]
            m33s = f"{t['Mk']:.1e}" if t["has_m33"] else "--"
            print(f"{tag:>7} {len(ts):>6} {n_m33:>5} {t['sep']:>5.2f} {t['Mi']:>8.1e} "
                  f"{t['Mj']:>8.1e} {m33s:>9} {t['rmid']:>6.1f}")
            key = (0 if t["has_m33"] else 1, t["rmid"])
            if best_overall is None or key < best_overall[0]:
                best_overall = (key, tag, t)
        else:
            print(f"{tag:>7} {0:>6}      --    --       --       --        --     --")
    if best_overall:
        _, tag, t = best_overall
        print(f"\nBEST triple: {tag}  MW-M31 sep={t['sep']:.2f} h^-1Mpc "
              f"M=({t['Mi']:.1e},{t['Mj']:.1e}) M33={t['Mk']:.1e} "
              f"r_from_observer={t['rmid']:.1f} h^-1Mpc  {'(has M33 companion)' if t['has_m33'] else '(no M33 at parent res)'}")


if __name__ == "__main__":
    main()
