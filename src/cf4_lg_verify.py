#!/usr/bin/env python
"""Visual validation of the confirmed constrained LG (c48f3) against the 3 constraints.

From the saved central particles + halos, show the LG in its environment: the MW-M31 pair,
Virgo (distance/direction), the Local Void (anti-Virgo underdensity), and the velocity
vectors (MW-M31 approach; LG bulk motion / Virgocentric infall). Panels: two supergalactic
density slices through the LG, plus an all-sky column-density map seen FROM the LG.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
VIRGO = (102.9, -2.3, 16.5); COMA = (89.0, 8.0, 90.0); GA = (155.0, -6.0, 65.0)
H = 0.746
VIRGO_D = VIRGO[2] * H


def sg(l, b, d=1.0):
    l, b = np.radians(l), np.radians(b)
    return d * np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lg", default="recon/lg_search/lg_c48f3.npz")
    ap.add_argument("--out", default="recon/cf4_lg_verify.png")
    args = ap.parse_args()
    from scipy.ndimage import gaussian_filter, map_coordinates
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    z = np.load(args.lg)
    cp = z["cen_pos"].astype(float); hp = z["halo_pos"].astype(float)
    hm = z["halo_mass"].astype(float); hv = z["halo_vel"].astype(float)
    sp = float(z["sp"]); L = float(z["L"]); c = L / 2.0
    vhat = sg(*VIRGO[:2])

    # find the winning MW-M31 pair (best chi2, same as selection)
    rh = np.linalg.norm(hp - c, axis=1)
    mw = np.flatnonzero((hm > 5e11) & (hm < 4e12) & (rh < 20)); big = hp[hm > 5e12]
    best = None
    for a in range(len(mw)):
        for b in range(a+1, len(mw)):
            i, j = mw[a], mw[b]; sep = np.linalg.norm(hp[i]-hp[j])
            if not (0.4 < sep < 0.9): continue
            mid = 0.5*(hp[i]+hp[j])
            if len(big) and np.linalg.norm(big-mid, axis=1).min() < 3.0: continue
            off = mid-c; lgv = np.linalg.norm(vhat*VIRGO_D - off)
            vrel = np.dot(hv[i]-hv[j], (hp[i]-hp[j])/sep)
            vlg = (hm[i]*hv[i]+hm[j]*hv[j])/(hm[i]+hm[j]); infall = np.dot(vlg, vhat)
            chi2 = (((lgv-VIRGO_D)/2.5)**2 + ((infall-200)/60)**2 + ((vrel+110)/40)**2
                    + ((np.log10(max(hm[i],hm[j]))-12.15)/0.2)**2
                    + ((np.log10(min(hm[i],hm[j]))-12.05)/0.2)**2 + ((sep-0.57)/0.12)**2)
            if best is None or chi2 < best[0]: best = (chi2, i, j, mid, vlg, infall, vrel, sep)
    chi2, i, j, lg, vlg, infall, vrel, sep = best
    virgo_pos = c + vhat * VIRGO_D           # Virgo location in the box frame

    # density around the LG (grid the central particles)
    Ng = 96; cell = 80.0 / Ng                 # cen_pos spans +-40 about box centre
    def grid(sl_axes):
        idx = np.floor((cp - (c-40)) / cell).astype(int)
        ok = np.all((idx >= 0) & (idx < Ng), axis=1); idx = idx[ok]
        f = np.zeros((Ng, Ng, Ng), np.float32); np.add.at(f, (idx[:,0], idx[:,1], idx[:,2]), 1.0)
        return gaussian_filter(f, 2.0/cell)
    dens = grid(None); dens = dens/dens.mean() - 1.0
    axg = (np.arange(Ng)+0.5)*cell + (c-40)   # box-frame coord of each cell centre

    fig = plt.figure(figsize=(15, 11))
    # two SG slices through the LG
    for panel, (u, v, w, ulab, vlab) in enumerate([(0, 1, 2, "SGX", "SGY"), (0, 2, 1, "SGX", "SGZ")]):
        ax = fig.add_subplot(2, 2, panel+1)
        kw = int((lg[w]-(c-40))/cell)
        sl = dens.take(kw, axis=w).T
        ext = [axg[0]-c, axg[-1]-c, axg[0]-c, axg[-1]-c]
        ax.imshow(sl, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="equal")
        ax.plot(hp[i][u]-c, hp[i][v]-c, "*", color="yellow", ms=18, mec="k", label="MW")
        ax.plot(hp[j][u]-c, hp[j][v]-c, "*", color="orange", ms=15, mec="k", label="M31")
        for nm, (l, b, d) in [("Virgo", VIRGO), ("Coma", COMA), ("GA", GA)]:
            x = sg(l, b, d)*H
            if abs(x[w]) < 20: ax.plot(x[u], x[v], "kD", ms=8); ax.annotate(nm, (x[u], x[v]), fontsize=9)
        ax.plot(0, 0, "g+", ms=12, mew=2)     # observer (box centre)
        # velocity arrows at the LG: MW-M31 approach (relative) and LG bulk (infall)
        sc = 0.02
        rel = (hv[i]-hv[j])
        ax.arrow((lg-c)[u], (lg-c)[v], sc*vlg[u], sc*vlg[v], color="cyan", width=0.15,
                 head_width=1.2, label="LG bulk vel")
        ax.arrow((hp[i]-c)[u], (hp[i]-c)[v], sc*rel[u], sc*rel[v], color="lime", width=0.1, head_width=1.0)
        ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
        ax.set_xlabel(f"{ulab} [$h^{{-1}}$Mpc]"); ax.set_ylabel(f"{vlab} [$h^{{-1}}$Mpc]")
        ax.set_title(f"{ulab}-{vlab} through LG  (cyan=LG bulk vel, lime=MW-M31 rel vel)")
        if panel == 0: ax.legend(loc="upper right", fontsize=8)

    # all-sky column density seen FROM the LG (0.7-20 h^-1Mpc), supergalactic Mollweide
    ax = fig.add_subplot(2, 1, 2, projection="mollweide")
    dc = grid(None); dc = dc                       # (1+delta) density on the grid
    nlon, nlat = 240, 120
    lon = np.linspace(-np.pi, np.pi, nlon); lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    LON, LAT = np.meshgrid(lon, lat)
    rx = np.cos(LAT)*np.cos(LON); ry = np.cos(LAT)*np.sin(LON); rz = np.sin(LAT)
    acc = np.zeros_like(LON)
    for rr in np.arange(0.7, 20, 0.5):
        px = (lg[0] + rr*rx - (c-40))/cell; py = (lg[1] + rr*ry - (c-40))/cell; pz = (lg[2] + rr*rz - (c-40))/cell
        acc += map_coordinates(dc, [px.ravel(), py.ravel(), pz.ravel()], order=1, mode="nearest").reshape(LON.shape)
    acc = gaussian_filter(np.log10(acc/acc.mean()+0.3), 1.5)
    ax.pcolormesh(LON, LAT, acc, cmap="RdBu_r", shading="auto")
    # mark Virgo & anti-Virgo (Local Void) directions as seen from the LG
    for nm, dvec, mk in [("Virgo", virgo_pos-lg, "kD"), ("Local Void?", -(virgo_pos-lg), "wo")]:
        dv = dvec/np.linalg.norm(dvec); vl = np.arctan2(dv[1], dv[0]); vb = np.arcsin(dv[2])
        ax.plot(vl, vb, mk, ms=10, mec="k"); ax.annotate(nm, (vl, vb), fontsize=9, color="k")
    ax.set_title("all-sky density FROM the LG (supergalactic): Virgo overdense, anti-Virgo = Local Void")
    ax.grid(True, alpha=0.3); ax.set_xticklabels([]); ax.set_yticklabels([])

    fig.suptitle(f"Constrained LG (c48f3) validation:  MW-M31 sep={sep:.2f}, approach={vrel:.0f}, "
                 f"LG-Virgo={np.linalg.norm(lg-virgo_pos):.1f} (t{VIRGO_D:.0f}), infall={infall:.0f} km/s (chi2={chi2:.1f})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(args.out, dpi=115)
    print(f"[verify] saved {args.out}")
    print(f"[verify] LG={lg}, Virgo dir dist={np.linalg.norm(lg-virgo_pos):.1f}, "
          f"infall={infall:.0f}, approach={vrel:.0f} km/s")


if __name__ == "__main__":
    main()
