#!/usr/bin/env python
"""CF4 (Cosmicflows-4) group-catalog ingest -> clean .npz.

Reads data/cf4_groups.csv (VizieR J/ApJ/944/94/groups, 38053 rows) and produces
a clean, box-ready catalog for the peculiar-velocity reconstruction.

Per group we keep / derive:
  pgc       group principal-galaxy id (1PGC)
  ngal      galaxies in group
  dist      luminosity/TF distance [Mpc]        (Dist)
  v3k       CMB-frame recession velocity [km/s]  (V3k)
  vh, vls   heliocentric / local-sheet velocity [km/s]
  vpec      catalog peculiar velocity [km/s]     (Vpec)
  dm, e_dm  distance modulus DMzp and its error e_DMzp [mag]
  sig_lnd   fractional distance error  = e_DMzp * ln10/5   (log-distance sigma)
  sig_v     velocity error from distance error = H0 * dist * sig_lnd [km/s]
  nhat      (3,) supergalactic unit direction from (SGL,SGB)
  pos_dist  (3,) real-space SG position  = dist * nhat            [Mpc]
  pos_z     (3,) redshift-space SG position = (V3k/H0) * nhat     [Mpc]
  sgl,sgb   supergalactic long/lat [deg]
  glon,glat galactic long/lat [deg]  (for Zone-of-Avoidance mask)

CF4 uses H0 = 74.6 km/s/Mpc.  Distances are in the CMB frame.
The supergalactic SGX/SGY/SGZ columns in the catalog are in km/s (|SG| ~ Vls),
so real-space position is built from Dist * n_hat, NOT from SGX/Y/Z.

Cuts (light QC only; final box-fit cut is done by the box loader):
  * finite Dist, DMzp, SGL, SGB
  * Dist in (dist_min, dist_max]
  * |GLAT| > b_cut  (Zone of Avoidance)

Output: data/cf4_clean.npz  (+ data/cf4_diag.png diagnostic).
"""
import os
import argparse
import numpy as np

LN10_5 = np.log(10.0) / 5.0
H0_CF4 = 74.6

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _read_csv(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([r[k].strip() for r in rows], dtype=object) for k in rows[0]}
    return cols, len(rows)


def _f(col):
    """object str array -> float array with '' -> nan."""
    out = np.full(col.shape, np.nan)
    for i, v in enumerate(col):
        if v != "":
            out[i] = float(v)
    return out


def sg_unit(sgl_deg, sgb_deg):
    """Supergalactic unit vectors from (SGL, SGB) in degrees -> (N,3)."""
    l = np.deg2rad(sgl_deg)
    b = np.deg2rad(sgb_deg)
    cb = np.cos(b)
    return np.stack([cb * np.cos(l), cb * np.sin(l), np.sin(b)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "cf4_groups.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cf4_clean.npz"))
    ap.add_argument("--H0", type=float, default=H0_CF4)
    ap.add_argument("--dist-min", type=float, default=1.0, help="Mpc, drop nearer")
    ap.add_argument("--dist-max", type=float, default=250.0, help="Mpc, drop farther")
    ap.add_argument("--b-cut", type=float, default=5.0, help="deg, |GLAT|>b_cut kept")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cols, nraw = _read_csv(args.csv)
    print(f"[ingest] read {nraw} rows from {os.path.relpath(args.csv, ROOT)}")

    pgc = _f(cols["1PGC"])
    ngal = _f(cols["Ngal"])
    dist = _f(cols["Dist"])
    v3k = _f(cols["V3k"])
    vh = _f(cols["Vh"])
    vls = _f(cols["Vls"])
    vpec = _f(cols["Vpec"])
    dm = _f(cols["DMzp"])
    e_dm = _f(cols["e_DMzp"])
    glon = _f(cols["GLON"])
    glat = _f(cols["GLAT"])
    sgl = _f(cols["SGL"])
    sgb = _f(cols["SGB"])

    # QC mask
    finite = np.isfinite(dist) & np.isfinite(dm) & np.isfinite(sgl) & np.isfinite(sgb) \
        & np.isfinite(glat) & np.isfinite(v3k)
    keep = finite & (dist > args.dist_min) & (dist <= args.dist_max) \
        & (np.abs(glat) > args.b_cut)
    print(f"[ingest] cuts: finite & {args.dist_min}<Dist<={args.dist_max} Mpc & "
          f"|GLAT|>{args.b_cut} deg  ->  {int(keep.sum())} groups kept")

    def sel(a):
        return a[keep]

    pgc, ngal, dist, v3k, vh, vls, vpec, dm, e_dm, glon, glat, sgl, sgb = \
        map(sel, (pgc, ngal, dist, v3k, vh, vls, vpec, dm, e_dm, glon, glat, sgl, sgb))

    sig_lnd = e_dm * LN10_5
    # guard: a few groups have e_DMzp=0 (single high-quality indicator); floor it.
    sig_lnd = np.clip(sig_lnd, 0.02, None)
    sig_v = args.H0 * dist * sig_lnd

    nhat = sg_unit(sgl, sgb)                       # (N,3) SG unit vectors
    pos_dist = dist[:, None] * nhat                # real-space SG position [Mpc]
    pos_z = (v3k / args.H0)[:, None] * nhat        # redshift-space SG position [Mpc]

    # cross-check n_hat against normalized catalog SGX/Y/Z (should be ~parallel)
    sgx = _f(cols["SGX"])[keep]
    sgy = _f(cols["SGY"])[keep]
    sgz = _f(cols["SGZ"])[keep]
    sgv = np.stack([sgx, sgy, sgz], axis=1)
    sgn = sgv / (np.linalg.norm(sgv, axis=1, keepdims=True) + 1e-9)
    cosang = np.sum(sgn * nhat, axis=1)
    good = np.isfinite(cosang)
    print(f"[ingest] n_hat vs SGX/Y/Z direction: median cos={np.median(cosang[good]):.5f} "
          f"(expect ~1.0), min={np.min(cosang[good]):.4f}")

    # radial selection function (number density vs r), for the box loader's W(r)
    r = dist
    nbins = 24
    r_edges = np.linspace(0, args.dist_max, nbins + 1)
    counts, _ = np.histogram(r, bins=r_edges)
    shell_vol = (4 * np.pi / 3) * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    nbar = counts / np.maximum(shell_vol, 1e-9)
    r_cen = 0.5 * (r_edges[1:] + r_edges[:-1])

    out = dict(
        pgc=pgc.astype(np.int64), ngal=ngal.astype(np.int32),
        dist=dist, v3k=v3k, vh=vh, vls=vls, vpec=vpec,
        dm=dm, e_dm=e_dm, sig_lnd=sig_lnd, sig_v=sig_v,
        nhat=nhat, pos_dist=pos_dist, pos_z=pos_z,
        sgl=sgl, sgb=sgb, glon=glon, glat=glat,
        H0=np.float64(args.H0), d_max=np.float64(args.dist_max),
        d_min=np.float64(args.dist_min), b_cut=np.float64(args.b_cut),
        seln_r=r_cen, seln_nbar=nbar,
    )
    np.savez(args.out, **out)
    print(f"[ingest] wrote {os.path.relpath(args.out, ROOT)}  "
          f"({os.path.getsize(args.out)/1e6:.2f} MB)")

    print("[ingest] summary of kept sample:")
    print(f"    dist   [Mpc] : med {np.median(dist):.1f}  ({dist.min():.1f}..{dist.max():.1f})")
    print(f"    vpec [km/s]  : std {np.std(vpec):.1f}  med {np.median(vpec):+.1f}")
    print(f"    sig_lnd      : med {np.median(sig_lnd):.3f}")
    print(f"    sig_v [km/s] : med {np.median(sig_v):.1f}  (cf. vpec std -> S/N~{np.std(vpec)/np.median(sig_v):.2f})")

    if not args.no_plot:
        _plot(out, os.path.join(ROOT, "data", "cf4_diag.png"))


def _plot(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(d["dist"], bins=60, color="C0")
    ax[0, 0].set_xlabel("Dist [Mpc]"); ax[0, 0].set_title("distance")
    ax[0, 1].hist(d["vpec"], bins=80, color="C1")
    ax[0, 1].set_xlabel("Vpec [km/s]"); ax[0, 1].set_title(f"peculiar vel (std {d['vpec'].std():.0f})")
    ax[0, 2].hist(d["sig_lnd"], bins=60, color="C2")
    ax[0, 2].set_xlabel("sig_lnd"); ax[0, 2].set_title(f"log-dist error (med {np.median(d['sig_lnd']):.3f})")
    p = d["pos_dist"]
    sc = ax[1, 0].scatter(p[:, 0], p[:, 1], s=2, c=d["vpec"], cmap="RdBu_r",
                          vmin=-1000, vmax=1000)
    ax[1, 0].set_xlabel("SGX [Mpc]"); ax[1, 0].set_ylabel("SGY [Mpc]")
    ax[1, 0].set_title("SGX-SGY (color=Vpec)"); ax[1, 0].set_aspect("equal")
    plt.colorbar(sc, ax=ax[1, 0], fraction=0.046)
    ax[1, 1].scatter(p[:, 0], p[:, 2], s=2, c=d["vpec"], cmap="RdBu_r",
                     vmin=-1000, vmax=1000)
    ax[1, 1].set_xlabel("SGX [Mpc]"); ax[1, 1].set_ylabel("SGZ [Mpc]")
    ax[1, 1].set_title("SGX-SGZ"); ax[1, 1].set_aspect("equal")
    ax[1, 2].plot(d["seln_r"], d["seln_nbar"], "o-", color="C3")
    ax[1, 2].set_yscale("log"); ax[1, 2].set_xlabel("r [Mpc]")
    ax[1, 2].set_ylabel("nbar [1/Mpc^3]"); ax[1, 2].set_title("radial selection")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[ingest] saved {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
