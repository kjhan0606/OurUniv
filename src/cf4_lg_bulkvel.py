#!/usr/bin/env python
"""Gate check (fable): the LG bulk velocity at the observer vs the 620 km/s CMB dipole.

CF4 constrains peculiar velocities, so the observer-region bulk flow (v_LG) SHOULD reproduce the
LG's CMB-dipole motion (~620 km/s toward the apex). If a candidate's observer isn't moving ~600
km/s toward the apex, that realization is suspect and shouldn't get a flagship zoom. One 576^3
forward per candidate.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CMB dipole apex for the Local Group (Kogut+1993 / Planck): galactic (l,b)=(276.4, 29.3), 620 km/s
CMB_GAL = (276.4, 29.3)
CMB_V = 620.0


def gal_to_sg(l, b):
    """Galactic (deg) -> supergalactic unit vector, via astropy if present else a fixed matrix."""
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").supergalactic
        sgl, sgb = c.sgl.deg, c.sgb.deg
    except Exception:
        # de Vaucouleurs SG pole gal (47.37,+6.32), SGL0 at gal(137.37,0). Build rotation.
        def uvec(lo, bo):
            lo, bo = np.radians(lo), np.radians(bo)
            return np.array([np.cos(bo)*np.cos(lo), np.cos(bo)*np.sin(lo), np.sin(bo)])
        zsg = uvec(47.37, 6.32); x0 = uvec(137.37, 0.0)
        ysg = np.cross(zsg, x0); ysg /= np.linalg.norm(ysg); xsg = np.cross(ysg, zsg)
        v = uvec(l, b); sgl = np.degrees(np.arctan2(v@ysg, v@xsg)) % 360; sgb = np.degrees(np.arcsin(v@zsg))
    l_, b_ = np.radians(sgl), np.radians(sgb)
    return np.array([np.cos(b_)*np.cos(l_), np.cos(b_)*np.sin(l_), np.sin(b_)]), sgl, sgb


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cr6.npz")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--label", default="cr6_e19")
    args = ap.parse_args()
    from mock_pipeline import make_forward, VUNIT_KMS
    from cf4_make_ic import embed_ic
    import jax.numpy as jnp

    z = np.load(args.recon)
    s = z["s_out"].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    L = Nc*spc; c = L/2.0; sp = L/args.Nfine
    cmb_sg, sgl, sgb = gal_to_sg(*CMB_GAL)

    s_fine = embed_ic(s, args.Nfine, args.seed)
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    vel = np.asarray(ptcl.vel).astype(np.float64) * VUNIT_KMS

    print(f"[bulk] {args.label}: CMB apex SG=({sgl:.0f},{sgb:.0f}) expect |v|~{CMB_V:.0f} km/s", flush=True)
    for R in (2.5, 5.0, 8.0):
        sel = np.linalg.norm(pos - c, axis=1) < R
        v = vel[sel].mean(0); mag = np.linalg.norm(v)
        cosang = float(np.dot(v/mag, cmb_sg)); ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        vdir = v/mag
        print(f"[bulk] R={R:>4} Mpc/h  N={sel.sum():>8}  |v_LG|={mag:>6.0f} km/s  "
              f"angle-to-CMB-apex={ang:>5.1f} deg  (SGdir=[{vdir[0]:+.2f},{vdir[1]:+.2f},{vdir[2]:+.2f}])",
              flush=True)


if __name__ == "__main__":
    main()
