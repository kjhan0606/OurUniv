#!/usr/bin/env python
"""Closed-loop test of the constrained fine IC: forward it and compare mock galaxies to CF4.

Take the constrained fine initial conditions (cf4_make_ic.py), run the pmwd forward, find
halos with Friends-of-Friends, populate them with an HOD, and assign line-of-sight peculiar
velocities (observer at the box centre). Then compare the mock galaxies to the real CF4:
  - do the mock galaxies cluster where the CF4 groups are (density cross-correlation)?
  - do the known clusters (Virgo/Coma/Great Attractor) appear at their positions?
  - is the mock bulk flow consistent with the observed one?
This is the posterior-predictive check. The IC was constrained by the CF4 velocities, so the
forward-evolved mock should reproduce the observed structure and flow.

Run on GPU:  sbatch src/cf4_job.slurm src/cf4_ic_test.py --ic recon/cf4_ic_fine.npz
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "0")   # float32 forward -> half the memory (needed at 768^3)
import argparse
import numpy as np


CLUSTERS = {"Virgo": (102.9, -2.3, 16.0), "Coma": (89.0, 8.0, 90.0),
            "Centaurus": (156.0, -11.0, 45.0), "Norma/GA": (155.0, -6.0, 65.0),
            "Perseus": (340.0, -13.0, 73.0), "Fornax": (236.0, -44.0, 19.0)}


def sg(sgl, sgb, d):
    l, b = np.radians(sgl), np.radians(sgb)
    return d * np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mock_pipeline import make_forward, RHO_CRIT, VUNIT_KMS
    from fof import fof
    from hod import populate, line_of_sight_vpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "cf4_ic_fine.npz"))
    ap.add_argument("--real-npz", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "cf4_clean.npz"))
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--b", type=float, default=0.2)
    ap.add_argument("--n-min", type=int, default=20)
    ap.add_argument("--fof-region", type=float, default=160.0,
                    help="central cube half-extent*2 [h^-1Mpc] to FoF; full 768^3 is too big")
    ap.add_argument("--fof-buffer", type=float, default=8.0,
                    help="buffer beyond the region so edge halos are complete [h^-1Mpc]")
    ap.add_argument("--fof", choices=["opfof", "scipy"], default="opfof",
                    help="opfof: Juhan Kim's MPI FoF on the full box; scipy: central-cube fallback")
    ap.add_argument("--fof-nid", type=int, default=16, help="OPFoF MPI ranks")
    ap.add_argument("--fof-nfile", type=int, default=32, help="OPFoF z-slab files (multiple of nid)")
    ap.add_argument("--logMmin", type=float, default=12.3)
    ap.add_argument("--sigma-logM", type=float, default=0.35)
    ap.add_argument("--logM0", type=float, default=11.8)
    ap.add_argument("--logM1", type=float, default=13.2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "cf4_ic_test"))
    # two-stage split: the 768^3 pmwd forward needs a big-memory GPU (h200), but OPFoF wants
    # a big-RAM CPU node (lageunha). --forward-only dumps particles; --from-ptcl reloads them.
    ap.add_argument("--dump-ptcl", default=None, help="save pmwd particles to this .npz")
    ap.add_argument("--forward-only", action="store_true", help="stop after the forward+dump")
    ap.add_argument("--from-ptcl", default=None, help="load particles from a --dump-ptcl file, skip forward")
    args = ap.parse_args()

    import time
    z = np.load(args.ic)
    N = int(z["Nfine"]); sp = float(z["spacing"]); L = N * sp; h = float(z["h"])
    m_p = args.Om * RHO_CRIT * sp ** 3
    print(f"[test] fine IC {N}^3 @ {sp:.3f} h^-1Mpc (L={L:.0f}) m_p={m_p:.2e} Msun/h", flush=True)

    if args.from_ptcl:
        zp = np.load(args.from_ptcl)
        pos = zp["pos"].astype(np.float64); vel = zp["vel"].astype(np.float64)
        dens_cube = zp["dens_cube"]
        print(f"[test] loaded {pos.shape[0]:.0f} particles from {args.from_ptcl}", flush=True)
    else:
        import jax, jax.numpy as jnp
        s = z["s_fine"].astype(np.float32)
        t0 = time.time()
        conf, cosmo, forward = make_forward(N, sp, jnp.float32)
        dens, ptcl = forward(jnp.asarray(s.reshape(N, N, N))); dens.block_until_ready()
        pos = np.asarray(ptcl.pos(), np.float64); vel = np.asarray(ptcl.vel, np.float64) * VUNIT_KMS
        # observer-centred present-density cube (+-35 h^-1Mpc) for the Hong et al. all-sky comparison
        ci = N // 2; ncu = int(round(35.0 / sp))
        dens_cube = np.asarray(dens[ci-ncu:ci+ncu, ci-ncu:ci+ncu, ci-ncu:ci+ncu], np.float32)
        print(f"[test] pmwd forward {pos.shape[0]:.0f} ptcl in {time.time()-t0:.0f}s, "
              f"1D vel rms {vel.std(0).mean():.0f} km/s", flush=True)
        if args.dump_ptcl:
            np.savez(args.dump_ptcl, pos=pos.astype(np.float32), vel=vel.astype(np.float32),
                     dens_cube=dens_cube, N=N, spacing=sp, L=L, h=h)
            print(f"[test] dumped particles -> {args.dump_ptcl} "
                  f"({os.path.getsize(args.dump_ptcl)/1e9:.1f} GB)", flush=True)
        if args.forward_only:
            return

    t0 = time.time()

    t0 = time.time()
    hod = dict(logMmin=args.logMmin, sigma_logM=args.sigma_logM, logM0=args.logM0,
               logM1=args.logM1, alpha=args.alpha)
    c = L / 2.0
    observer = np.array([c] * 3)

    def scipy_subregion():
        # scipy pair search cannot handle the full 768^3 (4.5e8), so FoF the central cube
        # (CF4-constrained volume) non-periodically; mean_sep stays the global spacing sp.
        half = args.fof_region / 2.0 + args.fof_buffer
        m = np.all(np.abs(pos - c) < half, axis=1)
        print(f"[test] scipy FoF central cube {2*half:.0f} h^-1Mpc: {int(m.sum())} of "
              f"{pos.shape[0]:.0f} particles", flush=True)
        h_ = fof(pos[m], vel[m], L=L, mean_sep=sp, b=args.b, n_min=args.n_min,
                 m_particle=m_p, periodic=False, verbose=True)
        return h_, pos[m], vel[m]

    if args.fof == "opfof":
        try:
            import opfof_io as oi
            print(f"[test] OPFoF (MPI) full box: {pos.shape[0]:.0f} particles, "
                  f"{args.fof_nfile} z-slabs, {args.fof_nid} ranks", flush=True)
            halos = oi.fof_opfof(pos, vel, L=L, nx=N, nstep=1, nfile=args.fof_nfile,
                                 nid=args.fof_nid, outdir=os.path.join(args.out, "opfof_work"),
                                 m_particle=m_p, nmin=args.n_min, verbose=True)
            pos_h, vel_h = pos, vel
        except Exception as e:
            print(f"[test] OPFoF failed ({e}); falling back to scipy central-cube FoF", flush=True)
            halos, pos_h, vel_h = scipy_subregion()
    else:
        halos, pos_h, vel_h = scipy_subregion()
    gal = populate(halos, pos_all=pos_h, vel_all=vel_h, params=hod, mode="particles",
                   Om=args.Om, seed=7, verbose=True)
    gp = gal["pos"]; vpec = line_of_sight_vpec(gal, observer) if gp.shape[0] else np.zeros(0)
    print(f"[test] FoF {len(halos['mass'])} halos (Mmax {halos['mass'].max():.1e}) | "
          f"HOD {gp.shape[0]} mock galaxies | v_pec rms {np.std(vpec):.0f} km/s "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # --- compare to observed CF4 ---
    zc = np.load(args.real_npz)
    cf4 = zc["pos_dist"].astype(np.float64) * h + c              # box frame
    # density cross-correlation on a 4 h^-1Mpc grid (smoothed)
    from scipy.ndimage import gaussian_filter
    Ng = 96; cell = L / Ng
    def grid(p):
        idx = np.floor((p % L) / cell).astype(int) % Ng
        f = np.zeros((Ng, Ng, Ng), np.float32); np.add.at(f, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        f = gaussian_filter(f, 1.5); return f / f.mean() - 1.0
    dm = grid(gp); dc = grid(cf4)
    # restrict to the CF4 sphere (r<180 h^-1Mpc from centre) for a fair comparison
    ax = (np.arange(Ng) + 0.5) * cell - c
    Rcmp = args.fof_region / 2.0
    RX, RY, RZ = np.meshgrid(ax, ax, ax, indexing="ij"); msk = np.sqrt(RX**2+RY**2+RZ**2) < Rcmp
    r = np.corrcoef(dm[msk], dc[msk])[0, 1]
    print(f"[test] mock-vs-CF4 galaxy density cross-corr (r<{Rcmp:.0f}, 4 h^-1Mpc smooth) = {r:.3f}", flush=True)
    # bulk flow of the mock
    rp = np.linalg.norm(gp - c, axis=1)
    for R in (30, 50, 80):
        mm = rp < R
        if mm.sum():
            bf = gal["vel"][mm].mean(0)
            print(f"[test] mock bulk flow R<{R}: |V|={np.linalg.norm(bf):.0f} km/s "
                  f"SG=({bf[0]:.0f},{bf[1]:.0f},{bf[2]:.0f})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "mock_catalog.npz"),
             pos=gp.astype(np.float32), vel=gal["vel"].astype(np.float32),
             vpec=vpec.astype(np.float32), mass=gal["mass"].astype(np.float32),
             dens_cube=dens_cube, dens_cube_spacing=sp, L=L, h=h, observer=observer)

    # figure: mock galaxies vs CF4 groups, SG slice
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax2 = plt.subplots(1, 2, figsize=(15, 7.2))
    for a, (p, ttl, col) in zip(ax2, [(gp, "mock galaxies (from constrained IC)", "C1"),
                                       (cf4, "observed CF4 groups", "C0")]):
        ps = p - c; sel = np.abs(ps[:, 2]) < 8
        a.scatter(ps[sel, 0], ps[sel, 1], s=3, c=col, alpha=0.4, lw=0)
        for nm, (sgl, sgb, dd) in CLUSTERS.items():
            x = sg(sgl, sgb, dd) * h
            if abs(x[2]) < 30 and max(abs(x[0]), abs(x[1])) < 150:
                a.plot(x[0], x[1], "k*", ms=11); a.annotate(nm, (x[0], x[1]), fontsize=8)
        a.plot(0, 0, "+", color="lime", ms=12, mew=2)
        lim = args.fof_region / 2.0 + 10
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_title(ttl)
        a.set_xlabel("SGX [$h^{-1}$Mpc]"); a.set_ylabel("SGY [$h^{-1}$Mpc]")
    fig.suptitle(f"Closed-loop test: constrained IC -> pmwd -> FoF -> HOD mock galaxies "
                 f"vs CF4 (density corr {r:.2f})", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "cf4_ic_test.png"), dpi=110)
    print(f"[test] saved {os.path.join(args.out, 'cf4_ic_test.png')}", flush=True)


if __name__ == "__main__":
    main()
