#!/usr/bin/env python
"""CF4 training-data pipeline: pmwd PM -> FoF halos -> HOD galaxies -> v_pec.

Realizes the methodology (user, 2026-07): generate supervised training pairs by
running the differentiable pmwd PM forward, finding halos with Friends-of-Friends,
populating them with an HOD, and assigning each mock galaxy a peculiar velocity.
The learned map is (background matter density field  <->  galaxy peculiar-velocity
field), so the trained network can recover the present matter density of the
local universe from CF4's sparse peculiar velocities.

One sample:
    s ~ N(0, I)                              whitened initial field
    lin = linear_modes(s)                    initial density
    ptcl = nbody(lpt(lin))                   z=0 particles (pos [Mpc/h], vel)
    delta_m = scatter(ptcl) - 1              present matter overdensity field
    halos = FoF(ptcl, b=0.2)                 dark-matter halos
    gal   = HOD(halos)                       mock galaxies (pos, 3D vel)
    v_pec = v_gal . r_hat  (observer=centre) line-of-sight peculiar velocity

pmwd units: length = Mpc/h; velocity_code * 100 = km/s (verified). H0=1 internal.

Output npz (one training pair, or a shard):
    s          (N,N,N)   whitened initial field  [network target for IC recovery]
    delta_m    (N,N,N)   present matter overdensity  [the field to be recovered]
    gal_pos    (Ng,3)    galaxy positions [Mpc/h]
    gal_vel    (Ng,3)    galaxy velocities [km/s]
    gal_vpec   (Ng,)     line-of-sight peculiar velocity [km/s]
    gal_mass   (Ng,)     host halo mass [Msun/h]
    gal_iscen  (Ng,)     central flag
    halo_mass  (Nh,)     halo mass function [Msun/h]
    L, N, spacing, m_particle, Om, hod_params

Run on GPU via Slurm:  sbatch src/cf4_job.slurm src/mock_pipeline.py [--N .. --spacing ..]
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import json
import time
import numpy as np

RHO_CRIT = 2.775e11   # Msun/h / (Mpc/h)^3
VUNIT_KMS = 100.0     # pmwd code velocity -> km/s


def make_forward(N, spacing, dtype, return_dens=True):
    import jax
    from pmwd import (Configuration, SimpleLCDM, boltzmann,
                      linear_modes, lpt, nbody, scatter)
    conf = Configuration(ptcl_spacing=float(spacing), ptcl_grid_shape=(N,) * 3,
                         mesh_shape=1, float_dtype=dtype)
    cosmo = boltzmann(SimpleLCDM(conf), conf)

    def forward(s):
        lin = linear_modes(s.reshape(N, N, N), cosmo, conf)
        ptcl, obs = lpt(lin, cosmo, conf)
        ptcl, obs = nbody(ptcl, obs, cosmo, conf)
        if return_dens:
            return scatter(ptcl, conf), ptcl   # dens mesh + particles
        return ptcl                            # particles only (saves the N^3 mesh in memory)

    return conf, cosmo, jax.jit(forward)


def run_sample(N, spacing, seed, Om, hod_params, b=0.2, n_min=20,
               vmode="particles", dtype=None, fwd=None, verbose=True):
    """fwd: optional prebuilt (conf, cosmo, forward) to avoid re-jitting per seed."""
    import jax
    import jax.numpy as jnp
    from fof import fof
    from hod import populate, line_of_sight_vpec, n_cen, n_sat
    if dtype is None:
        dtype = jnp.float32

    L = N * spacing
    m_p = Om * RHO_CRIT * spacing ** 3
    conf, cosmo, forward = fwd if fwd is not None else make_forward(N, spacing, dtype)

    t0 = time.time()
    s = jax.random.normal(jax.random.PRNGKey(int(seed)), (N ** 3,))
    dens, ptcl = forward(s)
    dens.block_until_ready()
    pos = np.asarray(ptcl.pos(), np.float64)              # [Mpc/h] in [0,L)
    vel = np.asarray(ptcl.vel, np.float64) * VUNIT_KMS     # [km/s]
    delta_m = np.asarray(dens, np.float32) - 1.0
    t_pm = time.time() - t0
    if verbose:
        print(f"[pipe] N={N} L={L:.0f} Mpc/h  Np={pos.shape[0]}  m_p={m_p:.2e} Msun/h "
              f"| PM {t_pm:.1f}s  1D vel rms={vel.std(0).mean():.0f} km/s")

    t0 = time.time()
    halos = fof(pos, vel, L=L, b=b, n_min=n_min, m_particle=m_p, verbose=verbose)
    t_fof = time.time() - t0

    t0 = time.time()
    gal = populate(halos, pos_all=pos, vel_all=vel, params=hod_params,
                   mode=vmode, Om=Om, seed=int(seed) * 7 + 1, verbose=verbose)
    t_hod = time.time() - t0

    observer = np.array([L / 2.0] * 3)
    vpec = line_of_sight_vpec(gal, observer) if gal["pos"].shape[0] else np.zeros(0)
    if verbose:
        nh = halos["mass"]
        print(f"[pipe] FoF {t_fof:.1f}s ({len(nh)} halos, "
              f"Mmax={nh.max():.2e}) | HOD {t_hod:.1f}s ({gal['pos'].shape[0]} gal) "
              f"| v_pec rms={np.std(vpec):.0f} km/s")

    return dict(
        s=np.asarray(s.reshape(N, N, N), np.float32),
        delta_m=delta_m,
        gal_pos=gal["pos"].astype(np.float32),
        gal_vel=gal["vel"].astype(np.float32),
        gal_vpec=vpec.astype(np.float32),
        gal_mass=gal["mass"].astype(np.float32),
        gal_iscen=gal["is_cen"],
        halo_mass=halos["mass"].astype(np.float32),
        halo_pos=halos["pos"].astype(np.float32),
        halo_vel=halos["vel"].astype(np.float32),
        L=np.float64(L), N=np.int64(N), spacing=np.float64(spacing),
        m_particle=np.float64(m_p), Om=np.float64(Om),
        hod_params=json.dumps(hod_params),
        seed=np.int64(seed),
    )


def diagnostic(rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hod import n_cen, n_sat, DEFAULT_HOD
    L = float(rec["L"]); N = int(rec["N"])
    dm = rec["delta_m"]
    gp = rec["gal_pos"]; gm = rec["gal_mass"]; iscen = rec["gal_iscen"]
    hm = rec["halo_mass"]
    p = json.loads(rec["hod_params"]) if isinstance(rec["hod_params"], str) else DEFAULT_HOD

    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    # (0,0) matter density slab + galaxies in a thin slice
    zc = N // 2
    slab = np.log10(2.0 + dm[:, :, max(zc - 2, 0):zc + 3].mean(axis=2))
    ax[0, 0].imshow(slab.T, origin="lower", extent=[0, L, 0, L], cmap="magma")
    cell = L / N
    sel = np.abs(gp[:, 2] - L / 2) < 3 * cell
    ax[0, 0].scatter(gp[sel, 0], gp[sel, 1], s=4, c="cyan", alpha=0.6, lw=0)
    ax[0, 0].set_title(f"matter density + galaxies (z-slice)\n{gp.shape[0]} gal total")
    ax[0, 0].set_xlabel("Mpc/h"); ax[0, 0].set_ylabel("Mpc/h")

    # (0,1) halo mass function
    mb = np.logspace(np.log10(max(hm.min(), 1e11)), np.log10(hm.max() + 1), 24)
    ax[0, 1].hist(hm, bins=mb, color="C0")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel("M_halo [Msun/h]"); ax[0, 1].set_ylabel("N")
    ax[0, 1].set_title(f"halo mass function ({len(hm)} halos)")

    # (0,2) HOD occupation: measured vs analytic
    bins = np.logspace(np.log10(hm.min()), np.log10(hm.max() + 1), 14)
    bc = np.sqrt(bins[:-1] * bins[1:])
    nhalo, _ = np.histogram(hm, bins=bins)
    ngal, _ = np.histogram(gm, bins=bins)
    meas = ngal / np.maximum(nhalo, 1)
    ana = n_cen(bc, p["logMmin"], p["sigma_logM"]) + \
        n_sat(bc, p["logMmin"], p["sigma_logM"], p["logM0"], p["logM1"], p["alpha"])
    ax[0, 2].loglog(bc, np.maximum(meas, 1e-3), "o-", label="measured")
    ax[0, 2].loglog(bc, np.maximum(ana, 1e-3), "k--", label="Zheng07")
    ax[0, 2].set_xlabel("M_halo [Msun/h]"); ax[0, 2].set_ylabel("<N_gal>")
    ax[0, 2].set_title("HOD occupation"); ax[0, 2].legend()

    # (1,0) peculiar velocity distribution
    ax[1, 0].hist(rec["gal_vpec"], bins=60, color="C1")
    ax[1, 0].set_xlabel("v_pec,los [km/s]")
    ax[1, 0].set_title(f"galaxy peculiar velocity (rms {np.std(rec['gal_vpec']):.0f})")

    # (1,1) v_pec vs radius (bulk flow / infall structure)
    r = np.linalg.norm(gp - L / 2, axis=1)
    ax[1, 1].scatter(r, rec["gal_vpec"], s=3, alpha=0.3, c=np.log10(gm), cmap="viridis")
    ax[1, 1].set_xlabel("r from centre [Mpc/h]"); ax[1, 1].set_ylabel("v_pec [km/s]")
    ax[1, 1].set_title("v_pec vs distance (color=logM)")

    # (1,2) central vs satellite counts and f_sat
    ncen = int(iscen.sum()); nsat = int((~iscen).sum())
    ax[1, 2].bar(["central", "satellite"], [ncen, nsat], color=["C2", "C3"])
    ax[1, 2].set_title(f"f_sat = {nsat/max(ncen+nsat,1):.3f}")
    ax[1, 2].set_ylabel("count")

    fig.suptitle(f"CF4 mock pipeline: pmwd -> FoF -> HOD -> v_pec  "
                 f"(N={N}, L={L:.0f} Mpc/h, m_p={float(rec['m_particle']):.1e})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    print(f"[pipe] saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--spacing", type=float, default=1.0, help="Mpc/h per particle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--b", type=float, default=0.2, help="FoF linking length / mean_sep")
    ap.add_argument("--n-min", type=int, default=20, help="min FoF members")
    ap.add_argument("--vmode", choices=["particles", "nfw"], default="particles")
    ap.add_argument("--logMmin", type=float, default=12.6)
    ap.add_argument("--sigma-logM", type=float, default=0.4)
    ap.add_argument("--logM0", type=float, default=12.0)
    ap.add_argument("--logM1", type=float, default=13.7)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "mock_pipeline.npz"))
    ap.add_argument("--fig", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "mock_pipeline.png"))
    args = ap.parse_args()

    import jax
    print(f"[env] jax {jax.__version__} dev={jax.devices()[0]}")
    hod_params = dict(logMmin=args.logMmin, sigma_logM=args.sigma_logM,
                      logM0=args.logM0, logM1=args.logM1, alpha=args.alpha)
    rec = run_sample(args.N, args.spacing, args.seed, args.Om, hod_params,
                     b=args.b, n_min=args.n_min, vmode=args.vmode)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **rec)
    print(f"[pipe] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    diagnostic(rec, args.fig)


if __name__ == "__main__":
    main()
