#!/usr/bin/env python
"""Validate the REAL CF4 reconstruction against known local structures.

No ground truth exists for real data, so we check physical consistency:
  1. forward the recovered s -> z=0 matter density + velocity field (pmwd, matched cosmo)
  2. render the supergalactic SGX-SGY plane (our position at box centre); mark the known
     attractors (Virgo, Coma, Great Attractor, Perseus-Pisces). A correct reconstruction
     puts overdensities at their measured supergalactic positions.
  3. overlay the CF4 groups used in the fit (in the slab)
  4. measure the reconstructed BULK FLOW (mean peculiar velocity in R<50 Mpc/h) and
     compare magnitude/direction to the known CF4 value (~few hundred km/s).

Run on GPU:  sbatch src/cf4_job.slurm src/cf4_real_validate.py --pred recon/cf4_map_cf4_real192.npz
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import numpy as np

# Known attractors: (name, SGL[deg], SGB[deg], dist[Mpc])
ATTRACTORS = [
    ("Virgo", 102.9, -2.3, 16.0),
    ("Coma", 89.0, 8.0, 90.0),
    ("GreatAttr", 155.0, -8.0, 55.0),
    ("Perseus-Pisces", 340.0, -5.0, 65.0),
    ("Fornax", 236.0, -44.0, 19.0),
]


def sg_to_xyz(sgl, sgb, d):
    l = np.radians(sgl); b = np.radians(sgb)
    return d * np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mock_pipeline import VUNIT_KMS

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="cf4_map_*.npz (uses s_out = full-power CR)")
    ap.add_argument("--field", default="s_out", choices=["s_out", "s_map"])
    ap.add_argument("--real-npz", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "cf4_clean.npz"))
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--A-s-1e9", type=float, default=1.63, help="match the recon (1.63=sigma8~0.81)")
    ap.add_argument("--smooth", type=float, default=8.0, help="Gaussian smoothing [Mpc/h]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P = np.load(args.pred)
    s = P[args.field].astype(np.float64)
    N = int(P["N"]); sp = float(P["spacing"]); L = N * sp
    zc = np.load(args.real_npz); h = float(zc["H0"]) / 100.0
    print(f"[val] {args.pred} field={args.field} N={N} L={L:.0f} Mpc/h h={h:.3f}", flush=True)

    import jax, jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody, scatter
    conf = Configuration(ptcl_spacing=float(sp), ptcl_grid_shape=(N,) * 3,
                         mesh_shape=1, float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=args.Om, h=h, A_s_1e9=args.A_s_1e9), conf)
    lin = linear_modes(jnp.asarray(s.reshape(N, N, N)), cosmo, conf)
    ptcl, obs = lpt(lin, cosmo, conf); ptcl, obs = nbody(ptcl, obs, cosmo, conf)
    dens = np.asarray(scatter(ptcl, conf), np.float64) - 1.0
    vel = np.asarray(ptcl.vel, np.float64) * VUNIT_KMS
    pos = np.asarray(ptcl.pos(), np.float64)

    # bulk flow: mean velocity of particles within R<50 Mpc/h of centre
    c = L / 2.0
    rp = np.linalg.norm(pos - c, axis=1)
    for R in (30.0, 50.0, 80.0):
        m = rp < R
        bf = vel[m].mean(0)
        print(f"[bulk] R<{R:.0f} Mpc/h: |V|={np.linalg.norm(bf):.0f} km/s "
              f"dir SG=({bf[0]:.0f},{bf[1]:.0f},{bf[2]:.0f})", flush=True)

    # smoothed density for the slice (Gaussian in Fourier)
    kf = np.fft.fftfreq(N, d=sp) * 2 * np.pi; kr = np.fft.rfftfreq(N, d=sp) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    W = np.exp(-0.5 * (KX**2 + KY**2 + KZ**2) * args.smooth**2)
    dsm = np.fft.irfftn(np.fft.rfftn(dens) * W, s=(N, N, N), axes=(0, 1, 2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # supergalactic slice: SGZ ~ 0 slab, axes SGX (Mpc/h) vs SGY (Mpc/h), observer at centre
    cell = sp; kz0 = N // 2
    slab = dsm[:, :, max(kz0 - 2, 0):kz0 + 3].mean(2)
    ext = [-c, L - c, -c, L - c]                             # Mpc/h relative to observer
    fig, ax = plt.subplots(1, 2, figsize=(17, 8))
    im = ax[0].imshow(np.arcsinh(5 * slab).T, origin="lower", extent=ext, cmap="magma",
                      vmin=np.percentile(np.arcsinh(5*slab), 2),
                      vmax=np.percentile(np.arcsinh(5*slab), 99))
    # attractors (project to SGX-SGY, in Mpc/h)
    for name, sgl, sgb, d in ATTRACTORS:
        xyz = sg_to_xyz(sgl, sgb, d) * h                     # Mpc -> Mpc/h
        if abs(xyz[2]) < 40 and max(abs(xyz[0]), abs(xyz[1])) < c:
            ax[0].plot(xyz[0], xyz[1], "c*", ms=16, mec="w")
            ax[0].annotate(name, (xyz[0], xyz[1]), color="cyan", fontsize=10,
                           xytext=(6, 6), textcoords="offset points")
    # CF4 groups in the slab
    gp = zc["pos_dist"] * h                                  # Mpc/h, observer at origin
    sel = np.abs(gp[:, 2]) < 3 * cell
    ax[0].scatter(gp[sel, 0], gp[sel, 1], s=2, c="w", alpha=0.25, lw=0)
    ax[0].plot(0, 0, "+", color="lime", ms=14, mew=2)        # us
    ax[0].set_xlim(-120, 120); ax[0].set_ylim(-120, 120)
    ax[0].set_xlabel("SGX [Mpc/h]"); ax[0].set_ylabel("SGY [Mpc/h]")
    ax[0].set_title(f"CF4 reconstruction: z=0 density (SGZ~0, {args.smooth:.0f} Mpc/h smooth)\n"
                    "cyan*=known attractors, white=CF4 groups, green+=us")
    plt.colorbar(im, ax=ax[0], fraction=0.046, label="asinh(5 δ)")

    # power spectrum sanity
    def Pk(f):
        fk = np.fft.rfftn(f); km = np.sqrt(KX**2+KY**2+KZ**2)
        nb = N//2; e = np.linspace(0, km.max(), nb+1); idx = np.clip(np.digitize(km.ravel(), e)-1, 0, nb-1)
        p = np.array([(np.abs(fk)**2).ravel()[idx==b].mean() if (idx==b).any() else np.nan for b in range(nb)])
        kk = np.array([km.ravel()[idx==b].mean() if (idx==b).any() else np.nan for b in range(nb)])
        return kk, p
    kk, pk = Pk(dens)
    ax[1].loglog(kk, pk, "C0-")
    ax[1].set_xlabel("k [h/Mpc]"); ax[1].set_ylabel("P(k)")
    ax[1].set_title(f"reconstructed z=0 density P(k)\nstd(delta)={dens.std():.3f}")
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = args.out or args.pred.replace(".npz", "_val.png")
    fig.savefig(out, dpi=110)
    print(f"[val] saved {out}", flush=True)


if __name__ == "__main__":
    main()
