#!/usr/bin/env python
"""Lagrangian zoom masks for lagRAMSES constrained runs (MW/M31/Virgo/Coma).

Given the reconstructed CF4 initial field, forward it to z=0 with pmwd, find the
particles that collapse into a target structure (Local Group at the box centre = our
position; Virgo; Coma), and trace them back to their INITIAL (Lagrangian) positions.
That Lagrangian region is the high-resolution patch lagRAMSES must refine.

Outputs per target:
  * lagrangian point cloud (initial positions of the target's z=0 particles)  [.npz]
  * refinement region summary (centre + bounding box + ellipsoid axes, Mpc/h and in
    box-fraction 0..1 as RAMSES namelist wants) -- printed + saved in the npz
  * figure: z=0 target selection (SG slice) + its Lagrangian region (initial slice)

The Lagrangian region SHAPE is set by the constrained large-scale collapse; lagRAMSES
re-simulates that patch at high resolution (adding small-scale power) -> MW/M31/M33/LMC,
cluster galaxies as correct-environment analogs (tiered plan, see cf4-target-resolution).

Run on GPU:  sbatch src/cf4_job.slurm src/cf4_zoom_mask.py --pred recon/cf4_map_cf4_real192.npz
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import numpy as np

# targets in supergalactic (SGL,SGB deg, dist Mpc) + selection radius [Mpc/h]
TARGETS = {
    "LocalGroup": (None, None, 0.0, 4.0),          # box centre = our position
    "Virgo": (102.9, -2.3, 16.0, 5.0),
    "Coma": (89.0, 8.0, 90.0, 8.0),
}


def sg_to_xyz(sgl, sgb, d):
    l = np.radians(sgl); b = np.radians(sgb)
    return d * np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="cf4_map_*.npz (uses s_out full-power IC)")
    ap.add_argument("--field", default="s_out")
    ap.add_argument("--real-npz", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "cf4_clean.npz"))
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--A-s-1e9", type=float, default=1.63)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS.keys()))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "zoom"))
    args = ap.parse_args()

    P = np.load(args.pred); s = P[args.field].astype(np.float64)
    N = int(P["N"]); sp = float(P["spacing"]); L = N * sp
    h = float(np.load(args.real_npz)["H0"]) / 100.0
    os.makedirs(args.out, exist_ok=True)
    print(f"[zoom] {args.pred} field={args.field} N={N} L={L:.0f} Mpc/h h={h:.3f}", flush=True)

    import jax, jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody
    conf = Configuration(ptcl_spacing=float(sp), ptcl_grid_shape=(N,) * 3,
                         mesh_shape=1, float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=args.Om, h=h, A_s_1e9=args.A_s_1e9), conf)
    lin = linear_modes(jnp.asarray(s.reshape(N, N, N)), cosmo, conf)
    ptcl, obs = lpt(lin, cosmo, conf); ptcl, obs = nbody(ptcl, obs, cosmo, conf)
    z0 = np.asarray(ptcl.pos(), np.float64)                 # (Np,3) final [Mpc/h] in [0,L)
    # Lagrangian positions: regular grid, C-order matching pmwd particle order
    lag = (np.indices((N, N, N)).reshape(3, -1).T.astype(np.float64)) * sp
    c = L / 2.0                                             # observer at box centre

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {}
    for name in args.targets:
        sgl, sgb, dist, R = TARGETS[name]
        Xt = c + (np.zeros(3) if sgl is None else sg_to_xyz(sgl, sgb, dist) * h)  # box frame
        # z=0 particles within R of the target (min-image for periodicity)
        dv = z0 - Xt
        dv -= L * np.round(dv / L)
        sel = np.linalg.norm(dv, axis=1) < R
        nsel = int(sel.sum())
        if nsel == 0:
            print(f"[zoom] {name}: 0 particles in R<{R} Mpc/h at {Xt-c} — skip", flush=True)
            continue
        lag_sel = lag[sel]
        # Lagrangian region descriptors (unwrap around its own mean for periodicity)
        lmean = lag_sel.mean(0)
        lu = lag_sel - L * np.round((lag_sel - lmean) / L)
        lcen = lu.mean(0); lu -= lcen; lcen = lcen % L
        cov = np.cov(lu.T); evals, evecs = np.linalg.eigh(cov)
        axes = 2.0 * np.sqrt(np.maximum(evals, 0))          # ~2-sigma semi-axes [Mpc/h]
        bbox = (lu.min(0), lu.max(0))                       # relative to lcen
        edge_margin = min(lcen.min(), (L - lcen).min())     # dist of region centre to box edge
        print(f"[zoom] {name}: {nsel} ptcl | z0 centre SG={Xt-c} | "
              f"Lagrangian centre(box)={lcen} | ellipsoid semi-axes={axes} Mpc/h | "
              f"bbox={bbox[1]-bbox[0]} Mpc/h | centre-to-edge={edge_margin:.0f} Mpc/h", flush=True)
        # save mask (Lagrangian points + region params, incl. box-fraction for RAMSES)
        outf = os.path.join(args.out, f"zoom_{name}.npz")
        np.savez(outf, lagrangian=lag_sel.astype(np.float32),
                 z0_centre_box=Xt, z0_centre_sg=Xt - c,
                 lag_centre_box=lcen, ellipsoid_semiaxes=axes, ellipsoid_evecs=evecs,
                 bbox_lo=lcen + bbox[0], bbox_hi=lcen + bbox[1],
                 centre_frac=lcen / L, semiaxes_frac=axes / L,
                 N=N, spacing=sp, L=L, radius=R, target=name)
        summary[name] = dict(nsel=nsel, lcen=lcen, axes=axes, margin=edge_margin, Xt=Xt)

        # figure: z0 selection + Lagrangian region
        fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
        # z=0 slab around target
        zt = Xt[2]
        m0 = np.abs(((z0[:, 2] - zt + L/2) % L) - L/2) < 6
        ax[0].scatter(z0[m0, 0] - c, z0[m0, 1] - c, s=1, c="0.6", alpha=0.3, lw=0)
        ax[0].scatter(z0[sel, 0] - c, z0[sel, 1] - c, s=3, c="crimson", lw=0)
        ax[0].plot(Xt[0]-c, Xt[1]-c, "b*", ms=14)
        ax[0].set_title(f"{name}: z=0 selection (R<{R} Mpc/h, {nsel} ptcl)")
        ax[0].set_xlabel("SGX [Mpc/h]"); ax[0].set_ylabel("SGY [Mpc/h]")
        ax[0].set_xlim(Xt[0]-c-40, Xt[0]-c+40); ax[0].set_ylim(Xt[1]-c-40, Xt[1]-c+40)
        # Lagrangian region
        ax[1].scatter(lag_sel[:, 0], lag_sel[:, 1], s=3, c="teal", lw=0)
        ax[1].plot(lcen[0], lcen[1], "r+", ms=12, mew=2)
        ax[1].set_title(f"{name}: Lagrangian region (initial)\nsemi-axes {axes.round(1)} Mpc/h")
        ax[1].set_xlabel("Lag X [Mpc/h]"); ax[1].set_ylabel("Lag Y [Mpc/h]")
        ax[1].set_aspect("equal")
        fig.tight_layout(); figp = os.path.join(args.out, f"zoom_{name}.png")
        fig.savefig(figp, dpi=110); plt.close(fig)
        print(f"[zoom] saved {outf} + {figp}", flush=True)

    print("\n[zoom] === RAMSES refinement regions (box-fraction, for namelist) ===", flush=True)
    for name, d in summary.items():
        cf = d["lcen"] / L; af = d["axes"] / L
        print(f"  {name}: xc,yc,zc = {cf[0]:.4f},{cf[1]:.4f},{cf[2]:.4f}  "
              f"rx,ry,rz = {af[0]:.4f},{af[1]:.4f},{af[2]:.4f}  "
              f"(margin {d['margin']:.0f} Mpc/h)", flush=True)
    print(f"[zoom] done: {list(summary.keys())}", flush=True)


if __name__ == "__main__":
    main()
