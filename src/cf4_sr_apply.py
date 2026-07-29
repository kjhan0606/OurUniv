#!/usr/bin/env python
"""Apply the TNG-trained position-conditioned super-resolution to the real CF4 reconstruction.

Produces the super-resolved galaxy field over the local volume. For each 32 h^-1Mpc sub-cube
of a central region we feed the model two conditioning fields.
  coarse   the CF4-reconstructed matter overdensity at 2 h^-1Mpc (the velocity-constrained
           large-scale field, forward-evolved from the recovered initial conditions)
  n_obs    the real CF4 groups in that region gridded on the fine grid (the observed peaks)
The model generates the fine (0.5 h^-1Mpc) galaxy field. The observed groups pin the biased
peaks where CF4 has data and the field is TNG-statistical elsewhere. We tile and stitch the
sub-cubes into the region and render the supergalactic slice with the known clusters.

Both pmwd (float32) and the diffusion net run in single precision, so one process suffices.
Run on GPU:  sbatch src/cf4_job.slurm src/cf4_sr_apply.py --region 128
"""
import os
os.environ["JAX_ENABLE_X64"] = "0"
import argparse, pickle
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf4_train_diffusion as D
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERS = {"Virgo": (102.9, -2.3, 16.0), "Coma": (89.0, 8.0, 90.0),
            "Centaurus": (156.0, -11.0, 45.0), "Norma/GA": (155.0, -6.0, 65.0),
            "Perseus": (340.0, -13.0, 73.0), "Fornax": (236.0, -44.0, 19.0)}


def sg(sgl, sgb, d):
    l, b = np.radians(sgl), np.radians(sgb)
    return d * np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default=os.path.join(ROOT, "recon", "cf4_map_cf4_real192.npz"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "recon", "cf4_sr_tng.eqx"))
    ap.add_argument("--real-npz", default=os.path.join(ROOT, "data", "cf4_clean.npz"))
    ap.add_argument("--region", type=float, default=128.0, help="central cube side [h^-1Mpc]")
    ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--smooth-obs", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "recon", "cf4_sr_local"))
    args = ap.parse_args()

    with open(args.ckpt, "rb") as fh:
        m = pickle.load(fh); sub = m["sub"]; cf = m["cf"]; T = m["T"]
        model = D.UNet3D(2, C=m["C"], key=jax.random.PRNGKey(0))
        model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x)
                                       and jnp.issubdtype(x.dtype, jnp.floating) else x, model)
        model = eqx.tree_deserialise_leaves(fh, model)
        # force float32 AFTER load (deserialize can reintroduce float64)
        model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x)
                                       and jnp.issubdtype(x.dtype, jnp.floating) else x, model)
    betas, alphas, abars = D.make_schedule(T)
    fine = 0.5; csize = fine * cf                                   # coarse cell = 2 h^-1Mpc
    subL = sub * fine                                              # 32 h^-1Mpc
    print(f"[apply] SR sub={sub} coarse-factor={cf} sub={subL:.0f} h^-1Mpc region={args.region:.0f}",
          flush=True)

    # --- coarse: CF4-reconstructed z=0 matter overdensity at 2 h^-1Mpc ---
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody, scatter
    z = np.load(args.recon); s = z["s_out"].astype(np.float32); N = int(z["N"]); spc = float(z["spacing"])
    L = N * spc
    zc = np.load(args.real_npz); h = float(zc["H0"]) / 100.0
    conf = Configuration(ptcl_spacing=spc, ptcl_grid_shape=(N,)*3, mesh_shape=1, float_dtype=jnp.float32)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=0.31, h=h, A_s_1e9=1.63), conf)
    lin = linear_modes(jnp.asarray(s.reshape(N, N, N)), cosmo, conf)
    p, o = lpt(lin, cosmo, conf); p, o = nbody(p, o, cosmo, conf)
    delta_c = np.asarray(scatter(p, conf), np.float32) - 1.0        # (192,192,192) 2 h^-1Mpc
    assert abs(spc - csize) < 1e-6, f"recon cell {spc} != SR coarse {csize}"

    # --- observed CF4 groups in the box frame ---
    c = L / 2.0
    gal = zc["pos_dist"].astype(np.float64) * h + c                # box frame [h^-1Mpc]

    # --- central region tiling with 50% OVERLAP for seam-free feather blending ---
    r_c0 = int(round((c - args.region / 2) / csize))              # coarse index of region start
    nsub = int(round(args.region / subL))
    Nf = nsub * sub                                               # fine cells across region
    region_lo = r_c0 * csize
    stride = sub // 2                                             # overlap by half a sub-cube
    fine_offs = list(range(0, Nf - sub + 1, stride))
    cs = sub // cf                                                # coarse cells per sub-cube
    # Hann window (tapers to ~0 at sub-cube edges) so overlapping tiles blend smoothly
    w1 = np.maximum(np.hanning(sub + 2)[1:-1], 1e-3).astype(np.float32)
    W = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).astype(np.float32)
    print(f"[apply] region [{region_lo-c:.0f},{region_lo-c+args.region:.0f}] h^-1Mpc, fine {Nf}^3, "
          f"{len(fine_offs)}^3={len(fine_offs)**3} overlapping sub-cubes (stride {stride})", flush=True)

    conds = []; offs = []
    for fx in fine_offs:
        for fy in fine_offs:
            for fz in fine_offs:
                cc = r_c0 + np.array([fx, fy, fz]) // cf          # coarse index of this sub-cube
                blk = delta_c[cc[0]:cc[0]+cs, cc[1]:cc[1]+cs, cc[2]:cc[2]+cs]
                rho = 1.0 + blk; dc = rho / max(rho.mean(), 1e-8) - 1.0   # local overdensity
                dcu = np.repeat(np.repeat(np.repeat(dc, cf, 0), cf, 1), cf, 2)  # -> sub^3
                lo = region_lo + np.array([fx, fy, fz]) * fine
                rel = gal - lo
                mm = np.all((rel >= 0) & (rel < subL), axis=1)
                idx = np.floor(rel[mm] / fine).astype(int)
                nog = np.zeros((sub, sub, sub), np.float32)
                if idx.shape[0]:
                    np.add.at(nog, (idx[:, 0] % sub, idx[:, 1] % sub, idx[:, 2] % sub), 1.0)
                    nog = gaussian_filter(nog, args.smooth_obs, mode="constant")
                    nbar = nog.mean(); nog = nog / nbar - 1.0 if nbar > 0 else nog
                cond = np.stack([dcu, nog.astype(np.float32)], 0)
                cond = (cond - m["cmu"][:, None, None, None]) / m["csd"][:, None, None, None]
                cond = np.clip(np.nan_to_num(cond, nan=0.0), -8.0, 8.0)   # keep in-distribution
                conds.append(cond); offs.append((fx, fy, fz))
    conds = np.array(conds, np.float32)                        # pmwd may enable x64; force f32
    print(f"[apply] {conds.shape[0]} overlapping sub-cubes to super-resolve", flush=True)

    # DDIM sampling (batched)
    abars32 = jnp.asarray(np.asarray(abars, np.float32))       # force float32 (pmwd x64 guard)
    @eqx.filter_jit
    def ddim(model, S, cnd, t, tp, key):
        eps = jax.vmap(lambda s, c: model(s, c, t.astype(jnp.float32) / T))(S, cnd)
        ab = abars32[t]; abp = jnp.where(tp >= 0, abars32[jnp.maximum(tp, 0)], jnp.float32(1.0))
        s0 = (S - jnp.sqrt(1 - ab) * eps) / jnp.sqrt(ab)
        return jnp.sqrt(abp) * s0 + jnp.sqrt(jnp.maximum(1 - abp, 0)) * eps
    ts = np.linspace(T - 1, 0, args.ddim_steps).round().astype(int)
    seq = [(int(ts[i]), int(ts[i+1]) if i+1 < len(ts) else -1) for i in range(len(ts))]
    fine_grid = np.zeros((Nf, Nf, Nf), np.float32)                # feather-weighted accumulator
    wsum = np.zeros((Nf, Nf, Nf), np.float32)
    key = jax.random.PRNGKey(7); B = 8
    import time; t0 = time.time()
    for b in range(0, conds.shape[0], B):
        cb = jnp.asarray(conds[b:b+B], jnp.float32); nb = cb.shape[0]
        key, k0 = jax.random.split(key)
        S = jax.random.normal(k0, (nb, sub, sub, sub), dtype=jnp.float32)
        for (t, tp) in seq:
            key, kt = jax.random.split(key)
            S = ddim(model, S, cb, jnp.array(t), jnp.array(tp), kt)
        gen = np.nan_to_num(np.asarray(S) * m["ysd"] + m["ymu"], nan=0.0)   # asinh(overdensity)
        for j in range(nb):
            fx, fy, fz = offs[b+j]
            fine_grid[fx:fx+sub, fy:fy+sub, fz:fz+sub] += W * gen[j]
            wsum[fx:fx+sub, fy:fy+sub, fz:fz+sub] += W
        print(f"[apply] {b+nb}/{conds.shape[0]} [{time.time()-t0:.0f}s]", flush=True)
    fine_grid = fine_grid / np.maximum(wsum, 1e-6)                # feather blend -> seam-free

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "fine_field.npy"), fine_grid)
    print(f"[apply] saved fine_field.npy {fine_grid.shape} (asinh overdensity, {fine} h^-1Mpc)", flush=True)

    # figure: SG slice of the super-resolved field + CF4 galaxies + clusters
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    kf = Nf // 2; slab = np.sinh(fine_grid[:, :, kf-2:kf+3].mean(2)).T
    ext = [region_lo - c, region_lo - c + args.region] * 2
    fig, ax = plt.subplots(figsize=(9, 8.5))
    im = ax.imshow(np.arcsinh(3*slab), origin="lower", extent=ext, cmap="magma")
    galsg = gal - c; sel = np.abs(galsg[:, 2]) < 3
    ax.scatter(galsg[sel, 0], galsg[sel, 1], s=3, c="cyan", alpha=0.4, lw=0, label="CF4 groups")
    for nm, (sgl, sgb, dd) in CLUSTERS.items():
        x = sg(sgl, sgb, dd) * h
        if abs(x[2]) < 20 and abs(x[0]) < args.region/2 and abs(x[1]) < args.region/2:
            ax.plot(x[0], x[1], "w*", ms=13, mec="k"); ax.annotate(nm, (x[0], x[1]), color="w", fontsize=9)
    ax.plot(0, 0, "+", color="lime", ms=12, mew=2)
    ax.set_xlabel("SGX [$h^{-1}$Mpc]"); ax.set_ylabel("SGY [$h^{-1}$Mpc]")
    ax.set_title(f"CF4 super-resolved galaxy field ({fine} $h^{{-1}}$Mpc)\nTNG-trained, "
                 f"conditioned on CF4 groups")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "cf4_sr_local.png"), dpi=120)
    print(f"[apply] saved {os.path.join(args.out, 'cf4_sr_local.png')}", flush=True)


if __name__ == "__main__":
    main()
