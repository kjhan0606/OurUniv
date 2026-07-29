#!/usr/bin/env python
"""High-resolution sub-cube training data (Hong+2021 method) at 0.3125 Mpc/h voxel.

Runs one large parent box (pmwd N -> FoF -> HOD -> v_pec, observer at box centre),
grids the full-box observable + target at the parent resolution, then TILES it into
sub^3 sub-cubes (Hong+2021: 20-40 Mpc/h sub-cubes). Each sub-cube is a training
sample at the fine voxel size; its global offset is stored so the trainer can build
LOS-geometry coordinate channels (position relative to the observer at parent centre).

Parent: N=256, spacing=0.3125 -> 80 Mpc/h box, 0.3125 Mpc/h voxel (== Hong TNG100).
Sub-cube: 64^3 -> 20 Mpc/h each; stride 64 -> 4^3=64 disjoint sub-cubes/parent.

One Slurm array task = one parent box = one shard of (n_sub) samples. Parents are
kept disjoint per split (no large-scale leakage).

Store per sub-cube: n_gal, vlos, delta_m (sub^3 float32) + offset (3 int) + parent_N.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import json
import time
import numpy as np
from cf4_gen_train import cic_deposit


def extract_subcubes(n_gal, vlos, delta_m, sub, stride):
    """Tile (N,N,N) fields into sub^3 blocks. Returns lists + offsets (cells)."""
    N = n_gal.shape[0]
    offs = range(0, N - sub + 1, stride)
    NG, VL, DM, OFF = [], [], [], []
    for ox in offs:
        for oy in offs:
            for oz in offs:
                sl = (slice(ox, ox + sub), slice(oy, oy + sub), slice(oz, oz + sub))
                NG.append(n_gal[sl]); VL.append(vlos[sl]); DM.append(delta_m[sl])
                OFF.append((ox, oy, oz))
    return (np.array(NG, np.float32), np.array(VL, np.float32),
            np.array(DM, np.float32), np.array(OFF, np.int64))


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mock_pipeline import run_sample, make_forward, RHO_CRIT

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=256, help="parent grid")
    ap.add_argument("--spacing", type=float, default=0.3125, help="Mpc/h per cell")
    ap.add_argument("--sub", type=int, default=64, help="sub-cube grid size")
    ap.add_argument("--stride", type=int, default=64, help="tiling stride (cells)")
    ap.add_argument("--n-parents", type=int, default=1, help="parent boxes this task")
    ap.add_argument("--seed-start", type=int, default=None)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--b", type=float, default=0.2)
    ap.add_argument("--n-min", type=int, default=20)
    ap.add_argument("--logMmin", type=float, default=12.0)
    ap.add_argument("--sigma-logM", type=float, default=0.35)
    ap.add_argument("--logM0", type=float, default=11.5)
    ap.add_argument("--logM1", type=float, default=13.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--vnoise-frac", type=float, default=0.0)
    ap.add_argument("--sigma-nl", type=float, default=200.0)
    ap.add_argument("--sel-scale", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data_hires"))
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    print(f"[env] jax {jax.__version__} dev={jax.devices()[0]}")
    N, sub = args.N, args.sub
    L = N * args.spacing
    base = {"train": 0, "val": 1_000_000, "test": 2_000_000}[args.split]
    arr = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed0 = args.seed_start if args.seed_start is not None else base + arr * args.n_parents
    outdir = os.path.join(args.out, args.split)
    os.makedirs(outdir, exist_ok=True)
    hod = dict(logMmin=args.logMmin, sigma_logM=args.sigma_logM, logM0=args.logM0,
               logM1=args.logM1, alpha=args.alpha)
    vnoise_coef = args.vnoise_frac * 100.0
    print(f"[gen] hires sub-cube: parent N={N} L={L:.0f} Mpc/h voxel={args.spacing} "
          f"sub={sub} ({sub*args.spacing:.0f} Mpc/h) stride={args.stride}")
    print(f"[gen] split={args.split} parents [{seed0},{seed0+args.n_parents}) "
          f"vnoise={args.vnoise_frac} sel={args.sel_scale}")

    fwd = make_forward(N, args.spacing, jnp.float32)
    NG, VL, DM, OFF, SEED = [], [], [], [], []
    t0 = time.time()
    for pi in range(args.n_parents):
        seed = seed0 + pi
        rec = run_sample(N, args.spacing, seed, args.Om, hod,
                         b=args.b, n_min=args.n_min, fwd=fwd, verbose=(pi == 0))
        gp = rec["gal_pos"].astype(np.float64)
        vpec = rec["gal_vpec"].astype(np.float64)
        rng = np.random.default_rng(int(seed) * 13 + 7)
        r = np.linalg.norm(gp - L / 2.0, axis=1)
        if args.sel_scale > 0 and gp.shape[0] > 0:
            keep = rng.random(gp.shape[0]) < np.exp(-r / args.sel_scale)
            gp, vpec, r = gp[keep], vpec[keep], r[keep]
        if vnoise_coef > 0 and gp.shape[0] > 0:
            sigv = np.sqrt((vnoise_coef * r) ** 2 + args.sigma_nl ** 2)
            vpec = vpec + rng.normal(0.0, 1.0, gp.shape[0]) * sigv
        n_gal = cic_deposit(gp, 1.0, N, L)
        vlos = cic_deposit(gp, vpec, N, L)
        delta_m = rec["delta_m"].astype(np.float32)
        ng, vl, dm, off = extract_subcubes(n_gal, vlos, delta_m, sub, args.stride)
        NG.append(ng); VL.append(vl); DM.append(dm); OFF.append(off)
        SEED.append(np.full(off.shape[0], seed, np.int64))
        print(f"[gen] parent {pi+1}/{args.n_parents} seed={seed} "
              f"gal={gp.shape[0]} -> {off.shape[0]} sub-cubes "
              f"(<gal/sub>={ng.reshape(ng.shape[0],-1).sum(1).mean():.0f}) "
              f"[{time.time()-t0:.0f}s]", flush=True)

    NG = np.concatenate(NG); VL = np.concatenate(VL)
    DM = np.concatenate(DM); OFF = np.concatenate(OFF); SEED = np.concatenate(SEED)
    fname = os.path.join(outdir, f"shard_{seed0:09d}_{NG.shape[0]:04d}.npz")
    meta = dict(N=N, spacing=args.spacing, L=L, sub=sub, stride=args.stride,
                Om=args.Om, hod=hod, split=args.split, vnoise_frac=args.vnoise_frac,
                sel_scale=args.sel_scale, n_parents=args.n_parents, seed_start=seed0)
    np.savez(fname, n_gal=NG, vlos=VL, delta_m=DM, offset=OFF, seed=SEED,
             parent_N=np.int64(N), spacing=np.float64(args.spacing),
             meta=json.dumps(meta))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gen] wrote {fname} ({os.path.getsize(fname)/1e6:.1f} MB, "
          f"{NG.shape[0]} sub-cubes) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
