#!/usr/bin/env python
"""Generate amortized-net training shards: (galaxy observable  ->  matter density).

For each seed, runs the full mock pipeline (pmwd -> FoF -> HOD -> v_pec) via
mock_pipeline.run_sample, then grids the sparse galaxy catalog (positions +
line-of-sight peculiar velocities, observer at box centre) into network-ready
fields with periodic CIC deposit:

  n_gal (N,N,N)   galaxy number field   (biased density tracer)
  vlos  (N,N,N)   LOS peculiar-velocity momentum  = CIC-deposit of v_pec,los [km/s]

Target:
  delta_m (N,N,N) present matter overdensity  (the field to recover)
  s       (N,N,N) whitened initial field       (kept for later IC-recovery target)

The 3 coordinate channels (LOS geometry) are constant across samples, so they are
NOT stored -- the trainer appends them. Shards mirror wp2_gen_data layout.

Disjoint seed namespaces per split (no leakage):
  train 0.., val 1_000_000.., test 2_000_000..

Slurm (one shard per array task, different seed window):
  sbatch --array=0-7 src/cf4_job.slurm src/cf4_gen_train.py \
      --N 64 --spacing 2.0 --per-task 32 --split train
  (array task k does seeds [k*per_task, (k+1)*per_task))
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import json
import time
import numpy as np


def cic_deposit(pos, values, N, L):
    """Periodic Cloud-In-Cell deposit. pos (Ng,3) in [0,L); values scalar or (Ng,)."""
    field = np.zeros((N, N, N), dtype=np.float64)
    if pos.shape[0] == 0:
        return field
    cs = L / N
    x = (pos % L) / cs
    i0 = np.floor(x).astype(np.int64)
    f = x - i0
    vals = np.broadcast_to(values, (pos.shape[0],)).astype(np.float64)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                w = wx * wy * wz * vals
                ii = (i0[:, 0] + dx) % N
                jj = (i0[:, 1] + dy) % N
                kk = (i0[:, 2] + dz) % N
                np.add.at(field, (ii, jj, kk), w)
    return field


def grid_sample(rec, vnoise_coef=0.0, sigma_nl=200.0, sel_scale=0.0, seed=0):
    """rec from mock_pipeline.run_sample -> gridded (n_gal, vlos, delta_m, s).

    Realistic-observable options (observer at box centre):
      sel_scale > 0 : radial selection, keep prob = exp(-r/sel_scale) [Mpc/h]
                      (flux-limited-like sparsification, denser near the observer).
      vnoise_coef>0 : distance-error velocity noise, sigma_v(r) =
                      sqrt((vnoise_coef * r)^2 + sigma_nl^2) [km/s]; r in Mpc/h.
                      vnoise_coef = 0.2*100 = 20 km/s per Mpc/h mirrors CF4
                      (sig_lnd~0.2, H0*d*sig_lnd), giving S/N<1 at the box edge.
    The delta_m / s TARGETS stay clean; only the observable is degraded.
    """
    N = int(rec["N"]); L = float(rec["L"])
    gp = rec["gal_pos"].astype(np.float64)
    vpec = rec["gal_vpec"].astype(np.float64)
    rng = np.random.default_rng(int(seed) * 13 + 7)
    r = np.linalg.norm(gp - L / 2.0, axis=1)
    if sel_scale > 0 and gp.shape[0] > 0:
        keep = rng.random(gp.shape[0]) < np.exp(-r / sel_scale)
        gp, vpec, r = gp[keep], vpec[keep], r[keep]
    if vnoise_coef > 0 and gp.shape[0] > 0:
        sigv = np.sqrt((vnoise_coef * r) ** 2 + sigma_nl ** 2)
        vpec = vpec + rng.normal(0.0, 1.0, gp.shape[0]) * sigv
    n_gal = cic_deposit(gp, 1.0, N, L)
    vlos = cic_deposit(gp, vpec, N, L)
    return dict(
        n_gal=n_gal.astype(np.float32),
        vlos=vlos.astype(np.float32),
        delta_m=rec["delta_m"].astype(np.float32),
        s=rec["s"].astype(np.float32),
        n_gal_total=np.int64(gp.shape[0]),
    )


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mock_pipeline import run_sample, make_forward

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--per-task", type=int, default=32, help="samples in this shard")
    ap.add_argument("--seed-start", type=int, default=None,
                    help="explicit first seed for a SINGLE (non-array) shard; ignores array id")
    ap.add_argument("--seed-base", type=int, default=None,
                    help="base seed added to arr*per_task across an array (disjoint shards); "
                         "default = split base (train 0 / val 1e6 / test 2e6)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--b", type=float, default=0.2)
    ap.add_argument("--n-min", type=int, default=10)
    ap.add_argument("--logMmin", type=float, default=12.8)
    ap.add_argument("--sigma-logM", type=float, default=0.4)
    ap.add_argument("--logM0", type=float, default=12.3)
    ap.add_argument("--logM1", type=float, default=13.9)
    ap.add_argument("--alpha", type=float, default=1.0)
    # realistic-observable degradation (CF4-like): distance-error velocity noise +
    # radial selection. Defaults 0 = clean (backward compatible).
    ap.add_argument("--vnoise-frac", type=float, default=0.0,
                    help="distance-error frac (sig_lnd); sigma_v=sqrt((100*frac*r)^2+sigma_nl^2)")
    ap.add_argument("--sigma-nl", type=float, default=200.0, help="km/s intrinsic v noise")
    ap.add_argument("--sel-scale", type=float, default=0.0,
                    help="Mpc/h radial selection scale; keep prob=exp(-r/scale) (0=off)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data_train"))
    args = ap.parse_args()

    import jax
    print(f"[env] jax {jax.__version__} dev={jax.devices()[0]}")

    base = args.seed_base if args.seed_base is not None else \
        {"train": 0, "val": 1_000_000, "test": 2_000_000}[args.split]
    arr = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed0 = args.seed_start if args.seed_start is not None else base + arr * args.per_task
    outdir = os.path.join(args.out, args.split)
    os.makedirs(outdir, exist_ok=True)
    hod = dict(logMmin=args.logMmin, sigma_logM=args.sigma_logM, logM0=args.logM0,
               logM1=args.logM1, alpha=args.alpha)
    vnoise_coef = args.vnoise_frac * 100.0   # km/s per Mpc/h  (0.2 -> 20 = CF4-like)
    print(f"[gen] split={args.split} shard seeds [{seed0},{seed0+args.per_task}) "
          f"N={args.N} spacing={args.spacing} L={args.N*args.spacing:.0f} Mpc/h")
    print(f"[gen] observable: vnoise_frac={args.vnoise_frac} (coef={vnoise_coef} km/s per Mpc/h, "
          f"sigma_nl={args.sigma_nl}) sel_scale={args.sel_scale} Mpc/h "
          f"({'REALISTIC' if (vnoise_coef>0 or args.sel_scale>0) else 'CLEAN'})")

    N = args.N
    buf = {k: np.empty((args.per_task, N, N, N), np.float32)
           for k in ("n_gal", "vlos", "delta_m", "s")}
    buf["seed"] = np.empty(args.per_task, np.int64)
    buf["n_gal_total"] = np.empty(args.per_task, np.int64)

    import jax.numpy as jnp
    fwd = make_forward(N, args.spacing, jnp.float32)   # build/jit forward ONCE
    t0 = time.time()
    for j in range(args.per_task):
        seed = seed0 + j
        rec = run_sample(N, args.spacing, seed, args.Om, hod,
                         b=args.b, n_min=args.n_min, fwd=fwd, verbose=(j == 0))
        g = grid_sample(rec, vnoise_coef=vnoise_coef, sigma_nl=args.sigma_nl,
                        sel_scale=args.sel_scale, seed=seed)
        for k in ("n_gal", "vlos", "delta_m", "s"):
            buf[k][j] = g[k]
        buf["seed"][j] = seed
        buf["n_gal_total"][j] = g["n_gal_total"]
        if (j + 1) % 8 == 0 or j == args.per_task - 1:
            rate = (j + 1) / (time.time() - t0)
            print(f"[gen] {j+1}/{args.per_task}  <n_gal>={buf['n_gal_total'][:j+1].mean():.0f} "
                  f"({rate:.2f} samp/s)", flush=True)

    fname = os.path.join(outdir, f"shard_{seed0:09d}_{args.per_task:04d}.npz")
    meta = dict(N=N, spacing=args.spacing, L=N * args.spacing, Om=args.Om,
                b=args.b, n_min=args.n_min, hod=hod, split=args.split,
                seed_start=seed0, per_task=args.per_task)
    np.savez(fname, meta=json.dumps(meta), **buf)
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gen] wrote {fname} ({os.path.getsize(fname)/1e6:.1f} MB) in "
          f"{time.time()-t0:.0f}s  mean galaxies/box={buf['n_gal_total'].mean():.0f}")


if __name__ == "__main__":
    main()
