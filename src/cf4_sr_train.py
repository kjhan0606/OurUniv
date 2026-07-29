#!/usr/bin/env python
"""Position-conditioned super-resolution by conditional diffusion, trained on TNG300.

The model maps a coarse constrained density and the sparse OBSERVED galaxy positions to the
full fine galaxy field. The observed-position channel pins the generated biased peaks to the
data where it exists, and the model fills the rest with TNG-learned bias and clustering.

  conditioning c = [ d_coarse (upsampled),  n_obs (sparse observed galaxies) ]
  target       y = asinh(n_fine)   (full TNG galaxy count on the fine grid)

DDPM eps-prediction on y conditioned on c, with the 3D conditional U-Net of the CF4 IC model.
Octahedral augmentation is exact because TNG is periodic and statistically isotropic.

modes: train | sample. Run on GPU via src/cf4_job.slurm.
"""
import os
os.environ["JAX_ENABLE_X64"] = "0"
import glob, time, argparse, pickle
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
import optax

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf4_train_diffusion as D          # UNet3D, make_schedule, build_basis_np


def upsample(coarse, factor):
    """nearest-neighbour upsample a (n,c,c,c) or (c,c,c) field by integer factor."""
    return np.repeat(np.repeat(np.repeat(coarse, factor, -1), factor, -2), factor, -3)


def load_shard(f, sub, cf):
    with np.load(f) as z:
        nf = z["n_fine"].astype(np.float32)                 # (n,sub,sub,sub)
        no = z["n_obs"].astype(np.float32)
        dc = z["d_coarse"].astype(np.float32)               # (n,csub,csub,csub)
    y = np.arcsinh(nf)                                       # target
    dcu = upsample(dc, cf)                                   # (n,sub,sub,sub)
    obs = np.arcsinh(no)
    cond = np.stack([dcu, obs], 1)                           # (n,2,sub,sub,sub)
    return y, cond


def augment(Y, C, rng):
    for i in range(Y.shape[0]):
        perm = tuple(rng.permutation(3))
        fl = tuple(a for a in range(3) if rng.random() < 0.5)
        def iso(a):
            if perm != (0, 1, 2): a = np.transpose(a, perm)
            if fl: a = np.flip(a, fl)
            return a
        Y[i] = iso(Y[i])
        for k in range(C.shape[1]): C[i, k] = iso(C[i, k])
    return Y, C


def train(args):
    trf = sorted(glob.glob(os.path.join(args.data, "train", "shard_*.npz")))
    with open(os.path.join(args.data, "meta.json")) as fh:
        import json; meta = json.load(fh)
    sub = meta["sub"]; cf = meta["coarse_factor"]
    # standardization stats from one pass
    ys, cs = [], []
    for f in trf:
        y, c = load_shard(f, sub, cf); ys.append(y); cs.append(c)
    Y = np.concatenate(ys); C = np.concatenate(cs)
    ymu, ysd = float(Y.mean()), float(Y.std() + 1e-6)
    cmu = C.reshape(-1, 2, sub**3).mean((0, 2)); csd = C.reshape(-1, 2, sub**3).std((0, 2)) + 1e-6
    Yn = (Y - ymu) / ysd
    Cn = (C - cmu[None, :, None, None, None]) / csd[None, :, None, None, None]
    print(f"[sr] train {Yn.shape[0]} sub-cubes sub={sub} | y mu={ymu:.3f} sd={ysd:.3f} "
          f"| cond mu={cmu} sd={csd}", flush=True)

    betas, alphas, abars = D.make_schedule(args.T)
    key = jax.random.PRNGKey(args.seed); key, mk = jax.random.split(key)
    model = D.UNet3D(2, C=args.C, key=mk, remat=bool(args.remat))
    npar = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"[sr] UNet3D C={args.C} params={npar/1e6:.2f}M remat={bool(args.remat)}", flush=True)
    opt = optax.adam(args.lr); ostate = opt.init(eqx.filter(model, eqx.is_array))

    def loss_one(m, y0, c, t, eps):
        ab = abars[t]; yt = jnp.sqrt(ab) * y0 + jnp.sqrt(1 - ab) * eps
        return jnp.mean((m(yt, c, t.astype(jnp.float32) / args.T) - eps) ** 2)

    @eqx.filter_jit
    def step(m, ostate, yb, cb, tb, eb):
        def bl(mm): return jnp.mean(jax.vmap(lambda y, c, t, e: loss_one(mm, y, c, t, e))(yb, cb, tb, eb))
        l, gr = eqx.filter_value_and_grad(bl)(m)
        up, ostate = opt.update(gr, ostate, m)
        return eqx.apply_updates(m, up), ostate, l

    def save(path, m):
        with open(path, "wb") as fh:
            pickle.dump({"ymu": ymu, "ysd": ysd, "cmu": cmu, "csd": csd, "C": args.C,
                         "T": args.T, "sub": sub, "cf": cf}, fh)
            eqx.tree_serialise_leaves(fh, m)

    bs = args.batch; rng = np.random.default_rng(args.seed); n = Yn.shape[0]
    t0 = time.time(); best = 1e9
    for ep in range(args.epochs):
        order = rng.permutation(n); tl = 0.0; nb = 0
        for i in range(0, n - bs + 1, bs):
            idx = order[i:i + bs]
            yb = Yn[idx].copy(); cb = Cn[idx].copy()
            if args.augment: yb, cb = augment(yb, cb, rng)
            key, k1, k2 = jax.random.split(key, 3)
            tb = jax.random.randint(k1, (bs,), 0, args.T)
            eb = jax.random.normal(k2, (bs, sub, sub, sub))
            model, ostate, l = step(model, ostate, jnp.asarray(yb), jnp.asarray(cb), tb, eb)
            tl += float(l); nb += 1
        print(f"[ep {ep:3d}] train {tl/max(nb,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
        if tl / max(nb, 1) < best:
            best = tl / max(nb, 1); save(args.ckpt, model)
        save(args.ckpt + ".last", model)
    print(f"[sr] done best {best:.4f}", flush=True)


def sample(args):
    with open(args.ckpt, "rb") as fh:
        m = pickle.load(fh); sub = m["sub"]; cf = m["cf"]; T = m["T"]
        model = D.UNet3D(2, C=m["C"], key=jax.random.PRNGKey(0))
        model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x)
                                       and jnp.issubdtype(x.dtype, jnp.floating) else x, model)
        model = eqx.tree_deserialise_leaves(fh, model)
    betas, alphas, abars = D.make_schedule(T)
    vf = sorted(glob.glob(os.path.join(args.data, "val", "shard_*.npz")))
    Y, C = load_shard(vf[0], sub, cf)
    Cn = (C - m["cmu"][None, :, None, None, None]) / m["csd"][None, :, None, None, None]

    @eqx.filter_jit
    def ddim(model, S, c, t, tp, key):
        eps = jax.vmap(lambda s: model(s, c, t.astype(jnp.float32) / T))(S)
        ab = abars[t]; abp = jnp.where(tp >= 0, abars[jnp.maximum(tp, 0)], 1.0)
        s0 = (S - jnp.sqrt(1 - ab) * eps) / jnp.sqrt(ab)
        return jnp.sqrt(abp) * s0 + jnp.sqrt(jnp.maximum(1 - abp, 0)) * eps

    ts = np.linspace(T - 1, 0, args.ddim_steps).round().astype(int)
    seq = [(int(ts[i]), int(ts[i + 1]) if i + 1 < len(ts) else -1) for i in range(len(ts))]
    no = min(args.nobj, Y.shape[0]); outs = []
    key = jax.random.PRNGKey(1)
    for o in range(no):
        c = jnp.asarray(Cn[o]); key, k0 = jax.random.split(key)
        S = jax.random.normal(k0, (1, sub, sub, sub))
        for (t, tp) in seq:
            key, kt = jax.random.split(key)
            S = ddim(model, S, c, jnp.array(t), jnp.array(tp), kt)
        gen = np.asarray(S[0]) * m["ysd"] + m["ymu"]            # generated asinh(overdensity)
        outs.append(gen)
    g = np.array(outs); tr = Y[:no]                              # both in asinh(overdensity) space
    np.savez(os.path.join(args.out, f"cf4_sr_{args.tag}.npz"),
             gen=g, true=tr, obs=C[:no, 1], coarse=C[:no, 0])
    print(f"[sr-sample] {no} sub-cubes | gen std={g.std():.3f} true std={tr.std():.3f}", flush=True)
    # cross-correlation generated vs true (the key: does the generated field match?)
    rs = []
    for o in range(no):
        a = g[o] - g[o].mean(); b = tr[o] - tr[o].mean()
        rs.append(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2) + 1e-12))
    print(f"[sr-sample] gen-vs-true corr = {np.mean(rs):.3f}  "
          f"(peaks pinned to observed positions + TNG statistics elsewhere)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "sample"], required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data_sr"))
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon", "cf4_sr.eqx"))
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--remat", type=int, default=1)
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--nobj", type=int, default=8)
    ap.add_argument("--tag", default="tng")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon"))
    args = ap.parse_args()
    print(f"[env] jax {jax.__version__} dev={jax.devices()[0]} mode={args.mode}", flush=True)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    (train if args.mode == "train" else sample)(args)


if __name__ == "__main__":
    main()
