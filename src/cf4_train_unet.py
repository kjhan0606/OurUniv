#!/usr/bin/env python
"""Amortized net (U-Net): galaxy peculiar-velocity observable -> matter density.

Upgrade of cf4_train_cnn.py's flat 5-layer CNN (receptive field only +-10 Mpc/h,
< the 30-60 Mpc velocity coherence) to a **periodic 3D U-Net** with 3 downsampling
levels: 64 -> 32 -> 16 -> 8^3 bottleneck = GLOBAL receptive field. This is the
right inductive bias for the long-range v -> delta inverse (delta ~ -div v),
and the same backbone reused later for the conditional-diffusion posterior.

Architecture adapted from CIRCLE wp2_score.py UNet3D (equinox), made periodic
(wrap padding) and un-conditioned (deterministic posterior-mean regressor).

Same data/observable/target/eval harness as cf4_train_cnn.py (imported), so the
r(k) numbers are directly comparable to the flat-CNN baseline.

Run (Slurm): sbatch src/cf4_job.slurm src/cf4_train_unet.py --data data_train --ablation full
"""
import os
os.environ["JAX_ENABLE_X64"] = "0"   # float32 (before any jax import)
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# reuse the exact data pipeline + metrics from the flat-CNN trainer
from cf4_train_cnn import (load_split, build_inputs, IN_CH, rk_curve, augment_batch)


PAD_MODE = "wrap"   # "wrap" for periodic full-box; "reflect" for non-periodic sub-cubes


def pwrap(x, p=1):
    """Pad the 3 spatial dims of a (C,D,H,W) tensor (PAD_MODE: wrap|reflect)."""
    return jnp.pad(x, ((0, 0), (p, p), (p, p), (p, p)), mode=PAD_MODE)


def load_split_subcube(data_dir, split):
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, split, "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no sub-cube shards in {data_dir}/{split}")
    ng, vl, dm, off = [], [], [], []
    pN = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        ng.append(z["n_gal"]); vl.append(z["vlos"]); dm.append(z["delta_m"])
        off.append(z["offset"]); pN = int(z["parent_N"])
    return (np.concatenate(ng).astype(np.float32), np.concatenate(vl).astype(np.float32),
            np.concatenate(dm).astype(np.float32), np.concatenate(off).astype(np.int64), pN)


def build_inputs_subcube(ng, vl, offsets, parent_N, norm):
    """5-ch input with per-sample GLOBAL-position coords (LOS geometry vs observer)."""
    ng_mu, ng_sd, vl_mu, vl_sd = norm
    n, sub = ng.shape[0], ng.shape[-1]
    X = np.empty((n, 5, sub, sub, sub), np.float32)
    X[:, 0] = (ng - ng_mu) / ng_sd
    X[:, 1] = (vl - vl_mu) / vl_sd
    base = np.arange(sub) + 0.5
    half = parent_N / 2.0
    for i in range(n):
        ox, oy, oz = offsets[i]
        cx = (ox + base - half) / half
        cy = (oy + base - half) / half
        cz = (oz + base - half) / half
        CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")
        X[i, 2] = CX; X[i, 3] = CY; X[i, 4] = CZ
    return X


class Block(eqx.Module):
    c1: eqx.nn.Conv3d
    c2: eqx.nn.Conv3d
    n1: eqx.nn.GroupNorm
    n2: eqx.nn.GroupNorm

    def __init__(self, cin, cout, key):
        k1, k2 = jax.random.split(key, 2)
        self.c1 = eqx.nn.Conv3d(cin, cout, 3, padding=0, key=k1)   # periodic via pwrap
        self.c2 = eqx.nn.Conv3d(cout, cout, 3, padding=0, key=k2)
        g = min(8, cout)
        self.n1 = eqx.nn.GroupNorm(g, cout)
        self.n2 = eqx.nn.GroupNorm(g, cout)

    def __call__(self, x):
        h = jax.nn.silu(self.n1(self.c1(pwrap(x))))
        h = jax.nn.silu(self.n2(self.c2(pwrap(h))))
        return h


class UNet3D(eqx.Module):
    inb: Block
    d1: Block
    d2: Block
    d3: Block
    mid: Block
    u3: Block
    u2: Block
    u1: Block
    out: eqx.nn.Conv3d

    def __init__(self, in_ch=IN_CH, C=24, key=None):
        ks = jax.random.split(key, 9)
        self.inb = Block(in_ch, C, ks[0])
        self.d1 = Block(C, 2 * C, ks[1])
        self.d2 = Block(2 * C, 4 * C, ks[2])
        self.d3 = Block(4 * C, 8 * C, ks[3])
        self.mid = Block(8 * C, 8 * C, ks[4])
        self.u3 = Block(8 * C + 4 * C, 4 * C, ks[5])
        self.u2 = Block(4 * C + 2 * C, 2 * C, ks[6])
        self.u1 = Block(2 * C + C, C, ks[7])
        self.out = eqx.nn.Conv3d(C, 1, 1, key=ks[8])

    def __call__(self, x):                       # x: (in_ch,N,N,N)
        h0 = self.inb(x)                          # (C, 64)
        h1 = self.d1(self._down(h0))              # (2C, 32)
        h2 = self.d2(self._down(h1))              # (4C, 16)
        h3 = self.d3(self._down(h2))              # (8C, 8)  <- global RF
        m = self.mid(h3)
        u = self.u3(jnp.concatenate([self._up(m), h2], 0))
        u = self.u2(jnp.concatenate([self._up(u), h1], 0))
        u = self.u1(jnp.concatenate([self._up(u), h0], 0))
        return self.out(u)[0]                      # (N,N,N)

    @staticmethod
    def _down(x):
        return eqx.nn.MaxPool3d(2, 2)(x)

    @staticmethod
    def _up(x):
        c, a, b, d = x.shape
        return jax.image.resize(x, (c, a * 2, b * 2, d * 2), method="nearest")


def batched_forward(model, X, bs):
    fwd = eqx.filter_jit(lambda xb: jax.vmap(model)(xb))
    return np.concatenate([np.asarray(fwd(jnp.asarray(X[i:i + bs])))
                           for i in range(0, X.shape[0], bs)], 0)


def batched_mse(model, X, Y, bs):
    fwd = eqx.filter_jit(lambda xb: jax.vmap(model)(xb))
    tot, n = 0.0, X.shape[0]
    for i in range(0, n, bs):
        p = np.asarray(fwd(jnp.asarray(X[i:i + bs])))
        tot += float(np.mean((p - Y[i:i + bs]) ** 2)) * (min(i + bs, n) - i)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_train")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--C", type=int, default=24, help="base channel width")
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--ablation", choices=["full", "novel", "nogal"], default="full")
    ap.add_argument("--augment", action="store_true", help="octahedral augmentation")
    ap.add_argument("--subcube", action="store_true",
                    help="sub-cube data (global-position coords + reflect padding)")
    ap.add_argument("--pad-mode", choices=["wrap", "reflect"], default=None)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    print(f"[env] jax {jax.__version__} eqx {eqx.__version__} dev={jax.devices()[0]}")
    tf = lambda d: np.arcsinh(d).astype(np.float32)

    global PAD_MODE
    PAD_MODE = args.pad_mode or ("reflect" if args.subcube else "wrap")
    if args.subcube and args.augment:
        print("[warn] octahedral augment invalid for fixed-position sub-cubes; disabling")
        args.augment = False

    if args.subcube:
        ng_tr, vl_tr, dm_tr, off_tr, pN = load_split_subcube(args.data, "train")
        ng_va, vl_va, dm_va, off_va, _ = load_split_subcube(args.data, "val")
        ng_te, vl_te, dm_te, off_te, _ = load_split_subcube(args.data, "test")
        args.N = ng_tr.shape[-1]
        norm = (ng_tr.mean(), ng_tr.std() + 1e-8, vl_tr.mean(), vl_tr.std() + 1e-8)
        Xtr = build_inputs_subcube(ng_tr, vl_tr, off_tr, pN, norm)
        Xva = build_inputs_subcube(ng_va, vl_va, off_va, pN, norm)
        Xte = build_inputs_subcube(ng_te, vl_te, off_te, pN, norm)
        box = args.N * args.spacing
        print(f"[data] SUB-CUBE mode: parent_N={pN} sub={args.N} voxel={args.spacing} "
              f"box(sub)={box:.0f} Mpc/h pad={PAD_MODE}")
    else:
        box = args.N * args.spacing
        ng_tr, vl_tr, dm_tr = load_split(args.data, "train")
        ng_va, vl_va, dm_va = load_split(args.data, "val")
        ng_te, vl_te, dm_te = load_split(args.data, "test")
        norm = (ng_tr.mean(), ng_tr.std() + 1e-8, vl_tr.mean(), vl_tr.std() + 1e-8)
        Xtr = build_inputs(ng_tr, vl_tr, args.N, norm)
        Xva = build_inputs(ng_va, vl_va, args.N, norm)
        Xte = build_inputs(ng_te, vl_te, args.N, norm)
    if args.ablation == "novel":
        Xtr[:, 1] = 0; Xva[:, 1] = 0; Xte[:, 1] = 0
    elif args.ablation == "nogal":
        Xtr[:, 0] = 0; Xva[:, 0] = 0; Xte[:, 0] = 0

    ytr = tf(dm_tr); yva = tf(dm_va); yte = tf(dm_te)
    ymu, ysd = ytr.mean(), ytr.std() + 1e-8
    Ytr = ((ytr - ymu) / ysd).astype(np.float32)
    Yva = ((yva - ymu) / ysd).astype(np.float32)
    Yte_true = ((yte - ymu) / ysd).astype(np.float32)
    base_va = float(np.mean(Yva ** 2))
    print(f"[data] train {Xtr.shape} val {Xva.shape} test {Xte.shape} | "
          f"ablation={args.ablation} | baseline val MSE={base_va:.4f}")

    key = jax.random.PRNGKey(0)
    model = UNet3D(in_ch=IN_CH, C=args.C, key=key)
    npar = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"[model] periodic 3D U-Net (3 levels, C={args.C}) -> {npar/1e6:.2f}M params")

    ntr = Xtr.shape[0]
    steps_per_ep = (ntr + args.batch - 1) // args.batch
    sched = optax.cosine_decay_schedule(args.lr, args.epochs * steps_per_ep, alpha=0.05)
    opt = optax.adamw(sched, weight_decay=args.wd)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        def loss_fn(m):
            pred = jax.vmap(m)(xb)
            return jnp.mean((pred - yb) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    best = 1e9; best_ep = 0; best_model = model
    rng = np.random.default_rng(0); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(ntr); tl = 0.0
        for i in range(0, ntr, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = Xtr[idx], Ytr[idx]
            if args.augment:
                xb, yb = augment_batch(xb, yb, rng)
            model, opt_state, loss = step(model, opt_state,
                                          jnp.asarray(xb), jnp.asarray(yb))
            tl += float(loss) * len(idx)
        tl /= ntr
        vl = batched_mse(model, Xva, Yva, args.batch)
        if vl < best:
            best = vl; best_ep = ep; best_model = model
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"[ep {ep:3d}] train_mse={tl:.4f} val_mse={vl:.4f} "
                  f"varexp={1-vl/base_va:.3f} ({time.time()-t0:.0f}s)", flush=True)

    model = best_model
    print(f"[select] best val_mse={best:.4f} at epoch {best_ep}")
    pred = batched_forward(model, Xte, args.batch)
    r_vox = float(np.corrcoef(Yte_true.ravel(), pred.ravel())[0, 1])
    kc, rk = rk_curve(Yte_true, pred, box)
    print("\n================ reserved TEST (U-Net) ================")
    print(f"ablation={args.ablation}  voxel r = {r_vox:.3f}")
    for i in range(len(kc)):
        if np.isfinite(rk[i]):
            print(f"{kc[i]:9.4f} {rk[i]:7.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].imshow(Yte_true[0].mean(0), cmap="magma"); ax[0].set_title("true asinh(delta_m)")
    ax[1].imshow(pred[0].mean(0), cmap="magma"); ax[1].set_title("U-Net predicted")
    ax[2].axhline(0, ls=":", c="k"); ax[2].axhline(1, ls=":", c="k")
    ax[2].semilogx(kc, rk, "o-", ms=4)
    ax[2].set_xlabel("k [h/Mpc]"); ax[2].set_ylabel("r(k)"); ax[2].set_ylim(-0.1, 1.05)
    ax[2].set_title(f"U-Net recovery (voxel r={r_vox:.3f}, {args.ablation})")
    fig.tight_layout()
    out = os.path.join(args.out, f"cf4_unet_{args.ablation}.png")
    fig.savefig(out, dpi=120)
    np.savez(os.path.join(args.out, f"cf4_unet_pred_{args.ablation}.npz"),
             true=Yte_true, pred=pred, kc=kc, rk=rk, r_vox=r_vox, best_val=best)
    print(f"\n[fig] saved {out}\n[done] best val_mse={best:.4f} varexp={1-best/base_va:.3f}")


if __name__ == "__main__":
    main()
