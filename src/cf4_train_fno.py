#!/usr/bin/env python
"""Amortized net (FNO): galaxy peculiar-velocity observable -> matter density.

Fourier Neural Operator benchmark alongside the U-Net. The v -> delta inverse is
nearly diagonal in Fourier space (v(k) ~ delta(k)/k), so an operator that mixes
channels per Fourier mode is a natural inductive bias -- and it is inherently
GLOBAL (each mode couples the whole box) and resolution-agnostic (helps when N
is scaled up).

SpectralConv3d: rfftn -> keep `modes` low frequencies (4 (dim1,dim2) corners x
low dim3) -> per-mode complex channel mixing -> irfftn; plus a pointwise (1x1)
residual path for high-frequency detail. Complex weights stored as real (re,im)
so optax/adamw optimize real params.

Same data/observable/target/eval harness as the CNN/U-Net (imported), so r(k)
is directly comparable.

Run (Slurm): sbatch src/cf4_job.slurm src/cf4_train_fno.py --data data_train --ablation full
"""
import os
os.environ["JAX_ENABLE_X64"] = "0"
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from cf4_train_cnn import load_split, build_inputs, IN_CH, rk_curve


def init_fno(key, in_ch=IN_CH, width=16, modes=8, layers=4, proj=64):
    ks = jax.random.split(key, 4 + 2 * layers)
    p = {}
    # lift (pointwise) in_ch -> width, and project width -> proj -> 1
    p["lift"] = jax.random.normal(ks[0], (width, in_ch)) * np.sqrt(1.0 / in_ch)
    p["lift_b"] = jnp.zeros((width,))
    p["proj1"] = jax.random.normal(ks[1], (proj, width)) * np.sqrt(1.0 / width)
    p["proj1_b"] = jnp.zeros((proj,))
    p["proj2"] = jax.random.normal(ks[2], (1, proj)) * np.sqrt(1.0 / proj)
    p["proj2_b"] = jnp.zeros((1,))
    p["layers"] = []
    scale = 1.0 / (width * width)
    for l in range(layers):
        kl = jax.random.split(ks[4 + l], 2)
        # 4 corner complex weights (re,im): (4,2,width,width,modes,modes,modes_z)
        R = jax.random.normal(kl[0], (4, 2, width, width, modes, modes, modes)) * scale
        W = jax.random.normal(kl[1], (width, width)) * np.sqrt(1.0 / width)  # pointwise
        b = jnp.zeros((width,))
        p["layers"].append({"R": R, "W": W, "b": b})
    p["_meta"] = (width, modes, layers)
    return p


def _spectral(h, R, modes):
    """h: (width,N,N,N) real -> spectral-mixed (width,N,N,N) real."""
    C, N, _, _ = h.shape
    hf = jnp.fft.rfftn(h, axes=(1, 2, 3))          # (C,N,N,N//2+1) complex
    m = modes
    out = jnp.zeros_like(hf)
    corners = [(slice(0, m), slice(0, m)), (slice(0, m), slice(N - m, N)),
               (slice(N - m, N), slice(0, m)), (slice(N - m, N), slice(N - m, N))]
    for c, (s1, s2) in enumerate(corners):
        blk = hf[:, s1, s2, :m]                     # (C,m,m,m) complex
        Rr = R[c, 0]; Ri = R[c, 1]                  # (Co,Ci,m,m,m)
        br, bi = blk.real, blk.imag
        # complex einsum: (Rr+iRi)(br+ibi)
        or_ = jnp.einsum("oixyz,ixyz->oxyz", Rr, br) - jnp.einsum("oixyz,ixyz->oxyz", Ri, bi)
        oi_ = jnp.einsum("oixyz,ixyz->oxyz", Rr, bi) + jnp.einsum("oixyz,ixyz->oxyz", Ri, br)
        out = out.at[:, s1, s2, :m].set(or_ + 1j * oi_)
    return jnp.fft.irfftn(out, s=(N, N, N), axes=(1, 2, 3))


def fno_forward(p, x):
    """x: (in_ch,N,N,N) -> (N,N,N)."""
    width, modes, layers = p["_meta"]
    h = jnp.einsum("wc,cxyz->wxyz", p["lift"], x) + p["lift_b"][:, None, None, None]
    for l in range(layers):
        L = p["layers"][l]
        sp = _spectral(h, L["R"], modes)
        pw = jnp.einsum("oi,ixyz->oxyz", L["W"], h) + L["b"][:, None, None, None]
        h = jax.nn.gelu(sp + pw)
    h = jax.nn.gelu(jnp.einsum("pw,wxyz->pxyz", p["proj1"], h) + p["proj1_b"][:, None, None, None])
    h = jnp.einsum("op,pxyz->oxyz", p["proj2"], h) + p["proj2_b"][:, None, None, None]
    return h[0]


def batched_forward(p, X, bs):
    f = jax.jit(lambda xb: jax.vmap(lambda a: fno_forward(p, a))(xb))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + bs])))
                           for i in range(0, X.shape[0], bs)], 0)


def batched_mse(p, X, Y, bs):
    f = jax.jit(lambda xb: jax.vmap(lambda a: fno_forward(p, a))(xb))
    tot, n = 0.0, X.shape[0]
    for i in range(0, n, bs):
        pr = np.asarray(f(jnp.asarray(X[i:i + bs])))
        tot += float(np.mean((pr - Y[i:i + bs]) ** 2)) * (min(i + bs, n) - i)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_train")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--ablation", choices=["full", "novel", "nogal"], default="full")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    print(f"[env] jax {jax.__version__} dev={jax.devices()[0]}")
    box = args.N * args.spacing
    tf = lambda d: np.arcsinh(d).astype(np.float32)

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
    print(f"[data] train {Xtr.shape} | ablation={args.ablation} | baseline val MSE={base_va:.4f}")

    key = jax.random.PRNGKey(0)
    p = init_fno(key, in_ch=IN_CH, width=args.width, modes=args.modes, layers=args.layers)
    npar = sum(int(np.asarray(a).size) for a in jax.tree_util.tree_leaves(
        {k: v for k, v in p.items() if k != "_meta"}))
    print(f"[model] FNO3D width={args.width} modes={args.modes} layers={args.layers} "
          f"-> {npar/1e6:.2f}M params")

    ntr = Xtr.shape[0]
    steps = (ntr + args.batch - 1) // args.batch * args.epochs
    sched = optax.cosine_decay_schedule(args.lr, steps, alpha=0.05)
    opt = optax.adamw(sched, weight_decay=args.wd)
    # optimize only array leaves (skip _meta tuple)
    trainable = {k: v for k, v in p.items() if k != "_meta"}
    opt_state = opt.init(trainable)

    @jax.jit
    def step(trainable, opt_state, xb, yb):
        def loss_fn(tp):
            pp = dict(tp); pp["_meta"] = p["_meta"]
            pred = jax.vmap(lambda a: fno_forward(pp, a))(xb)
            return jnp.mean((pred - yb) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(trainable)
        updates, opt_state = opt.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        return trainable, opt_state, loss

    best = 1e9; best_ep = 0; best_tp = trainable
    rng = np.random.default_rng(0); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(ntr); tl = 0.0
        for i in range(0, ntr, args.batch):
            idx = perm[i:i + args.batch]
            trainable, opt_state, loss = step(trainable, opt_state,
                                              jnp.asarray(Xtr[idx]), jnp.asarray(Ytr[idx]))
            tl += float(loss) * len(idx)
        tl /= ntr
        pp = dict(trainable); pp["_meta"] = p["_meta"]
        vl = batched_mse(pp, Xva, Yva, args.batch)
        if vl < best:
            best = vl; best_ep = ep; best_tp = jax.tree_util.tree_map(lambda a: a, trainable)
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"[ep {ep:3d}] train_mse={tl:.4f} val_mse={vl:.4f} "
                  f"varexp={1-vl/base_va:.3f} ({time.time()-t0:.0f}s)", flush=True)

    pp = dict(best_tp); pp["_meta"] = p["_meta"]
    print(f"[select] best val_mse={best:.4f} at epoch {best_ep}")
    pred = batched_forward(pp, Xte, args.batch)
    r_vox = float(np.corrcoef(Yte_true.ravel(), pred.ravel())[0, 1])
    kc, rk = rk_curve(Yte_true, pred, box)
    print("\n================ reserved TEST (FNO) ================")
    print(f"ablation={args.ablation}  voxel r = {r_vox:.3f}")
    for i in range(len(kc)):
        if np.isfinite(rk[i]):
            print(f"{kc[i]:9.4f} {rk[i]:7.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].imshow(Yte_true[0].mean(0), cmap="magma"); ax[0].set_title("true asinh(delta_m)")
    ax[1].imshow(pred[0].mean(0), cmap="magma"); ax[1].set_title("FNO predicted")
    ax[2].axhline(0, ls=":", c="k"); ax[2].axhline(1, ls=":", c="k")
    ax[2].semilogx(kc, rk, "o-", ms=4)
    ax[2].set_xlabel("k [h/Mpc]"); ax[2].set_ylabel("r(k)"); ax[2].set_ylim(-0.1, 1.05)
    ax[2].set_title(f"FNO recovery (voxel r={r_vox:.3f}, {args.ablation})")
    fig.tight_layout()
    out = os.path.join(args.out, f"cf4_fno_{args.ablation}.png")
    fig.savefig(out, dpi=120)
    np.savez(os.path.join(args.out, f"cf4_fno_pred_{args.ablation}.npz"),
             true=Yte_true, pred=pred, kc=kc, rk=rk, r_vox=r_vox, best_val=best)
    print(f"\n[fig] saved {out}\n[done] best val_mse={best:.4f} varexp={1-best/base_va:.3f}")


if __name__ == "__main__":
    main()
