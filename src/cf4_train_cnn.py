#!/usr/bin/env python
"""Amortized net: galaxy peculiar-velocity observable  ->  matter density field.

Trains the deterministic regressor  (n_gal, v_los, coords) -> asinh(delta_m)
under L2 (posterior mean). Adapted from CIRCLE's wp2_train_cnn.py (periodic 3D
CNN in JAX, hand-written Adam), extended to a multi-channel input.

Input channels (per cell):
  0  n_gal        galaxy number field (CIC)          [biased density tracer]
  1  v_los        LOS peculiar-velocity momentum      [the velocity observable]
  2  c_x          normalized position (x-centre)/(L/2)   } LOS geometry, so a
  3  c_y          normalized position (y-centre)/(L/2)   } translation-equivariant
  4  c_z          normalized position (z-centre)/(L/2)   } CNN can use the observer

Target:
  asinh(delta_m)   compressive transform of the present matter overdensity
                   (bounds the halo-core dynamic range for stable L2 training)

Metrics: train/val MSE, reserved-test voxel r and scale-dependent r(k) between
predicted and true asinh(delta_m). The velocity-cosmology claim: large scales
recovered (r(k)->1 at low k).

Run (Slurm): sbatch src/cf4_job.slurm src/cf4_train_cnn.py --data data_train --N 64 --epochs 120
"""
import os
os.environ["JAX_ENABLE_X64"] = "0"   # force float32 training (override slurm's x64=1)
import glob
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

IN_CH = 5
CHANNELS = ("n_gal", "vlos")   # stored channels; coords appended at load


def load_split(data_dir, split):
    files = sorted(glob.glob(os.path.join(data_dir, split, "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards in {os.path.join(data_dir, split)}")
    ng, vl, dm = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        ng.append(z["n_gal"]); vl.append(z["vlos"]); dm.append(z["delta_m"])
    return (np.concatenate(ng).astype(np.float32),
            np.concatenate(vl).astype(np.float32),
            np.concatenate(dm).astype(np.float32))


def coord_channels(N):
    """3 constant channels: normalized position relative to box centre, in [-1,1)."""
    c = (np.arange(N, dtype=np.float32) + 0.5) / N * 2 - 1.0
    CX, CY, CZ = np.meshgrid(c, c, c, indexing="ij")
    return np.stack([CX, CY, CZ], axis=0)   # (3,N,N,N)


def build_inputs(ng, vl, N, norm):
    """(n_samples, 5, N,N,N) input tensor with per-channel normalization + coords."""
    (ng_mu, ng_sd, vl_mu, vl_sd) = norm
    n = ng.shape[0]
    coords = coord_channels(N)[None]                      # (1,3,N,N,N)
    coords = np.broadcast_to(coords, (n, 3, N, N, N))
    a = ((ng - ng_mu) / ng_sd)[:, None]
    b = ((vl - vl_mu) / vl_sd)[:, None]
    return np.concatenate([a, b, coords], axis=1).astype(np.float32)


_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def _iso3(f, p, flips):
    """Apply a cube isometry (axis perm + flips) to a (N,N,N) scalar field."""
    f = np.transpose(f, p)
    for ax in flips:
        f = np.flip(f, ax)
    return f


def augment_batch(Xb, Yb, rng):
    """Octahedral augmentation about the box centre (label-preserving).

    Applies a random cube isometry to the PHYSICAL channels (0=n_gal, 1=vlos) and
    the target, leaving the coordinate channels (2,3,4) standard. Valid because the
    data is isotropic about the centre and rotating vlos as a scalar equals the
    radial velocity of the rotated velocity field (v.rhat invariant under g).
    """
    Xo = np.array(Xb); Yo = np.array(Yb)
    for i in range(Xb.shape[0]):
        p = _PERMS[rng.integers(6)]
        flips = [ax for ax in range(3) if rng.random() < 0.5]
        for c in (0, 1):
            Xo[i, c] = _iso3(Xo[i, c], p, flips)
        if Yo.ndim == 5:
            Yo[i, 0] = _iso3(Yo[i, 0], p, flips)
        else:
            Yo[i] = _iso3(Yo[i], p, flips)
    return np.ascontiguousarray(Xo), np.ascontiguousarray(Yo)


def init_params(key, in_ch=IN_CH, depth=5, ch=48, ksize=3):
    chans = [in_ch] + [ch] * (depth - 1) + [1]
    params = []
    for i in range(depth):
        cin, cout = chans[i], chans[i + 1]
        key, k = jax.random.split(key)
        fan_in = cin * ksize ** 3
        W = (jax.random.normal(k, (cout, cin, ksize, ksize, ksize))
             * np.sqrt(2.0 / fan_in)).astype(jnp.float32)
        params.append((W, jnp.zeros((cout,), jnp.float32)))
    return params


DN = jax.lax.conv_dimension_numbers((1, 1, 4, 4, 4), (1, 1, 3, 3, 3),
                                    ("NCDHW", "OIDHW", "NCDHW"))


def forward(params, x):
    h = x
    n = len(params)
    for i, (W, b) in enumerate(params):
        h = jnp.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap")
        h = jax.lax.conv_general_dilated(h, W, (1, 1, 1), "VALID", dimension_numbers=DN)
        h = h + b[None, :, None, None, None]
        if i < n - 1:
            h = jax.nn.relu(h)
    return h


def mse(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)


def batched_mse(params, X, Y, bs):
    tot, n = 0.0, X.shape[0]
    for i in range(0, n, bs):
        tot += float(mse(params, jnp.asarray(X[i:i + bs]), jnp.asarray(Y[i:i + bs]))) \
            * (min(i + bs, n) - i)
    return tot / n


def batched_forward(params, X, bs):
    return np.concatenate([np.asarray(forward(params, jnp.asarray(X[i:i + bs])))
                           for i in range(0, X.shape[0], bs)], 0)


@jax.jit
def adam_step(params, m, v, x, y, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    loss, grads = jax.value_and_grad(mse)(params, x, y)
    new_p, new_m, new_v = [], [], []
    for (W, b), (gW, gb), (mW, mb), (vW, vb) in zip(params, grads, m, v):
        nmW = b1 * mW + (1 - b1) * gW; nvW = b2 * vW + (1 - b2) * gW ** 2
        nmb = b1 * mb + (1 - b1) * gb; nvb = b2 * vb + (1 - b2) * gb ** 2
        mhW = nmW / (1 - b1 ** t); vhW = nvW / (1 - b2 ** t)
        mhb = nmb / (1 - b1 ** t); vhb = nvb / (1 - b2 ** t)
        new_p.append((W - lr * mhW / (jnp.sqrt(vhW) + eps),
                      b - lr * mhb / (jnp.sqrt(vhb) + eps)))
        new_m.append((nmW, nmb)); new_v.append((nvW, nvb))
    return new_p, new_m, new_v, loss


def rk_curve(strue, spred, box):
    n = strue.shape[-1]; dx = box / n
    kx = 2 * np.pi * np.fft.fftfreq(n, d=dx); kz = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    kk = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    kf = 2 * np.pi / box; kny = np.pi / dx
    bins = np.arange(0.5 * kf, kny, kf); m = kk > 0
    idx = np.digitize(kk[m], bins)
    Xc = np.zeros(len(bins)); Pt = np.zeros(len(bins)); Pp = np.zeros(len(bins))
    cnt = np.zeros(len(bins))
    for a, bb in zip(strue, spred):
        ta = np.fft.rfftn(a); pb = np.fft.rfftn(bb)
        cross = np.real(ta * np.conj(pb))[m]; pt = (np.abs(ta) ** 2)[m]; pp = (np.abs(pb) ** 2)[m]
        for j in range(1, len(bins)):
            sel = idx == j
            if sel.any():
                Xc[j] += cross[sel].sum(); Pt[j] += pt[sel].sum()
                Pp[j] += pp[sel].sum(); cnt[j] += sel.sum()
    good = cnt > 0; kc = 0.5 * (bins[:-1] + bins[1:])
    r = np.full(len(bins), np.nan); r[good] = Xc[good] / np.sqrt(Pt[good] * Pp[good] + 1e-30)
    return kc, r[1:len(kc) + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_train")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--ablation", choices=["full", "novel", "nogal"], default="full",
                    help="full=all; novel=zero velocity ch; nogal=zero n_gal ch")
    ap.add_argument("--augment", action="store_true", help="octahedral augmentation")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    print(f"[env] jax {jax.__version__} dev={jax.devices()[0]}")
    box = args.N * args.spacing
    tf = lambda d: np.arcsinh(d).astype(np.float32)   # compressive target transform

    ng_tr, vl_tr, dm_tr = load_split(args.data, "train")
    ng_va, vl_va, dm_va = load_split(args.data, "val")
    ng_te, vl_te, dm_te = load_split(args.data, "test")

    norm = (ng_tr.mean(), ng_tr.std() + 1e-8, vl_tr.mean(), vl_tr.std() + 1e-8)
    Xtr = build_inputs(ng_tr, vl_tr, args.N, norm)
    Xva = build_inputs(ng_va, vl_va, args.N, norm)
    Xte = build_inputs(ng_te, vl_te, args.N, norm)
    if args.ablation == "novel":       # ablate velocity information
        Xtr[:, 1] = 0; Xva[:, 1] = 0; Xte[:, 1] = 0
    elif args.ablation == "nogal":     # ablate galaxy-count (density) information
        Xtr[:, 0] = 0; Xva[:, 0] = 0; Xte[:, 0] = 0

    ytr = tf(dm_tr); yva = tf(dm_va); yte = tf(dm_te)
    ymu, ysd = ytr.mean(), ytr.std() + 1e-8
    Ytr = ((ytr - ymu) / ysd)[:, None]
    Yva = ((yva - ymu) / ysd)[:, None]
    Yte_true = ((yte - ymu) / ysd)                      # normalized target (test)
    base_va = float(np.mean(((yva - ymu) / ysd) ** 2))  # baseline = predict mean (0)
    print(f"[data] train {Xtr.shape} val {Xva.shape} test {Xte.shape} | "
          f"ablation={args.ablation} | baseline val MSE={base_va:.4f}")

    key = jax.random.PRNGKey(0)
    params = init_params(key, in_ch=IN_CH, depth=args.depth, ch=args.ch)
    m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    v = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    npar = sum(int(W.size + b.size) for W, b in params)
    print(f"[model] periodic 3D CNN depth={args.depth} ch={args.ch} in={IN_CH} "
          f"-> {npar/1e3:.1f}k params")

    ntr = Xtr.shape[0]; t = 0; best = 1e9; best_ep = 0
    best_params = [(jnp.array(W), jnp.array(b)) for W, b in params]
    t0 = time.time()
    rng = np.random.default_rng(0)
    for ep in range(1, args.epochs + 1):
        lr_ep = args.lr * (0.5 ** (ep // 50))     # step LR decay every 50 epochs
        perm = rng.permutation(ntr); tl = 0.0
        for i in range(0, ntr, args.batch):
            idx = perm[i:i + args.batch]; t += 1
            xb, yb = Xtr[idx], Ytr[idx]
            if args.augment:
                xb, yb = augment_batch(xb, yb, rng)
            params, m, v, loss = adam_step(params, m, v,
                                           jnp.asarray(xb), jnp.asarray(yb),
                                           t, lr_ep)
            tl += float(loss) * len(idx)
        tl /= ntr
        vl = batched_mse(params, Xva, Yva, args.batch)
        if vl < best:                              # keep BEST params, not final
            best = vl; best_ep = ep
            best_params = [(jnp.array(W), jnp.array(b)) for W, b in params]
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"[ep {ep:3d}] train_mse={tl:.4f} val_mse={vl:.4f} "
                  f"varexp={1-vl/base_va:.3f} lr={lr_ep:.1e} ({time.time()-t0:.0f}s)", flush=True)

    params = best_params                            # evaluate the best checkpoint
    print(f"[select] best val_mse={best:.4f} at epoch {best_ep}")
    pred = batched_forward(params, Xte, args.batch)[:, 0]
    r_vox = float(np.corrcoef(Yte_true.ravel(), pred.ravel())[0, 1])
    kc, rk = rk_curve(Yte_true, pred, box)
    print("\n================ reserved TEST ================")
    print(f"ablation={args.ablation}  voxel r = {r_vox:.3f}  (n_test={len(pred)})")
    print(f"{'k[h/Mpc]':>9} {'r(k)':>7}")
    for i in range(len(kc)):
        if np.isfinite(rk[i]):
            print(f"{kc[i]:9.4f} {rk[i]:7.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].imshow(Yte_true[0].mean(0), cmap="magma"); ax[0].set_title("true asinh(delta_m)")
    ax[1].imshow(pred[0].mean(0), cmap="magma"); ax[1].set_title("predicted")
    ax[2].axhline(0, ls=":", c="k"); ax[2].axhline(1, ls=":", c="k")
    ax[2].semilogx(kc, rk, "o-", ms=4)
    ax[2].set_xlabel("k [h/Mpc]"); ax[2].set_ylabel("cross-corr r(k)")
    ax[2].set_ylim(-0.1, 1.05)
    ax[2].set_title(f"recovery (voxel r={r_vox:.3f}, {args.ablation})")
    fig.tight_layout()
    out = os.path.join(args.out, f"cf4_cnn_{args.ablation}.png")
    fig.savefig(out, dpi=120)
    np.savez(os.path.join(args.out, f"cf4_cnn_pred_{args.ablation}.npz"),
             true=Yte_true, pred=pred, kc=kc, rk=rk, r_vox=r_vox, best_val=best)
    print(f"\n[fig] saved {out}\n[done] best val_mse={best:.4f} varexp={1-best/base_va:.3f}")


if __name__ == "__main__":
    main()
