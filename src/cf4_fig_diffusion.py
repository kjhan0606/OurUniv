#!/usr/bin/env python
"""Standalone (float32) figure: the amortized conditional-diffusion reverse chain.
Must force JAX_ENABLE_X64=0 BEFORE importing jax (the diffusion net trains in float32)."""
import os
os.environ["JAX_ENABLE_X64"] = "0"
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
import jax, jax.numpy as jnp, equinox as eqx
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
import cf4_train_diffusion as M

FIG = os.path.join(ROOT, "overleaf_cf4", "figs"); REC = os.path.join(ROOT, "recon")

ck = os.path.join(REC, "cf4_diff_n64_full_last.eqx")
with open(ck, "rb") as fh:
    meta = pickle.load(fh); N = meta["N"]; nc = meta["n_cond"]
    model = M.UNet3D(nc, C=meta["C"], key=jax.random.PRNGKey(0))
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x)
                                   and jnp.issubdtype(x.dtype, jnp.floating) else x, model)
    model = eqx.tree_deserialise_leaves(fh, model)
T = meta["T"]; betas, alphas, abars = M.make_schedule(T)
f = sorted(glob.glob(os.path.join(ROOT, "data_train", "test", "shard_*.npz")))[0]
Strue, obs = M.shard_arrays(f, N, meta["ablation"]); obs = M.standardize(obs, meta["stats"])
o = 3; cond = jnp.asarray(obs[o])


@eqx.filter_jit
def step(model, S, obs, t, key):
    eps = jax.vmap(lambda s: model(s, obs, t.astype(jnp.float32) / T))(S)
    ab = abars[t]; a = alphas[t]; b = betas[t]
    mean = (S - b / jnp.sqrt(1 - ab) * eps) / jnp.sqrt(a)
    return mean + jnp.where(t > 0, jnp.sqrt(b), 0.0) * jax.random.normal(key, S.shape)


key = jax.random.PRNGKey(0)
S = jax.random.normal(key, (1, N, N, N))
snaps = {T: np.asarray(S[0])}
want = {int(0.75 * T), int(0.5 * T), int(0.25 * T), 0}
for t in range(T - 1, -1, -1):
    key, k = jax.random.split(key)
    S = step(model, S, cond, jnp.array(t), k)
    if t in want:
        snaps[t] = np.asarray(S[0])
k0 = N // 2


from scipy.ndimage import gaussian_filter
SMOOTH = 2.0  # cells


def sl(fld):
    return fld[:, :, k0 - 1:k0 + 2].mean(2).T


def sls(fld):
    # smoothed slab. s is white noise, so the recoverable large-scale phases (r(k) at low k)
    # are invisible under the dominant unconstrained small-scale power in a raw image. We
    # smooth the s-fields to reveal the large-scale structure the model actually recovers.
    return gaussian_filter(fld[:, :, k0 - 1:k0 + 2].mean(2).T, SMOOTH)


def dc(fld):
    # remove the k=0 (DC) mode. The IC mean density is fixed to zero, so the DC mode of s
    # is unphysical and unconstrained; centering makes the structure comparison fair.
    return fld - fld.mean()


order = [T, int(0.75 * T), int(0.5 * T), int(0.25 * T), 0]
fig, ax = plt.subplots(2, 5, figsize=(17, 7))
strue_c = dc(Strue[o]); srec_c = dc(snaps[0])
top = [(obs[o, 0], "conditioning: $n_{\\rm gal}$", False),
       (obs[o, 1], "conditioning: $v_{\\rm los}$", False),
       (strue_c, "true initial $s$ (smoothed)", True),
       (srec_c, "recovered $\\hat s$ (smoothed)", True),
       (srec_c - strue_c, "residual $\\hat s - s$", True)]
for a, (fld, title, iss) in zip(ax[0], top):
    img = sls(fld) if iss else sl(fld)
    vv = np.percentile(np.abs(img), 98) + 1e-6
    a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-vv, vmax=vv)
    a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
for a, t in zip(ax[1], order):
    img = sls(dc(snaps[t])); vv = np.percentile(np.abs(img), 98) + 1e-6
    a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-vv, vmax=vv)
    a.set_title(f"$s_t$, $t/T={t / T:.2f}$", fontsize=10); a.set_xticks([]); a.set_yticks([])
ax[0][0].set_ylabel("inputs / target", fontsize=11)
ax[1][0].set_ylabel("reverse diffusion", fontsize=11)
fig.suptitle("Amortized conditional diffusion $p(s\\,|\\,n_{\\rm gal},v_{\\rm los})$. Reverse "
             "denoising chain (bottom, noise to field) conditioned on the galaxy observable",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "fig_diffusion.pdf")); plt.close(fig)
print("[fig] fig_diffusion.pdf saved")
