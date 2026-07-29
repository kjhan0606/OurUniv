#!/usr/bin/env python
"""Publication figures for the AI4CF4 paper (JCAP style).

Generates (into overleaf_cf4/figs/):
  fig_ic_density.pdf     the reconstruction in field space: observed CF4 galaxies+v_los ->
                         recovered INITIAL density delta(a_start) -> forward z=0 density ->
                         z=0 with known clusters. [needs pmwd]
  fig_diffusion.pdf      the amortized conditional-diffusion process: input observable +
                         reverse DDPM chain s_T -> s_0 conditioned on it, vs the true s. [JAX]
  fig_rk.pdf             amortized vs explicit-MAP r(k) + power-completion transfer function.
  fig_scorecard.pdf      density-at-cluster (constrained) vs random baseline.
Run on GPU: sbatch src/cf4_job.slurm src/cf4_paper_figs.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "overleaf_cf4", "figs")
REC = os.path.join(ROOT, "recon")
os.makedirs(FIG, exist_ok=True)

CLUSTERS = {"Virgo": (102.9, -2.3, 16.0), "Coma": (89.0, 8.0, 90.0),
            "Centaurus": (156.0, -11.0, 45.0), "Norma/GA": (155.0, -6.0, 65.0),
            "Perseus": (340.0, -13.0, 73.0), "Fornax": (236.0, -44.0, 19.0)}


def sg(sgl, sgb, d):
    l, b = np.radians(sgl), np.radians(sgb)
    return d * np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def smooth(field, N, sp, R):
    kf = np.fft.fftfreq(N, d=sp)*2*np.pi; kr = np.fft.rfftfreq(N, d=sp)*2*np.pi
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    W = np.exp(-0.5*(KX**2+KY**2+KZ**2)*R**2)
    return np.fft.irfftn(np.fft.rfftn(field)*W, s=(N,)*3, axes=(0, 1, 2))


def fig_ic_density():
    import jax, jax.numpy as jnp
    from pmwd import (Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody, scatter)
    P = np.load(os.path.join(REC, "cf4_map_cf4_real192.npz"))
    s_out = P["s_out"].astype(np.float64); s_map = P["s_map"].astype(np.float64)
    N = int(P["N"]); spc = float(P["spacing"]); L = N*spc
    zc = np.load(os.path.join(ROOT, "data", "cf4_clean.npz")); h = float(zc["H0"])/100
    conf = Configuration(ptcl_spacing=float(spc), ptcl_grid_shape=(N,)*3, mesh_shape=1,
                         float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=0.31, h=h, A_s_1e9=1.63), conf)
    # initial density delta(a_start) from the constrained mean (clean structure)
    a0 = 0.02
    dic = np.array(linear_modes(jnp.asarray(s_map.reshape(N, N, N)), cosmo, conf,
                                a=a0, real=True), np.float64)
    dic = smooth(dic - dic.mean(), N, spc, 4.0)
    # z=0 density from the full-power realization
    lin = linear_modes(jnp.asarray(s_out.reshape(N, N, N)), cosmo, conf)
    ptcl, o = lpt(lin, cosmo, conf); ptcl, o = nbody(ptcl, o, cosmo, conf)
    d0 = np.asarray(scatter(ptcl, conf), np.float64) - 1.0
    d0s = smooth(d0, N, spc, 5.0)
    # constrained z=0 (s_map) for the clean structure panel
    linm = linear_modes(jnp.asarray(s_map.reshape(N, N, N)), cosmo, conf)
    pm, om = lpt(linm, cosmo, conf); pm, om = nbody(pm, om, cosmo, conf)
    dc = smooth(np.asarray(scatter(pm, conf), np.float64) - 1.0, N, spc, 5.0)

    c = L/2; k0 = N//2
    ext = [-c, L-c, -c, L-c]
    def slab(f): return f[:, :, k0-1:k0+2].mean(2).T

    fig, ax = plt.subplots(1, 4, figsize=(19, 5.0))
    # (a) observed galaxies + v_los
    gp = zc["pos_dist"]*h; vp = zc["vpec"]
    sel = np.abs(gp[:, 2]) < 8
    sc0 = ax[0].scatter(gp[sel, 0], gp[sel, 1], c=np.clip(vp[sel], -1500, 1500), s=5,
                        cmap="coolwarm", lw=0)
    ax[0].plot(0, 0, "k+", ms=10, mew=2)
    ax[0].set_title("(a) CF4 observed: galaxies + $v_{\\rm los}$")
    plt.colorbar(sc0, ax=ax[0], fraction=0.046, label="$v_{\\rm los}$ [km/s]")
    # (b) recovered INITIAL density delta(a_start)
    v = np.percentile(np.abs(slab(dic)), 99)
    im1 = ax[1].imshow(slab(dic), origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[1].set_title("(b) recovered IC: $\\delta(a_{\\rm start})$")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    # (c) constrained present density
    im2 = ax[2].imshow(np.arcsinh(4*slab(dc)), origin="lower", extent=ext, cmap="magma")
    ax[2].set_title("(c) constrained $z{=}0$ density")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, label="$\\sinh^{-1}(4\\delta)$")
    # (d) z=0 full realization + clusters
    im3 = ax[3].imshow(np.arcsinh(4*slab(d0s)), origin="lower", extent=ext, cmap="magma")
    for nm, (sgl, sgb, dd) in CLUSTERS.items():
        x = sg(sgl, sgb, dd)*h
        if abs(x[2]) < 40 and max(abs(x[0]), abs(x[1])) < c-5:
            ax[3].plot(x[0], x[1], "c*", ms=11, mec="w")
            ax[3].annotate(nm, (x[0], x[1]), color="cyan", fontsize=8,
                           xytext=(4, 4), textcoords="offset points")
    ax[3].set_title("(d) $z{=}0$ realization + clusters")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)
    for a in ax:
        a.set_xlim(-120, 120); a.set_ylim(-120, 120)
        a.set_xlabel("SGX [$h^{-1}$Mpc]")
    ax[0].set_ylabel("SGY [$h^{-1}$Mpc]")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ic_density.pdf")); plt.close(fig)
    print("[fig] fig_ic_density.pdf", flush=True)


def fig_diffusion():
    """Amortized conditional-diffusion reverse chain on an N=64 test object."""
    import jax, jax.numpy as jnp, equinox as eqx, pickle
    os.environ["JAX_ENABLE_X64"] = "0"
    import importlib
    import cf4_train_diffusion as M
    ck = os.path.join(REC, "cf4_diff_n64_full_last.eqx")
    with open(ck, "rb") as fh:
        meta = pickle.load(fh); N = meta["N"]; nc = meta["n_cond"]
        model = M.UNet3D(nc, C=meta["C"], key=jax.random.PRNGKey(0))
        model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x)
                                       and jnp.issubdtype(x.dtype, jnp.floating) else x, model)
        model = eqx.tree_deserialise_leaves(fh, model)
    T = meta["T"]; betas, alphas, abars = M.make_schedule(T)
    import glob
    f = sorted(glob.glob(os.path.join(ROOT, "data_train", "test", "shard_*.npz")))[0]
    Strue, obs = M.shard_arrays(f, N, meta["ablation"])
    obs = M.standardize(obs, meta["stats"])
    o = 3
    cond = jnp.asarray(obs[o])

    @eqx.filter_jit
    def step(model, S, obs, t, key):
        eps = jax.vmap(lambda s: model(s, obs, t.astype(jnp.float32)/T))(S)
        ab = abars[t]; a = alphas[t]; b = betas[t]
        mean = (S - b/jnp.sqrt(1-ab)*eps)/jnp.sqrt(a)
        return mean + jnp.where(t > 0, jnp.sqrt(b), 0.0)*jax.random.normal(key, S.shape)
    key = jax.random.PRNGKey(0)
    S = jax.random.normal(key, (1, N, N, N))
    snaps = {T: np.asarray(S[0])}
    want = [int(0.75*T), int(0.5*T), int(0.25*T), int(0.1*T), 0]
    for t in range(T-1, -1, -1):
        key, k = jax.random.split(key)
        S = step(model, S, cond, jnp.array(t), k)
        if t in want:
            snaps[t] = np.asarray(S[0])
    k0 = N//2
    order = [T, int(0.75*T), int(0.5*T), int(0.25*T), 0]
    fig, ax = plt.subplots(2, 5, figsize=(17, 7))
    # top row: input observable channels + true s
    ng = obs[o, 0]; vl = obs[o, 1]
    def sl(f): return f[:, :, k0-1:k0+2].mean(2).T
    for a, f, t in zip(ax[0], [ng, vl, Strue[o], snaps[0], snaps[0]-Strue[o]],
                       ["cond: $n_{\\rm gal}$", "cond: $v_{\\rm los}$", "true $s$",
                        "recovered $\\hat s$ ($t{=}0$)", "residual"]):
        vv = np.percentile(np.abs(sl(f)), 98) + 1e-6
        a.imshow(sl(f), origin="lower", cmap="RdBu_r", vmin=-vv, vmax=vv)
        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])
    # bottom row: reverse diffusion chain
    for a, t in zip(ax[1], order):
        f = snaps[t]; vv = np.percentile(np.abs(sl(f)), 98) + 1e-6
        a.imshow(sl(f), origin="lower", cmap="RdBu_r", vmin=-vv, vmax=vv)
        a.set_title(f"$s_t$, $t/T={t/T:.2f}$", fontsize=10); a.set_xticks([]); a.set_yticks([])
    ax[0][0].set_ylabel("inputs / target", fontsize=10)
    ax[1][0].set_ylabel("reverse diffusion", fontsize=10)
    fig.suptitle("Amortized conditional diffusion: $p(s\\,|\\,n_{\\rm gal},v_{\\rm los})$ "
                 "reverse chain (bottom) from noise to the recovered initial field", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "fig_diffusion.pdf")); plt.close(fig)
    os.environ["JAX_ENABLE_X64"] = "1"
    print("[fig] fig_diffusion.pdf", flush=True)


def fig_rk():
    amo = np.load(os.path.join(REC, "cf4_diff_n64_full_last.npz"))["rk"]
    amo_m = np.nanmean(amo, 0)
    mp64 = np.load(os.path.join(REC, "cf4_map_map64_kstar.npz"))
    mp192 = np.load(os.path.join(REC, "cf4_map_map192_kstar.npz"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    k64 = (np.arange(len(amo_m))+0.5)/64
    ax[0].plot(k64, amo_m, "o-", ms=3, label="amortized diffusion (N=64)", color="C0")
    ax[0].plot(k64, mp64["rk_map"], "s-", ms=3, label="explicit MAP (N=64)", color="C3")
    k192 = (np.arange(len(mp192["rk_map"]))+0.5)/192
    ax[0].plot(k192, mp192["rk_map"], "^-", ms=2, label="explicit MAP (N=192)", color="C2")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlim(0, 0.5); ax[0].set_ylim(-0.05, 1.02)
    ax[0].set_xlabel("$k$ [cell$^{-1}$]"); ax[0].set_ylabel("$r(k)$: recovered vs true $s$")
    ax[0].set_title("(a) initial-field recovery"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    # transfer function / power-completion (from map192)
    smap = mp192["s_map"].astype(np.float64); strue = mp192["s_true"].astype(np.float64)
    N = smap.shape[0]
    kf = np.fft.fftfreq(N); kr = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij"); km = np.sqrt(KX**2+KY**2+KZ**2)
    nb = N//2; idx = np.clip(np.digitize(km.ravel(), np.linspace(0, .5, nb+1))-1, 0, nb-1)
    Pm = np.abs(np.fft.rfftn(smap))**2; Pt = np.abs(np.fft.rfftn(strue))**2
    T2 = np.array([Pm.ravel()[idx == b].mean()/Pt.ravel()[idx == b].mean() if (idx == b).any()
                   else np.nan for b in range(nb)])
    kk = (np.arange(nb)+0.5)/N
    ax[1].plot(kk, np.sqrt(np.clip(T2, 0, None)), "-", color="C4", label="$T(k)=\\sqrt{P_{\\rm MAP}/P_{\\rm prior}}$")
    ax[1].plot(kk, mp192["rk_map"], "--", color="C2", label="$r_{\\rm MAP}(k)$")
    ax[1].axhline(1, color="k", lw=0.5, ls=":")
    ax[1].set_xlim(0, 0.5); ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("$k$ [cell$^{-1}$]"); ax[1].set_ylabel("amplitude")
    ax[1].set_title("(b) MAP transfer function (power-completion)"); ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rk.pdf")); plt.close(fig)
    print("[fig] fig_rk.pdf", flush=True)


def fig_scorecard():
    # values from the density-at-cluster run
    dat = {"Virgo": 27.4, "Perseus": 8.3, "Centaurus": 5.05, "Fornax": 3.12,
           "Coma": 2.64, "Norma/GA": -0.86, "Hydra": -3.36}
    names = list(dat.keys()); vals = [dat[n] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = ["C2" if v > 0 else "C3" for v in vals]
    ax[0].bar(names, vals, color=colors)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axhline(0.12, color="0.4", ls="--", lw=1, label="random baseline")
    ax[0].fill_between([-.5, len(names)-.5], -0.35, 0.35, color="0.7", alpha=0.4)
    ax[0].set_ylabel("$\\delta$ at cluster [$\\sigma$]")
    ax[0].set_title("(a) constrained density at known clusters")
    ax[0].tick_params(axis="x", rotation=45); ax[0].legend(fontsize=9)
    # discriminator
    ax[1].bar(["reconstruction", "random\n(mean$\\pm\\sigma$)"], [6.04, 0.12],
              yerr=[0, 0.35], color=["C2", "0.6"], capsize=5)
    ax[1].set_ylabel("mean $\\delta$ at 7 clusters [$\\sigma$]")
    ax[1].set_title("(b) constrained vs random: $16.8\\sigma$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_scorecard.pdf")); plt.close(fig)
    print("[fig] fig_scorecard.pdf", flush=True)


if __name__ == "__main__":
    for fn in (fig_scorecard, fig_rk, fig_ic_density, fig_diffusion):
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[fig] {fn.__name__} FAILED: {str(e)[:120]}", flush=True)
