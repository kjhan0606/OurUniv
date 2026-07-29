#!/usr/bin/env python
"""CF4 Stage-B: amortized conditional diffusion  observable -> initial field s.

Recovers the whitened INITIAL field s ~ N(0,I) of a full periodic cubic box directly
from the present-day galaxy observable (the CF4 premise: present peculiar-velocity
field -> initial conditions). Adapted from CIRCLE poc/wp2_score.py, whose amortized
DDPM IC-recovery conditions on the CLEAN matter density; here we re-condition on the
REALISTIC galaxy observable our FoF->HOD->v_pec pipeline produces:

  conditioning channels (per cell):
    n_gal   biased galaxy number field (CIC)          [full ablation only]
    vlos    line-of-sight peculiar-velocity field (CIC, observer at box centre)
    r_geom  radial distance from centre / (L/2)        [rotation-invariant geometry]

  target: s  (white-noise initial field; periodic + cubic by construction)

DDPM eps-prediction (Ho et al. 2020): s_t = sqrt(abar) s0 + sqrt(1-abar) eps; the
3D conditional U-Net predicts eps from (s_t, cond, t); loss ||eps-eps_theta||^2.
Reverse ancestral sampling conditioned on a test observable draws posterior samples
p(s | observable) with NO pmwd at inference. Unconstrained small scales relax to the
Gaussian prior (correct: those modes are erased by nonlinear evolution).

Checkpoint selection = pooled low-k basis std(z) on held-out objects (val eps-loss is
a proven non-predictor of posterior calibration; wp2_score note). Octahedral (48-elt)
augmentation only -- observer at box centre breaks translation invariance, so no shifts;
r_geom is rotation-invariant so it is rebuilt, never rotated.

modes:
  train   : train eps-net on data_train/{train,val}; checkpoint .eqx
  sample  : load ckpt, posterior-sample s for test objects; write r(k), z-score, npz
"""
import os
os.environ["JAX_ENABLE_X64"] = "0"          # NN trains/samples in float32

import glob, time, argparse, pickle
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
COND_KEYS = ("n_gal", "vlos", "r_geom")     # conditioning channel order


# ---------------- diffusion schedule ----------------
def make_schedule(T, beta_min=1e-4, beta_max=2e-2):
    betas = np.linspace(beta_min, beta_max, T, dtype=np.float64)
    abars = np.cumprod(1.0 - betas)
    return (jnp.asarray(betas, jnp.float32), jnp.asarray(1.0 - betas, jnp.float32),
            jnp.asarray(abars, jnp.float32))


# ---------------- 3D conditional U-Net (equinox) ----------------
def sinusoidal(t, dim):
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / (half - 1))
    a = t * freqs
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)])


class Block(eqx.Module):
    c1: eqx.nn.Conv3d
    c2: eqx.nn.Conv3d
    n1: eqx.nn.GroupNorm
    n2: eqx.nn.GroupNorm
    temb: eqx.nn.Linear

    def __init__(self, cin, cout, tdim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.c1 = eqx.nn.Conv3d(cin, cout, 3, padding=1, key=k1)
        self.c2 = eqx.nn.Conv3d(cout, cout, 3, padding=1, key=k2)
        g = min(8, cout)
        self.n1 = eqx.nn.GroupNorm(g, cout); self.n2 = eqx.nn.GroupNorm(g, cout)
        self.temb = eqx.nn.Linear(tdim, cout, key=k3)

    def __call__(self, x, t):
        h = jax.nn.silu(self.n1(self.c1(x)))
        h = h + self.temb(t)[:, None, None, None]
        h = jax.nn.silu(self.n2(self.c2(h)))
        return h


class UNet3D(eqx.Module):
    """3-level conditional U-Net. Input = s_t (1 ch) + cond (n_cond ch)."""
    inb: Block
    d1: Block
    d2: Block
    mid: Block
    u2: Block
    u1: Block
    out: eqx.nn.Conv3d
    tmlp1: eqx.nn.Linear
    tmlp2: eqx.nn.Linear
    tdim: int = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(self, n_cond, C=32, tdim=128, key=None, remat=False):
        ks = jax.random.split(key, 9)
        self.tdim = tdim; self.n_cond = n_cond; self.remat = remat
        self.tmlp1 = eqx.nn.Linear(tdim, tdim, key=ks[0])
        self.tmlp2 = eqx.nn.Linear(tdim, tdim, key=ks[1])
        self.inb = Block(1 + n_cond, C, tdim, ks[2])
        self.d1 = Block(C, 2 * C, tdim, ks[3])
        self.d2 = Block(2 * C, 4 * C, tdim, ks[4])
        self.mid = Block(4 * C, 4 * C, tdim, ks[5])
        self.u2 = Block(4 * C + 2 * C, 2 * C, tdim, ks[6])
        self.u1 = Block(2 * C + C, C, tdim, ks[7])
        self.out = eqx.nn.Conv3d(C, 1, 1, key=ks[8])

    def __call__(self, s_t, cond, t):
        # s_t: (N,N,N); cond: (n_cond,N,N,N); t: scalar in [0,1]
        temb = sinusoidal(t, self.tdim)
        temb = jax.nn.silu(self.tmlp1(temb)); temb = self.tmlp2(temb)
        x = jnp.concatenate([s_t[None], cond], 0)          # (1+n_cond,N,N,N)
        # gradient checkpointing: recompute block activations in backward (3-5x less
        # memory) so large grids fit. Statics/closure handled by eqx.filter_checkpoint.
        blk = (lambda b: eqx.filter_checkpoint(b)) if self.remat else (lambda b: b)
        h0 = blk(self.inb)(x, temb)
        h1 = blk(self.d1)(self._down(h0), temb)
        h2 = blk(self.d2)(self._down(h1), temb)
        m = blk(self.mid)(h2, temb)
        u2 = blk(self.u2)(jnp.concatenate([self._up(m), h1], 0), temb)
        u1 = blk(self.u1)(jnp.concatenate([self._up(u2), h0], 0), temb)
        return self.out(u1)[0]

    @staticmethod
    def _down(x):
        return eqx.nn.MaxPool3d(2, 2)(x)

    @staticmethod
    def _up(x):
        c, a, b, d = x.shape
        return jax.image.resize(x, (c, a * 2, b * 2, d * 2), method="nearest")


# ---------------- Fourier basis for calibration z-score ----------------
def build_basis_np(N, P):
    kx = np.fft.fftfreq(N); kz = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    order = np.argsort(np.sqrt(KX**2 + KY**2 + KZ**2).ravel())
    cols = []
    for idx in order[1:]:
        spec = np.zeros(KX.size, dtype=complex); spec[idx] = 1.0
        f = np.fft.irfftn(spec.reshape(KX.shape), s=(N, N, N), axes=(0, 1, 2)).ravel()
        if np.linalg.norm(f) > 1e-8:
            cols.append(f)
        if len(cols) >= P:
            break
    Q, _ = np.linalg.qr(np.stack(cols, 1))
    return Q.T.astype(np.float32)


# ---------------- geometry channel (rotation-invariant) ----------------
def radial_channel(N):
    """r/(L/2) from box centre, in cell units normalized to [0, ~sqrt(3)]. Isotropic
    about the centre -> invariant under the octahedral group (augmentation-safe)."""
    c = (N - 1) / 2.0
    ax = (np.arange(N) - c)
    RX, RY, RZ = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(RX**2 + RY**2 + RZ**2) / (N / 2.0)
    return r.astype(np.float32)


# ---------------- data ----------------
def shard_arrays(f, N, ablation):
    """Load one shard -> (s (n,N,N,N), obs (n,n_cond,N,N,N)) with clean-sample filter.
    obs channels are RAW here (n_gal, vlos, r_geom); standardization applied later."""
    with np.load(f) as z:
        s = z["s"].astype(np.float32).reshape(-1, N, N, N)
        ng = z["n_gal"].astype(np.float32).reshape(-1, N, N, N)
        vl = z["vlos"].astype(np.float32).reshape(-1, N, N, N)
    good = (np.isfinite(s).reshape(s.shape[0], -1).all(1) &
            np.isfinite(ng).reshape(ng.shape[0], -1).all(1) &
            np.isfinite(vl).reshape(vl.shape[0], -1).all(1))
    s, ng, vl = s[good], ng[good], vl[good]
    if ablation == "nogal":
        ng = np.zeros_like(ng)
    r = np.broadcast_to(radial_channel(N), ng.shape)
    obs = np.stack([ng, vl, r], 1)                          # (n,3,N,N,N)
    return s, obs


def standardize(obs, stats):
    """Per-channel (x-mu)/sd in place; stats = list of (mu,sd) per cond channel."""
    for c, (mu, sd) in enumerate(stats):
        obs[:, c] = (obs[:, c] - mu) / sd
    return obs


# ---------------- augmentation (octahedral only) ----------------
def augment_batch(S, OBS, rng):
    """Random cube isometry (perm of axes + reflections) applied identically to s and
    every observable channel. The pmwd forward (isotropic, periodic cubic) commutes
    with axis permutations/reflections about the centre, and r_geom is invariant, so
    augmented (s, obs) are exact draws from the same distribution. No translation
    (observer fixed at box centre breaks it)."""
    n = S.shape[0]
    for i in range(n):
        perm = tuple(rng.permutation(3))
        fl = tuple(ax for ax in range(3) if rng.random() < 0.5)
        def iso(a):                                         # a: (N,N,N)
            if perm != (0, 1, 2):
                a = np.transpose(a, perm)
            if fl:
                a = np.flip(a, fl)
            return a
        S[i] = iso(S[i])
        for c in range(OBS.shape[1]):
            OBS[i, c] = iso(OBS[i, c])
    return S, OBS


# ---------------- train ----------------
def train(args):
    N = args.N
    tr_files = sorted(glob.glob(os.path.join(args.data, "train", "shard_*.npz")))
    assert tr_files, f"no train shards in {args.data}/train"
    n_cond = len(COND_KEYS)

    # one streaming pass: per-channel standardization stats + valid count
    csum = np.zeros(n_cond); csq = np.zeros(n_cond); ccnt = 0; ntr = 0
    for f in tr_files:
        _, obs = shard_arrays(f, N, args.ablation)
        ntr += obs.shape[0]
        for c in range(n_cond):
            d = obs[:, c].astype(np.float64)
            csum[c] += d.sum(); csq[c] += (d**2).sum()
        ccnt += obs[:, 0].size
    stats = [(csum[c]/ccnt, float(np.sqrt(max(csq[c]/ccnt-(csum[c]/ccnt)**2, 1e-12)))+1e-8)
             for c in range(n_cond)]
    print(f"[data] train {ntr} ({len(tr_files)} shards) ablation={args.ablation}", flush=True)
    for c, k in enumerate(COND_KEYS):
        print(f"       cond[{k}] mu={stats[c][0]:.4g} sd={stats[c][1]:.4g}", flush=True)

    # val (load fully; small)
    Sval, OBSval = [], []
    for f in sorted(glob.glob(os.path.join(args.data, "val", "shard_*.npz"))):
        s, o = shard_arrays(f, N, args.ablation); Sval.append(s); OBSval.append(o)
    Sval = np.concatenate(Sval); OBSval = standardize(np.concatenate(OBSval), stats)

    betas, alphas, abars = make_schedule(args.T)
    key = jax.random.PRNGKey(args.seed)
    key, mk = jax.random.split(key)
    model = UNet3D(n_cond, C=args.C, key=mk, remat=bool(args.remat))
    nparam = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"[model] UNet3D C={args.C} n_cond={n_cond} params={nparam/1e6:.2f}M "
          f"remat={bool(args.remat)}", flush=True)
    opt = optax.adam(args.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # calibration probe set (a few val objects; disjoint would be ideal but val works)
    ncal = min(args.calib_objs, Sval.shape[0])
    Ecal = build_basis_np(N, args.calib_P)
    a_true_cal = Sval[:ncal].reshape(ncal, -1) @ Ecal.T
    OBScal = jnp.asarray(OBSval[:ncal])

    def loss_one(model, s0, obs, t_idx, eps):
        ab = abars[t_idx]
        s_t = jnp.sqrt(ab) * s0 + jnp.sqrt(1.0 - ab) * eps
        pred = model(s_t, obs, t_idx.astype(jnp.float32) / args.T)
        return jnp.mean((pred - eps) ** 2)

    @eqx.filter_jit
    def step(model, opt_state, s0b, ob, tb, epsb):
        def bl(m):
            return jnp.mean(jax.vmap(lambda s, o, t, e: loss_one(m, s, o, t, e))(s0b, ob, tb, epsb))
        loss, grads = eqx.filter_value_and_grad(bl)(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss

    @eqx.filter_jit
    def val_loss(model, s0b, ob, tb, epsb):
        return jnp.mean(jax.vmap(lambda s, o, t, e: loss_one(model, s, o, t, e))(s0b, ob, tb, epsb))

    @eqx.filter_jit
    def ddpm_step(model, S, obs, t_idx, key):
        eps = jax.vmap(lambda s: model(s, obs, t_idx.astype(jnp.float32) / args.T))(S)
        ab = abars[t_idx]; a = alphas[t_idx]; b = betas[t_idx]
        mean = (S - b / jnp.sqrt(1.0 - ab) * eps) / jnp.sqrt(a)
        z = jax.random.normal(key, S.shape)
        return mean + jnp.where(t_idx > 0, jnp.sqrt(b), 0.0) * z

    def calib_metric(model, ep):
        zs = []
        for o in range(ncal):
            obs = OBScal[o]
            key = jax.random.PRNGKey(777000 + o)
            key, k0 = jax.random.split(key)
            S = jax.random.normal(k0, (args.calib_samples, N, N, N))
            for t in range(args.T - 1, -1, -1):
                key, kt = jax.random.split(key)
                S = ddpm_step(model, S, obs, jnp.array(t), kt)
            a_samp = np.asarray(S).reshape(args.calib_samples, -1) @ Ecal.T
            zs.append((a_true_cal[o] - a_samp.mean(0)) / (a_samp.std(0) + 1e-8))
        Z = np.array(zs)
        lo, al = float(Z[:, :6].std()), float(Z.std())
        print(f"    [calib ep {ep:3d}] low-k std(z)={lo:.3f} all-band={al:.3f}", flush=True)
        return lo

    def save_ckpt(path, model):
        with open(path, "wb") as fh:
            pickle.dump({"stats": stats, "C": args.C, "T": args.T, "N": N,
                         "n_cond": n_cond, "ablation": args.ablation}, fh)
            eqx.tree_serialise_leaves(fh, model)

    bs = args.batch
    rng = np.random.default_rng(args.seed)
    t0 = time.time(); best = 1e9
    for ep in range(args.epochs):
        tl = 0.0; nb = 0
        for si in rng.permutation(len(tr_files)):
            S_sh, O_sh = shard_arrays(tr_files[si], N, args.ablation)
            O_sh = standardize(O_sh, stats)
            m = S_sh.shape[0]; order = rng.permutation(m)
            for i in range(0, m - bs + 1, bs):
                idx = order[i:i + bs]
                s0np, onp = S_sh[idx].copy(), O_sh[idx].copy()
                if args.augment:
                    s0np, onp = augment_batch(s0np, onp, rng)
                key, k1, k2 = jax.random.split(key, 3)
                tb = jax.random.randint(k1, (bs,), 0, args.T)
                epsb = jax.random.normal(k2, (bs, N, N, N))
                model, opt_state, loss = step(model, opt_state, jnp.asarray(s0np),
                                              jnp.asarray(onp), tb, epsb)
                tl += float(loss); nb += 1
            del S_sh, O_sh
        # val monitor
        key, k1, k2 = jax.random.split(key, 3)
        nv = min(64, Sval.shape[0]); nv = (nv // bs) * bs or bs
        vidx = rng.permutation(Sval.shape[0])[:nv]
        tv = jax.random.randint(k1, (len(vidx),), 0, args.T)
        ev = jax.random.normal(k2, (len(vidx), N, N, N))
        vsum = 0.0; vn = 0
        for j in range(0, len(vidx) - bs + 1, bs):
            sub = vidx[j:j + bs]
            vsum += float(val_loss(model, jnp.asarray(Sval[sub]), jnp.asarray(OBSval[sub]),
                                   tv[j:j+bs], ev[j:j+bs])) * len(sub); vn += len(sub)
        vl = vsum / max(vn, 1)
        print(f"[ep {ep:3d}] train {tl/max(nb,1):.4f} val {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if args.calib_every > 0 and (ep % args.calib_every == args.calib_every - 1
                                     or ep == args.epochs - 1):
            lo = calib_metric(model, ep)
            score = abs(np.log(max(lo, 1e-6)))
            if score < best:
                best = score; save_ckpt(args.ckpt, model)
                print(f"    [ckpt] saved (low-k std(z) {lo:.3f})", flush=True)
        elif args.calib_every <= 0 and vl < best:
            best = vl; save_ckpt(args.ckpt, model)
            print(f"    [ckpt] saved (val {vl:.4f})", flush=True)
        save_ckpt(args.ckpt + ".last", model)
    print(f"[done] best {'|log std(z)|' if args.calib_every>0 else 'val'} {best:.4f}", flush=True)


# ---------------- sample ----------------
def _rk_curve(A, B, N):
    """Cross-correlation r(k) between fields A,B on a cubic grid (radial k bins)."""
    Ak = np.fft.rfftn(A); Bk = np.fft.rfftn(B)
    kf = np.fft.fftfreq(N); kr = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    nb = N // 2
    edges = np.linspace(0, 0.5, nb + 1)
    idx = np.digitize(kmag.ravel(), edges) - 1
    out = np.full(nb, np.nan)
    aa = (np.abs(Ak)**2).ravel(); bb = (np.abs(Bk)**2).ravel()
    ab = np.real(Ak * np.conj(Bk)).ravel()
    for b in range(nb):
        m = idx == b
        if m.sum() > 0:
            den = np.sqrt(aa[m].sum() * bb[m].sum())
            out[b] = ab[m].sum() / den if den > 0 else np.nan
    return out


def sample(args):
    with open(args.ckpt, "rb") as fh:
        meta = pickle.load(fh)
        N = meta["N"]; n_cond = meta["n_cond"]
        model = UNet3D(n_cond, C=meta["C"], key=jax.random.PRNGKey(0))
        model = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32) if eqx.is_array(x)
            and jnp.issubdtype(x.dtype, jnp.floating) else x, model)
        model = eqx.tree_deserialise_leaves(fh, model)
    stats, T = meta["stats"], meta["T"]
    betas, alphas, abars = make_schedule(T)
    tf = sorted(glob.glob(os.path.join(args.data, "test", "shard_*.npz")))
    Stest, OBStest = [], []
    for f in tf:
        s, o = shard_arrays(f, N, meta["ablation"]); Stest.append(s); OBStest.append(o)
    Stest = np.concatenate(Stest); OBStest = standardize(np.concatenate(OBStest), stats)
    o0, o1 = args.obj_start, min(args.obj_start + args.nobj, Stest.shape[0])
    Enp = build_basis_np(N, args.P)

    @eqx.filter_jit
    def ddpm_step(model, S, obs, t_idx, key):
        eps = jax.vmap(lambda s: model(s, obs, t_idx.astype(jnp.float32) / T))(S)
        ab = abars[t_idx]; a = alphas[t_idx]; b = betas[t_idx]
        mean = (S - b / jnp.sqrt(1.0 - ab) * eps) / jnp.sqrt(a)
        z = jax.random.normal(key, S.shape)
        return mean + jnp.where(t_idx > 0, jnp.sqrt(b), 0.0) * z

    @eqx.filter_jit
    def ddim_step(model, S, obs, t_idx, tprev_idx, eta, key):
        eps = jax.vmap(lambda s: model(s, obs, t_idx.astype(jnp.float32) / T))(S)
        ab = abars[t_idx]
        ab_prev = jnp.where(tprev_idx >= 0, abars[jnp.maximum(tprev_idx, 0)], 1.0)
        s0 = (S - jnp.sqrt(1.0 - ab) * eps) / jnp.sqrt(ab)
        sigma = eta * jnp.sqrt((1.0 - ab_prev) / (1.0 - ab)) * jnp.sqrt(1.0 - ab / ab_prev)
        dir_xt = jnp.sqrt(jnp.maximum(1.0 - ab_prev - sigma ** 2, 0.0)) * eps
        z = jax.random.normal(key, S.shape)
        return jnp.sqrt(ab_prev) * s0 + dir_xt + sigma * z

    if args.ddim_steps > 0:                                # DDIM timestep subsequence
        ts = np.linspace(T - 1, 0, args.ddim_steps).round().astype(int)
        tseq = [(int(ts[i]), int(ts[i + 1]) if i + 1 < len(ts) else -1)
                for i in range(len(ts))]
        print(f"[sample] DDIM {args.ddim_steps} steps eta={args.ddim_eta}", flush=True)
    else:
        print(f"[sample] DDPM ancestral {T} steps", flush=True)

    def draw(obs, b, key):
        key, k0 = jax.random.split(key)
        S = jax.random.normal(k0, (b, N, N, N))
        if args.ddim_steps > 0:
            for (t, tp) in tseq:
                key, kt = jax.random.split(key)
                S = ddim_step(model, S, obs, jnp.array(t), jnp.array(tp),
                              jnp.float32(args.ddim_eta), kt)
        else:
            for t in range(T - 1, -1, -1):
                key, kt = jax.random.split(key)
                S = ddpm_step(model, S, obs, jnp.array(t), kt)
        return S, key

    Z, RK, means, samples = [], [], [], []
    for o in range(o0, o1):
        obs = jnp.asarray(OBStest[o])
        key = jax.random.PRNGKey(1000 + o)
        outs = []
        done = 0
        while done < args.n_samples:
            b = min(args.batch, args.n_samples - done)
            S, key = draw(obs, b, key)
            outs.append(np.asarray(S)); done += b
        Sall = np.concatenate(outs, 0)                     # (n_samples,N,N,N)
        smean = Sall.mean(0)
        a_samp = Sall.reshape(Sall.shape[0], -1) @ Enp.T
        a_true = Enp @ Stest[o].ravel()
        zc = (a_true - a_samp.mean(0)) / (a_samp.std(0) + 1e-8)
        rk = _rk_curve(smean, Stest[o], N)
        Z.append(zc); RK.append(rk); means.append(smean.astype(np.float32))
        samples.append(Sall[0].astype(np.float32))         # one realization for GRAFIC/IC use
        print(f"[obj {o}] low-k std(z)={zc[:6].std():.2f} all-std(z)={zc.std():.2f} "
              f"r(k) low={np.nanmean(rk[:4]):.2f} mid={np.nanmean(rk[4:12]):.2f} "
              f"post-mean/true corr(k<0.1)={np.nanmean(rk[:max(1,int(0.1*N))]):.2f}", flush=True)
    RK = np.array(RK); Z = np.array(Z)
    os.makedirs(args.out, exist_ok=True)
    outf = os.path.join(args.out, f"cf4_diff_{args.tag}.npz")
    np.savez(outf, rk=RK, z=Z, post_mean=np.array(means),
             post_sample=np.array(samples), obj=np.arange(o0, o1),
             ablation=meta["ablation"])
    print(f"[out] {outf}  pooled low-k std(z)={Z[:, :6].std():.3f} "
          f"mean r(k) low={np.nanmean(RK[:, :4]):.3f} mid={np.nanmean(RK[:, 4:12]):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "sample"], required=True)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data_train"))
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(HERE), "recon", "cf4_diff.eqx"))
    ap.add_argument("--ablation", choices=["full", "nogal"], default="full")
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--remat", type=int, default=0, help="gradient checkpointing (big grids)")
    ap.add_argument("--calib_every", type=int, default=25)
    ap.add_argument("--calib_objs", type=int, default=4)
    ap.add_argument("--calib_samples", type=int, default=8)
    ap.add_argument("--calib_P", type=int, default=12)
    # sample-mode
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--nobj", type=int, default=8)
    ap.add_argument("--obj_start", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--ddim_steps", type=int, default=0, help="0=full DDPM; else DDIM steps")
    ap.add_argument("--ddim_eta", type=float, default=0.5, help="DDIM stochasticity (1=DDPM-like)")
    ap.add_argument("--tag", default="n64")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "recon"))
    args = ap.parse_args()
    print(f"[env] jax {jax.__version__} eqx {eqx.__version__} dev={jax.devices()[0]} "
          f"mode={args.mode} N={args.N} ablation={args.ablation}", flush=True)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    (train if args.mode == "train" else sample)(args)


if __name__ == "__main__":
    main()
