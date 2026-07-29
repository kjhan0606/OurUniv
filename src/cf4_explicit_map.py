#!/usr/bin/env python
"""CF4 explicit forward-model MAP: fit the initial field s to galaxy peculiar velocities.

The amortized diffusion recovers s only modestly (r~0.4 large-scale) because inverting
nonlinear gravity to the IC is hard. For HIGH-fidelity constrained ICs (lagRAMSES: MW,
M31, M33, Virgo, Coma) we optimize s directly against the data with the exact,
DIFFERENTIABLE pmwd forward -- the BORG/CIRCLE explicit branch.

Likelihood (velocity-based constrained realization; NO FoF/HOD in the loop -- galaxy
positions are fixed DATA, only the velocity FIELD is predicted, so it is differentiable):

    s  --pmwd-->  z=0 matter velocity field v(x)   (CIC momentum / mass)
    v_los_pred(gal) = interp(v, x_gal) . r_hat(gal)          [x_gal fixed, observed]
    -log post(s) = 1/2 sum_g (v_los_pred - v_los_obs)^2 / sigma_v^2  +  1/2 ||s||^2

s ~ N(0,I) white-noise prior => periodic + cubic IC by construction (user constraint).
Optimize with L-BFGS (curvature-aware; CIRCLE wp2_warmstart_opt found Adam does not
exploit a good init, quasi-Newton can). Optional warm start from the amortized s.

Validates on a mock (r(k) of s_MAP vs s_true), then exports s_MAP -> GRAFIC (2LPT).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import time
import numpy as np


def _rk_curve(A, B, N):
    Ak = np.fft.rfftn(A); Bk = np.fft.rfftn(B)
    kf = np.fft.fftfreq(N); kr = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    nb = N // 2
    edges = np.linspace(0, 0.5, nb + 1)
    idx = np.digitize(kmag.ravel(), edges) - 1
    aa = (np.abs(Ak)**2).ravel(); bb = (np.abs(Bk)**2).ravel()
    ab = np.real(Ak * np.conj(Bk)).ravel()
    out = np.full(nb, np.nan)
    for b in range(nb):
        m = idx == b
        if m.sum() > 0:
            den = np.sqrt(aa[m].sum() * bb[m].sum())
            out[b] = ab[m].sum() / den if den > 0 else np.nan
    return out


def power_complete(s_map, N, seed):
    """Constrained realization by POWER-COMPLETION (data-only; no second solve).

    The MAP is a Wiener-suppressed estimate: its power P_MAP(k) = T(k)^2 * P_prior(k)
    with T(k) measurable from the MAP itself (prior for the whitened field is white,
    P_prior = N^3). Keep the MAP's phases and add ONLY the complementary power
    sqrt(1 - T^2) as a fresh white field -> full prior power at every k, and
    r_CR(k) = r_MAP(k) * T(k) (keeps the constraint where constrained, random else).
    Much better than Hoffman-Ribak here (HR is drowned by the MAP's amplitude shrink)."""
    kf = np.fft.fftfreq(N); kr = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    nb = N // 2; edges = np.linspace(0, 0.5, nb + 1)
    idx = np.clip(np.digitize(kmag.ravel(), edges) - 1, 0, nb - 1)
    P_prior = float(N ** 3)
    smap_k = np.fft.rfftn(s_map)
    v = (np.abs(smap_k) ** 2).ravel()
    T2 = np.array([v[idx == b].mean() if (idx == b).any() else 0.0 for b in range(nb)])
    T2 = np.clip(T2 / P_prior, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    xi_k = np.fft.rfftn(rng.standard_normal((N, N, N)))
    filt = np.sqrt(np.maximum(1.0 - T2, 0.0))[idx].reshape(kmag.shape)
    return s_map + np.fft.irfftn(xi_k * filt, s=(N, N, N), axes=(0, 1, 2))


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mock_pipeline import run_sample, VUNIT_KMS

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=2000000, help="mock seed (test namespace)")
    ap.add_argument("--real-npz", default=None,
                    help="real CF4 catalog (cf4_clean.npz): fit observed v_pec instead of a mock")
    ap.add_argument("--h", type=float, default=0.7, help="little-h (real data: 0.746)")
    ap.add_argument("--vpec-max", type=float, default=3000.0, help="drop |v_pec|>this (km/s)")
    ap.add_argument("--sig-floor", type=float, default=50.0, help="floor on sigma_v (km/s)")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction of galaxies withheld from the fit (saved for VELMOD validation)")
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--A-s-1e9", type=float, default=2.0,
                    help="primordial amplitude; 2.0->sigma8~0.90 (pmwd default), "
                         "1.63->sigma8~0.81 (Planck; use for real data)")
    ap.add_argument("--sigma-v", type=float, default=150.0, help="velocity noise [km/s]")
    ap.add_argument("--vnoise-frac", type=float, default=0.0,
                    help="distance-error velocity noise coef (0.2 -> CF4-like); adds to data")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--chi2-target", type=float, default=1.2,
                    help="discrepancy-principle stop: freeze s when chi2/N <= this")
    ap.add_argument("--pc-seed", type=int, default=42,
                    help=">=0: power-completion constrained realization (default; full power, "
                         "keeps MAP phases; set -1 to disable and export the bare MAP)")
    ap.add_argument("--cr-seed", type=int, default=-1,
                    help=">=0: Hoffman-Ribak instead of power-completion (comparison only)")
    ap.add_argument("--delta-constraints", action="store_true",
                    help="add explicit linear-density constraints at known cluster / Local-Void "
                         "positions -> pins structure beyond the velocity-constrained ~15 h^-1Mpc")
    ap.add_argument("--dc-target", type=float, default=2.0,
                    help="target linear delta_L(R) at each named cluster")
    ap.add_argument("--dc-void", type=float, default=-0.6,
                    help="target linear delta_L(R) at the Local Void (anti-Virgo)")
    ap.add_argument("--dc-sigma", type=float, default=1.0,
                    help="constraint sigma (loose = a nudge, not a hard pin)")
    ap.add_argument("--dc-R", type=float, default=5.0,
                    help="Gaussian smoothing radius for the density constraint [h^-1Mpc]")
    ap.add_argument("--load-smap-npz", default=None,
                    help="reuse a saved posterior-mean s_map (skip the obs MAP solve) -> fast "
                         "cr_seed generation: only the HR mock solve runs (K*=--iters)")
    ap.add_argument("--warm-npz", default=None, help="amortized npz for warm start")
    ap.add_argument("--warm-key", default="post_sample")
    ap.add_argument("--warm-obj", type=int, default=0)
    ap.add_argument("--grafic-out", default=None, help="write s_MAP -> GRAFIC dir")
    ap.add_argument("--astart", type=float, default=0.02)
    ap.add_argument("--tag", default="map64")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recon"))
    args = ap.parse_args()

    import jax, jax.numpy as jnp, optax
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody
    print(f"[env] jax {jax.__version__} dev={jax.devices()[0]} N={args.N}", flush=True)
    N, sp = args.N, args.spacing
    L = N * sp

    hh = args.h
    if args.real_npz:
        # --- REAL CF4 catalog: fixed observed positions + peculiar velocities ---
        z = np.load(args.real_npz)
        hh = float(z["H0"]) / 100.0
        pos_mpch = z["pos_dist"].astype(np.float64) * hh    # Mpc -> Mpc/h (observer at origin)
        observer = np.array([L / 2.0] * 3)
        gal_all = pos_mpch + observer                       # box frame, observer at centre
        r = np.linalg.norm(pos_mpch, axis=1)
        vpec = z["vpec"].astype(np.float64)
        sig = z["sig_v"].astype(np.float64)
        # robust cuts: inside box; drop unphysical |v_pec|>3000 (bad-distance outliers
        # dominate chi2); floor sigma_v (a few tiny-error points must not dominate).
        keep = ((r > 1e-6) & (r < 0.98 * L / 2.0)
                & (np.abs(vpec) < args.vpec_max))
        gal = gal_all[keep]; r = r[keep]
        rhat = z["nhat"].astype(np.float64)[keep]
        v_obs = vpec[keep]                                  # km/s LOS peculiar velocity
        sig = np.maximum(sig[keep], args.sig_floor)         # sigma_v floor
        s_true = None
        chi2_0 = float(np.mean((v_obs / sig) ** 2))
        print(f"[data] REAL CF4: {gal.shape[0]}/{len(keep)} groups (|v_pec|<{args.vpec_max}, "
              f"sig>={args.sig_floor}) | h={hh:.3f} v_pec rms={v_obs.std():.0f} "
              f"sig_v med={np.median(sig):.0f} km/s S/N med={np.median(np.abs(v_obs)/sig):.2f} "
              f"chi2(s=0)/N={chi2_0:.2f}", flush=True)
    else:
        # --- mock data (fixed galaxy positions + observed LOS velocities) ---
        hod = dict(logMmin=12.8, sigma_logM=0.4, logM0=12.3, logM1=13.9, alpha=1.0)
        rec = run_sample(N, sp, args.seed, args.Om, hod, verbose=True)
        s_true = rec["s"].astype(np.float64)
        gal = rec["gal_pos"].astype(np.float64)             # [Mpc/h], fixed
        observer = np.array([L / 2.0] * 3)
        d = gal - observer
        r = np.linalg.norm(d, axis=1)
        keep = r > 1e-6
        gal, d, r = gal[keep], d[keep], r[keep]
        rhat = d / r[:, None]
        v_obs = rec["gal_vpec"].astype(np.float64)[keep]    # km/s (LOS)
        rng = np.random.default_rng(args.seed)
        sig = np.sqrt((args.vnoise_frac * 100.0 * r) ** 2 + args.sigma_v ** 2)
        v_obs = v_obs + rng.normal(0, 1, v_obs.shape) * sig  # add measurement noise
        print(f"[data] {gal.shape[0]} galaxies | v_los rms={v_obs.std():.0f} km/s "
              f"sigma_v med={np.median(sig):.0f} km/s", flush=True)
    # held-out split (VELMOD validation): withhold a fraction from the FIT; predict on it.
    held = None
    if args.holdout > 0 and gal.shape[0] > 20:
        rngh = np.random.default_rng(777)
        hm = rngh.random(gal.shape[0]) < args.holdout
        held = dict(gal=gal[hm].astype(np.float32), rhat=rhat[hm].astype(np.float32),
                    v_obs=v_obs[hm].astype(np.float32), sig=sig[hm].astype(np.float32))
        gal, rhat, v_obs, sig = gal[~hm], rhat[~hm], v_obs[~hm], sig[~hm]
        print(f"[holdout] fit on {gal.shape[0]}, held out {held['gal'].shape[0]} for VELMOD", flush=True)
    inv_sig2 = 1.0 / sig ** 2

    # --- differentiable pmwd velocity-field forward (cosmology matched to the data) ---
    conf = Configuration(ptcl_spacing=float(sp), ptcl_grid_shape=(N,) * 3,
                         mesh_shape=1, float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=args.Om, h=hh, A_s_1e9=args.A_s_1e9), conf)

    have_truth = s_true is not None
    def rkrep(A):                                           # r(k) vs truth (mock) or NaN (real)
        return _rk_curve(A, s_true, N) if have_truth else np.full(N // 2, np.nan)
    gal_j = jnp.asarray(gal); rhat_j = jnp.asarray(rhat)
    v_obs_j = jnp.asarray(v_obs); w_j = jnp.asarray(inv_sig2)
    cell = L / N

    def cic_deposit(pos, w):                                # (Ng,) or (Ng,) weights
        x = (pos % L) / cell
        i0 = jnp.floor(x).astype(jnp.int32); f = x - i0
        grid = jnp.zeros((N, N, N), pos.dtype)
        for dx in (0, 1):
            wx = jnp.where(dx, f[:, 0], 1 - f[:, 0])
            for dy in (0, 1):
                wy = jnp.where(dy, f[:, 1], 1 - f[:, 1])
                for dz in (0, 1):
                    wz = jnp.where(dz, f[:, 2], 1 - f[:, 2])
                    ww = wx * wy * wz * w
                    ii = (i0[:, 0] + dx) % N; jj = (i0[:, 1] + dy) % N; kk = (i0[:, 2] + dz) % N
                    grid = grid.at[ii, jj, kk].add(ww)
        return grid

    def cic_read(grid, pos):                                # trilinear sample (Ng,)
        x = (pos % L) / cell
        i0 = jnp.floor(x).astype(jnp.int32); f = x - i0
        out = 0.0
        for dx in (0, 1):
            wx = jnp.where(dx, f[:, 0], 1 - f[:, 0])
            for dy in (0, 1):
                wy = jnp.where(dy, f[:, 1], 1 - f[:, 1])
                for dz in (0, 1):
                    wz = jnp.where(dz, f[:, 2], 1 - f[:, 2])
                    ii = (i0[:, 0] + dx) % N; jj = (i0[:, 1] + dy) % N; kk = (i0[:, 2] + dz) % N
                    out = out + wx * wy * wz * grid[ii, jj, kk]
        return out

    def vfield(s):
        lin = linear_modes(s.reshape(N, N, N), cosmo, conf)
        ptcl, obs = lpt(lin, cosmo, conf)
        ptcl, obs = nbody(ptcl, obs, cosmo, conf)
        pos = ptcl.pos()                                    # (Np,3) [Mpc/h]
        vel = ptcl.vel * VUNIT_KMS                          # (Np,3) km/s
        mass = cic_deposit(pos, jnp.ones(pos.shape[0], pos.dtype))
        eps = 1e-6
        vx = cic_deposit(pos, vel[:, 0]) / (mass + eps)
        vy = cic_deposit(pos, vel[:, 1]) / (mass + eps)
        vz = cic_deposit(pos, vel[:, 2]) / (mass + eps)
        return vx, vy, vz

    def forward_vlos(s):
        vx, vy, vz = vfield(s)
        vlx = cic_read(vx, gal_j); vly = cic_read(vy, gal_j); vlz = cic_read(vz, gal_j)
        return vlx * rhat_j[:, 0] + vly * rhat_j[:, 1] + vlz * rhat_j[:, 2]

    # --- optional explicit linear-density constraints (known clusters + Local Void) ---
    # Each g_c(s) = [Gaussian_R * delta_L(s; a=1)](x_c) is a LINEAR functional of s, so it is
    # cheap and differentiable, and folds cleanly into the same MAP / Hoffman-Ribak machinery.
    # Velocities alone only pin the near ~15 h^-1Mpc; these nudge the phases of distant clusters.
    CL = [("Virgo", 102.9, -2.3, 16.5), ("Fornax", 236.5, -45.7, 19.0),
          ("Centaurus", 156.0, -11.5, 45.0), ("Hydra", 139.5, -37.5, 50.0),
          ("Norma", 149.5, -7.2, 67.0), ("Perseus", 347.0, -13.0, 74.0)]

    def _sg(l, b, d):
        l, b = np.radians(l), np.radians(b)
        return d * np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])

    use_dc = args.delta_constraints
    dc_tar_j = None; dc_sig = None
    if use_dc:
        oc = L / 2.0
        # Local Void: Tully+2019 places it dominantly toward +SGZ (SGL~78, SGB~+74, ~23 Mpc),
        # NOT anti-Virgo. The Local Sheet's ~260 km/s motion is away from +SGZ (the void push).
        dc_pos = np.vstack([[oc + _sg(l, b, d*hh) for _, l, b, d in CL],
                            oc + _sg(78.0, 74.0, 23.0*hh)])          # +Local Void (+SGZ)
        dc_tar = np.array([args.dc_target]*len(CL) + [args.dc_void])
        dc_sig = np.array([args.dc_sigma]*len(CL) + [max(args.dc_sigma*0.5, 0.3)])
        dc_pos_j = jnp.asarray(dc_pos); dc_tar_j = jnp.asarray(dc_tar)
        dc_w_j = jnp.asarray(1.0 / dc_sig ** 2)
        kf1 = np.fft.fftfreq(N, d=cell) * 2*np.pi; kr1 = np.fft.rfftfreq(N, d=cell) * 2*np.pi
        KX, KY, KZ = np.meshgrid(kf1, kf1, kr1, indexing="ij")
        gauss_k = jnp.asarray(np.exp(-0.5 * (KX**2 + KY**2 + KZ**2) * args.dc_R ** 2))
        print(f"[dc] {len(dc_tar)} linear-density constraints (R={args.dc_R} h^-1Mpc): "
              f"{len(CL)} clusters @delta={args.dc_target}, Local Void @delta={args.dc_void}",
              flush=True)

        def dens_funcs(s):
            d = linear_modes(s.reshape(N, N, N), cosmo, conf, a=1.0, real=True)
            d = d - jnp.mean(d)
            ds = jnp.fft.irfftn(jnp.fft.rfftn(d) * gauss_k, s=(N, N, N))
            return cic_read(ds, dc_pos_j)

    def neg_log_post(s, target, d_target):
        vlos = forward_vlos(s)
        misfit = 0.5 * jnp.sum(w_j * (vlos - target) ** 2)
        prior = 0.5 * jnp.sum(s ** 2)
        total = misfit + prior
        if d_target is not None:
            total = total + 0.5 * jnp.sum(dc_w_j * (dens_funcs(s) - d_target) ** 2)
        return total

    ndata = gal.shape[0]

    def solve(target, s_init, label, d_target=None, discrepancy=True, fixed_iters=None):
        """L-BFGS MAP for a given data target. Returns (s_stop, iters_used).
        discrepancy=True: stop at the first iter with chi2/N <= chi2_target (the real-data
          sweet spot; avoids noise-overfitting seen on sparse data). Used for the MAP.
        fixed_iters=K: run EXACTLY K iterations, return the final iterate. Used for the HR
          mock solve so both MAPs are the SAME operator (same iteration count K*), which is
          essential -- real data (model-mismatch chi2 floor) and mock data stop at very
          different iters under a chi2 criterion, breaking the constrained realization."""
        vg = jax.jit(jax.value_and_grad(lambda s: neg_log_post(s, target, d_target)))
        opt = optax.lbfgs(memory_size=15, linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20))
        s = jnp.asarray(s_init); state = opt.init(s)

        @jax.jit
        def do_step(s, state):
            val, grad = vg(s)
            updates, state = opt.update(grad, state, s, value=val, grad=grad,
                                        value_fn=lambda z: neg_log_post(z, target, d_target))
            return optax.apply_updates(s, updates), state, val

        nmax = fixed_iters if fixed_iters is not None else args.iters
        t0 = time.time(); s_stop = np.asarray(s).copy(); stopped = -1
        for it in range(nmax):
            s, state, val = do_step(s, state)
            sn = np.asarray(s)
            chi2 = float(2 * float(val) - np.sum(sn ** 2)) / ndata
            if discrepancy and fixed_iters is None and stopped < 0 and chi2 <= args.chi2_target:
                s_stop = sn.copy(); stopped = it
                break                                       # stop at the sweet spot
            if it % 20 == 0 or it == nmax - 1:
                rk = rkrep(sn.reshape(N, N, N))
                print(f"[{label} {it:4d}] -logP={float(val):.3e} chi2/N={chi2:.3f} "
                      f"r(k)low={np.nanmean(rk[:4]):.3f} std(s)={sn.std():.3f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        if stopped < 0:                                     # never hit target -> final iterate
            s_stop = np.asarray(s); stopped = nmax - 1
        print(f"[{label}] stop it={stopped} (disc={discrepancy} fixed={fixed_iters})", flush=True)
        return s_stop, stopped + 1

    # --- warm start for the observed-data solve ---
    if args.warm_npz:
        with np.load(args.warm_npz) as z:
            arr = z[args.warm_key]
            s0 = (arr[args.warm_obj] if arr.ndim == 4 else arr).astype(np.float64).ravel()
        print(f"[warm] init from {args.warm_npz}[{args.warm_key}][{args.warm_obj}] "
              f"r(k)low={np.nanmean(rkrep(s0.reshape(N,N,N))[:4]):.3f}", flush=True)
    else:
        s0 = np.zeros(N ** 3); print("[warm] cold start s=0", flush=True)

    # MAP for real data: discrepancy stop at the sweet spot -> gives K* iterations.
    if args.load_smap_npz:
        zc = np.load(args.load_smap_npz)
        s_map_a = zc["s_map"].astype(np.float64).ravel(); kstar = args.iters
        print(f"[warm] REUSED s_map from {args.load_smap_npz} (std={s_map_a.std():.4f}); "
              f"skipping obs MAP solve, K*={kstar} for the HR mock solve", flush=True)
    else:
        s_map_a, kstar = solve(v_obs_j, s0, "map", d_target=dc_tar_j, discrepancy=True)
    s_map = s_map_a.reshape(N, N, N)
    rk_map = rkrep(s_map)
    print(f"[done] s_MAP r(k) low={np.nanmean(rk_map[:4]):.3f} mid={np.nanmean(rk_map[4:12]):.3f} "
          f"std={s_map.std():.3f} K*={kstar} (shrunk: prior-dominated null space)", flush=True)

    s_out = s_map; rk_out = rk_map; kind = "MAP"
    if args.cr_seed < 0 and args.pc_seed >= 0:
        # DEFAULT: power-completion constrained realization (keeps MAP phases, adds only
        # the complementary power sqrt(1-T^2)). Data-only, no second solve, best r_CR.
        s_pc = power_complete(s_map, N, args.pc_seed)
        rk_pc = rkrep(s_pc)
        print(f"[PC] power-complete s_CR r(k) low={np.nanmean(rk_pc[:4]):.3f} "
              f"mid={np.nanmean(rk_pc[4:12]):.3f} std={s_pc.std():.3f} (full power)", flush=True)
        s_out = s_pc; rk_out = rk_pc; kind = "PC"
    elif args.cr_seed >= 0:
        # Hoffman-Ribak constrained realization: full LCDM power + data constraint.
        # s_CR = s_rand - MAP(mock from s_rand) + MAP(obs), BOTH run for K* iterations
        # (same operator, at the real-data sweet spot) so convergence is symmetric.
        rng2 = np.random.default_rng(args.cr_seed)
        s_rand = rng2.standard_normal(N ** 3)
        d_rand = np.asarray(forward_vlos(jnp.asarray(s_rand)))
        d_rand = d_rand + rng2.normal(0, 1, d_rand.shape) * sig    # matched noise
        # same constraint operator on the mock: mock density "obs" = g_c(s_rand) + matched noise
        dtar_rand = None
        if use_dc:
            g_rand = np.asarray(dens_funcs(jnp.asarray(s_rand)))
            dtar_rand = jnp.asarray(g_rand + rng2.normal(0, 1, g_rand.shape) * dc_sig)
        s_map_rand, _ = solve(jnp.asarray(d_rand), np.zeros(N ** 3), "cr_rand",
                              d_target=dtar_rand, discrepancy=False, fixed_iters=kstar)
        s_cr = (s_rand - s_map_rand.ravel() + s_map.ravel()).reshape(N, N, N)
        rk_cr = rkrep(s_cr)
        print(f"[CR] Hoffman-Ribak s_CR r(k) low={np.nanmean(rk_cr[:4]):.3f} "
              f"mid={np.nanmean(rk_cr[4:12]):.3f} std={s_cr.std():.3f} "
              f"(full power restored)", flush=True)
        if use_dc:                                            # achieved linear delta_L(R) at each site
            g_cr = np.asarray(dens_funcs(jnp.asarray(s_cr)))
            names = [n for n, *_ in CL] + ["Void"]
            print("[dc] achieved linear delta_L(R) on s_CR: " +
                  ", ".join(f"{n}={g:.2f}(t{t:.1f})" for n, g, t in zip(names, g_cr, np.asarray(dc_tar_j))),
                  flush=True)
        s_out = s_cr; rk_out = rk_cr; kind = "CR"

    os.makedirs(args.out, exist_ok=True)
    outf = os.path.join(args.out, f"cf4_map_{args.tag}.npz")
    save_kw = dict(s_map=s_map.astype(np.float32), s_out=s_out.astype(np.float32),
                   rk_map=rk_map, rk_out=rk_out, kind=kind, N=N, spacing=sp,
                   L=L, hh=hh, Om=args.Om, A_s_1e9=args.A_s_1e9)
    if s_true is not None:
        save_kw["s_true"] = s_true.astype(np.float32)
    if held is not None:
        save_kw.update({f"held_{k}": v for k, v in held.items()})
    np.savez(outf, **save_kw)
    print(f"[out] {outf} (export field = {kind})", flush=True)

    if args.grafic_out:
        import grafic_io as G
        from pmwd import growth
        a = args.astart
        delta = np.array(linear_modes(jnp.asarray(s_out.reshape(N, N, N)), cosmo, conf,
                                      a=a, real=True), np.float64)
        delta -= delta.mean()
        Om = float(cosmo.Omega_m); h = float(cosmo.h); OL = 1 - Om
        D = float(growth(a, cosmo, conf, order=1, deriv=0))
        f1 = float(growth(a, cosmo, conf, order=1, deriv=1)) / D
        H_a = 100.0 * h * np.sqrt(Om / a ** 3 + OL)
        vx, vy, vz = G.lpt2_velocity(delta, L / h, a, H_a, f1)
        G.write_grafic_ic(args.grafic_out, delta.astype(np.float32), vx, vy, vz,
                          L_mpc_h=L, h=h, astart=a, omega_m=Om, omega_l=OL)
        print(f"[grafic] wrote s_{kind} IC -> {args.grafic_out}", flush=True)


if __name__ == "__main__":
    main()
