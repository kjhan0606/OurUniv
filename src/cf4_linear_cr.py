#!/usr/bin/env python
"""Exact linear-Gaussian Wiener filter and constrained realizations for CF4.

This is the statistically controlled generalization of ``power_complete``.
The earlier routine had the correct aim--restore Wiener-suppressed high-k power
with random phases--but approximated the residual covariance by an isotropic
k-shell filter.  The method below samples the full linear posterior covariance.
It also replaces subtracting two nonlinear MAP optimizations as if they were a
Hoffman--Ribak operator.  The model is

    s ~ N(0, I)
    u = A s + B q + epsilon
    q ~ N(0, Lambda),  epsilon ~ N(0, N)

where ``s`` is the white-noise LCDM initial field, ``A`` is the linear-theory
radial-velocity response, and ``B q`` marginalizes a constant external bulk
flow and the CF4 distance-scale zero point.  Matrix-free conjugate gradients
solve in data space,

    C = A A^T + B Lambda B^T + N.

Matheron's rule then gives an exact conditional draw (up to the reported CG
residual):

    s_CR = xi + A^T C^-1 (u_obs - A xi - B q0 - epsilon0).

The default velocity estimator is the Watkins--Feldman log-distance estimator,
which is linear in the measured distance modulus and hence has Gaussian
measurement errors.  Galaxy/group positions use the minimum-variance blend of
the bias-corrected distance and redshift distance described for CF4 bulk-flow
work.  A deterministic held-out split is available for posterior-predictive
validation; do not select realization seeds on that held-out set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

C_KMS = 299792.458
LN10 = np.log(10.0)


def modified_cz(v_cmb: np.ndarray, omega_m: float) -> np.ndarray:
    """Low-z cosmographic correction used in CF4 velocity analyses."""
    z = np.asarray(v_cmb, np.float64) / C_KMS
    q0 = 1.5 * omega_m - 1.0
    j0 = 1.0
    zmod = z * (
        1.0
        + 0.5 * (1.0 - q0) * z
        - (1.0 - q0 - 3.0 * q0**2 + j0) * z**2 / 6.0
    )
    return C_KMS * zmod


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stratified_holdout(cz: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Deterministic radial-stratified holdout mask."""
    out = np.zeros(cz.size, dtype=bool)
    if fraction <= 0:
        return out
    rng = np.random.default_rng(seed)
    edges = np.quantile(cz, np.linspace(0.0, 1.0, 11))
    edges[0] -= 1.0
    edges[-1] += 1.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((cz > lo) & (cz <= hi))
        rng.shuffle(idx)
        out[idx[: int(np.round(fraction * idx.size))]] = True
    return out


def prepare_catalog(args: argparse.Namespace) -> dict[str, np.ndarray]:
    z = np.load(args.catalog)
    hcat = float(z["H0"]) / 100.0
    h0 = float(z["H0"])
    dm_all = z["dm"].astype(np.float64)
    edm_all = z["e_dm"].astype(np.float64)
    dist_all = z["dist"].astype(np.float64)
    nhat_all = z["nhat"].astype(np.float64)
    pgc_all = z["pgc"].astype(np.int64)
    cz_all = modified_cz(z["v3k"].astype(np.float64), args.Om)

    # Unbiased distance and its exact lognormal variance.
    ksig_all = (LN10 / 5.0) * edm_all
    dc_all = dist_all * np.exp(-0.5 * ksig_all**2)
    sig_c_all = dc_all * np.sqrt(np.expm1(ksig_all**2))
    dz_all = cz_all / h0
    sig_z = args.position_sigma / h0
    wc = 1.0 / np.maximum(sig_c_all, 1e-4) ** 2
    wz = 1.0 / sig_z**2
    dpos_wf_all = (wc * dc_all + wz * dz_all) / (wc + wz)

    # Watkins--Feldman: Gaussian because it is linear in distance modulus.
    cz_positive = np.where(cz_all > 0, cz_all, np.nan)
    vobs_wf_all = cz_all * LN10 * (
        np.log10(cz_positive / h0) - (dm_all - 25.0) / 5.0
    )
    sig_wf_all = (
        cz_all * LN10 * np.maximum(edm_all, args.edm_floor) / 5.0
    )

    rmax_hmpc = args.box_size / 2.0 * args.radial_fraction
    radius_wf_hmpc = np.abs(dpos_wf_all) * hcat
    finite_common = (
        np.isfinite(cz_all)
        & np.isfinite(dm_all)
        & np.isfinite(edm_all)
        & np.isfinite(dc_all)
        & np.isfinite(sig_c_all)
        & (edm_all > 0)
    )
    keep_wf = (
        finite_common
        & np.isfinite(vobs_wf_all)
        & np.isfinite(sig_wf_all)
        & (cz_all >= args.cz_min)
        & (cz_all <= args.cz_max)
        & (radius_wf_hmpc < rmax_hmpc)
        & (np.abs(vobs_wf_all) < args.vmax)
    )

    # At small redshift the WF15 expansion fails when |u| is comparable to cz
    # and is undefined for negative cz.  For precise nearby distances, use the
    # direct low-z relation u = cz - H0*d.  The distance is debiased for its
    # lognormal measurement distribution and its exact variance is propagated.
    use_local = args.local_distance_max > 0
    keep_local = np.zeros_like(keep_wf)
    vobs_local_all = cz_all - h0 * dc_all
    sig_local_all = h0 * sig_c_all
    if use_local:
        keep_local = (
            finite_common
            & np.isfinite(vobs_local_all)
            & np.isfinite(sig_local_all)
            & (dc_all <= args.local_distance_max)
            & (edm_all <= args.local_edm_max)
            & (np.abs(vobs_local_all) <= args.local_vmax)
            & (dc_all * hcat < rmax_hmpc)
        )
        # Each catalog row enters exactly one likelihood.
        keep_wf &= ~keep_local

    idx_wf = np.flatnonzero(keep_wf)
    idx_local = np.flatnonzero(keep_local)
    raw_idx = np.concatenate((idx_wf, idx_local))
    kind = np.concatenate(
        (
            np.zeros(idx_wf.size, dtype=np.int8),
            np.ones(idx_local.size, dtype=np.int8),
        )
    )
    cz = np.concatenate((cz_all[idx_wf], cz_all[idx_local]))
    nhat = np.concatenate((nhat_all[idx_wf], nhat_all[idx_local]), axis=0)
    dpos = np.concatenate((dpos_wf_all[idx_wf], dc_all[idx_local]))
    vobs = np.concatenate((vobs_wf_all[idx_wf], vobs_local_all[idx_local]))
    sig_measure = np.concatenate((sig_wf_all[idx_wf], sig_local_all[idx_local]))
    variance = np.concatenate(
        (
            (args.error_scale * sig_wf_all[idx_wf]) ** 2 + args.sigma_nl**2,
            (args.local_error_scale * sig_local_all[idx_local]) ** 2
            + args.local_sigma_nl**2,
        )
    )
    pos_hmpc = dpos[:, None] * nhat * hcat + args.box_size / 2.0
    pgc = pgc_all[raw_idx]

    # B columns: external bulk flow [km/s] and delta-H0 [km/s/Mpc].
    B = np.column_stack((nhat, -dpos))
    q_std = np.array(
        [args.bulk_prior, args.bulk_prior, args.bulk_prior, args.h0_prior],
        dtype=np.float64,
    )

    if use_local:
        hold = np.zeros(cz.size, dtype=bool)
        for label in (0, 1):
            select = np.flatnonzero(kind == label)
            radial_coordinate = cz[select] if label == 0 else dpos[select]
            local_hold = stratified_holdout(
                radial_coordinate, args.holdout, args.split_seed + label
            )
            hold[select[local_hold]] = True
    else:
        # Preserve the v1 split bit-for-bit when the hybrid likelihood is off.
        hold = stratified_holdout(cz, args.holdout, args.split_seed)
    return {
        "raw_idx": raw_idx,
        "pgc": pgc,
        "cz": cz,
        "pos": pos_hmpc,
        "rhat": nhat,
        "vobs": vobs,
        "sig_measure": sig_measure,
        "variance": variance,
        "likelihood_kind": kind,
        "B": B,
        "q_std": q_std,
        "holdout": hold,
    }


def build_forward(
    pos: np.ndarray,
    rhat: np.ndarray,
    args: argparse.Namespace,
):
    """Build the JAX linear radial-velocity operator A."""
    import jax
    import jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, growth
    from pmwd.boltzmann import linear_power

    dtype = jnp.float64 if args.float64 else jnp.float32
    N = args.N
    L = args.box_size
    spacing = L / N
    conf = Configuration(
        ptcl_spacing=float(spacing),
        ptcl_grid_shape=(N,) * 3,
        mesh_shape=1,
        # pmwd's background ODE is presently internally float64.  We evaluate
        # the transfer once in float64, then cast the fixed Fourier multiplier
        # so all repeated CG operator calls can run in the requested dtype.
        cosmo_dtype=jnp.float64,
        float_dtype=dtype,
    )
    cosmo = boltzmann(
        SimpleLCDM(
            conf,
            Omega_m=args.Om,
            Omega_b=args.Ob,
            h=args.h,
            A_s_1e9=args.A_s_1e9,
            n_s=args.ns,
        ),
        conf,
    )
    D1 = growth(1.0, cosmo, conf, order=1, deriv=0)
    dD1 = growth(1.0, cosmo, conf, order=1, deriv=1)
    f_growth = (dD1 / D1).astype(dtype)

    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=spacing)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=spacing)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag64 = jnp.asarray(np.sqrt(KX**2 + KY**2 + KZ**2), dtype=jnp.float64)
    delta_amp = jnp.sqrt(linear_power(kmag64, 1.0, cosmo, conf) * L**3).astype(dtype)
    kvec = tuple(jnp.asarray(x, dtype=dtype) for x in (KX, KY, KZ))
    k2 = kvec[0] ** 2 + kvec[1] ** 2 + kvec[2] ** 2
    k2_safe = jnp.where(k2 > 0, k2, jnp.ones_like(k2))

    pos_j = jnp.asarray(pos, dtype=dtype)
    rhat_j = jnp.asarray(rhat, dtype=dtype)
    x = (pos_j % L) / spacing
    i0 = jnp.floor(x).astype(jnp.int32)
    frac = x - i0

    def cic_read(grid):
        out = jnp.zeros(pos_j.shape[0], dtype=dtype)
        for dx in (0, 1):
            wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
            for dy in (0, 1):
                wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
                for dz_ in (0, 1):
                    wz = frac[:, 2] if dz_ else 1.0 - frac[:, 2]
                    ii = (i0[:, 0] + dx) % N
                    jj = (i0[:, 1] + dy) % N
                    kk = (i0[:, 2] + dz_) % N
                    out = out + wx * wy * wz * grid[ii, jj, kk]
        return out

    def forward(s):
        sk = jnp.fft.rfftn(s.reshape((N, N, N)), norm="ortho")
        delta = jnp.fft.irfftn(sk * delta_amp, s=(N, N, N)) / spacing**3
        dk = jnp.fft.rfftn(delta)
        components = []
        for ki in kvec:
            vk = 1j * jnp.asarray(100.0, dtype) * f_growth * ki / k2_safe * dk
            vk = jnp.where(k2 > 0, vk, jnp.zeros_like(vk))
            components.append(cic_read(jnp.fft.irfftn(vk, s=(N, N, N))))
        return (
            components[0] * rhat_j[:, 0]
            + components[1] * rhat_j[:, 1]
            + components[2] * rhat_j[:, 2]
        )

    A = jax.jit(forward)
    zero = jnp.zeros((N, N, N), dtype=dtype)

    @jax.jit
    def AT(y):
        return jax.grad(lambda s: jnp.vdot(forward(s), y))(zero)

    return A, AT, float(f_growth), np.dtype(np.float64 if args.float64 else np.float32)


def build_observer_density_probe(args: argparse.Namespace):
    """Linear z=0 Gaussian-smoothed density at the observer (box centre)."""
    import jax
    import jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann
    from pmwd.boltzmann import linear_power

    dtype = jnp.float64 if args.float64 else jnp.float32
    N = args.N
    L = args.box_size
    spacing = L / N
    conf = Configuration(
        ptcl_spacing=float(spacing),
        ptcl_grid_shape=(N,) * 3,
        mesh_shape=1,
        cosmo_dtype=jnp.float64,
        float_dtype=dtype,
    )
    cosmo = boltzmann(
        SimpleLCDM(
            conf,
            Omega_m=args.Om,
            Omega_b=args.Ob,
            h=args.h,
            A_s_1e9=args.A_s_1e9,
            n_s=args.ns,
        ),
        conf,
    )
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=spacing)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=spacing)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    delta_amp = jnp.sqrt(
        linear_power(jnp.asarray(kmag, jnp.float64), 1.0, cosmo, conf) * L**3
    ).astype(dtype)
    window = jnp.asarray(
        np.exp(-0.5 * kmag**2 * args.observer_delta_radius**2), dtype=dtype
    )
    centre = N // 2

    def probe(s):
        sk = jnp.fft.rfftn(s.reshape((N, N, N)), norm="ortho")
        delta = jnp.fft.irfftn(sk * delta_amp, s=(N, N, N)) / spacing**3
        smooth = jnp.fft.irfftn(
            jnp.fft.rfftn(delta) * window, s=(N, N, N)
        )
        return jnp.atleast_1d(smooth[centre, centre, centre])

    D = jax.jit(probe)
    zero = jnp.zeros((N, N, N), dtype=dtype)

    @jax.jit
    def DT(y):
        return jax.grad(lambda s: jnp.vdot(probe(s), y))(zero)

    return D, DT


def cg_solve(matvec, rhs, precond_diag, args):
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    M = jax.jit(lambda x: x / precond_diag)
    t0 = time.time()
    sol, _ = cg(matvec, rhs, tol=args.cg_tol, atol=0.0, maxiter=args.cg_maxiter, M=M)
    sol.block_until_ready()
    resid = rhs - matvec(sol)
    rel = float(jnp.linalg.norm(resid) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
    return sol, rel, time.time() - t0


def shell_statistics(field: np.ndarray, box_size: float) -> list[dict[str, float]]:
    N = field.shape[0]
    fk = np.fft.rfftn(field, norm="ortho")
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=box_size / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=box_size / N)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kmin = 2.0 * np.pi / box_size
    kny = np.pi * N / box_size
    edges = np.geomspace(kmin, kny, 13)
    rows = []
    power = np.abs(fk) ** 2
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (kmag >= lo) & (kmag < hi)
        if m.any():
            rows.append(
                {
                    "k_lo": float(lo),
                    "k_hi": float(hi),
                    "k_mean": float(kmag[m].mean()),
                    "nmode_rfft": int(m.sum()),
                    "white_power_ratio": float(power[m].mean()),
                }
            )
    return rows


def field_statistics(field: np.ndarray, box_size: float) -> dict:
    from scipy.stats import kurtosis, skew

    x = np.asarray(field, np.float64).ravel()
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "skew": float(skew(x)),
        "excess_kurtosis": float(kurtosis(x)),
        "shells": shell_statistics(field, box_size),
    }


def posterior_predictive(
    A_hold,
    data: dict[str, np.ndarray],
    hold: np.ndarray,
    mean_s,
    mean_q: np.ndarray,
    sample_s: list[np.ndarray],
    sample_q: list[np.ndarray],
) -> dict:
    if not hold.any():
        return {}
    vobs = data["vobs"][hold]
    var = data["variance"][hold]
    B = data["B"][hold]
    latent_mean = np.asarray(A_hold(mean_s), np.float64) + B @ mean_q
    draws = np.stack(
        [np.asarray(A_hold(s), np.float64) + B @ q for s, q in zip(sample_s, sample_q)]
    )
    latent_var = draws.var(axis=0, ddof=1) if len(draws) > 1 else np.zeros_like(vobs)
    pred_var = var + latent_var
    z = (vobs - latent_mean) / np.sqrt(pred_var)
    logp = -0.5 * np.sum(np.log(2.0 * np.pi * pred_var) + (vobs - latent_mean) ** 2 / pred_var)
    # Noise-only comparison with the same measurement model.
    logp0 = -0.5 * np.sum(np.log(2.0 * np.pi * var) + vobs**2 / var)
    return {
        "n": int(vobs.size),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "coverage_1sigma": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_2sigma": float(np.mean(np.abs(z) <= 2.0)),
        "log_predictive_density": float(logp),
        "noise_only_log_density": float(logp0),
        "delta_log_score": float(logp - logp0),
        "latent_rms_kms": float(np.sqrt(np.mean(latent_mean**2))),
        "posterior_sd_median_kms": float(np.median(np.sqrt(latent_var))),
    }


def parse_seeds(text: str) -> list[int]:
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one sample seed is required")
    return seeds


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--catalog", default=str(root / "data" / "cf4_clean.npz"))
    ap.add_argument("--outdir", default=str(root / "recon" / "linear_cr"))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--box-size", type=float, default=384.0, help="Mpc/h")
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--Ob", type=float, default=0.05)
    ap.add_argument("--h", type=float, default=0.746)
    ap.add_argument("--A-s-1e9", type=float, default=1.63)
    ap.add_argument("--ns", type=float, default=0.96)
    ap.add_argument("--cz-min", type=float, default=1000.0, help="km/s")
    ap.add_argument("--cz-max", type=float, default=18000.0, help="km/s")
    ap.add_argument("--vmax", type=float, default=6000.0, help="WF15 estimator cut [km/s]")
    ap.add_argument("--edm-floor", type=float, default=0.04343, help="mag")
    ap.add_argument("--sigma-nl", type=float, default=250.0, help="km/s")
    ap.add_argument(
        "--error-scale",
        type=float,
        default=1.0,
        help="global multiplier on catalog distance-modulus velocity errors",
    )
    ap.add_argument("--position-sigma", type=float, default=300.0, help="km/s")
    ap.add_argument("--radial-fraction", type=float, default=0.95)
    ap.add_argument("--bulk-prior", type=float, default=150.0, help="1D km/s")
    ap.add_argument("--h0-prior", type=float, default=3.0, help="km/s/Mpc")
    ap.add_argument(
        "--local-distance-max",
        type=float,
        default=0.0,
        help="Mpc; positive enables direct u=cz-H0*d likelihood for nearby rows",
    )
    ap.add_argument(
        "--local-edm-max", type=float, default=0.25, help="maximum local DM error [mag]"
    )
    ap.add_argument(
        "--local-vmax", type=float, default=1500.0, help="local |u| cut [km/s]"
    )
    ap.add_argument(
        "--local-sigma-nl",
        type=float,
        default=200.0,
        help="nearby nonlinear velocity dispersion [km/s]",
    )
    ap.add_argument(
        "--local-error-scale",
        type=float,
        default=1.0,
        help="multiplier on propagated nearby distance errors",
    )
    ap.add_argument(
        "--observer-delta",
        type=float,
        default=0.0,
        help="optional z=0 linear density target at the box-centre observer",
    )
    ap.add_argument(
        "--observer-delta-sigma",
        type=float,
        default=0.0,
        help="enable the observer-density Gaussian constraint when positive",
    )
    ap.add_argument(
        "--observer-delta-radius",
        type=float,
        default=5.0,
        help="Gaussian smoothing radius of observer density [Mpc/h]",
    )
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=20260729)
    ap.add_argument("--sample-seeds", type=parse_seeds, default=parse_seeds("1,2,3,4"))
    ap.add_argument("--precond-probes", type=int, default=4)
    ap.add_argument("--cg-tol", type=float, default=3e-5)
    ap.add_argument("--cg-maxiter", type=int, default=500)
    ap.add_argument("--float64", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = prepare_catalog(args)
    hold = data["holdout"]
    train = ~hold
    print(
        f"[data] kept={len(train)} train={train.sum()} holdout={hold.sum()} "
        f"cz={data['cz'].min():.0f}..{data['cz'].max():.0f} km/s "
        f"sig_med={np.median(np.sqrt(data['variance'])):.0f} km/s",
        flush=True,
    )
    kinds = data["likelihood_kind"]
    print(
        f"[data] likelihood rows: WF15={int(np.sum(kinds == 0))} "
        f"local-direct={int(np.sum(kinds == 1))}",
        flush=True,
    )

    A, AT, f_growth, npdtype = build_forward(data["pos"][train], data["rhat"][train], args)
    import jax
    import jax.numpy as jnp

    scale = jnp.asarray(np.sqrt(data["variance"][train]), dtype=npdtype)
    velocity_dnorm = jnp.asarray(data["vobs"][train], dtype=npdtype) / scale
    velocity_Bn = jnp.asarray(data["B"][train], dtype=npdtype) / scale[:, None]
    qvar = jnp.asarray(data["q_std"] ** 2, dtype=npdtype)

    use_observer_density = args.observer_delta_sigma > 0
    if use_observer_density:
        Dobs, DTobs = build_observer_density_probe(args)
        obs_sigma = jnp.asarray(args.observer_delta_sigma, dtype=npdtype)
        dnorm = jnp.concatenate(
            (
                velocity_dnorm,
                jnp.asarray([args.observer_delta], dtype=npdtype) / obs_sigma,
            )
        )
        Bn = jnp.concatenate(
            (
                velocity_Bn,
                jnp.zeros((1, velocity_Bn.shape[1]), dtype=npdtype),
            ),
            axis=0,
        )
        An = jax.jit(
            lambda s: jnp.concatenate((A(s) / scale, Dobs(s) / obs_sigma))
        )
        ATn = jax.jit(
            lambda y: AT(y[:-1] / scale) + DTobs(y[-1:] / obs_sigma)
        )
        print(
            f"[observer] delta_L Gaussian R={args.observer_delta_radius:.1f} Mpc/h "
            f"target={args.observer_delta:+.3f}+/-{args.observer_delta_sigma:.3f}",
            flush=True,
        )
    else:
        dnorm = velocity_dnorm
        Bn = velocity_Bn
        An = jax.jit(lambda s: A(s) / scale)
        ATn = jax.jit(lambda y: AT(y / scale))

    @jax.jit
    def Cnorm(y):
        return y + An(ATn(y)) + Bn @ (qvar * (Bn.T @ y))

    # Compile and prove the implemented adjoint numerically before sampling.
    rng = np.random.default_rng(913)
    sx = jnp.asarray(rng.standard_normal((args.N,) * 3), dtype=npdtype)
    dy = jnp.asarray(rng.standard_normal(dnorm.size), dtype=npdtype)
    lhs = float(jnp.vdot(An(sx), dy))
    rhs = float(jnp.vdot(sx, ATn(dy)))
    adjoint_rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    print(f"[operator] f={f_growth:.5f} adjoint_rel={adjoint_rel:.3e}", flush=True)

    probe_power = np.zeros(dnorm.size, dtype=np.float64)
    for i in range(args.precond_probes):
        sprobe = jnp.asarray(rng.standard_normal((args.N,) * 3), dtype=npdtype)
        probe_power += np.asarray(An(sprobe), np.float64) ** 2
    probe_power /= args.precond_probes
    nuisance_diag = np.sum(np.asarray(Bn, np.float64) ** 2 * data["q_std"][None, :] ** 2, axis=1)
    precond_diag = jnp.asarray(1.0 + probe_power + nuisance_diag, dtype=npdtype)

    alpha_mean, mean_rel, mean_sec = cg_solve(Cnorm, dnorm, precond_diag, args)
    mean_s = ATn(alpha_mean)
    mean_q = np.asarray(qvar * (Bn.T @ alpha_mean), np.float64)
    train_resid = np.asarray(dnorm - An(mean_s) - Bn @ jnp.asarray(mean_q, dtype=npdtype))
    print(
        f"[mean] cg_rel={mean_rel:.3e} sec={mean_sec:.1f} "
        f"norm_resid_rms={np.sqrt(np.mean(train_resid**2)):.3f} "
        f"q=(bulk {mean_q[:3]}, dH0 {mean_q[3]:+.3f})",
        flush=True,
    )
    if use_observer_density:
        print(
            f"[observer] posterior-mean delta_L="
            f"{float(Dobs(mean_s)[0]):+.4f}",
            flush=True,
        )

    samples_jax = []
    samples_np = []
    qs = []
    sample_meta = []
    for seed in args.sample_seeds:
        rs = np.random.default_rng(seed)
        xi = jnp.asarray(rs.standard_normal((args.N,) * 3), dtype=npdtype)
        q0 = jnp.asarray(rs.standard_normal(4) * data["q_std"], dtype=npdtype)
        eps0 = jnp.asarray(rs.standard_normal(dnorm.size), dtype=npdtype)
        sample_rhs = dnorm - An(xi) - Bn @ q0 - eps0
        alpha, rel, sec = cg_solve(Cnorm, sample_rhs, precond_diag, args)
        scr = xi + ATn(alpha)
        qcr = np.asarray(q0 + qvar * (Bn.T @ alpha), np.float64)
        scr_np = np.asarray(scr, np.float32)
        samples_jax.append(scr)
        samples_np.append(scr_np)
        qs.append(qcr)
        stats = field_statistics(scr_np, args.box_size)
        sample_meta.append({"seed": seed, "cg_rel": rel, "seconds": sec, "q": qcr.tolist(), **stats})
        print(
            f"[sample {seed}] cg_rel={rel:.3e} sec={sec:.1f} "
            f"std={stats['std']:.5f} skew={stats['skew']:+.4f} "
            f"kurt={stats['excess_kurtosis']:+.4f}",
            flush=True,
        )
        if use_observer_density:
            print(
                f"[observer sample {seed}] delta_L="
                f"{float(Dobs(scr)[0]):+.4f}",
                flush=True,
            )

    held_diag = {}
    if hold.any():
        A_hold, _, _, _ = build_forward(data["pos"][hold], data["rhat"][hold], args)
        held_diag = posterior_predictive(
            A_hold, data, hold, mean_s, mean_q, samples_jax, qs
        )
        print(
            f"[holdout] n={held_diag['n']} z={held_diag['z_mean']:+.3f}"
            f"+/-{held_diag['z_std']:.3f} cov68={held_diag['coverage_1sigma']:.3f} "
            f"cov95={held_diag['coverage_2sigma']:.3f} "
            f"dlogscore={held_diag['delta_log_score']:+.1f}",
            flush=True,
        )

    mean_np = np.asarray(mean_s, np.float32)
    common = {
        "s_map": mean_np,
        "kind": np.array("LINEAR_GAUSSIAN_CR"),
        "N": np.int64(args.N),
        "spacing": np.float64(args.box_size / args.N),
        "L": np.float64(args.box_size),
        "hh": np.float64(args.h),
        "Om": np.float64(args.Om),
        "Ob": np.float64(args.Ob),
        "A_s_1e9": np.float64(args.A_s_1e9),
        "ns": np.float64(args.ns),
        "train_raw_idx": data["raw_idx"][train],
        "holdout_raw_idx": data["raw_idx"][hold],
    }
    output_files = []
    for seed, scr, qcr in zip(args.sample_seeds, samples_np, qs):
        path = outdir / f"cf4_linear_cr_{args.tag}_s{seed}.npz"
        np.savez(
            path,
            **common,
            s_out=scr,
            sample_seed=np.int64(seed),
            nuisance_q=qcr,
        )
        output_files.append(str(path))

    manifest = {
        "status": "diagnostic" if hold.any() else "all_data",
        "method": "linear Gaussian WF + Matheron constrained realization",
        "catalog": os.path.abspath(args.catalog),
        "catalog_sha256": sha256_file(args.catalog),
        "configuration": vars(args) | {"sample_seeds": args.sample_seeds},
        "n_kept": int(len(train)),
        "n_train": int(train.sum()),
        "n_holdout": int(hold.sum()),
        "growth_rate": f_growth,
        "adjoint_relative_error": adjoint_rel,
        "mean_cg_relative_residual": mean_rel,
        "mean_nuisance_q": mean_q.tolist(),
        "train_normalized_residual_rms": float(np.sqrt(np.mean(train_resid**2))),
        "heldout": held_diag,
        "samples": sample_meta,
        "outputs": output_files,
    }
    manifest_path = outdir / f"manifest_{args.tag}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"[out] {manifest_path}", flush=True)
    for path in output_files:
        print(f"[out] {path}", flush=True)


if __name__ == "__main__":
    main()
