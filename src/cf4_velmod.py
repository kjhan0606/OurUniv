#!/usr/bin/env python
"""VELMOD-style validation of the CF4 reconstruction against HELD-OUT velocities.

The reconstruction was fit to a subset of CF4 galaxies (--holdout withheld the rest).
Here we predict LOS velocities at the held-out galaxy positions and compare to their
observed velocities with the VELMOD / "Velocity Field Olympics" likelihood model:

    u_obs,i = beta * u_pred,i  +  V_ext . r_hat_i  +  noise,   noise ~ N(0, sig_i^2 + sig_v^2)

  beta   : velocity-amplitude scaling (1 = reconstruction has the right fsigma8/sigma8;
           !=1 flags an amplitude bias -- the direct test of our sigma8 calibration).
  V_ext  : residual external bulk flow the reconstruction volume does NOT explain
           (small |V_ext| = the box captures the flow).
  sig_v  : extra scatter beyond measurement error (nonlinear/thermal; smaller = better).

Max-likelihood: for fixed sig_v, (beta, V_ext) are a weighted linear least-squares; sig_v
is optimized in 1D (the ln|C| term). We also fit a NULL model (beta=0: bulk flow + noise
only) -> Delta(-2lnL)/Delta BIC quantifies how much the RECONSTRUCTION adds beyond a mere
bulk flow (the "does it capture real structure" significance). All on held-out data =
non-circular. Point-by-point residuals localise where structures are mis-reconstructed.

Run on GPU: sbatch src/cf4_job.slurm src/cf4_velmod.py --pred recon/cf4_map_cf4_real192.npz
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import numpy as np


def fit_linear(upred, rhat, uobs, w):
    """Weighted LSQ for p=[beta,Vx,Vy,Vz] in uobs = beta*upred + V.rhat. Returns p, resid."""
    A = np.column_stack([upred, rhat[:, 0], rhat[:, 1], rhat[:, 2]])
    W = w[:, None]
    ATA = A.T @ (W * A); ATb = A.T @ (w * uobs)
    p = np.linalg.solve(ATA, ATb)
    resid = uobs - A @ p
    return p, resid


def neg2logL(sig_v, upred, rhat, uobs, sig, null=False):
    """Profile -2lnL over (beta,V_ext) at this sig_v. null=True forces beta=0."""
    var = sig ** 2 + sig_v ** 2
    w = 1.0 / var
    if null:
        A = rhat; ATA = A.T @ (w[:, None] * A); ATb = A.T @ (w * uobs)
        V = np.linalg.solve(ATA, ATb); p = np.concatenate([[0.0], V])
        resid = uobs - A @ V
    else:
        p, resid = fit_linear(upred, rhat, uobs, w)
    return float(np.sum(w * resid ** 2) + np.sum(np.log(var))), p, resid


def profile_sigv(upred, rhat, uobs, sig, null=False):
    from scipy.optimize import minimize_scalar
    f = lambda sv: neg2logL(abs(sv), upred, rhat, uobs, sig, null)[0]
    res = minimize_scalar(f, bounds=(1.0, 3000.0), method="bounded")
    sv = abs(res.x)
    val, p, resid = neg2logL(sv, upred, rhat, uobs, sig, null)
    return sv, p, resid, val


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mock_pipeline import VUNIT_KMS
    from cf4_gen_train import cic_deposit

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="cf4_map_*.npz with s_out + held_* arrays")
    ap.add_argument("--field", default="s_out")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P = np.load(args.pred)
    if "held_gal" not in P:
        raise SystemExit("no held-out set in npz -- rerun cf4_explicit_map.py with --holdout 0.2")
    s = P[args.field].astype(np.float64)
    N = int(P["N"]); sp = float(P["spacing"]); L = float(P["L"])
    hh = float(P["hh"]); Om = float(P["Om"]); As = float(P["A_s_1e9"])
    gal = P["held_gal"].astype(np.float64); rhat = P["held_rhat"].astype(np.float64)
    uobs = P["held_v_obs"].astype(np.float64); sig = P["held_sig"].astype(np.float64)
    print(f"[velmod] {args.pred} field={args.field} | held-out {gal.shape[0]} galaxies "
          f"N={N} L={L:.0f} h={hh:.3f} As={As}", flush=True)

    import jax, jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody
    conf = Configuration(ptcl_spacing=float(sp), ptcl_grid_shape=(N,) * 3,
                         mesh_shape=1, float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=Om, h=hh, A_s_1e9=As), conf)
    lin = linear_modes(jnp.asarray(s.reshape(N, N, N)), cosmo, conf)
    ptcl, obs = lpt(lin, cosmo, conf); ptcl, obs = nbody(ptcl, obs, cosmo, conf)
    pos = np.asarray(ptcl.pos(), np.float64); vel = np.asarray(ptcl.vel, np.float64) * VUNIT_KMS
    # velocity field on the grid (CIC momentum / mass), then trilinear read at held positions
    mass = cic_deposit(pos, 1.0, N, L)
    eps = 1e-8
    vfield = np.stack([cic_deposit(pos, vel[:, a], N, L) / (mass + eps) for a in range(3)], 0)

    cell = L / N
    x = (gal % L) / cell
    i0 = np.floor(x).astype(int); f = x - i0
    upred_vec = np.zeros((gal.shape[0], 3))
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                wgt = wx * wy * wz
                ii = (i0[:, 0] + dx) % N; jj = (i0[:, 1] + dy) % N; kk = (i0[:, 2] + dz) % N
                upred_vec += wgt[:, None] * vfield[:, ii, jj, kk].T
    upred = np.sum(upred_vec * rhat, axis=1)                # predicted LOS velocity

    # VELMOD fits: reconstruction model vs null (bulk-flow-only)
    sv, p, resid, m2 = profile_sigv(upred, rhat, uobs, sig, null=False)
    beta = p[0]; Vext = p[1:]; Vmag = np.linalg.norm(Vext)
    sv0, p0, resid0, m2_0 = profile_sigv(upred, rhat, uobs, sig, null=True)
    n = gal.shape[0]
    dBIC = (m2_0) - (m2 + 1 * np.log(n))                    # null has 1 fewer param (beta)
    # noise-aware quality: chi2/N of the model, and binned correlation
    var = sig ** 2 + sv ** 2
    chi2N = np.sum(resid ** 2 / var) / n
    # binned correlation: sort by upred, average in ~15 bins (beats per-object noise)
    order = np.argsort(upred); nb = 15; edges = np.linspace(0, n, nb + 1).astype(int)
    ubp, uob = [], []
    for b in range(nb):
        idx = order[edges[b]:edges[b + 1]]
        if len(idx):
            wl = 1.0 / var[idx]
            ubp.append(np.sum(wl * (beta * upred[idx] + upred_vec[idx] @ Vext * 0)) / wl.sum())
            uob.append(np.sum(wl * uobs[idx]) / wl.sum())
    ubp, uob = np.array(ubp), np.array(uob)
    rbin = np.corrcoef(ubp, uob)[0, 1] if len(ubp) > 2 else np.nan

    print(f"\n[VELMOD held-out results]")
    print(f"  beta  = {beta:.3f}   (1 = correct velocity amplitude / sigma8; !=1 -> bias)")
    print(f"  V_ext = {Vmag:.0f} km/s  SG=({Vext[0]:.0f},{Vext[1]:.0f},{Vext[2]:.0f})  "
          f"(residual flow the box does NOT explain)")
    print(f"  sig_v = {sv:.0f} km/s   (intrinsic scatter beyond measurement error)")
    print(f"  chi2/N (model) = {chi2N:.3f}")
    print(f"  binned corr (recon vs obs, noise-averaged) = {rbin:.3f}")
    print(f"  null(sigv={sv0:.0f}): -2lnL={m2_0:.1f} | recon: -2lnL={m2:.1f} | "
          f"Delta(-2lnL)={m2_0-m2:.1f}  Delta BIC={dBIC:.1f}")
    verdict = ("recon STRONGLY beats bulk-flow-only" if dBIC > 10 else
               "recon beats bulk-flow-only" if dBIC > 2 else
               "recon NOT better than a bulk flow")
    print(f"  => {verdict} (dBIC>10 decisive)")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    ax[0].errorbar(beta * upred, uobs, yerr=sig, fmt=".", ms=3, alpha=0.2, elinewidth=0.4)
    lim = np.percentile(np.abs(np.r_[beta * upred, uobs]), 99)
    ax[0].plot(ubp, uob, "gs", ms=7, label="binned (noise-averaged)")
    ax[0].plot([-lim, lim], [-lim, lim], "r--", label="1:1")
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel("beta * u_pred [km/s]"); ax[0].set_ylabel("u_obs (held-out) [km/s]")
    ax[0].set_title(f"VELMOD: beta={beta:.2f} sig_v={sv:.0f} rbin={rbin:.2f}")
    ax[0].legend()
    ax[1].scatter(upred, uobs, s=6, c=resid, cmap="coolwarm", vmin=-2*sv, vmax=2*sv)
    ax[1].set_xlabel("u_pred [km/s]"); ax[1].set_ylabel("u_obs [km/s]")
    ax[1].set_title(f"residuals | dBIC={dBIC:.0f} ({'recon helps' if dBIC>2 else 'no gain'})")
    fig.tight_layout()
    out = args.out or args.pred.replace(".npz", "_velmod.png")
    fig.savefig(out, dpi=110)
    print(f"[velmod] saved {out}", flush=True)


if __name__ == "__main__":
    main()
