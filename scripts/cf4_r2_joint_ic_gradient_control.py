"""Bounded actual-data CF4+2M++ IC-gradient mechanics control.

N128 source counts and shell exposure are conservatively restricted to N32.
The physical forward is one unconditional 384-cMpc/h PMWD trajectory. This
checks differentiation of one same-state likelihood, not tracer calibration,
posterior inference, or a map at the R2 target resolution.
"""

import json
import os
from pathlib import Path
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_2mpp_joint_likelihood_jax import poisson_log_likelihood_jax
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_continuous_tracer import predict_continuous_intensity
from cf4_z0_physical_field import read_centred

CATALOGUE = Path('/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1')
SELECTION = Path('/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1')
CF4 = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_joint_ic_gradient_n32_v1')


def aggregate_sparse(keys, values):
    pop, cell = np.divmod(keys.astype(np.int64), 128**3)
    xyz = np.array(np.unravel_index(cell, (128,)*3)).T//4
    coarse = pop*32**3 + np.ravel_multi_index(xyz.T, (32,)*3)
    full = np.bincount(coarse, weights=values, minlength=6*32**3)
    return full.reshape(6,32,32,32).astype(np.int32)


def data():
    contract = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    c = contract['common_cosmology']
    selection_report = json.loads((SELECTION/'result.json').read_text())
    if selection_report['cosmology']['h'] != c['h'] or selection_report['occupied_zero_selection_keys']:
        raise ValueError('selection provenance/support mismatch')
    with np.load(CATALOGUE/'counts_3_sparse.npz', allow_pickle=False) as f:
        counts = {label: aggregate_sparse(f[f'{label}_keys'], f[f'{label}_counts'])
                  for label in ('all','train','holdout')}
    np.testing.assert_array_equal(counts['train']+counts['holdout'], counts['all'])
    with np.load(CATALOGUE/'survival_marks.npz', allow_pickle=False) as f:
        yes, no = f['yes'], f['no']
    survival = (yes+1)/(yes+no+2)
    coarse = np.empty((6,6,32,32,32), dtype=np.float32)
    with h5py.File(SELECTION/'selection_3.h5', 'r') as f:
        fine = f['selection_shells']
        for x in range(32):
            slab = fine[:,:,4*x:4*x+4]
            coarse[:,:,x] = slab.reshape(6,6,4,32,4,32,4).mean(axis=(2,4,6))
    exposure = np.einsum('ps,psijk->pijk', survival, coarse, optimize=True)
    with np.load(CF4, allow_pickle=False) as f:
        cf4 = {k:f[k] for k in ('radial_observed','CF4_pos','CF4_rhat',
            'CF4_variance','CF4_B','CF4_q_std','CF4_holdout')}
    if np.any(exposure[counts['all']>0] <= 0) or np.any(cf4['CF4_variance'] <= 0):
        raise ValueError('observation support/variance failure')
    prior = contract['published_prior']
    nbar = np.asarray(prior['original_mean_count_per_cell_bright_first']) * (
        12./prior['original_cell_cMpc_h'])**3
    bias = np.asarray(prior['linear_regime_bias_bright_first'])
    if not np.all(np.isfinite(nbar)) or np.any(nbar <= 0):
        raise ValueError('invalid sourced rate prior')
    return c, counts, exposure, cf4, nbar, bias


def radial_log_likelihood(predicted, observed, variance, B, q_std):
    """Marginalize shared four-component Gaussian bulk/H0 nuisance exactly."""
    residual = (observed-predicted)/jnp.sqrt(variance)
    U = B*q_std[None,:]/jnp.sqrt(variance)[:,None]
    gram = jnp.eye(4, dtype=residual.dtype)+U.T@U
    z = U.T@residual
    quadratic = residual@residual-z@jnp.linalg.solve(gram,z)
    _sign, logdet_small = jnp.linalg.slogdet(gram)
    return -.5*(quadratic+jnp.sum(jnp.log(variance))+logdet_small
                 +len(observed)*jnp.log(2*jnp.pi))


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    c, counts, exposure, cf4, nbar, bias = data()
    base = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    settings = {key:base[key] for key in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
    for key, field in (('Om','Omega_m'),('Ob','Omega_b'),('h','h'),
                       ('A_s_1e9','A_s_1e9'),('ns','ns')):
        if settings['cosmology'][key] != c[field]:
            raise ValueError('PM/observation cosmology mismatch')
    settings.update(n=32,box_cMpc_h=384.)
    evolve, _, conf, _, particle_mass = make_dynamics(settings)
    mass = jnp.full((32**3,), particle_mass)
    exposure = jnp.asarray(exposure, dtype=jnp.float64)
    train_count = jnp.asarray(counts['train'])
    holdout_count = jnp.asarray(counts['holdout'])
    nbar = jnp.asarray(nbar)
    bias = jnp.asarray(bias)
    train = ~cf4['CF4_holdout']
    cf4_pos = jnp.asarray(cf4['CF4_pos'][train])
    cf4_rhat = jnp.asarray(cf4['CF4_rhat'][train])
    observed = jnp.asarray(cf4['radial_observed'][train])
    variance = jnp.asarray(cf4['CF4_variance'][train])
    B = jnp.asarray(cf4['CF4_B'][train])
    q_std = jnp.asarray(cf4['CF4_q_std'])
    # Small direct-covariance oracle for the four shared CF4 nuisance modes.
    ncheck = 8
    u = np.asarray(B[:ncheck])*np.asarray(q_std)[None,:]
    cov = np.diag(np.asarray(variance[:ncheck]))+u@u.T
    sign, logdet = np.linalg.slogdet(cov)
    if sign != 1:
        raise ValueError('CF4 marginal covariance not positive definite')
    y = np.asarray(observed[:ncheck])
    direct = -.5*(y@np.linalg.solve(cov,y)+logdet+ncheck*np.log(2*np.pi))
    woodbury = float(radial_log_likelihood(jnp.zeros(ncheck), observed[:ncheck],
        variance[:ncheck],B[:ncheck],q_std))
    np.testing.assert_allclose(woodbury,direct,rtol=1e-10,atol=1e-8)
    nuisance = json.loads((ROOT/'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())['inference_model']
    fog = jnp.asarray(nuisance['FoG_prior_median_km_s'])
    redshift = jnp.asarray(nuisance['fixed_redshift_error_km_s'])

    def forward(white):
        pos, vel = evolve(white)
        field = particle_grid(pos, vel, mass, conf)
        rho = field['rho']
        vector = jnp.moveaxis(field['mean_velocity_km_s'], -1, 0)
        intensity = predict_continuous_intensity(rho, vector, exposure,
            nbar, bias, jnp.zeros(6), box=384., observer=jnp.asarray([192.,192.,192.]),
            hubble=c['H0_km_s_Mpc'], little_h=c['h'],
            sigma_fog=fog, sigma_redshift=redshift)
        velocity = jnp.stack([read_centred(vector[k], cf4_pos, 384., .5)
                              for k in range(3)], axis=-1)
        radial = jnp.sum(velocity*cf4_rhat, axis=1)
        return intensity, radial

    def objective(white):
        intensity, radial = forward(white)
        count_ll = poisson_log_likelihood_jax(train_count, .6*intensity)
        cf4_ll = radial_log_likelihood(radial,observed,variance,B,q_std)
        prior = .5*jnp.vdot(white,white)
        return prior-count_ll-cf4_ll

    rng = np.random.default_rng(2026092601)
    white = jnp.asarray(rng.standard_normal(32**3))
    direction = jnp.asarray(rng.standard_normal(32**3))
    direction /= jnp.sqrt(jnp.mean(direction**2))
    t0 = time.monotonic()
    intensity, radial = jax.jit(forward)(white)
    intensity, radial = np.asarray(intensity), np.asarray(radial)
    forward_seconds = time.monotonic()-t0
    train_np = counts['train']
    hold_np = counts['holdout']
    if not np.isfinite(intensity).all() or np.any(intensity[train_np>0] <= 0):
        report = dict(classification='NO_GO_N32_POSITIVE_COUNT_SUPPORT',
                      training_occupied_zero_intensity=int(np.count_nonzero(
                          intensity[train_np>0]<=0)), forward_seconds=forward_seconds,
                      gradient_executed=False, posterior=False)
    else:
        t1 = time.monotonic()
        value, gradient = jax.jit(jax.value_and_grad(objective))(white)
        gradient.block_until_ready()
        gradient_seconds = time.monotonic()-t1
        epsilon = .005
        plus = float(jax.jit(objective)(white+epsilon*direction))
        minus = float(jax.jit(objective)(white-epsilon*direction))
        finite_difference = (plus-minus)/(2*epsilon)
        tangent = float(jnp.vdot(gradient,direction))
        relative_error = abs(finite_difference-tangent)/max(1.,abs(finite_difference),abs(tangent))
        report = dict(classification='ACTUAL_DATA_N32_JOINT_IC_GRADIENT_CONTROL',
            source_commit=os.environ['EXPECTED_COMMIT'], grid=32,box_cMpc_h=384.,
            dx_cMpc_h=12., common_cosmology=c,
            CF4_training_rows=int(train.sum()),CF4_heldout_rows=int((~train).sum()),
            galaxy_training_counts=int(train_np.sum()),
            galaxy_heldout_counts=int(hold_np.sum()),
            galaxy_training_occupied_zero_intensity=0,
            galaxy_holdout_occupied_zero_intensity=int(np.count_nonzero(intensity[hold_np>0]<=0)),
            min_training_occupied_intensity=float(intensity[train_np>0].min()),
            predicted_CF4_radial_rms_km_s=float(np.sqrt(np.mean(radial**2))),
            objective=float(value),gradient_rms=float(np.sqrt(np.mean(np.asarray(gradient)**2))),
            directional_gradient=tangent,finite_difference=finite_difference,
            relative_directional_error=relative_error,forward_seconds=forward_seconds,
            gradient_seconds=gradient_seconds,
            fixed_published_rate_and_bias_only=True,
            survival_uncertainty_marginalized=False,FoG_calibrated=False,
            CF4_and_count_holdouts_excluded_from_objective=True,
            actual_N128_or_N256_joint_gradient=False,
            posterior_or_LG_identification=False)
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
