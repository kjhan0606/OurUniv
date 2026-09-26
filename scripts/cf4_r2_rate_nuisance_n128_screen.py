"""One actual-data N128 joint IC-gradient/cost screen, not posterior inference.

The six count rates are analytically marginalized under an explicitly broad,
uncalibrated Gamma law. Survival posterior means and published biases/FoG are
still fixed development inputs. No native LG truth identity enters this run.
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
sys.path.insert(0, str(ROOT / 'src'))
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_continuous_tracer import predict_continuous_intensity
from cf4_r2_rate_marginal import gamma_poisson_log_marginal_sparse
from cf4_z0_physical_field import read_centred
from cf4_r2_joint_ic_gradient_control import radial_log_likelihood

CATALOGUE = Path('/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1')
SELECTION = Path('/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1')
CF4 = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz')
PM_STATE = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_unconditional_v1/state.npz')
VERSION = os.environ.get('CF4_R2_SCREEN_VERSION','v1')
if VERSION not in ('v1','v2'):
    raise ValueError('screen version must be v1 or v2')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_rate_nuisance_n128_screen_'+VERSION)
N = 128
BOX = 384.


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    c = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    common, prior = c['common_cosmology'], c['published_prior']
    nuisance = json.loads((ROOT/'config/cf4_r2_rate_nuisance_v1.json').read_text())
    selection_report = json.loads((SELECTION/'result.json').read_text())
    if selection_report['cosmology']['h'] != common['h'] or selection_report['occupied_zero_selection_keys']:
        raise ValueError('selection cosmology/support mismatch')
    with np.load(CATALOGUE/'counts_3_sparse.npz', allow_pickle=False) as f:
        train_keys = f['train_keys'].astype(np.int32)
        train_counts = f['train_counts'].astype(np.int32)
        hold_keys = f['holdout_keys'].astype(np.int32)
        hold_counts = f['holdout_counts'].astype(np.int32)
    with np.load(CATALOGUE/'survival_marks.npz', allow_pickle=False) as f:
        yes, no = f['yes'], f['no']
    if yes.shape != (6,6) or no.shape != (6,6):
        raise ValueError('survival mark geometry mismatch')
    survival_mean = (yes+1)/(yes+no+2)
    with h5py.File(SELECTION/'selection_3.h5', 'r') as f:
        shells = f['selection_shells'][:]
    exposure = np.einsum('ps,psijk->pijk', survival_mean, shells, optimize=True)
    del shells
    flat_exposure = exposure.reshape(-1)
    if np.any(flat_exposure[train_keys] <= 0) or np.any(flat_exposure[hold_keys] <= 0):
        raise ValueError('observed cell outside selection support')
    with np.load(CF4, allow_pickle=False) as f:
        cf4 = {k:f[k] for k in ('radial_observed','CF4_pos','CF4_rhat',
            'CF4_variance','CF4_B','CF4_q_std','CF4_holdout')}
    train = ~cf4['CF4_holdout']
    if np.any(cf4['CF4_variance'][train] <= 0):
        raise ValueError('nonpositive CF4 variance')
    base = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    settings = {key:base[key] for key in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
    for key, field in (('Om','Omega_m'),('Ob','Omega_b'),('h','h'),
                       ('A_s_1e9','A_s_1e9'),('ns','ns')):
        if settings['cosmology'][key] != common[field]:
            raise ValueError('PM/observation cosmology mismatch')
    settings.update(n=N, box_cMpc_h=BOX)
    prior_mean = np.asarray(prior['original_mean_count_per_cell_bright_first'],dtype=np.float64)
    prior_mean *= (BOX/N/prior['original_cell_cMpc_h'])**3
    shape = np.full(6, nuisance['rate_shape'], dtype=np.float64)
    bias = jnp.asarray(prior['linear_regime_bias_bright_first'])
    fixed = json.loads((ROOT/'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())['inference_model']
    fog = jnp.asarray(fixed['FoG_prior_median_km_s'])
    redshift = jnp.asarray(fixed['fixed_redshift_error_km_s'])
    exposure_j = jnp.asarray(exposure,dtype=jnp.float64)
    keys_j = jnp.asarray(train_keys)
    counts_j = jnp.asarray(train_counts)
    mean_j, shape_j = jnp.asarray(prior_mean), jnp.asarray(shape)
    cf4_pos = jnp.asarray(cf4['CF4_pos'][train])
    cf4_rhat = jnp.asarray(cf4['CF4_rhat'][train])
    observed = jnp.asarray(cf4['radial_observed'][train])
    variance = jnp.asarray(cf4['CF4_variance'][train])
    B = jnp.asarray(cf4['CF4_B'][train])
    q_std = jnp.asarray(cf4['CF4_q_std'])
    observer = jnp.asarray([BOX/2]*3)
    zeros = jnp.zeros(6)
    ones = jnp.ones(6)

    def intensity_and_radial(rho, vector):
        unit = nuisance['training_fraction']*predict_continuous_intensity(
            rho,vector,exposure_j,ones,bias,zeros,box=BOX,observer=observer,
            hubble=common['H0_km_s_Mpc'],little_h=common['h'],
            sigma_fog=fog,sigma_redshift=redshift)
        velocity = jnp.stack([read_centred(vector[k],cf4_pos,BOX,.5)
                              for k in range(3)],axis=-1)
        return unit,jnp.sum(velocity*cf4_rhat,axis=1)

    print('loaded actual observations; checking saved unconditional PM state support',flush=True)
    with np.load(PM_STATE,allow_pickle=False) as f:
        saved_rho = jnp.asarray(f['rho'],dtype=jnp.float64)
        saved_velocity = jnp.asarray(f['velocity_km_s'],dtype=jnp.float64)
    start = time.monotonic()
    saved_unit, saved_radial = jax.jit(intensity_and_radial)(saved_rho,saved_velocity)
    saved_unit = np.asarray(saved_unit)
    saved_radial = np.asarray(saved_radial)
    if not np.isfinite(saved_unit).all() or np.any(saved_unit.reshape(-1)[train_keys] <= 0):
        raise ValueError('saved-state positive-count support failure')
    support_seconds = time.monotonic()-start
    saved_rate_ll = {str(alpha):float(gamma_poisson_log_marginal_sparse(
        jnp.asarray(saved_unit),keys_j,counts_j,mean_j,jnp.full((6,),alpha)))
        for alpha in (0.5,1.,2.)}
    if not all(np.isfinite(list(saved_rate_ll.values()))):
        raise FloatingPointError('rate-marginal saved-state likelihood nonfinite')
    print('saved-state support passed; compiling one N128 PM+IC gradient',flush=True)
    evolve, _initial, conf, _cosmo, particle_mass = make_dynamics(settings)
    mass = jnp.full((N**3,),particle_mass)

    def objective_parts(white):
        position, velocity = evolve(white)
        field = particle_grid(position,velocity,mass,conf)
        vector = jnp.moveaxis(field['mean_velocity_km_s'],-1,0)
        unit, radial = intensity_and_radial(field['rho'],vector)
        count_ll = gamma_poisson_log_marginal_sparse(unit,keys_j,counts_j,mean_j,shape_j)
        cf4_ll = radial_log_likelihood(radial,observed,variance,B,q_std)
        return jnp.stack((.5*jnp.vdot(white,white),-count_ll,-cf4_ll))

    def objective(white):
        return jnp.sum(objective_parts(white))

    white = jnp.asarray(np.random.default_rng(2026092501).standard_normal(N**3))
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    start = time.monotonic()
    value, gradient = value_and_grad(white)
    gradient.block_until_ready()
    gradient_seconds = time.monotonic()-start
    gradient_np = np.asarray(gradient)
    value_f = float(value)
    if not np.isfinite(value_f) or not np.isfinite(gradient_np).all():
        raise FloatingPointError('N128 objective or gradient nonfinite')
    report = dict(classification='ACTUAL_DATA_N128_RATE_MARGINAL_IC_GRADIENT_SCREEN',
        source_commit=os.environ['EXPECTED_COMMIT'],grid=N,box_cMpc_h=BOX,
        dx_cMpc_h=BOX/N,particle_count=N**3,
        CF4_training_rows=int(train.sum()),CF4_holdout_rows=int((~train).sum()),
        galaxy_training_count=int(train_counts.sum()),galaxy_holdout_count=int(hold_counts.sum()),
        galaxy_training_occupied_keys=int(len(train_keys)),
        saved_state_occupied_zero_intensity=int(np.count_nonzero(saved_unit.reshape(-1)[train_keys]<=0)),
        saved_state_holdout_zero_intensity=int(np.count_nonzero(saved_unit.reshape(-1)[hold_keys]<=0)),
        saved_state_rate_log_likelihood_shape_sensitivity=saved_rate_ll,
        saved_state_radial_rms_km_s=float(np.sqrt(np.mean(saved_radial**2))),
        objective=value_f,gradient_rms=float(np.sqrt(np.mean(gradient_np**2))),
        gradient_max_abs=float(np.max(np.abs(gradient_np))),
        saved_state_support_seconds=support_seconds,gradient_compile_and_evaluate_seconds=gradient_seconds,
        gamma_prior_shape=float(shape[0]),gamma_prior_mean_per_cell=prior_mean.tolist(),
        survival_Beta_marginalized=False,bias_calibrated=False,FoG_calibrated=False,
        derivative_finite_step_validated=False,actual_data_posterior=False,
        target_N256_map_or_LG_roles_delivered=False)
    if VERSION == 'v2':
        start = time.monotonic()
        _warm_value, warm_gradient = value_and_grad(white)
        warm_gradient.block_until_ready()
        report['warm_gradient_seconds'] = time.monotonic()-start
        direction = np.random.default_rng(2026092602).standard_normal(N**3)
        direction /= np.sqrt(np.mean(direction**2))
        direction_j = jnp.asarray(direction)
        tangent = float(jnp.vdot(gradient,direction_j))
        parts = jax.jit(objective_parts)
        baseline = np.asarray(parts(white))
        ladder = []
        for epsilon in (1e-4,5e-5,2e-5,1e-5):
            plus = np.asarray(parts(white+epsilon*direction_j))
            minus = np.asarray(parts(white-epsilon*direction_j))
            derivative = (plus-minus)/(2*epsilon)
            ladder.append(dict(epsilon=epsilon,
                prior_derivative=float(derivative[0]),
                count_derivative=float(derivative[1]),
                CF4_derivative=float(derivative[2]),
                total_derivative=float(derivative.sum())))
        report['objective_components'] = dict(zip(
            ('white_prior','negative_rate_marginal_count_ll','negative_CF4_ll'),
            baseline.tolist()))
        report['directional_autodiff'] = tangent
        report['directional_finite_difference_ladder'] = ladder
        report['relative_smallest_step_discrepancy'] = abs(
            tangent-ladder[-1]['total_derivative'])/max(
            1.,abs(tangent),abs(ladder[-1]['total_derivative']))
        report['one_direction_step_ladder_computed'] = True
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
