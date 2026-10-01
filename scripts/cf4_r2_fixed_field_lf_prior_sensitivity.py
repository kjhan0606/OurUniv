#!/usr/bin/env python3
"""One fixed-field, prior-centred LF-shape sensitivity; not a fit/calibration."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_linked_fp_sparse_train import SOURCE
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_raw_volume_target import tracer_geometry, tracer_masses
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_resolution_target import source_geometry_at_resolution
from cf4_r2_count_operator_closure_mock import radial_population_bins

jax.config.update('jax_enable_x64', True)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
N_SOURCE, N_COUNT, BOX = 256, 128, 384.0
NIC, NNUISANCE = N_SOURCE**3, 24
SOURCE_CHUNK = 2_097_152
RADIAL_EDGES = np.arange(0.0, 192.0 + 12.0, 12.0)
PROFILE = BASE/'r2_n256_frozen_field_tracer_profile_20261002_v3/result.json'
CHECKPOINT = BASE/'r2_n256_gl2_metric6000_control_a_v1/chain_a_accepted_checkpoint.npz'
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
CLOSURE = BASE/'r2_n256_count_operator_closure_20261002_v1/result.json'
OUT = Path(os.environ.get('CF4_R2_LF_SENSITIVITY_OUT_DIR',
    str(BASE/'r2_n256_lf_prior_center_sensitivity_20261002_v1')))
_OUTPUT_CREATED = False


def pearson(observed, expected, minimum_expected=5.0):
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    use = exp >= minimum_expected
    if not np.any(use):
        raise ValueError('no aggregate bins meet the expected-count threshold')
    return dict(statistic=float(np.sum((obs[use]-exp[use])**2/exp[use])),
        bins=int(use.sum()), expected_count_threshold=float(minimum_expected),
        interpretation='in-sample aggregate descriptive discrepancy, not a calibrated test')


def _run():
    global _OUTPUT_CREATED
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('submit this fixed-field N256 forward check to Slurm with a GPU')
    OUT.mkdir(parents=True, exist_ok=False)
    _OUTPUT_CREATED = True
    started = time.monotonic()

    profile = json.loads(PROFILE.read_text())
    closure = json.loads(CLOSURE.read_text())
    if profile.get('status') != 'FROZEN_FIELD_TRACER_PROFILE_BOUNDED_NOT_CONVERGED':
        raise ValueError('the frozen-field nuisance profile is not the expected reference')
    if closure.get('status') != 'COUNT_OPERATOR_CLOSURE_COMPLETE_NOT_CALIBRATION':
        raise ValueError('the exact-reference TSC/NGP closure result is unavailable')
    tracer0 = np.asarray(profile['initial']['tracer_coordinates'], dtype=np.float64)
    tracer1 = tracer0.copy()
    tracer1[7:9] = 0.0  # prior-centre alpha/mstar white coordinates; all others frozen
    baseline_score = float(closure['tsc_reference']['training_count_log_score'])
    baseline_table = np.zeros((6, len(RADIAL_EDGES)-1), dtype=np.float64)
    for row in closure['radial_population_counts']:
        ibin = int(np.flatnonzero(RADIAL_EDGES == row['radius_lower_cMpc_h'])[0])
        baseline_table[row['population'], ibin] = row['expected_tsc']

    with np.load(CHECKPOINT, allow_pickle=False) as f:
        q = np.asarray(f['canonical'], dtype=np.float64)
        completed = int(f['completed_transitions'])
    if q.shape != (NIC+NNUISANCE,) or completed != 1 or not np.isfinite(q).all():
        raise ValueError('saved accepted endpoint-A field checkpoint is invalid')
    if not np.array_equal(tracer0, q[NIC:NIC+9]):
        raise ValueError('baseline nuisance coordinates do not match endpoint A')

    with np.load(SOURCE, allow_pickle=False) as f:
        source_data = {name: f[name].copy() for name in
            ('positions','angular','radial_table','modulus_table','redshift_table')}
    source = source_geometry_at_resolution(source_data, N_SOURCE)
    source_pos, angular = jnp.asarray(source['positions']), jnp.asarray(source['angular'])
    growth = dict(observer=jnp.full(3, BOX/2), box_size_cMpc_h=BOX,
        hubble_km_s_Mpc=74.6, little_h=.746,
        radius_table_cMpc_h=jnp.asarray(source_data['radial_table']),
        modulus_table_h=jnp.asarray(source_data['modulus_table']),
        redshift_table=jnp.asarray(source_data['redshift_table']), grid_size=N_COUNT,
        radial_min_cMpc_h=5., radial_max_cMpc_h=180.)
    with np.load(SPLIT, allow_pickle=False) as f:
        keys = np.asarray(f['train_keys'], dtype=np.int64)
        counts = np.asarray(f['train_counts'], dtype=np.int64)
        exposure, _ = build_population_exposure_masks(N_COUNT,
            f['heldout_flat_voxels'], f['train_window_excluded_keys'], ())

    parent = json.loads((BASE/'r2_n256_dynamics_profile_v1/result.json').read_text())
    if parent.get('status') != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
        raise ValueError('saved N256 PMWD settings unavailable')
    evolve, _, config, _, particle_mass = make_dynamics(parent['settings'])
    particle_masses = jnp.full(NIC, particle_mass)

    @jax.jit
    def recover_field(white):
        pos, vel = evolve(white)
        state = particle_grid(pos, vel, particle_masses, config)
        return state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0)

    field_started = time.monotonic()
    rho_native, velocity_native = recover_field(jnp.asarray(q[:NIC]))
    jax.block_until_ready((rho_native, velocity_native))
    density, velocity = native_mass_momentum_to_count_cells(
        rho_native, velocity_native, BOX)
    velocity = jnp.moveaxis(velocity, 0, -1).reshape(-1, 3)

    candidate_j = jnp.asarray(tracer1)
    masses = tracer_masses(density, candidate_j)*(BOX/N_SOURCE/3.)**3
    geometry = tracer_geometry(candidate_j, growth)
    tic = time.monotonic()

    @jax.jit
    def predict(d, v, m):
        return predict_chunked_volume_intensity(source_pos, v, m, angular,
            source_chunk_size=SOURCE_CHUNK, source_spacing=BOX/N_SOURCE,
            volume_order=2, deposition='tsc', order=4, segments=8, **geometry)

    intensity = predict(density, velocity, masses)
    jax.block_until_ready(intensity)
    intensity_host = np.asarray(jax.device_get(intensity), dtype=np.float64)
    if not np.isfinite(intensity_host).all() or np.any(intensity_host < 0.0):
        raise FloatingPointError('candidate TSC count means are invalid')
    exposure_j = jnp.asarray(exposure)
    keys_j, counts_j = jnp.asarray(keys), jnp.asarray(counts)
    candidate_score = float(sparse_marked_poisson_log_likelihood(
        intensity, keys_j, counts_j, selected_voxel_mask=exposure_j))
    candidate_table = np.zeros_like(baseline_table)
    radial_bin = radial_population_bins()
    flat = intensity_host.reshape(6, -1)
    exp_flat = exposure.reshape(6, -1)
    for pop in range(6):
        for ibin in range(candidate_table.shape[1]):
            selected = exp_flat[pop] & (radial_bin == ibin)
            candidate_table[pop, ibin] = flat[pop, selected].sum(dtype=np.float64)
    observed_table = np.zeros_like(baseline_table, dtype=np.int64)
    for row in profile['initial']['radial_by_population']:
        ibin = int(np.flatnonzero(RADIAL_EDGES == row['radius_lower_cMpc_h'])[0])
        observed_table[row['population'], ibin] = row['observed_count']

    baseline_prior = float(profile['initial']['gaussian_tracer_prior_nll'])
    candidate_prior = 0.5*float(tracer1 @ tracer1)
    baseline_obj = baseline_prior-baseline_score
    candidate_obj = candidate_prior-candidate_score
    baseline_l1 = float(np.sum(np.abs(observed_table-baseline_table)))
    candidate_l1 = float(np.sum(np.abs(observed_table-candidate_table)))
    bright_shells = []
    for lo in (36., 48., 60., 72., 84.):
        ibin = int(np.flatnonzero(RADIAL_EDGES == lo)[0])
        o = observed_table[:3, ibin].astype(np.float64)
        b = baseline_table[:3, ibin]
        c = candidate_table[:3, ibin]
        bright_shells.append(dict(radius_cMpc_h=f'{lo:g}-{lo+12:g}',
            observed_population0_share=float(o[0]/o.sum()),
            baseline_tsc_population0_share=float(b[0]/b.sum()),
            prior_center_tsc_population0_share=float(c[0]/c.sum())))
    report = dict(status='FIXED_FIELD_LF_PRIOR_CENTER_SENSITIVITY_COMPLETE_NOT_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        classification='ONE_SAVED_N256_FIELD_ONE_PRIOR_CENTRE_LF_POINT_TSC_FORWARD_ONLY',
        active_factor='2M++ marked-count factor only; raw-FP factor not reevaluated',
        field_checkpoint=str(CHECKPOINT), profile_reference=str(PROFILE),
        exact_baseline_TSC_table_reference=str(CLOSURE),
        PMWD_forward_replays=1, PMWD_forward_seconds=time.monotonic()-field_started,
        candidate_TSC_seconds=time.monotonic()-tic,
        initial_tracer_coordinates=tracer0.tolist(), candidate_tracer_coordinates=tracer1.tolist(),
        alpha_white_initial=float(tracer0[7]), mstar_white_initial=float(tracer0[8]),
        alpha_white_candidate=0.0, mstar_white_candidate=0.0,
        candidate_mstar=-23.28, candidate_alpha=-.94,
        count_training_log_score=dict(baseline=baseline_score, candidate=candidate_score,
            delta_candidate_minus_baseline=candidate_score-baseline_score),
        Gaussian_tracer_prior_nll=dict(baseline=baseline_prior, candidate=candidate_prior,
            delta_candidate_minus_baseline=candidate_prior-baseline_prior),
        count_plus_prior_conditional_objective=dict(baseline=baseline_obj,
            candidate=candidate_obj, delta_candidate_minus_baseline=candidate_obj-baseline_obj),
        training_aggregate=dict(
            baseline_total_mean=float(baseline_table.sum()),
            candidate_total_mean=float(candidate_table.sum()),
            observed_total=int(observed_table.sum()),
            baseline_radial_population_L1=baseline_l1,
            candidate_radial_population_L1=candidate_l1,
            baseline_pearson_expected_ge5=pearson(observed_table, baseline_table),
            candidate_pearson_expected_ge5=pearson(observed_table, candidate_table),
            bright_share_by_shell=bright_shells),
        heldout_outcome_values_read=False, heldout_scored=False,
        field_fitted=False, chain_transitions=0, PMWD_adjoints=0,
        count_law_changed=False, posterior_claim=False,
        interpretation='Same training count data at two fixed nuisance points; the candidate is the prior centre, not an optimized point. The raw-FP factor is not reevaluated. Any count training-score change is sensitivity only. This does not calibrate selection, luminosity survival/bias, RSD/FoG, or validate the z=0 posterior.',
        Q_GOAL='test whether the current population-split residual is sensitive to prior-centred LF/K transfer on the same actual z=0 field, before further R2 likelihood work',
        Q_LEAN='one saved-field replay and one active-TSC forward count map at one predeclared prior-centre point; no optimizer, chain, heldout, mock or new law',
        MW_M31='remain role-ambiguous; future observables must constrain both roles on the same NEW field',
        M33='unresolved; future observables must constrain it on the same NEW field at <=0.3 cMpc/h',
        native_truth_id_role='calibration/evaluation only')
    report['elapsed_seconds'] = time.monotonic()-started
    report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


def main():
    try:
        _run()
    except Exception as error:
        result = OUT/'result.json'
        if _OUTPUT_CREATED and not result.exists():
            try:
                (OUT/'failure.json').write_text(json.dumps(dict(
                    status='FIXED_FIELD_LF_PRIOR_CENTER_SENSITIVITY_FAILED',
                    error=f'{type(error).__name__}: {error}'), indent=2)+'\n')
            except Exception:
                pass
        raise


if __name__ == '__main__':
    main()
