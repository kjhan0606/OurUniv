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
    str(BASE/'r2_n256_lf_prior_center_sensitivity_20261002_v2')))
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


def rate_scaled_score(score, observed_total, expected_total, scale):
    if min(observed_total, expected_total, scale) <= 0.0:
        raise ValueError('positive observed/expected totals and rate scale required')
    return float(score + observed_total*np.log(scale) - (scale-1.0)*expected_total)


def aggregate_score_delta(observed, baseline_expected, candidate_expected):
    obs = np.asarray(observed, dtype=np.float64)
    base = np.asarray(baseline_expected, dtype=np.float64)
    cand = np.asarray(candidate_expected, dtype=np.float64)
    if obs.shape != base.shape or base.shape != cand.shape or np.any(base < 0) or np.any(cand < 0):
        raise ValueError('invalid observed/baseline/candidate aggregate tables')
    if np.any((obs > 0) & ((base <= 0) | (cand <= 0))):
        raise ValueError('positive observed aggregate has zero expected count')
    positive = obs > 0
    log_ratio = np.zeros_like(obs)
    log_ratio[positive] = np.log(cand[positive]/base[positive])
    return float(np.sum(obs*log_ratio - (cand-base)))


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
    field_seconds = time.monotonic()-field_started

    geometry = tracer_geometry(jnp.asarray(tracer1), growth)

    @jax.jit
    def predict(d, v, m):
        return predict_chunked_volume_intensity(source_pos, v, m, angular,
            source_chunk_size=SOURCE_CHUNK, source_spacing=BOX/N_SOURCE,
            volume_order=2, deposition='tsc', order=4, segments=8, **geometry)

    def expected_table(host_intensity):
        table = np.zeros_like(baseline_table)
        flat = host_intensity.reshape(6, -1)
        exp_flat = exposure.reshape(6, -1)
        radial_bin = radial_population_bins()
        for pop in range(6):
            for ibin in range(table.shape[1]):
                selected = exp_flat[pop] & (radial_bin == ibin)
                table[pop, ibin] = flat[pop, selected].sum(dtype=np.float64)
        return table

    observed_table = np.zeros_like(baseline_table, dtype=np.int64)
    for row in profile['initial']['radial_by_population']:
        ibin = int(np.flatnonzero(RADIAL_EDGES == row['radius_lower_cMpc_h'])[0])
        observed_table[row['population'], ibin] = row['observed_count']
    observed_total = int(observed_table.sum())
    if int(counts.sum()) != observed_total:
        raise ValueError('saved profile and training split disagree on training total')
    baseline_mean_total = float(baseline_table.sum())
    if abs(baseline_mean_total-float(closure['tsc_exposed_total_mean'])) > 1e-6:
        raise ValueError('baseline radial table does not sum to the saved TSC exposure mean')
    exposure_j = jnp.asarray(exposure)
    keys_j, counts_j = jnp.asarray(keys), jnp.asarray(counts)

    tic = time.monotonic()
    candidate_masses = tracer_masses(density, jnp.asarray(tracer1))*(BOX/N_SOURCE/3.)**3
    candidate_intensity = predict(density, velocity, candidate_masses)
    jax.block_until_ready(candidate_intensity)
    candidate_host = np.asarray(jax.device_get(candidate_intensity), dtype=np.float64)
    if not np.isfinite(candidate_host).all() or np.any(candidate_host < 0.0):
        raise FloatingPointError('prior-centre TSC count means are invalid')
    candidate_score = float(sparse_marked_poisson_log_likelihood(
        candidate_intensity, keys_j, counts_j, selected_voxel_mask=exposure_j))
    candidate_table = expected_table(candidate_host)
    candidate_mean_total = float(candidate_table.sum())
    candidate_tsc_seconds = time.monotonic()-tic
    del candidate_intensity, candidate_masses

    candidate_scale = observed_total/candidate_mean_total
    candidate_delta_u0 = 0.5*float(np.log(candidate_scale))
    tracer_profiled = tracer1.copy()
    tracer_profiled[0] += candidate_delta_u0
    tic = time.monotonic()
    profiled_masses = tracer_masses(density, jnp.asarray(tracer_profiled))*(BOX/N_SOURCE/3.)**3
    profiled_intensity = predict(density, velocity, profiled_masses)
    jax.block_until_ready(profiled_intensity)
    profiled_host = np.asarray(jax.device_get(profiled_intensity), dtype=np.float64)
    if not np.isfinite(profiled_host).all() or np.any(profiled_host < 0.0):
        raise FloatingPointError('rate-profiled TSC count means are invalid')
    profiled_score = float(sparse_marked_poisson_log_likelihood(
        profiled_intensity, keys_j, counts_j, selected_voxel_mask=exposure_j))
    profiled_table = expected_table(profiled_host)
    profiled_mean_total = float(profiled_table.sum())
    profiled_tsc_seconds = time.monotonic()-tic
    rate_identity_error = float(np.max(np.abs(profiled_host-candidate_host*candidate_scale)))
    if rate_identity_error > 1e-8*max(1.0,float(np.max(profiled_host))):
        raise AssertionError('explicit prior-centre TSC map failed the exp(2u0) scaling identity')

    predicted_profiled_score = rate_scaled_score(
        candidate_score, observed_total, candidate_mean_total, candidate_scale)
    score_identity_error = abs(profiled_score-predicted_profiled_score)
    if score_identity_error > 1e-6:
        raise AssertionError('explicit TSC score disagrees with scalar-rate Poisson identity')

    baseline_scale = observed_total/baseline_mean_total
    baseline_delta_u0 = 0.5*float(np.log(baseline_scale))
    baseline_profiled_table = baseline_table*baseline_scale
    baseline_profiled_score = rate_scaled_score(
        baseline_score, observed_total, baseline_mean_total, baseline_scale)
    baseline_prior = float(profile['initial']['gaussian_tracer_prior_nll'])
    baseline_profiled_prior = baseline_prior + 0.5*((tracer0[0]+baseline_delta_u0)**2-tracer0[0]**2)
    candidate_prior = 0.5*float(tracer1 @ tracer1)
    profiled_prior = candidate_prior + 0.5*((tracer_profiled[0]-tracer1[0])
        *(tracer_profiled[0]+tracer1[0]))
    baseline_obj = baseline_profiled_prior-baseline_profiled_score
    candidate_obj = profiled_prior-profiled_score
    aggregate_delta = aggregate_score_delta(
        observed_table, baseline_profiled_table, profiled_table)
    full_delta = profiled_score-baseline_profiled_score
    within_bin_delta = full_delta-aggregate_delta

    def shares_and_metrics(table):
        bright = []
        for lo in (36., 48., 60., 72., 84.):
            ibin = int(np.flatnonzero(RADIAL_EDGES == lo)[0])
            o = observed_table[:3, ibin].astype(np.float64)
            m = table[:3, ibin]
            bright.append(dict(radius_cMpc_h=f'{lo:g}-{lo+12:g}',
                observed_population0_share=float(o[0]/o.sum()),
                model_population0_share=float(m[0]/m.sum())))
        return dict(total_mean=float(table.sum()),
            radial_population_L1=float(np.sum(np.abs(observed_table-table))),
            pearson_expected_ge5=pearson(observed_table, table),
            bright_population0_share_by_shell=bright,
            radial_population_mean_table=table.tolist())

    report = dict(status='FIXED_FIELD_LF_RATE_PROFILE_SENSITIVITY_COMPLETE_NOT_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        classification='ONE_SAVED_N256_FIELD_ONE_PRIOR_CENTRE_LF_POINT_TSC_FORWARD_ONLY',
        active_factor='2M++ marked-count factor only; raw-FP factor not reevaluated',
        field_checkpoint=str(CHECKPOINT), profile_reference=str(PROFILE),
        exact_baseline_TSC_table_reference=str(CLOSURE),
        PMWD_forward_replays=1, PMWD_forward_seconds=field_seconds,
        candidate_unprofiled_TSC_seconds=candidate_tsc_seconds,
        candidate_rate_profiled_TSC_seconds=profiled_tsc_seconds,
        initial_tracer_coordinates=tracer0.tolist(), candidate_tracer_coordinates=tracer1.tolist(),
        rate_profiled_candidate_tracer_coordinates=tracer_profiled.tolist(),
        alpha_white_initial=float(tracer0[7]), mstar_white_initial=float(tracer0[8]),
        alpha_white_candidate=0.0, mstar_white_candidate=0.0,
        candidate_mstar=-23.28, candidate_alpha=-.94,
        scalar_total_rate_profile=dict(
            observed_training_total=observed_total,
            prior_center_unprofiled_expected=candidate_mean_total,
            prior_center_scale=float(candidate_scale),
            prior_center_delta_u0=float(candidate_delta_u0),
            profiled_baseline_scale=float(baseline_scale),
            profiled_baseline_delta_u0=float(baseline_delta_u0),
            prior_center_expected_after_explicit_TSC=profiled_mean_total,
            max_abs_TSC_map_error_to_scalar_identity=rate_identity_error,
            explicit_vs_analytic_score_abs_error=score_identity_error,
            candidate_score_profiled_by_total_rate=float(predicted_profiled_score)),
        count_training_log_score=dict(
            baseline_unprofiled=baseline_score,
            baseline_rate_profiled=baseline_profiled_score,
            candidate_prior_center_unprofiled=candidate_score,
            candidate_prior_center_rate_profiled=profiled_score,
            delta_rate_profiled_candidate_minus_baseline=full_delta),
        Gaussian_tracer_prior_nll=dict(
            baseline_unprofiled=baseline_prior,
            baseline_rate_profiled=baseline_profiled_prior,
            candidate_prior_center_unprofiled=candidate_prior,
            candidate_prior_center_rate_profiled=profiled_prior,
            delta_rate_profiled_candidate_minus_baseline=profiled_prior-baseline_profiled_prior),
        count_plus_prior_conditional_objective=dict(
            baseline_unprofiled=baseline_prior-baseline_score,
            baseline_rate_profiled=baseline_obj,
            candidate_prior_center_unprofiled=candidate_prior-candidate_score,
            candidate_prior_center_rate_profiled=candidate_obj,
            delta_rate_profiled_candidate_minus_baseline=candidate_obj-baseline_obj),
        score_delta_decomposition=dict(
            full_voxel_delta=full_delta,
            radial_x_population_aggregate_delta=aggregate_delta,
            within_aggregate_bin_conditional_allocation_delta=within_bin_delta,
            sum_check=aggregate_delta+within_bin_delta),
        training_aggregate=dict(
            observed_total=observed_total,
            observed_radial_population_count_table=observed_table.tolist(),
            baseline_unprofiled=shares_and_metrics(baseline_table),
            baseline_rate_profiled=shares_and_metrics(baseline_profiled_table),
            candidate_prior_center_unprofiled=shares_and_metrics(candidate_table),
            candidate_prior_center_rate_profiled=shares_and_metrics(profiled_table)),
        heldout_outcome_values_read=False, heldout_scored=False,
        field_fitted=False, chain_transitions=0, PMWD_adjoints=0,
        count_law_changed=False, posterior_claim=False,
        interpretation='Same training count data and one fixed field. Compare endpoint-A baseline with one predeclared prior-centre alpha/mstar point, using a one-dimensional total-rate profile; all five bias coordinates and sigma_los remain fixed. The raw-FP factor is not reevaluated. Training-score change is sensitivity only and does not calibrate selection, luminosity survival/bias, RSD/FoG, or validate the z=0 posterior.',
        Q_GOAL='test whether the current population-split residual is sensitive to prior-centred LF/K transfer on the same actual z=0 field, before further R2 likelihood work',
        Q_LEAN='one saved-field replay, one prior-centre LF point, and its exact scalar-rate TSC rescaling; no optimizer, chain, heldout, mock or new law',
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
