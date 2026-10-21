#!/usr/bin/env python3
"""One fixed-field voxel-count operator-closure mock; not an R2 fit/calibration."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_linked_fp_sparse_train import SOURCE
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_raw_volume_target import tracer_geometry, tracer_masses
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_resolution_target import source_geometry_at_resolution

jax.config.update('jax_enable_x64', True)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
N_SOURCE, N_COUNT, BOX = 256, 128, 384.0
NIC, NNUISANCE = N_SOURCE**3, 24
SOURCE_CHUNK = 2_097_152
RADIAL_EDGES = np.arange(0.0, 192.0 + 12.0, 12.0)
PROFILE = BASE/'r2_n256_frozen_field_tracer_profile_20261002_v3/result.json'
CHECKPOINT = BASE/'r2_n256_gl2_metric6000_control_a_v1/chain_a_accepted_checkpoint.npz'
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
OUT = Path(os.environ.get('CF4_R2_CLOSURE_OUT_DIR',
    str(BASE/'r2_n256_count_operator_closure_20261002_v1')))
SEED = 20261002
_OUTPUT_CREATED = False


def radial_population_bins(grid_size=N_COUNT, box_size=BOX, edges=RADIAL_EDGES):
    """Use the existing observed-cell-centre assignment, labeling the final bin as a tail."""
    flat = np.arange(grid_size**3, dtype=np.int64)
    i, j, k = flat//grid_size**2, (flat//grid_size) % grid_size, flat % grid_size
    scale = float(box_size)/grid_size
    x, y, z = (i+.5)*scale-box_size/2, (j+.5)*scale-box_size/2, (k+.5)*scale-box_size/2
    radius = np.sqrt(x*x+y*y+z*z)
    return np.clip(np.searchsorted(edges, radius, side='right')-1,
                   0, len(edges)-2).astype(np.int16)


def aggregate_expected(intensity, exposure, radial_bin):
    expected = np.asarray(intensity, dtype=np.float64).reshape(6, -1)
    mask = np.asarray(exposure, dtype=bool).reshape(6, -1)
    out = np.zeros((6, len(RADIAL_EDGES)-1), dtype=np.float64)
    for pop in range(6):
        for rbin in range(out.shape[1]):
            selected = mask[pop] & (radial_bin == rbin)
            out[pop, rbin] = expected[pop, selected].sum(dtype=np.float64)
    return out


def aggregate_observed(counts, radial_bin):
    count = np.asarray(counts, dtype=np.int64).reshape(6, -1)
    out = np.zeros((6, len(RADIAL_EDGES)-1), dtype=np.int64)
    for pop in range(6):
        for rbin in range(out.shape[1]):
            out[pop, rbin] = count[pop, radial_bin == rbin].sum(dtype=np.int64)
    return out


def table_rows(observed, expected_tsc, expected_ngp):
    rows = []
    for pop in range(6):
        for rbin in range(len(RADIAL_EDGES)-1):
            lo, hi = RADIAL_EDGES[rbin:rbin+2]
            o, et, en = int(observed[pop, rbin]), float(expected_tsc[pop, rbin]), float(expected_ngp[pop, rbin])
            rows.append(dict(population=pop,
                radius_lower_cMpc_h=float(lo),
                radius_upper_cMpc_h=(None if rbin == len(RADIAL_EDGES)-2 else float(hi)),
                radius_bin_label=(f'>={lo:g}' if rbin == len(RADIAL_EDGES)-2 else f'{lo:g}-{hi:g}'),
                mock_count=o, expected_ngp=en, expected_tsc=et,
                mock_to_ngp=float(o/en) if en > 0 else None,
                mock_to_tsc=float(o/et) if et > 0 else None))
    return rows


def pearson_summary(observed, expected, minimum_expected=5.0):
    obs, exp = np.asarray(observed, dtype=np.float64).ravel(), np.asarray(expected, dtype=np.float64).ravel()
    use = exp >= minimum_expected
    if not np.any(use):
        raise ValueError('no aggregate bins meet the predeclared Pearson expected-count threshold')
    value = float(np.sum((obs[use]-exp[use])**2/exp[use]))
    dof = int(use.sum())
    probability = float(chi2.sf(value, dof))
    lower, upper = (float(chi2.ppf(.025, dof)), float(chi2.ppf(.975, dof)))
    return dict(statistic=value, degrees_of_freedom=dof, p_value=probability,
        central_95_interval=[lower, upper], inside_central_95=(lower <= value <= upper),
        expected_count_threshold=minimum_expected,
        interpretation='aggregate one-mock Pearson diagnostic; approximate chi-square reference, not validation')


def bright_population_zero_share(observed, expected):
    rows = []
    for lo in (48., 60., 72., 96.):
        rbin = int(np.flatnonzero(RADIAL_EDGES == lo)[0])
        obs = np.asarray(observed[:3, rbin], dtype=np.float64)
        exp = np.asarray(expected[:3, rbin], dtype=np.float64)
        n = float(obs.sum()); total = float(exp.sum())
        p = float(exp[0]/total) if total > 0 else None
        share = float(obs[0]/n) if n > 0 else None
        z = ((share-p)/np.sqrt(p*(1-p)/n) if n > 0 and p is not None and 0 < p < 1 else None)
        rows.append(dict(radius_cMpc_h=f'{lo:g}-{lo+12:g}',
            population0_observed_share=share, population0_model_share=p,
            bright_mock_count=int(n), poisson_standardized_difference=float(z) if z is not None else None))
    return rows


def sparse_log_score(intensity, keys, counts, exposure):
    flat = np.asarray(intensity, dtype=np.float64).reshape(-1)
    key = np.asarray(keys, dtype=np.int64); count = np.asarray(counts, dtype=np.int64)
    if np.any(flat[key] <= 0) or not np.isfinite(flat).all():
        return float('-inf')
    from scipy.special import gammaln
    return float(np.sum(count*np.log(flat[key])-gammaln(count+1.))
        - np.sum(flat.reshape(6, -1)*np.asarray(exposure).reshape(6, -1)))


def _run():
    global _OUTPUT_CREATED
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('submit this bounded forward-only calculation to Slurm with a GPU')
    OUT.mkdir(parents=True, exist_ok=False)
    _OUTPUT_CREATED = True
    started = time.monotonic()
    report = dict(status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        classification='ONE_FIXED_FIELD_VOXEL_COUNT_OPERATOR_CLOSURE_NOT_CALIBRATION',
        field_checkpoint=str(CHECKPOINT), profile_reference=str(PROFILE),
        fixed_tracer_coordinates='saved endpoint A initial nine count-tracer coordinates, not the profiled values',
        synthetic_poisson_seed=SEED, source_volume_order=2, los_order=4, los_segments=8,
        observed_count_voxel_rule='floor(x/dx), periodic, matching cf4_r2_common_catalogue.py',
        model_mean_voxel_rule='active TSC versus diagnostic NGP; same source, LF-transfer, angular selection, RSD/FoG and radial cut',
        synthetic_count_rule='independent Poisson counts from the fixed-field NGP mean on graph-closed training-exposed cells only',
        heldout_outcome_values_read=False, heldout_scored=False, actual_rows_generated=False,
        count_law_changed=False, field_fitted=False, chain_transitions=0, PMWD_adjoints=0,
        Q_GOAL='test whether the known TSC-prediction versus NGP-catalogue mapping can explain the current R2 population-by-radius training residual; no LG identification',
        Q_LEAN='one saved N256 field, one PMWD forward replay, two forward count operators, one seeded Poisson voxel-count draw; no adjoint, fit, sampler, heldout outcome or raw-object mock archive',
        MW_M31='remain role-ambiguous; future observables must constrain these roles on the same NEW field',
        M33='unresolved; future observables must constrain M33 on the same NEW field at <=0.3 cMpc/h')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        tmp = OUT/'result.json.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(OUT/'result.json')

    save()
    parent = json.loads((BASE/'r2_n256_dynamics_profile_v1/result.json').read_text())
    profile = json.loads(PROFILE.read_text())
    if parent.get('status') != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
        raise ValueError('saved N256 PMWD settings unavailable')
    if profile.get('status') != 'FROZEN_FIELD_TRACER_PROFILE_BOUNDED_NOT_CONVERGED':
        raise ValueError('the bounded frozen-field profile did not complete as expected')
    with np.load(CHECKPOINT, allow_pickle=False) as f:
        q = np.asarray(f['canonical'], dtype=np.float64)
        completed = int(f['completed_transitions'])
    if q.shape != (NIC+NNUISANCE,) or completed != 1 or not np.isfinite(q).all():
        raise ValueError('saved A endpoint shape/status mismatch')
    tracer = np.asarray(profile['initial']['tracer_coordinates'], dtype=np.float64)
    if not np.array_equal(tracer, q[NIC:NIC+9]):
        raise ValueError('mock nuisance values must exactly equal frozen endpoint A start')

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
    radial_bin = radial_population_bins()

    evolve, _, config, _, particle_mass = make_dynamics(parent['settings'])
    particle_masses = jnp.full(NIC, particle_mass)

    @jax.jit
    def recover_field(white):
        pos, vel = evolve(white)
        state = particle_grid(pos, vel, particle_masses, config)
        return state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0)

    tic = time.monotonic()
    rho_native, velocity_native = recover_field(jnp.asarray(q[:NIC]))
    jax.block_until_ready((rho_native, velocity_native))
    density, velocity = native_mass_momentum_to_count_cells(rho_native, velocity_native, BOX)
    velocity = jnp.moveaxis(velocity, 0, -1).reshape(-1, 3)
    report.update(PMWD_forward_replays=1, PMWD_forward_seconds=time.monotonic()-tic)
    save()

    tracer_j = jnp.asarray(tracer)
    masses = tracer_masses(density, tracer_j)*(BOX/N_SOURCE/3.)**3
    geometry = tracer_geometry(tracer_j, growth)
    key_j, count_j, exposure_j = jnp.asarray(keys), jnp.asarray(counts), jnp.asarray(exposure)

    means = {}
    for deposition in ('tsc','ngp'):
        tic = time.monotonic()
        @jax.jit
        def predict(d, v, m):
            return predict_chunked_volume_intensity(source_pos, v, m, angular,
                source_chunk_size=SOURCE_CHUNK, source_spacing=BOX/N_SOURCE,
                volume_order=2, order=4, segments=8, deposition=deposition,
                **geometry)
        predicted = predict(density, velocity, masses)
        jax.block_until_ready(predicted)
        host = np.asarray(jax.device_get(predicted), dtype=np.float64)
        if not np.isfinite(host).all() or np.any(host < 0.):
            raise FloatingPointError(f'{deposition}: predicted count mean invalid')
        means[deposition] = host
        report[f'{deposition}_forward_seconds'] = time.monotonic()-tic
        report[f'{deposition}_exposed_total_mean'] = float(
            (host.reshape(6,-1)*exposure.reshape(6,-1)).sum(dtype=np.float64))
        del predicted
        save()

    score_tsc = float(sparse_marked_poisson_log_likelihood(means['tsc'], key_j,
        count_j, selected_voxel_mask=exposure_j))
    expected_tsc = aggregate_expected(means['tsc'], exposure, radial_bin)
    expected_ngp = aggregate_expected(means['ngp'], exposure, radial_bin)
    expected_reference = np.asarray([[r['expected_training_count'] for r in rows]
        for rows in [[x for x in profile['initial']['radial_by_population'] if x['population']==p]
                     for p in range(6)]], dtype=np.float64)
    score_error = abs(score_tsc-float(profile['initial']['count_log_score']))
    table_error = float(np.max(np.abs(expected_tsc-expected_reference)))
    if score_error > 1e-7 or table_error > 1e-6:
        raise AssertionError('TSC reference did not reproduce the saved A training score/table')

    generator = np.random.default_rng(SEED)
    mock_counts = np.zeros((6,N_COUNT**3), dtype=np.int64)
    tsc_flat, ngp_flat = means['tsc'].reshape(6,-1), means['ngp'].reshape(6,-1)
    exposure_flat = exposure.reshape(6,-1)
    for pop in range(6):
        ids = np.flatnonzero(exposure_flat[pop])
        rate = ngp_flat[pop,ids]
        if not np.isfinite(rate).all() or np.any(rate < 0.):
            raise FloatingPointError('NGP training exposure has invalid Poisson rates')
        mock_counts[pop,ids] = generator.poisson(rate)
    mock_table = aggregate_observed(mock_counts, radial_bin)
    rows = table_rows(mock_table, expected_tsc, expected_ngp)
    pearson_ngp = pearson_summary(mock_table, expected_ngp)
    pearson_tsc = pearson_summary(mock_table, expected_tsc)
    tsc_bins = expected_tsc.ravel(); ngp_bins = expected_ngp.ravel()
    use = tsc_bins >= 5.
    mean_operator_delta = float(np.sum((ngp_bins[use]-tsc_bins[use])**2/tsc_bins[use]))
    high_bright_ngp = bright_population_zero_share(mock_table, expected_ngp)
    high_bright_tsc = bright_population_zero_share(mock_table, expected_tsc)
    edge = [r for r in rows if r['radius_bin_label'] == '>=180']
    mock_keys = np.flatnonzero(mock_counts.ravel()).astype(np.int64)
    mock_values = mock_counts.ravel()[mock_keys]
    mock_score_tsc = sparse_log_score(means['tsc'], mock_keys, mock_values, exposure)
    mock_score_ngp = sparse_log_score(means['ngp'], mock_keys, mock_values, exposure)
    report.update(status='COUNT_OPERATOR_CLOSURE_COMPLETE_NOT_CALIBRATION',
        tsc_reference=dict(training_count_log_score=score_tsc,
            absolute_score_error_to_saved_A=score_error,
            max_radial_population_mean_error_to_saved_A=table_error,
            exact_reference_gate_passed=True),
        synthetic_mock=dict(seed=SEED, observed_training_count=int(mock_counts.sum()),
            nonzero_population_voxel_keys=int(len(mock_keys)),
            poisson_check_against_generating_ngp_mean=pearson_ngp,
            poisson_check_against_active_tsc_mean=pearson_tsc,
            approximate_noncentrality_from_mean_operator_difference=mean_operator_delta,
            log_score_under_generating_ngp=mock_score_ngp,
            log_score_under_active_tsc=mock_score_tsc,
            bright_population0_share_against_ngp=high_bright_ngp,
            bright_population0_share_against_tsc=high_bright_tsc,
            edge_bins_ge_180=edge),
        radial_population_counts=rows,
        TSC_vs_NGP_mean_table_max_abs_difference=float(np.max(np.abs(expected_tsc-expected_ngp))),
        interpretation='one fixed saved field and initial endpoint-A tracer state; this isolates the voxel-deposition/count-operator difference under the existing analytic source-mark transfer. It does not validate the transfer law, selection/bias/survival, RSD/FoG calibration, source-group dependence, actual CF4 prediction, posterior or LG roles.',
        heldout_outcome_values_read=False, heldout_scored=False,
        actual_rows_generated=False, actual_count_law_changed=False)
    save()
    print(json.dumps(report, allow_nan=False), flush=True)


def main():
    try:
        _run()
    except Exception as error:
        result = OUT/'result.json'
        if _OUTPUT_CREATED and result.exists():
            try:
                report = json.loads(result.read_text())
                report.update(status='COUNT_OPERATOR_CLOSURE_FAILED_NOT_CALIBRATION',
                              error=f'{type(error).__name__}: {error}')
                temp = OUT/'result.json.tmp'
                temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                temp.replace(result)
            except Exception:
                pass
        raise


if __name__ == '__main__':
    main()
