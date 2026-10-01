#!/usr/bin/env python3
"""Compare same-state R2 training counts with their selected count-model mean.

This is a deterministic PMWD forward replay of two already saved accepted
states. It reads training count values and heldout/buffer geometry only to
construct the original training exposure; heldout outcomes are never read.
It does not fit, sample, score heldout data, or modify the observation law.
"""
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


BASE = Path('/gpfs/kjhan/CF4/z0_density')
N_SOURCE = 256
N_COUNT = 128
BOX = 384.0
NIC = N_SOURCE**3
NNUISANCE = 24
CHUNK = 2_097_152
CHECKPOINTS = {
    'a': BASE/'r2_n256_gl2_metric6000_control_a_v1/chain_a_accepted_checkpoint.npz',
    'b': BASE/'r2_n256_gl2_metric6000_control_b_v1/chain_b_accepted_checkpoint.npz',
}
COMPONENT_REPORT = BASE/'r2_n256_lowk_component_attribution_20261002_v2/result.json'
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
OUT = Path(os.environ.get('CF4_R2_OUT_DIR',
    str(BASE/'r2_n256_training_count_residual_20261002_v1')))
RADIAL_EDGES = np.arange(0.0, 192.0 + 12.0, 12.0)
INTENSITY_QUANTILES = 5


def _ratio(observed, expected):
    return float(observed/expected) if expected > 0.0 else None


def _residual(observed, expected):
    return float((observed-expected)/np.sqrt(expected)) if expected > 0.0 else None


def summarize_training_counts(intensity, keys, counts, exposure, *, grid_size,
                              box_size, radial_edges=RADIAL_EDGES,
                              intensity_quantiles=INTENSITY_QUANTILES):
    """Aggregate observed/predicted counts over exposed cells only.

    Radial distance is the radius of the observed count-cell centre, not a
    latent true distance. Intensity quantiles are defined within each
    population/radial bin from the model means in its training-exposed cells.
    Returned residuals are descriptive aggregate discrepancies, not
    independent-cell tests or heldout predictions.
    """
    n = int(grid_size)
    nvoxel = n**3
    expected = np.asarray(intensity, dtype=np.float64)
    if expected.shape != (6, n, n, n):
        raise ValueError('intensity must have shape (6, grid, grid, grid)')
    expected = expected.reshape(6, nvoxel)
    key = np.asarray(keys)
    observed_count = np.asarray(counts, dtype=np.int64)
    train = np.asarray(exposure, dtype=bool)
    edges = np.asarray(radial_edges, dtype=np.float64)
    if key.ndim != 1 or observed_count.shape != key.shape:
        raise ValueError('sparse training keys/counts must be aligned vectors')
    if train.shape != (6*nvoxel,):
        raise ValueError('population-major training exposure has the wrong size')
    if (not np.isfinite(expected).all() or np.any(expected < 0.0)
            or np.any(observed_count < 0) or not np.isfinite(edges).all()
            or len(edges) < 2 or np.any(np.diff(edges) <= 0.0)
            or intensity_quantiles < 2):
        raise ValueError('invalid expected counts, observations, bins, or quantiles')
    if (np.any(key < 0) or np.any(key >= 6*nvoxel)
            or not np.issubdtype(key.dtype, np.integer)):
        raise ValueError('training keys are outside population-major geometry')
    if len(np.unique(key)) != len(key):
        raise ValueError('training count keys must already be coalesced')
    if len(key) and not np.all(train[key]):
        raise ValueError('an observed training key is outside training exposure')

    flat = np.arange(nvoxel, dtype=np.int64)
    ijk = np.column_stack((flat//(n*n), (flat//n) % n, flat % n))
    coordinate = (ijk.astype(np.float64)+.5)*(float(box_size)/n)-float(box_size)/2.0
    radius = np.linalg.norm(coordinate, axis=1)
    radial_bin = np.searchsorted(edges, radius, side='right')-1
    radial_bin[radius == edges[-1]] = len(edges)-2
    in_radial_range = (radial_bin >= 0) & (radial_bin < len(edges)-1)
    # Bit 0/1/2 indicate x/y/z >= 0. Names below state signs explicitly.
    sector = ((coordinate[:, 0] >= 0).astype(np.uint8)
              | ((coordinate[:, 1] >= 0).astype(np.uint8) << 1)
              | ((coordinate[:, 2] >= 0).astype(np.uint8) << 2))
    sector_names = ('---', '+--', '-+-', '++-', '--+', '+-+', '-++', '+++')
    key_population = key//nvoxel
    key_voxel = key % nvoxel
    key_radial = radial_bin[key_voxel]
    key_sector = sector[key_voxel]
    obs_in_range = (key_radial >= 0) & (key_radial < len(edges)-1)
    if len(key) and not np.all(obs_in_range):
        raise ValueError('an observed training key lies outside the declared radial bins')

    def row(population, rbin, sector_id, intensity_bin, cell_ids, cell_values,
            observed_value, upper_edges=()):
        prediction = float(np.sum(cell_values, dtype=np.float64))
        return dict(population=int(population),
            radius_lower_cMpc_h=float(edges[rbin]),
            radius_upper_cMpc_h=float(edges[rbin+1]),
            sector=None if sector_id is None else sector_names[sector_id],
            model_intensity_quantile=intensity_bin,
            exposed_cells=int(len(cell_ids)), observed_count=int(observed_value),
            expected_training_count=prediction,
            observed_to_expected=_ratio(float(observed_value), prediction),
            aggregate_poisson_residual_descriptive_only=_residual(
                float(observed_value), prediction),
            quantile_upper_edges=None if upper_edges is None else
                [float(x) for x in upper_edges])

    radial_rows = []
    sector_rows = []
    quantile_rows = []
    total_observed = int(observed_count.sum())
    total_expected = 0.0
    training_exposed_cells = 0
    for pop in range(6):
        pop_exposure = train.reshape(6, nvoxel)[pop]
        pop_expected = expected[pop]
        total_expected += float(pop_expected[pop_exposure].sum(dtype=np.float64))
        training_exposed_cells += int(pop_exposure.sum())
        for rbin in range(len(edges)-1):
            in_shell = in_radial_range & (radial_bin == rbin)
            cells = np.flatnonzero(pop_exposure & in_shell)
            row_obs = (key_population == pop) & obs_in_range & (key_radial == rbin)
            obs_shell = int(observed_count[row_obs].sum(dtype=np.int64))
            radial_rows.append(row(pop, rbin, None, None, cells,
                                   pop_expected[cells], obs_shell))

            for sector_id in range(8):
                sector_cells = cells[sector[cells] == sector_id]
                row_sector = row_obs & (key_sector == sector_id)
                sector_rows.append(row(pop, rbin, sector_id, None, sector_cells,
                    pop_expected[sector_cells],
                    int(observed_count[row_sector].sum(dtype=np.int64))))

            if not len(cells):
                continue
            values = pop_expected[cells]
            cuts = np.unique(np.quantile(values,
                np.arange(1, intensity_quantiles)/intensity_quantiles))
            assignments = np.searchsorted(cuts, values, side='right')
            cell_obs = row_obs & (key_population == pop)
            obs_voxels = key_voxel[cell_obs]
            obs_values = observed_count[cell_obs]
            if len(obs_voxels):
                location = np.searchsorted(cells, obs_voxels)
                if (np.any(location >= len(cells))
                        or not np.array_equal(cells[location], obs_voxels)):
                    raise AssertionError('observed count was not mapped to an exposed shell cell')
                observed_by_quantile = np.bincount(assignments[location],
                    weights=obs_values.astype(np.float64),
                    minlength=len(cuts)+1).astype(np.int64)
            else:
                observed_by_quantile = np.zeros(len(cuts)+1, dtype=np.int64)
            for qbin in range(len(cuts)+1):
                selected = assignments == qbin
                selected_cells = cells[selected]
                if not len(selected_cells):
                    continue
                upper = cuts[qbin:qbin+1] if qbin < len(cuts) else ()
                quantile_rows.append(row(pop, rbin, None, qbin, selected_cells,
                    values[selected], int(observed_by_quantile[qbin]), upper))

    if int(train.sum()) != training_exposed_cells:
        raise AssertionError('training exposure accounting changed')
    return dict(
        radius_definition='observed count-cell centre, relative to [192,192,192] cMpc/h; not latent true distance',
        radial_edges_cMpc_h=edges.tolist(),
        sector_definition='geometric sign octants relative to box centre; only training-exposed cells included',
        intensity_quantile_definition=(
            f'{intensity_quantiles} requested quantiles within each population/radial bin; '
            'duplicate cut values are merged, and rows list resulting bins'),
        total=dict(observed_training_count=total_observed,
            expected_training_count=total_expected,
            observed_to_expected=_ratio(float(total_observed), total_expected),
            training_exposed_population_cells=training_exposed_cells,
            training_exposure_fraction_by_population=(
                train.reshape(6, nvoxel).mean(axis=1).astype(float).tolist())),
        radial_by_population=radial_rows,
        radial_population_by_exposed_octant=sector_rows,
        radial_population_by_model_intensity_quantile=quantile_rows,
        interpretation='same-state fitted training residual only; aggregate Poisson residual is descriptive, not a significance test or heldout prediction')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('run under Slurm with a GPU')
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        classification='SAME_STATE_TRAINING_COUNT_RESIDUAL_NOT_POSTERIOR',
        grid_source=N_SOURCE, grid_observed=N_COUNT, box_cMpc_h=BOX,
        source_resolution_cMpc_h=BOX/N_SOURCE,
        deterministic_PM_forward_replays=0, chain_transitions=0,
        heldout_count_or_mark_values_read=False, heldout_scored=False,
        split_geometry_used_only_to_construct_training_exposure=True,
        fitted_values_or_observation_law_changed=False,
        Q_GOAL='locate the R2 partial count-factor mismatch at the same accepted z=0 field states before revising the model; this is upstream of same-new-field LG conditioning, not MW/M31/M33 identification',
        Q_LEAN='two saved states, two deterministic field replays and count predictions only; no adjoints, chains, heldout scores, map promotion, or independent gravity simulation',
        MW_M31='remain role-ambiguous; no truth identity used to seed/select a field',
        M33='unresolved; eventual observables must constrain these roles on the same NEW field at <=0.3 cMpc/h',
        states={})

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        temporary = OUT/'result.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(OUT/'result.json')

    save()
    try:
        parent_report = json.loads((BASE/'r2_n256_dynamics_profile_v1/result.json').read_text())
        if parent_report.get('status') != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
            raise ValueError('frozen N256 PMWD configuration is unavailable')
        component = json.loads(COMPONENT_REPORT.read_text())
        if component.get('status') != 'COMPONENT_ATTRIBUTION_COMPLETE_NOT_POSTERIOR':
            raise ValueError('same-state count component reference is unavailable')
        settings = parent_report['settings']
        evolve, _, config, _, particle_mass = make_dynamics(settings)
        particle_mass_array = jnp.full(NIC, particle_mass)

        with np.load(SOURCE, allow_pickle=False) as archive:
            source_table = {name: archive[name].copy() for name in
                ('positions', 'angular', 'radial_table', 'modulus_table', 'redshift_table')}
        source_xy = source_geometry_at_resolution(source_table, N_SOURCE)
        source = {**source_xy,
            'radial_table': source_table['radial_table'],
            'modulus_table': source_table['modulus_table'],
            'redshift_table': source_table['redshift_table']}
        source = {name: jnp.asarray(value) for name, value in source.items()}

        with np.load(SPLIT, allow_pickle=False) as split:
            # Only training count values plus frozen split/exposure geometry.
            # Heldout outcome arrays are intentionally never accessed.
            keys = np.asarray(split['train_keys'], dtype=np.int64)
            counts = np.asarray(split['train_counts'], dtype=np.int64)
            training_exposure, _ = build_population_exposure_masks(
                N_COUNT, split['heldout_flat_voxels'],
                split['train_window_excluded_keys'], split['heldout_window_excluded_keys'])
        exposure_jax = jnp.asarray(training_exposure)
        keys_jax, counts_jax = jnp.asarray(keys), jnp.asarray(counts)
        growth = dict(observer=jnp.full(3, BOX/2), box_size_cMpc_h=BOX,
            hubble_km_s_Mpc=74.6, little_h=.746,
            radius_table_cMpc_h=source['radial_table'],
            modulus_table_h=source['modulus_table'],
            redshift_table=source['redshift_table'], grid_size=N_COUNT,
            radial_min_cMpc_h=5., radial_max_cMpc_h=180.)

        @jax.jit
        def forward_field(white):
            position, velocity = evolve(white)
            state = particle_grid(position, velocity, particle_mass_array, config)
            return state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0)

        states = {}
        for label in ('a', 'b'):
            endpoint = component['checkpoints'][label]
            checkpoint_path = Path(endpoint['checkpoint'])
            if checkpoint_path != CHECKPOINTS[label]:
                raise ValueError(f'{label}: checkpoint provenance path changed')
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                q = np.asarray(checkpoint['canonical'], dtype=np.float64)
                transitions = int(checkpoint['completed_transitions'])
            if q.shape != (NIC+NNUISANCE,) or transitions != 1 or not np.isfinite(q).all():
                raise ValueError(f'{label}: saved accepted endpoint has invalid layout')
            tracer = jnp.asarray(q[NIC:NIC+9])
            tic = time.monotonic()
            rho_native, velocity_native = forward_field(jnp.asarray(q[:NIC]))
            jax.block_until_ready((rho_native, velocity_native))
            density, velocity = native_mass_momentum_to_count_cells(
                rho_native, velocity_native, BOX)
            velocity = jnp.moveaxis(velocity, 0, -1).reshape(-1, 3)
            intrinsic = tracer_masses(density, tracer)*(BOX/N_SOURCE/3.)**3
            geometry = tracer_geometry(tracer, growth)
            intensity = predict_chunked_volume_intensity(
                source['positions'], velocity, intrinsic, source['angular'],
                source_chunk_size=CHUNK, source_spacing=BOX/N_SOURCE,
                volume_order=2, order=4, segments=8, **geometry)
            intensity.block_until_ready()
            score = float(sparse_marked_poisson_log_likelihood(
                intensity, keys_jax, counts_jax, selected_voxel_mask=exposure_jax))
            reference_score = float(endpoint['component_scores']['count_2mpp'])
            score_error = abs(score-reference_score)
            if not np.isfinite(score) or score_error > 1e-7:
                raise AssertionError(f'{label}: exact selected count score mismatch {score_error:.3g}')
            summary = summarize_training_counts(np.asarray(intensity), keys,
                counts, training_exposure, grid_size=N_COUNT, box_size=BOX)
            states[label] = dict(
                checkpoint=str(checkpoint_path), accepted_transition=transitions,
                pmwd_forward_seconds=time.monotonic()-tic,
                count_score=score, component_reference_count_score=reference_score,
                count_score_absolute_error=score_error,
                expected_training_count=summary['total']['expected_training_count'],
                observed_training_count=summary['total']['observed_training_count'],
                observed_to_expected=summary['total']['observed_to_expected'],
                radial_by_population=summary['radial_by_population'],
                radial_population_by_exposed_octant=summary['radial_population_by_exposed_octant'],
                radial_population_by_model_intensity_quantile=summary['radial_population_by_model_intensity_quantile'],
                summary_definitions={key: summary[key] for key in
                    ('radius_definition', 'sector_definition',
                     'intensity_quantile_definition', 'interpretation')})
            report['states'] = states
            report['deterministic_PM_forward_replays'] += 1
            save()
            print(json.dumps({label: dict(pmwd_forward_seconds=states[label]['pmwd_forward_seconds'],
                count_score_absolute_error=score_error,
                observed_to_expected=summary['total']['observed_to_expected'])}), flush=True)
            del intensity, intrinsic, density, velocity, rho_native, velocity_native

        report.update(status='SAME_STATE_TRAINING_COUNT_RESIDUAL_COMPLETE_NOT_POSTERIOR',
            inference_limits='A and B are correlated accepted states in one short lineage; aggregate residuals are descriptive, not heldout/predictive/posterior checks. The count observation law remains partial and uncalibrated.')
        save()
        print(json.dumps(report, allow_nan=False), flush=True)
    except Exception as error:
        report.update(status='TRAINING_COUNT_RESIDUAL_FAILED_OR_INCOMPLETE_NOT_POSTERIOR',
            error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    main()
