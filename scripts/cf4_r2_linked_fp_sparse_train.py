"""Continuous linked-FP mark factors on the frozen training singleton graph.

The source neighborhood is a conservative 8-sigma spatial index around the
observed count voxel. One frozen group is checked against the all-source
calculation before the same operator is evaluated on every eligible training
singleton. This remains an unconditional-state mechanics diagnostic: group
association/inclusion and FP covariance are not calibrated.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/gpfs/kjhan/CF4/z0_density')
DEFAULT_OUT = BASE/'r2_linked_fp_sparse_train_v1'
DEFAULT_SPLIT = BASE/'r2_sky_closed_split_v5/split.npz'
OUT = Path(os.environ.get('CF4_R2_OUT_DIR', str(DEFAULT_OUT)))
SPLIT = Path(os.environ.get('CF4_R2_SPLIT_PATH', str(DEFAULT_SPLIT)))
POINTS = BASE/'r2_point_mark_manifest_v1/points.npz'
FP = BASE/'r2_source_observation_assembly_v1/observations.npz'
GROUP = BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
SOURCE = BASE/'r2_marked_source_geometry_v1/geometry.npz'
STATE = BASE/'r2_pm128_unconditional_v1/state.npz'
CROSSMATCH = ROOT/'data/cf4_2mpp_crossmatch_v1.csv'
COSMO = ROOT/'config/cf4_r2_common_cosmology_v1.json'
N, BOX = 128, 384.
SIGMA_LOS_KM_S = 100.
TAIL_SIGMA = 8.
MAX_PAD_WIDTH = 8192


def select_training_single_mark_links(options, membership, *, include_grouped=False):
    """Retain one scored FP and one secure point per source group.

    ``options`` must come from load_train_singletons's frozen structural/role
    checks (one source FP, one linked point, no anchor). Grouped physical
    membership does not turn a single distance mark into multiple measurements.
    Enabling it is a CONDITIONAL WORKING model, not a calibrated group law.
    The source group redshift is only the FP distance-indicator reference;
    the linked point's individual redshift supplies the conditioning kernel.
    """
    allowed={'source_ungrouped_catalogue_present','source_ungrouped_catalogue_absent'}
    if include_grouped:
        allowed.add('source_grouped_catalogue_present')
    chosen=[o for o in options if membership[o[3]] in allowed]
    if any(len({o[k] for o in chosen})!=len(chosen) for k in (0,1,2,3)):
        raise ValueError('single-mark cohort repeats a source group, point or FP row')
    return chosen


def load_train_singletons(split_path=SPLIT):
    with np.load(split_path, allow_pickle=False) as f:
        recno = f['point_recno'].copy()
        labels = f['fp_source_group'].copy()
        roles = f['fp_role'].copy()
    with np.load(POINTS, allow_pickle=False) as f:
        point = {k: f[k].copy() for k in
                 ('population', 'flat_cell', 'radius_cMpc_h')}
    with np.load(FP, allow_pickle=False) as f:
        pgc = f['PGC'].copy()
        source_group = f['source_group'].copy()
    with np.load(GROUP, allow_pickle=False) as f:
        np.testing.assert_array_equal(labels, f['group_labels'])
        row_group = f['row_group'].copy()
        anchor_group = f['anchor_group'].copy()
    if np.any(~np.isin(roles, (0, 1, 2))):
        raise ValueError('unknown frozen FP role')
    row_count = np.bincount(row_group, minlength=len(labels))
    anchor_count = np.bincount(anchor_group, minlength=len(labels))
    rec_index = {int(value): i for i, value in enumerate(recno)}
    label_index = {str(value): i for i, value in enumerate(labels)}
    pgc_group = {int(p): label_index[str(group)]
                 for p, group in zip(pgc, source_group)}
    links = {}
    with CROSSMATCH.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['match_class'] != 'secure_joint_mark' or not row['twompp_recno']:
                continue
            i = rec_index.get(int(row['twompp_recno']))
            g = pgc_group.get(int(row['PGC']))
            if i is not None and g is not None:
                links.setdefault(g, set()).add(i)
    options = []
    for g, indices in links.items():
        if (roles[g] == 0 and len(indices) == 1 and row_count[g] == 1
                and anchor_count[g] == 0):
            i = next(iter(indices))
            radius = float(point['radius_cMpc_h'][i])
            if 5. <= radius <= 180.:
                row = int(np.flatnonzero(row_group == g)[0])
                options.append((str(labels[g]), g, i, row))
    options.sort()
    if not options or len({item[1] for item in options}) != len(options):
        raise ValueError('frozen train-only singleton graph is empty or duplicated')
    train_rows = np.asarray([item[3] for item in options], dtype=np.int64)
    with np.load(GROUP, allow_pickle=False) as f:
        fp = {k: f[k][train_rows].copy() for k in
              ('dz_row', 'eta_mean', 'eta_std', 'eta_alpha')}
    options = [(*item, train_index) for train_index, item in enumerate(options)]
    return options, point, fp


def source_neighborhoods(shifted, options, point, hubble, little_h):
    spacing = BOX/N
    sigma_radius = little_h*SIGMA_LOS_KM_S/hubble
    support_radius = math.sqrt(3.)*1.5*spacing
    search_radius = TAIL_SIGMA*sigma_radius+support_radius
    bucket_radius = int(math.ceil(search_radius/spacing))+1
    bins = (np.floor(shifted/spacing).astype(np.int32) % N)
    flat_bin = np.ravel_multi_index(bins.T, (N,)*3)
    order = np.argsort(flat_bin, kind='stable')
    counts = np.bincount(flat_bin, minlength=N**3)
    starts = np.r_[0, np.cumsum(counts)]
    offsets = np.arange(-bucket_radius, bucket_radius+1, dtype=np.int32)
    ox, oy, oz = np.meshgrid(offsets, offsets, offsets, indexing='ij')
    bin_offsets = np.stack((ox.ravel(), oy.ravel(), oz.ravel()), axis=1)
    neighborhoods = []
    for _label, _g, i, _row, _train_index in options:
        voxel = np.asarray(np.unravel_index(int(point['flat_cell'][i]), (N,)*3),
                           dtype=np.int32)
        center = (voxel+.5)*spacing
        target_bins = (voxel[None, :]+bin_offsets) % N
        flat_targets = np.ravel_multi_index(target_bins.T, (N,)*3)
        pieces = [order[starts[b]:starts[b+1]] for b in flat_targets
                  if starts[b] < starts[b+1]]
        if not pieces:
            raise ValueError(f'no spatial candidates for {_label}')
        ids = np.unique(np.concatenate(pieces))
        delta = (shifted[ids]-center+BOX/2.) % BOX-BOX/2.
        ids = ids[np.sum(delta*delta, axis=1) <= search_radius**2+1e-9]
        if not len(ids):
            raise ValueError(f'empty exact neighborhood for {_label}')
        neighborhoods.append(ids.astype(np.int32, copy=False))
    return neighborhoods, sigma_radius, search_radius


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    expected_commit = os.environ.get('CF4_EXPECTED_COMMIT')
    if not expected_commit:
        raise RuntimeError('CF4_EXPECTED_COMMIT must pin the submitted source')
    source_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if source_commit != expected_commit:
        raise RuntimeError(f'source commit mismatch: {source_commit} != {expected_commit}')
    dirty = subprocess.run(
        ['git', 'diff', '--quiet', expected_commit, '--',
         'scripts/cf4_r2_linked_fp_sparse_train.py',
         'scripts/run_cf4_r2_linked_fp_sparse_train.sbatch'], cwd=ROOT)
    if dirty.returncode:
        raise RuntimeError('submitted diagnostic sources changed after commit')
    if OUT.exists():
        raise FileExistsError(OUT)

    from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax
    from cf4_r2_fp_distance import fp_log_likelihood_ratio
    from cf4_r2_marked_tracer_jax import (
        conditional_single_link_logfactor, intrinsic_biased_source_masses,
        intrinsic_lf_bin_fractions, predict_source_marked_radial_key_density,
    )
    from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells

    started = time.monotonic()
    options, point, fp = load_train_singletons()
    with np.load(STATE, allow_pickle=False) as f:
        rho = jnp.asarray(f['rho'], dtype=jnp.float64)
        velocity = jnp.asarray(f['velocity_km_s'], dtype=jnp.float64)
    with np.load(SOURCE, allow_pickle=False) as f:
        source = {k: jnp.asarray(f[k]) for k in f.files}
    rho_cell, velocity_cell = native_mass_momentum_to_count_cells(rho, velocity, BOX)
    fraction = jnp.sum(intrinsic_lf_bin_fractions()[1:4])
    intrinsic = intrinsic_biased_source_masses(
        rho_cell, jnp.log(fraction), jnp.ones(5),
        reference_interval=(-25., -21.))
    source_velocity = jnp.moveaxis(velocity_cell, 0, -1).reshape(-1, 3)
    cosmology = json.loads(COSMO.read_text())['common_cosmology']
    observer = jnp.full(3, BOX/2.)
    shifted = jax.jit(lambda p, v: observer_centred_spherical_rsd_jax(
        p, v, observer, BOX, cosmology['H0_km_s_Mpc'],
        little_h=cosmology['h'], scale_factor=1.)[0])(
            source['positions'], source_velocity)
    shifted.block_until_ready()
    shifted_host = np.asarray(shifted)
    neighborhoods, sigma_radius, search_radius = source_neighborhoods(
        shifted_host, options, point, cosmology['H0_km_s_Mpc'], cosmology['h'])
    width = 1 << int(math.ceil(math.log2(max(map(len, neighborhoods)))))
    if width > MAX_PAD_WIDTH:
        raise ValueError(f'candidate width {width} exceeds explicit cap {MAX_PAD_WIDTH}')

    paths = (SPLIT, POINTS, FP, GROUP, SOURCE, STATE, CROSSMATCH, COSMO)
    report = dict(
        classification='R2_V6_TRAIN_SINGLE_LINKED_FP_FIXED_STATE_FULL_SOURCE_COMPARISON',
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=source_commit, N=N, box_cMpc_h=BOX,
        training_one_countpoint_one_FProw_groups=len(options),
        candidate_padding_width=width, source_neighborhood='state-frozen periodic '
            'shifted-source bins; 8-sigma plus TSC support; not a live-field list',
        sigma_los_cMpc_h=sigma_radius, spatial_search_radius_cMpc_h=search_radius,
        Gaussian_tail_probability_beyond_8sigma=math.erfc(TAIL_SIGMA/math.sqrt(2.)),
        association_probability_assumed_field_independent_constant=True,
        group_selection_calibrated=False, FP_group_covariance_calibrated=False,
        heldout_FP_marks_used=0, heldout_count_keys_used=0,
        field_fit=False, sampler=False, R2_posterior=False, N256=False,
        MW_M31_M33='New-field roles remain latent; unresolved M33; no truth IDs used; '
                    'their observables must constrain the same evolved field.',
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths})
    if SPLIT != DEFAULT_SPLIT:
        v5_labels = {option[0] for option in load_train_singletons(DEFAULT_SPLIT)[0]}
        v6_labels = {option[0] for option in options}
        report['v5_v6_training_linked_group_identity_delta'] = dict(
            v5_only=sorted(v5_labels-v6_labels),
            v6_only=sorted(v6_labels-v5_labels),
            v5_count=len(v5_labels), v6_count=len(v6_labels))
    OUT.mkdir(parents=True)

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(
            json.dumps(report, indent=2, allow_nan=False)+'\n')

    save()
    radial_args = dict(
        observer=observer, box_size_cMpc_h=BOX,
        hubble_km_s_Mpc=cosmology['H0_km_s_Mpc'], little_h=cosmology['h'],
        radius_table_cMpc_h=source['radial_table'],
        modulus_table_h=source['modulus_table'], redshift_table=source['redshift_table'],
        grid_size=N, sigma_los_km_s=SIGMA_LOS_KM_S,
        radial_min_cMpc_h=5., radial_max_cMpc_h=180.)

    def make_scorer(population):
        def score(pos, vel, mass, angular, voxel, observed_radius, dz,
                  eta_mean, eta_std, eta_alpha):
            density = predict_source_marked_radial_key_density(
                pos, vel, mass, angular, population, voxel, observed_radius,
                **radial_args)
            relative = (pos-observer+BOX/2.) % BOX-BOX/2.
            true_radius = jnp.linalg.norm(relative, axis=1)
            eta = jnp.log10(dz/true_radius)
            log_mark = fp_log_likelihood_ratio(
                eta, 0., eta_mean, eta_std, eta_alpha)
            marks = jnp.broadcast_to(log_mark[None, :], density.shape)
            factor = conditional_single_link_logfactor(
                density, jnp.zeros_like(density), marks)
            return factor, jnp.sum(density, axis=1)
        return jax.jit(score)

    scorers = {pop: make_scorer(pop) for pop in range(6)}

    def evaluate(option, ids, pad_width):
        label, _g, i, row, train_index = option
        ids_pad = np.full(pad_width, int(ids[0]), dtype=np.int32)
        ids_pad[:len(ids)] = ids
        active = (np.arange(pad_width) < len(ids)).astype(np.float64)
        ix = jnp.asarray(ids_pad)
        pos = source['positions'][ix]
        vel = source_velocity[ix]
        mass = intrinsic[:, ix]*jnp.asarray(active)[None, :]
        angular = source['angular'][:, ix]
        voxel = np.asarray(np.unravel_index(
            int(point['flat_cell'][i]), (N,)*3), dtype=np.int32)
        pop = int(point['population'][i])
        result = scorers[pop](
            pos, vel, mass, angular, jnp.asarray(voxel),
            float(point['radius_cMpc_h'][i]), float(fp['dz_row'][train_index]),
            float(fp['eta_mean'][train_index]), float(fp['eta_std'][train_index]),
            float(fp['eta_alpha'][train_index]))
        return label, pop, voxel, float(point['radius_cMpc_h'][i]), len(ids), result

    try:
        # This identity/geometry-frozen control was selected before inspecting marks.
        control_index = next((k for k, option in enumerate(options)
                              if option[0] == 'P1085367'), None)
        if control_index is None:
            raise ValueError('frozen all-source support-control group disappeared')
        control = options[control_index]
        sparse = evaluate(control, neighborhoods[control_index], width)
        label, _pop, voxel, radius, _candidate_count, (sparse_score, sparse_bins) = sparse
        train_index = control[4]
        i = control[2]
        pop = int(point['population'][i])
        exact_score, exact_bins = scorers[pop](
            source['positions'], source_velocity, intrinsic, source['angular'],
            jnp.asarray(voxel), radius, float(fp['dz_row'][train_index]),
            float(fp['eta_mean'][train_index]), float(fp['eta_std'][train_index]),
            float(fp['eta_alpha'][train_index]))
        exact_score, exact_bins = float(exact_score), np.asarray(exact_bins)
        sparse_score, sparse_bins = float(sparse_score), np.asarray(sparse_bins)
        relative_bins = np.abs(sparse_bins-exact_bins)/np.maximum(np.abs(exact_bins),1e-30)
        score_error = abs(sparse_score-exact_score)
        report['all_source_control'] = dict(
            source_group=label, count_population=pop, count_voxel=voxel.tolist(),
            observed_radius_cMpc_h=radius,
            candidate_sources=len(neighborhoods[control_index]),
            total_sources=int(source['positions'].shape[0]),
            sparse_minus_full_conditional_logfactor= sparse_score-exact_score,
            max_relative_trueK_density_sum_error=float(np.max(relative_bins)),
            sparse_conditional_logfactor=sparse_score,
            full_conditional_logfactor=exact_score)
        save()
        if score_error > 1e-8 or np.max(relative_bins) > 1e-8:
            raise AssertionError('8-sigma source neighborhood disagrees with full source')

        rows = []
        max_score_error = (0., None)
        max_bin_error = (0., None)
        for k, option in enumerate(options):
            if k == control_index:
                result = sparse
            else:
                result = evaluate(option, neighborhoods[k], width)
            label, pop, voxel, radius, candidate_count, (score, bins) = result
            bins = np.asarray(bins)
            if (not np.isfinite(float(score)) or not np.isfinite(bins).all()
                    or np.sum(bins) <= 0):
                raise FloatingPointError(f'conditional source support failed for {label}')
            if k == control_index:
                full_score, full_bins = exact_score, exact_bins
            else:
                full_score, full_bins = scorers[pop](
                    source['positions'], source_velocity, intrinsic,
                    source['angular'], jnp.asarray(voxel), radius,
                    float(fp['dz_row'][option[4]]),
                    float(fp['eta_mean'][option[4]]),
                    float(fp['eta_std'][option[4]]),
                    float(fp['eta_alpha'][option[4]]))
                full_score, full_bins = float(full_score), np.asarray(full_bins)
            score_error = abs(float(score)-full_score)
            relative_bins = np.abs(bins-full_bins)/np.maximum(np.abs(full_bins),1e-30)
            bin_error = float(np.max(relative_bins))
            if score_error > max_score_error[0]:
                max_score_error = (score_error, label)
            if bin_error > max_bin_error[0]:
                max_bin_error = (bin_error, label)
            if score_error > 1e-8 or bin_error > 1e-8:
                raise AssertionError(
                    f'fixed-state sparse/full mismatch for {label}: '
                    f'logfactor={score_error:.3g}, trueK={bin_error:.3g}')
            rows.append(dict(source_group=label, population=pop,
                             voxel=[int(v) for v in voxel],
                             observed_radius_cMpc_h=radius,
                             candidate_sources=candidate_count,
                             conditional_FP_logfactor=float(score),
                             full_source_FP_logfactor=full_score,
                             sparse_minus_full_conditional_logfactor=
                                 float(score)-full_score,
                             max_relative_trueK_density_sum_error=bin_error))
            if (k+1) % 32 == 0 or k+1 == len(options):
                report['completed_training_groups'] = k+1
                report['groups'] = rows
                report['largest_full_source_discrepancies'] = dict(
                    conditional_logfactor_abs=(max_score_error[0],max_score_error[1]),
                    trueK_density_relative=(max_bin_error[0],max_bin_error[1]))
                report['partial_score_range'] = [
                    min(row['conditional_FP_logfactor'] for row in rows),
                    max(row['conditional_FP_logfactor'] for row in rows)]
                save()
        score_values = np.asarray([row['conditional_FP_logfactor'] for row in rows])
        candidate_sizes = np.asarray([row['candidate_sources'] for row in rows])
        report.update(
            groups=rows,
            candidate_count_quantiles=dict(zip(
                ('p50','p90','p99','max'),
                np.quantile(candidate_sizes,[.5,.9,.99,1.]).tolist())),
            conditional_FP_logfactor_quantiles=dict(zip(
                ('p05','p50','p95','min','max'),
                [*np.quantile(score_values,[.05,.5,.95]).tolist(),
                 float(score_values.min()),float(score_values.max())])),
            largest_full_source_discrepancies=dict(
                conditional_logfactor_abs=(max_score_error[0],max_score_error[1]),
                trueK_density_relative=(max_bin_error[0],max_bin_error[1])),
            status='COMPLETED_V6_FIXED_STATE_FULL_SOURCE_TRAINING_COMPARISON_NOT_CALIBRATION')
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    save()
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
