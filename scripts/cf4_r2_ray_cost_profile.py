"""Bounded training-key benchmark for map-aware voxel ray exposure integration.

This profiles geometric ray/cell intersections and the six existing radial
selection channels. It reads ``train_keys`` only: no counts, heldout arrays,
field, likelihood, fit, or PM evolution are involved.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from itertools import product

import healpy as hp
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from cf4_actual_selection import base
from cf4_twompp_joint_information_budget_pilot_v1 import (
    _cosmology_distance_table, schechter_fraction)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
DATA = BASE / 'r2_common_catalogue_128_v1'
PREVIOUS = BASE / 'r2_common_selection_128_v1'
TECHNICAL_INPUT = Path(
    '/gpfs/kjhan/CF4/kf_design/twompp_disjoint_tracer_v4/pilot/result.json')
N = 128
DX = 3.0
BOX = 384.0
HALF = BOX / 2
RADIAL_EDGES = np.array([5., 30., 60., 90., 120., 150., 180.])
PROFILE_RADIAL_BINS = np.array([0., 15., 30., 60., 90., 120., 150., 180., 186.])
REFERENCE_KEYS = {
    3153974: {512: 0.1327339, 1024: 0.1326210, 2048: 0.1326252},
    5070912: {512: 0.0033372, 1024: 0.0033882, 2048: 0.0034077},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ray_box_intervals(directions, low, high):
    """Positive-distance slab intersections for unit rays and one voxel."""
    directions = np.asarray(directions, dtype=np.float64)
    enter = np.full(directions.shape[1], 0.0)
    leave = np.full(directions.shape[1], np.inf)
    intersects_parallel = np.ones(directions.shape[1], dtype=bool)
    for axis in range(3):
        component = directions[axis]
        nonzero = np.abs(component) > 1e-15
        axis_enter = np.full(component.shape, -np.inf)
        axis_leave = np.full(component.shape, np.inf)
        t0 = np.zeros(component.shape)
        t1 = np.zeros(component.shape)
        t0[nonzero] = low[axis] / component[nonzero]
        t1[nonzero] = high[axis] / component[nonzero]
        axis_enter[nonzero] = np.minimum(t0[nonzero], t1[nonzero])
        axis_leave[nonzero] = np.maximum(t0[nonzero], t1[nonzero])
        enter = np.maximum(enter, axis_enter)
        leave = np.minimum(leave, axis_leave)
        intersects_parallel &= nonzero | ((low[axis] <= 0.) & (high[axis] >= 0.))
    valid = intersects_parallel & (leave > enter)
    return enter, leave, valid


def self_test_ray_box_intervals():
    root3 = np.sqrt(3.)
    diagonal = np.array([[1 / root3], [1 / root3], [1 / root3]])
    enter, leave, valid = ray_box_intervals(
        diagonal, np.array([1., 1., 1.]), np.array([2., 2., 2.]))
    if not valid.tolist() == [True]:
        raise AssertionError('ray/box hit and parallel-miss cases changed')
    if not np.isclose(enter[0], root3) or not np.isclose(leave[0], 2 * root3):
        raise AssertionError('diagonal ray/box interval is incorrect')
    axis = np.array([[1.], [0.], [0.]])
    enter, leave, valid = ray_box_intervals(
        axis, np.array([-1., -1., -1.]), np.array([1., 1., 1.]))
    if not valid.tolist() == [True] or not np.isclose(enter[0], 0.) or not np.isclose(leave[0], 1.):
        raise AssertionError('axis ray/box interval is incorrect')
    _, _, valid = ray_box_intervals(
        axis, np.array([1., 2., -1.]), np.array([2., 3., 1.]))
    if not valid.tolist() == [False]:
        raise AssertionError('parallel ray outside a slab must miss')
    rng = np.random.default_rng(41)
    low = np.array([0., 0., 0.])
    high = np.array([3., 3., 3.])
    center = (low + high) / 2
    cap_axis, cap = candidate_cap(center, low, high, np.eye(3))
    points = rng.uniform(low, high, size=(4096, 3))
    norms = np.linalg.norm(points, axis=1)
    points = points[norms > 0.]
    cosines = (points @ cap_axis) / np.linalg.norm(points, axis=1)
    if np.arccos(np.clip(np.min(cosines), -1., 1.)) > cap + 1e-12:
        raise AssertionError('corner-derived angular cap excludes an interior ray')
    return 4


def voxel_geometry(flat):
    ijk = np.array(np.unravel_index(int(flat), (N,) * 3), dtype=np.int64)
    low = ijk * DX - HALF
    high = low + DX
    center = (low + high) / 2
    return ijk, low, high, center


def candidate_cap(center_sky, low, high, rotation):
    center_radius = float(np.linalg.norm(center_sky))
    if center_radius == 0.:
        return np.array([1., 0., 0.]), np.pi
    axis = center_sky / center_radius
    vertices = np.asarray(list(product(*zip(low, high))), dtype=np.float64)
    vertex_sky = (rotation @ vertices.T).T
    vertex_norm = np.linalg.norm(vertex_sky, axis=1)
    nonzero = vertex_norm > 0.
    if not np.any(nonzero):
        return axis, np.pi
    cosine = (vertex_sky[nonzero] @ axis) / vertex_norm[nonzero]
    # Extreme voxel rays are its vertices; a one-microradian pad covers roundoff.
    cap = float(np.arccos(np.clip(np.min(cosine), -1., 1.)) + 1e-6)
    return axis, min(np.pi, cap)


def ray_geometry(flat, nside, rotation, candidate_limit):
    ijk, low, high, center = voxel_geometry(flat)
    center_sky = rotation @ center
    axis, cap = candidate_cap(center_sky, low, high, rotation)
    nominal_candidate_count = (hp.nside2npix(nside)
        * (1. - np.cos(cap)) / 2.)
    if nominal_candidate_count > candidate_limit:
        return {
            'skipped': True,
            'skip_reason': 'estimated_cap_pixel_count_over_bounded_limit',
            'cap_rad': cap,
            'estimated_candidate_pixels': float(nominal_candidate_count),
            'voxel_ijk': ijk.tolist(),
        }

    t0 = time.perf_counter()
    candidates = hp.query_disc(nside, axis, cap, inclusive=True, nest=False)
    sky_dirs = np.asarray(hp.pix2vec(nside, candidates, nest=False))
    directions = rotation.T @ sky_dirs
    enter, leave, intersects = ray_box_intervals(directions, low, high)
    rlo = np.maximum(enter, RADIAL_EDGES[0])
    rhi = np.minimum(leave, RADIAL_EDGES[-1])
    valid = intersects & (rhi > rlo)
    if np.any(valid):
        native_pixels = hp.vec2pix(512, *sky_dirs[:, valid], nest=False)
        valid_low = rlo[valid]
        valid_high = rhi[valid]
    else:
        native_pixels = np.empty(0, dtype=np.int64)
        valid_low = np.empty(0, dtype=np.float64)
        valid_high = np.empty(0, dtype=np.float64)
    geometry_seconds = time.perf_counter() - t0
    return {
        'skipped': False,
        'cap_rad': cap,
        'candidate_pixels': int(len(candidates)),
        'intersecting_center_pixels': int(np.count_nonzero(valid)),
        'geometry_seconds': geometry_seconds,
        'native_pixels': native_pixels,
        'rlo': valid_low,
        'rhi': valid_high,
        'voxel_ijk': ijk.tolist(),
    }


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('ray cost profile must run through Slurm')
    out_value = os.environ.get('CF4_R2_OUT_DIR')
    expected_commit = os.environ.get('CF4_EXPECTED_COMMIT')
    if not out_value or not expected_commit:
        raise RuntimeError('CF4_R2_OUT_DIR and CF4_EXPECTED_COMMIT are required')
    out = Path(out_value)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    test_count = self_test_ray_box_intervals()

    tracer = json.loads((ROOT / 'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    cosmology_contract = json.loads((ROOT / 'config/cf4_r2_common_cosmology_v1.json').read_text())
    c = cosmology_contract['common_cosmology']
    cosmology = dict(h=c['h'], H0_km_s_Mpc=c['H0_km_s_Mpc'],
                     Omega_m=c['Omega_m'], Omega_b=c['Omega_b'], Tcmb_K=c['Tcmb_K'])
    map_paths = [tracer['inputs'][name]['path'] for name in
                 ('completeness_11_5', 'completeness_12_5')]
    technical = json.loads(TECHNICAL_INPUT.read_text())
    if technical.get('status') != 'PASS_TECHNICAL_INPUT_GATE_NO_FIELD_INFERENCE':
        raise ValueError('pinned map technical-input gate is not passing')
    expected_hashes = [technical['bound_inputs'][name]['sha256'] for name in
                       ('completeness_11_5', 'completeness_12_5')]
    actual_hashes = [sha256(path) for path in map_paths]
    if actual_hashes != expected_hashes:
        raise ValueError('angular map digest differs from pinned technical-input result')
    maps = [base.load_completeness_map(path, 512) for path in map_paths]
    rotation = SkyCoord(CartesianRepresentation(np.eye(3)),
                        frame='supergalactic').icrs.cartesian.xyz.value

    # Strict training-only access. Do not load counts or any holdout/all-key arrays.
    count_path = DATA / 'counts_3_sparse.npz'
    with np.load(count_path, allow_pickle=False) as data:
        keys = np.asarray(data['train_keys'], dtype=np.int64).copy()
    if keys.ndim != 1 or np.any(keys < 0) or np.any(keys >= 6 * N**3):
        raise ValueError('training-key namespace is malformed')
    if len(np.unique(keys)) != len(keys):
        raise ValueError('training keys are not unique')
    keys_digest = hashlib.sha256(keys.tobytes()).hexdigest()
    flats = np.unique(keys % N**3)
    ijk = np.array(np.unravel_index(flats, (N,) * 3)).T
    centers = ((ijk + .5) * DX - HALF)
    radii = np.linalg.norm(centers, axis=1)
    radial_bin = np.clip(np.searchsorted(PROFILE_RADIAL_BINS, radii, side='right') - 1,
                         0, len(PROFILE_RADIAL_BINS) - 2)
    rng = np.random.default_rng(20261004)
    selected = []
    per_bin_sample_cap = int(os.environ.get('CF4_R2_GEOMETRIES_PER_BIN', '2'))
    for b in range(len(PROFILE_RADIAL_BINS) - 1):
        candidates = flats[radial_bin == b]
        take = min(per_bin_sample_cap, len(candidates))
        if take:
            selected.extend(np.sort(rng.choice(candidates, size=take, replace=False)).tolist())
    # Preserve two previously measured, difficult TRAINING controls as code checks.
    key_set = set(keys.tolist())
    fixed_controls = {}
    for key, values in REFERENCE_KEYS.items():
        if key in key_set:
            flat = int(key % N**3)
            if flat not in selected:
                selected.append(flat)
            fixed_controls[flat] = {
                'training_key': int(key), 'population': int(key // N**3),
                'prior_ray_exposure': {str(nside): float(value)
                                       for nside, value in values.items()},
            }
    selected = sorted(set(int(value) for value in selected))
    if not selected:
        raise ValueError('no training-only geometry cells selected')

    edges = RADIAL_EDGES
    rtab = np.linspace(edges[0], edges[-1], 200001)
    dl = _cosmology_distance_table(rtab, cosmology) * cosmology['h']
    absolute = tracer['tracer_design']['absolute_K_edges']
    radial = np.array([
        schechter_fraction(dl, None if p // 3 == 0 else 11.5,
            11.5 if p // 3 == 0 else 12.5, absolute[p % 3], absolute[p % 3 + 1],
            -23.28, -.94)
        for p in range(6)])
    cumulative = np.concatenate((np.zeros((6, 1)), np.cumsum(
        .5 * (rtab[1:]**2 * radial[:, 1:] + rtab[:-1]**2 * radial[:, :-1])
        * np.diff(rtab)[None, :], axis=1)), axis=1)

    nsides = (512, 1024, 2048)
    candidate_limit = int(os.environ.get('CF4_R2_MAX_CANDIDATE_PIXELS', '1500000'))
    detail = []
    for flat in selected:
        cell_ijk, low, high, center = voxel_geometry(flat)
        radius = float(np.linalg.norm(center))
        shell = int(np.clip(np.searchsorted(PROFILE_RADIAL_BINS, radius, side='right') - 1,
                            0, len(PROFILE_RADIAL_BINS) - 2))
        row = {'flat_voxel': int(flat), 'voxel_ijk': cell_ijk.tolist(),
               'center_radius_cMpc_h': radius, 'radial_bin': shell,
               'fixed_reference_control': fixed_controls.get(flat), 'nside': {}}
        for nside in nsides:
            geo = ray_geometry(flat, nside, rotation, candidate_limit)
            if geo['skipped']:
                row['nside'][str(nside)] = {k: v for k, v in geo.items()
                                             if k != 'native_pixels' and k != 'rlo' and k != 'rhi'}
                continue
            native = geo.pop('native_pixels')
            rlo, rhi = geo.pop('rlo'), geo.pop('rhi')
            t_channel = time.perf_counter()
            exposures = np.zeros(6, dtype=np.float64)
            for p in range(6):
                radial_mass = np.interp(rhi, rtab, cumulative[p]) - np.interp(
                    rlo, rtab, cumulative[p])
                completeness = maps[p // 3][native]
                exposures[p] = (hp.nside2pixarea(nside)
                    * np.dot(completeness, radial_mass) / DX**3)
            channel_seconds = time.perf_counter() - t_channel
            result = dict(geo)
            result.update(
                six_population_channel_seconds=channel_seconds,
                exposures_by_population=exposures.tolist(),
                estimated_streaming_geometry_bytes_upper_bound=int(
                    result['candidate_pixels'] * 128))
            if flat in fixed_controls:
                p = int(fixed_controls[flat]['population'])
                prior = fixed_controls[flat]['prior_ray_exposure'][str(nside)]
                result['prior_reference_exposure'] = prior
                result['absolute_difference_from_prior_reference'] = float(abs(exposures[p] - prior))
            row['nside'][str(nside)] = result
        detail.append(row)

    # Volume-equivalent voxel counts, used only for an explicitly rough cost projection.
    volume_counts = {}
    for b in range(len(PROFILE_RADIAL_BINS) - 1):
        lo, hi = PROFILE_RADIAL_BINS[b:b + 2]
        lo_shell = max(lo, 5.)
        hi_shell = min(hi, 180.)
        if hi_shell <= lo_shell:
            count = 0.
        else:
            count = (4. * np.pi / 3. * (hi_shell**3 - lo_shell**3) / DX**3)
        volume_counts[str(b)] = float(count)
    projections = {}
    for nside in nsides:
        geometry_by_bin = {}
        channel_by_bin = {}
        candidates_by_bin = {}
        intervals_by_bin = {}
        samples_by_bin = {}
        skipped_by_bin = {}
        for row in detail:
            value = row['nside'][str(nside)]
            if value.get('skipped'):
                b = str(row['radial_bin'])
                skipped_by_bin[b] = skipped_by_bin.get(b, 0) + 1
                continue
            b = str(row['radial_bin'])
            for mapping, key, field in ((geometry_by_bin, b, 'geometry_seconds'),
                                        (channel_by_bin, b, 'six_population_channel_seconds'),
                                        (candidates_by_bin, b, 'candidate_pixels'),
                                        (intervals_by_bin, b, 'intersecting_center_pixels')):
                mapping.setdefault(key, []).append(float(value[field]))
            samples_by_bin[b] = samples_by_bin.get(b, 0) + 1
        projected_geometry_s = 0.
        projected_channels_s = 0.
        projected_candidates = 0.
        projected_intervals = 0.
        unprofiled_bins = []
        for b, voxel_count in volume_counts.items():
            if voxel_count <= 0:
                continue
            if b not in geometry_by_bin:
                unprofiled_bins.append(int(b))
                continue
            projected_geometry_s += voxel_count * float(np.mean(geometry_by_bin[b]))
            projected_channels_s += voxel_count * float(np.mean(channel_by_bin[b]))
            projected_candidates += voxel_count * float(np.mean(candidates_by_bin[b]))
            projected_intervals += voxel_count * float(np.mean(intervals_by_bin[b]))
        projections[str(nside)] = {
            'volume_equivalent_geometry_voxels_total': float(sum(volume_counts.values())),
            'training_sample_count_by_radial_bin': samples_by_bin,
            'candidate-cap-skipped_sample_count_by_radial_bin': skipped_by_bin,
            'unprofiled_radial_bins': unprofiled_bins,
            'projected_geometry_seconds_profiled_bins_only': projected_geometry_s,
            'projected_six_channel_seconds_profiled_bins_only': projected_channels_s,
            'projected_wall_seconds_profiled_bins_only': projected_geometry_s + projected_channels_s,
            'projected_candidate_pixels_profiled_bins_only': projected_candidates,
            'projected_intersecting_ray_cell_intervals_profiled_bins_only': projected_intervals,
            'projection_excludes_unprofiled_bins': bool(unprofiled_bins),
            'interpretation': 'Training-key-selected geometry controls, volume-weighted by spherical-shell voxel-equivalent counts. This is a rough CPU-work extrapolation, not a full-grid runtime guarantee or accuracy certificate.',
        }

    report = {
        'classification': 'TRAINING_ONLY_MAP_AWARE_RAY_CELL_COST_PROFILE',
        'status': 'BOUNDED_COST_MEASUREMENT_ONLY_NOT_SELECTION_CERTIFICATION',
        'source_commit': expected_commit,
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'cosmology': cosmology,
        'grid': {'N': N, 'dx_cMpc_h': DX, 'box_cMpc_h': BOX},
        'angular_maps': {'native_nside': 512, 'hashes': actual_hashes,
                         'technical_input_status': technical['status']},
        'training_keys_only': True,
        'training_counts_read': False,
        'all_keys_or_holdout_arrays_read': False,
        'heldout_scored': False,
        'training_key_count': int(len(keys)),
        'training_key_sha256': keys_digest,
        'unique_training_geometry_voxels': int(len(flats)),
        'sampled_training_geometry_voxels': int(len(selected)),
        'samples_per_radial_bin_cap': per_bin_sample_cap,
        'radial_bin_edges_cMpc_h': PROFILE_RADIAL_BINS.tolist(),
        'ray_nsides': list(nsides),
        'candidate_pixel_limit_per_voxel': candidate_limit,
        'ray_geometry_self_tests_passed': test_count,
        'fixed_prior_reference_controls': fixed_controls,
        'samples': detail,
        'rough_full_grid_projections': projections,
        'volume_equivalent_voxels_by_radial_bin': volume_counts,
        'elapsed_seconds': time.monotonic() - started,
        'host_peak_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'uses_pixel_values_without_smoothing_or_nearest_replacement': True,
        'angular_quadrature_certified': False,
        'selection_calibration_complete': False,
        'quantitative_joint_likelihood_ready': False,
        'R2_complete': False,
        'Q_GOAL': 'Measure cost of a map-preserving selection integral needed for the same CF4-conditioned z=0 field; no field or posterior is inferred.',
        'Q_LEAN': 'A deterministic small training-geometry sample measures streaming ray/cell and six-channel work; no full-grid integration, heldout read, fit, or simulation.',
        'MW_M31_M33': 'MW/M31 roles remain ambiguous and M33 unresolved; future observables must constrain those same roles in the NEW evolved field at LG <=0.3 cMpc/h, with native truth IDs reserved for calibration/evaluation.',
        'open_R2_limitations': ['angular quadrature accuracy', 'selection/bias calibration', 'multi-member covariance', 'shared-latent count/mark conditional'],
    }
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'classification': report['classification'],
        'sampled_training_geometry_voxels': report['sampled_training_geometry_voxels'],
        'rough_full_grid_projections': projections,
        'elapsed_seconds': report['elapsed_seconds'],
        'host_peak_GiB': report['host_peak_GiB'],
    }, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
