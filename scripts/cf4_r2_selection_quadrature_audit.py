"""Training-key-only voxel selection sensitivity; no fit or field inference."""
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from cf4_actual_selection import base
from cf4_twompp_joint_information_budget_pilot_v1 import (
    _cosmology_distance_table, schechter_fraction)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
DATA = BASE / 'r2_common_catalogue_128_v1'
PREVIOUS = BASE / 'r2_common_selection_128_v1'
TECHNICAL_INPUT = Path('/gpfs/kjhan/CF4/kf_design/twompp_disjoint_tracer_v4/pilot/result.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('selection quadrature audit must run through Slurm')
    output_dir = os.environ.get('CF4_R2_OUT_DIR')
    source_commit = os.environ.get('CF4_EXPECTED_COMMIT')
    if not output_dir or not source_commit:
        raise RuntimeError('CF4_R2_OUT_DIR and CF4_EXPECTED_COMMIT are required')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    started = time.monotonic()

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
    expected_map_hashes = [technical['bound_inputs'][name]['sha256'] for name in
                           ('completeness_11_5', 'completeness_12_5')]
    actual_map_hashes = [sha256(path) for path in map_paths]
    if actual_map_hashes != expected_map_hashes:
        raise ValueError('angular map digest differs from the pinned technical-input result')
    maps = [base.load_completeness_map(path, 512) for path in map_paths]
    rotation = SkyCoord(CartesianRepresentation(np.eye(3)),
                        frame='supergalactic').icrs.cartesian.xyz.value

    # Deliberately load only training keys; do not touch all_keys, holdout_keys,
    # train_counts, or holdout_counts in this evaluation-preserving rerun.
    count_path = DATA / 'counts_3_sparse.npz'
    with np.load(count_path, allow_pickle=False) as data:
        keys = data['train_keys'].copy()
    train_keys_digest = hashlib.sha256(keys.tobytes()).hexdigest()
    n, dx, order = 128, 3., 8
    population = keys // n**3
    flat = keys % n**3
    ijk = np.array(np.unravel_index(flat, (n,) * 3)).T
    centers = ((ijk + .5) * dx - 192.).T
    exposure8 = np.zeros(len(keys), dtype=np.float64)
    rows_by_population = [np.flatnonzero(population == p) for p in range(6)]

    edges = np.array([5., 30., 60., 90., 120., 150., 180.])
    rtab = np.linspace(edges[0], edges[-1], 200001)
    dl = _cosmology_distance_table(rtab, cosmology) * cosmology['h']
    absolute = tracer['tracer_design']['absolute_K_edges']
    radial = np.array([
        schechter_fraction(dl, None if p // 3 == 0 else 11.5,
            11.5 if p // 3 == 0 else 12.5, absolute[p % 3], absolute[p % 3 + 1],
            -23.28, -.94)
        for p in range(6)])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for a, b, d in product(range(order), repeat=3):
        xyz = centers + dx / 2 * nodes[[a, b, d], None]
        radius = np.linalg.norm(xyz, axis=0)
        active = (radius >= edges[0]) & (radius <= edges[-1])
        rows = np.flatnonzero(active)
        if not rows.size:
            continue
        r = radius[rows]
        pixels = hp.vec2pix(512, *(rotation @ xyz[:, rows]), nest=False)
        weight = weights[a] * weights[b] * weights[d] / 8
        active_population = population[rows]
        for p in range(6):
            local = np.flatnonzero(active_population == p)
            if local.size:
                selected = rows[local]
                exposure8[selected] += (weight * maps[p // 3][pixels[local]]
                    * np.interp(r[local], rtab, radial[p]))

    previous_result = json.loads((PREVIOUS / 'result.json').read_text())
    if previous_result.get('quadrature_order') != 6:
        raise ValueError('the bound predecessor is not the order-six integral')
    previous_cube_path = PREVIOUS / 'selection_3.h5'
    exposure6 = np.zeros(len(keys), dtype=np.float64)
    with h5py.File(previous_cube_path, 'r') as h:
        if int(h.attrs['quadrature_order']) != 6:
            raise ValueError('saved selection cube does not have order six')
        if h.attrs['status'] != 'INTEGRATION_COMPLETE_NOT_CALIBRATED':
            raise ValueError('saved order-six selection cube is incomplete or unresolved')
        cube = h['selection_shells']
        for p, p_rows in enumerate(rows_by_population):
            if p_rows.size:
                collapsed = np.asarray(cube[p], dtype=np.float64).sum(axis=0)
                exposure6[p_rows] = collapsed[ijk[p_rows, 0], ijk[p_rows, 1], ijk[p_rows, 2]]

    relative = np.abs(exposure8 - exposure6) / np.maximum(np.abs(exposure6), 1e-30)
    absolute_difference = np.abs(exposure8 - exposure6)
    largest = np.argsort(relative)[-10:][::-1]
    report = {
        'classification': 'TRAINING_ONLY_ORDER8_VS_SAVED_ORDER6_CELL_SELECTION_SENSITIVITY',
        'status': 'TRAINING_DIAGNOSTIC_ONLY_NOT_CALIBRATED_NOT_FIELD_INFERENCE',
        'source_commit': source_commit,
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'cosmology': cosmology,
        'grid': {'N': n, 'dx_cMpc_h': dx, 'box_cMpc_h': 384.},
        'quadrature': {'saved_order': 6, 'new_order': order,
                       'node_layout': 'tensor_product_Gauss_Legendre_per_voxel'},
        'training_keys_only': True,
        'heldout_keys_or_counts_read_by_this_run': False,
        'training_population_cell_keys': int(len(keys)),
        'training_keys_sha256': train_keys_digest,
        'count_archive_path': str(count_path),
        'order6_zero_exposure_training_keys': int(np.count_nonzero(exposure6 <= 0)),
        'order8_zero_exposure_training_keys': int(np.count_nonzero(exposure8 <= 0)),
        'relative_exposure_difference_max': float(np.max(relative)),
        'relative_exposure_difference_p95': float(np.percentile(relative, 95)),
        'absolute_exposure_difference_max': float(np.max(absolute_difference)),
        'absolute_exposure_difference_p95': float(np.percentile(absolute_difference, 95)),
        'largest_training_key_changes': [
            {'key': int(keys[i]), 'population': int(population[i]), 'voxel_ijk': ijk[i].tolist(),
             'order6_exposure': float(exposure6[i]), 'order8_exposure': float(exposure8[i]),
             'relative_difference': float(relative[i]),
             'absolute_difference': float(absolute_difference[i])}
            for i in largest],
        'map_hashes': actual_map_hashes,
        'technical_input_gate': technical['status'],
        'global_order8_integral_reference': str(PREVIOUS.parent / 'r2_selection_quadrature_20261003_v2/result.json'),
        'elapsed_seconds': time.monotonic() - started,
        'host_peak_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'selection_calibration_complete': False,
        'angular_quadrature_certified': False,
        'quantitative_joint_likelihood_ready': False,
        'Q_GOAL': 'Training-only numerical sensitivity of the selection term for the same CF4-conditioned z=0 field; no field is inferred.',
        'Q_LEAN': 'One sparse training-cell order-eight comparison against the saved order-six cube; no global reintegration, heldout read, fit, or simulation.',
        'LG_same_new_field_requirement': 'MW/M31 roles remain ambiguous and M33 unresolved; later observables must constrain those same roles in the NEW evolved field at <=0.3 cMpc/h, native truth IDs calibration/evaluation-only.',
    }
    output_dir.mkdir(parents=True)
    (output_dir / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in (
        'classification', 'relative_exposure_difference_max',
        'relative_exposure_difference_p95', 'order8_zero_exposure_training_keys',
        'elapsed_seconds', 'host_peak_GiB')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
