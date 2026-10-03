"""One-order selection-integral sensitivity check; no fit or field inference."""
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
TECHNICAL_INPUT = Path('/gpfs/kjhan/CF4/kf_design/twompp_disjoint_tracer_v4/pilot/result.json')
DATA = BASE / 'r2_common_catalogue_128_v1'
PREVIOUS = BASE / 'r2_common_selection_128_v1'
OUT = Path(os.environ['CF4_R2_OUT_DIR'])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('selection quadrature audit must run through Slurm')
    if OUT.exists():
        raise FileExistsError(OUT)
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

    edges = np.array([5., 30., 60., 90., 120., 150., 180.])
    rtab = np.linspace(edges[0], edges[-1], 200001)
    dl = _cosmology_distance_table(rtab, cosmology) * cosmology['h']
    absolute = tracer['tracer_design']['absolute_K_edges']

    def fraction(distance, population):
        apparent, abs_bin = divmod(population, 3)
        return schechter_fraction(distance, None if apparent == 0 else 11.5,
            11.5 if apparent == 0 else 12.5, absolute[abs_bin],
            absolute[abs_bin + 1], -23.28, -.94)

    radial = np.array([fraction(dl, population) for population in range(6)])
    n, dx, order, slab_width = 128, 3., 8, 4
    axis = (np.arange(n) + .5) * dx - 192.
    nodes, weights = np.polynomial.legendre.leggauss(order)
    with np.load(DATA / 'counts_3_sparse.npz', allow_pickle=False) as data:
        keys = data['all_keys']
    occupied_pop = keys // n**3
    flat = keys % n**3
    occupied_xyz = np.array(np.unravel_index(flat, (n,) * 3)).T
    occupied_exposure = np.zeros(len(keys), dtype=np.float64)
    total = np.zeros((6, 6), dtype=np.float64)

    for lower in range(0, n, slab_width):
        width = min(slab_width, n - lower)
        centers = np.array(np.meshgrid(axis[lower:lower + width], axis, axis,
                                       indexing='ij')).reshape(3, -1)
        block = np.zeros((6, 6, centers.shape[1]), dtype=np.float64)
        for a, b, d in product(range(order), repeat=3):
            xyz = centers + dx / 2 * nodes[[a, b, d], None]
            radius = np.linalg.norm(xyz, axis=0)
            active = (radius >= edges[0]) & (radius <= edges[-1])
            ids = np.flatnonzero(active)
            if not ids.size:
                continue
            r = radius[ids]
            pixels = hp.vec2pix(512, *(rotation @ xyz[:, ids]), nest=False)
            shell = np.clip(np.searchsorted(edges, r, side='right') - 1, 0, 5)
            weight = weights[a] * weights[b] * weights[d] / 8
            for population in range(6):
                block[population, shell, ids] += (
                    weight * maps[population // 3][pixels] * np.interp(r, rtab, radial[population]))
        total += block.sum(axis=2) * dx**3

        in_slab = ((occupied_xyz[:, 0] >= lower)
                   & (occupied_xyz[:, 0] < lower + width))
        selected = np.flatnonzero(in_slab)
        if selected.size:
            collapsed = block.sum(axis=1).reshape(6, width, n, n)
            local = occupied_xyz[selected].copy()
            local[:, 0] -= lower
            occupied_exposure[selected] = collapsed[
                occupied_pop[selected], local[:, 0], local[:, 1], local[:, 2]]

    previous_result_path = PREVIOUS / 'result.json'
    previous_result = json.loads(previous_result_path.read_text())
    if previous_result.get('quadrature_order') != 6:
        raise ValueError('the bound predecessor is not the order-six integral')
    if previous_result.get('observed_cells') != len(keys):
        raise ValueError('count-key cohort differs from the saved order-six result')
    previous_h5_path = PREVIOUS / 'selection_3.h5'
    order6_exposure = np.zeros(len(keys), dtype=np.float64)
    with h5py.File(previous_h5_path, 'r') as h:
        if int(h.attrs['quadrature_order']) != 6:
            raise ValueError('saved selection cube does not have order six')
        if h.attrs['status'] != 'INTEGRATION_COMPLETE_NOT_CALIBRATED':
            raise ValueError('saved order-six selection cube is incomplete or unresolved')
        cube = h['selection_shells']
        for population in range(6):
            selected = np.flatnonzero(occupied_pop == population)
            if selected.size:
                collapsed = np.asarray(cube[population], dtype=np.float64).sum(axis=0)
                xyz = occupied_xyz[selected]
                order6_exposure[selected] = collapsed[xyz[:, 0], xyz[:, 1], xyz[:, 2]]

    volume6 = np.asarray(previous_result['effective_volume_cMpc_h3'], dtype=np.float64)
    volume_abs = np.abs(total - volume6)
    volume_rel = volume_abs[volume6 > 0] / volume6[volume6 > 0]
    exposure_rel = np.abs(occupied_exposure - order6_exposure) / np.maximum(
        np.abs(order6_exposure), 1e-30)
    largest = np.argsort(exposure_rel)[-10:][::-1]
    report = {
        'classification': 'ORDER8_VS_SAVED_ORDER6_SELECTION_QUADRATURE_SENSITIVITY',
        'status': 'DIAGNOSTIC_ONLY_NOT_CALIBRATED_NOT_FIELD_INFERENCE',
        'source_commit': os.environ['CF4_EXPECTED_COMMIT'],
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'cosmology': cosmology,
        'grid': {'N': n, 'dx_cMpc_h': dx, 'box_cMpc_h': 384.},
        'quadrature': {'saved_order': 6, 'new_order': order,
                       'node_layout': 'tensor_product_Gauss_Legendre_per_voxel'},
        'inputs': {
            'count_keys': str(DATA / 'counts_3_sparse.npz'),
            'count_keys_sha256': sha256(DATA / 'counts_3_sparse.npz'),
            'saved_selection_result': str(previous_result_path),
            'saved_selection_cube': str(previous_h5_path),
            'maps': [{'path': str(path), 'sha256': sha256(path)} for path in map_paths],
            'technical_input_gate': technical['status'],
        },
        'global_effective_volume_cMpc_h3_order6': volume6.tolist(),
        'global_effective_volume_cMpc_h3_order8': total.tolist(),
        'global_effective_volume_absolute_difference': volume_abs.tolist(),
        'global_positive_order6_cells_max_relative_difference': float(np.max(volume_rel)),
        'global_positive_order6_cells_p95_relative_difference': float(np.percentile(volume_rel, 95)),
        'observed_population_cell_keys': int(len(keys)),
        'order6_zero_exposure_keys': int(np.count_nonzero(order6_exposure <= 0)),
        'order8_zero_exposure_keys': int(np.count_nonzero(occupied_exposure <= 0)),
        'observed_key_relative_exposure_max': float(np.max(exposure_rel)),
        'observed_key_relative_exposure_p95': float(np.percentile(exposure_rel, 95)),
        'largest_observed_key_relative_changes': [
            {'key': int(keys[i]), 'population': int(occupied_pop[i]),
             'voxel_ijk': occupied_xyz[i].tolist(),
             'order6_exposure': float(order6_exposure[i]),
             'order8_exposure': float(occupied_exposure[i]),
             'relative_difference': float(exposure_rel[i])}
            for i in largest],
        'elapsed_seconds': time.monotonic() - started,
        'host_peak_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'selection_calibration_complete': False,
        'angular_quadrature_certified': False,
        'quantitative_joint_likelihood_ready': False,
        'MW_M31_role_ambiguous': True,
        'M33_unresolved': True,
        'native_truth_ids_used_for_field_selection': False,
        'Q_GOAL': 'Numerical sensitivity of the existing CF4-conditioned count selection term; no field is inferred.',
        'Q_LEAN': 'One order-eight full-grid integral compared with the saved order-six result; no full rerun, fit, heldout score, or simulation.',
        'LG_same_new_field_requirement': 'MW/M31 roles remain ambiguous and M33 unresolved; later observables must constrain those same roles in the NEW evolved field at <=0.3 cMpc/h, with native truth IDs calibration/evaluation-only.',
    }
    OUT.mkdir(parents=True)
    (OUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in (
        'classification', 'global_positive_order6_cells_max_relative_difference',
        'observed_key_relative_exposure_max', 'order8_zero_exposure_keys',
        'elapsed_seconds', 'host_peak_GiB')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
