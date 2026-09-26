"""Rebase actual 2M++ row datum to the frozen CF4/PM cosmology.

Reuses only observer-independent frozen survivor and split marks, keyed by
original recno. It does not reuse the incompatible selection integral.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_actual_selection import base, magnitude_h, mark_counts

NATIVE = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2')
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1')


def sparse_counts(pop, cell, used, train, held, n):
    key = pop.astype(np.int64)*n**3 + np.ravel_multi_index(cell.T, (n,)*3)
    result = {}
    for label, mask in (('all', used), ('train', used & train), ('holdout', used & held)):
        result[f'{label}_keys'], result[f'{label}_counts'] = np.unique(
            key[mask], return_counts=True)
    return result


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('submit catalogue preparation through Slurm')
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    common = contract['common_cosmology']
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    galaxy = base.load_catalog(tracer['inputs']['twompp_catalog']['path'])
    model = dict(h=common['h'], H0_km_s_Mpc=common['H0_km_s_Mpc'],
                 Omega_m=common['Omega_m'], Omega_b=common['Omega_b'],
                 Tcmb_K=common['Tcmb_K'])
    radius, magnitude = base.distance_and_absolute_magnitude(
        galaxy['Vcmb'], galaxy['Ksmag'], model)
    eligible, _, apparent, absolute = base.classify_disjoint_tracer(
        galaxy, set(), radius, magnitude_h(magnitude, common['h']),
        tracer['tracer_design'])
    with np.load(NATIVE/'galaxy_native.npz', allow_pickle=False) as old:
        previous = {key: old[key] for key in ('recno', 'survives', 'calibration',
                                               'train', 'holdout', 'population')}
    order = np.argsort(galaxy['recno'])
    idx = order[np.searchsorted(galaxy['recno'][order], previous['recno'])]
    np.testing.assert_array_equal(galaxy['recno'][idx], previous['recno'])
    # A newly eligible row has no frozen survivor mark and must not be silently
    # omitted; the source-data audit found none for this candidate cosmology.
    if np.count_nonzero(eligible) != np.count_nonzero(eligible[idx]):
        raise RuntimeError('newly eligible rows require fresh mark preparation')
    keep = eligible[idx]
    retained = {key: value[keep] for key, value in previous.items()}
    ids = idx[keep]
    population = (3*apparent[ids] + absolute[ids]).astype(np.int8)
    edges = np.array([5., 30., 60., 90., 120., 150., 180.])
    shell = np.clip(np.searchsorted(edges, radius[ids], side='right')-1, 0, 5)
    direction = base.supergalactic_unit_vectors(galaxy['RA'], galaxy['DEC'])[ids]
    pos = radius[ids, None]*direction+192.
    cell = np.floor(pos/(384./128)).astype(int)
    if np.any(cell < 0) or np.any(cell >= 128):
        raise ValueError('eligible row outside N128 box')
    used = retained['survives'] & ~retained['calibration']
    counts = sparse_counts(population, cell, used, retained['train'],
                           retained['holdout'], 128)
    yes, no = mark_counts(population, shell, retained['calibration'],
                          retained['survives'])
    if int(counts['all_counts'].sum()) != int(counts['train_counts'].sum()
                                             + counts['holdout_counts'].sum()):
        raise AssertionError('train/holdout totals not conserved')
    if int((yes+no).sum()) != int(retained['calibration'].sum()):
        raise AssertionError('survivor mark totals not conserved')
    published = contract['published_prior']
    nbar = np.asarray(published['original_mean_count_per_cell_bright_first']) * (
        3./published['original_cell_cMpc_h'])**3
    if nbar.shape != (6,) or np.any(nbar <= 0):
        raise ValueError('invalid published rate')
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'counts_3_sparse.npz', **counts)
    np.savez_compressed(OUT/'rows.npz', recno=retained['recno'],
        population=population, shell=shell, position_cMpc_h=pos.astype(np.float32),
        survives=retained['survives'], calibration=retained['calibration'],
        train=retained['train'], holdout=retained['holdout'])
    np.savez_compressed(OUT/'survival_marks.npz', yes=yes, no=no)
    report = dict(classification='R2_COMMON_COSMOLOGY_CATALOGUE_N128_NO_SELECTION',
                  common_cosmology=common, source_catalogue=tracer['inputs']['twompp_catalog'],
                  retained_parent_rows=int(len(ids)), dropped_parent_rows=int((~keep).sum()),
                  used_counts=int(counts['all_counts'].sum()),
                  train_counts=int(counts['train_counts'].sum()),
                  holdout_counts=int(counts['holdout_counts'].sum()),
                  calibration_parent_marks=int((yes+no).sum()),
                  calibration_survivor_marks=int(yes.sum()),
                  population_counts=np.bincount(population[used], minlength=6).tolist(),
                  external_rate_per_N128_cell_bright_first=nbar.tolist(),
                  external_bias_bright_first=published['linear_regime_bias_bright_first'],
                  selection_integral_recomputed=False,
                  quantitative_joint_likelihood_ready=False,
                  posterior_or_LG_identification=False)
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
