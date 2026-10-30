"""Conservatively aggregate native 1.5 Mpc/h observations to 3 Mpc/h.

This preserves the original row ownership and selection *volume average*.
It creates no new matter-field information or calibrated tracer nuisance.
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_native_128_observations_v1')
FINE_N, COARSE_N = 256, 128


def coarsen_sparse(keys, counts):
    """Sum disjoint 256-grid population/cell keys into 128-grid cells."""
    if keys.shape != counts.shape or np.any(counts <= 0):
        raise ValueError('invalid sparse count arrays')
    pop, cell = np.divmod(keys.astype(np.int64), FINE_N**3)
    if np.any(pop >= 6) or np.any(pop < 0) or np.any(cell < 0):
        raise ValueError('invalid population/cell keys')
    x, y, z = np.unravel_index(cell, (FINE_N,)*3)
    parent = pop * COARSE_N**3 + np.ravel_multi_index((x//2, y//2, z//2), (COARSE_N,)*3)
    unique, inverse = np.unique(parent, return_inverse=True)
    sums = np.bincount(inverse, weights=counts, minlength=len(unique)).astype(np.int64)
    if sums.sum() != counts.sum():
        raise AssertionError('count mass was not conserved')
    return unique, sums


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('submit numerical aggregation through Slurm')
    source_select = SOURCE / 'selection_1p5_v2/selection.h5'
    source_counts = SOURCE / 'native_data_v2/counts_1p5_sparse.npz'
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    with np.load(source_counts, allow_pickle=False) as source:
        counts = {label: coarsen_sparse(source[f'{label}_keys'], source[f'{label}_counts'])
                  for label in ('all', 'train', 'holdout')}
    np.savez_compressed(OUTPUT / 'counts_3_sparse.npz',
                        **{f'{label}_{field}': value for label, (keys, values) in counts.items()
                           for field, value in (('keys', keys), ('counts', values))})
    total = np.zeros((6, 6), dtype=np.float64)
    with h5py.File(source_select, 'r') as source, h5py.File(OUTPUT / 'selection_3.h5', 'x') as result:
        fine = source['selection_shells']
        if fine.shape != (6, 6, FINE_N, FINE_N, FINE_N):
            raise ValueError('native selection shape changed')
        coarse = result.create_dataset('selection_shells', shape=(6, 6, COARSE_N, COARSE_N, COARSE_N),
                                       dtype='f4', chunks=(1, 1, 2, 32, 32),
                                       compression='gzip', compression_opts=1)
        for low in range(0, COARSE_N, 2):
            fine_block = fine[:, :, 2*low:2*(low+2)]
            parent = fine_block.reshape(6, 6, 2, 2, COARSE_N, 2, COARSE_N, 2).mean(axis=(3, 5, 7))
            coarse[:, :, low:low+2] = parent
            total += parent.astype(np.float64).sum(axis=(2, 3, 4)) * 3.0**3
        result.attrs.update(status='VOLUME_AVERAGED_NATIVE_SELECTION_NOT_CALIBRATED',
                            source=str(source_select), box_cMpc_h=384., cell_cMpc_h=3.)
    with h5py.File(OUTPUT / 'selection_3.h5', 'r') as result:
        selection = result['selection_shells'][:].sum(axis=1)
    key = counts['all'][0]
    pop, cell = np.divmod(key, COARSE_N**3)
    x, y, z = np.unravel_index(cell, (COARSE_N,)*3)
    unsupported = int(np.count_nonzero(selection[pop, x, y, z] <= 0))
    train = dict(zip(*counts['train']))
    heldout = dict(zip(*counts['holdout']))
    if unsupported or any(train.get(int(k), 0) + heldout.get(int(k), 0) != int(v)
                          for k, v in zip(*counts['all'])):
        raise RuntimeError('native count support or train/holdout partition failed')
    report = dict(classification='NATIVE_COUNTS_AND_VOLUME_AVERAGED_SELECTION_N128',
                  source_selection=str(source_select), source_counts=str(source_counts),
                  count_totals={label: int(values.sum()) for label, (_keys, values) in counts.items()},
                  occupied_cells={label: int(len(keys)) for label, (keys, _values) in counts.items()},
                  occupied_zero_selection_cells=unsupported,
                  shell_effective_volume_cMpc_h3=total.tolist(),
                  source_cell_cMpc_h=1.5, output_cell_cMpc_h=3.0,
                  interpolation_or_new_information=False, survival_bias_calibrated=False,
                  N128_PM_state=False, R2_posterior=False)
    (OUTPUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('classification', 'count_totals', 'occupied_cells',
                                           'occupied_zero_selection_cells')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
