"""Preserved inclusive N128 count view and source-selection support only.

No CF4 joint likelihood, rate calibration, posterior, or catalogue promotion.
Runs in one short Slurm allocation because the 216-selection-quadrature HDF5
input is a numerical data product.  Imputed rows were already excluded from
the frozen eligible-parent row list.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1')
SELECTION = Path('/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1/selection_3.h5')
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1')
N = 128


def sparse(keys, mask):
    return np.unique(keys[mask], return_counts=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm job required')
    if OUT.exists():
        raise FileExistsError(OUT)
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    source = tracer['inputs']['cf4_twompp_crossmatch']
    path = ROOT/source['path']
    raw = path.read_bytes()
    if len(raw) != source['bytes'] or hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('crossmatch source binding changed')
    matches = set()
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['match_class'] != 'unmatched':
                matches.add(int(row['twompp_recno']))
    if len(matches) != 17007:
        raise ValueError('frozen crossmatch cardinality changed')
    with np.load(PARENT/'rows.npz', allow_pickle=False) as saved:
        rows = {key:saved[key] for key in saved.files}
    match = np.isin(rows['recno'], np.fromiter(matches,dtype=np.int64))
    old_used = rows['survives'] & ~rows['calibration']
    if np.any(match & rows['survives']):
        raise AssertionError('frozen survivor flag includes crossmatch')
    cell = np.floor(rows['position_cMpc_h']/3.).astype(np.int32)
    if np.any(cell < 0) or np.any(cell >= N):
        raise ValueError('eligible count row outside N128 box')
    keys = rows['population'].astype(np.int64)*N**3 + np.ravel_multi_index(cell.T,(N,)*3)
    with np.load(PARENT/'counts_3_sparse.npz',allow_pickle=False) as old:
        k, count = sparse(keys,old_used)
        np.testing.assert_array_equal(k,old['all_keys'])
        np.testing.assert_array_equal(count,old['all_counts'])
    exposure = np.full(len(keys),np.nan,dtype=np.float32)
    with h5py.File(SELECTION,'r') as handle:
        if handle.attrs['status'] != 'INTEGRATION_COMPLETE_NOT_CALIBRATED':
            raise ValueError('N128 selection source is incomplete')
        selected = handle['selection_shells']
        if selected.shape != (6,6,N,N,N):
            raise ValueError('unexpected selection grid')
        for lo in range(0,N,4):
            take = np.flatnonzero((cell[:,0]>=lo)&(cell[:,0]<lo+4))
            if take.size:
                slab = selected[:,:,lo:lo+4].sum(axis=1)
                exposure[take] = slab[rows['population'][take],cell[take,0]-lo,
                                      cell[take,1],cell[take,2]]
    if not np.all(np.isfinite(exposure)) or np.any(exposure < 0):
        raise ValueError('invalid source selection at eligible count rows')
    masks = dict(parent=np.ones(len(keys),dtype=bool), train=rows['train'],
                 holdout=rows['holdout'], calibration=rows['calibration'])
    counts = {}
    for label, mask in masks.items():
        counts[f'{label}_keys'], counts[f'{label}_counts'] = sparse(keys,mask)
    noncrossmatch_failure = ~match & ~rows['survives']
    report = dict(classification='R2_INCLUSIVE_N128_COUNT_DIAGNOSTIC_NO_LIKELIHOOD',
        source_crossmatch_sha256=source['sha256'], N=N, dx_cMpc_h=3.,
        parent=int(len(keys)), train=int(rows['train'].sum()),
        holdout=int(rows['holdout'].sum()), calibration=int(rows['calibration'].sum()),
        crossmatched_parent=int(match.sum()),
        noncrossmatched_old_survivors=int(rows['survives'].sum()),
        noncrossmatch_metadata_anomalies=int(noncrossmatch_failure.sum()),
        zero_exposure_parent=int(np.count_nonzero(exposure==0)),
        zero_exposure_crossmatched=int(np.count_nonzero((exposure==0)&match)),
        zero_exposure_noncrossmatch_anomalies=int(np.count_nonzero((exposure==0)&noncrossmatch_failure)),
        old_disjoint_count_recovered_exactly=True,
        inclusion_rule='all frozen eligible parent rows, irrespective of CF4 match; diagnostics only',
        limitations='No joint CF4 factor, model calibration, metadata-anomaly resolution, heldout predictive score, or posterior. Positive-exposure counts are not sufficient for validity.')
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'inclusive_counts_3_sparse.npz',**counts)
    np.savez_compressed(OUT/'row_diagnostics.npz',recno=rows['recno'],population=rows['population'],
        position_cMpc_h=rows['position_cMpc_h'],match=match,
        source_survivor=rows['survives'],train=rows['train'],holdout=rows['holdout'],
        calibration=rows['calibration'],exposure=exposure)
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
