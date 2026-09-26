"""Check source HEALPix support and catalogue marks for the inclusive view.

This is an object-level diagnostic, not an angle-only mask construction or a
selection correction.  Run through Slurm; do not change the frozen catalogue.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import healpy as hp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cf4_twompp_disjoint_tracer_pilot_v3 import load_catalog, base

PARENT = Path('/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1')
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_point_selection_audit_v1')


def check_binding(entry):
    path = Path(entry['path'])
    if not path.is_absolute():
        path = ROOT/path
    raw = path.read_bytes()
    if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError(f'bound source changed: {path}')
    return path


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm job required')
    if OUT.exists():
        raise FileExistsError(OUT)
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    inputs = tracer['inputs']
    source = load_catalog(check_binding(inputs['twompp_catalog']))
    bright = base.load_completeness_map(check_binding(inputs['completeness_11_5']),512)
    faint = base.load_completeness_map(check_binding(inputs['completeness_12_5']),512)
    old = json.loads((ROOT/'config/cf4_datum_bearing_z0_twompp_datum_builder_program_v2.json').read_text())
    excluded = check_binding(old['bindings']['excluded_recnos'])
    with excluded.open(newline='',encoding='utf-8') as handle:
        prior = {int(row['recno']) for row in csv.DictReader(handle)}
    if len(prior) != 319:
        raise ValueError('old exception manifest changed')
    with np.load(PARENT/'row_diagnostics.npz',allow_pickle=False) as saved:
        rows = {key:saved[key] for key in saved.files}
    order = np.argsort(source['recno'])
    ids = order[np.searchsorted(source['recno'][order],rows['recno'])]
    np.testing.assert_array_equal(source['recno'][ids],rows['recno'])
    pixel = hp.ang2pix(512,np.deg2rad(90-source['DEC'][ids]),
                       np.deg2rad(source['RA'][ids]),nest=False)
    is_faint = rows['population']>=3
    map_value = np.where(is_faint,faint[pixel],bright[pixel])
    mark = np.where(is_faint,source['c12_5'][ids],source['c11_5'][ids])
    point_zero = map_value<=0
    mismatch = ~np.isfinite(mark) | (np.abs(mark-map_value)>.05)
    prior_flag = np.isin(rows['recno'],np.fromiter(prior,dtype=np.int64))
    quality = ~point_zero & ~mismatch & ~prior_flag
    np.testing.assert_array_equal(rows['source_survivor'],~rows['match'] & quality)
    categories = {}
    for label, mask in (('all',np.ones(len(ids),dtype=bool)),
                        ('matched',rows['match']),('nonmatched',~rows['match'])):
        categories[label] = dict(parent=int(mask.sum()),point_map_zero=int(np.count_nonzero(mask&point_zero)),
            mark_mismatch=int(np.count_nonzero(mask&mismatch)),
            prior_exception=int(np.count_nonzero(mask&prior_flag)),
            any_quality_failure=int(np.count_nonzero(mask&~quality)))
    report = dict(classification='R2_POINT_SELECTION_AUDIT_NO_MODEL',
        source_crossmatch_free_quality_reconstructs_old_survivor_exactly=True,
        categories=categories, cf4_match_independent_quality_rule=True,
        recommendation_limit='Pointwise zero exposure or mark mismatch is not an angle-only mask and cannot be repaired by arbitrary sky thinning. No catalogue changed or likelihood evaluated.')
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
