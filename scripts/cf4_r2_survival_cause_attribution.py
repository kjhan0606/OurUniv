"""Attribute calibration-mark sky variation to CF4 crossmatch versus remainder.

This is a fixed-mark diagnostic only. It does not fit a selection law or use
the 2M++ count holdout, IC phases, native halo truth, or any LG role label.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from cf4_r2_survival_holdout_screen import sky_mark_test

ROOT = Path(__file__).resolve().parents[1]
ROWS = Path('/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1/rows.npz')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_survival_cause_attribution_v1')


def sky_result(rows, success):
    selected = {key:rows[key] for key in ('position_cMpc_h','population','shell','calibration')}
    selected['survives'] = success
    group = rows['population'].astype(np.int32)*6+rows['shell'].astype(np.int32)
    mark = rows['calibration']
    yes = np.bincount(group[mark & success],minlength=36).reshape(6,6)
    no = np.bincount(group[mark & ~success],minlength=36).reshape(6,6)
    return sky_mark_test(selected,yes,no)


def octant_table(rows, match):
    offset = rows['position_cMpc_h']-192.
    octant = ((offset[:,0]>=0).astype(np.int32)*4
              +(offset[:,1]>=0).astype(np.int32)*2
              +(offset[:,2]>=0).astype(np.int32))
    calibration = rows['calibration']
    selected = (rows['population']==3)&(rows['shell']==5)&calibration
    table = []
    for sector in range(8):
        take = selected & (octant==sector)
        table.append(dict(octant=sector,parent=int(take.sum()),
                          crossmatched=int(np.count_nonzero(take & match)),
                          surviving=int(np.count_nonzero(take & rows['survives'])),
                          noncrossmatch_other_failure=int(np.count_nonzero(
                              take & ~match & ~rows['survives']))))
    return table


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm job required')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    bound = tracer['inputs']['cf4_twompp_crossmatch']
    path = ROOT/bound['path']
    raw = path.read_bytes()
    if len(raw)!=bound['bytes'] or hashlib.sha256(raw).hexdigest()!=bound['sha256']:
        raise ValueError('crossmatch input binding changed')
    matches = set()
    with path.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['match_class'].strip()!='unmatched':
                matches.add(int(row['twompp_recno']))
    if len(matches)!=17007:
        raise ValueError('crossmatch unique target count changed')
    with np.load(ROWS,allow_pickle=False) as f:
        rows = {key:f[key] for key in ('recno','position_cMpc_h','population',
                                       'shell','calibration','survives')}
    match = np.isin(rows['recno'],np.fromiter(matches,dtype=np.int64))
    if np.any(match & rows['survives']):
        raise AssertionError('crossmatched row incorrectly marked surviving')
    original = sky_result(rows,rows['survives'])
    crossmatch = sky_result(rows,match)
    subset = ~match
    residual_rows = {key:value[subset] for key,value in rows.items()}
    residual = sky_result(residual_rows,residual_rows['survives'])
    calibration = rows['calibration']
    failure = calibration & ~rows['survives']
    report = dict(classification='R2_SURVIVAL_SKY_CAUSE_ATTRIBUTION',
        source_commit=os.environ['EXPECTED_COMMIT'],
        crossmatch_binding_sha256=bound['sha256'],
        total_calibration_marks=int(calibration.sum()),
        total_calibration_failures=int(failure.sum()),
        calibration_failures_with_CF4_crossmatch=int(np.count_nonzero(failure & match)),
        calibration_failures_without_CF4_crossmatch=int(np.count_nonzero(failure & ~match)),
        original_survival_octant_test=original,
        CF4_crossmatch_octant_test=crossmatch,
        noncrossmatched_residual_survival_octant_test=residual,
        top_original_stratum_population3_shell5_octants=octant_table(rows,match),
        interpretation_limit='Attribution of binary mark failures, not a causal proof or a volume selection function',
        count_holdout_used=False,posterior_or_selection_model_fit=False)
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
