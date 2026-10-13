"""Can the published CF4 group V3k be reconstructed from member Vcmb?

Read-only, source-bound catalogue comparison. It does not assert an averaging
algorithm from numerical proximity and does not fit a galaxy/field likelihood.
"""

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ROOT/'data/cf4_groups.csv'
GALAXIES = ROOT/'data/cf4_galaxies.csv'
MAPPING = ROOT/'data/cf4_2mpp_crossmatch_v1.csv'
ELIGIBLE = Path('/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1/row_diagnostics.npz')
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_group_redshift_source_audit_v1')
HASHES = {
    GROUPS:'bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334',
    GALAXIES:'28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709',
    MAPPING:'64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf',
}


def rows(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def quantiles(values):
    if not values:
        return None
    return {key:float(value) for key,value in zip(
        ('median','p90','p99','maximum'),np.percentile(values,(50,90,99,100)))}


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm job required')
    if OUT.exists():
        raise FileExistsError(OUT)
    for path,expected in HASHES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'frozen source changed: {path}')
    with np.load(ELIGIBLE,allow_pickle=False) as saved:
        eligible = set(map(int,saved['recno']))
    secure_groups = set()
    for row in rows(MAPPING):
        if row['match_class']=='secure_joint_mark' and int(row['twompp_recno']) in eligible:
            secure_groups.add(row['1PGC'])
    members = defaultdict(list)
    for row in rows(GALAXIES):
        if row['Vcmb'].strip():
            members[row['1PGC']].append(float(row['Vcmb']))
    differences = defaultdict(list)
    counts = defaultdict(int)
    n_equals = defaultdict(int)
    n_source = defaultdict(list)
    for row in rows(GROUPS):
        key = row['1PGC']
        if not row['V3k'].strip() or key not in members:
            continue
        values = np.asarray(members[key])
        group_v = float(row['V3k'])
        source_n = int(row['Ngal'])
        for label,selected in (('all',True),('eligible_secure',key in secure_groups),
                               ('eligible_secure_multimember',key in secure_groups and len(values)>1)):
            if not selected:
                continue
            counts[label] += 1
            n_equals[label] += int(source_n==len(values))
            n_source[label].append(len(values))
            differences[label+'_mean'].append(abs(group_v-float(values.mean())))
            differences[label+'_median'].append(abs(group_v-float(np.median(values))))
    report = dict(classification='CF4_GROUP_V3K_SOURCE_AGGREGATION_AUDIT_NOT_ALGORITHM_PROOF',
        source_sha256={path.name:digest for path,digest in HASHES.items()},
        group_count=counts,group_Ngal_equals_individual_row_count=n_equals,
        individual_member_count_quantiles={label:quantiles(vals) for label,vals in n_source.items()},
        absolute_V3k_minus_individual_aggregate_km_s={label:quantiles(vals) for label,vals in differences.items()},
        group_v3k_is_deterministic_mean_established=False,
        note='Near agreement alone cannot establish catalogue group-V3k construction, missing members, weights, frame conversion, or independent measurement errors. No likelihood/posterior changed.')
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
