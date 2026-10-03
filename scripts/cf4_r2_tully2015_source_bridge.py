"""Source-only CF4/Tully-2015/2M++ group bridge, never a likelihood.

The archived 2015 CDS tables are not assumed identical to the later EDD CF4
group construction. PGC1 and the published 2M++ cross-ID are checked rather
than silently treating CF4 1PGC, Tully Nest, and Lavaux-Hudson GID as equal.
"""

import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_tully2015_source_bridge_v1')
HASHES = {
    'tully2015_ReadMe': '66ff93d0ba029e91b9c2f13fb5c62f7e874f04ed6a1b8c021d3c20a7a7c3762a',
    'tully2015_table3.dat.gz': 'dd85454fd68acd0ef7aeb432a4a336d07b31df7a79b37ab18b695fb0b4483618',
    'tully2015_table4.dat.gz': '902541fe765d50955038a60730ced8884151eb3963908dba127596298e33f771',
    'tully2015_table5.dat.gz': '337b9f24484ae34c973a7187fa002807b345e47e76ffabce066549783e2b2844',
    'cf4_groups.csv': 'bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334',
    'cf4_galaxies.csv': '28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709',
    '2mpp_catalog.csv': '05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d',
    'cf4_2mpp_crossmatch_v1.csv': '64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf',
}


def bound(name):
    path = ROOT / 'data' / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError(f'changed source: {name}')
    return path


def rows(name):
    with bound(name).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def records(name):
    with gzip.open(bound(name), 'rt', encoding='ascii') as stream:
        return [line.rstrip('\n') for line in stream]


def integer(line, start, stop):
    value = line[start:stop].strip()
    return int(value) if value else None


def stats(values):
    if not values:
        return None
    x = np.asarray(values, dtype=np.float64)
    return {'n': len(values), 'median': float(np.median(x)),
            'p90': float(np.percentile(x, 90)), 'p99': float(np.percentile(x, 99)),
            'maximum': float(np.max(x))}


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    bound('tully2015_ReadMe')
    table3, table4, table5 = (records('tully2015_table3.dat.gz'),
                              records('tully2015_table4.dat.gz'),
                              records('tully2015_table5.dat.gz'))
    if tuple(map(len, (table3, table4, table5))) != (25474, 43038, 43038):
        raise ValueError('CDS row counts differ from published ReadMe')

    # ReadMe byte columns are one-indexed, inclusive; Python slices are zero-indexed.
    nest = {}
    by_pgc1 = defaultdict(list)
    for line in table3:
        group = {'nest': integer(line, 3, 9), 'nmb': integer(line, 10, 13),
                 'pgc1': integer(line, 14, 21), 'vcmb': integer(line, 46, 51)}
        if group['nest'] in nest:
            raise ValueError('duplicate Tully nest')
        nest[group['nest']] = group
        by_pgc1[group['pgc1']].append(group)
    members = defaultdict(list)
    for line in table4:
        members[integer(line, 3, 9)].append((integer(line, 10, 17),
                                              integer(line, 64, 69)))
    if set(members) != set(nest):
        raise ValueError('Tully group/member coverage differs')
    bad_membership = [gid for gid, group in nest.items()
                      if group['nmb'] != len(members[gid])]
    mean_error = [abs(group['vcmb'] - np.mean([v for _, v in members[gid]]))
                  for gid, group in nest.items() if len(members[gid]) > 1]

    # The combined table supplies an explicit, *published* 2M++ cross-ID.
    table5_by_pgc = defaultdict(list)
    by_nest = defaultdict(list)
    for line in table5:
        item = {'pgc': integer(line, 0, 7), 'nest': integer(line, 99, 105),
                'pgc1': integer(line, 110, 117), 'gid2mpp': integer(line, 241, 245),
                'group_vcmb': integer(line, 172, 177)}
        table5_by_pgc[item['pgc']].append(item)
        by_nest[item['nest']].append(item)
    inconsistent_combined_nest = sum(any(x['pgc1'] != group['pgc1']
                                         or x['group_vcmb'] != group['vcmb']
                                         for x in by_nest[gid])
                                     for gid, group in nest.items())

    cf4 = rows('cf4_groups.csv')
    cf4_galaxies = rows('cf4_galaxies.csv')
    mpp = {r['recno']: r for r in rows('2mpp_catalog.csv')}
    matches = rows('cf4_2mpp_crossmatch_v1.csv')
    if len(cf4) != 38053 or len(mpp) != 72973:
        raise ValueError('local source cardinality changed')
    cf4_by_pgc = {int(r['1PGC']): r for r in cf4}
    cf4_to_tully = {}
    for pgc, row in cf4_by_pgc.items():
        candidates = by_pgc1.get(pgc, ())
        if len(candidates) == 1:
            cf4_to_tully[pgc] = candidates[0]
    dm_members = Counter(int(r['1PGC']) for r in cf4_galaxies)
    group_delta_cmb = []
    group_delta_3k = []
    group_ngal_vs_tully = []
    for pgc, group in cf4_to_tully.items():
        row = cf4_by_pgc[pgc]
        group_delta_cmb.append(abs(float(row['Vcmb']) - group['vcmb']))
        group_delta_3k.append(abs(float(row['V3k']) - group['vcmb']))
        group_ngal_vs_tully.append(abs(int(row['Ngal']) - group['nmb']))

    # Eligible secure crossmatches test group relationships; no inferred join
    # is promoted when a PGC or nest has multiple catalogued associations.
    eligible_path = Path('/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1/row_diagnostics.npz')
    with np.load(eligible_path, allow_pickle=False) as saved:
        eligible = set(map(int, saved['recno']))
    secure = 0
    secure_with_unique_tully = 0
    same_pgc_tully_nest = 0
    t5_gid_equal_mpp_gid = 0
    t5_gid_disagree_mpp_gid = 0
    t5_gid_missing_either = 0
    cf4_group_to_tully_nests = defaultdict(set)
    for row in matches:
        if row['match_class'] != 'secure_joint_mark' or int(row['twompp_recno']) not in eligible:
            continue
        secure += 1
        pgc = int(row['PGC'])
        cf4_pgc = int(row['1PGC'])
        candidates = table5_by_pgc.get(pgc, ())
        if len(candidates) != 1:
            continue
        item = candidates[0]
        secure_with_unique_tully += 1
        if item['nest'] in nest:
            cf4_group_to_tully_nests[cf4_pgc].add(item['nest'])
        group = cf4_to_tully.get(cf4_pgc)
        if group is not None and item['nest'] == group['nest']:
            same_pgc_tully_nest += 1
        local_gid = mpp[row['twompp_recno']]['GID']
        published_gid = item['gid2mpp']
        if not local_gid or not published_gid:
            t5_gid_missing_either += 1
        elif int(local_gid) == published_gid:
            t5_gid_equal_mpp_gid += 1
        else:
            t5_gid_disagree_mpp_gid += 1

    result = {
        'classification': 'R2_TULLY2015_SOURCE_BRIDGE_ONLY',
        'source_url': 'https://cdsarc.cds.unistra.fr/ftp/J/AJ/149/171/',
        'source_sha256': HASHES,
        'tully_nests': len(nest), 'tully_members': len(table4),
        'tully_combined_rows': len(table5),
        'tully_nmb_membership_mismatches': len(bad_membership),
        'tully_group_velocity_minus_member_mean_km_s': stats(mean_error),
        'tully_combined_nest_internal_inconsistencies': inconsistent_combined_nest,
        'cf4_groups_with_unique_pgc1_tully_match': len(cf4_to_tully),
        'cf4_groups_with_multiple_pgc1_tully_matches': sum(len(v) > 1 for v in by_pgc1.values()
                                                            if v and v[0]['pgc1'] in cf4_by_pgc),
        'cf4_vcmb_minus_tully_group_vcmba_km_s': stats(group_delta_cmb),
        'cf4_v3k_minus_tully_group_vcmba_km_s_frame_mismatched_diagnostic_only': stats(group_delta_3k),
        'cf4_ngal_minus_tully_nmb_absolute': stats(group_ngal_vs_tully),
        'cf4_distance_members_with_unique_tully_group': sum(dm_members[pgc]
                                                            for pgc in cf4_to_tully),
        'eligible_secure_cf4_2mpp_matches': secure,
        'secure_with_unique_tully_member_PGC': secure_with_unique_tully,
        'secure_member_nest_equals_cf4_pgc1_matched_nest': same_pgc_tully_nest,
        'secure_member_published_2mpp_gid_equals_local_gid': t5_gid_equal_mpp_gid,
        'secure_member_published_2mpp_gid_differs_local_gid': t5_gid_disagree_mpp_gid,
        'secure_member_published_or_local_2mpp_gid_missing': t5_gid_missing_either,
        'cf4_groups_with_multiple_tully_nests_via_secure_members': sum(
            len(v) > 1 for v in cf4_group_to_tully_nests.values()),
        'limitation': ('2015 CDS tables may differ from later EDD CF4 group revisions. '
                       'A PGC1 or published 2M++ cross-ID is evidence of a catalogue '
                       'association, not identity of grouping algorithms or a survey '
                       'selection law. V3k and adjusted CMB group velocities are in '
                       'different frames. No mock, normalized joint likelihood, '
                       'calibrated bias or actual-data posterior is produced.'),
    }
    OUT.mkdir(parents=True)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
