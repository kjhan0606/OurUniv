"""Build a source-only ownership ledger for the active v6 training factors.

No mark values, held-out counts, field state, likelihood scores, or truth
identities are read. This is bookkeeping, not a joint likelihood.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/gpfs/kjhan/CF4/z0_density')
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
POINTS = BASE/'r2_point_mark_manifest_v1/points.npz'
FP = BASE/'r2_source_observation_assembly_v1/observations.npz'
GROUP = BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
CROSSMATCH = ROOT/'data/cf4_2mpp_crossmatch_v1.csv'
NGRID = 128
EXPECTED_HASHES = {
    SPLIT: '9528b54a1e9dc876052771c21c0330ebba5f9b2b56966ab3cbe593a71cfc2b74',
    POINTS: '9377de48a3cc045b8432266ee5124d3540411fdcddd0d83b8036aa6744d11139',
    FP: '90d6aecca7f62cd09767bd444df99cc60b8a557b21605cbdc31afdc38ea10b28',
    GROUP: 'f80e34099894c3fdd090e45e775d5352a1d06b19583be47c11558d5af5941b38',
    CROSSMATCH: '64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_ledger(train_keys, train_counts, points, fp_groups, crossmatch_rows):
    """Return count-cell, CF4-group, FP-row and edge ownership records."""
    keys = np.asarray(train_keys, dtype=np.int64)
    counts = np.asarray(train_counts, dtype=np.int64)
    if (keys.ndim != 1 or counts.shape != keys.shape or not len(keys)
            or len(np.unique(keys)) != len(keys) or np.any(keys < 0)
            or np.any(counts <= 0)):
        raise ValueError('v6 training count key/value contract is invalid')
    count_by_key = dict(zip(map(int, keys), map(int, counts)))
    if set(fp_groups) != {str(row['label']) for row in fp_groups.values()}:
        raise ValueError('FP group map key/label mismatch')

    point_by_recno = {}
    point_multiplicity = Counter()
    for row in points:
        recno = int(row['recno'])
        if recno in point_by_recno:
            raise ValueError(f'duplicate 2M++ recno {recno}')
        role = int(row['role'])
        if role not in (0, 1, 2):
            raise ValueError(f'unknown point role {role}')
        key = int(row['population'])*NGRID**3 + int(row['flat_cell'])
        if role == 0:
            if key not in count_by_key:
                raise ValueError(f'training point {recno} maps outside v6 training keys')
            point_multiplicity[key] += 1
        point_by_recno[recno] = dict(role=role, key=key)
    if {key: point_multiplicity[key] for key in count_by_key} != count_by_key:
        mismatches = [key for key in count_by_key
                      if point_multiplicity[key] != count_by_key[key]]
        raise ValueError(f'point manifest does not reconstruct v6 counts at {len(mismatches)} keys')

    group_recno = defaultdict(set)
    group_ambiguous_edges = Counter()
    group_edge_classes = defaultdict(Counter)
    pgc_groups = defaultdict(set)
    for group in fp_groups.values():
        for _, pgc in group['fp_rows']:
            pgc_groups[int(pgc)].add(str(group['label']))

    edge_rows = []
    for raw in crossmatch_rows:
        pgc = int(raw['PGC'])
        labels = sorted(pgc_groups.get(pgc, ()))
        if not labels:
            continue
        match_class = str(raw['match_class']).strip()
        raw_recno = str(raw.get('twompp_recno', '')).strip()
        recno = int(raw_recno) if raw_recno else None
        status = 'ambiguous_association'
        if len(labels) != 1:
            status = 'CF4_group_assignment_collision'
        elif match_class == 'secure_joint_mark' and recno is not None:
            point = point_by_recno.get(recno)
            if point is None:
                status = 'secure_edge_recno_not_in_point_manifest'
            elif point['role'] != 0:
                status = 'secure_edge_outside_training_fold'
            else:
                status = 'secure_training_count_link'
                group_recno[labels[0]].add(recno)
        if len(labels) == 1:
            group_edge_classes[labels[0]][match_class] += 1
            if status != 'secure_training_count_link':
                group_ambiguous_edges[labels[0]] += 1
        elif len(labels) > 1:
            for candidate_label in labels:
                group_edge_classes[candidate_label][match_class] += 1
                group_ambiguous_edges[candidate_label] += 1
        edge_rows.append(dict(PGC=pgc, candidate_group_labels=';'.join(labels),
            match_class=match_class, twompp_recno='' if recno is None else recno,
            status=status, ownership='count_grid_owns_cell_count; CF4_group_owns_mark'))

    recno_groups = defaultdict(set)
    for label, recnos in group_recno.items():
        for recno in recnos:
            recno_groups[recno].add(label)
    conflicting_recnos = {recno for recno, labels in recno_groups.items() if len(labels) > 1}
    training_key_links = defaultdict(set)
    for label, recnos in group_recno.items():
        for recno in recnos:
            point = point_by_recno[recno]
            training_key_links[point['key']].add((recno, label))

    count_rows = []
    for key in sorted(count_by_key):
        population, flat_cell = divmod(key, NGRID**3)
        links = sorted(training_key_links.get(key, ()))
        linked_groups = sorted({label for _, label in links})
        count_rows.append(dict(count_key=key, population=population,
            flat_cell=flat_cell, count=count_by_key[key],
            factor_owner='2Mpp_population_voxel_count_once',
            training_source_points=int(point_multiplicity[key]),
            secure_linked_recno_count=len({recno for recno, _ in links}),
            linked_CF4_group_labels=';'.join(linked_groups),
            linkage_class=('count_only' if not linked_groups else
                'count_with_one_CF4_group' if len(linked_groups) == 1 else
                'count_with_multiple_CF4_groups')))

    group_rows = []
    fp_rows = []
    for label in sorted(fp_groups):
        group = fp_groups[label]
        recnos = sorted(group_recno.get(label, ()))
        collision = any(recno in conflicting_recnos for recno in recnos)
        anchor_count = int(group['anchor_count'])
        if collision:
            category = 'association_conflict_unresolved'
        elif not recnos and group_ambiguous_edges[label]:
            category = 'ambiguous_association_unresolved'
        elif not recnos and anchor_count == 0:
            category = 'unanchored_selected_group_conditional'
        elif not recnos:
            category = 'anchor_marked_group_without_direct_count_link'
        elif len(recnos) == 1:
            category = 'one_linked_count_point'
        else:
            category = 'multi_member_shared_latent_candidate'
        group_rows.append(dict(source_group_label=label,
            FP_row_count=len(group['fp_rows']), anchor_row_count=anchor_count,
            secure_training_recno_count=len(recnos),
            secure_training_recnos=';'.join(map(str, recnos)),
            ambiguous_edge_count=int(group_ambiguous_edges[label]),
            secure_edge_classes=json.dumps(dict(group_edge_classes[label]), sort_keys=True),
            category=category,
            mark_factor_owner='CF4_group_marks_once; member rows share group factor',
            association_collision=bool(collision)))
        for row_id, pgc in group['fp_rows']:
            fp_rows.append(dict(fp_row_index=int(row_id), PGC=int(pgc),
                source_group_label=label,
                mark_owner=f'CF4_group_marks_once:{label}',
                individual_2Mpp_redshift_factor='forbidden_if_same_secure_object'))

    if len({row['count_key'] for row in count_rows}) != len(keys):
        raise AssertionError('a v6 count factor was omitted or duplicated')
    if len({row['fp_row_index'] for row in fp_rows}) != len(fp_rows):
        raise AssertionError('a CF4 FP row has multiple group-factor owners')
    return count_rows, group_rows, fp_rows, edge_rows, conflicting_recnos


def _write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CF4_EXPECTED_COMMIT'):
        raise RuntimeError('Slurm job and pinned source commit are required')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD^{commit}'],
                                     cwd=ROOT, text=True).strip()
    expected = subprocess.check_output(
        ['git', 'rev-parse', '--verify', f"{os.environ['CF4_EXPECTED_COMMIT']}^{{commit}}"],
        cwd=ROOT, text=True).strip()
    if actual != expected:
        raise RuntimeError('submitted source revision mismatch')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    if out != BASE/'r2_v6_factor_ownership_ledger_20261004_v1' or out.exists():
        raise FileExistsError('unexpected or previously used output path')
    started = time.monotonic()
    hashes = {path: sha256(path) for path in EXPECTED_HASHES}
    if hashes != EXPECTED_HASHES:
        raise ValueError('one or more frozen v6 source hashes changed')

    with np.load(SPLIT, allow_pickle=False) as split:
        train_keys = split['train_keys'].astype(np.int64, copy=True)
        train_counts = split['train_counts'].astype(np.int64, copy=True)
        point_recno = split['point_recno'].astype(np.int64, copy=True)
        point_role = split['point_role'].astype(np.int8, copy=True)
        labels = split['fp_source_group'].astype(str)
        fp_role = split['fp_role'].astype(np.int8, copy=True)
    with np.load(POINTS, allow_pickle=False) as source_points:
        recnos = source_points['recno'].astype(np.int64, copy=True)
        populations = source_points['population'].astype(np.int64, copy=True)
        flat_cells = source_points['flat_cell'].astype(np.int64, copy=True)
    with np.load(FP, allow_pickle=False) as observations:
        pgcs = observations['PGC'].astype(np.int64, copy=True)
        fp_labels = observations['source_group'].astype(str)
    with np.load(GROUP, allow_pickle=False) as groups:
        group_labels = groups['group_labels'].astype(str)
        row_group = groups['row_group'].astype(np.int64, copy=True)
        anchor_group = groups['anchor_group'].astype(np.int64, copy=True)

    if not (len(point_recno) == len(point_role) and len(labels) == len(fp_role)
            and len(recnos) == len(populations) == len(flat_cells)
            and len(pgcs) == len(fp_labels) and np.array_equal(labels, group_labels)
            and np.array_equal(fp_labels, labels[row_group])):
        raise ValueError('frozen split, point, FP and group arrays are misaligned')
    if np.any(~np.isin(fp_role, (0, 1, 2))):
        raise ValueError('unknown FP split role')
    point_role_by_recno = dict(zip(map(int, point_recno), map(int, point_role)))
    if len(point_role_by_recno) != len(point_recno):
        raise ValueError('v6 split repeats a point recno')
    point_map = []
    for recno, pop, flat in zip(recnos, populations, flat_cells):
        key = int(pop)*NGRID**3+int(flat)
        role = point_role_by_recno.get(int(recno), 2)
        point_map.append(dict(recno=int(recno), role=int(role),
                              population=int(pop), flat_cell=int(flat)))
    fp_by_group = {str(label): {'label': str(label), 'fp_rows': [], 'anchor_count': 0}
                   for label, role in zip(labels, fp_role) if int(role) == 0}
    for row_index, (pgc, label) in enumerate(zip(pgcs, fp_labels)):
        label = str(label)
        if label in fp_by_group:
            fp_by_group[label]['fp_rows'].append((row_index, int(pgc)))
    anchor_counts = np.bincount(anchor_group, minlength=len(labels))
    for index, label in enumerate(labels):
        label = str(label)
        if label in fp_by_group:
            fp_by_group[label]['anchor_count'] = int(anchor_counts[index])
    if len(fp_by_group) != 6028:
        raise ValueError(f'expected 6,028 frozen v6 training FP groups, got {len(fp_by_group)}')

    training_pgcs = {int(pgc) for group in fp_by_group.values()
                     for _, pgc in group['fp_rows']}
    crossmatch_rows = []
    with CROSSMATCH.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            try:
                pgc = int(row['PGC'])
            except (KeyError, ValueError):
                continue
            if pgc in training_pgcs:
                crossmatch_rows.append(row)

    count_rows, group_rows, fp_rows, edge_rows, conflicts = make_ledger(
        train_keys, train_counts, point_map, fp_by_group, crossmatch_rows)
    if int(train_counts.sum()) != 47121:
        raise ValueError('active v6 target is not the pinned 47,121-count split')
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out/'count_factors.csv', count_rows, list(count_rows[0]))
    _write_csv(out/'group_factors.csv', group_rows, list(group_rows[0]))
    _write_csv(out/'fp_mark_ownership.csv', fp_rows, list(fp_rows[0]))
    _write_csv(out/'crossmatch_edges.csv', edge_rows, list(edge_rows[0]))
    category_counts = Counter(row['category'] for row in group_rows)
    key_class_counts = Counter(row['linkage_class'] for row in count_rows)
    result = dict(status='V6_TRAINING_FACTOR_OWNERSHIP_LEDGER_NOT_JOINT_LIKELIHOOD',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=actual,
        split='r2_sky_closed_split_v6', training_count_factor_keys=len(count_rows),
        training_count_total=int(train_counts.sum()),
        training_point_key_reconstruction='exact',
        count_factor_key_classes=dict(key_class_counts),
        training_CF4_groups=len(group_rows), training_FP_mark_rows=len(fp_rows),
        group_factor_classes=dict(category_counts),
        crossmatch_training_edges=len(edge_rows),
        crossmatch_status_counts=dict(Counter(row['status'] for row in edge_rows)),
        recno_group_conflict_count=len(conflicts),
        unique_count_factor_owners=len({row['count_key'] for row in count_rows}),
        unique_group_factor_owners=len({row['source_group_label'] for row in group_rows}),
        heldout_count_values_read=False, heldout_mark_values_read=False,
        likelihood_scores_read=False, field_state_read=False, PM_evolutions=0,
        shared_latent_count_mark_kernel_wired=False, R2_complete=False,
        selection_group_inclusion_calibrated=False,
        MW_M31='ambiguous roles must constrain the same NEW evolved LG field',
        M33='unresolved; same NEW field at <=0.3 cMpc/h required',
        source_sha256={str(path): digest for path, digest in hashes.items()},
        outputs={name: sha256(out/name) for name in
            ('count_factors.csv','group_factors.csv','fp_mark_ownership.csv','crossmatch_edges.csv')},
        elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n',
                                   encoding='utf-8')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
