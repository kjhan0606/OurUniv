"""Freeze a score-blind, training-only multi-member CF4/2M++ graph control."""
import csv
import hashlib
import json
import os
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


def eligible_multilink_training_groups(labels, fp_role, fp_row_count,
                                        anchor_count, group_members,
                                        point_role_by_recno):
    eligible = []
    for index, label in enumerate(labels):
        members = sorted(group_members.get(str(label), ()))
        if (int(fp_role[index]) == 0 and int(fp_row_count[index]) == 1
                and int(anchor_count[index]) == 0 and len(members) >= 2
                and all(point_role_by_recno.get(recno, 2) == 0 for recno in members)):
            eligible.append(str(label))
    return sorted(eligible)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('submit this source graph census to Slurm')
    expected = os.environ.get('CF4_EXPECTED_COMMIT')
    if not expected:
        raise RuntimeError('CF4_EXPECTED_COMMIT must pin submitted source')

    def resolve(revision):
        return subprocess.check_output(
            ['git', 'rev-parse', '--verify', f'{revision}^{{commit}}'],
            cwd=ROOT, text=True).strip()

    expected_revision, source_revision = resolve(expected), resolve('HEAD')
    if expected_revision != source_revision:
        raise RuntimeError('source commit mismatch')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    with np.load(SPLIT, allow_pickle=False) as data:
        point_recno = data['point_recno'].astype(np.int64, copy=True)
        point_role = data['point_role'].astype(np.int8, copy=True)
        labels = data['fp_source_group'].astype(str)
        fp_role = data['fp_role'].astype(np.int8, copy=True)
    with np.load(POINTS, allow_pickle=False) as data:
        point_recno_catalogue = data['recno'].astype(np.int64, copy=True)
        population = data['population'].astype(np.int8, copy=True)
        flat_cell = data['flat_cell'].astype(np.int64, copy=True)
        radius = data['radius_cMpc_h'].astype(np.float64, copy=True)
    with np.load(FP, allow_pickle=False) as data:
        pgc = data['PGC'].astype(np.int64, copy=True)
        source_group = data['source_group'].astype(str)
    with np.load(GROUP, allow_pickle=False) as data:
        group_labels = data['group_labels'].astype(str)
        row_group = data['row_group'].astype(np.int64, copy=True)
        anchor_group = data['anchor_group'].astype(np.int64, copy=True)

    np.testing.assert_array_equal(labels, group_labels)
    if (len(point_recno) != len(point_role) or len(labels) != len(fp_role)
            or len(point_recno_catalogue) != len(population)
            or len(population) != len(flat_cell) or len(flat_cell) != len(radius)):
        raise ValueError('v6 split, point catalogue and group geometry are misaligned')
    point_role_by_recno = {int(recno): int(role)
                           for recno, role in zip(point_recno, point_role)}
    point_index = {int(recno): i for i, recno in enumerate(point_recno_catalogue)}
    label_index = {str(label): i for i, label in enumerate(labels)}
    pgc_group = {}
    fp_pgcs_by_group = {}
    for object_id, label in zip(pgc, source_group):
        label = str(label)
        if label not in label_index:
            continue
        old = pgc_group.setdefault(int(object_id), label)
        if old != label:
            raise ValueError(f'PGC {object_id} maps to multiple source groups')
        fp_pgcs_by_group.setdefault(label, set()).add(int(object_id))

    group_members = {}
    excluded_edges = 0
    with CROSSMATCH.open(newline='', encoding='utf-8') as stream:
        for edge in csv.DictReader(stream):
            if edge['match_class'] != 'secure_joint_mark' or not edge['twompp_recno']:
                continue
            label = pgc_group.get(int(edge['PGC']))
            recno = int(edge['twompp_recno'])
            if label is None or recno not in point_index:
                excluded_edges += 1
                continue
            group_members.setdefault(label, set()).add(recno)

    fp_row_count = np.bincount(row_group, minlength=len(labels))
    anchor_count = np.bincount(anchor_group, minlength=len(labels))
    all_degrees = np.asarray([len(group_members.get(str(label), ()))
                              for label in labels], dtype=np.int32)
    role_summary = {}
    for role, name in ((0, 'training'), (1, 'heldout'), (2, 'buffer')):
        degrees = all_degrees[fp_role == role]
        role_summary[name] = dict(groups=int(len(degrees)), zero_direct_points=int(np.sum(degrees == 0)),
            one_direct_point=int(np.sum(degrees == 1)),
            multiple_direct_points=int(np.sum(degrees >= 2)),
            multiple_with_one_FP_row_no_anchor=int(np.sum(
                (degrees >= 2) & (fp_row_count[fp_role == role] == 1)
                & (anchor_count[fp_role == role] == 0))))

    eligible = eligible_multilink_training_groups(
        labels, fp_role, fp_row_count, anchor_count, group_members,
        point_role_by_recno)
    selected = None
    if eligible:
        label = eligible[0]
        members = sorted(group_members[label])
        selected = dict(source_group_label=label,
            selection_rule='lexicographically first eligible v6 training group; no mark scores read',
            FP_rows=int(fp_row_count[label_index[label]]),
            anchor_rows=int(anchor_count[label_index[label]]),
            member_recno=members,
            member_training_role=[point_role_by_recno[recno] for recno in members],
            member_population=[int(population[point_index[recno]]) for recno in members],
            member_flat_cell=[int(flat_cell[point_index[recno]]) for recno in members],
            member_radius_cMpc_h=[float(radius[point_index[recno]]) for recno in members],
            distinct_member_count=len(members),
            distinct_population_voxel_keys=len({
                (int(population[point_index[recno]]), int(flat_cell[point_index[recno]]))
                for recno in members}),
            associated_PGC=sorted(fp_pgcs_by_group.get(label, ())))

    report = dict(status='V6_MULTIMEMBER_GRAPH_CENSUS_NOT_LIKELIHOOD',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=source_revision,
        split='r2_sky_closed_split_v6', source_graph_role_counts=role_summary,
        distinct_direct_secure_training_group_member_pairs=int(sum(
            len(group_members.get(str(label), ())) for i, label in enumerate(labels)
            if fp_role[i] == 0)),
        training_group_count=len(labels[fp_role == 0]),
        eligible_training_multilink_group_count=len(eligible),
        selected_training_control=selected, excluded_unjoined_edges=excluded_edges,
        heldout_mark_values_read=False, likelihood_scores_read=False,
        field_state_read=False, PM_evolutions=0, field_fit=False, sampler=False,
        posterior_promoted=False, R2_complete=False,
        MW_M31='roles remain ambiguous on the same NEW inferred field',
        M33='unresolved; later observables must constrain the same NEW field at <=0.3 cMpc/h',
        calibration_status='group inclusion, member-redshift covariance and CF4 mark covariance remain uncalibrated',
        input_sha256={str(path): sha256(path)
            for path in (SPLIT, POINTS, FP, GROUP, CROSSMATCH)},
        elapsed_seconds=time.monotonic()-started)
    temp = out/'result.json.tmp'
    temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temp.replace(out/'result.json')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
