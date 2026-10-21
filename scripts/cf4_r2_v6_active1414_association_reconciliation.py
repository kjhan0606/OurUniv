"""Reconcile the active conditional-FP rows against the v6 ownership ledger.

Only training identifiers and structural selection fields are used. No held-
out values, raw marks, field state, likelihood or truth IDs are read. This
certifies cohort overlap; it does not recalibrate associations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import subprocess
import time

import numpy as np

from cf4_r2_linked_fp_sparse_train import (
    FP, load_train_singletons, select_training_single_mark_links,
)
from cf4_r2_v6_factor_ownership_ledger import EXPECTED_HASHES, sha256


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/gpfs/kjhan/CF4/z0_density')
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
MIXTURE = BASE/'r2_raw_source_mixture_v1/raw_source_mixture.npz'
LEDGER = BASE/'r2_v6_factor_ownership_ledger_20261004_v2'
OUT = BASE/'r2_v6_active1414_association_reconciliation_20261004_v1'
EXPECTED_ACTIVE_ROWS = 1414


def reconcile_selected_rows(chosen, fp_pgcs, active_pgcs, group_rows, edge_rows):
    """Return row-level association status after exact active-cohort matching."""
    if len(chosen) != len(active_pgcs) or len(chosen) != len(fp_pgcs):
        raise ValueError('selected rows, FP rows and active mixture IDs differ in length')
    groups = {str(row['source_group_label']): row for row in group_rows}
    if len(groups) != len(group_rows):
        raise ValueError('ownership ledger repeats a source-group label')
    edge_by_pgc = {}
    for edge in edge_rows:
        edge_by_pgc.setdefault(int(edge['PGC']), []).append(edge)

    rows = []
    seen_groups, seen_pgcs = set(), set()
    for option, fp_pgc, active_pgc in zip(chosen, fp_pgcs, active_pgcs):
        label, _group_index, point_index, row_index, _train_index = option
        label, fp_pgc, active_pgc = str(label), int(fp_pgc), int(active_pgc)
        if fp_pgc != active_pgc:
            raise ValueError(f'active mixture order/identity mismatch at {label}')
        if label in seen_groups or fp_pgc in seen_pgcs:
            raise ValueError('active conditional cohort repeats a group or FP row')
        seen_groups.add(label)
        seen_pgcs.add(fp_pgc)
        ledger = groups.get(label)
        if ledger is None:
            raise ValueError(f'active group absent from ownership ledger: {label}')
        if int(ledger['FP_row_count']) != 1:
            raise ValueError(f'active group is not a one-row mark factor: {label}')
        if int(ledger['secure_training_recno_count']) != 1:
            raise ValueError(f'active group is not linked to one training point: {label}')
        if int(ledger['anchor_row_count']) != 0:
            raise ValueError(f'active group unexpectedly contains an anchor: {label}')
        matching_edges = [edge for edge in edge_by_pgc.get(fp_pgc, ())
                          if label in str(edge['candidate_group_labels']).split(';')]
        ambiguous_count = int(ledger['ambiguous_edge_count'])
        collision = str(ledger['association_collision']).lower() == 'true'
        rows.append(dict(
            source_group_label=label,
            fp_pgc=fp_pgc,
            point_index=int(point_index),
            fp_row_index=int(row_index),
            ledger_category=str(ledger['category']),
            ambiguous_edge_count=ambiguous_count,
            association_collision=collision,
            associated_edge_statuses=';'.join(sorted({str(e['status']) for e in matching_edges})),
            unresolved_for_conditional_use=bool(
                ledger['category'] != 'one_linked_count_point' or ambiguous_count or collision),
        ))
    if seen_pgcs != set(map(int, active_pgcs)):
        raise ValueError('active conditional FP IDs are not unique and exactly represented')
    return rows


def clean_conditional_row_indices(rows):
    """Keep only one-link groups with no unresolved association edge."""
    return [index for index, row in enumerate(rows)
            if row['ledger_category'] == 'one_linked_count_point'
            and not row['unresolved_for_conditional_use']]


def read_verified_group_ledger():
    with (LEDGER/'result.json').open(encoding='utf-8') as stream:
        result = json.load(stream)
    if (result.get('status') != 'V6_TRAINING_FACTOR_OWNERSHIP_LEDGER_NOT_JOINT_LIKELIHOOD'
            or result.get('training_count_total') != 47121):
        raise ValueError('ownership ledger is not the verified active v6 result')
    expected_sources = {str(path): digest for path, digest in EXPECTED_HASHES.items()}
    if result.get('source_sha256') != expected_sources:
        raise ValueError('ownership ledger does not certify the frozen v6 source hashes')
    files = {name: LEDGER/name for name in ('group_factors.csv', 'crossmatch_edges.csv')}
    for name, path in files.items():
        if sha256(path) != result['outputs'][name]:
            raise ValueError(f'ownership ledger output hash mismatch: {name}')
    with files['group_factors.csv'].open(newline='', encoding='utf-8') as stream:
        groups = list(csv.DictReader(stream))
    with files['crossmatch_edges.csv'].open(newline='', encoding='utf-8') as stream:
        edges = list(csv.DictReader(stream))
    return groups, edges, result


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
    if OUT != BASE/'r2_v6_active1414_association_reconciliation_20261004_v1' or OUT.exists():
        raise FileExistsError('unexpected or previously used output path')
    started = time.monotonic()

    source_hashes = {path: sha256(path) for path in EXPECTED_HASHES}
    if source_hashes != EXPECTED_HASHES:
        raise ValueError('frozen v6 source hashes changed since ownership ledger')
    options, point, _ = load_train_singletons(SPLIT, include_fp_parameters=False)
    with np.load(FP, allow_pickle=False) as fp:
        fp_pgcs = fp['PGC'].astype(np.int64, copy=True)
        membership = fp['membership_state'].astype(str)
    with np.load(MIXTURE, allow_pickle=False) as mixture:
        active_pgcs = mixture['PGC'].astype(np.int64, copy=True)
        active_population = mixture['population'].astype(np.int64, copy=True)
    selected = select_training_single_mark_links(
        options, membership, include_grouped=True)
    chosen = [option for pop in range(6) for option in selected
              if int(point['population'][option[2]]) == pop]
    if len(chosen) != EXPECTED_ACTIVE_ROWS or len(active_pgcs) != EXPECTED_ACTIVE_ROWS:
        raise ValueError('active inclusive-v6 conditional cohort is not the frozen 1,414 rows')
    expected_population = np.asarray([point['population'][option[2]] for option in chosen])
    if not np.array_equal(active_population, expected_population):
        raise ValueError('active mixture population IDs do not align with selected training links')
    selected_pgcs = np.asarray([fp_pgcs[option[3]] for option in chosen], dtype=np.int64)

    group_rows, edge_rows, ledger_result = read_verified_group_ledger()
    ledger_files = {name: LEDGER/name for name in ('group_factors.csv', 'crossmatch_edges.csv')}
    rows = reconcile_selected_rows(chosen, selected_pgcs, active_pgcs, group_rows, edge_rows)
    category_counts = Counter(row['ledger_category'] for row in rows)
    edge_status_counts = Counter(status for row in rows
                                 for status in row['associated_edge_statuses'].split(';') if status)
    unresolved = [row for row in rows if row['unresolved_for_conditional_use']]

    OUT.mkdir(parents=True, exist_ok=False)
    columns = list(rows[0])
    with (OUT/'selected_association_rows.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        status='V6_ACTIVE_CONDITIONAL_COHORT_ASSOCIATION_RECONCILED_NOT_LIKELIHOOD',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=actual,
        active_split='r2_sky_closed_split_v6', active_conditional_rows=len(rows),
        exact_ordered_PGC_match_to_current_raw_mixture=True,
        membership_categories=['source_ungrouped_catalogue_present',
                               'source_ungrouped_catalogue_absent',
                               'source_grouped_catalogue_present'],
        ledger_group_classes_in_active_cohort=dict(sorted(category_counts.items())),
        associated_edge_statuses=dict(sorted(edge_status_counts.items())),
        unresolved_association_rows=len(unresolved),
        unresolved_association_PGCs=[row['fp_pgc'] for row in unresolved],
        clean_conditional_rows=len(clean_conditional_row_indices(rows)),
        source_conditioning_radius_wiring=(
            'linked 2M++ point radius carried into active target; fixed-state profile pending'),
        heldout_values_read=False, raw_mark_values_read=False,
        field_or_likelihood_read=False, PM_evolutions=0, posterior_or_fit=False,
        R2_complete=False,
        MW_M31='roles remain ambiguous; observables must constrain the same NEW evolved field',
        M33='unresolved; same NEW field at <=0.3 cMpc/h required',
        frozen_source_sha256={str(path): digest for path, digest in source_hashes.items()},
        ledger_csv_sha256={name: sha256(path) for name, path in ledger_files.items()},
        selected_association_rows_sha256=hashlib.sha256(
            (OUT/'selected_association_rows.csv').read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started)
    (OUT/'result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n',
                                   encoding='utf-8')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
