"""Inspect ONE author release: archive/schema/identities only, never FP values.

Acceptance bits are derived outcomes. Optional inspection is restricted to
unambiguously training-linked, graph-isolated groups; all others stay sealed.
This is not a source selection fit or an amendment to the frozen v6 sample.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import time
import zipfile

import numpy as np

from cf4_r2_sky_closed_split import close_group_marks


ARCHIVE = Path('/gpfs/kjhan/CF4/external/6dfgs_2014_source_probe_20261107/'
               'stu1198_Supplementary_Data.zip')
EXPECTED_SHA = '474c82f5f1338d1ed457ee7e86806388960e6b0b7c79f5cbebfb1343cb074283'
SOURCE_ID = re.compile(r'(?<!\w)g\d{7}[+-]\d{6}(?!\w)')
XSC_ID = re.compile(r'(?<!\w)2MASX\s*J\d{8}[+-]\d{7}(?!\w)')
ROOT = Path(__file__).resolve().parents[1]
SPLIT = Path('/gpfs/kjhan/CF4/z0_density/r2_sky_closed_split_v6/split.npz')


def training_status_probe(records):
    """Diagnostic subset ONLY; never define the eventual native science cohort.

    Exact XSC identity -> existing count point -> existing CF4 crossmatch.
    Unknown, duplicate, conflicting or non-training links seal the ENTIRE
    native group, propagated over all associated CF4 groups. No nearest-match
    tuning, sky-cut changes, or new FP/morphology criterion chooses this set.
    """
    inputs = {SPLIT: '9528b54a1e9dc876052771c21c0330ebba5f9b2b56966ab3cbe593a71cfc2b74',
        ROOT/'data/2mpp_catalog.csv': '05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d',
        ROOT/'data/cf4_2mpp_crossmatch_v1.csv': '64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf'}
    for path, expected in inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'identity/split input changed: {path.name}')
    with np.load(SPLIT, allow_pickle=False) as f:
        points = dict(zip(map(int, f['point_recno']), map(int, f['point_role'])))
        cf4 = dict(zip(map(int, f['cf4_pgc']), map(int, f['cf4_role'])))
    names, links = {}, {}
    with (ROOT/'data/2mpp_catalog.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            # CSV Name is the XSC HHMMSSss+DDMMSSs suffix (no rounding).
            names.setdefault(row['Name'].strip(), []).append(int(row['recno']))
    with (ROOT/'data/cf4_2mpp_crossmatch_v1.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['twompp_recno'] and row['1PGC']:
                links.setdefault(int(row['twompp_recno']), []).append(
                    (int(row['PGC']), int(row['1PGC']), row['match_class']))
    group_labels, row_safe, row_links, status = [], [], [], Counter()
    for source, xsc, line in records:
        fields = line.split()
        if len(fields) != 31 or fields[:2] != [source, xsc]:
            raise ValueError('pinned author table column/identity layout changed')
        group = int(fields[20])  # Catalogue identity only; not a physical distance.
        label = f'6DF:G{group}' if group > 0 else f'6DF:S{source}'
        recnos = names.get(xsc.removeprefix('2MASXJ'), [])
        edges = [edge for recno in recnos for edge in links.get(recno, [])]
        secure = (len(recnos) == 1 and len(edges) == 1
                  and edges[0][2] == 'secure_joint_mark')
        safe = (secure and points.get(recnos[0], -1) == 0
                and cf4.get(edges[0][1], -1) == 0)
        status['direct_training' if safe else 'sealed_unknown_conflict_or_nontraining'] += 1
        group_labels.append(label)
        row_safe.append(safe)
        row_links.append(edges)
    unsafe_groups = {g for g, safe in zip(group_labels, row_safe) if not safe}
    edge_cf4, edge_source = [], []
    for group, edges in zip(group_labels, row_links):
        for _, cfgroup, _ in edges:
            edge_cf4.append(cfgroup)
            edge_source.append(group)
    edge_source = np.asarray(edge_source, dtype=str)
    _, closed = close_group_marks(np.zeros(0, dtype=bool), np.array([], dtype=int),
        np.array([], dtype=int), [g for g, role in cf4.items() if role != 0],
        np.asarray(edge_cf4, dtype=int), edge_source,
        np.array([g in unsafe_groups for g in edge_source]))
    sealed_groups = unsafe_groups | closed
    selected = [i for i, (group, safe) in enumerate(zip(group_labels, row_safe))
                if safe and group not in sealed_groups]
    bits, pv = [], []
    for i in selected:
        fields = records[i][2].split()
        code, inclusion = fields[25], fields[26]
        if not re.fullmatch(r'[01]{6}', code) or inclusion not in ('0', '1'):
            raise ValueError('unexpected native J selection codes; do not repair')
        bits.append(code)
        pv.append(int(inclusion))
    return dict(scope='CONSERVATIVE_TRAINING_METADATA_PROBE_NOT_SCIENCE_COHORT',
        identity_inputs={str(p): sha for p, sha in inputs.items()},
        direct_status_counts=dict(status), inspected_rows=len(selected),
        inspected_native_groups=len({group_labels[i] for i in selected}),
        sealed_rows=len(records)-len(selected),
        training_J_sample_code_counts=dict(Counter(bits)),
        training_J_PV_included=int(sum(pv)), training_J_PV_excluded=len(pv)-sum(pv),
        training_J_residual_cut_failed=sum(code[5] == '0' for code in bits),
        training_J_residual_only_failed=sum(code == '111110' for code in bits),
        all_native_identity_roles_resolved=False, native_science_sample_selected=False,
        numerical_FP_values_parsed=False, heldout_acceptance_bits_parsed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-status-probe', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    started = time.monotonic()
    content = ARCHIVE.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    if sha != EXPECTED_SHA or len(content) > 10_000_000:
        raise ValueError('author archive differs from pinned bounded download')
    with zipfile.ZipFile(io.BytesIO(content)) as outer:
        members = [dict(name=x.filename, bytes=x.file_size, crc=x.CRC) for x in outer.infolist()]
        raw = outer.read('campbell_table8.zip')
    with zipfile.ZipFile(io.BytesIO(raw)) as inner:
        # Inspect in memory; never extract arbitrary archive paths or executables.
        entries = [x for x in inner.infolist() if not x.is_dir()
                   and not x.filename.startswith('__MACOSX/')
                   and not Path(x.filename).name.startswith('.')]
        if len(entries) != 1 or entries[0].file_size > 10_000_000:
            raise ValueError(f'unexpected nested table entries: {[x.filename for x in entries]}')
        table_name = entries[0].filename
        table = inner.read(table_name)
    identities, records, header, columns = [], [], [], Counter()
    for line in table.decode('utf-8').splitlines():
        if not line.strip():
            continue
        source, xsc = SOURCE_ID.search(line), XSC_ID.search(line)
        if source is None and xsc is None and not identities:
            header.append(line)  # before the first identified record only
            continue
        if source is None or xsc is None:
            raise ValueError('non-header line missing a source identity; no science fallback')
        identities.append((source.group(), re.sub(r'\s+', '', xsc.group())))
        records.append((*identities[-1], line))
        columns[len(line.split())] += 1
    if len(identities) != 11102 or any(len({r[k] for r in identities}) != 11102 for k in (0, 1)):
        raise ValueError('expected 11102 distinct raw-table identities; review version/schema')
    probe = training_status_probe(records) if args.training_status_probe else None
    # Numerical FP/velocity/distance values remain unparsed. The optional
    # probe opens only native group identities and isolated training flags.
    args.output_dir.mkdir(parents=False, exist_ok=False)
    with (args.output_dir/'identities.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('sixdf_id', 'two_mass_id'))
        writer.writerows(identities)
    result = dict(status='AUTHOR_PRE_FINAL_CUT_TABLE_IDENTITIES_AVAILABLE_NOT_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        archive=str(ARCHIVE), archive_sha256=sha, archive_bytes=len(content),
        official_landing='https://academic.oup.com/mnras/article/443/2/1231/1060810',
        outer_members=members, nested_table=table_name, table_bytes=len(table),
        table_sha256=hashlib.sha256(table).hexdigest(), header=header,
        rows=len(identities), distinct_sixdf_ids=11102, distinct_two_mass_ids=11102,
        whitespace_field_counts=dict(columns), training_status_probe=probe,
        acceptance_bits_parsed_for_training_only=bool(probe and probe['inspected_rows']),
        published_reject_retention_is_row_verified=bool(probe and probe['training_J_residual_cut_failed']),
        FP_or_distance_values_parsed=False, heldout_scored=False,
        frozen_split_changed=False, source_selection_calibrated=False,
        source_identity_split_linkage_complete=False, live_target_changed=False,
        R2_complete=False, elapsed_seconds=time.monotonic()-started,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (args.output_dir/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
