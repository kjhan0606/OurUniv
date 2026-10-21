"""Write the score-blind v6 selected-mark holdout identity manifest.

Only IDs, split roles, source-membership labels and factor ownership are read.
Do not add marks, counts, point positions/radii, field states or fit outputs to
this preparatory step. Measurement extraction is reserved for valid post-fit
evaluation.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from cf4_r2_mark_holdout_manifest import build_role_aware_mark_holdout_manifest
from cf4_r2_v6_factor_ownership_ledger import EXPECTED_HASHES

BASE = Path('/gpfs/kjhan/CF4/z0_density')
SPLIT = BASE / 'r2_sky_closed_split_v6/split.npz'
SPLIT_RESULT = BASE / 'r2_sky_closed_split_v6/result.json'
FP = BASE / 'r2_source_observation_assembly_v1/observations.npz'
GROUP = BASE / 'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
M2PP = ROOT / 'data/2mpp_catalog.csv'
TEMPEL_TABLE = Path('/gpfs/kjhan/CF4/external/tempel2017_cds/table1_parent_positions_v2.tsv')
CROSSMATCH = ROOT / 'data/cf4_2mpp_crossmatch_v1.csv'
OUT = BASE / 'r2_mark_holdout_manifest_20261021_v1'
EXPECTED_PRELEDGER = 197


def metadata_digest(parts):
    """Hash only the explicitly consumed ID/role metadata, never source files."""
    digest = hashlib.sha256()
    for name, values in parts:
        digest.update(str(name).encode('utf-8') + b'\0')
        for value in values:
            if isinstance(value, (tuple, list)):
                payload = '\t'.join(map(str, value))
            else:
                payload = str(value)
            digest.update(payload.encode('utf-8') + b'\n')
    return digest.hexdigest()


def load_m2pp_gids_score_blind(path):
    """Read only recno/GID columns, not the catalogue's measured quantities."""
    mapping = {}
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if 'recno' not in header or 'GID' not in header:
            raise ValueError('2M++ source lacks recno/GID columns')
        recno_column, gid_column = header.index('recno'), header.index('GID')
        for row in reader:
            recno = int(row[recno_column])
            gid_text = row[gid_column].strip()
            gid = int(gid_text) if gid_text else -1
            if recno in mapping:
                raise ValueError(f'duplicate 2M++ recno {recno}')
            mapping[recno] = gid
    if not mapping:
        raise ValueError('2M++ identity map is empty')
    return mapping


def load_tempel_parent_ids_score_blind(path, n_parent):
    """Reconstruct the graph's object-to-parent index from ID columns only."""
    n_single = 0
    pairs = []
    for raw in path.open(encoding='utf-8'):
        if not raw.strip() or raw.startswith('#'):
            continue
        fields = raw.rstrip('\n').split('\t')
        if len(fields) < 3 or not fields[0].strip().isdigit():
            continue
        object_id, group_id = int(fields[1]), int(fields[2])
        if group_id > 88_662:
            raise ValueError('Tempel GroupID exceeds the frozen parent table')
        if group_id > 0:
            parent_index = group_id - 1
        else:
            parent_index = 88_662 + n_single
            n_single += 1
        pairs.append((object_id, parent_index))
    if len(pairs) != 584_449 or n_single != n_parent - 88_662:
        raise ValueError('Tempel parent identity rows differ from the frozen graph contract')
    if len({object_id for object_id, _ in pairs}) != len(pairs):
        raise ValueError('duplicate Tempel object identity')
    pairs.sort()
    return pairs


def main():
    if OUT.exists():
        raise FileExistsError(OUT)

    # Use the recorded, frozen source hashes as provenance. Do not recompute
    # whole-file digests here: the NPZ/catalogue containers also hold sealed
    # measurements/counts, while this step is intentionally score-blind.
    expected = {Path(path): digest for path, digest in EXPECTED_HASHES.items()}
    split_result = json.loads(SPLIT_RESULT.read_text(encoding='utf-8'))
    source_hashes = split_result.get('source_sha256', {})
    for path in (SPLIT, FP, GROUP, CROSSMATCH):
        if path not in expected:
            raise ValueError(f'frozen ownership ledger omits metadata source: {path}')
    for path in (M2PP, TEMPEL_TABLE):
        if str(path) not in source_hashes:
            raise ValueError(f'graph-closure result omits identity source: {path}')

    with np.load(SPLIT, allow_pickle=False) as z:
        group_labels = z['fp_source_group'].astype(str)
        fp_role = z['fp_role'].astype(np.int8)
        point_recno = z['point_recno'].astype(np.int64)
        point_role = z['point_role'].astype(np.int8)
        cf4_ids = z['cf4_pgc'].astype(np.int64)
        cf4_roles = z['cf4_role'].astype(np.int8)
        parent_kind = z['tempel_parent_kind'].astype(np.int8)
        parent_id = z['tempel_parent_id'].astype(np.int64)
        parent_roles = z['tempel_parent_role'].astype(np.int8)
        gid_values = z['m2pp_gid'].astype(np.int64)
        gid_roles = z['m2pp_gid_role'].astype(np.int8)
    if len(parent_kind) != len(parent_id) or len(parent_id) != len(parent_roles):
        raise ValueError('frozen Tempel parent role arrays disagree')
    if len(gid_values) != len(gid_roles):
        raise ValueError('frozen 2M++ GID role arrays disagree')
    parent_role_by_index = dict(enumerate(map(int, parent_roles)))
    if len(np.unique(cf4_ids)) != len(cf4_ids) or len(np.unique(gid_values)) != len(gid_values):
        raise ValueError('frozen CF4/GID role identity arrays contain duplicates')
    cf4_role_by_id = dict(zip(map(int, cf4_ids), map(int, cf4_roles)))
    gid_role_by_id = dict(zip(map(int, gid_values), map(int, gid_roles)))

    # These are source identities/membership labels, not scored FP measurements.
    with np.load(FP, allow_pickle=False) as z:
        fp_pgc = z['PGC'].astype(np.int64)
        fp_source_group = z['source_group'].astype(str)
        membership_state = z['membership_state'].astype(str)
        source_obj_id = z['source_obj_id'].astype(np.int64)
        cf4_group = z['CF4_group'].astype(np.int64)
    with np.load(GROUP, allow_pickle=False) as z:
        row_group = z['row_group'].astype(np.int64)
        anchor_group = z['anchor_group'].astype(np.int64)

    with CROSSMATCH.open(newline='', encoding='utf-8') as stream:
        crossmatch_rows = [dict(PGC=row['PGC'],
            twompp_recno=row['twompp_recno'], match_class=row['match_class'])
            for row in csv.DictReader(stream)]

    # Reproduce the graph's exact identity indexes without parsing any
    # Tempel/2M++ position, distance, velocity or magnitude columns.
    tempel_pairs = load_tempel_parent_ids_score_blind(TEMPEL_TABLE, len(parent_id))
    parent_index_by_objid = dict(tempel_pairs)
    m2pp_gid_by_recno = load_m2pp_gids_score_blind(M2PP)

    manifest, rejected = build_role_aware_mark_holdout_manifest(
        fp_pgc=fp_pgc,
        fp_source_group=fp_source_group,
        membership_state=membership_state,
        source_obj_id=source_obj_id,
        cf4_group=cf4_group,
        group_labels=group_labels,
        fp_role=fp_role,
        row_group=row_group,
        anchor_group=anchor_group,
        point_recno=point_recno,
        point_role=point_role,
        crossmatch_rows=crossmatch_rows,
        parent_index_by_objid=parent_index_by_objid,
        parent_role_by_index=parent_role_by_index,
        m2pp_gid_by_recno=m2pp_gid_by_recno,
        m2pp_gid_role=gid_role_by_id,
        cf4_role_by_id=cf4_role_by_id,
        target_role=1)

    preledger_candidates = len(manifest) + sum(
        bool(row.get('preledger_candidate')) for row in rejected)
    if preledger_candidates != EXPECTED_PRELEDGER:
        raise ValueError(
            f'role-1 single-mark metadata cohort changed: '
            f'{preledger_candidates} != {EXPECTED_PRELEDGER}')
    rejection_counts = {}
    for row in rejected:
        for reason in row['reasons']:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    metadata_hashes = {
        'v6_roles': metadata_digest([
            ('fp_source_group', group_labels), ('fp_role', fp_role),
            ('point_recno', point_recno), ('point_role', point_role),
            ('cf4_pgc', sorted(cf4_role_by_id.items())),
            ('tempel_parent_kind_id_role', zip(parent_kind, parent_id, parent_roles)),
            ('m2pp_gid_role', sorted(gid_role_by_id.items()))]),
        'FP_identity_and_membership': metadata_digest([
            ('PGC', fp_pgc), ('source_group', fp_source_group),
            ('membership_state', membership_state), ('source_obj_id', source_obj_id),
            ('CF4_group', cf4_group)]),
        'group_ownership': metadata_digest([
            ('row_group', row_group), ('anchor_group', anchor_group)]),
        'crossmatch_identity_edges': metadata_digest([
            ('edge', (row['PGC'], row['twompp_recno'], row['match_class']))
            for row in crossmatch_rows]),
        'Tempel_object_parent_ids': metadata_digest([('pair', pair) for pair in tempel_pairs]),
        '2Mpp_recno_GID': metadata_digest([('pair', pair)
            for pair in sorted(m2pp_gid_by_recno.items())]),
    }
    report = dict(
        status='V6_ROLE1_MARK_HOLDOUT_IDENTITY_MANIFEST_NOT_MEASUREMENTS',
        split='r2_sky_closed_split_v6',
        preledger_single_mark_candidates=preledger_candidates,
        certified_role_aware_candidates=len(manifest),
        rejected_candidates=len(rejected),
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
        heldout_measurements_read=False,
        heldout_counts_or_keys_read=False,
        point_positions_or_radii_read=False,
        field_states_or_fit_outputs_read=False,
        catalog_measurement_columns_accessed=False,
        raw_marks_or_training_mixture_weights_materialized=False,
        driver_supplied_prior_candidate_count=EXPECTED_PRELEDGER,
        recorded_frozen_source_sha256={
            str(path): expected[path] for path in (SPLIT, FP, GROUP, CROSSMATCH)},
        graph_closure_identity_source_sha256={
            str(path): source_hashes[str(path)] for path in (M2PP, TEMPEL_TABLE)},
        used_metadata_sha256=metadata_hashes,
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(
            (ROOT / 'src/cf4_r2_mark_holdout_manifest.py').read_bytes()).hexdigest(),
        whole_file_hashes_recomputed=False,
        split_result_path=str(SPLIT_RESULT),
        operator_contract=(
            'future post-fit: shared population draw; optical covariance and '
            'joint cut normalization; within-bin K density; secure linked '
            '2M++ point radius for source conditioning; separate CF4 row '
            'redshift reference; regenerate source support/weights per draw'),
        training_only_preprocessing=(
            'population fit, richness centering, nuisance scaling and all '
            'preprocessing must be fitted on v6 training data only'),
        limits=[
            'identity/graph closure is leakage control, not physical membership or selection calibration',
            'selected-mark prediction cannot identify missing eligibility, targeting, redshift-success, FP-quality or group-inclusion probabilities',
            'candidate conditional mark score is not a full-survey calibrated likelihood',
            'MW/M31 remain role-ambiguous; M33 unresolved; all observables must constrain those same roles on the same NEW evolved LG field at <=0.3 cMpc/h',
            'native truth IDs are calibration/evaluation only and never candidate selectors',
            'R2 remains NO-GO; this manifest is neither a posterior nor an exit audit'],
    )
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUT / 'rejected.json').write_text(
        json.dumps(rejected, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUT / 'result.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
