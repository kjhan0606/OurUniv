"""Freeze an N128 sky count holdout and graph-closed CF4/FP mark roles.

No likelihood, field fit, or simulation is performed. The sky partition is
chosen from survey geometry alone before inspecting count residuals. Run
only inside Slurm.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cf4_r2_sky_closed_split import close_group_marks, octant_from_xyz

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE / 'r2_sky_closed_split_v5'
CHOICE = BASE / 'r2_sky_footprint_choice_v1/result.json'
ANCHORS = BASE / 'r2_cross_method_anchors_v1/anchors.npz'
POINTS = BASE / 'r2_point_mark_manifest_v1/points.npz'
EDGES = BASE / 'r2_point_mark_manifest_v1/edges.npz'
NATIVE = BASE / 'bundle_c_v1/native_data_v2/CF4_native.npz'
FP = BASE / 'r2_source_observation_assembly_v1/observations.npz'
PARENT = BASE / 'r2_inclusive_count_diagnostic_v1/inclusive_counts_3_sparse.npz'
CROSSMATCH = ROOT / 'data/cf4_2mpp_crossmatch_v1.csv'
N = 128


def sparse_counts(keys):
    return np.unique(keys, return_counts=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    choice = json.loads(CHOICE.read_text())
    if (choice['classification'] != 'R2_GEOMETRY_ONLY_HOLDOUT_CHOICE'
            or not choice['no_observation_values_or_model_scores_read']
            or choice['chosen_octant'] not in choice['eligible_octants']):
        raise ValueError('sky choice was not frozen by geometry-only rule')
    held_octant = int(choice['chosen_octant'])
    with np.load(POINTS, allow_pickle=False) as f:
        recno = f['recno'].astype(np.int64)
        pop = f['population'].astype(np.int64)
        flat = f['flat_cell'].astype(np.int64)
    with np.load(EDGES, allow_pickle=False) as f:
        edge_point = f['point_index'].astype(np.int64)
        edge_cf4 = f['group_1pgc'].astype(np.int64)
    with np.load(NATIVE, allow_pickle=False) as f:
        native_id = f['CF4_pgc'].astype(np.int64)
        native_pos = f['CF4_pos'].astype(np.float64)
    with np.load(FP, allow_pickle=False) as f:
        fp_cf4 = f['CF4_group'].astype(np.int64)
        fp_source = f['source_group'].copy()
        fp_pgc = f['PGC'].astype(np.int64)
        fp_directions = f['directions'].astype(np.float64)
    with np.load(ANCHORS, allow_pickle=False) as f:
        anchor_cf4 = f['CF4_group'].astype(np.int64)
        anchor_source = f['source_group'].copy()
    if (len(np.unique(recno)) != len(recno) or len(np.unique(native_id)) != len(native_id)
            or np.any(flat < 0) or np.any(flat >= N**3)
            or np.any(pop < 0) or np.any(pop >= 6)
            or native_pos.shape != (len(native_id), 3)
            or fp_directions.shape != (len(fp_cf4), 3)
            or len(anchor_cf4) != len(anchor_source)
            or len(np.unique(fp_pgc)) != len(fp_pgc)
            or np.any(~np.isin(anchor_source, np.unique(fp_source)))
            or not np.isfinite(native_pos).all()
            or not np.isfinite(fp_directions).all()):
        raise ValueError('source geometry or identities changed')
    # N128 has a cell boundary exactly through the observer at (192,192,192).
    # Thus the integer voxel octant is also the complete observed-count mask.
    all_flat = np.arange(N**3, dtype=np.int32)
    ijk = np.stack(np.unravel_index(all_flat, (N,)*3), axis=1)
    voxel_held = octant_from_xyz(ijk-64) == held_octant
    point_held = voxel_held[flat]
    native_oct = octant_from_xyz(native_pos-192.)
    fp_row_held = octant_from_xyz(fp_directions) == held_octant
    source_pgc = {int(p): group for p, group in zip(fp_pgc, fp_source)}
    point_recno = {int(rec): index for index, rec in enumerate(recno)}
    point_fp_edges = set()
    with CROSSMATCH.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['match_class'] != 'secure_joint_mark' or not row['twompp_recno']:
                continue
            index = point_recno.get(int(row['twompp_recno']))
            group = source_pgc.get(int(row['PGC']))
            if index is not None and group is not None:
                point_fp_edges.add((index, group))
    direct_point = np.array([pair[0] for pair in sorted(point_fp_edges)], dtype=np.int64)
    direct_source = np.array([pair[1] for pair in sorted(point_fp_edges)],
                             dtype=fp_source.dtype)
    direct_held_fp = np.unique(direct_source[point_held[direct_point]])
    direct_train_fp = np.unique(direct_source[~point_held[direct_point]])
    graph_cf4 = np.r_[fp_cf4, anchor_cf4]
    graph_source = np.r_[fp_source, anchor_source]
    graph_sky_seed = np.r_[fp_row_held | np.isin(fp_source, direct_held_fp),
                           np.zeros(len(anchor_cf4), dtype=bool)]
    held_cf4, held_fp = close_group_marks(
        point_held, edge_point, edge_cf4,
        native_id[native_oct == held_octant], graph_cf4,
        graph_source, graph_sky_seed)
    native_closed = np.isin(native_id, list(held_cf4))
    fp_labels, fp_row_index = np.unique(fp_source, return_inverse=True)
    fp_seed_count = np.bincount(fp_row_index, weights=fp_row_held.astype(int),
                                minlength=len(fp_labels))
    fp_rows = np.bincount(fp_row_index, minlength=len(fp_labels))
    fp_closed = np.isin(fp_labels, list(held_fp))
    if np.any((native_oct == held_octant) & ~native_closed):
        raise AssertionError('sky CF4 group escaped closure')
    if np.any(fp_row_held & ~fp_closed[fp_row_index]):
        raise AssertionError('sky FP group escaped closure')
    if np.any(point_held[edge_point] & ~np.isin(edge_cf4, list(held_cf4))):
        raise AssertionError('heldout point leaks into training CF4 group')
    if np.any(np.isin(graph_cf4, list(held_cf4))
              & ~np.isin(graph_source, list(held_fp))):
        raise AssertionError('heldout CF4/anchor group leaks into training FP group')
    if np.any(~np.isin(direct_held_fp, list(held_fp))):
        raise AssertionError('heldout direct FP member redshift leaks to training')

    # A group with a member or linked FP source across the sky boundary is
    # withheld from training, but it is a boundary buffer, not validation.
    # Codes: 0 training, 1 internally sky-held validation, 2 buffer.
    cf4_with_train_point = np.unique(edge_cf4[~point_held[edge_point]])
    complete_sky_fp = fp_labels[fp_seed_count == fp_rows]
    cf4_with_outside_fp = np.unique(graph_cf4[
        ~np.isin(graph_source, complete_sky_fp)])
    cf4_with_fp_train_member = np.unique(graph_cf4[
        np.isin(graph_source, direct_train_fp)])
    native_validate = ((native_oct == held_octant)
                       & ~np.isin(native_id, cf4_with_train_point)
                       & ~np.isin(native_id, cf4_with_outside_fp)
                       & ~np.isin(native_id, cf4_with_fp_train_member))
    fp_cross_cf4 = np.union1d(cf4_with_train_point,
                              native_id[native_oct != held_octant])
    fp_with_cross_cf4 = np.unique(graph_source[
        np.isin(graph_cf4, fp_cross_cf4)])
    fp_validate = ((fp_seed_count == fp_rows)
                   & ~np.isin(fp_labels, fp_with_cross_cf4)
                   & ~np.isin(fp_labels, direct_train_fp))
    native_role = np.where(~native_closed, 0,
                           np.where(native_validate, 1, 2)).astype(np.int8)
    fp_role = np.where(~fp_closed, 0,
                       np.where(fp_validate, 1, 2)).astype(np.int8)
    key = pop * N**3 + flat
    train_key, train_count = sparse_counts(key[~point_held])
    held_key, held_count = sparse_counts(key[point_held])
    all_key, all_count = sparse_counts(key)
    with np.load(PARENT, allow_pickle=False) as f:
        np.testing.assert_array_equal(all_key, f['parent_keys'])
        np.testing.assert_array_equal(all_count, f['parent_counts'])
    if np.intersect1d(train_key, held_key).size:
        raise AssertionError('count key straddles sky split')
    np.testing.assert_array_equal(np.sort(np.r_[train_key, held_key]), all_key)
    if (not np.all(~voxel_held[train_key % N**3])
            or not np.all(voxel_held[held_key % N**3])):
        raise AssertionError('count sky mask disagrees with point partition')
    with np.load(BASE/'r2_sky_closed_split_v4/split.npz', allow_pickle=False) as old:
        np.testing.assert_array_equal(old['cf4_pgc'], native_id)
        np.testing.assert_array_equal(old['fp_source_group'], fp_labels)
        changed_cf4 = int(np.count_nonzero(old['cf4_role'] != native_role))
        changed_fp = int(np.count_nonzero(old['fp_role'] != fp_role))
    report = dict(
        classification='R2_SOURCE_BOUND_SKY_CLOSED_SPLIT_NOT_LIKELIHOOD',
        job_id=os.environ['SLURM_JOB_ID'], heldout_octant=held_octant,
        reason='geometry-only footprint choice before R2 field fitting',
        choice_report=str(CHOICE),
        geometry='N128 observed voxels; observer at grid boundary (192,192,192) cMpc/h',
        point_train=int((~point_held).sum()), point_heldout=int(point_held.sum()),
        heldout_voxels=int(voxel_held.sum()),
        native_cf4_roles=np.bincount(native_role, minlength=3).tolist(),
        fp_source_roles=np.bincount(fp_role, minlength=3).tolist(),
        cf4_graph_held_ids=len(held_cf4), fp_graph_held_labels=len(held_fp),
        heldout_point_edges=int(point_held[edge_point].sum()),
        direct_fp_member_edges=int(len(direct_point)),
        heldout_direct_fp_member_edges=int(point_held[direct_point].sum()),
        cross_method_anchor_edges=int(len(anchor_cf4)),
        train_cf4_edges_to_heldout_points=0,
        train_fp_or_anchor_rows_to_heldout_cf4_groups=0,
        train_fp_groups_to_heldout_direct_members=0,
        changed_CF4_roles_from_v4=changed_cf4,
        changed_FP_roles_from_v4=changed_fp,
        six_population_parent_projection_exact=True,
        old_native_and_row_hash_splits_used=False,
        interpretation='The count holdout is a sky-volume observation window. '
                       'Training CF4/FP marks are closed against heldout 2M++ '
                       'members; boundary-crossing marks are a buffer. Even '
                       'internally held marks are not an independent survey '
                       'or a calibrated posterior test.',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (POINTS, EDGES, NATIVE, FP, ANCHORS, PARENT,
                                 CROSSMATCH, CHOICE)},
        mw_m31_m33='No component identities seeded; unresolved M33 remains latent. '
                   'Their observables must constrain the same NEW evolved field.',
        rate_bias_group_selection_and_sampler_calibrated=False,
        R2_posterior=False)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'split.npz', point_recno=recno,
                        point_heldout=point_held,
                        heldout_flat_voxels=all_flat[voxel_held],
                        train_keys=train_key.astype(np.int32),
                        train_counts=train_count.astype(np.int32),
                        heldout_keys=held_key.astype(np.int32),
                        heldout_counts=held_count.astype(np.int32),
                        cf4_pgc=native_id, cf4_role=native_role,
                        fp_source_group=fp_labels, fp_role=fp_role)
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
