"""Build a source-identity-graph-closed R2 holdout from frozen v5 roles.

The closure is for leakage control only. Tempel--2M++ links are observed
catalogue associations, not calibrated physical membership probabilities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_r2_graph_closure import close_graph_roles  # noqa: E402

BASE = Path("/gpfs/kjhan/CF4/z0_density")
OUT = BASE / "r2_sky_closed_split_v6"
V5 = BASE / "r2_sky_closed_split_v5/split.npz"
POINTS = BASE / "r2_point_mark_manifest_v1/points.npz"
EDGES = BASE / "r2_point_mark_manifest_v1/edges.npz"
FP = BASE / "r2_source_observation_assembly_v1/observations.npz"
ANCHORS = BASE / "r2_cross_method_anchors_v1/anchors.npz"
PARENTS = BASE / "r2_tempel_parent_denominator_v1/parents.npz"
LINKS = BASE / "r2_tempel_parent_denominator_v1/independent_m2pp_links.npz"
TEMPEL_TABLE = Path("/gpfs/kjhan/CF4/external/tempel2017_cds/table1_parent_positions_v2.tsv")
M2PP = ROOT / "data/2mpp_catalog.csv"
CROSSMATCH = ROOT / "data/cf4_2mpp_crossmatch_v1.csv"
N, NTEMPEL_GROUPS = 128, 88_662


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_m2pp_gids(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"recno", "GID"}.issubset(reader.fieldnames or []):
            raise ValueError("2M++ source lacks recno/GID columns")
        for row in reader:
            recno = int(row["recno"])
            gid_text = row["GID"].strip()
            gid = int(gid_text) if gid_text else -1
            if recno in result:
                raise ValueError(f"duplicate 2M++ recno {recno}")
            result[recno] = gid
    if not result:
        raise ValueError("2M++ source catalogue is empty")
    return result


def load_tempel_parent_index(path: Path, n_parent: int):
    objids: list[int] = []
    parent_indices: list[int] = []
    group_ids: list[int] = []
    n_single = 0
    for raw in path.open(encoding="utf-8"):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.rstrip("\n").split("\t")
        if len(fields) < 3 or not fields[0].strip().isdigit():
            continue
        objid = int(fields[1])
        group_id = int(fields[2])
        if group_id > NTEMPEL_GROUPS:
            raise ValueError("Tempel GroupID exceeds the frozen parent table")
        if group_id > 0:
            parent = group_id - 1
        else:
            parent = NTEMPEL_GROUPS + n_single
            n_single += 1
        objids.append(objid)
        parent_indices.append(parent)
        group_ids.append(group_id)
    objid_array = np.asarray(objids, dtype=np.uint64)
    parent_array = np.asarray(parent_indices, dtype=np.int64)
    group_array = np.asarray(group_ids, dtype=np.int32)
    if (len(objid_array) != 584_449 or n_single != n_parent-NTEMPEL_GROUPS
            or len(np.unique(objid_array)) != len(objid_array)):
        raise ValueError("Tempel source rows/unique object IDs differ from frozen contract")
    order = np.argsort(objid_array)
    return objid_array[order], parent_array[order], group_array[order]


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    expected_commit = os.environ.get("CF4_EXPECTED_COMMIT")
    if not expected_commit:
        raise RuntimeError("CF4_EXPECTED_COMMIT must pin the submitted source")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != expected_commit:
        raise RuntimeError(f"source commit mismatch: {commit} != {expected_commit}")
    tracked_sources = [ROOT / "scripts/cf4_r2_graph_closed_split.py",
                       ROOT / "scripts/run_cf4_r2_graph_closed_split.sbatch",
                       ROOT / "src/cf4_r2_graph_closure.py"]
    if subprocess.run(["git", "diff", "--quiet", expected_commit, "--",
                       *[str(p.relative_to(ROOT)) for p in tracked_sources]],
                      cwd=ROOT).returncode:
        raise RuntimeError("submitted graph-closure sources changed after commit")
    if OUT.exists():
        raise FileExistsError(OUT)

    with np.load(V5, allow_pickle=False) as z:
        point_recno = z["point_recno"].copy()
        base_heldout = z["point_heldout"].astype(bool)
        cf4_ids = z["cf4_pgc"].astype(np.int64)
        cf4_old_roles = z["cf4_role"].astype(np.int8)
        fp_labels = z["fp_source_group"].astype(str)
        fp_old_roles = z["fp_role"].astype(np.int8)
        old_keys = z["train_keys"].astype(np.int64)
    with np.load(POINTS, allow_pickle=False) as z:
        point_pop = z["population"].astype(np.int64)
        point_flat = z["flat_cell"].astype(np.int64)
    with np.load(EDGES, allow_pickle=False) as z:
        edge_point = z["point_index"].astype(np.int64)
        edge_cf4 = z["group_1pgc"].astype(np.int64)
    with np.load(FP, allow_pickle=False) as z:
        fp_pgc = z["PGC"].astype(np.int64)
        fp_source = z["source_group"].astype(str)
        fp_cf4 = z["CF4_group"].astype(np.int64)
        fp_objid_raw = z["source_obj_id"].astype(np.int64)
    with np.load(ANCHORS, allow_pickle=False) as z:
        anchor_cf4 = z["CF4_group"].astype(np.int64)
        anchor_source = z["source_group"].astype(str)
    with np.load(PARENTS, allow_pickle=False) as z:
        parent_kind = z["parent_kind"].astype(np.int8)
        parent_id = z["parent_id"].astype(np.int64)
        old_parent_role = z["role"].astype(np.int8)
    with np.load(LINKS, allow_pickle=False) as z:
        linked_recno = z["m2pp_recno"].astype(np.int64)
        linked_gid = z["m2pp_gid"].astype(np.int64)
        linked_parent = z["parent_index"].astype(np.int64)

    if (len(point_recno) != 57_238 or len(np.unique(point_recno)) != len(point_recno)
            or len(point_pop) != len(point_recno) or len(point_flat) != len(point_recno)
            or len(base_heldout) != len(point_recno)
            or np.any(~np.isin(cf4_old_roles, (0, 1, 2)))
            or np.any(~np.isin(fp_old_roles, (0, 1, 2)))
            or len(np.unique(fp_pgc)) != len(fp_pgc)
            or not np.array_equal(np.unique(fp_source), fp_labels)
            or np.any(~np.isin(anchor_source, fp_labels))
            or np.any(linked_parent < 0) or np.any(linked_parent >= len(parent_id))):
        raise ValueError("frozen split, source identities or link dimensions changed")
    point_index = {int(rec): i for i, rec in enumerate(point_recno)}
    fp_index = {int(pgc): str(label) for pgc, label in zip(fp_pgc, fp_source, strict=True)}
    node_roles: dict[tuple, int] = {}
    for i, held in enumerate(base_heldout):
        node_roles[("point", i)] = int(held)
    for gid, role in zip(cf4_ids, cf4_old_roles, strict=True):
        node_roles[("cf4", int(gid))] = int(role)
    for label, role in zip(fp_labels, fp_old_roles, strict=True):
        node_roles[("fp", str(label))] = int(role)

    edges: list[tuple[tuple, tuple]] = []
    count_key = point_pop * N**3 + point_flat
    for i, key in enumerate(count_key):
        edges.append((("point", i), ("count_key", int(key))))

    m2pp_gid_by_recno = load_m2pp_gids(M2PP)
    for i, recno in enumerate(point_recno):
        if int(recno) not in m2pp_gid_by_recno:
            raise ValueError(f"eligible count recno {recno} absent from 2M++ source")
        gid = m2pp_gid_by_recno[int(recno)]
        if gid >= 0:
            edges.append((("point", i), ("m2pp_gid", gid)))

    if not (len(linked_recno) == len(linked_gid) == len(linked_parent)):
        raise ValueError("independent Tempel--2M++ link arrays disagree")
    for recno, gid, parent in zip(linked_recno, linked_gid, linked_parent, strict=True):
        parent_node = ("tempel_parent", int(parent))
        point_i = point_index.get(int(recno))
        if point_i is not None:
            edges.append((("point", point_i), parent_node))
        if gid >= 0:
            gid_node = ("m2pp_gid", int(gid))
            edges.append((gid_node, parent_node))

    for point_i, group in zip(edge_point, edge_cf4, strict=True):
        if point_i < 0 or point_i >= len(point_recno):
            raise ValueError("point-to-CF4 edge has an invalid point index")
        if group >= 0:
            edges.append((("point", int(point_i)), ("cf4", int(group))))
    for label, group in zip(fp_source, fp_cf4, strict=True):
        if group >= 0:
            edges.append((("fp", str(label)), ("cf4", int(group))))
    for group, label in zip(anchor_cf4, anchor_source, strict=True):
        if group >= 0:
            edges.append((("cf4", int(group)), ("fp", str(label))))

    # Secure direct member links are retained as graph edges; ambiguous or
    # non-secure catalogue matches do not become invented identities.
    rec_index = point_index
    secure_direct_edges: set[tuple[int, str]] = set()
    with CROSSMATCH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["match_class"] != "secure_joint_mark" or not row["twompp_recno"]:
                continue
            point_i = rec_index.get(int(row["twompp_recno"]))
            label = fp_index.get(int(row["PGC"]))
            if point_i is not None and label is not None:
                secure_direct_edges.add((point_i, label))
    for point_i, label in secure_direct_edges:
        edges.append((("point", point_i), ("fp", label)))

    objid_sorted, parent_sorted, tempel_group_sorted = load_tempel_parent_index(
        TEMPEL_TABLE, len(parent_id))
    if (not np.array_equal(parent_kind, np.r_[np.zeros(NTEMPEL_GROUPS, dtype=np.int8),
                                               np.ones(len(parent_id)-NTEMPEL_GROUPS,
                                                        dtype=np.int8)])):
        raise ValueError("Tempel parent index convention differs from frozen source")
    if np.any(fp_objid_raw < 0):
        raise ValueError("negative FP source object ID")
    fp_objid = fp_objid_raw.astype(np.uint64)
    fp_pos = np.searchsorted(objid_sorted, fp_objid)
    fp_found = fp_pos < len(objid_sorted)
    fp_found &= objid_sorted[np.minimum(fp_pos, len(objid_sorted)-1)] == fp_objid
    source_parent_mismatches = 0
    for i in np.flatnonzero(fp_found):
        parent = int(parent_sorted[fp_pos[i]])
        label = str(fp_source[i])
        if label.startswith("T"):
            try:
                expected_group = int(label[1:])
            except ValueError as exc:
                raise ValueError(f"malformed Tempel source label {label}") from exc
            mismatch = int(tempel_group_sorted[fp_pos[i]]) != expected_group
        else:
            mismatch = int(tempel_group_sorted[fp_pos[i]]) != 0
        if mismatch:
            source_parent_mismatches += 1
        edges.append((("fp", label), ("tempel_parent", parent)))
    if source_parent_mismatches:
        raise ValueError(f"{source_parent_mismatches} FP source-parent identity conflicts")

    final, graph_summary = close_graph_roles(node_roles, edges)
    point_role = np.asarray([final[("point", i)] for i in range(len(point_recno))],
                            dtype=np.int8)
    cf4_role = np.asarray([final[("cf4", int(gid))] for gid in cf4_ids], dtype=np.int8)
    fp_role = np.asarray([final[("fp", str(label))] for label in fp_labels], dtype=np.int8)
    if np.any(~np.isin(point_role, (0, 1, 2))):
        raise AssertionError("eligible count point remained unassigned")

    all_keys, all_counts = np.unique(count_key, return_counts=True)
    role_keys = {}
    for role in (0, 1, 2):
        role_keys[role] = np.unique(count_key[point_role == role])
    if any(np.intersect1d(role_keys[a], role_keys[b]).size
           for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("a population/count voxel crosses final graph roles")
    for role in (0, 1, 2):
        ids, counts = np.unique(count_key[point_role == role], return_counts=True)
        role_keys[role] = (ids.astype(np.int32), counts.astype(np.int32))
    joined_keys = np.concatenate([role_keys[r][0] for r in (0, 1, 2)])
    joined_counts = np.concatenate([role_keys[r][1] for r in (0, 1, 2)])
    order = np.argsort(joined_keys)
    if (not np.array_equal(joined_keys[order], all_keys)
            or not np.array_equal(joined_counts[order], all_counts)):
        raise AssertionError("train/heldout/buffer count projection is not exact")

    with np.load(V5, allow_pickle=False) as z:
        held_voxels = z["heldout_flat_voxels"].astype(np.int32)
    base_hold_voxel = np.zeros(N**3, dtype=bool)
    base_hold_voxel[held_voxels] = True
    buffer_keys = role_keys[2][0].astype(np.int64)
    buffer_flat = buffer_keys % (N**3)
    buffer_was_heldout = base_hold_voxel[buffer_flat]
    parent_nodes = [("tempel_parent", i) for i in range(len(parent_id))]
    parent_role = np.asarray([final.get(node, 3) for node in parent_nodes], dtype=np.int8)
    gid_nodes = sorted(key for key in final if key[0] == "m2pp_gid")
    gid_values = np.asarray([key[1] for key in gid_nodes], dtype=np.int64)
    gid_role = np.asarray([final[key] for key in gid_nodes], dtype=np.int8)
    changed_cf4 = int(np.count_nonzero(cf4_role != cf4_old_roles))
    changed_fp = int(np.count_nonzero(fp_role != fp_old_roles))
    source_hashes = {str(path): sha256(path) for path in
                     (V5, POINTS, EDGES, FP, ANCHORS, PARENTS, LINKS,
                      TEMPEL_TABLE, M2PP, CROSSMATCH)}
    report = dict(
        classification="R2_SOURCE_GRAPH_CLOSED_HOLDOUT_NOT_LIKELIHOOD_OR_POSTERIOR",
        job_id=os.environ["SLURM_JOB_ID"], source_commit=commit,
        base_split="v5 geometry octant retained; identity graph components that mix roles are buffered",
        role_codes={"0": "train", "1": "heldout", "2": "buffer", "3": "unassigned Tempel/GID"},
        graph=graph_summary,
        m2pp_catalogue_rows=len(m2pp_gid_by_recno),
        point_role_counts=np.bincount(point_role, minlength=3).tolist(),
        cf4_role_counts=np.bincount(cf4_role, minlength=3).tolist(),
        fp_source_role_counts=np.bincount(fp_role, minlength=3).tolist(),
        tempel_parent_role_counts=np.bincount(parent_role, minlength=4).tolist(),
        Tempel_parent_mixed_in_full_source=int(np.count_nonzero(old_parent_role == 3)),
        Tempel_parent_mixed_inside_withdrawn_proxy_window=int(np.count_nonzero(
            (old_parent_role == 3) & np.load(PARENTS, allow_pickle=False)[
                "approx_mask_support_and_tempel_z_window"])),
        Tempel_FP_source_identity_mismatches=source_parent_mismatches,
        Tempel_FP_source_rows_linked=int(fp_found.sum()),
        direct_secure_count_FP_edges=len(secure_direct_edges),
        graph_association_limits="Reciprocal 10-arcsec/300-km-s-1 Tempel--2M++ links and published secure crossmatches are used only to conservatively prevent split leakage; they are not physical membership probabilities or a calibrated selection law.",
        train_count_keys=int(len(role_keys[0][0])),
        heldout_count_keys=int(len(role_keys[1][0])),
        buffer_count_keys=int(len(buffer_keys)),
        buffer_keys_from_base_heldout=int(buffer_was_heldout.sum()),
        buffer_keys_from_base_training=int((~buffer_was_heldout).sum()),
        old_train_to_buffer_points=int(np.count_nonzero((point_role == 2) & ~base_heldout)),
        old_heldout_to_buffer_points=int(np.count_nonzero((point_role == 2) & base_heldout)),
        CF4_roles_changed_from_v5=changed_cf4,
        FP_roles_changed_from_v5=changed_fp,
        exact_count_projection=True,
        heldout_values_or_model_scores_read=False,
        field_fit=False, sampler=False, R2_posterior=False, N256=False,
        new_gravity_runs=0,
        MW_M31_M33="No component identities seed or select candidates. MW/M31 remain ambiguous latent roles on a NEW evolved field; M33 remains explicitly unresolved when unsupported, and their observables must constrain that same field.",
        Q_GOAL="Restores source-graph separation needed before prospective CF4+galaxy heldout scoring; it does not itself constrain or reconstruct a field.",
        Q_LEAN="One identity-only union/closure over already-frozen sources, without new likelihood, simulation, random mask proxy, score sweep or sampler.",
        source_sha256=source_hashes,
        prior_split_path=str(V5),
        split_path=str(OUT/"split.npz"))

    OUT.mkdir(parents=True)
    np.savez_compressed(
        OUT/"split.npz",
        point_recno=point_recno,
        point_role=point_role,
        point_heldout=point_role == 1,
        point_buffer=point_role == 2,
        point_base_heldout=base_heldout,
        heldout_flat_voxels=held_voxels,
        train_keys=role_keys[0][0], train_counts=role_keys[0][1],
        heldout_keys=role_keys[1][0], heldout_counts=role_keys[1][1],
        buffer_keys=role_keys[2][0], buffer_counts=role_keys[2][1],
        train_window_excluded_keys=buffer_keys[~buffer_was_heldout].astype(np.int32),
        heldout_window_excluded_keys=buffer_keys[buffer_was_heldout].astype(np.int32),
        cf4_pgc=cf4_ids, cf4_role=cf4_role,
        fp_source_group=fp_labels, fp_role=fp_role,
        tempel_parent_kind=parent_kind, tempel_parent_id=parent_id,
        tempel_parent_role=parent_role,
        m2pp_gid=gid_values, m2pp_gid_role=gid_role)
    (OUT/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
