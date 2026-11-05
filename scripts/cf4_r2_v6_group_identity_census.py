"""Census catalogue-group concordance for the frozen v6 train multi-link set."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from collections import Counter

import numpy as np

from cf4_r2_t10106_source_identity import catalogue_relation
from cf4_r2_v6_multimember_graph_census import eligible_multilink_training_groups


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4/z0_density")
SPLIT = BASE / "r2_sky_closed_split_v6/split.npz"
POINTS = BASE / "r2_point_mark_manifest_v1/points.npz"
FP = BASE / "r2_source_observation_assembly_v1/observations.npz"
GROUP = BASE / "r2_hierarchical_field_geometry_v1/geometry_q257.npz"
CENSUS = BASE / "r2_v6_multimember_graph_census_20261003_v2/result.json"
EXPECTED_CENSUS_SHA = "2b82b2e76b9b6e9eb2641bec8a31f21b6393ee0de2e269b6fb2ceaf298d67341"
OUT = BASE / "r2_v6_multimember_group_identity_20261003_v1"
SOURCES = {
    SPLIT: "9528b54a1e9dc876052771c21c0330ebba5f9b2b56966ab3cbe593a71cfc2b74",
    POINTS: "9377de48a3cc045b8432266ee5124d3540411fdcddd0d83b8036aa6744d11139",
    FP: "90d6aecca7f62cd09767bd444df99cc60b8a557b21605cbdc31afdc38ea10b28",
    GROUP: "f80e34099894c3fdd090e45e775d5352a1d06b19583be47c11558d5af5941b38",
    ROOT / "data/cf4_2mpp_crossmatch_v1.csv": "64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf",
    ROOT / "data/cf4_galaxies.csv": "28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709",
    ROOT / "data/2mpp_catalog.csv": "05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d",
}


def source_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_relation_census(group_members, edges_by_group, cf4_groups_by_pgc,
                          gid_by_recno):
    rows = []
    for label in sorted(group_members):
        recnos = sorted(group_members[label])
        edges = edges_by_group.get(label, ())
        pgcs = sorted({int(edge[0]) for edge in edges})
        cf4_membership = [cf4_groups_by_pgc.get(pgc, ()) for pgc in pgcs]
        point_gids = []
        for recno in recnos:
            gid = str(gid_by_recno.get(recno, "")).strip()
            point_gids.append(() if gid in ("", "-1") else (gid,))
        consistency = Counter()
        for pgc, edge_group, _ in edges:
            assignments = {str(value).strip() for value in cf4_groups_by_pgc.get(int(pgc), ())}
            assignments -= {"", "-1"}
            edge_group = str(edge_group).strip()
            if not assignments:
                consistency["member_group_missing"] += 1
            elif len(assignments) > 1:
                consistency["member_group_ambiguous"] += 1
            elif edge_group in ("", "-1"):
                consistency["crossmatch_group_missing"] += 1
            elif edge_group in assignments:
                consistency["match"] += 1
            else:
                consistency["mismatch"] += 1
        rows.append({
            "source_group_label": label,
            "count_member_recno_count": len(recnos),
            "linked_PGC_count": len(pgcs),
            "secure_edge_count": len(edges),
            "CF4_catalogue_group_relation": catalogue_relation(cf4_membership),
            "2mpp_catalogue_group_relation": catalogue_relation(point_gids),
            "crossmatch_vs_CF4_member_table": dict(consistency),
        })
    return rows


def read_selected_rows(path, columns, predicate):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} lacks expected identity columns")
        return [{name: row[name] for name in columns} for row in reader if predicate(row)]


def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CF4_EXPECTED_COMMIT"):
        raise RuntimeError("Slurm job and pinned source commit are required")
    expected = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{os.environ['CF4_EXPECTED_COMMIT']}^{{commit}}"],
        cwd=ROOT, text=True).strip()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if expected != actual:
        raise RuntimeError("source commit mismatch")
    if Path(os.environ["CF4_R2_OUT_DIR"]) != OUT or OUT.exists():
        raise FileExistsError("unexpected or already-used output path")
    started = time.monotonic()
    hashes = {str(path): source_hash(path) for path in SOURCES}
    if hashes != {str(path): digest for path, digest in SOURCES.items()}:
        raise ValueError("one or more frozen source hashes changed")
    if source_hash(CENSUS) != EXPECTED_CENSUS_SHA:
        raise ValueError("frozen score-blind graph census changed")
    frozen_census = json.loads(CENSUS.read_text(encoding="utf-8"))
    if (frozen_census.get("source_commit") != "e9c06d3fdb30f2aa6cadfe091ad5d316c1c5224a"
            or frozen_census.get("eligible_training_multilink_group_count") != 272
            or frozen_census.get("selected_training_control", {}).get("source_group_label") != "T10106"):
        raise ValueError("frozen v6 multi-link census contract changed")
    hashes[str(CENSUS)] = EXPECTED_CENSUS_SHA

    with np.load(SPLIT, allow_pickle=False) as data:
        point_recno = data["point_recno"].astype(np.int64, copy=True)
        point_role = data["point_role"].astype(np.int8, copy=True)
        labels = data["fp_source_group"].astype(str)
        fp_role = data["fp_role"].astype(np.int8, copy=True)
    with np.load(POINTS, allow_pickle=False) as data:
        catalogue_recno = data["recno"].astype(np.int64, copy=True)
    with np.load(FP, allow_pickle=False) as data:
        pgc = data["PGC"].astype(np.int64, copy=True)
        source_group = data["source_group"].astype(str)
    with np.load(GROUP, allow_pickle=False) as data:
        group_labels = data["group_labels"].astype(str)
        row_group = data["row_group"].astype(np.int64, copy=True)
        anchor_group = data["anchor_group"].astype(np.int64, copy=True)

    np.testing.assert_array_equal(labels, group_labels)
    if (len(point_recno) != len(point_role) or len(labels) != len(fp_role)
            or len(np.unique(point_recno)) != len(point_recno)
            or len(np.unique(catalogue_recno)) != len(catalogue_recno)):
        raise ValueError("split arrays are misaligned")
    if np.any(row_group < 0) or np.any(row_group >= len(labels)):
        raise ValueError("FP row has an invalid frozen source-group index")
    np.testing.assert_array_equal(source_group, labels[row_group])
    point_role_by_recno = dict(zip(map(int, point_recno), map(int, point_role)))
    point_index = {int(recno): i for i, recno in enumerate(catalogue_recno)}
    label_index = {str(label): i for i, label in enumerate(labels)}
    pgc_group = {}
    for object_id, label in zip(pgc, source_group):
        old = pgc_group.setdefault(int(object_id), str(label))
        if old != str(label):
            raise ValueError(f"PGC {object_id} maps to multiple Tempel source groups")

    group_members, edges_by_group = {}, {}
    crossmatch_path = ROOT / "data/cf4_2mpp_crossmatch_v1.csv"
    with crossmatch_path.open(newline="", encoding="utf-8") as stream:
        for edge in csv.DictReader(stream):
            if edge["match_class"] != "secure_joint_mark" or not edge["twompp_recno"]:
                continue
            pgc_id, recno = int(edge["PGC"]), int(edge["twompp_recno"])
            label = pgc_group.get(pgc_id)
            if label is None or recno not in point_index:
                continue
            group_members.setdefault(label, set()).add(recno)
            edges_by_group.setdefault(label, []).append((pgc_id, edge["1PGC"], recno))

    fp_rows = np.bincount(row_group, minlength=len(labels))
    anchor_rows = np.bincount(anchor_group, minlength=len(labels))
    eligible = eligible_multilink_training_groups(
        labels, fp_role, fp_rows, anchor_rows, group_members, point_role_by_recno)
    if len(eligible) != 272:
        raise ValueError(f"v6 eligible training group count changed: {len(eligible)}")
    target_pgcs = {int(edge[0]) for label in eligible for edge in edges_by_group[label]}
    target_recnos = {recno for label in eligible for recno in group_members[label]}

    member_rows = read_selected_rows(
        ROOT / "data/cf4_galaxies.csv", ("PGC", "1PGC"),
        lambda row: bool(row["PGC"].strip()) and int(row["PGC"]) in target_pgcs)
    cf4_groups = {}
    for row in member_rows:
        cf4_groups.setdefault(int(row["PGC"]), set()).add(row["1PGC"].strip())
    point_rows = read_selected_rows(
        ROOT / "data/2mpp_catalog.csv", ("recno", "GID"),
        lambda row: int(row["recno"]) in target_recnos)
    point_row_recnos = [int(row["recno"]) for row in point_rows]
    if len(point_row_recnos) != len(set(point_row_recnos)):
        raise ValueError("2M++ source repeats a linked recno")
    gid_by_recno = {int(row["recno"]): row["GID"].strip() for row in point_rows}
    if set(gid_by_recno) != target_recnos:
        raise ValueError("not every eligible point recno joined to the 2M++ source table")

    groups = group_relation_census(group_members={label: group_members[label] for label in eligible},
        edges_by_group=edges_by_group, cf4_groups_by_pgc=cf4_groups,
        gid_by_recno=gid_by_recno)
    pair_counts = Counter((row["CF4_catalogue_group_relation"],
                           row["2mpp_catalogue_group_relation"]) for row in groups)
    control = next(row for row in groups if row["source_group_label"] == "T10106")
    if (control["count_member_recno_count"] != 2 or control["linked_PGC_count"] != 2
            or control["secure_edge_count"] != 2
            or control["CF4_catalogue_group_relation"] != "shared_catalogue_group"
            or control["2mpp_catalogue_group_relation"] != "shared_catalogue_group"
            or control["crossmatch_vs_CF4_member_table"] != {"match": 2}):
        raise ValueError("full-cohort result does not reproduce the frozen T10106 bridge")
    report = {
        "status": "V6_TRAIN_MULTIMEMBER_CATALOGUE_IDENTITY_CENSUS_NOT_PHYSICAL_MEMBERSHIP",
        "job_id": os.environ["SLURM_JOB_ID"], "source_commit": actual,
        "split": "r2_sky_closed_split_v6", "eligible_training_group_count": len(groups),
        "identity_relation_counts": {f"{cf4}|{mpp}": n for (cf4, mpp), n in pair_counts.items()},
        "crossmatch_CF4_group_member_table_counts": dict(Counter(
            status for row in groups for status, n in row["crossmatch_vs_CF4_member_table"].items()
            for _ in range(n))),
        "T10106_control": control, "groups": groups,
        "input_sha256": hashes,
        "heldout_measurements_read": False, "FP_distance_marks_read": False,
        "likelihood_scores_read": False, "field_state_read": False,
        "PM_evolutions": 0, "posterior_promoted": False, "R2_complete": False,
        "limit": "Catalogue concordance does not establish physical membership, inclusion, covariance or a same-field likelihood.",
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
