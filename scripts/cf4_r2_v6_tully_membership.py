"""Crosswalk the frozen v6 training links to archived Tully-2015 Nest IDs."""

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import time

import numpy as np

from cf4_r2_t10106_source_identity import catalogue_relation
from cf4_r2_tully2015_source_bridge import HASHES as TULLY_HASHES, integer, records
from cf4_r2_v6_redshift_overlap import (
    FP, FP_SHA, IDENTITY_CENSUS, IDENTITY_CENSUS_SHA, POINTS, POINTS_SHA,
    SOURCE_HASHES, eligible_secure_edge, selected_rows,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4/z0_density")
Z0_RESULT = BASE / "r2_v6_redshift_overlap_20261003_v3/result.json"
Z0_RESULT_SHA = "47fd803bab67eb96094489a4c8d725314a43ac896e315a55bbcfbbf741d13da0"
OUT = BASE / "r2_v6_tully_membership_20261003_v1"


def summarize_tully_membership(groups, parent_by_pgc1, table4_by_pgc, table5_by_pgc):
    result = []
    for label, edges in sorted(groups.items()):
        member_assignments, parent_assignments = [], []
        parent_status, table5_table4_status = Counter(), Counter()
        for pgc, raw_cf4_group, _recno, _relation in edges:
            cf4_group = str(raw_cf4_group).strip()
            t4_nests = {str(nest) for nest in table4_by_pgc.get(pgc, ())
                        if nest not in (None, 0)}
            t5_nests = {str(nest) for nest in table5_by_pgc.get(pgc, ())
                        if nest not in (None, 0)}
            parent_nests = {str(nest) for nest in parent_by_pgc1.get(cf4_group, ())
                            if nest not in (None, 0)}
            member_assignments.append(t4_nests)
            parent_assignments.append(parent_nests)

            if len(t4_nests) == len(t5_nests) == 1:
                table5_table4_status["match" if t4_nests == t5_nests else "mismatch"] += 1
            elif not t4_nests or not t5_nests:
                table5_table4_status["missing"] += 1
            else:
                table5_table4_status["ambiguous"] += 1

            if not t4_nests:
                parent_status["member_missing"] += 1
            elif len(t4_nests) > 1:
                parent_status["member_ambiguous"] += 1
            elif not parent_nests:
                parent_status["parent_missing"] += 1
            elif len(parent_nests) > 1:
                parent_status["parent_ambiguous"] += 1
            elif t4_nests == parent_nests:
                parent_status["member_matches_CF4_parent_Nest"] += 1
            else:
                parent_status["member_differs_from_CF4_parent_Nest"] += 1

        result.append({
            "source_group_label": label,
            "prior_CF4_2mpp_relation_class": edges[0][3],
            "secure_member_pair_count": len(edges),
            "Tully_member_Nest_relation": catalogue_relation(member_assignments),
            "Tully_parent_Nest_relation_from_CF4_1PGC": catalogue_relation(parent_assignments),
            "member_vs_parent_status": dict(parent_status),
            "Tully_table5_vs_table4_membership": dict(table5_table4_status),
        })
    return result


def aggregate_by_relation(rows):
    strata = defaultdict(list)
    for row in rows:
        strata[row["prior_CF4_2mpp_relation_class"]].append(row)
    output = {}
    for relation, members in sorted(strata.items()):
        member_status, table_status = Counter(), Counter()
        nest_relations, parent_relations = Counter(), Counter()
        for row in members:
            member_status.update(row["member_vs_parent_status"])
            table_status.update(row["Tully_table5_vs_table4_membership"])
            nest_relations[row["Tully_member_Nest_relation"]] += 1
            parent_relations[row["Tully_parent_Nest_relation_from_CF4_1PGC"]] += 1
        output[relation] = {
            "group_count": len(members),
            "secure_member_pair_count": sum(row["secure_member_pair_count"] for row in members),
            "Tully_member_Nest_relation_counts": dict(nest_relations),
            "Tully_parent_Nest_relation_counts": dict(parent_relations),
            "member_vs_parent_status_counts": dict(member_status),
            "Tully_table5_vs_table4_status_counts": dict(table_status),
        }
    return output


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if (sha256(Z0_RESULT) != Z0_RESULT_SHA
            or sha256(IDENTITY_CENSUS) != IDENTITY_CENSUS_SHA
            or sha256(FP) != FP_SHA or sha256(POINTS) != POINTS_SHA):
        raise ValueError("frozen v6 result or source manifest changed")
    identity = json.loads(IDENTITY_CENSUS.read_text(encoding="utf-8"))
    prior = json.loads(Z0_RESULT.read_text(encoding="utf-8"))
    prior_groups = {row["source_group_label"]: row for row in prior["groups"]}
    labels = set(prior_groups)
    if (identity.get("eligible_training_group_count") != 272 or len(labels) != 272
            or prior.get("group_count") != 272 or prior.get("secure_member_pair_count") != 828
            or labels != {row["source_group_label"] for row in identity["groups"]}):
        raise ValueError("frozen v6 cohort changed")

    crossmatch_path = ROOT / "data/cf4_2mpp_crossmatch_v1.csv"
    if sha256(crossmatch_path) != SOURCE_HASHES["cf4_2mpp_crossmatch_v1.csv"]:
        raise ValueError("frozen CF4/2M++ crossmatch changed")
    with np.load(FP, allow_pickle=False) as data:
        pgcs, source_groups = data["PGC"].astype(np.int64), data["source_group"].astype(str)
    pgc_group = {}
    for pgc, label in zip(pgcs, source_groups):
        if label in labels:
            if int(pgc) in pgc_group and pgc_group[int(pgc)] != label:
                raise ValueError("selected FP PGC appears in multiple source groups")
            pgc_group[int(pgc)] = label
    with np.load(POINTS, allow_pickle=False) as data:
        point_recnos = set(map(int, data["recno"]))
    edge_rows = selected_rows(crossmatch_path,
        ("PGC", "1PGC", "twompp_recno", "match_class"),
        lambda row: eligible_secure_edge(row, pgc_group, point_recnos))
    groups = defaultdict(list)
    for row in edge_rows:
        label = pgc_group[int(row["PGC"])]
        groups[label].append((int(row["PGC"]), row["1PGC"], int(row["twompp_recno"]),
                              prior_groups[label]["relation_class"]))
    if set(groups) != labels or sum(map(len, groups.values())) != 828:
        raise ValueError("selected v6 Tully cohort differs from frozen 272/828 contract")

    table3, table4, table5 = (records("tully2015_table3.dat.gz"),
                              records("tully2015_table4.dat.gz"),
                              records("tully2015_table5.dat.gz"))
    if (len(table3), len(table4), len(table5)) != (25474, 43038, 43038):
        raise ValueError("frozen Tully 2015 source row counts changed")
    parent_by_pgc1 = defaultdict(set)
    for line in table3:
        nest, pgc1 = integer(line, 3, 9), integer(line, 14, 21)
        if nest is not None and pgc1 is not None:
            parent_by_pgc1[str(pgc1)].add(nest)
    table4_by_pgc = defaultdict(set)
    for line in table4:
        pgc, nest = integer(line, 10, 17), integer(line, 3, 9)
        if pgc is not None and nest is not None:
            table4_by_pgc[pgc].add(nest)
    table5_by_pgc = defaultdict(set)
    for line in table5:
        pgc, nest = integer(line, 0, 7), integer(line, 99, 105)
        if pgc is not None and nest is not None:
            table5_by_pgc[pgc].add(nest)

    rows = summarize_tully_membership(groups, parent_by_pgc1, table4_by_pgc, table5_by_pgc)
    if {row["source_group_label"] for row in rows} != labels:
        raise ValueError("Tully membership readout lost a frozen v6 group")
    prior_relation_counts = Counter(row["relation_class"] for row in prior["groups"])
    current_relation_counts = Counter(row["prior_CF4_2mpp_relation_class"] for row in rows)
    if prior_relation_counts != current_relation_counts:
        raise ValueError("CF4/2M++ relation strata differ from frozen prior readout")

    report = {
        "status": "V6_TRAIN_TULLY2015_MEMBER_CROSSWALK_NOT_COVARIANCE_CALIBRATION",
        "job_id": os.environ["SLURM_JOB_ID"], "source_commit": actual,
        "group_count": len(rows), "secure_member_pair_count": 828,
        "relation_summaries": aggregate_by_relation(rows), "groups": rows,
        "Tully_2015_tables_sha256": {name: TULLY_HASHES[name] for name in (
            "tully2015_table3.dat.gz", "tully2015_table4.dat.gz", "tully2015_table5.dat.gz")},
        "CF4_2mpp_crossmatch_sha256": SOURCE_HASHES["cf4_2mpp_crossmatch_v1.csv"],
        "FP_sha256": FP_SHA, "point_manifest_sha256": POINTS_SHA,
        "identity_census_sha256": IDENTITY_CENSUS_SHA,
        "prior_redshift_overlap_sha256": Z0_RESULT_SHA,
        "heldout_measurement_values_used": False, "velocity_values_used": False,
        "FP_distance_marks_read": False, "likelihood_scores_read": False,
        "field_state_read": False, "PM_evolutions": 0,
        "posterior_promoted": False, "R2_complete": False,
        "limitation": ("This is a catalogue-member identity crosswalk using the archived Tully 2015 Nest table. "
                       "It does not calibrate the later EDD CF4 group revision, selection, or covariance."),
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
