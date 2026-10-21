"""Summarize source redshift overlap in the frozen v6 training cohort."""

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


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4/z0_density")
FP = BASE / "r2_source_observation_assembly_v1/observations.npz"
POINTS = BASE / "r2_point_mark_manifest_v1/points.npz"
GRAPH_CENSUS = BASE / "r2_v6_multimember_graph_census_20261003_v2/result.json"
IDENTITY_CENSUS = BASE / "r2_v6_multimember_group_identity_20261003_v1/result.json"
OUT = BASE / "r2_v6_redshift_overlap_20261003_v3"
GRAPH_CENSUS_SHA = "2b82b2e76b9b6e9eb2641bec8a31f21b6393ee0de2e269b6fb2ceaf298d67341"
IDENTITY_CENSUS_SHA = "0bf09ede7230a17e71ceb983941d55fe4705be0e2eefb2362f11448737ef3692"
FP_SHA = "90d6aecca7f62cd09767bd444df99cc60b8a557b21605cbdc31afdc38ea10b28"
POINTS_SHA = "9377de48a3cc045b8432266ee5124d3540411fdcddd0d83b8036aa6744d11139"
SOURCE_HASHES = {
    "cf4_2mpp_crossmatch_v1.csv": "64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf",
    "cf4_galaxies.csv": "28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709",
    "cf4_groups.csv": "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334",
    "2mpp_catalog.csv": "05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d",
    "2mpp_groups.csv": "e83bcad0ebe97f6048f36b3235f66babaadf995fc2257302313160f35980057a",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_rows(path, columns, predicate):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{Path(path).name} lacks required columns")
        return [{key: row[key] for key in columns} for row in reader if predicate(row)]


def eligible_secure_edge(row, pgc_group, point_recnos):
    return (row["match_class"] == "secure_joint_mark" and row["twompp_recno"].strip()
            and int(row["PGC"]) in pgc_group
            and int(row["twompp_recno"]) in point_recnos)


def number(value):
    value = str(value).strip()
    if not value:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "median": None, "p90": None, "maximum": None}
    return {"n": int(values.size), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "maximum": float(np.max(values))}


METRICS = (
    "abs_CF4_individual_Vcmb_minus_2mpp_individual_Vcmb_km_s",
    "abs_CF4_group_Vcmb_minus_linked_CF4_member_Vcmb_km_s",
    "abs_2mpp_group_Vcmb_minus_linked_2mpp_member_Vcmb_km_s",
    "abs_CF4_group_Vcmb_minus_linked_2mpp_group_Vcmb_km_s",
)


def summarize_group_velocity_overlap(groups, cf4_group, cf4_groups_by_pgc,
                                    cf4_vcmb_by_pgc, mpp_point, mpp_group):
    result = []
    for label, edges in sorted(groups.items()):
        cf4_relation_members, mpp_relation_members = [], []
        member_offsets, cf4_group_member_offsets, mpp_group_member_offsets = [], [], []
        group_pair_offsets = {}
        mismatches = ambiguous_member_velocities = 0
        for pgc, edge_group, recno in edges:
            pgc, recno = int(pgc), int(recno)
            cf4_ids = {str(item).strip() for item in cf4_groups_by_pgc.get(pgc, ())} - {"", "-1"}
            edge_group = str(edge_group).strip()
            if edge_group not in ("", "-1") and edge_group not in cf4_ids:
                mismatches += 1
            cf4_relation_members.append(cf4_ids)

            point = mpp_point[recno]
            gid = str(point["GID"]).strip()
            mpp_relation_members.append(set() if gid in ("", "-1") else {gid})
            member_v = [number(v) for v in cf4_vcmb_by_pgc.get(pgc, ())]
            member_v = [v for v in member_v if v is not None]
            if len(member_v) > 1:
                ambiguous_member_velocities += 1
            cf4_v, mpp_v = (member_v[0] if len(member_v) == 1 else None), number(point["Vcmb"])
            if cf4_v is not None and mpp_v is not None:
                member_offsets.append(abs(cf4_v - mpp_v))

            for cf4_id in cf4_ids:
                cf4_v_group = number(cf4_group.get(cf4_id, {}).get("Vcmb"))
                mpp_v_group = number(mpp_group.get(gid, {}).get("Vcmb")) if gid not in ("", "-1") else None
                if cf4_v_group is not None and cf4_v is not None:
                    cf4_group_member_offsets.append(abs(cf4_v_group - cf4_v))
                if cf4_v_group is not None and mpp_v_group is not None:
                    group_pair_offsets[(cf4_id, gid)] = abs(cf4_v_group - mpp_v_group)
            mpp_v_group = number(mpp_group.get(gid, {}).get("Vcmb")) if gid not in ("", "-1") else None
            if mpp_v_group is not None and mpp_v is not None:
                mpp_group_member_offsets.append(abs(mpp_v_group - mpp_v))

        samples = {
            METRICS[0]: member_offsets,
            METRICS[1]: cf4_group_member_offsets,
            METRICS[2]: mpp_group_member_offsets,
            METRICS[3]: list(group_pair_offsets.values()),
        }
        cf4_relation = catalogue_relation(cf4_relation_members)
        mpp_relation = catalogue_relation(mpp_relation_members)
        result.append({
            "source_group_label": label,
            "relation_class": f"{cf4_relation}|{mpp_relation}",
            "secure_member_pair_count": len(edges),
            "CF4_member_table_edge_group_mismatches": mismatches,
            "ambiguous_CF4_member_velocity_records": ambiguous_member_velocities,
            **{metric: describe(values) for metric, values in samples.items()},
            "_samples": samples,
            "_group_pairs": group_pair_offsets,
        })
    return result


def aggregate(rows):
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation_class"]].append(row)
    result = {}
    for relation, members in sorted(by_relation.items()):
        group_pairs = {}
        for row in members:
            group_pairs.update(row["_group_pairs"])
        result[relation] = {
            "group_count": len(members),
            "secure_member_pair_count": sum(row["secure_member_pair_count"] for row in members),
            "CF4_member_table_edge_group_mismatches": sum(
                row["CF4_member_table_edge_group_mismatches"] for row in members),
            "ambiguous_CF4_member_velocity_records": sum(
                row["ambiguous_CF4_member_velocity_records"] for row in members),
            **{metric: describe(
                list(group_pairs.values()) if metric == METRICS[3]
                else [value for row in members for value in row["_samples"][metric]])
               for metric in METRICS},
        }
    return result


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
    if (sha256(IDENTITY_CENSUS) != IDENTITY_CENSUS_SHA or sha256(FP) != FP_SHA
            or sha256(POINTS) != POINTS_SHA):
        raise ValueError("frozen identity census or source/point graph changed")
    census = json.loads(IDENTITY_CENSUS.read_text(encoding="utf-8"))
    labels = {row["source_group_label"] for row in census["groups"]}
    if (census.get("status") != "V6_TRAIN_MULTIMEMBER_CATALOGUE_IDENTITY_CENSUS_NOT_PHYSICAL_MEMBERSHIP"
            or census.get("eligible_training_group_count") != 272 or len(labels) != 272
            or census.get("input_sha256", {}).get(str(GRAPH_CENSUS)) != GRAPH_CENSUS_SHA
            or census.get("input_sha256", {}).get(str(POINTS)) != POINTS_SHA
            or census.get("input_sha256", {}).get(str(FP)) != FP_SHA):
        raise ValueError("frozen v6 training cohort contract changed")

    paths = {name: ROOT / "data" / name for name in SOURCE_HASHES}
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    if source_hashes != SOURCE_HASHES:
        raise ValueError("frozen source catalogues changed")
    with np.load(FP, allow_pickle=False) as data:
        pgcs, source_groups = data["PGC"].astype(np.int64), data["source_group"].astype(str)
    pgc_group = {}
    for pgc, group in zip(pgcs, source_groups):
        if group in labels:
            if int(pgc) in pgc_group and pgc_group[int(pgc)] != group:
                raise ValueError(f"PGC {pgc} appears in multiple selected source groups")
            pgc_group[int(pgc)] = group
    with np.load(POINTS, allow_pickle=False) as data:
        point_recnos = set(map(int, data["recno"]))

    groups = defaultdict(list)
    selected_edges = selected_rows(paths["cf4_2mpp_crossmatch_v1.csv"],
        ("PGC", "1PGC", "twompp_recno", "match_class"),
        lambda row: eligible_secure_edge(row, pgc_group, point_recnos))
    for edge in selected_edges:
        groups[pgc_group[int(edge["PGC"])]].append(
            (int(edge["PGC"]), edge["1PGC"], int(edge["twompp_recno"])))
    expected_links = sum(row["secure_edge_count"] for row in census["groups"])
    if set(groups) != labels or sum(map(len, groups.values())) != expected_links:
        raise ValueError(
            f"frozen cohort mismatch: groups={len(groups)}/{len(labels)}, "
            f"links={sum(map(len, groups.values()))}/{expected_links}")

    cf4_pairs = {(pgc, str(group).strip()) for edges in groups.values()
                 for pgc, group, _ in edges}
    recnos = {recno for edges in groups.values() for _, _, recno in edges}
    cf4_group_ids = {group for _, group in cf4_pairs if group not in ("", "-1")}
    cf4_group_rows = selected_rows(
        paths["cf4_groups.csv"], ("1PGC", "Vcmb"),
        lambda row: row["1PGC"].strip() in cf4_group_ids)
    cf4_groups = {row["1PGC"].strip(): row for row in cf4_group_rows}
    if len(cf4_groups) != len(cf4_group_rows):
        raise ValueError("selected CF4 group catalogue repeats a 1PGC")
    member_groups, member_velocities = {}, defaultdict(list)
    for row in selected_rows(paths["cf4_galaxies.csv"], ("PGC", "1PGC", "Vcmb"),
            lambda row: bool(row["PGC"].strip())
                and (int(row["PGC"]), row["1PGC"].strip()) in cf4_pairs):
        pgc = int(row["PGC"])
        member_groups.setdefault(pgc, set()).add(row["1PGC"].strip())
        velocity = number(row["Vcmb"])
        if velocity is not None and velocity not in member_velocities[pgc]:
            member_velocities[pgc].append(velocity)
    mpp_rows = selected_rows(paths["2mpp_catalog.csv"], ("recno", "GID", "Vcmb"),
                             lambda row: int(row["recno"]) in recnos)
    mpp_points = {int(row["recno"]): row for row in mpp_rows}
    if len(mpp_points) != len(mpp_rows) or set(mpp_points) != recnos:
        raise ValueError("selected secure 2M++ recnos are duplicated or missing")
    gids = {row["GID"].strip() for row in mpp_rows if row["GID"].strip() not in ("", "-1")}
    mpp_group_rows = selected_rows(paths["2mpp_groups.csv"], ("GID", "Vcmb"),
                                   lambda row: row["GID"].strip() in gids)
    mpp_groups = {row["GID"].strip(): row for row in mpp_group_rows}
    if len(mpp_groups) != len(mpp_group_rows):
        raise ValueError("selected 2M++ group catalogue repeats a GID")

    rows = summarize_group_velocity_overlap(
        groups, cf4_groups, member_groups, member_velocities, mpp_points, mpp_groups)
    if any(row["CF4_member_table_edge_group_mismatches"] for row in rows):
        raise ValueError("secure crossmatch CF4 group IDs disagree with member table")
    relation_counts = Counter(row["relation_class"] for row in rows)
    if dict(relation_counts) != census["identity_relation_counts"]:
        raise ValueError("relation classes differ from the frozen identity census")
    control = next(row for row in rows if row["source_group_label"] == "T10106")
    if (control["secure_member_pair_count"] != 2
            or control["relation_class"] != "shared_catalogue_group|shared_catalogue_group"):
        raise ValueError("T10106 control changed")

    report = {
        "status": "V6_TRAIN_GROUP_REDSHIFT_OVERLAP_SOURCE_READOUT_NOT_COVARIANCE_CALIBRATION",
        "job_id": os.environ["SLURM_JOB_ID"], "source_commit": actual,
        "group_count": len(rows), "secure_member_pair_count": expected_links,
        "relation_counts": dict(relation_counts), "relation_summaries": aggregate(rows),
        "T10106_control": {key: value for key, value in control.items()
                            if key not in ("_samples", "_group_pairs")},
        "groups": [{key: value for key, value in row.items()
                    if key not in ("_samples", "_group_pairs")} for row in rows],
        "source_sha256": source_hashes, "FP_graph_sha256": FP_SHA,
        "count_point_manifest_sha256": POINTS_SHA,
        "identity_census_sha256": IDENTITY_CENSUS_SHA, "graph_census_sha256": GRAPH_CENSUS_SHA,
        "heldout_measurement_values_used": False, "FP_distance_marks_read": False,
        "likelihood_scores_read": False, "field_state_read": False, "PM_evolutions": 0,
        "posterior_promoted": False, "R2_complete": False,
        "limits": [
            "CF4 1PGC, 2M++ GID, and v6 Tempel source labels are distinct namespaces.",
            "Secure crossmatches pair catalogue rows but do not prove physical group membership.",
            "Velocity offsets are matched-source descriptions, not independent errors or a fitted covariance law.",
            "CF4 distance-contributor membership does not identify the systemic-velocity member process.",
            "MW/M31 roles remain ambiguous and M33 unresolved; eventual observables must constrain the same NEW field at LG <=0.3 cMpc/h.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
