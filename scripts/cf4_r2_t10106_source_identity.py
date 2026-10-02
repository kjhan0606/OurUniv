"""Score-blind catalogue identity bridge for the frozen v6 T10106 control."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4/z0_density")
CENSUS = BASE / "r2_v6_multimember_graph_census_20261003_v2/result.json"
OUT = Path("/gpfs/kjhan/CF4/z0_density/r2_t10106_source_identity_20261003_v1")
SOURCES = {
    "cf4_2mpp_crossmatch_v1.csv": "64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf",
    "cf4_galaxies.csv": "28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709",
    "2mpp_catalog.csv": "05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d",
}
CONTROL = {
    "source_group_label": "T10106",
    "FP_rows": 2,
    "anchor_rows": 0,
    "member_recno": [52802, 52824],
    "member_training_role": [0, 0],
    "member_population": [3, 3],
    "member_flat_cell": [930008, 946135],
    "member_radius_cMpc_h": [125.56221619447454, 121.6602137466458],
    "distinct_member_count": 2,
    "distinct_population_voxel_keys": 2,
    "associated_PGC": [54049, 54054],
    "selection_rule": (
        "among v6 train groups with >=2 direct count members, >=1 FP row, and "
        "all members training: fewest anchors, then fewest FP rows, then lexical label"),
}


def catalogue_relation(member_groups):
    groups = [tuple(sorted({str(value).strip() for value in values} - {"", "-1"}))
              for values in member_groups]
    if all(not values for values in groups):
        return "all_unassigned"
    if any(len(values) > 1 for values in groups):
        return "ambiguous_member_grouping"
    if any(not values for values in groups):
        return "partly_unassigned"
    return ("shared_catalogue_group" if len({values[0] for values in groups}) == 1
            else "distinct_catalogue_groups")


def summarize_bridge(expected_recnos, expected_pgcs, secure_edges,
                     cf4_groups_by_pgc, twompp_gid_by_recno):
    recnos = tuple(sorted(map(int, expected_recnos)))
    pgcs = tuple(sorted(map(int, expected_pgcs)))
    if not recnos or len(set(recnos)) != len(recnos) or not pgcs or len(set(pgcs)) != len(pgcs):
        raise ValueError("frozen member IDs must be nonempty and unique")
    selected = [edge for edge in secure_edges if int(edge[0]) in set(pgcs)]
    if len(selected) != len(pgcs) or len({int(edge[0]) for edge in selected}) != len(pgcs):
        raise ValueError("each selected PGC must have exactly one secure edge")
    edge_by_pgc = {int(pgc): (str(group).strip(), int(recno))
                   for pgc, group, recno in selected}
    if tuple(sorted(recno for _, recno in edge_by_pgc.values())) != recnos:
        raise ValueError("secure links do not reproduce the frozen member recnos")

    cf4_groups, gids, members = [], [], []
    for pgc in pgcs:
        edge_group, recno = edge_by_pgc[pgc]
        edge_group = edge_group if edge_group not in ("", "-1") else ""
        member_groups = tuple(sorted({str(value).strip()
                                      for value in cf4_groups_by_pgc.get(pgc, ())}
                                     - {"", "-1"}))
        gid_text = str(twompp_gid_by_recno.get(recno, "")).strip()
        gid = None if gid_text in ("", "-1") else gid_text
        cf4_groups.append(member_groups)
        gids.append(() if gid is None else (gid,))
        members.append({"PGC": pgc, "2mpp_recno": recno,
                        "crossmatch_CF4_1PGC": edge_group or None,
                        "CF4_member_1PGC": list(member_groups),
                        "2mpp_GID": gid,
                        "crossmatch_CF4_group_matches_member_catalogue":
                            (not edge_group or edge_group in member_groups)})
    return {"members": members,
            "CF4_catalogue_group_relation": catalogue_relation(cf4_groups),
            "2mpp_catalogue_group_relation": catalogue_relation(gids),
            "cross_catalogue_IDs_equated": False}


def read_rows(path, columns):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} lacks required identifier columns")
        return [{name: row[name] for name in columns} for row in reader]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    control = census.get("selected_training_control")
    if census.get("source_commit") != "e9c06d3fdb30f2aa6cadfe091ad5d316c1c5224a" or control != CONTROL:
        raise ValueError("frozen score-blind training control changed")

    paths = {name: ROOT / "data" / name for name in SOURCES}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != SOURCES:
        raise ValueError("frozen source catalogue hash changed")
    target_pgcs = set(CONTROL["associated_PGC"])
    edges = [(int(row["PGC"]), row["1PGC"], int(row["twompp_recno"]))
             for row in read_rows(paths["cf4_2mpp_crossmatch_v1.csv"],
                                  ("PGC", "1PGC", "twompp_recno", "match_class"))
             if row["match_class"] == "secure_joint_mark" and row["twompp_recno"]
             and int(row["PGC"]) in target_pgcs]
    cf4_rows = read_rows(paths["cf4_galaxies.csv"], ("PGC", "1PGC"))
    cf4_groups = {}
    for row in cf4_rows:
        if row["PGC"].strip() and int(row["PGC"]) in target_pgcs:
            cf4_groups.setdefault(int(row["PGC"]), set()).add(row["1PGC"].strip())
    twompp_rows = read_rows(paths["2mpp_catalog.csv"], ("recno", "GID"))
    gid_by_recno = {int(row["recno"]): row["GID"].strip() for row in twompp_rows}
    result = summarize_bridge(CONTROL["member_recno"], CONTROL["associated_PGC"],
                              edges, cf4_groups, gid_by_recno)
    report = {"status": "T10106_CATALOGUE_IDENTITY_ONLY_NOT_PHYSICAL_MEMBERSHIP",
              "job_id": os.environ["SLURM_JOB_ID"], "source_commit": actual,
              "frozen_control": CONTROL, "catalogue_group_bridge": result,
              "source_sha256": hashes, "census_result_sha256": sha256(CENSUS),
              "used_CF4_or_FP_distance_marks": False, "used_heldout_rows": False,
              "used_likelihood_or_field_state": False, "PM_evolutions": 0,
              "elapsed_seconds": time.monotonic() - started,
              "limit": "Catalogue assignments do not calibrate physical membership, selection, covariance, or a field likelihood."}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
