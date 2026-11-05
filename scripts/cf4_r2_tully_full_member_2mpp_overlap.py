"""Source-only crosswalk of all Tully-2015 members to the local 2M++ rows.

This freezes object/member ownership only. It does not read redshift, magnitude,
CF4 mark, field, or held-out outcome values and is not a covariance calibration.
"""

import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4/z0_density")
SPLIT = BASE / "r2_sky_closed_split_v6/split.npz"
POINTS = BASE / "r2_point_mark_manifest_v1/points.npz"
OUT = BASE / "r2_tully_full_member_2mpp_overlap_20261003_v2"
MAX_SEPARATION_ARCSEC = 3.0

INPUT_HASHES = {
    "data/tully2015_table3.dat.gz": "dd85454fd68acd0ef7aeb432a4a336d07b31df7a79b37ab18b695fb0b4483618",
    "data/tully2015_table4.dat.gz": "902541fe765d50955038a60730ced8884151eb3963908dba127596298e33f771",
    "data/tully2015_table5.dat.gz": "337b9f24484ae34c973a7187fa002807b345e47e76ffabce066549783e2b2844",
    "data/2mrs_huchra2012_table3.dat.gz": "14a40e14dea131afbc2ff525e42b39fdc4094cf9d06d9a43952257eff80f1790",
    "data/2mpp_catalog.csv": "05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d",
    str(SPLIT): "9528b54a1e9dc876052771c21c0330ebba5f9b2b56966ab3cbe593a71cfc2b74",
    str(POINTS): "9377de48a3cc045b8432266ee5124d3540411fdcddd0d83b8036aa6744d11139",
}

TULLY_ROWS = {"table3": 25474, "table4": 43038, "table5": 43038}
HUCHRA_ROWS = 44599
TWOMPP_ROWS = 72973


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int(text, label):
    value = text.strip()
    if not value:
        raise ValueError(f"missing integer field: {label}")
    return int(value)


def parse_float(text, label):
    value = float(text)
    if not np.isfinite(value):
        raise ValueError(f"nonfinite coordinate: {label}")
    return value


def unit_vectors(lon_deg, lat_deg):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64) % 360.0)
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    if lon.ndim != 1 or lat.shape != lon.shape or not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("invalid longitude/latitude arrays")
    if np.any(np.abs(lat) > np.pi / 2 + 1e-14):
        raise ValueError("latitude outside [-90, 90] degrees")
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def crossmatch_coordinates(source_lon, source_lat, target_lon, target_lat,
                           max_separation_arcsec=MAX_SEPARATION_ARCSEC):
    """Return reciprocal unique positional matches and candidate counts.

    The strict radius-unique rule intentionally leaves close blends unresolved;
    no redshift-based rescue is performed.
    """
    source = unit_vectors(source_lon, source_lat)
    target = unit_vectors(target_lon, target_lat)
    if not len(source) or not len(target):
        raise ValueError("both coordinate catalogues must be nonempty")
    if max_separation_arcsec <= 0 or not np.isfinite(max_separation_arcsec):
        raise ValueError("maximum separation must be finite and positive")

    radius = 2.0 * np.sin(np.deg2rad(max_separation_arcsec / 3600.0) / 2.0)
    source_tree, target_tree = cKDTree(source), cKDTree(target)
    target_candidates = target_tree.query_ball_point(source, radius)
    source_candidates = source_tree.query_ball_point(target, radius)
    source_count = np.fromiter((len(items) for items in target_candidates), dtype=np.int32,
                               count=len(source))
    target_count = np.fromiter((len(items) for items in source_candidates), dtype=np.int32,
                               count=len(target))

    matches = []
    for source_i, candidates in enumerate(target_candidates):
        if len(candidates) != 1:
            continue
        target_i = int(candidates[0])
        if target_count[target_i] != 1:
            continue
        chord = float(np.linalg.norm(source[source_i] - target[target_i]))
        angle = 2.0 * np.arcsin(min(1.0, chord / 2.0))
        separation = float(np.rad2deg(angle) * 3600.0)
        if separation <= max_separation_arcsec:
            matches.append((source_i, target_i, separation))

    return {
        "matches": matches,
        "source_candidate_count": source_count,
        "target_candidate_count": target_count,
    }


def _read_tully_tables():
    def lines(path):
        with gzip.open(path, "rt", encoding="ascii") as stream:
            return [line.rstrip("\n") for line in stream]

    t3 = lines(ROOT / "data/tully2015_table3.dat.gz")
    t4 = lines(ROOT / "data/tully2015_table4.dat.gz")
    t5 = lines(ROOT / "data/tully2015_table5.dat.gz")
    if tuple(map(len, (t3, t4, t5))) != tuple(TULLY_ROWS.values()):
        raise ValueError("Tully 2015 table row counts differ from pinned ReadMe")

    summary = {}
    for line in t3:
        nest = parse_int(line[3:9], "Tully table3 Nest")
        nmb = parse_int(line[10:13], "Tully table3 Nmb")
        pgc1 = parse_int(line[14:21], "Tully table3 PGC1")
        if nest in summary:
            raise ValueError(f"duplicate Tully Nest {nest}")
        summary[nest] = {"Nmb": nmb, "PGC1": pgc1}

    table4_by_pgc = {}
    for line in t4:
        nest = parse_int(line[3:9], "Tully table4 Nest")
        pgc = parse_int(line[10:17], "Tully table4 PGC")
        if pgc in table4_by_pgc:
            raise ValueError(f"Tully table4 duplicates PGC {pgc}")
        table4_by_pgc[pgc] = nest

    members = []
    member_keys = set()
    for line in t5:
        row = {
            "PGC": parse_int(line[0:7], "Tully table5 PGC"),
            "GLON": parse_float(line[8:16], "Tully table5 GLON"),
            "GLAT": parse_float(line[17:25], "Tully table5 GLAT"),
            "Nest": parse_int(line[99:105], "Tully table5 Nest"),
            "Nmb": parse_int(line[106:109], "Tully table5 Nmb"),
            "PGC1": parse_int(line[110:117], "Tully table5 PGC1"),
        }
        key = (row["Nest"], row["PGC"])
        if key in member_keys:
            raise ValueError(f"duplicate Tully table5 membership {key}")
        member_keys.add(key)
        members.append(row)

    if len({row["PGC"] for row in members}) != len(members):
        raise ValueError("Tully table5 duplicates a PGC identity")
    table5_nest_counts = Counter()
    for row in members:
        nest = row["Nest"]
        table5_nest_counts[nest] += 1
        if nest == 0:
            continue
        if nest not in summary:
            raise ValueError(f"Tully table5 refers to unknown nonzero Nest {nest}")
        group = summary[nest]
        if row["PGC1"] != group["PGC1"] or row["Nmb"] != group["Nmb"]:
            raise ValueError(f"Tully table5 parent/count metadata mismatch for Nest {nest}")
    if set(summary) != (set(table5_nest_counts) - {0}):
        raise ValueError("Tully table3 and nonzero table5 Nest sets differ")
    if set(summary) != set(Counter(table4_by_pgc.values())):
        raise ValueError("Tully table3 and table4 Nest sets differ")
    return members, summary, table4_by_pgc


def _read_huchra_table3():
    records = []
    with gzip.open(ROOT / "data/2mrs_huchra2012_table3.dat.gz", "rt", encoding="ascii") as stream:
        for line in stream:
            records.append({
                "ID": line[0:16].strip(),
                "RAdeg": parse_float(line[17:26], "2MRS table3 RAdeg"),
                "DEdeg": parse_float(line[27:36], "2MRS table3 DEdeg"),
                "GLON": parse_float(line[37:46], "2MRS table3 GLON"),
                "GLAT": parse_float(line[47:56], "2MRS table3 GLAT"),
            })
    if len(records) != HUCHRA_ROWS:
        raise ValueError("Huchra 2MRS table3 row count differs from its ReadMe")
    ids = [row["ID"] for row in records]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("Huchra 2MRS table3 IDs are missing or duplicated")
    return records


def _read_2mpp_catalog():
    name_to_recno = {}
    real_positions = []
    all_recnos = set()
    zoa_fake_rows = 0
    with (ROOT / "data/2mpp_catalog.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        required = {"recno", "Name", "Ref", "_RA", "_DE"}
        if header is None or not required.issubset(header):
            raise ValueError("2M++ catalogue lacks recno/Name/Ref/_RA/_DE")
        column = {name: header.index(name) for name in required}
        row_count = 0
        for row in reader:
            row_count += 1
            recno = parse_int(row[column["recno"]], "2M++ recno")
            name = row[column["Name"]].strip()
            if recno in all_recnos or not name:
                raise ValueError("2M++ recno or Name is missing/duplicated")
            all_recnos.add(recno)
            if row[column["Ref"]].strip().lower() == "zoa":
                zoa_fake_rows += 1
                continue
            if name in name_to_recno:
                raise ValueError("duplicate real-galaxy 2M++ Name")
            name_to_recno[name] = recno
            real_positions.append((
                recno,
                parse_float(row[column["_RA"]], "2M++ _RA"),
                parse_float(row[column["_DE"]], "2M++ _DE"),
            ))
    if row_count != TWOMPP_ROWS:
        raise ValueError("2M++ row count changed")
    if len(real_positions) != 69160 or zoa_fake_rows != 3813:
        raise ValueError("2M++ real/ZoA row counts differ from the pinned ReadMe")
    return name_to_recno, real_positions, {
        "total_rows": row_count, "real_galaxy_rows": len(real_positions),
        "zoa_fake_rows": zoa_fake_rows,
    }


def summarize_overlap(members, nest_summary, table4_by_pgc, huchra, twompp_by_name,
                      twompp_positions, twompp_catalog_counts, point_role):
    crosswalk = crossmatch_coordinates(
        [row["GLON"] for row in members], [row["GLAT"] for row in members],
        [row["GLON"] for row in huchra], [row["GLAT"] for row in huchra])
    huchra_to_2mpp = crossmatch_coordinates(
        [row["RAdeg"] for row in huchra], [row["DEdeg"] for row in huchra],
        [row[1] for row in twompp_positions], [row[2] for row in twompp_positions])
    huchra_to_id = [row["ID"] for row in huchra]
    member_match = {source_i: (target_i, separation)
                    for source_i, target_i, separation in crosswalk["matches"]}
    huchra_position_match = {source_i: (target_i, separation)
                             for source_i, target_i, separation in huchra_to_2mpp["matches"]}

    per_nest = defaultdict(Counter)
    mapping_rows = []
    match_separations = []
    status_counts = Counter()
    matched_recno_to_pgc = {}
    exact_name_position_relation_counts = Counter()
    position_only_candidate_counts = Counter()
    position_only_candidate_roles = Counter()
    table5_by_pgc = {row["PGC"]: row for row in members}
    table4_counts = Counter(table4_by_pgc.values())
    table5_counts = Counter(row["Nest"] for row in members)
    association_disagreements = []
    for pgc in sorted(set(table4_by_pgc) | set(table5_by_pgc)):
        nest4 = table4_by_pgc.get(pgc)
        row5 = table5_by_pgc.get(pgc)
        nest5 = row5["Nest"] if row5 is not None else None
        if nest4 != nest5:
            association_disagreements.append({
                "Tully_PGC": pgc, "Tully_Table4_Nest": nest4,
                "Tully_Table5_Nest": nest5,
                "relation": ("table4_only_member" if row5 is None else
                             "table5_only_member" if nest4 is None else
                             "table5_nest0_but_table4_grouped" if nest5 == 0 else
                             "different_nonzero_nests"),
            })
    for index, member in enumerate(members):
        nest = member["Nest"]
        group = per_nest[nest]
        group["Tully_Table5_rows"] += 1
        candidates = int(crosswalk["source_candidate_count"][index])
        matched = member_match.get(index)
        huchra_id = ""
        separation = None
        recno = None
        role_name = ""
        positional_recno = None
        positional_separation = None
        positional_role_name = ""
        name_position_relation = "not_evaluated_no_2mrs_match"
        table4_nest = table4_by_pgc.get(member["PGC"])
        table3 = nest_summary.get(nest, {})
        if table4_nest is None:
            association_status = "table5_only_member"
        elif table4_nest == nest:
            association_status = "table4_table5_nest_agree"
        elif nest == 0:
            association_status = "table5_nest0_but_table4_grouped"
        else:
            association_status = "different_nonzero_nests"

        if candidates == 0:
            status = "no_2mrs_position_within_3arcsec"
            group["no_2mrs_position_within_3arcsec"] += 1
        elif matched is None:
            status = "ambiguous_or_nonreciprocal_position"
            group["ambiguous_or_nonreciprocal_position"] += 1
        else:
            h_index, separation = matched
            huchra_id = huchra_to_id[h_index]
            match_separations.append(separation)
            position_candidate_count = int(
                huchra_to_2mpp["source_candidate_count"][h_index])
            positional = huchra_position_match.get(h_index)
            if positional is not None:
                position_index, positional_separation = positional
                positional_recno = int(twompp_positions[position_index][0])
                positional_role = point_role.get(positional_recno)
                positional_role_name = (
                    {0: "training", 1: "heldout", 2: "buffer"}[positional_role]
                    if positional_role is not None else "outside_v6_parent")
            if huchra_id not in twompp_by_name:
                status = "2mrs_match_not_in_local_2mpp"
                group["2mrs_match_not_in_local_2mpp"] += 1
                if positional is not None:
                    name_position_relation = "position_only_candidate_not_promoted"
                    position_only_candidate_counts["unique_position_only_candidate"] += 1
                    position_only_candidate_roles[positional_role_name] += 1
                elif position_candidate_count > 1:
                    name_position_relation = "multiple_position_candidates_not_promoted"
                    position_only_candidate_counts["multiple_position_candidates"] += 1
                else:
                    name_position_relation = "no_position_candidate"
                    position_only_candidate_counts["no_position_candidate"] += 1
            else:
                recno = int(twompp_by_name[huchra_id])
                if positional_recno == recno:
                    name_position_relation = "exact_name_and_position_agree"
                elif positional_recno is None:
                    name_position_relation = "exact_name_without_unique_position_match"
                else:
                    name_position_relation = "exact_name_position_conflict"
                exact_name_position_relation_counts[name_position_relation] += 1
                if recno in matched_recno_to_pgc:
                    raise ValueError("two Tully members map to one 2M++ recno")
                matched_recno_to_pgc[recno] = member["PGC"]
                role = point_role.get(recno)
                if role is None:
                    status = "2mpp_member_outside_v6_parent"
                    group["2mpp_member_outside_v6_parent"] += 1
                else:
                    role_name = {0: "training", 1: "heldout", 2: "buffer"}[role]
                    status = f"2mpp_member_{role_name}"
                    group[status] += 1

        status_counts[status] += 1
        if recno is not None:
            group["matched_local_2mpp_total"] += 1
        mapping_rows.append({
            "Tully_PGC": member["PGC"], "Tully_Table5_Nest": nest,
            "Tully_Table4_Nest_by_PGC": table4_nest,
            "Tully_Table3_PGC1": table3.get("PGC1"),
            "Tully_Table3_Nmb": table3.get("Nmb"),
            "Tully_Table5_PGC1": member["PGC1"], "Tully_Table5_Nmb": member["Nmb"],
            "Tully_Table4_Table5_association_status": association_status,
            "Huchra_2MASS_ID": huchra_id, "separation_arcsec": separation,
            "2mpp_recno": recno, "v6_point_role": role_name,
            "2mpp_position_candidate_recno_not_identity": positional_recno,
            "2mpp_position_candidate_separation_arcsec": positional_separation,
            "2mpp_position_candidate_v6_role": positional_role_name,
            "2mpp_Name_position_relation": name_position_relation,
            "identity_status": status,
        })

    nest_rows = []
    complete, partial, none = 0, 0, 0
    for nest in sorted(set(nest_summary) | set(table4_counts) | set(table5_counts)):
        expected = nest_summary.get(nest, {})
        counts = per_nest[nest]
        local_matches = counts["matched_local_2mpp_total"]
        table5_rows = table5_counts[nest]
        if nest != 0 and table5_rows and local_matches == table5_rows:
            complete += 1
        elif nest != 0 and local_matches == 0:
            none += 1
        elif nest != 0:
            partial += 1
        nmb = expected.get("Nmb")
        nest_rows.append({
            "Tully_Nest": nest,
            "Tully_Table3_PGC1": expected.get("PGC1"),
            "Tully_Table3_Nmb": nmb,
            "Tully_Table4_member_rows": table4_counts[nest],
            "Tully_Table5_member_rows": table5_rows,
            "Table4_minus_Table3_Nmb": (table4_counts[nest] - nmb) if nmb is not None else None,
            "Table5_minus_Table3_Nmb": (table5_rows - nmb) if nmb is not None else None,
            **dict(sorted(counts.items())),
        })

    candidate_counts = Counter(map(int, crosswalk["source_candidate_count"]))
    target_candidate_counts = Counter(map(int, crosswalk["target_candidate_count"]))
    summary = {
        "Tully_member_rows": len(members), "Tully_Nests": len(nest_summary),
        "Huchra_2MRS_rows": len(huchra),
        "local_2Mpp_total_rows": twompp_catalog_counts["total_rows"],
        "local_2Mpp_real_galaxy_rows": twompp_catalog_counts["real_galaxy_rows"],
        "local_2Mpp_ZoA_fake_rows_excluded_from_identity_matching":
            twompp_catalog_counts["zoa_fake_rows"],
        "mutual_unique_position_matches_within_3arcsec": len(member_match),
        "no_Huchra_candidate_within_3arcsec": candidate_counts[0],
        "multiple_Huchra_candidates_within_3arcsec": sum(v for k, v in candidate_counts.items() if k > 1),
        "Tully_members_not_reciprocal_unique_matches": len(members) - len(member_match),
        "Huchra_rows_with_multiple_Tully_candidates_within_3arcsec": sum(
            v for k, v in target_candidate_counts.items() if k > 1),
        "position_match_separation_arcsec": {
            "n": len(match_separations),
            "median": float(np.median(match_separations)) if match_separations else None,
            "p90": float(np.percentile(match_separations, 90)) if match_separations else None,
            "maximum": float(np.max(match_separations)) if match_separations else None,
        },
        "member_identity_status_counts": dict(sorted(status_counts.items())),
        "exact_Name_position_relation_counts": dict(sorted(exact_name_position_relation_counts.items())),
        "name_unmatched_Huchra_rows_position_only_candidates_not_promoted":
            dict(sorted(position_only_candidate_counts.items())),
        "position_only_candidate_v6_role_counts_not_promoted":
            dict(sorted(position_only_candidate_roles.items())),
        "Huchra_to_2Mpp_mutual_unique_position_matches_within_3arcsec": len(huchra_position_match),
        "Huchra_to_2Mpp_no_position_candidate_within_3arcsec": int(
            np.count_nonzero(huchra_to_2mpp["source_candidate_count"] == 0)),
        "Huchra_to_2Mpp_multiple_position_candidates_within_3arcsec": int(
            np.count_nonzero(huchra_to_2mpp["source_candidate_count"] > 1)),
        "Tully_Nests_all_table5_rows_in_local_2mpp": complete,
        "Tully_Nests_partly_in_local_2mpp": partial,
        "Tully_Nests_with_no_local_2mpp_matches": none,
        "Tully_Table4_member_rows": sum(table4_counts.values()),
        "Tully_Table5_member_rows": sum(table5_counts.values()),
        "Tully_Table4_only_PGC_rows_without_Table5_position": sum(
            pgc not in table5_by_pgc for pgc in table4_by_pgc),
        "Tully_Table5_only_PGC_rows_without_Table4_member": sum(
            pgc not in table4_by_pgc for pgc in table5_by_pgc),
        "Tully_Table4_Table5_membership_association_disagreements": len(association_disagreements),
        "Tully_Table3_Nmb_mismatch_nests_vs_Table4_rows": sum(
            table4_counts[nest] != int(group["Nmb"]) for nest, group in nest_summary.items()),
        "Tully_Table3_Nmb_mismatch_nests_vs_Table5_rows": sum(
            table5_counts[nest] != int(group["Nmb"]) for nest, group in nest_summary.items()),
        "Tully_member_rows_in_local_2mpp_by_exact_Name": sum(
            counts["matched_local_2mpp_total"] for counts in per_nest.values()),
        "Tully_member_rows_in_v6_training": sum(
            counts["2mpp_member_training"] for counts in per_nest.values()),
        "Tully_member_rows_in_v6_heldout_identity_graph": sum(
            counts["2mpp_member_heldout"] for counts in per_nest.values()),
        "Tully_member_rows_in_v6_buffer_identity_graph": sum(
            counts["2mpp_member_buffer"] for counts in per_nest.values()),
        "Tully_member_rows_in_local_2mpp_but_outside_v6_parent": sum(
            counts["2mpp_member_outside_v6_parent"] for counts in per_nest.values()),
        "heldout_redshift_values_or_mark_scores_used": False,
        "CF4_observed_group_or_FP_values_used": False,
        "velocities_or_magnitudes_used": False,
        "field_state_or_likelihood_read": False,
    }
    return summary, nest_rows, mapping_rows, association_disagreements


def run_regressions():
    wrap = crossmatch_coordinates([359.9999], [12.0], [0.0001], [12.0])
    assert len(wrap["matches"]) == 1
    expected_arcsec = 0.72 * np.cos(np.deg2rad(12.0))
    assert abs(wrap["matches"][0][2] - expected_arcsec) < 1e-5

    collision = crossmatch_coordinates([10.0, 10.0002], [0.0, 0.0], [10.0001], [0.0])
    assert not collision["matches"]
    assert list(collision["source_candidate_count"]) == [1, 1]
    assert list(collision["target_candidate_count"]) == [2]

    vectors = unit_vectors([0.0, 360.0], [-90.0, -90.0])
    assert np.allclose(vectors[0], vectors[1], atol=1e-14)
    return {"tests": 3, "status": "PASS", "scope": "synthetic coordinate-geometry regressions"}


def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CF4_EXPECTED_COMMIT"):
        raise RuntimeError("Slurm job and pinned source commit are required")
    expected = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{os.environ['CF4_EXPECTED_COMMIT']}^{{commit}}"],
        cwd=ROOT, text=True).strip()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    if expected != actual:
        raise RuntimeError("source commit mismatch")
    if Path(os.environ["CF4_R2_OUT_DIR"]) != OUT or OUT.exists():
        raise FileExistsError("unexpected or already-used output path")

    started = time.monotonic()
    test_result = run_regressions()
    for relative, expected_hash in INPUT_HASHES.items():
        path = Path(relative) if Path(relative).is_absolute() else ROOT / relative
        if sha256(path) != expected_hash:
            raise ValueError(f"pinned source changed: {relative}")

    members, nest_summary, table4_by_pgc = _read_tully_tables()
    huchra = _read_huchra_table3()
    twompp_by_name, twompp_positions, twompp_catalog_counts = _read_2mpp_catalog()
    with np.load(SPLIT, allow_pickle=False) as split, np.load(POINTS, allow_pickle=False) as points:
        split_recno = split["point_recno"].astype(np.int64)
        point_role = split["point_role"].astype(np.int8)
        point_recno = points["recno"].astype(np.int64)
        if (len(split_recno) != len(point_role) or len(set(map(int, split_recno))) != len(split_recno)
                or set(map(int, split_recno)) != set(map(int, point_recno))):
            raise ValueError("v6 split roles and v6 point manifest do not align")
        if not set(map(int, point_role)).issubset({0, 1, 2}):
            raise ValueError("unknown v6 point role")
        role_by_recno = {int(recno): int(role) for recno, role in zip(split_recno, point_role)}

    summary, nest_rows, mapping_rows, association_disagreements = summarize_overlap(
        members, nest_summary, table4_by_pgc, huchra, twompp_by_name,
        twompp_positions, twompp_catalog_counts, role_by_recno)
    OUT.mkdir(parents=True, exist_ok=False)
    mapping_path = OUT / "tully_member_2mpp_identity.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    result = {
        "status": "TULLY_FULL_MEMBER_2MPP_IDENTITY_CROSSWALK_NOT_COVARIANCE_CALIBRATION",
        "job_id": os.environ["SLURM_JOB_ID"], "source_commit": actual,
        "method": ("mutual unique-neighbour angular match between Tully Table5 and Huchra 2MRS "
                   "within 3 arcsec, followed by exact 2MASS ID-to-real-2M++ Name join; a separate "
                   "3 arcsec Huchra-to-2M++ position crossmatch records alias candidates but never "
                   "promotes them to identity; published Tully Table4/Table5 associations remain separate"),
        "tests": test_result, "summary": summary, "nest_rows": nest_rows,
        "Tully_Table4_Table5_association_disagreements": association_disagreements,
        "mapping_path": str(mapping_path), "mapping_sha256": sha256(mapping_path),
        "input_sha256": {name: digest for name, digest in INPUT_HASHES.items()},
        "elapsed_seconds": time.monotonic() - started,
        "interpretation_limit": ("This is object/member identity and v6 split-role bookkeeping only. "
            "It does not establish identical spectra/redshift errors, Tully-to-CF4 group selection, "
            "physical membership probabilities, covariance, or a likelihood. Published Tully "
            "Table4/Table5 membership discrepancies are retained rather than reconciled implicitly. "
            "PGC33946 remains measurement/source-unresolved even if its object identity joins."),
        "posterior_promoted": False, "R2_complete": False, "new_gravity_runs": 0,
        "MW_M31": "roles remain ambiguous on the same NEW inferred field",
        "M33": "unresolved; eventual observables constrain the same NEW field at LG <=0.3 cMpc/h",
        "native_truth_ids_used_for_candidate_selection": False,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
