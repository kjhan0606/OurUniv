"""Classify raw 2M++ redshift sources for secure CF4 links in Tully Nests."""

import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import time

import numpy as np

from cf4_r2_v6_redshift_overlap import (
    FP, FP_SHA, IDENTITY_CENSUS, IDENTITY_CENSUS_SHA, POINTS, POINTS_SHA,
    SOURCE_HASHES, eligible_secure_edge, number, selected_rows,
)
from cf4_r2_tully2015_source_bridge import HASHES as TULLY_HASHES, integer, records


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4/z0_density")
PRIOR = BASE / "r2_v6_redshift_overlap_20261003_v3/result.json"
PRIOR_SHA = "47fd803bab67eb96094489a4c8d725314a43ac896e315a55bbcfbbf741d13da0"
MEMBERSHIP = BASE / "r2_v6_tully_membership_20261003_v1/result.json"
MEMBERSHIP_SHA = "10a53c2dd5d10487670bc898dbce0f73e496103dafd78859fe22b62508703594"
OUT = BASE / "r2_v6_tully_velocity_sources_20261003_v5"
TWOMRS_TABLE3 = ROOT / "data/2mrs_huchra2012_table3.dat.gz"
TWOMRS_TABLE3_SHA = "14a40e14dea131afbc2ff525e42b39fdc4094cf9d06d9a43952257eff80f1790"
EXPLICIT_2MRS_PREFIX = "20112MRS."


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "median": None, "p90": None, "maximum": None}
    return {"n": int(values.size), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "maximum": float(np.max(values))}


def summarize_source_rows(rows):
    by_reference = defaultdict(list)
    by_class = defaultdict(list)
    exact_equal = Counter()
    rounded_equal = Counter()
    for row in rows:
        reference = row["reference"] or "MISSING"
        source_class = ("missing_reference" if reference == "MISSING" else
                        "explicit_2MRS_source_code" if reference.startswith(EXPLICIT_2MRS_PREFIX)
                        else "other_or_unresolved_reference")
        by_reference[reference].append(row["abs_delta_km_s"])
        by_class[source_class].append(row["abs_delta_km_s"])
        exact_equal[reference] += row["abs_delta_km_s"] == 0.0
        rounded_equal[reference] += row["abs_delta_km_s"] <= 0.5
    return {
        "linked_member_count": len(rows),
        "reference_counts": {key: len(vals) for key, vals in sorted(by_reference.items())},
        "reference_class_counts": {key: len(vals) for key, vals in sorted(by_class.items())},
        "absolute_velocity_agreement_counts": {
            "exact_equal": sum(row["abs_delta_km_s"] == 0.0 for row in rows),
            "equal_within_0p5_km_s": sum(row["abs_delta_km_s"] <= 0.5 for row in rows),
        },
        "absolute_CF4_individual_minus_2mpp_point_Vcmb_km_s": describe(
            [row["abs_delta_km_s"] for row in rows]),
        "by_reference_class": {
            key: {"count": len(vals), "abs_delta_km_s": describe(vals),
                  "exact_equal_count": sum(value == 0.0 for value in vals),
                  "equal_within_0p5_km_s_count": sum(value <= 0.5 for value in vals)}
            for key, vals in sorted(by_class.items())
        },
        "by_reference": {
            key: {"count": len(vals), "abs_delta_km_s": describe(vals),
                  "exact_equal_count": exact_equal[key],
                  "equal_within_0p5_km_s_count": rounded_equal[key]}
            for key, vals in sorted(by_reference.items())
        },
        "explicit_2MRS_reference_codes": {
            key: len(vals) for key, vals in sorted(by_reference.items())
            if key.startswith(EXPLICIT_2MRS_PREFIX)
        },
    }


def parse_2mrs_table3(stream):
    """Read the official Huchra+2012 fixed-width ID and adopted-cz reference."""
    by_id = {}
    for line_number, raw in enumerate(stream, start=1):
        line = raw.rstrip(b"\r\n")
        if len(line) < 16:
            raise ValueError(f"short 2MRS table3 ID row {line_number}: {len(line)} bytes")
        # CDS omits trailing blank fixed-width fields, including the redshift
        # reference on rows without an adopted cz. Right-pad those fields.
        line = line.ljust(204, b" ")
        object_id = line[0:16].decode("ascii").strip()
        reference = line[185:204].decode("ascii").strip()
        if not object_id:
            raise ValueError(f"missing 2MRS ID on row {line_number}")
        if object_id in by_id:
            raise ValueError(f"duplicate 2MRS ID {object_id}")
        by_id[object_id] = reference
    return by_id


def summarize_2mrs_id_source_join(rows):
    status_counts = Counter()
    source_pairs = Counter()
    velocity_deltas = defaultdict(list)
    for row in rows:
        mpp_ref = row["2mpp_reference"] or "MISSING"
        mrs_ref = row["2mrs_reference"] or "MISSING"
        source_pairs[(mpp_ref, mrs_ref)] += 1
        if not row["2mrs_id_match"]:
            status = "not_in_2mrs_main_table"
        elif mpp_ref == "MISSING" or mrs_ref == "MISSING":
            status = "matched_id_reference_missing"
        elif mpp_ref == mrs_ref:
            status = "matched_id_same_reference_code"
        else:
            status = "matched_id_different_reference_code"
        status_counts[status] += 1
        if row.get("abs_velocity_delta_km_s") is not None:
            velocity_deltas[status].append(row["abs_velocity_delta_km_s"])
    return {
        "selected_tully_member_count": len(rows),
        "2mrs_main_table_id_match_count": sum(row["2mrs_id_match"] for row in rows),
        "source_join_status_counts": dict(sorted(status_counts.items())),
        "same_reference_code_count": status_counts["matched_id_same_reference_code"],
        "CF4_2mpp_abs_Vcmb_difference_km_s_by_source_status": {
            status: describe(values) for status, values in sorted(velocity_deltas.items())
        },
        "reference_code_pairs": [
            {"2mpp_Ref": left, "2mrs_r_cz": right, "count": count}
            for (left, right), count in sorted(source_pairs.items())
        ],
    }


def selected_tully_members(edges, table3, table4):
    parent_by_pgc1 = defaultdict(set)
    for line in table3:
        nest, pgc1 = integer(line, 3, 9), integer(line, 14, 21)
        if nest is not None and pgc1 is not None:
            parent_by_pgc1[str(pgc1)].add(nest)
    member_by_pgc = defaultdict(set)
    for line in table4:
        pgc, nest = integer(line, 10, 17), integer(line, 3, 9)
        if pgc is not None and nest is not None:
            member_by_pgc[pgc].add(nest)

    included, missing, mismatch, ambiguous = [], 0, 0, 0
    for pgc, one_pgc, recno, group in edges:
        member_nests = member_by_pgc.get(pgc, set())
        parent_nests = parent_by_pgc1.get(str(one_pgc).strip(), set())
        if not member_nests or not parent_nests:
            missing += 1
        elif len(member_nests) != 1 or len(parent_nests) != 1:
            ambiguous += 1
        elif member_nests != parent_nests:
            mismatch += 1
        else:
            included.append((pgc, one_pgc, recno))
    return included, {"missing": missing, "mismatch": mismatch, "ambiguous": ambiguous}


def summarize_tully_member_coverage(edges, selected, table3, table4):
    nmb_by_nest = {}
    parent_by_pgc1 = defaultdict(set)
    for line in table3:
        nest, nmb, pgc1 = integer(line, 3, 9), integer(line, 10, 13), integer(line, 14, 21)
        if nest is None or nmb is None or pgc1 is None:
            continue
        if nest in nmb_by_nest:
            raise ValueError("duplicate Tully Nest in group catalogue")
        nmb_by_nest[nest] = nmb
        parent_by_pgc1[str(pgc1)].add(nest)
    member_nests_by_pgc = defaultdict(set)
    for line in table4:
        nest, pgc = integer(line, 3, 9), integer(line, 10, 17)
        if nest is not None and pgc is not None:
            member_nests_by_pgc[pgc].add(nest)

    parent_nests = set()
    no_unique_parent = 0
    for _pgc, pgc1, _recno, _group in edges:
        candidates = parent_by_pgc1.get(str(pgc1).strip(), set())
        if len(candidates) == 1:
            parent_nests.update(candidates)
        else:
            no_unique_parent += 1
    linked_by_nest = Counter()
    for pgc, pgc1, _recno in selected:
        members = member_nests_by_pgc.get(pgc, set())
        parents = parent_by_pgc1.get(str(pgc1).strip(), set())
        if len(members) != 1 or len(parents) != 1 or members != parents:
            raise ValueError("selected member lacks a unique matching Tully parent Nest")
        linked_by_nest[next(iter(members))] += 1
    if not parent_nests.issuperset(linked_by_nest):
        raise ValueError("linked member Nest missing from parent-Nest cohort")

    fractions = []
    total_nmb = 0
    all_members_linked = 0
    for nest in sorted(parent_nests):
        if nest not in nmb_by_nest:
            raise ValueError(f"missing Tully member count for Nest {nest}")
        nmb, linked = nmb_by_nest[nest], linked_by_nest[nest]
        if linked > nmb:
            raise ValueError(f"linked member count exceeds Tully Nmb for Nest {nest}")
        total_nmb += nmb
        all_members_linked += linked == nmb
        fractions.append(linked / nmb)
    bins = Counter()
    for fraction in fractions:
        if fraction == 0:
            bins["zero"] += 1
        elif fraction <= 0.25:
            bins["(0,0.25]"] += 1
        elif fraction <= 0.5:
            bins["(0.25,0.5]"] += 1
        elif fraction < 1:
            bins["(0.5,1)"] += 1
        else:
            bins["1"] += 1
    linked_total = sum(linked_by_nest.values())
    return {
        "secure_v6_edge_count": len(edges),
        "edges_without_unique_parent_nest": no_unique_parent,
        "unique_Tully_parent_Nest_count": len(parent_nests),
        "sum_Tully_Nmb_over_parent_Nests": total_nmb,
        "secure_links_that_are_Tully_table4_members": linked_total,
        "aggregate_linked_fraction_of_Tully_Nmb": linked_total / total_nmb,
        "Nests_with_any_linked_Tully_members": sum(value > 0 for value in linked_by_nest.values()),
        "Nests_with_all_Tully_members_linked": all_members_linked,
        "per_Nest_linked_fraction": describe(fractions),
        "per_Nest_fraction_bins": dict(sorted(bins.items())),
    }


def main():
    if not os.environ.get("CF4_EXPECTED_COMMIT"):
        raise RuntimeError("pinned source commit is required")
    expected = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{os.environ['CF4_EXPECTED_COMMIT']}^{{commit}}"],
        cwd=ROOT, text=True).strip()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    if actual != expected:
        raise RuntimeError("source commit mismatch")
    if Path(os.environ["CF4_R2_OUT_DIR"]) != OUT or OUT.exists():
        raise FileExistsError("unexpected or already-used output path")
    started = time.monotonic()

    if (sha256(PRIOR) != PRIOR_SHA or sha256(MEMBERSHIP) != MEMBERSHIP_SHA
            or sha256(IDENTITY_CENSUS) != IDENTITY_CENSUS_SHA
            or sha256(FP) != FP_SHA or sha256(POINTS) != POINTS_SHA
            or sha256(TWOMRS_TABLE3) != TWOMRS_TABLE3_SHA):
        raise ValueError("frozen v6 input/result hash changed")
    for name in ("cf4_2mpp_crossmatch_v1.csv", "cf4_galaxies.csv", "2mpp_catalog.csv"):
        if sha256(ROOT / "data" / name) != SOURCE_HASHES[name]:
            raise ValueError(f"frozen source changed: {name}")
    for name in ("tully2015_table3.dat.gz", "tully2015_table4.dat.gz"):
        if sha256(ROOT / "data" / name) != TULLY_HASHES[name]:
            raise ValueError(f"frozen source changed: {name}")

    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    identity = json.loads(IDENTITY_CENSUS.read_text(encoding="utf-8"))
    membership = json.loads(MEMBERSHIP.read_text(encoding="utf-8"))
    if (prior.get("group_count") != 272 or prior.get("secure_member_pair_count") != 828
            or membership.get("group_count") != 272
            or membership.get("secure_member_pair_count") != 828
            or identity.get("eligible_training_group_count") != 272):
        raise ValueError("frozen v6 272/828 cohort changed")

    labels = {row["source_group_label"] for row in prior["groups"]}
    with np.load(FP, allow_pickle=False) as data:
        pgcs = data["PGC"].astype(np.int64)
        source_groups = data["source_group"].astype(str)
    pgc_group = {}
    for pgc, label in zip(pgcs, source_groups):
        if label in labels:
            if int(pgc) in pgc_group and pgc_group[int(pgc)] != label:
                raise ValueError("selected PGC belongs to multiple FP source groups")
            pgc_group[int(pgc)] = label
    with np.load(POINTS, allow_pickle=False) as data:
        point_recnos = set(map(int, data["recno"]))

    crossmatch = ROOT / "data/cf4_2mpp_crossmatch_v1.csv"
    raw_edges = selected_rows(crossmatch, ("PGC", "1PGC", "twompp_recno", "match_class"),
                              lambda row: eligible_secure_edge(row, pgc_group, point_recnos))
    edges = [(int(row["PGC"]), row["1PGC"], int(row["twompp_recno"]),
              pgc_group[int(row["PGC"])]) for row in raw_edges]
    if len(edges) != 828:
        raise ValueError("frozen v6 secure edge count changed")

    table3 = records("tully2015_table3.dat.gz")
    table4 = records("tully2015_table4.dat.gz")
    if (len(table3), len(table4)) != (25474, 43038):
        raise ValueError("Tully source row counts changed")
    selected, membership_status = selected_tully_members(edges, table3, table4)
    if (len(selected) != 441 or membership_status !=
            {"missing": 387, "mismatch": 0, "ambiguous": 0}):
        raise ValueError("Tully member cohort differs from committed crosswalk")
    member_coverage = summarize_tully_member_coverage(edges, selected, table3, table4)

    recno_to_pgc = {recno: pgc for pgc, _one_pgc, recno in selected}
    if len(recno_to_pgc) != 441:
        raise ValueError("one 2M++ row maps to multiple selected PGCs")
    selected_recnos = set(recno_to_pgc)
    mpp_rows = selected_rows(ROOT / "data/2mpp_catalog.csv", ("recno", "Name", "Vcmb", "Ref"),
                             lambda row: int(row["recno"]) in selected_recnos)
    mpp_by_recno = {int(row["recno"]): row for row in mpp_rows}
    if len(mpp_by_recno) != 441 or set(mpp_by_recno) != selected_recnos:
        raise ValueError("Tully-member 2M++ source rows are missing or duplicated")

    selected_pgcs = {pgc for pgc, _one_pgc, _recno in selected}
    expected_cf4_group = {pgc: str(one_pgc).strip() for pgc, one_pgc, _recno in selected}
    cf4_rows = selected_rows(ROOT / "data/cf4_galaxies.csv", ("PGC", "1PGC", "Vcmb"),
                             lambda row: row["PGC"].strip()
                             and int(row["PGC"]) in selected_pgcs)
    by_pgc = defaultdict(list)
    for row in cf4_rows:
        if row["1PGC"].strip() != expected_cf4_group[int(row["PGC"])]:
            raise ValueError("CF4 individual row has a different 1PGC than the frozen link")
        if number(row["Vcmb"]) is not None:
            by_pgc[int(row["PGC"])].append(number(row["Vcmb"]))
    if set(by_pgc) != selected_pgcs or any(len(set(vals)) != 1 for vals in by_pgc.values()):
        raise ValueError("selected CF4 individual raw Vcmb is absent or ambiguous")

    rows = []
    for pgc, _one_pgc, recno in selected:
        point = mpp_by_recno[recno]
        cf4_v, mpp_v = by_pgc[pgc][0], number(point["Vcmb"])
        if mpp_v is None:
            raise ValueError("selected 2M++ raw Vcmb is nonfinite")
        rows.append({"pgc": pgc, "2mpp_name": point["Name"].strip(),
                     "reference": point["Ref"].strip(),
                     "abs_delta_km_s": abs(cf4_v - mpp_v)})

    if len({row["2mpp_name"] for row in rows}) != len(rows):
        raise ValueError("selected Tully members have duplicate 2M++ names")
    with gzip.open(TWOMRS_TABLE3, "rb") as stream:
        twomrs_by_id = parse_2mrs_table3(stream)
    if len(twomrs_by_id) != 44599:
        raise ValueError("official 2MRS main-table row count changed")
    source_join_rows = [{
        "pgc": row["pgc"], "2mpp_name": row["2mpp_name"],
        "2mpp_reference": row["reference"],
        "2mrs_id_match": row["2mpp_name"] in twomrs_by_id,
        "2mrs_reference": twomrs_by_id.get(row["2mpp_name"], ""),
        "abs_velocity_delta_km_s": row["abs_delta_km_s"],
    } for row in rows]

    report = {
        "status": "V6_TULLY_MEMBER_RAW_VELOCITY_SOURCE_CROSSWALK_NOT_COVARIANCE",
        "job_id": os.environ.get("SLURM_JOB_ID"), "source_commit": actual,
        "execution_mode": ("slurm" if os.environ.get("SLURM_JOB_ID")
                           else "bounded_local_metadata_join"),
        "v6_group_count": 272, "secure_member_pair_count": 828,
        "Tully_member_link_count": len(selected),
        "Tully_member_absent_from_archived_table4_count": membership_status["missing"],
        "Tully_membership_mismatch_count": membership_status["mismatch"],
        "Tully_membership_ambiguous_count": membership_status["ambiguous"],
        "2mpp_velocity_reference_semantics": (
            "Ref is the source bibcode. Only explicit 20112MRS.* codes are classified as 2MRS; "
            "other literature codes remain unresolved because a compiled reference need not identify "
            "whether that redshift was also incorporated into 2MRS."
        ),
        "velocity_comparison_semantics": (
            "Compares only raw individual CMB-frame Vcmb in CF4 and 2M++; "
            "does not use group Vcmb or Tully adjusted Vcmba."
        ),
        "source_summary": summarize_source_rows(rows),
        "2mrs_main_catalog_source_join": summarize_2mrs_id_source_join(source_join_rows),
        "Tully_parent_Nest_member_coverage": member_coverage,
        "input_sha256": {
            "prior_redshift_overlap": PRIOR_SHA,
            "prior_tully_membership": MEMBERSHIP_SHA,
            "identity_census": IDENTITY_CENSUS_SHA,
            "fp_observations": FP_SHA,
            "count_point_manifest": POINTS_SHA,
            "2mrs_huchra2012_table3.dat.gz": TWOMRS_TABLE3_SHA,
            **{name: SOURCE_HASHES[name] for name in (
                "cf4_2mpp_crossmatch_v1.csv", "cf4_galaxies.csv", "2mpp_catalog.csv")},
            **{name: TULLY_HASHES[name] for name in (
                "tully2015_table3.dat.gz", "tully2015_table4.dat.gz")},
        },
        "Tully_adjusted_Vcmba_read": False,
        "heldout_values_read": False, "likelihood_or_field_read": False,
        "PM_evolutions": 0, "posterior_promoted": False, "R2_complete": False,
        "limitation": (
            "An exact 2MASS ID and matching 2M++ Ref/2MRS r_cz bibcode identify the same "
            "catalogued object and cited publication, not proof of an identical spectrum or "
            "measurement. The 2MRS cz and 2M++ Vcmb frames are not numerically compared here; "
            "no group-mean covariance law is inferred and no adjusted Tully Vcmba is compared."
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
