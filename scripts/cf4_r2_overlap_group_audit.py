"""Bounded source-level CF4/2M++ overlap audit; no likelihood or candidate fit.

Uses only the three frozen CSVs, the frozen N128 row list and native CF4
group list.  It never changes a catalogue or interprets a matched galaxy as
an independent CF4 density observation.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRACER = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json"
GROUPS = ROOT / "data/cf4_groups.csv"
GROUPS_SHA256 = "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334"
ROWS = Path("/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1/rows.npz")
NATIVE = Path("/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz")
OLD_BUILDER = ROOT / "config/cf4_datum_bearing_z0_twompp_datum_builder_program_v2.json"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bound(path, entry):
    raw = path.read_bytes()
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"frozen source changed: {path}")


def quantiles(values):
    if not values:
        return None
    return {"median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)), "maximum": float(np.max(values))}


def audit():
    tracer = json.loads(TRACER.read_text())
    match_path = ROOT / tracer["inputs"]["cf4_twompp_crossmatch"]["path"]
    twompp_path = ROOT / tracer["inputs"]["twompp_catalog"]["path"]
    bound(match_path, tracer["inputs"]["cf4_twompp_crossmatch"])
    bound(twompp_path, tracer["inputs"]["twompp_catalog"])
    if hashlib.sha256(GROUPS.read_bytes()).hexdigest() != GROUPS_SHA256:
        raise ValueError("frozen CF4 group source changed")
    groups = {row["1PGC"]: row for row in read_csv(GROUPS)}
    twompp = {int(row["recno"]): row for row in read_csv(twompp_path)}
    with np.load(ROWS, allow_pickle=False) as saved:
        eligible = set(map(int, saved["recno"]))
        calibration = set(map(int, saved["recno"][saved["calibration"]]))
        survivors = set(map(int, saved["recno"][saved["survives"]]))
    with np.load(NATIVE, allow_pickle=False) as saved:
        native_pgc = set(map(int, saved["CF4_pgc"]))
    if len(groups) != 38053 or len(twompp) != 72973:
        raise ValueError("source row counts changed")

    by_class = defaultdict(set)
    group_rows = defaultdict(list)
    deltas = []
    group_deltas = []
    group_sizes = []
    native_secure_targets = set()
    missing_groups = 0
    for row in read_csv(match_path):
        kind = row["match_class"]
        if kind == "unmatched":
            continue
        target = int(row["twompp_recno"])
        if target not in twompp:
            raise ValueError("crossmatch target missing from 2M++")
        by_class[kind].add(target)
        if target not in eligible or kind != "secure_joint_mark":
            continue
        group = groups.get(row["1PGC"])
        if group is None:
            missing_groups += 1
            continue
        group_rows[row["1PGC"]].append(target)
        group_sizes.append(int(group["Ngal"]))
        deltas.append(abs(float(row["delta_vcmb_kms"])))
        group_deltas.append(abs(float(twompp[target]["Vcmb"]) - float(group["V3k"])))
        if int(row["1PGC"]) in native_pgc:
            native_secure_targets.add(target)
    all_matched = set().union(*by_class.values())
    old_binding = json.loads(OLD_BUILDER.read_text())["bindings"]["excluded_recnos"]
    old_path = Path(old_binding["path"])
    bound(old_path, old_binding)
    old_excluded = {int(row["recno"]) for row in read_csv(old_path)}
    if len(old_excluded) != 319:
        raise ValueError("old metadata exclusion count changed")
    residual_failures = calibration - survivors - all_matched
    eligible_class = {key: len(values & eligible) for key, values in by_class.items()}
    report = {
        "classification": "R2_OVERLAP_GROUP_SOURCE_AUDIT_NOT_LIKELIHOOD",
        "frozen_crossmatch_sha256": tracer["inputs"]["cf4_twompp_crossmatch"]["sha256"],
        "frozen_CF4_groups_sha256": GROUPS_SHA256,
        "source_rows": {"CF4_groups": len(groups), "2Mpp": len(twompp),
                        "N128_eligible_parent": len(eligible)},
        "full_catalogue_unique_targets_by_class": {key: len(value) for key, value in by_class.items()},
        "full_catalogue_all_nonunmatched_unique_targets": len(all_matched),
        "N128_eligible_unique_targets_by_class": eligible_class,
        "N128_eligible_all_nonunmatched_unique_targets": len(eligible & all_matched),
        "N128_calibration_all_nonunmatched_unique_targets": len(calibration & all_matched),
        "N128_calibration_noncrossmatch_failures": len(residual_failures),
        "N128_calibration_noncrossmatch_failures_in_prior_metadata_exclusions": len(residual_failures & old_excluded),
        "N128_calibration_noncrossmatch_failures_new_map_rule_only": len(residual_failures - old_excluded),
        "N128_eligible_secure_distinct_groups_with_canonical_row": len(group_rows),
        "N128_eligible_secure_targets_missing_canonical_group": missing_groups,
        "N128_eligible_secure_targets_in_native_CF4_groups": len(native_secure_targets),
        "N128_eligible_secure_group_membership_gt1": sum(n > 1 for n in group_sizes),
        "N128_eligible_secure_groups_with_multiple_matched_members": sum(len(v) > 1 for v in group_rows.values()),
        "N128_eligible_secure_abs_CF4_individual_minus_2Mpp_Vcmb_km_s": quantiles(deltas),
        "N128_eligible_secure_abs_CF4_group_V3k_minus_2Mpp_Vcmb_km_s": quantiles(group_deltas),
        "interpretation": "CF4 grouped V3k and individual 2M++ Vcmb are distinct catalogue fields. Differences and multiple matched members must be modeled or bounded before inclusive counts can share a CF4 factor. No count or CF4 datum is promoted here.",
    }
    if len(all_matched) != 17007 or len(eligible & all_matched) < 1:
        raise ValueError("frozen overlap totals are inconsistent")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
