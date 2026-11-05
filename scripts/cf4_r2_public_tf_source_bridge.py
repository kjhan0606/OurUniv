"""Read-only PGC bridge from the public 2020 CF4 TF source to R2 marks."""

import csv
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/gpfs/kjhan/CF4/source_catalogues/J_ApJ_902_145")
BASE = Path("/gpfs/kjhan/CF4/z0_density")
CF4 = ROOT / "data/cf4_galaxies.csv"
BRIDGE = BASE / "r2_tf_matched_point_bridge_v1/tf_groups_linked.npz"
EDGES = BASE / "r2_point_mark_manifest_v1/edges.npz"
OUT = BASE / "r2_public_tf_source_bridge_v1"
EXPECTED = {
    "ReadMe": "45fa3cca5346aff29bcf9785e7b08e8ab34b1d67e80b359c0d28bdbd18e42581",
    "table1.dat.gz": "59a16536ff62aa63729f1b1a6fc11155b0f63613e80ebd8c3469966385a64b0c",
    "table4.dat.gz": "7199960877ccd38b4f499b342e9732951d32e7bc643bb90a08211ab524de7246",
}


def fixed_catalogue(name, width, expected_rows, value_column=None):
    rows = {}
    with gzip.open(SOURCE / name, "rt", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            record = line.rstrip("\r\n")
            # CDS table4 omits trailing empty columns on some records; its
            # ReadMe length is a maximum, unlike the always-full table1.
            minimum = width if value_column is None else 7
            if not minimum <= len(record) <= width:
                raise ValueError(f"{name}:{line_number}: unexpected record width")
            pgc = int(record[:7])
            if pgc in rows:
                raise ValueError(f"{name}: duplicate PGC {pgc}")
            rows[pgc] = (float(record[value_column].strip())
                         if value_column and record[value_column].strip() else None)
    if len(rows) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows}, got {len(rows)}")
    return rows


def percentile(values):
    return ([float(x) for x in np.percentile(values, (50, 90, 99))]
            if values else None)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    hashes = {}
    for name, expected in EXPECTED.items():
        digest = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"public source hash changed: {name}")
        hashes[name] = digest
    table1 = fixed_catalogue("table1.dat.gz", 499, 10737)
    table4 = fixed_catalogue("table4.dat.gz", 100, 9792, slice(26, 31))

    cf4_by_recno = {}
    tf_members = defaultdict(set)
    with CF4.open(newline="") as stream:
        for row in csv.DictReader(stream):
            recno = int(row["recno"])
            if recno in cf4_by_recno:
                raise ValueError("duplicate CF4 recno")
            cf4_by_recno[recno] = row
            if row["1PGC"] and row["DMtf"]:
                tf_members[int(row["1PGC"])].add(int(row["PGC"]))

    with np.load(BRIDGE, allow_pickle=False) as saved:
        group = saved["group_pgc"].astype(np.int64)
        point = saved["secure_point_index"].astype(np.int64)
        holdout = saved["holdout"].astype(bool)
        ngal = saved["source_Ngal"].astype(np.int64)
    if len(group) != 8502 or len(set(group.tolist())) != len(group):
        raise ValueError("TF source group set changed")
    with np.load(EDGES, allow_pickle=False) as saved:
        edge_group = saved["group_1pgc"].astype(np.int64)
        edge_point = saved["point_index"].astype(np.int64)
        edge_recno = saved["cf4_recno"].astype(np.int64)
        edge_class = saved["match_class_code"].astype(np.int64)
    secure_edge = defaultdict(list)
    for g, p, r, c in zip(edge_group, edge_point, edge_recno, edge_class, strict=True):
        if c == 1:
            secure_edge[(int(g), int(p))].append(int(r))

    all_group_coverage = Counter()
    matched = {"train": Counter(), "holdout": Counter()}
    dm_delta = {"train": [], "holdout": []}
    missing_examples = []
    for g, p, h, n in zip(group, point, holdout, ngal, strict=True):
        members = tf_members.get(int(g), set())
        if members:
            all_group_coverage["has_cf4_tf_member"] += 1
        if any(m in table1 for m in members):
            all_group_coverage["has_2020_raw_tf_member"] += 1
        if any(m in table4 for m in members):
            all_group_coverage["has_2020_distance_member"] += 1
        if p < 0:
            continue
        if n != 1:
            raise ValueError("matched TF bridge contains a non-singleton")
        split = "holdout" if h else "train"
        count = matched[split]
        count["secure_matched_groups"] += 1
        rows = secure_edge[(int(g), int(p))]
        if len(rows) != 1:
            raise ValueError("matched group is not uniquely bound to a secure edge")
        source_row = cf4_by_recno.get(rows[0])
        if source_row is None or int(source_row["1PGC"]) != int(g):
            count["missing_or_wrong_cf4_edge_recno"] += 1
            continue
        pgc = int(source_row["PGC"])
        if pgc not in members:
            count["edge_member_lacks_cf4_tf_mark"] += 1
        if pgc in table1:
            count["edge_member_in_2020_raw_tf"] += 1
        elif len(missing_examples) < 8:
            missing_examples.append({"group_1pgc": int(g), "cf4_member_pgc": pgc,
                                     "split": split})
        if pgc in table4:
            count["edge_member_in_2020_distance"] += 1
            if source_row["DMtf"] and table4[pgc] is not None:
                delta = float(source_row["DMtf"]) - table4[pgc]
                if math.isfinite(delta):
                    dm_delta[split].append(abs(delta))
                    count["comparable_individual_dm"] += 1

    report = {
        "classification": "PUBLIC_TF_SOURCE_PGC_OVERLAP_ONLY",
        "job_id": os.environ["SLURM_JOB_ID"],
        "source": "CDS J/ApJ/902/145 Kourkchi+2020, tables 1 and 4",
        "public_source_sha256": hashes,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (CF4, BRIDGE, EDGES)},
        "source_rows": {"table1_raw": len(table1), "table4_distances": len(table4)},
        "tf_only_groups": int(len(group)),
        "all_tf_only_group_coverage": dict(all_group_coverage),
        "secure_matched_singleton": {k: dict(v) for k, v in matched.items()},
        "abs_cf4_member_DMtf_minus_2020_DMbest_mag_p50_p90_p99":
            {k: percentile(v) for k, v in dm_delta.items()},
        "missing_raw_member_examples_first_eight": missing_examples,
        "heldout_used_for_fit": False,
        "R2_posterior": False,
        "limitations": [
            "PGC overlap is identity evidence, not a calibrated TF selection or group-inclusion denominator.",
            "The 2020 individual TF catalogue may not reproduce the 2023 heterogeneous grouped DMtf value or error.",
            "Individual raw-TF likelihood would replace, not multiply, an overlapping group-DMtf factor.",
            "No FP or cross-method source covariance is inferred here.",
        ],
    }
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
