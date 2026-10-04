"""Summarize the official SDSS-PV mock ensemble for R2 error calibration.

This is a training-side observation-law diagnostic only. It does not fit a
correction, score held-out data, construct a density field, or run gravity.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path


EXPECTED_BYTES = 10_643_218_721
EXPECTED_MD5 = "9ba3e8876f6f08a2af00d30cbf1c6cd9"
ARCHIVE_URL = "https://zenodo.org/records/6824749/files/mocks.tar.gz"
REQUIRED = {
    "ID", "cen_flag", "parenthalomass", "subhalomass", "z_obs",
    "z_obs_cen", "vxcen", "vycen", "vzcen", "vx", "vy", "vz",
    "r", "er", "s", "es", "i", "ei", "logdist_true", "logdist",
    "logdist_err", "logdist_alpha",
}
HOST_KEY = ("parenthalomass", "z_obs_cen", "vxcen", "vycen", "vzcen")
RICHNESS_BINS = ((1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, math.inf, "10+"))
MOCK_NAME = re.compile(r"^mocks/MOCK_HAMHOD_SDSS_v5_R(\d+)\.([0-7])_err_corr$")


class DigestReader:
    """Non-seeking wrapper used to checksum bytes as the tar stream consumes them."""

    def __init__(self, handle):
        self.handle = handle
        self.md5 = hashlib.md5()
        self.bytes_read = 0

    def read(self, size=-1):
        block = self.handle.read(size)
        self.md5.update(block)
        self.bytes_read += len(block)
        return block

    def close(self):
        # tarfile closes its stream wrapper; the caller owns the underlying file.
        pass


class _ReadableMember(io.RawIOBase):
    """Adapt tarfile's non-seekable member stream for csv/TextIOWrapper."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.source.read(len(buffer))
        if not block:
            return 0
        buffer[:len(block)] = block
        return len(block)


def richness_bin(n):
    for low, high, label in RICHNESS_BINS:
        if low <= n <= high:
            return label
    raise ValueError(f"invalid mock host richness: {n}")


def _new_bin():
    return dict(
        galaxies=0, groups=0, z_sum=0.0, z2_sum=0.0,
        cover68=0, cover95=0, alpha_sum=0.0,
        group_centered_sq=0.0, expected_group_var=0.0,
    )


def _finish_bin(value):
    n = value["galaxies"]
    ng = value["groups"]
    if n == 0:
        return {"galaxies": 0, "groups": 0}
    mean = value["z_sum"] / n
    return {
        "galaxies": n,
        "groups": ng,
        "standardized_residual_mean": mean,
        "standardized_residual_sd": math.sqrt(max(0.0, value["z2_sum"] / n - mean * mean)),
        "coverage_abs_z_le_1": value["cover68"] / n,
        "coverage_abs_z_le_1_96": value["cover95"] / n,
        "mean_skew_alpha": value["alpha_sum"] / n,
        # Ratio near one is the independent reported-error reference. This is
        # a descriptive group-mean diagnostic, not an adopted inflation factor.
        "group_mean_variance_ratio_vs_independent": (
            value["group_centered_sq"] / value["expected_group_var"]
            if value["expected_group_var"] > 0 else None
        ),
    }


def summarize_catalog(name, member_file):
    match = MOCK_NAME.match(name)
    if not match:
        raise ValueError(f"unexpected mock member name: {name}")
    simulation, observer = match.group(1), int(match.group(2))

    text = io.TextIOWrapper(io.BufferedReader(_ReadableMember(member_file)),
                            encoding="utf-8", newline="")
    reader = csv.DictReader(text)
    if reader.fieldnames is None or not REQUIRED.issubset(reader.fieldnames):
        missing = sorted(REQUIRED - set(reader.fieldnames or ()))
        raise ValueError(f"{name}: missing required columns {missing}")

    groups = {}
    id_to_key = {}
    key_to_ids = collections.defaultdict(set)
    rows = 0
    residual_sum = 0.0
    central_rows = 0

    for row in reader:
        rows += 1
        try:
            gid = row["ID"]
            key = tuple(float(row[c]) for c in HOST_KEY)
            observed = float(row["logdist"])
            truth = float(row["logdist_true"])
            sigma = float(row["logdist_err"])
            alpha = float(row["logdist_alpha"])
            z_obs = float(row["z_obs"])
            fp_marks = tuple(float(row[c]) for c in ("r", "er", "s", "es", "i", "ei"))
            central = int(row["cen_flag"])
            parent_mass = float(row["parenthalomass"])
            subhalo_mass = float(row["subhalomass"])
            center_v = tuple(float(row[c]) for c in ("vxcen", "vycen", "vzcen"))
            galaxy_v = tuple(float(row[c]) for c in ("vx", "vy", "vz"))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"{name}: malformed row {rows}") from exc
        vals = (observed, truth, sigma, alpha, z_obs, parent_mass, subhalo_mass,
                *fp_marks, *key, *center_v, *galaxy_v)
        if (not gid or not all(math.isfinite(x) for x in vals) or sigma <= 0
                or min(fp_marks[1], fp_marks[3], fp_marks[5]) <= 0 or central not in (0, 1)):
            raise ValueError(f"{name}: nonfinite/invalid data at row {rows}")

        previous = id_to_key.setdefault(gid, key)
        if previous != key:
            raise ValueError(f"{name}: repeated ID maps to multiple host keys: {gid}")
        key_to_ids[key].add(gid)
        group = groups.setdefault(key, {"members": [], "central_count": 0, "ids": set()})
        group["ids"].add(gid)
        group["central_count"] += central
        if central:
            central_rows += 1
            if not math.isclose(parent_mass, subhalo_mass, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"{name}: central/parent halo mass mismatch at row {rows}")
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-10) for a, b in zip(center_v, galaxy_v)):
                raise ValueError(f"{name}: central/parent velocity mismatch at row {rows}")
        residual = observed - truth
        group["members"].append((residual, sigma, alpha))
        residual_sum += residual

    # Closing the adapter does not close the tar stream itself.
    text.close()
    if rows == 0:
        raise ValueError(f"{name}: empty mock catalogue")
    if any(v["central_count"] > 1 for v in groups.values()):
        raise ValueError(f"{name}: a host key has multiple selected centrals")
    if any(len(ids) != 1 for ids in key_to_ids.values()):
        raise ValueError(f"{name}: host key collision across IDs")

    global_mean = residual_sum / rows
    bins = {label: _new_bin() for _, _, label in RICHNESS_BINS}
    no_central_groups = 0
    for group in groups.values():
        members = group["members"]
        n = len(members)
        b = bins[richness_bin(n)]
        b["groups"] += 1
        if group["central_count"] == 0:
            no_central_groups += 1
        group_mean = sum(v[0] for v in members) / n
        expected_var = sum(v[1] * v[1] for v in members) / (n * n)
        b["group_centered_sq"] += (group_mean - global_mean) ** 2
        b["expected_group_var"] += expected_var
        for residual, sigma, alpha in members:
            z = residual / sigma
            b["galaxies"] += 1
            b["z_sum"] += z
            b["z2_sum"] += z * z
            b["cover68"] += int(abs(z) <= 1.0)
            b["cover95"] += int(abs(z) <= 1.96)
            b["alpha_sum"] += alpha

    return {
        "member": name,
        "simulation": simulation,
        "observer": observer,
        "galaxies": rows,
        "host_groups": len(groups),
        "unique_repeated_IDs": len(id_to_key),
        "groups_without_selected_central": no_central_groups,
        "central_galaxies": central_rows,
        "catalogue_mean_logdist_minus_truth": global_mean,
        "bins": {k: _finish_bin(v) | {
            "_group_centered_sq": v["group_centered_sq"],
            "_expected_group_var": v["expected_group_var"],
            "_z_sum": v["z_sum"], "_z2_sum": v["z2_sum"],
            "_cover68": v["cover68"], "_cover95": v["cover95"],
            "_alpha_sum": v["alpha_sum"],
        } for k, v in bins.items()},
    }


def _quantiles(values):
    values = sorted(values)
    if not values:
        return None
    def at(q):
        pos = q * (len(values) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        return values[lo] * (1 - frac) + values[hi] * frac
    return {"q05": at(0.05), "q50": at(0.50), "q95": at(0.95)}


def summarize_archive(archive_path):
    archive_path = Path(archive_path)
    archive_bytes = archive_path.stat().st_size
    if archive_bytes != EXPECTED_BYTES:
        raise ValueError(f"archive size mismatch: {archive_bytes}")
    catalogues = []
    digest_reader = None
    with archive_path.open("rb") as raw:
        digest_reader = DigestReader(raw)
        with tarfile.open(fileobj=digest_reader, mode="r|gz") as tar:
            for member in tar:
                if member.isdir():
                    continue
                if not member.isfile() or not member.name.startswith("mocks/"):
                    raise ValueError(f"unexpected tar member: {member.name}")
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    raise ValueError(f"cannot read mock member: {member.name}")
                catalogues.append(summarize_catalog(member.name, fileobj))
                if len(catalogues) % 128 == 0:
                    print(f"validated_mock_catalogues={len(catalogues)}/2048", flush=True)
        while digest_reader.read(8 * 1024 * 1024):
            pass
    if digest_reader.bytes_read != EXPECTED_BYTES or digest_reader.md5.hexdigest() != EXPECTED_MD5:
        raise ValueError(
            f"archive integrity mismatch: bytes={digest_reader.bytes_read}, md5={digest_reader.md5.hexdigest()}"
        )
    if len(catalogues) != 2048:
        raise ValueError(f"expected2048 mock catalogues, found{len(catalogues)}")
    by_simulation = collections.defaultdict(list)
    for item in catalogues:
        by_simulation[item["simulation"]].append(item)
    if len(by_simulation) != 256 or any(sorted(x["observer"] for x in v) != list(range(8)) for v in by_simulation.values()):
        raise ValueError("expected8 distinct observers in each of256 simulation boxes")

    bins = {}
    for _, _, label in RICHNESS_BINS:
        items = [c["bins"][label] for c in catalogues]
        galaxies = sum(x["galaxies"] for x in items)
        zsum = sum(x["_z_sum"] for x in items)
        z2sum = sum(x["_z2_sum"] for x in items)
        num = sum(x["_group_centered_sq"] for x in items)
        den = sum(x["_expected_group_var"] for x in items)
        box_ratio = []
        for sample in by_simulation.values():
            bitems = [c["bins"][label] for c in sample]
            bnum = sum(x["_group_centered_sq"] for x in bitems)
            bden = sum(x["_expected_group_var"] for x in bitems)
            if bden > 0:
                box_ratio.append(bnum / bden)
        mean = zsum / galaxies if galaxies else None
        bins[label] = {
            "galaxies": galaxies,
            "host_groups": sum(x["groups"] for x in items),
            "standardized_residual_mean": mean,
            "standardized_residual_sd": math.sqrt(max(0.0, z2sum / galaxies - mean * mean)) if galaxies else None,
            "coverage_abs_z_le_1": sum(x["_cover68"] for x in items) / galaxies if galaxies else None,
            "coverage_abs_z_le_1_96": sum(x["_cover95"] for x in items) / galaxies if galaxies else None,
            "mean_skew_alpha": sum(x["_alpha_sum"] for x in items) / galaxies if galaxies else None,
            "group_mean_variance_ratio_vs_independent": num / den if den else None,
            "box_level_group_mean_variance_ratio_q05_q50_q95": _quantiles(box_ratio),
        }
    box_offsets = [
        sum(c["catalogue_mean_logdist_minus_truth"] for c in sample) / len(sample)
        for sample in by_simulation.values()
    ]
    return {
        "classification": "SDSS_PV_MOCK_FP_ERROR_AND_HOST_GROUP_COVARIANCE_DIAGNOSTIC_NOT_FULL_CF4_LIKELIHOOD",
        "source": ARCHIVE_URL,
        "source_version": "Zenodo 1.1.0, record 6824749",
        "archive_bytes": digest_reader.bytes_read,
        "archive_md5": digest_reader.md5.hexdigest(),
        "mock_catalogues": len(catalogues),
        "independent_simulation_boxes": len(by_simulation),
        "observers_per_box": 8,
        "training_overlap_context": {
            "exact_linked_training_rows": 1414,
            "pgc_set_matches_SDSS_PV_public_catalogue": True,
            "NgroupT17_quantiles_q00_q25_q50_q75_q100": [1, 1, 3, 4, 33],
        },
        "host_group_key": {
            "fields": list(HOST_KEY),
            "validation": "one composite key per repeated ID and one ID per composite key in every mock; zero selected centrals allowed because a halo central may fail the SDSS-PV selection",
            "richness_name": "number of selected mock rows sharing this host key; NOT Tempel17 Ngroup",
        },
        "richness_bins": bins,
        "catalogue_level_mean_residual_q05_q50_q95": _quantiles(
            [x["catalogue_mean_logdist_minus_truth"] for x in catalogues]
        ),
        "simulation_box_mean_residual_q05_q50_q95": _quantiles(box_offsets),
        "groups_without_selected_central": sum(x["groups_without_selected_central"] for x in catalogues),
        "groups_total": sum(x["host_groups"] for x in catalogues),
        "limits": [
            "This calibrates only the SDSS-PV FP mark/error and host-group covariance component.",
            "The mock selection omits redshift-success effects and does not model the 2M++ count catalogue or the heterogeneous full CF4 observation law.",
            "Mock host richness is not Tempel17 Ngroup; no Tempel group finder is run here, so absolute group inclusion remains uncalibrated.",
            "No likelihood parameter, selection correction, FoG number, or covariance inflation is adopted from this diagnostic.",
            "No heldout outcome, candidate field, MW/M31/M33 identity, density fit, or gravity evolution is used.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = summarize_archive(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir / "result.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("classification", "mock_catalogues", "independent_simulation_boxes", "archive_md5")}, flush=True))


if __name__ == "__main__":
    main()
