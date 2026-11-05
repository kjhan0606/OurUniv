"""Fail-closed CF4--2M++ positional/redshift crossmatch.

The canonical mapping is an audit artifact, not a truth-editing mechanism.  Only
mutual, one-to-one nearest neighbours that pass both frozen gates receive the
``secure_joint_mark`` label.  Every other case is retained in a quarantine or
unmatched class using the precedence documented in ``CLASS_PRECEDENCE``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from cf4_2mpp_validate import CATALOG_HEADER


CF4_REQUIRED_COLUMNS = (
    "recno",
    "PGC",
    "1PGC",
    "Vcmb",
    "RAJ2000",
    "DEJ2000",
)
OUTPUT_HEADER = (
    "cf4_recno",
    "PGC",
    "1PGC",
    "twompp_recno",
    "twompp_Name",
    "sep_arcsec",
    "delta_vcmb_kms",
    "twompp_Cln",
    "mutual_nearest",
    "match_class",
)

SECURE_SEPARATION_ARCSEC = 3.0
EXTENDED_SEPARATION_ARCSEC = 30.0
VELOCITY_DIFFERENCE_KMS = 300.0

CLASS_PRECEDENCE = (
    "secure_joint_mark",
    "coordinate_redshift_conflict",
    "nonreciprocal_collision",
    "extended_review_candidate",
    "unmatched",
)


class CrossmatchError(ValueError):
    """A fail-closed input, matching, or publication contract violation."""


@dataclass(frozen=True)
class CF4Galaxy:
    recno: int
    pgc: str
    one_pgc: str
    vcmb: float
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True)
class TwoMppGalaxy:
    recno: int
    name: str
    vcmb: float
    ra_deg: float
    dec_deg: float
    cln: int


@dataclass(frozen=True)
class MappingRow:
    cf4: CF4Galaxy
    twompp: TwoMppGalaxy | None
    separation_arcsec: float | None
    delta_vcmb_kms: float | None
    mutual_nearest: bool
    match_class: str


@dataclass(frozen=True)
class CrossmatchResult:
    mapping_bytes: bytes
    mapping_sha256: str
    summary: dict[str, object]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[tuple[str, ...]], str]:
    raw = path.read_bytes()
    try:
        parsed = list(
            csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise CrossmatchError(f"cannot parse {path} as UTF-8 CSV") from exc
    if not parsed:
        raise CrossmatchError(f"{path} is empty")
    header = tuple(parsed[0])
    if len(set(header)) != len(header):
        raise CrossmatchError(f"{path} header contains duplicate column names")
    rows = [tuple(row) for row in parsed[1:]]
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise CrossmatchError(
                f"{path} row {row_number} has {len(row)} columns; "
                f"expected {len(header)}"
            )
    return header, rows, _sha256(raw)


def _parse_int(value: str, label: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CrossmatchError(f"{label} must be an integer, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise CrossmatchError(f"{label} must be >= {minimum}, got {parsed}")
    return parsed


def _parse_float(
    value: str,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_exclusive: bool = False,
) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise CrossmatchError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise CrossmatchError(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise CrossmatchError(f"{label} must be >= {minimum}, got {parsed}")
    if maximum is not None:
        invalid = parsed >= maximum if maximum_exclusive else parsed > maximum
        if invalid:
            relation = "<" if maximum_exclusive else "<="
            raise CrossmatchError(f"{label} must be {relation} {maximum}")
    return parsed


def _load_cf4(path: Path) -> tuple[list[CF4Galaxy], str]:
    header, rows, raw_sha = _read_csv(path)
    missing = [column for column in CF4_REQUIRED_COLUMNS if column not in header]
    if missing:
        raise CrossmatchError(f"CF4 header missing required columns: {missing}")
    positions = {name: header.index(name) for name in CF4_REQUIRED_COLUMNS}
    galaxies: list[CF4Galaxy] = []
    recnos: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        recno = _parse_int(
            row[positions["recno"]], f"CF4.recno row {row_number}", minimum=1
        )
        if recno in recnos:
            raise CrossmatchError(f"CF4.recno values must be unique: {recno}")
        recnos.add(recno)
        pgc = row[positions["PGC"]].strip()
        one_pgc = row[positions["1PGC"]].strip()
        if not pgc or not one_pgc:
            raise CrossmatchError(f"CF4 PGC and 1PGC row {row_number} must be nonempty")
        galaxies.append(
            CF4Galaxy(
                recno=recno,
                pgc=pgc,
                one_pgc=one_pgc,
                vcmb=_parse_float(
                    row[positions["Vcmb"]], f"CF4.Vcmb row {row_number}"
                ),
                ra_deg=_parse_float(
                    row[positions["RAJ2000"]],
                    f"CF4.RAJ2000 row {row_number}",
                    minimum=0.0,
                    maximum=360.0,
                    maximum_exclusive=True,
                ),
                dec_deg=_parse_float(
                    row[positions["DEJ2000"]],
                    f"CF4.DEJ2000 row {row_number}",
                    minimum=-90.0,
                    maximum=90.0,
                ),
            )
        )
    if not galaxies:
        raise CrossmatchError("CF4 input contains no galaxy rows")
    galaxies.sort(key=lambda item: item.recno)
    return galaxies, raw_sha


def _load_twompp(
    path: Path,
) -> tuple[list[TwoMppGalaxy], int, int, str]:
    header, rows, raw_sha = _read_csv(path)
    if header != CATALOG_HEADER:
        raise CrossmatchError(f"2M++ catalog header mismatch: {header!r}")
    galaxies: list[TwoMppGalaxy] = []
    recnos: set[int] = set()
    names: set[str] = set()
    zoa_count = 0
    cln_real_count = 0
    for row_number, row in enumerate(rows, start=2):
        values = dict(zip(CATALOG_HEADER, row))
        recno = _parse_int(
            values["recno"], f"2M++.recno row {row_number}", minimum=1
        )
        if recno in recnos:
            raise CrossmatchError(f"2M++.recno values must be unique: {recno}")
        recnos.add(recno)
        name = values["Name"].strip()
        if not name:
            raise CrossmatchError(f"2M++.Name row {row_number} is empty")
        if name in names:
            raise CrossmatchError(f"2M++.Name values must be unique: {name!r}")
        names.add(name)
        vcmb = _parse_float(values["Vcmb"], f"2M++.Vcmb row {row_number}")
        ra_deg = _parse_float(
            values["_RA"],
            f"2M++._RA row {row_number}",
            minimum=0.0,
            maximum=360.0,
            maximum_exclusive=True,
        )
        dec_deg = _parse_float(
            values["_DE"],
            f"2M++._DE row {row_number}",
            minimum=-90.0,
            maximum=90.0,
        )
        cln = _parse_int(values["Cln"], f"2M++.Cln row {row_number}")
        if cln not in (0, 1):
            raise CrossmatchError(f"2M++.Cln row {row_number} must be 0 or 1")
        normalized_ref = values["Ref"].strip()
        if not normalized_ref:
            raise CrossmatchError(f"2M++.Ref row {row_number} is empty")
        if normalized_ref == "zoa":
            zoa_count += 1
            continue
        cln_real_count += cln
        galaxies.append(
            TwoMppGalaxy(
                recno=recno,
                name=name,
                vcmb=vcmb,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                cln=cln,
            )
        )
    galaxies.sort(key=lambda item: item.recno)
    return galaxies, zoa_count, cln_real_count, raw_sha


def _unit_vectors(ra_deg: Sequence[float], dec_deg: Sequence[float]) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def _separation_arcsec(first: np.ndarray, second: np.ndarray) -> float:
    chord = float(np.linalg.norm(first - second))
    radians = 2.0 * math.asin(min(1.0, 0.5 * chord))
    return math.degrees(radians) * 3600.0


def _classify(separation: float, delta_vcmb: float, mutual: bool) -> str:
    velocity_pass = abs(delta_vcmb) <= VELOCITY_DIFFERENCE_KMS
    if mutual and separation <= SECURE_SEPARATION_ARCSEC and velocity_pass:
        return "secure_joint_mark"
    if separation <= SECURE_SEPARATION_ARCSEC and not velocity_pass:
        return "coordinate_redshift_conflict"
    if separation <= SECURE_SEPARATION_ARCSEC and velocity_pass:
        return "nonreciprocal_collision"
    if (
        SECURE_SEPARATION_ARCSEC < separation <= EXTENDED_SEPARATION_ARCSEC
        and velocity_pass
    ):
        return "extended_review_candidate"
    return "unmatched"


def _mapping_rows(
    cf4: Sequence[CF4Galaxy], twompp: Sequence[TwoMppGalaxy]
) -> list[MappingRow]:
    if not twompp:
        return [
            MappingRow(item, None, None, None, False, "unmatched") for item in cf4
        ]

    cf4_vectors = _unit_vectors(
        [item.ra_deg for item in cf4], [item.dec_deg for item in cf4]
    )
    twompp_vectors = _unit_vectors(
        [item.ra_deg for item in twompp], [item.dec_deg for item in twompp]
    )
    cf4_to_twompp = np.asarray(
        cKDTree(twompp_vectors).query(cf4_vectors, k=1)[1], dtype=int
    )
    twompp_to_cf4 = np.asarray(
        cKDTree(cf4_vectors).query(twompp_vectors, k=1)[1], dtype=int
    )

    mapped: list[MappingRow] = []
    for cf4_index, twompp_index in enumerate(cf4_to_twompp):
        target = twompp[int(twompp_index)]
        separation = _separation_arcsec(
            cf4_vectors[cf4_index], twompp_vectors[int(twompp_index)]
        )
        delta_vcmb = cf4[cf4_index].vcmb - target.vcmb
        mutual = int(twompp_to_cf4[int(twompp_index)]) == cf4_index
        match_class = _classify(separation, delta_vcmb, mutual)
        if match_class == "unmatched":
            mapped.append(
                MappingRow(cf4[cf4_index], None, None, None, False, match_class)
            )
        else:
            mapped.append(
                MappingRow(
                    cf4[cf4_index],
                    target,
                    separation,
                    delta_vcmb,
                    mutual,
                    match_class,
                )
            )

    secure_targets = [
        item.twompp.recno
        for item in mapped
        if item.match_class == "secure_joint_mark" and item.twompp is not None
    ]
    if len(secure_targets) != len(set(secure_targets)):
        raise CrossmatchError("secure mapping is not one-to-one in 2M++ recno")
    return mapped


def _serialize_mapping(rows: Sequence[MappingRow]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(OUTPUT_HEADER)
    for item in sorted(rows, key=lambda row: row.cf4.recno):
        if item.twompp is None:
            twompp_values: tuple[str, ...] = ("", "", "", "", "", "0")
        else:
            twompp_values = (
                str(item.twompp.recno),
                item.twompp.name,
                format(item.separation_arcsec, ".9f"),
                format(item.delta_vcmb_kms, ".6f"),
                str(item.twompp.cln),
                "1" if item.mutual_nearest else "0",
            )
        writer.writerow(
            (
                str(item.cf4.recno),
                item.cf4.pgc,
                item.cf4.one_pgc,
                *twompp_values,
                item.match_class,
            )
        )
    return output.getvalue().encode("utf-8")


def build_crossmatch(
    cf4_galaxies_path: str | Path, twompp_catalog_path: str | Path
) -> CrossmatchResult:
    """Build a deterministic mapping and a JSON-serializable audit summary."""

    cf4_path = Path(cf4_galaxies_path)
    twompp_path = Path(twompp_catalog_path)
    cf4, cf4_sha = _load_cf4(cf4_path)
    twompp, zoa_count, cln_real_count, twompp_sha = _load_twompp(twompp_path)
    rows = _mapping_rows(cf4, twompp)
    mapping_bytes = _serialize_mapping(rows)
    mapping_sha = _sha256(mapping_bytes)
    class_counts = Counter(item.match_class for item in rows)
    classified = [item for item in rows if item.twompp is not None]
    secure = [item for item in rows if item.match_class == "secure_joint_mark"]
    summary: dict[str, object] = {
        "schema": "ouruniv-cf4-2mpp-crossmatch-summary-v1",
        "status": "COMPLETE",
        "algorithm": {
            "nearest_neighbour": "bidirectional scipy.spatial.cKDTree on unit sphere",
            "class_precedence": list(CLASS_PRECEDENCE),
            "secure_joint_mark": (
                "mutual nearest AND separation <= 3.0 arcsec AND "
                "abs(delta Vcmb) <= 300 km/s; exact one-to-one"
            ),
            "delta_vcmb_definition": "CF4 Vcmb minus 2M++ Vcmb",
            "automatic_promotion_or_manual_truth_edit_allowed": False,
        },
        "thresholds": {
            "secure_separation_arcsec_max_inclusive": SECURE_SEPARATION_ARCSEC,
            "extended_review_separation_arcsec_max_inclusive": (
                EXTENDED_SEPARATION_ARCSEC
            ),
            "absolute_delta_vcmb_kms_max_inclusive": VELOCITY_DIFFERENCE_KMS,
        },
        "inputs": {
            "cf4_galaxies": {
                "path": str(cf4_path),
                "rows": len(cf4),
                "raw_sha256": cf4_sha,
            },
            "twompp_catalog": {
                "path": str(twompp_path),
                "rows_including_ZoA": len(twompp) + zoa_count,
                "real_rows_eligible": len(twompp),
                "ZoA_rows_excluded": zoa_count,
                "real_Cln_rows": cln_real_count,
                "raw_sha256": twompp_sha,
            },
        },
        "mapping": {
            "rows": len(rows),
            "canonical_order": "ascending integer CF4 recno",
            "sha256": mapping_sha,
            "class_counts": {
                name: class_counts.get(name, 0) for name in CLASS_PRECEDENCE
            },
            "unique_twompp_targets_in_non_unmatched": len(
                {item.twompp.recno for item in classified if item.twompp is not None}
            ),
            "unique_cf4_1PGC_total": len({item.cf4.one_pgc for item in rows}),
            "unique_cf4_1PGC_in_non_unmatched": len(
                {item.cf4.one_pgc for item in classified}
            ),
            "unique_cf4_1PGC_secure": len(
                {item.cf4.one_pgc for item in secure}
            ),
        },
        "result_policy": {
            "ZoA_selector": "Ref.strip() == 'zoa'",
            "ZoA_rows_allowed_as_matching_observations": False,
            "Cln_flag_retained_in_mapping": True,
            "Cln_radial_redshift_class": "imputed_latent",
            "Cln_radial_redshift_allowed_as_independent_observation": False,
            "secure_mapping_role": "joint-likelihood de-duplication mark",
            "quarantine_classes_may_be_auto_promoted": False,
        },
    }
    return CrossmatchResult(mapping_bytes, mapping_sha, summary)


def _write_temp(parent: Path, prefix: str, payload: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=prefix, suffix=".tmp", dir=parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _inode_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _publish_no_overwrite(
    temporary: Path, destination: Path
) -> tuple[int, int]:
    """Atomically publish a staged inode and return its ownership identity."""

    identity = _inode_identity(temporary)
    os.link(temporary, destination)
    if _inode_identity(destination) != identity:
        raise CrossmatchError("published path does not identify the staged inode")
    temporary.unlink()
    return identity


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> bool:
    """Remove path only while its exact device/inode is the one we published."""

    try:
        if _inode_identity(path) != identity:
            return False
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def publish_crossmatch(
    cf4_galaxies_path: str | Path,
    twompp_catalog_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, object]:
    """Validate, stage, then publish mapping first and bound COMPLETE summary last."""

    output = Path(output_path)
    summary_file = Path(summary_path)
    if output.resolve() == summary_file.resolve():
        raise CrossmatchError("output and summary paths must be distinct")
    if output.exists() or summary_file.exists():
        raise FileExistsError("refusing overwrite: output or summary already exists")
    if not output.parent.is_dir() or not summary_file.parent.is_dir():
        raise CrossmatchError("output and summary parent directories must already exist")

    result = build_crossmatch(cf4_galaxies_path, twompp_catalog_path)
    summary_payload = (
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    mapping_temp: Path | None = None
    summary_temp: Path | None = None
    mapping_published_identity: tuple[int, int] | None = None
    try:
        mapping_temp = _write_temp(output.parent, f".{output.name}.", result.mapping_bytes)
        summary_temp = _write_temp(
            summary_file.parent, f".{summary_file.name}.", summary_payload
        )
        if _sha256(mapping_temp.read_bytes()) != result.mapping_sha256:
            raise CrossmatchError("staged mapping SHA256 validation failed")
        parsed_summary = json.loads(summary_temp.read_text(encoding="utf-8"))
        if parsed_summary.get("status") != "COMPLETE":
            raise CrossmatchError("staged summary is not COMPLETE")
        if parsed_summary.get("mapping", {}).get("sha256") != result.mapping_sha256:
            raise CrossmatchError("staged summary does not bind the mapping SHA256")

        mapping_published_identity = _publish_no_overwrite(mapping_temp, output)
        mapping_temp = None
        _publish_no_overwrite(summary_temp, summary_file)
        summary_temp = None
        return result.summary
    except BaseException:
        if mapping_published_identity is not None:
            _unlink_if_owned(output, mapping_published_identity)
        raise
    finally:
        if mapping_temp is not None:
            mapping_temp.unlink(missing_ok=True)
        if summary_temp is not None:
            summary_temp.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf4-galaxies", required=True, type=Path)
    parser.add_argument("--twompp-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = publish_crossmatch(
            args.cf4_galaxies, args.twompp_catalog, args.output, args.summary
        )
    except (OSError, CrossmatchError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
