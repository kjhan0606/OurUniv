"""Fail-closed validator for the VizieR 2M++ catalog and group table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CATALOG_FILENAME = "2mpp_catalog.csv"
GROUPS_FILENAME = "2mpp_groups.csv"
README_FILENAME = "2mpp_ReadMe.txt"

CATALOG_HEADER = (
    "recno",
    "Name",
    "Ksmag",
    "HV",
    "e_HV",
    "Vcmb",
    "GID",
    "c11_5",
    "c12_5",
    "Cln",
    "M0",
    "M1",
    "M2",
    "Ref",
    "_RA",
    "_DE",
)
GROUPS_HEADER = (
    "recno",
    "GID",
    "GLON",
    "GLAT",
    "K2mag",
    "Rich",
    "HV",
    "Vcmb",
    "sigma",
    "_RA_icrs",
    "_DE_icrs",
)


class ValidationError(ValueError):
    """A fail-closed 2M++ contract violation."""


@dataclass(frozen=True)
class ValidationSpec:
    catalog_rows: int
    groups_rows: int
    catalog_canonical_sha256: str
    groups_canonical_sha256: str
    readme_sha256: str
    real_rows: int
    fake_zoa_rows: int
    cloned_redshift_rows: int
    known_orphan_gid: int = 5000
    known_orphan_rows: int = 3
    require_readme_markers: bool = True


DEFAULT_SPEC = ValidationSpec(
    catalog_rows=72_973,
    groups_rows=4_002,
    catalog_canonical_sha256=(
        "e761a9973f92e74520b81d36c5c7e76f739e47e2279da0567cdb2a92cf9d02ce"
    ),
    groups_canonical_sha256=(
        "c2959d7fbda188ae2496ce76743a9243b6fb260099550025db988d1df381f6fa"
    ),
    readme_sha256=(
        "0a4206da2c0e9997ff508909434816dfa095595d23eb6be4bee85fb8aee6c2d9"
    ),
    real_rows=69_160,
    fake_zoa_rows=3_813,
    cloned_redshift_rows=1_840,
)


@dataclass(frozen=True)
class CsvTable:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    raw_sha256: str
    canonical_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_csv_bytes(
    header: Sequence[str], rows: Sequence[Sequence[str]]
) -> bytes:
    """Serialize parsed rows sorted by integer recno using CSV/LF canon."""

    try:
        ordered = sorted(rows, key=lambda row: int(row[0]))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValidationError("recno must be a canonical integer in every row") from exc
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(ordered)
    return output.getvalue().encode("utf-8")


def canonical_csv_sha256(
    header: Sequence[str], rows: Sequence[Sequence[str]]
) -> str:
    """Return the order-invariant canonical recno-sorted CSV SHA256."""

    return _sha256(_canonical_csv_bytes(header, rows))


def _read_csv(
    path: Path, expected_filename: str, expected_header: Sequence[str]
) -> CsvTable:
    if path.name != expected_filename:
        raise ValidationError(
            f"expected filename {expected_filename!r}, got {path.name!r}"
        )
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
        parsed = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValidationError(f"cannot parse {expected_filename} as UTF-8 CSV") from exc
    if not parsed:
        raise ValidationError(f"{expected_filename} is empty")
    header = tuple(parsed[0])
    if header != tuple(expected_header):
        raise ValidationError(
            f"{expected_filename} header mismatch: {header!r}"
        )
    rows = tuple(tuple(row) for row in parsed[1:])
    for index, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValidationError(
                f"{expected_filename} row {index} has {len(row)} columns; "
                f"expected {len(header)}"
            )
    return CsvTable(
        header=header,
        rows=rows,
        raw_sha256=_sha256(raw),
        canonical_sha256=canonical_csv_sha256(header, rows),
    )


def _parse_int(value: str, label: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValidationError(f"{label} must be an integer, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise ValidationError(f"{label} must be >= {minimum}, got {parsed}")
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
        raise ValidationError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValidationError(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise ValidationError(f"{label} must be >= {minimum}, got {parsed}")
    if maximum is not None:
        invalid = parsed >= maximum if maximum_exclusive else parsed > maximum
        if invalid:
            relation = "<" if maximum_exclusive else "<="
            raise ValidationError(
                f"{label} must be {relation} {maximum}, got {parsed}"
            )
    return parsed


def _parse_flag(value: str, label: str) -> int:
    parsed = _parse_int(value, label)
    if parsed not in (0, 1):
        raise ValidationError(f"{label} must be 0 or 1, got {parsed}")
    return parsed


def _validate_recno(
    rows: Sequence[Sequence[str]], table_name: str, expected_count: int
) -> tuple[int, ...]:
    recnos = tuple(
        _parse_int(row[0], f"{table_name}.recno row {index}", minimum=1)
        for index, row in enumerate(rows, start=2)
    )
    if len(set(recnos)) != len(recnos):
        raise ValidationError(f"{table_name}.recno values must be unique")
    if set(recnos) != set(range(1, expected_count + 1)):
        raise ValidationError(
            f"{table_name}.recno must cover exactly 1..{expected_count}"
        )
    return recnos


def _validate_catalog(
    table: CsvTable, spec: ValidationSpec
) -> tuple[Counter[int], dict[str, int]]:
    if len(table.rows) != spec.catalog_rows:
        raise ValidationError(
            f"catalog row count {len(table.rows)} != {spec.catalog_rows}"
        )
    if table.canonical_sha256 != spec.catalog_canonical_sha256:
        raise ValidationError("catalog canonical recno-sorted SHA256 mismatch")
    _validate_recno(table.rows, "catalog", spec.catalog_rows)

    names: set[str] = set()
    gid_counts: Counter[int] = Counter()
    fake_count = 0
    clone_count = 0
    for row_number, row in enumerate(table.rows, start=2):
        values = dict(zip(CATALOG_HEADER, row))
        name = values["Name"]
        if not name:
            raise ValidationError(f"catalog.Name row {row_number} is empty")
        if name in names:
            raise ValidationError(f"catalog.Name is not unique: {name!r}")
        names.add(name)

        _parse_float(
            values["Ksmag"],
            f"catalog.Ksmag row {row_number}",
            minimum=0.0,
            maximum=12.5,
        )
        _parse_float(values["HV"], f"catalog.HV row {row_number}")
        _parse_float(values["Vcmb"], f"catalog.Vcmb row {row_number}")
        if values["e_HV"]:
            _parse_float(
                values["e_HV"],
                f"catalog.e_HV row {row_number}",
                minimum=0.0,
            )
        if values["GID"]:
            gid = _parse_int(
                values["GID"], f"catalog.GID row {row_number}", minimum=1
            )
            gid_counts[gid] += 1
        _parse_float(
            values["c11_5"],
            f"catalog.c11_5 row {row_number}",
            minimum=0.0,
            maximum=1.0,
        )
        if values["c12_5"]:
            _parse_float(
                values["c12_5"],
                f"catalog.c12_5 row {row_number}",
                minimum=0.0,
                maximum=1.0,
            )
        clone = _parse_flag(values["Cln"], f"catalog.Cln row {row_number}")
        _parse_flag(values["M0"], f"catalog.M0 row {row_number}")
        _parse_flag(values["M1"], f"catalog.M1 row {row_number}")
        _parse_flag(values["M2"], f"catalog.M2 row {row_number}")
        _parse_float(
            values["_RA"],
            f"catalog._RA row {row_number}",
            minimum=0.0,
            maximum=360.0,
            maximum_exclusive=True,
        )
        _parse_float(
            values["_DE"],
            f"catalog._DE row {row_number}",
            minimum=-90.0,
            maximum=90.0,
        )
        normalized_ref = values["Ref"].strip()
        if not normalized_ref:
            raise ValidationError(f"catalog.Ref row {row_number} is empty")
        fake_count += normalized_ref == "zoa"
        clone_count += clone

    real_count = len(table.rows) - fake_count
    if real_count != spec.real_rows or fake_count != spec.fake_zoa_rows:
        raise ValidationError(
            "catalog real/fake count mismatch: "
            f"real={real_count}, fake={fake_count}"
        )
    if clone_count != spec.cloned_redshift_rows:
        raise ValidationError(
            f"catalog cloned-redshift count {clone_count} != "
            f"{spec.cloned_redshift_rows}"
        )
    return gid_counts, {
        "unique_names": len(names),
        "real_rows": real_count,
        "fake_zoa_rows": fake_count,
        "cloned_redshift_rows": clone_count,
    }


def _validate_groups(table: CsvTable, spec: ValidationSpec) -> set[int]:
    if len(table.rows) != spec.groups_rows:
        raise ValidationError(
            f"groups row count {len(table.rows)} != {spec.groups_rows}"
        )
    if table.canonical_sha256 != spec.groups_canonical_sha256:
        raise ValidationError("groups canonical recno-sorted SHA256 mismatch")
    _validate_recno(table.rows, "groups", spec.groups_rows)

    gids: set[int] = set()
    for row_number, row in enumerate(table.rows, start=2):
        values = dict(zip(GROUPS_HEADER, row))
        gid = _parse_int(values["GID"], f"groups.GID row {row_number}", minimum=1)
        if gid in gids:
            raise ValidationError(f"groups.GID is not unique: {gid}")
        gids.add(gid)
        _parse_float(
            values["GLON"],
            f"groups.GLON row {row_number}",
            minimum=0.0,
            maximum=360.0,
            maximum_exclusive=True,
        )
        _parse_float(
            values["GLAT"],
            f"groups.GLAT row {row_number}",
            minimum=-90.0,
            maximum=90.0,
        )
        _parse_float(
            values["K2mag"],
            f"groups.K2mag row {row_number}",
            minimum=0.0,
            maximum=12.5,
        )
        _parse_int(values["Rich"], f"groups.Rich row {row_number}", minimum=1)
        _parse_float(values["HV"], f"groups.HV row {row_number}")
        _parse_float(values["Vcmb"], f"groups.Vcmb row {row_number}")
        _parse_float(
            values["sigma"],
            f"groups.sigma row {row_number}",
            minimum=0.0,
        )
        _parse_float(
            values["_RA_icrs"],
            f"groups._RA_icrs row {row_number}",
            minimum=0.0,
            maximum=360.0,
            maximum_exclusive=True,
        )
        _parse_float(
            values["_DE_icrs"],
            f"groups._DE_icrs row {row_number}",
            minimum=-90.0,
            maximum=90.0,
        )
    return gids


def validate_files(
    catalog_path: str | Path,
    groups_path: str | Path,
    readme_path: str | Path,
    spec: ValidationSpec = DEFAULT_SPEC,
) -> dict[str, object]:
    """Validate all 2M++ inputs and return a JSON-serializable result."""

    catalog_file = Path(catalog_path)
    groups_file = Path(groups_path)
    readme_file = Path(readme_path)
    if readme_file.name != README_FILENAME:
        raise ValidationError(
            f"expected filename {README_FILENAME!r}, got {readme_file.name!r}"
        )
    catalog = _read_csv(catalog_file, CATALOG_FILENAME, CATALOG_HEADER)
    groups = _read_csv(groups_file, GROUPS_FILENAME, GROUPS_HEADER)
    readme_bytes = readme_file.read_bytes()
    readme_sha = _sha256(readme_bytes)
    if readme_sha != spec.readme_sha256:
        raise ValidationError("ReadMe SHA256 mismatch")
    if spec.require_readme_markers:
        readme_text = readme_bytes.decode("utf-8")
        for marker in (
            "J/MNRAS/416/2840",
            "Number of real galaxies = 69160",
            "Number of fake galaxies in ZoA = 3813",
            "Number of groups = 4002",
        ):
            if marker not in readme_text:
                raise ValidationError(f"ReadMe missing required marker {marker!r}")

    catalog_gids, catalog_counts = _validate_catalog(catalog, spec)
    group_gids = _validate_groups(groups, spec)
    orphan_gids = set(catalog_gids) - group_gids
    if orphan_gids != {spec.known_orphan_gid}:
        raise ValidationError(
            f"unexpected orphan catalog GIDs: {sorted(orphan_gids)}"
        )
    if catalog_gids[spec.known_orphan_gid] != spec.known_orphan_rows:
        raise ValidationError(
            f"known orphan GID {spec.known_orphan_gid} has "
            f"{catalog_gids[spec.known_orphan_gid]} rows; "
            f"expected {spec.known_orphan_rows}"
        )
    if spec.known_orphan_gid in group_gids:
        raise ValidationError(
            f"known orphan GID {spec.known_orphan_gid} unexpectedly exists in groups"
        )

    return {
        "status": "PASS",
        "catalog": {
            "filename": CATALOG_FILENAME,
            "rows": len(catalog.rows),
            "raw_sha256": catalog.raw_sha256,
            "canonical_recno_sorted_sha256": catalog.canonical_sha256,
            **catalog_counts,
        },
        "groups": {
            "filename": GROUPS_FILENAME,
            "rows": len(groups.rows),
            "raw_sha256": groups.raw_sha256,
            "canonical_recno_sorted_sha256": groups.canonical_sha256,
            "unique_GIDs": len(group_gids),
        },
        "readme": {
            "filename": README_FILENAME,
            "sha256": readme_sha,
        },
        "linkage": {
            "known_orphan_GID": spec.known_orphan_gid,
            "known_orphan_rows": spec.known_orphan_rows,
            "other_orphan_GIDs": [],
        },
        "result_policy": {
            "fake_ZoA_selector": "Ref.strip() == 'zoa'",
            "fake_ZoA_class": "imputed_latent",
            "fake_ZoA_allowed_as_independent_observation": False,
            "cloned_redshift_selector": "Cln == 1",
            "cloned_redshift_class": "imputed_latent",
            "cloned_redshift_allowed_as_independent_observation": False,
            "raw_catalog_is_published_4_Mpc_h_density_map": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--readme", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate_files(args.catalog, args.groups, args.readme)
    except (OSError, ValidationError, UnicodeDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
