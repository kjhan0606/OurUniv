"""Object-level CF4--2M++ crossmatch manifest and factor-owner gate.

The existing crossmatch CSV is the canonical matching artifact.  This module
does not edit or promote matches; it derives a small, explicit manifest of
secure rows and binds every entry to its CF4 ``1PGC`` group.  The manifest is
what a likelihood caller must pass to the host-side ownership check.  Ambiguous
and unmatched classes never enter the secure manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SECURE_CLASS = "secure_joint_mark"
MAPPING_HEADER = (
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
FACTOR_OWNERSHIP = {
    "count_factor_owner": "2Mpp_grid_counts",
    "redshift_factor_owner": "CF4_group_marks_shared_redshift",
    "independent_twompp_redshift_factor": False,
    "group_key": "CF4_1PGC",
}
EXPECTED_MAPPING_SHA256 = "64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf"
EXPECTED_SUMMARY_SHA256 = "3e2e5841d62e9581c7437a28f07d6f5c3423b023749f99a678546d1d7d29752a"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MAPPING_PATH = PROJECT_ROOT / "data" / "cf4_2mpp_crossmatch_v1.csv"
EXPECTED_SUMMARY_PATH = PROJECT_ROOT / "config" / "cf4_2mpp_crossmatch_v1_result.json"
EXPECTED_THRESHOLDS = {
    "secure_separation_arcsec_max_inclusive": 3.0,
    "extended_review_separation_arcsec_max_inclusive": 30.0,
    "absolute_delta_vcmb_kms_max_inclusive": 300.0,
}


class CrossmatchManifestError(ValueError):
    """A malformed, inconsistent, or unsafe object-level manifest."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CrossmatchManifestError(f"{label} must be an integer") from exc
    if parsed < 1:
        raise CrossmatchManifestError(f"{label} must be positive")
    return parsed


def _float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CrossmatchManifestError(f"{label} must be numeric") from exc
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise CrossmatchManifestError(f"{label} must be finite")
    return parsed


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MAPPING_HEADER:
                raise CrossmatchManifestError(
                    f"mapping header must be exactly {MAPPING_HEADER!r}"
                )
            rows = list(reader)
    except OSError as exc:
        raise CrossmatchManifestError(f"cannot read mapping {path}") from exc
    if not rows:
        raise CrossmatchManifestError("mapping must contain at least one row")
    return rows


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise CrossmatchManifestError("manifest must be a JSON object")
    if manifest.get("schema") != "ouruniv-cf4-2mpp-secure-object-manifest-v1":
        raise CrossmatchManifestError("unexpected secure-object manifest schema")
    if manifest.get("status") != "VALIDATED_SECURE_OBJECT_MANIFEST":
        raise CrossmatchManifestError("manifest is not in validated status")
    ownership = manifest.get("factor_ownership")
    if ownership != FACTOR_OWNERSHIP:
        raise CrossmatchManifestError("factor ownership is not the frozen contract")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise CrossmatchManifestError("manifest source binding is required")
    for key in ("mapping_path", "mapping_sha256", "summary_path", "summary_sha256"):
        if not str(source.get(key, "")).strip():
            raise CrossmatchManifestError(f"manifest source binding is missing {key}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CrossmatchManifestError("manifest entries must be a non-empty list")
    secure_ids: list[str] = []
    twompp_ids: list[str] = []
    group_ids: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise CrossmatchManifestError(f"entry {index} is not an object")
        required = {
            "secure_object_id",
            "cf4_recno",
            "cf4_group_id",
            "group_index",
            "twompp_object_id",
            "twompp_recno",
            "match_class",
        }
        if not required.issubset(entry):
            raise CrossmatchManifestError(f"entry {index} is missing required fields")
        secure_id = str(entry["secure_object_id"]).strip()
        twompp_id = str(entry["twompp_object_id"]).strip()
        group_id = str(entry["cf4_group_id"]).strip()
        if not secure_id.startswith("cf4:") or not twompp_id.startswith("2mpp:"):
            raise CrossmatchManifestError(f"entry {index} has malformed object identity")
        if not group_id.startswith("cf4_1PGC:") or not group_id.removeprefix("cf4_1PGC:"):
            raise CrossmatchManifestError(f"entry {index} has malformed group identity")
        if entry["match_class"] != SECURE_CLASS:
            raise CrossmatchManifestError("quarantine or unmatched rows cannot enter secure manifest")
        cf4_recno = _int(str(entry["cf4_recno"]), f"entry {index} cf4_recno")
        twompp_recno = _int(str(entry["twompp_recno"]), f"entry {index} twompp_recno")
        try:
            group_index = int(entry["group_index"])
        except (TypeError, ValueError) as exc:
            raise CrossmatchManifestError(f"entry {index} group_index must be an integer") from exc
        if group_index < 0:
            raise CrossmatchManifestError(f"entry {index} group_index must be non-negative")
        if secure_id != f"cf4:{cf4_recno}" or twompp_id != f"2mpp:{twompp_recno}":
            raise CrossmatchManifestError(f"entry {index} identity does not bind its recno")
        secure_ids.append(secure_id)
        twompp_ids.append(twompp_id)
        group_ids.append(group_id)
    if len(set(secure_ids)) != len(secure_ids):
        raise CrossmatchManifestError("secure object identities must be unique")
    if len(set(twompp_ids)) != len(twompp_ids):
        raise CrossmatchManifestError("2M++ object identities must be unique")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or counts.get("secure_rows") != len(entries):
        raise CrossmatchManifestError("manifest secure-row count is inconsistent")
    group_indices = sorted({int(entry["group_index"]) for entry in entries})
    if group_indices != list(range(len(group_indices))):
        raise CrossmatchManifestError("group_index values must be contiguous from zero")
    if counts.get("secure_cf4_groups") != len(group_indices):
        raise CrossmatchManifestError("manifest group count is inconsistent")
    return dict(manifest)


def validate_secure_crossmatch_manifest(
    manifest: Mapping[str, Any],
    *,
    mapping_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    """Validate a manifest and rebind every entry to canonical source files."""

    value = _validate_manifest_shape(manifest)
    canonical = build_secure_crossmatch_manifest(mapping_path, summary_path)
    if value["source"] != canonical["source"]:
        raise CrossmatchManifestError("manifest source hashes do not match canonical files")
    if value["counts"] != canonical["counts"] or value["entries"] != canonical["entries"]:
        raise CrossmatchManifestError("manifest entries do not reproduce canonical secure rows")
    return value


def build_secure_crossmatch_manifest(
    mapping_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    """Derive and validate secure object/group identities from canonical files.

    ``1PGC`` is the CF4 group/member key.  It is retained explicitly so the
    shared-redshift latent is attached to a group rather than accidentally
    duplicated per galaxy.  This function is read-only and refuses any mapping
    whose hash, order, thresholds, or one-to-one secure assignment disagrees
    with its COMPLETE summary.
    """

    mapping = Path(mapping_path)
    summary_file = Path(summary_path)
    if mapping.resolve() != EXPECTED_MAPPING_PATH.resolve():
        raise CrossmatchManifestError("mapping path is not the frozen canonical repo path")
    if summary_file.resolve() != EXPECTED_SUMMARY_PATH.resolve():
        raise CrossmatchManifestError("summary path is not the frozen canonical repo path")
    rows = _read_rows(mapping)
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossmatchManifestError(f"cannot read crossmatch summary {summary_file}") from exc
    if summary.get("status") != "COMPLETE":
        raise CrossmatchManifestError("crossmatch summary is not COMPLETE")
    mapping_sha = _sha256(mapping)
    summary_sha = _sha256(summary_file)
    if mapping_sha != EXPECTED_MAPPING_SHA256:
        raise CrossmatchManifestError("mapping SHA256 is not the frozen v1 artifact")
    if summary_sha != EXPECTED_SUMMARY_SHA256:
        raise CrossmatchManifestError("summary SHA256 is not the frozen v1 artifact")
    if summary.get("mapping", {}).get("sha256") != mapping_sha:
        raise CrossmatchManifestError("mapping SHA256 does not match COMPLETE summary")
    declared_rows = summary.get("mapping", {}).get("rows")
    if declared_rows != len(rows):
        raise CrossmatchManifestError("mapping row count does not match COMPLETE summary")
    thresholds = summary.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(EXPECTED_THRESHOLDS):
        raise CrossmatchManifestError("crossmatch thresholds must match the frozen exact schema")
    for key, expected in EXPECTED_THRESHOLDS.items():
        try:
            actual = float(thresholds[key])
        except (TypeError, ValueError) as exc:
            raise CrossmatchManifestError(f"threshold {key} must be numeric") from exc
        if actual != expected:
            raise CrossmatchManifestError(f"threshold {key} does not match frozen value")
    max_sep = EXPECTED_THRESHOLDS["secure_separation_arcsec_max_inclusive"]
    max_dv = EXPECTED_THRESHOLDS["absolute_delta_vcmb_kms_max_inclusive"]
    class_counts: dict[str, int] = {}
    secure: list[dict[str, Any]] = []
    seen_cf4: set[int] = set()
    seen_twompp: set[int] = set()
    group_indices: dict[str, int] = {}
    previous_recno = 0
    for index, row in enumerate(rows, start=2):
        recno = _int(row["cf4_recno"], f"mapping row {index} cf4_recno")
        if recno <= previous_recno:
            raise CrossmatchManifestError("mapping must be strictly sorted by cf4_recno")
        previous_recno = recno
        if recno in seen_cf4:
            raise CrossmatchManifestError("CF4 recno is duplicated")
        seen_cf4.add(recno)
        class_counts[row["match_class"]] = class_counts.get(row["match_class"], 0) + 1
        if row["match_class"] != SECURE_CLASS:
            continue
        if not row["1PGC"].strip():
            raise CrossmatchManifestError(f"secure row {index} has no CF4 1PGC group")
        twompp_recno = _int(row["twompp_recno"], f"mapping row {index} twompp_recno")
        if twompp_recno in seen_twompp:
            raise CrossmatchManifestError("secure 2M++ recno is not one-to-one")
        seen_twompp.add(twompp_recno)
        if row["mutual_nearest"] != "1":
            raise CrossmatchManifestError("secure row is not marked mutual_nearest")
        if _float(row["sep_arcsec"], f"mapping row {index} sep_arcsec") > max_sep:
            raise CrossmatchManifestError("secure row exceeds separation threshold")
        if abs(_float(row["delta_vcmb_kms"], f"mapping row {index} delta_vcmb_kms")) > max_dv:
            raise CrossmatchManifestError("secure row exceeds velocity threshold")
        group_id = f"cf4_1PGC:{row['1PGC'].strip()}"
        if group_id not in group_indices:
            group_indices[group_id] = len(group_indices)
        group_index = group_indices[group_id]
        secure.append(
            {
                "secure_object_id": f"cf4:{recno}",
                "cf4_recno": recno,
                "cf4_group_id": group_id,
                "group_index": group_index,
                "twompp_object_id": f"2mpp:{twompp_recno}",
                "twompp_recno": twompp_recno,
                "match_class": SECURE_CLASS,
            }
        )
    expected_secure = summary.get("mapping", {}).get("class_counts", {}).get(SECURE_CLASS)
    if expected_secure != len(secure):
        raise CrossmatchManifestError("secure-row count does not match COMPLETE summary")
    if summary.get("mapping", {}).get("class_counts") != class_counts:
        raise CrossmatchManifestError("mapping class counts do not match COMPLETE summary")
    manifest: dict[str, Any] = {
        "schema": "ouruniv-cf4-2mpp-secure-object-manifest-v1",
        "status": "VALIDATED_SECURE_OBJECT_MANIFEST",
        "source": {
            "mapping_path": str(mapping.resolve()),
            "mapping_sha256": mapping_sha,
            "summary_path": str(summary_file.resolve()),
            "summary_sha256": summary_sha,
        },
        "factor_ownership": dict(FACTOR_OWNERSHIP),
        "group_semantics": "CF4 1PGC identifies one shared CF4 group-mark latent",
        "counts": {
            "mapping_rows": len(rows),
            "secure_rows": len(secure),
            "secure_cf4_groups": len({item["cf4_group_id"] for item in secure}),
        },
        "entries": secure,
    }
    return _validate_manifest_shape(manifest)


def load_secure_crossmatch_manifest(
    path: str | Path,
    *,
    mapping_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    """Load a JSON manifest and apply the same fail-closed validation."""

    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossmatchManifestError(f"cannot read manifest {manifest_path}") from exc
    return validate_secure_crossmatch_manifest(
        value, mapping_path=mapping_path, summary_path=summary_path
    )
