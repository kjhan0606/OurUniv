import csv
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cf4_2mpp_validate import (  # noqa: E402
    CATALOG_FILENAME,
    CATALOG_HEADER,
    GROUPS_FILENAME,
    GROUPS_HEADER,
    README_FILENAME,
    ValidationError,
    ValidationSpec,
    canonical_csv_sha256,
    validate_files,
)


def _catalog_rows():
    return [
        ["1", "A", "10.0", "100", "1", "110", "1", "0.9", "0.8", "0", "1", "0", "0", "ref-a", "10", "-5"],
        ["2", "B", "10.1", "200", "", "210", "1", "0.9", "0.8", "0", "1", "0", "0", "ref-b", "20", "5"],
        ["3", "Z1", "11.0", "300", "2", "310", "5000", "1.0", "0.0", "0", "0", "1", "0", "zoa", "30", "0"],
        ["4", "C", "11.1", "400", "3", "410", "5000", "1.0", "0.5", "1", "0", "0", "1", "ref-c", "40", "10"],
        ["5", "D", "11.2", "500", "4", "510", "5000", "1.0", "0.5", "0", "0", "0", "1", "ref-d", "50", "20"],
    ]


def _groups_rows():
    return [
        ["1", "1", "120", "30", "8.5", "2", "150", "160", "25", "12", "-4"]
    ]


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(header)
        csv.writer(handle, lineterminator="\n").writerows(rows)


def _make_dataset(tmp_path, catalog_rows=None, groups_rows=None, catalog_header=None):
    catalog_rows = _catalog_rows() if catalog_rows is None else catalog_rows
    groups_rows = _groups_rows() if groups_rows is None else groups_rows
    catalog_header = CATALOG_HEADER if catalog_header is None else catalog_header
    catalog = tmp_path / CATALOG_FILENAME
    groups = tmp_path / GROUPS_FILENAME
    readme = tmp_path / README_FILENAME
    _write_csv(catalog, catalog_header, catalog_rows)
    _write_csv(groups, GROUPS_HEADER, groups_rows)
    readme.write_text("synthetic 2M++ ReadMe\n", encoding="utf-8")
    spec = ValidationSpec(
        catalog_rows=len(catalog_rows),
        groups_rows=len(groups_rows),
        catalog_canonical_sha256=canonical_csv_sha256(
            catalog_header, catalog_rows
        ),
        groups_canonical_sha256=canonical_csv_sha256(
            GROUPS_HEADER, groups_rows
        ),
        readme_sha256=hashlib.sha256(readme.read_bytes()).hexdigest(),
        real_rows=sum(row[13] != "zoa" for row in catalog_rows),
        fake_zoa_rows=sum(row[13] == "zoa" for row in catalog_rows),
        cloned_redshift_rows=sum(row[9] == "1" for row in catalog_rows),
        require_readme_markers=False,
    )
    return catalog, groups, readme, spec


def test_canonical_hash_and_validation_are_row_order_invariant(tmp_path):
    catalog, groups, readme, spec = _make_dataset(tmp_path)
    first_raw = hashlib.sha256(catalog.read_bytes()).hexdigest()
    first = validate_files(catalog, groups, readme, spec)

    _write_csv(catalog, CATALOG_HEADER, list(reversed(_catalog_rows())))
    second_raw = hashlib.sha256(catalog.read_bytes()).hexdigest()
    second = validate_files(catalog, groups, readme, spec)

    assert first_raw != second_raw
    assert (
        first["catalog"]["canonical_recno_sorted_sha256"]
        == second["catalog"]["canonical_recno_sorted_sha256"]
        == spec.catalog_canonical_sha256
    )


def test_bad_header_fails_closed(tmp_path):
    bad_header = list(CATALOG_HEADER)
    bad_header[1] = "WrongName"
    catalog, groups, readme, spec = _make_dataset(
        tmp_path, catalog_header=bad_header
    )

    with pytest.raises(ValidationError, match="header mismatch"):
        validate_files(catalog, groups, readme, spec)


def test_bad_row_count_fails_closed(tmp_path):
    catalog, groups, readme, spec = _make_dataset(tmp_path)

    with pytest.raises(ValidationError, match="row count"):
        validate_files(catalog, groups, readme, replace(spec, catalog_rows=6))


def test_duplicate_recno_fails_closed(tmp_path):
    rows = _catalog_rows()
    rows[-1][0] = "1"
    catalog, groups, readme, spec = _make_dataset(tmp_path, catalog_rows=rows)

    with pytest.raises(ValidationError, match="recno values must be unique"):
        validate_files(catalog, groups, readme, spec)


def test_unexpected_orphan_gid_fails_closed(tmp_path):
    rows = _catalog_rows()
    rows[-1][6] = "4000"
    catalog, groups, readme, spec = _make_dataset(tmp_path, catalog_rows=rows)

    with pytest.raises(ValidationError, match="unexpected orphan"):
        validate_files(catalog, groups, readme, spec)


def test_fake_zoa_and_cloned_redshift_policy_is_imputed_latent(tmp_path):
    catalog, groups, readme, spec = _make_dataset(tmp_path)
    result = validate_files(catalog, groups, readme, spec)

    assert result["catalog"]["fake_zoa_rows"] == 1
    assert result["catalog"]["cloned_redshift_rows"] == 1
    assert result["result_policy"]["fake_ZoA_class"] == "imputed_latent"
    assert result["result_policy"]["cloned_redshift_class"] == "imputed_latent"
    assert not result["result_policy"][
        "fake_ZoA_allowed_as_independent_observation"
    ]
    assert not result["result_policy"][
        "cloned_redshift_allowed_as_independent_observation"
    ]
