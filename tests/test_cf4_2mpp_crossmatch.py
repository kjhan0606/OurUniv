import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cf4_2mpp_crossmatch as crossmatch_module  # noqa: E402
from cf4_2mpp_crossmatch import (  # noqa: E402
    CF4_REQUIRED_COLUMNS,
    CrossmatchError,
    build_crossmatch,
    publish_crossmatch,
)
from cf4_2mpp_validate import CATALOG_HEADER  # noqa: E402


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _cf4_row(recno, ra, dec=0.0, vcmb=1000.0, pgc=None, one_pgc=None):
    return [
        str(recno),
        str(recno if pgc is None else pgc),
        str(recno if one_pgc is None else one_pgc),
        str(vcmb),
        str(ra),
        str(dec),
    ]


def _twompp_row(
    recno,
    ra,
    dec=0.0,
    vcmb=1000.0,
    *,
    name=None,
    cln=0,
    ref="real",
):
    values = {
        "recno": str(recno),
        "Name": name or f"T{recno}",
        "Ksmag": "10.0",
        "HV": str(vcmb),
        "e_HV": "1",
        "Vcmb": str(vcmb),
        "GID": "",
        "c11_5": "1.0",
        "c12_5": "0.5",
        "Cln": str(cln),
        "M0": "1",
        "M1": "0",
        "M2": "0",
        "Ref": ref,
        "_RA": str(ra),
        "_DE": str(dec),
    }
    return [values[column] for column in CATALOG_HEADER]


def _dataset(tmp_path, cf4_rows, twompp_rows, *, suffix=""):
    cf4 = tmp_path / f"cf4{suffix}.csv"
    twompp = tmp_path / f"twompp{suffix}.csv"
    _write_csv(cf4, CF4_REQUIRED_COLUMNS, cf4_rows)
    _write_csv(twompp, CATALOG_HEADER, twompp_rows)
    return cf4, twompp


def _mapping_rows(result):
    text = result.mapping_bytes.decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def test_mutual_nearest_is_secure_and_summary_binds_canonical_output(tmp_path):
    cf4, twompp = _dataset(
        tmp_path,
        [_cf4_row(2, 20.0, one_pgc=10), _cf4_row(1, 10.0, one_pgc=10)],
        [_twompp_row(20, 20.0), _twompp_row(10, 10.0)],
    )

    result = build_crossmatch(cf4, twompp)
    rows = _mapping_rows(result)

    assert [row["cf4_recno"] for row in rows] == ["1", "2"]
    assert {row["match_class"] for row in rows} == {"secure_joint_mark"}
    assert all(row["mutual_nearest"] == "1" for row in rows)
    assert result.summary["mapping"]["class_counts"]["secure_joint_mark"] == 2
    assert result.summary["mapping"]["unique_twompp_targets_in_non_unmatched"] == 2
    assert result.summary["mapping"]["unique_cf4_1PGC_total"] == 1
    assert result.mapping_sha256 == hashlib.sha256(result.mapping_bytes).hexdigest()


def test_canonical_mapping_is_input_row_order_invariant(tmp_path):
    cf4_rows = [_cf4_row(2, 20.0), _cf4_row(1, 10.0)]
    twompp_rows = [_twompp_row(20, 20.0), _twompp_row(10, 10.0)]
    first_cf4, first_twompp = _dataset(tmp_path, cf4_rows, twompp_rows, suffix="a")
    second_cf4, second_twompp = _dataset(
        tmp_path, list(reversed(cf4_rows)), list(reversed(twompp_rows)), suffix="b"
    )

    first = build_crossmatch(first_cf4, first_twompp)
    second = build_crossmatch(second_cf4, second_twompp)

    assert first.mapping_bytes == second.mapping_bytes
    assert first.mapping_sha256 == second.mapping_sha256
    assert first.summary["mapping"] == second.summary["mapping"]
    assert (
        first.summary["inputs"]["cf4_galaxies"]["raw_sha256"]
        != second.summary["inputs"]["cf4_galaxies"]["raw_sha256"]
    )


def test_nonreciprocal_near_collision_is_quarantined(tmp_path):
    arcsec = 1.0 / 3600.0
    cf4, twompp = _dataset(
        tmp_path,
        [_cf4_row(1, 0.0), _cf4_row(2, arcsec)],
        [_twompp_row(10, 0.1 * arcsec)],
    )

    rows = _mapping_rows(build_crossmatch(cf4, twompp))

    assert rows[0]["match_class"] == "secure_joint_mark"
    assert rows[1]["match_class"] == "nonreciprocal_collision"
    assert rows[1]["mutual_nearest"] == "0"


def test_coordinate_redshift_conflict_precedes_reciprocity(tmp_path):
    cf4, twompp = _dataset(
        tmp_path,
        [_cf4_row(1, 10.0, vcmb=1500)],
        [_twompp_row(10, 10.0, vcmb=1000)],
    )

    row = _mapping_rows(build_crossmatch(cf4, twompp))[0]

    assert row["mutual_nearest"] == "1"
    assert row["match_class"] == "coordinate_redshift_conflict"
    assert row["delta_vcmb_kms"] == "500.000000"


def test_extended_review_candidate_is_not_promoted(tmp_path):
    ten_arcsec = 10.0 / 3600.0
    cf4, twompp = _dataset(
        tmp_path,
        [_cf4_row(1, 10.0, vcmb=1100)],
        [_twompp_row(10, 10.0 + ten_arcsec, vcmb=1000)],
    )

    row = _mapping_rows(build_crossmatch(cf4, twompp))[0]

    assert row["mutual_nearest"] == "1"
    assert row["match_class"] == "extended_review_candidate"


def test_zoa_is_excluded_from_matching_even_at_identical_position(tmp_path):
    cf4, twompp = _dataset(
        tmp_path,
        [_cf4_row(1, 10.0)],
        [_twompp_row(10, 10.0, ref="  zoa  ")],
    )

    result = build_crossmatch(cf4, twompp)
    row = _mapping_rows(result)[0]

    assert row["match_class"] == "unmatched"
    assert row["twompp_recno"] == ""
    assert result.summary["inputs"]["twompp_catalog"]["ZoA_rows_excluded"] == 1
    assert not result.summary["result_policy"][
        "ZoA_rows_allowed_as_matching_observations"
    ]


def test_cln_flag_is_retained_but_radial_redshift_is_not_independent(tmp_path):
    cf4, twompp = _dataset(
        tmp_path,
        [_cf4_row(1, 10.0)],
        [_twompp_row(10, 10.0, cln=1)],
    )

    result = build_crossmatch(cf4, twompp)
    row = _mapping_rows(result)[0]

    assert row["match_class"] == "secure_joint_mark"
    assert row["twompp_Cln"] == "1"
    assert result.summary["result_policy"]["Cln_flag_retained_in_mapping"]
    assert not result.summary["result_policy"][
        "Cln_radial_redshift_allowed_as_independent_observation"
    ]


@pytest.mark.parametrize("which", ["cf4", "twompp"])
def test_bad_header_fails_closed(tmp_path, which):
    cf4, twompp = _dataset(
        tmp_path, [_cf4_row(1, 10.0)], [_twompp_row(10, 10.0)]
    )
    if which == "cf4":
        _write_csv(cf4, CF4_REQUIRED_COLUMNS[:-1], [_cf4_row(1, 10.0)[:-1]])
        match = "missing required columns"
    else:
        bad_header = list(CATALOG_HEADER)
        bad_header[-1] = "wrong_DEC"
        _write_csv(twompp, bad_header, [_twompp_row(10, 10.0)])
        match = "header mismatch"

    with pytest.raises(CrossmatchError, match=match):
        build_crossmatch(cf4, twompp)


@pytest.mark.parametrize("which", ["cf4", "twompp"])
def test_nonfinite_coordinate_fails_closed(tmp_path, which):
    cf4_rows = [_cf4_row(1, "nan" if which == "cf4" else 10.0)]
    twompp_rows = [_twompp_row(10, "inf" if which == "twompp" else 10.0)]
    cf4, twompp = _dataset(tmp_path, cf4_rows, twompp_rows)

    with pytest.raises(CrossmatchError, match="must be finite"):
        build_crossmatch(cf4, twompp)


@pytest.mark.parametrize("which", ["cf4", "twompp"])
def test_duplicate_recno_fails_closed(tmp_path, which):
    cf4_rows = [_cf4_row(1, 10.0), _cf4_row(1 if which == "cf4" else 2, 20.0)]
    twompp_rows = [
        _twompp_row(10, 10.0),
        _twompp_row(10 if which == "twompp" else 20, 20.0),
    ]
    cf4, twompp = _dataset(tmp_path, cf4_rows, twompp_rows)

    with pytest.raises(CrossmatchError, match="recno values must be unique"):
        build_crossmatch(cf4, twompp)


@pytest.mark.parametrize("existing", ["output", "summary"])
def test_publish_refuses_overwrite(tmp_path, existing):
    cf4, twompp = _dataset(
        tmp_path, [_cf4_row(1, 10.0)], [_twompp_row(10, 10.0)]
    )
    output = tmp_path / "mapping.csv"
    summary = tmp_path / "summary.json"
    target = output if existing == "output" else summary
    target.write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        publish_crossmatch(cf4, twompp, output, summary)

    assert target.read_text(encoding="utf-8") == "do not overwrite\n"
    assert not (summary if existing == "output" else output).exists()


def test_publish_writes_complete_summary_bound_to_mapping(tmp_path):
    cf4, twompp = _dataset(
        tmp_path, [_cf4_row(1, 10.0)], [_twompp_row(10, 10.0)]
    )
    output = tmp_path / "mapping.csv"
    summary = tmp_path / "summary.json"

    returned = publish_crossmatch(cf4, twompp, output, summary)
    published = json.loads(summary.read_text(encoding="utf-8"))

    assert returned["status"] == published["status"] == "COMPLETE"
    assert published["mapping"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


def test_summary_publish_race_preserves_competitor_and_removes_owned_mapping(
    tmp_path, monkeypatch
):
    cf4, twompp = _dataset(
        tmp_path, [_cf4_row(1, 10.0)], [_twompp_row(10, 10.0)]
    )
    output = tmp_path / "mapping.csv"
    summary = tmp_path / "summary.json"
    original_link = crossmatch_module.os.link

    def race_summary_link(source, destination):
        if Path(destination) == summary:
            summary.write_text("competitor summary\n", encoding="utf-8")
        return original_link(source, destination)

    monkeypatch.setattr(crossmatch_module.os, "link", race_summary_link)

    with pytest.raises(FileExistsError):
        publish_crossmatch(cf4, twompp, output, summary)

    assert summary.read_text(encoding="utf-8") == "competitor summary\n"
    assert not output.exists()


def test_cleanup_preserves_competitor_replacement_mapping(tmp_path, monkeypatch):
    cf4, twompp = _dataset(
        tmp_path, [_cf4_row(1, 10.0)], [_twompp_row(10, 10.0)]
    )
    output = tmp_path / "mapping.csv"
    summary = tmp_path / "summary.json"
    original_link = crossmatch_module.os.link

    def replace_mapping_then_race_summary(source, destination):
        if Path(destination) == summary:
            output.unlink()
            output.write_text("competitor mapping\n", encoding="utf-8")
            summary.write_text("competitor summary\n", encoding="utf-8")
        return original_link(source, destination)

    monkeypatch.setattr(
        crossmatch_module.os, "link", replace_mapping_then_race_summary
    )

    with pytest.raises(FileExistsError):
        publish_crossmatch(cf4, twompp, output, summary)

    assert output.read_text(encoding="utf-8") == "competitor mapping\n"
    assert summary.read_text(encoding="utf-8") == "competitor summary\n"
