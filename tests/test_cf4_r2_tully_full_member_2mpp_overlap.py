from cf4_r2_tully_full_member_2mpp_overlap import (
    _read_tully_tables,
    crossmatch_coordinates,
    unit_vectors,
)

import numpy as np


def test_galactic_longitude_wrap_is_spherical():
    result = crossmatch_coordinates([359.9999], [12.0], [0.0001], [12.0])
    assert len(result["matches"]) == 1
    expected_arcsec = 0.72 * np.cos(np.deg2rad(12.0))
    assert abs(result["matches"][0][2] - expected_arcsec) < 1e-5


def test_close_pair_is_not_forced_into_one_identity():
    result = crossmatch_coordinates([10.0, 10.0002], [0.0, 0.0], [10.0001], [0.0])
    assert result["matches"] == []
    np.testing.assert_array_equal(result["source_candidate_count"], [1, 1])
    np.testing.assert_array_equal(result["target_candidate_count"], [2])


def test_longitude_wrap_at_pole_has_same_unit_direction():
    vectors = unit_vectors([0.0, 360.0], [-90.0, -90.0])
    np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-14)


def test_published_tully_table_membership_disagreements_are_preserved():
    members, groups, table4_by_pgc = _read_tully_tables()
    table5_by_pgc = {row["PGC"]: row for row in members}

    assert len(members) == 43038
    assert len(groups) == 25474
    assert set(table4_by_pgc) - set(table5_by_pgc) == {9067}
    assert set(table5_by_pgc) - set(table4_by_pgc) == {212964}
    assert {pgc for pgc, row in table5_by_pgc.items() if row["Nest"] == 0} == {
        40621, 41618, 42447
    }
    assert table4_by_pgc[40621] == table4_by_pgc[41618] == table4_by_pgc[42447] == 100002
    assert sum(nest == 200006 for nest in table4_by_pgc.values()) == 49
    assert groups[200006]["Nmb"] == 47


if __name__ == "__main__":
    test_galactic_longitude_wrap_is_spherical()
    test_close_pair_is_not_forced_into_one_identity()
    test_longitude_wrap_at_pole_has_same_unit_direction()
    test_published_tully_table_membership_disagreements_are_preserved()
    print("4 tests passed")
