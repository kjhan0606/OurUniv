import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cf4_r2_tempel_parent import (  # noqa: E402
    nearest_random_arcsec,
    parent_counts,
    parent_indices,
    unit_vectors,
    unique_velocity_matches,
)


def test_group_zero_rows_remain_distinct_singletons():
    parent, kind, ident = parent_indices(
        np.array([1, 1, 0, 0]), np.array([8, 9, 10, 11]), group_count=2)
    np.testing.assert_array_equal(parent, [0, 0, 2, 3])
    np.testing.assert_array_equal(kind, [0, 0, 1, 1])
    np.testing.assert_array_equal(ident, [1, 2, 10, 11])


def test_parent_counts_preserve_zero_and_multiple_parent_members():
    got = parent_counts(np.array([0, 0, 1, 2]), 4,
                        np.array([True, False, True, True]))
    np.testing.assert_array_equal(got, [1, 1, 1, 0])


def test_unit_vectors_and_zero_separation():
    xyz = unit_vectors(np.array([0.0, 90.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(xyz, [[1, 0, 0], [0, 1, 0]], atol=1e-14)
    d = nearest_random_arcsec(cKDTree(xyz), xyz)
    np.testing.assert_allclose(d, 0.0, atol=1e-8)


def test_velocity_link_is_one_to_one_and_rejects_reused_tempel_member():
    # Two source points competing for one target must not silently choose one.
    tempel_xyz = unit_vectors(np.array([0.0, 1.0]), np.zeros(2))
    source_xyz = unit_vectors(np.array([0.0001, 0.0002, 1.0001]), np.zeros(3))
    source_i, tempel_i, diag = unique_velocity_matches(
        tempel_xyz, np.array([1000.0, 2000.0]), source_xyz,
        np.array([1001.0, 1002.0, 2001.0]), max_sep_arcsec=10.0, max_dv_km_s=50.0)
    np.testing.assert_array_equal(source_i, [2])
    np.testing.assert_array_equal(tempel_i, [1])
    assert diag["ambiguous_tempel_rows"] == 1
    assert diag["source_rows_rejected_by_reuse"] == 2


def test_velocity_limit_prevents_sky_only_false_link():
    xyz = unit_vectors(np.array([15.0]), np.array([-3.0]))
    source_i, tempel_i, diag = unique_velocity_matches(
        xyz, np.array([5000.0]), xyz.copy(), np.array([6000.0]), max_dv_km_s=300.0)
    assert len(source_i) == len(tempel_i) == 0
    assert diag["unique_pairs"] == 0


if __name__ == "__main__":
    tests = [value for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"PASS {len(tests)} Tempel parent source tests")
