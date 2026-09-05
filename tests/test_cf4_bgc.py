import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cf4_bgc import (  # noqa: E402
    bgc_transform_from_reference,
    fixed_count_reference_median,
)
from cf4_linear_cr import raw_index_hash_holdout  # noqa: E402


def test_reference_median_uses_fixed_shifted_window():
    ref_cz = np.arange(10.0)
    ref_x = 10.0 * ref_cz
    targets = np.array([-2.0, 4.2, 20.0])
    med, count = fixed_count_reference_median(ref_cz, ref_x, targets, 5)
    np.testing.assert_allclose(med, [20.0, 50.0, 70.0])
    np.testing.assert_array_equal(count, [5, 5, 5])


def test_reference_bgc_does_not_leak_target_distance_into_medians():
    ref_cz = np.linspace(500.0, 30000.0, 101)
    ref_dist = ref_cz / 75.0 + np.sin(ref_cz / 2000.0)
    target_cz = np.array([3000.0, 9000.0, 17000.0])
    target_dist = target_cz / 75.0 + np.array([2.0, -3.0, 1.0])
    sigln = np.full(3, 0.12)

    first = bgc_transform_from_reference(
        target_cz,
        target_dist,
        sigln,
        ref_cz,
        ref_dist,
        h0=75.0,
        window=21,
    )
    changed_dist = target_dist.copy()
    changed_dist[1] *= 1.25
    second = bgc_transform_from_reference(
        target_cz,
        changed_dist,
        sigln,
        ref_cz,
        ref_dist,
        h0=75.0,
        window=21,
    )

    np.testing.assert_allclose(first.distance_median, second.distance_median)
    np.testing.assert_allclose(first.velocity_median, second.velocity_median)
    np.testing.assert_allclose(first.velocity[[0, 2]], second.velocity[[0, 2]])
    expected_change = -target_cz[1] * np.log(changed_dist[1] / target_dist[1])
    np.testing.assert_allclose(
        second.velocity[1] - first.velocity[1], expected_change
    )


def test_raw_index_hash_holdout_is_estimator_independent():
    first_idx = np.arange(0, 10000, dtype=np.int64)
    second_idx = np.arange(500, 10500, dtype=np.int64)
    first = raw_index_hash_holdout(first_idx, 0.2, 20260817)
    second = raw_index_hash_holdout(second_idx, 0.2, 20260817)
    common = np.intersect1d(first_idx[first], second_idx[second])
    expected = first_idx[first & np.isin(first_idx, second_idx)]
    np.testing.assert_array_equal(common, expected)
    assert 0.18 < first.mean() < 0.22
