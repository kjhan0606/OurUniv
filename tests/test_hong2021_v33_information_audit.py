import numpy as np

from hong2021_v33_information_audit import (
    FEATURE_NAMES,
    block_information_rows,
    source_balanced_standardization,
)
from hong2021_v31_copula import DOMAIN_ORDER


def test_exact_pooled_dispersion_adds_native_within_cell_second_moment():
    count = np.zeros((64, 64, 64), dtype=np.float64)
    velocity = np.zeros_like(count)
    dispersion = np.zeros_like(count)
    count[0, 0, 0] = 2
    velocity[0, 0, 0] = 5
    dispersion[0, 0, 0] = np.sqrt(50.0)
    backbone = np.zeros_like(count)
    truth = np.ones_like(count) * 0.2
    feature, target, diagnostic = block_information_rows(
        count, velocity, dispersion, backbone, truth, 4
    )
    assert feature.shape == (16, 16, 16, len(FEATURE_NAMES))
    assert np.isclose(feature[0, 0, 0, 1], 5.0)
    assert feature[0, 0, 0, 5] == 0
    assert np.isclose(feature[0, 0, 0, 6], 5.0)
    assert np.allclose(target, 0.2)
    assert diagnostic["blocks_with_strict_intrinsic_increment_fraction"] == 1 / 16**3


def test_exact_and_recoverable_match_when_no_native_cell_is_multiple():
    count = np.zeros((64, 64, 64), dtype=np.float64)
    velocity = np.zeros_like(count)
    dispersion = np.zeros_like(count)
    count[0, 0, 0] = 1
    count[1, 0, 0] = 1
    velocity[1, 0, 0] = 10
    feature, _, _ = block_information_rows(
        count, velocity, dispersion, np.zeros_like(count), np.zeros_like(count), 4
    )
    assert np.allclose(feature[..., 5], feature[..., 6])
    assert np.isclose(feature[0, 0, 0, 5], 5.0)


def test_v33_standardization_accepts_thirteen_features():
    rows = {
        domain: np.arange((index + 2) * len(FEATURE_NAMES), dtype=np.float64).reshape(
            index + 2, len(FEATURE_NAMES)
        )
        for index, domain in enumerate(DOMAIN_ORDER)
    }
    mean, std = source_balanced_standardization(rows)
    assert mean.shape == (len(FEATURE_NAMES),)
    assert std.shape == (len(FEATURE_NAMES),)
    assert np.all(std > 0)
