from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_prepare_simba import (
    conservative_resample_cube,
    choose_observer,
    extract_periodic_cube,
    overlap_matrix,
)


def test_overlap_matrix_is_constant_preserving() -> None:
    weights = overlap_matrix(256, 80)
    assert weights.shape == (80, 256)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=2.0e-14)
    np.testing.assert_allclose(weights @ np.ones(256), 1.0, atol=2.0e-14)


def test_conservative_resampling_preserves_constant_and_mean() -> None:
    generator = np.random.default_rng(7)
    value = generator.uniform(0.1, 5.0, size=(16, 16, 16)).astype(np.float32)
    result = conservative_resample_cube(value, 5)
    assert result.shape == (5, 5, 5)
    np.testing.assert_allclose(result.mean(), value.mean(), rtol=2.0e-6)
    constant = conservative_resample_cube(np.full((16, 16, 16), 3.5), 5)
    np.testing.assert_allclose(constant, 3.5, atol=1.0e-6)


def test_periodic_cube_wraps_each_axis() -> None:
    value = np.arange(8**3).reshape(8, 8, 8)
    result = extract_periodic_cube(value, np.array([7, -1, 6]), size=3)
    expected = value[np.ix_([7, 0, 1], [7, 0, 1], [6, 7, 0])]
    np.testing.assert_array_equal(result, expected)


def test_observer_selection_uses_only_mass_and_stable_tie_break() -> None:
    mass = np.array([1.0e9, 5.0e10, np.sqrt(4.0e10 * 1.0e11), 8.0e10])
    assert choose_observer(mass) == 2
    mass = np.array([6.0e10, 6.0e10])
    assert choose_observer(mass) == 0
