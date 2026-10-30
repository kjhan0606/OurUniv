import numpy as np
import pytest

from hong2021_v14_multiscale import (
    compose_residual,
    decompose_residual,
    fourier_band_masks,
    inverse_standardized_residual,
    standardize_residual,
)


def test_band_decomposition_exactly_reconstructs_field():
    generator = np.random.default_rng(14)
    field = generator.normal(size=(16, 16, 16)) + 0.37
    location, bands = decompose_residual(field, voxel_mpc_h=0.3125)
    reconstructed = compose_residual(location, bands)
    np.testing.assert_allclose(reconstructed, field, rtol=0, atol=2e-15)
    assert location == pytest.approx(field.mean())
    assert np.max(np.abs(bands.mean(axis=(1, 2, 3)))) < 1e-16


def test_standardize_and_inverse_use_only_supplied_location_scales():
    generator = np.random.default_rng(15)
    field = generator.normal(size=(16, 16, 16)) - 0.12
    scales = np.array([0.4, 0.7, 1.3, 2.1])
    location, standardized = standardize_residual(
        field, predicted_scales=scales, voxel_mpc_h=0.3125
    )
    restored = inverse_standardized_residual(
        standardized,
        predicted_location=location,
        predicted_scales=scales,
        voxel_mpc_h=0.3125,
    )
    np.testing.assert_allclose(restored, field, rtol=0, atol=3e-15)


def test_masks_exhaust_non_dc_modes():
    masks = fourier_band_masks(16, 0.3125)
    membership = masks.sum(axis=0)
    assert membership[0, 0, 0] == 0
    membership[0, 0, 0] = 1
    assert np.all(membership == 1)


def test_multiscale_transform_rejects_invalid_scales():
    field = np.ones((8, 8, 8))
    with pytest.raises(ValueError, match="positive"):
        standardize_residual(
            field,
            predicted_scales=np.array([1.0, 0.0, 1.0, 1.0]),
            voxel_mpc_h=0.3125,
        )
