from __future__ import annotations

import numpy as np
import pytest
import torch
import h5py

from hong2021_v14_multiscale import fourier_band_masks
from hong2021_v18_init import (
    EXPECTED_MODE_COUNTS,
    EXPECTED_MODE_COUNTS_BY_GRID,
    PriorMatchedInitializer,
    measure_band_mode_variances,
    prior_matched_init,
    prior_matched_spectral_std,
    sha256_file,
)


VARIANCE = [1836.1418485297647, 73.04195496322953, 10.368459581198783, 1.1250001849821358]


def test_v18_spectral_std_has_frozen_band_values_and_symmetry() -> None:
    sigma = 40.0
    value = prior_matched_spectral_std(
        64, 0.3125, sigma, VARIANCE, device=torch.device("cpu")
    )
    masks = fourier_band_masks(64, 0.3125)
    assert tuple(int(mask.sum()) for mask in masks) == EXPECTED_MODE_COUNTS
    assert value.dtype == torch.float64
    assert value[0, 0, 0].item() == sigma
    for mask, variance in zip(masks, VARIANCE, strict=True):
        expected = np.sqrt(sigma**2 + variance)
        actual = value[torch.from_numpy(mask)]
        assert torch.all(actual == actual[0])
        assert actual[0].item() == expected
    negative = torch.remainder(-torch.arange(64), 64)
    reflected = value[negative][:, negative][:, :, negative]
    assert torch.equal(value, reflected)


def test_v18_prior_matched_init_is_real_and_has_expected_mode_scaling() -> None:
    generator = torch.Generator().manual_seed(180061)
    noise = torch.randn((2, 1, 64, 64, 64), generator=generator)
    std = prior_matched_spectral_std(
        64, 0.3125, 40.0, VARIANCE, device=torch.device("cpu")
    )
    initialized, imaginary_ratio = prior_matched_init(noise, std)
    assert initialized.shape == noise.shape
    assert initialized.dtype == noise.dtype
    assert torch.isfinite(initialized).all()
    assert imaginary_ratio <= 1.0e-12
    before = torch.fft.fftn(noise.double(), dim=(-3, -2, -1), norm="ortho")
    after = torch.fft.fftn(initialized.double(), dim=(-3, -2, -1), norm="ortho")
    masks = fourier_band_masks(64, 0.3125)
    for mask, variance in zip(masks, VARIANCE, strict=True):
        selected = torch.from_numpy(mask)
        after_power = (after.real.square() + after.imag.square())[:, :, selected]
        before_power = (before.real.square() + before.imag.square())[:, :, selected]
        ratio = after_power.sum() / before_power.sum()
        expected = 40.0**2 + variance
        assert torch.abs(ratio - expected) / expected < 2.0e-7


def test_v18_inference_grid_mapping_has_frozen_counts_symmetry_and_scaling() -> None:
    grid = 80
    sigma = 40.0
    value = prior_matched_spectral_std(
        grid, 0.3125, sigma, VARIANCE, device=torch.device("cpu")
    )
    masks = fourier_band_masks(grid, 0.3125)
    assert tuple(int(mask.sum()) for mask in masks) == EXPECTED_MODE_COUNTS_BY_GRID[grid]
    assert sum(int(mask.sum()) for mask in masks) == grid**3 - 1
    negative = torch.remainder(-torch.arange(grid), grid)
    reflected = value[negative][:, negative][:, :, negative]
    assert torch.equal(value, reflected)
    generator = torch.Generator().manual_seed(180080)
    noise = torch.randn((1, 1, grid, grid, grid), generator=generator)
    initialized, imaginary_ratio = prior_matched_init(noise, value)
    assert imaginary_ratio <= 1.0e-12
    before = torch.fft.fftn(noise.double(), dim=(-3, -2, -1), norm="ortho")
    after = torch.fft.fftn(initialized.double(), dim=(-3, -2, -1), norm="ortho")
    for mask, variance in zip(masks, VARIANCE, strict=True):
        selected = torch.from_numpy(mask)
        after_power = (after.real.square() + after.imag.square())[:, :, selected]
        before_power = (before.real.square() + before.imag.square())[:, :, selected]
        ratio = after_power.sum() / before_power.sum()
        expected = sigma**2 + variance
        assert torch.abs(ratio - expected) / expected < 2.0e-7


def test_v18_initializer_consumes_no_rng_and_rejects_wrong_shape() -> None:
    std = prior_matched_spectral_std(
        64, 0.3125, 40.0, VARIANCE, device=torch.device("cpu")
    )
    initializer = PriorMatchedInitializer(std, maximum_imaginary_ratio=1.0e-12)
    first = torch.Generator().manual_seed(180062)
    second = torch.Generator().manual_seed(180062)
    noise = torch.randn((1, 1, 64, 64, 64), generator=first)
    reference = torch.randn((1, 1, 64, 64, 64), generator=second)
    assert torch.equal(noise, reference)
    state = first.get_state().clone()
    initializer(noise)
    assert torch.equal(first.get_state(), state)
    assert torch.equal(first.get_state(), second.get_state())
    with pytest.raises(ValueError, match="shape or dtype"):
        prior_matched_init(noise, std.float())


def test_v18_measurement_rejects_wrong_cache_schema(tmp_path) -> None:
    path = tmp_path / "wrong_schema.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "wrong"
        handle.create_dataset("standardized_residual", shape=(1, 1, 64, 64, 64), dtype="f4")
    first = {
        "path": str(path), "sha256": sha256_file(path),
        "objects": 1, "domain_attribute": "TNG100",
    }
    sources = {
        "TNG100": first,
        "SIMBA": first,
        "Swift": first,
    }
    with pytest.raises(ValueError, match="wrong standardized-cache schema"):
        measure_band_mode_variances(sources)
