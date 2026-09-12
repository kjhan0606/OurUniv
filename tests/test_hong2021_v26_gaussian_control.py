from __future__ import annotations

import numpy as np
import pytest
import torch

from hong2021_v14_multiscale import fourier_band_masks


@pytest.fixture(scope="module")
def module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/hong2021_v26_gaussian_control.py"
    spec = importlib.util.spec_from_file_location("hong2021_v26_gaussian_control", path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def test_spectral_standard_deviation_assigns_only_non_dc_modes(module):
    variance = [4.0, 9.0, 16.0, 25.0]
    result = module.latent_spectral_standard_deviation(
        64, 0.3125, variance, device=torch.device("cpu")
    ).numpy()
    assert result.dtype == np.float64
    assert result[0, 0, 0] == 0.0
    assert np.count_nonzero(result) == 64**3 - 1
    for mask, expected in zip(
        fourier_band_masks(64, 0.3125), np.sqrt(variance), strict=True
    ):
        assert np.all(result[mask] == expected)


def test_latent_gaussian_is_exactly_centered_and_reproducible(module):
    device = torch.device("cpu")
    spectral = module.latent_spectral_standard_deviation(
        64, 0.3125, [1.0, 1.0, 1.0, 1.0], device=device
    )
    first, first_imaginary = module.sample_latent_gaussian(
        ensemble=2,
        grid=64,
        spectral_std=spectral,
        generator=torch.Generator().manual_seed(19381),
        device=device,
    )
    second, second_imaginary = module.sample_latent_gaussian(
        ensemble=2,
        grid=64,
        spectral_std=spectral,
        generator=torch.Generator().manual_seed(19381),
        device=device,
    )
    assert torch.equal(first, second)
    assert first_imaginary == second_imaginary
    assert torch.max(torch.abs(first.double().mean(dim=(-3, -2, -1)))) < 1.0e-8
    assert first_imaginary < 1.0e-12


def test_invalid_latent_variance_is_rejected(module):
    with pytest.raises(ValueError, match="finite and positive"):
        module.latent_spectral_standard_deviation(
            64, 0.3125, [1.0, 0.0, 1.0, 1.0], device=torch.device("cpu")
        )
