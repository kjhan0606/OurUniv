from __future__ import annotations

import torch

from hong2021_v26_haar import (
    detail_dimensions,
    haar_analysis,
    haar_pyramid,
    haar_synthesis,
    inverse_haar_pyramid,
)


def test_one_level_roundtrip_and_parseval():
    value = torch.randn(
        2, 1, 8, 8, 8,
        generator=torch.Generator().manual_seed(18),
        dtype=torch.float64,
    )
    lowpass, details = haar_analysis(value)
    recovered = haar_synthesis(lowpass, details)
    assert torch.max(torch.abs(value - recovered)) < 2.0e-15
    input_energy = value.double().square().sum()
    coefficient_energy = lowpass.double().square().sum() + details.double().square().sum()
    assert torch.abs(input_energy - coefficient_energy) / input_energy < 2.0e-15


def test_six_levels_isolate_dc_and_cover_every_other_dimension():
    value = torch.randn(1, 1, 64, 64, 64, generator=torch.Generator().manual_seed(19))
    value -= value.mean(dim=(-3, -2, -1), keepdim=True)
    lowpass, details = haar_pyramid(value)
    assert lowpass.shape == (1, 1, 1, 1, 1)
    assert abs(float(lowpass)) < 2.0e-5
    assert [item.numel() for item in details] == detail_dimensions()
    assert sum(detail_dimensions()) == 64**3 - 1
    recovered = inverse_haar_pyramid(lowpass, details)
    assert torch.max(torch.abs(value - recovered)) < 2.0e-6


def test_constant_field_enters_only_coarsest_dc():
    value = torch.full((1, 1, 64, 64, 64), 0.125, dtype=torch.float64)
    lowpass, details = haar_pyramid(value)
    assert all(float(item.abs().max()) < 1.0e-14 for item in details)
    assert torch.allclose(
        lowpass, torch.tensor([[[[[64.0]]]]], dtype=torch.float64), atol=4.0e-13
    )
