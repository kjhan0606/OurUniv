from __future__ import annotations

import torch

from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_v27_flow import (
    ParentAlignedConditionalHaarSplineFlow,
    depth_to_space_3d,
    space_to_depth_3d,
)


def _flow(levels: int = 3) -> ParentAlignedConditionalHaarSplineFlow:
    return ParentAlignedConditionalHaarSplineFlow(
        detail_mean=[[0.0] * 7] * levels,
        detail_std=[[1.0] * 7] * levels,
        context_mean=[0.0] * len(FEATURE_NAMES),
        context_std=[1.0] * len(FEATURE_NAMES),
        hidden_channels=8,
        levels=levels,
        couplings=2,
    )


def test_space_depth_roundtrip_and_binary_phase_order():
    value = torch.arange(2 * 4 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4, 4)
    packed = space_to_depth_3d(value)
    assert packed.shape == (2, 8, 2, 2, 2)
    assert torch.equal(depth_to_space_3d(packed), value)
    assert torch.equal(packed[0, :, 0, 0, 0], value[0, 0, :2, :2, :2].reshape(-1))


def test_parent_aligned_context_retains_coarsest_reflection_phase():
    model = _flow().eval()
    condition = torch.randn(
        1, 4, 8, 8, 8, generator=torch.Generator().manual_seed(70)
    )
    reflected = condition.flip(-1)
    lowpass = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64)
    original = model.scale_context(
        condition, lowpass, model.standardized_global_context(condition)
    )
    changed = model.scale_context(
        reflected, lowpass, model.standardized_global_context(reflected)
    )
    assert original.shape == (1, 41, 1, 1, 1)
    assert torch.max(torch.abs(original - changed)) > 0.1
    original_phases = original[:, :32].reshape(1, 4, 2, 2, 2, 1, 1, 1)
    reflected_phases = changed[:, :32].reshape(1, 4, 2, 2, 2, 1, 1, 1)
    assert torch.allclose(reflected_phases, original_phases.flip(4), atol=2.0e-6)


def test_parent_aligned_flow_log_prob_sampling_and_exact_dc():
    model = _flow().eval()
    condition = torch.randn(
        2, 4, 8, 8, 8, generator=torch.Generator().manual_seed(71)
    )
    latent = torch.randn(
        2, 1, 8, 8, 8, generator=torch.Generator().manual_seed(72)
    )
    latent -= latent.mean(dim=(-3, -2, -1), keepdim=True)
    log_prob, diagnostic = model.log_prob(latent, condition)
    assert torch.isfinite(log_prob).all()
    assert diagnostic["scale_log_prob_coarse_to_fine"].shape == (2, 3)
    sample = model.sample(
        condition, generator=torch.Generator().manual_seed(73)
    )
    assert sample.shape == latent.shape
    assert torch.max(torch.abs(sample.mean(dim=(-3, -2, -1)))) < 2.0e-8


def test_registered_v27_parameter_count():
    model = ParentAlignedConditionalHaarSplineFlow(
        detail_mean=[[0.0] * 7] * 6,
        detail_std=[[1.0] * 7] * 6,
        context_mean=[0.0] * len(FEATURE_NAMES),
        context_std=[1.0] * len(FEATURE_NAMES),
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_787_032
