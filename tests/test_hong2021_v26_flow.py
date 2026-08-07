from __future__ import annotations

import torch

from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_v26_flow import ConditionalHaarSplineFlow, ConditionalSplineCoupling3D


def test_identity_initialized_coupling_roundtrip_and_logdet():
    layer = ConditionalSplineCoupling3D(
        (0, 1, 2), context_channels=5, hidden_channels=8, bins=8
    )
    value = torch.randn(2, 7, 4, 4, 4, generator=torch.Generator().manual_seed(51))
    context = torch.randn(2, 5, 4, 4, 4, generator=torch.Generator().manual_seed(52))
    transformed, forward_logdet = layer(value, context)
    recovered, inverse_logdet = layer.inverse(transformed, context)
    assert torch.max(torch.abs(transformed - value)) < 2.0e-6
    assert torch.max(torch.abs(recovered - value)) < 3.0e-6
    assert torch.max(torch.abs(forward_logdet + inverse_logdet)) < 2.0e-5


def _small_flow() -> ConditionalHaarSplineFlow:
    return ConditionalHaarSplineFlow(
        detail_mean=[[0.0] * 7] * 3,
        detail_std=[[1.0] * 7] * 3,
        context_mean=[0.0] * len(FEATURE_NAMES),
        context_std=[1.0] * len(FEATURE_NAMES),
        hidden_channels=8,
        levels=3,
        couplings=2,
    )


def test_flow_log_prob_is_finite_and_sampling_has_exact_dc():
    model = _small_flow()
    latent = torch.randn(2, 1, 8, 8, 8, generator=torch.Generator().manual_seed(53))
    latent -= latent.mean(dim=(-3, -2, -1), keepdim=True)
    condition = torch.randn(2, 4, 8, 8, 8, generator=torch.Generator().manual_seed(54))
    log_prob, diagnostics = model.log_prob(latent, condition)
    assert log_prob.shape == (2,)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(diagnostics["base_log_prob"]).all()
    assert diagnostics["scale_log_prob_coarse_to_fine"].shape == (2, 3)
    sample = model.sample(
        condition, generator=torch.Generator().manual_seed(55)
    )
    assert sample.shape == latent.shape
    assert torch.max(torch.abs(sample.mean(dim=(-3, -2, -1)))) < 2.0e-8


def test_sample_is_seed_reproducible():
    model = _small_flow().eval()
    condition = torch.randn(1, 4, 8, 8, 8, generator=torch.Generator().manual_seed(56))
    first = model.sample(condition, generator=torch.Generator().manual_seed(57))
    second = model.sample(condition, generator=torch.Generator().manual_seed(57))
    assert torch.equal(first, second)


def test_sampling_reports_roundoff_dc_projection():
    model = _small_flow().eval()
    condition = torch.randn(2, 4, 8, 8, 8, generator=torch.Generator().manual_seed(58))
    sample, diagnostics = model.sample_with_diagnostics(
        condition, generator=torch.Generator().manual_seed(59)
    )
    assert diagnostics["pre_center_mean"].shape == (2,)
    assert torch.max(torch.abs(diagnostics["post_center_mean"])) < 2.0e-8
    assert torch.max(torch.abs(sample.mean(dim=(-3, -2, -1)))) < 2.0e-8
