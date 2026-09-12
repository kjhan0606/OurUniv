from __future__ import annotations

import numpy as np
import torch

from hong2021_residual_diffusion import ConditionalResidualUNet
from hong2021_v14_mean_correction import (
    DOMAINS,
    correction_forward,
    select_correction_candidate,
    source_balanced_tail_weights,
)


def test_three_source_tail_weights_ignore_unequal_source_sizes() -> None:
    counts = {
        "TNG100": np.array([1000, 100, 10, 5, 1]),
        "SIMBA": np.array([10, 100, 1000, 5, 1]),
        "Swift-EAGLE": np.array([10, 100, 10, 500, 100]),
    }
    result = source_balanced_tail_weights(counts)
    probability = np.asarray(result["probability"]["equal_source"])
    weights = np.asarray(result["weights"])
    assert result["source_weight"] == 1.0 / 3.0
    np.testing.assert_allclose(np.sum(probability * weights), 1.0, rtol=1e-7)


def test_correction_forward_has_exact_zero_dc() -> None:
    model = ConditionalResidualUNet(base_channels=4)
    condition = torch.randn(2, 4, 8, 8, 8)
    actual = correction_forward(model, condition)
    torch.testing.assert_close(
        actual.mean(dim=(-3, -2, -1)), torch.zeros(2, 1), atol=2e-7, rtol=0
    )


def test_selection_requires_improvement_in_every_domain_and_minimax() -> None:
    baseline = {name: {"combined": 1.0} for name in DOMAINS}
    history = [
        {"step": 1000, "validation": {name: {"combined": value} for name, value in zip(DOMAINS, (0.8, 0.9, 0.95), strict=True)}},
        {"step": 3000, "validation": {name: {"combined": value} for name, value in zip(DOMAINS, (0.7, 1.01, 0.7), strict=True)}},
        {"step": 5000, "validation": {name: {"combined": value} for name, value in zip(DOMAINS, (0.9, 0.9, 0.9), strict=True)}},
    ]
    result = select_correction_candidate(history, baseline, [1000, 3000, 5000])
    assert result["selected_step"] == 5000
    assert result["candidates"][1]["improves_every_domain"] is False
