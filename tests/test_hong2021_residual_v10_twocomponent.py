from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_residual_diffusion import ConditionalResidualUNet
from hong2021_residual_v10_twocomponent import correction_forward


def test_correction_forward_has_exact_cube_dc_null() -> None:
    torch.manual_seed(4)
    model = ConditionalResidualUNet(base_channels=8)
    condition = torch.randn(2, 4, 8, 8, 8)
    value = correction_forward(model, condition)
    torch.testing.assert_close(
        value.mean(dim=(-3, -2, -1)), torch.zeros(2, 1), atol=1e-7, rtol=0
    )
