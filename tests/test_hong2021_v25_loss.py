from __future__ import annotations

import copy

import torch

from hong2021_residual_v9_tail import tail_balanced_edm_loss
from hong2021_v25_loss import proper_unweighted_edm_loss


class ScaleNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(
        self, noisy: torch.Tensor, condition: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        del condition, noise
        return self.scale * noisy


def test_v25_optimized_gradient_is_exactly_the_unweighted_gradient() -> None:
    generator = torch.Generator().manual_seed(250013)
    residual = torch.randn((2, 1, 4, 4, 4), generator=generator)
    condition = torch.randn((2, 4, 4, 4, 4), generator=generator)
    truth = torch.randn((2, 1, 4, 4, 4), generator=generator)
    weights = torch.tensor([0.9, 1.0, 1.3, 2.2, 4.0])
    proper_model = ScaleNetwork()
    inherited_model = copy.deepcopy(proper_model)
    proper_generator = torch.Generator().manual_seed(250014)
    inherited_generator = torch.Generator().manual_seed(250014)
    proper, diagnostic = proper_unweighted_edm_loss(
        proper_model,
        residual,
        condition,
        truth,
        weights,
        proper_generator,
        1.0,
        -0.5,
        1.2,
    )
    _, inherited_unweighted, _ = tail_balanced_edm_loss(
        inherited_model,
        residual,
        condition,
        truth,
        weights,
        inherited_generator,
        1.0,
        -0.5,
        1.2,
    )
    assert torch.equal(proper, inherited_unweighted)
    assert diagnostic.requires_grad is False
    proper.backward()
    inherited_unweighted.backward()
    assert torch.equal(proper_model.scale.grad, inherited_model.scale.grad)
    assert torch.equal(proper_generator.get_state(), inherited_generator.get_state())
