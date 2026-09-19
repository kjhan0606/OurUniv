from __future__ import annotations

import copy

import torch
from torch import nn

from hong2021_residual_v9_tail import tail_balanced_edm_loss
from hong2021_v19_edm import P_MEAN, P_STD
from hong2021_v19_preflight import KS_MAXIMUM, fixed_lognormal_ks, rng_stream_self_check


class TinyDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.125))

    def forward(
        self, noisy: torch.Tensor, condition: torch.Tensor, noise_level: torch.Tensor
    ) -> torch.Tensor:
        return self.scale * noisy + 0.01 * condition[:, :1]


def _one_step(model, generator, *, p_mean, p_std):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    residual = torch.linspace(-1, 1, 2 * 4**3).reshape(2, 1, 4, 4, 4)
    condition = torch.flip(residual, dims=(-1,))
    truth = residual * 0.2
    weights = torch.ones(5)
    optimizer.zero_grad(set_to_none=True)
    values = tail_balanced_edm_loss(
        model, residual, condition, truth, weights, generator,
        2.006363569349486, p_mean, p_std,
    )
    values[0].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return values, copy.deepcopy(model.state_dict()), generator.get_state().clone()


def test_v19_fixed_lognormal_draw_preflight() -> None:
    assert fixed_lognormal_ks() < KS_MAXIMUM


def test_v19_noise_change_preserves_e3_rng_stream() -> None:
    assert rng_stream_self_check() is True


def test_v19_shared_loss_is_e3_bit_identical_when_given_e3_constants() -> None:
    first = TinyDenoiser()
    second = copy.deepcopy(first)
    values_a, state_a, rng_a = _one_step(
        first, torch.Generator().manual_seed(190073),
        p_mean=0.18549829030348333, p_std=1.2,
    )
    values_b, state_b, rng_b = _one_step(
        second, torch.Generator().manual_seed(190073),
        p_mean=0.18549829030348333, p_std=1.2,
    )
    assert all(torch.equal(a, b) for a, b in zip(values_a, values_b, strict=True))
    assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)
    assert torch.equal(rng_a, rng_b)


def test_v19_actual_constants_change_values_but_not_rng_consumption() -> None:
    first = TinyDenoiser()
    second = copy.deepcopy(first)
    values_a, _, rng_a = _one_step(
        first, torch.Generator().manual_seed(190074),
        p_mean=0.18549829030348333, p_std=1.2,
    )
    values_b, _, rng_b = _one_step(
        second, torch.Generator().manual_seed(190074),
        p_mean=P_MEAN, p_std=P_STD,
    )
    assert not torch.equal(values_a[0], values_b[0])
    assert torch.equal(rng_a, rng_b)
