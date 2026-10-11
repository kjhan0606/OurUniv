from __future__ import annotations

import torch

from hong2021_residual_v6 import sample_edm
from scripts.hong2021_v24_terminal_sampler_audit import (
    FLOAT32_REPLAY_ATOL,
    replay_is_numerically_identical,
    sample_edm_with_trace,
)


class ZeroNetwork(torch.nn.Module):
    def forward(
        self, noisy: torch.Tensor, condition: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        del condition, noise
        return torch.zeros_like(noisy)


def test_traced_sampler_is_bit_identical_to_frozen_sampler() -> None:
    model = ZeroNetwork()
    condition = torch.zeros((2, 3, 4, 4, 4), dtype=torch.float32)
    first = torch.Generator().manual_seed(240041)
    second = torch.Generator().manual_seed(240041)
    expected = sample_edm(
        model, condition, first, 4, 0.002, 4.0, 7.0, 1.0
    )
    actual, trace = sample_edm_with_trace(
        model,
        condition,
        second,
        4,
        0.002,
        4.0,
        7.0,
        1.0,
        coordinates=[(0, 0, 0), (1, 1, 1)],
        trace_steps=(0, 1, 2, 4),
    )
    assert torch.equal(actual, expected)
    assert [row["step"] for row in trace if row["phase"] == "state" and row["member"] == 0] == [0, 1, 2, 4]
    assert torch.equal(first.get_state(), second.get_state())


def test_replay_integrity_allows_only_one_float32_epsilon() -> None:
    assert replay_is_numerically_identical(0.5 * FLOAT32_REPLAY_ATOL)
    assert replay_is_numerically_identical(FLOAT32_REPLAY_ATOL)
    assert not replay_is_numerically_identical(1.01 * FLOAT32_REPLAY_ATOL)
