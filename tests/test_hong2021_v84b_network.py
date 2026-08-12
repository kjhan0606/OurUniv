from __future__ import annotations

import numpy as np
import torch

from hong2021_v84b_network import (
    ConditionalSplicedTailUNet,
    INITIAL_LOWER_SCALE,
    INITIAL_TAIL_MASS,
    INITIAL_UPPER_SCALE,
    PARAMETERS,
    conditional_cdf,
    conditional_icdf,
    conditional_log_probability,
    spliced_parameters,
    upper_physical_second_moment_margin,
)


def test_spliced_initialization_and_parameter_shape() -> None:
    model = ConditionalSplicedTailUNet(base_channels=4)
    model.eval()
    observed = model(torch.zeros(1, 7, 16, 16, 16))
    assert observed.shape == (1, PARAMETERS, 16, 16, 16)
    _, weights, lower_scale, upper_scale = spliced_parameters(observed)
    assert torch.allclose(weights[:, 0], torch.full_like(weights[:, 0], INITIAL_TAIL_MASS))
    assert torch.allclose(weights[:, 2], torch.full_like(weights[:, 2], INITIAL_TAIL_MASS))
    assert torch.allclose(lower_scale, torch.full_like(lower_scale, INITIAL_LOWER_SCALE))
    assert torch.allclose(upper_scale, torch.full_like(upper_scale, INITIAL_UPPER_SCALE))


def test_spliced_cdf_icdf_round_trip_in_both_tails_and_centre() -> None:
    model = ConditionalSplicedTailUNet(base_channels=4)
    parameters = model.output.bias.detach().reshape(1, PARAMETERS, 1, 1, 1).expand(
        1, PARAMETERS, 2, 2, 2
    )
    uniform = torch.tensor(
        [1.0e-4, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999],
        dtype=torch.float32,
    ).reshape(1, 1, 2, 2, 2)
    residual = conditional_icdf(parameters, uniform)
    recovered = conditional_cdf(parameters, residual)
    assert torch.max(torch.abs(recovered - uniform)).item() < 2.0e-5
    assert torch.isfinite(conditional_log_probability(parameters, residual)).all()


def test_spliced_density_integrates_to_one() -> None:
    model = ConditionalSplicedTailUNet(base_channels=4)
    parameters = model.output.bias.detach().reshape(1, PARAMETERS, 1, 1, 1)
    grid = torch.linspace(-12.0, 12.0, 120001).reshape(-1, 1, 1, 1, 1)
    expanded = parameters.expand(len(grid), -1, -1, -1, -1)
    density = torch.exp(conditional_log_probability(expanded, grid))[:, 0, 0, 0, 0]
    integral = torch.trapezoid(density, grid[:, 0, 0, 0, 0])
    assert abs(float(integral) - 1.0) < 2.0e-4


def test_upper_tail_bound_guarantees_finite_physical_second_moment() -> None:
    assert upper_physical_second_moment_margin(0.09877202271987233) > 0.3
