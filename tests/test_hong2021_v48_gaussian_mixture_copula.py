import hashlib
from pathlib import Path

import numpy as np
import torch

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v48_development_gate import classify
from hong2021_v48_network import (
    INITIAL_BIASES,
    LocalMixtureUNet,
    gaussian_mixture_cdf,
    gaussian_mixture_inverse,
    gaussian_mixture_log_probability,
    mixture_parameters,
    parameter_count,
)
from hong2021_v48_sample import ENSEMBLE_SCHEMA
from hong2021_v48_train import (
    PARAMETERS,
    PROGRAM_SHA256,
    _finite_physical_log_moments,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_firewall_and_evaluator_schema() -> None:
    path = REPO / "config/hong2021_v48_identifiable_gaussian_mixture_copula_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"spatial_rank_transport": false' in text
    assert '"global_residual_scaling": false' in text
    assert '"posthoc_Ak": false' in text
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS


def test_network_shape_parameter_count_and_identifiable_initial_output() -> None:
    model = LocalMixtureUNet()
    assert parameter_count(model) == PARAMETERS
    with torch.no_grad():
        output = model(torch.zeros(1, 7, 16, 16, 16))
    assert output.shape == (1, 15, 16, 16, 16)
    expected = torch.tensor(INITIAL_BIASES).reshape(1, 15, 1, 1, 1)
    assert torch.count_nonzero(model.output.weight).item() == 0
    assert torch.max(torch.abs(output - expected)).item() <= 1.0e-7
    logits, locations, scales = mixture_parameters(expected)
    weights = torch.softmax(logits, dim=1)
    mean = torch.sum(weights * locations, dim=1)
    variance = torch.sum(
        weights
        * (torch.square(locations - mean[:, None]) + torch.square(scales)),
        dim=1,
    )
    assert abs(float(mean)) <= 1.0e-6
    assert abs(float(variance) - 1.0) <= 1.0e-6
    assert torch.unique(locations).numel() == 5


def test_identifiable_initialization_breaks_v44_gradient_symmetry() -> None:
    torch.manual_seed(144044)
    target = torch.randn(1, 1, 4, 4, 4)
    old = torch.zeros(15, requires_grad=True)
    old_loss = -gaussian_mixture_log_probability(
        old.reshape(1, 15, 1, 1, 1).expand(1, 15, 4, 4, 4), target
    ).mean()
    old_loss.backward()
    for start, stop in ((0, 5), (5, 10), (10, 15)):
        assert torch.max(torch.abs(old.grad[start:stop] - old.grad[start])).item() <= 1.0e-7

    new = torch.tensor(INITIAL_BIASES, requires_grad=True)
    new_loss = -gaussian_mixture_log_probability(
        new.reshape(1, 15, 1, 1, 1).expand(1, 15, 4, 4, 4), target
    ).mean()
    new_loss.backward()
    for start, stop in ((0, 5), (5, 10), (10, 15)):
        assert torch.max(torch.abs(new.grad[start:stop] - new.grad[start])).item() > 1.0e-8


def test_mixture_likelihood_inverse_is_accurate_and_monotone() -> None:
    generator = torch.Generator().manual_seed(44)
    parameters = torch.randn(1, 15, 1, 1, 64, generator=generator)
    uniform = torch.linspace(1.0e-4, 1.0 - 1.0e-4, 64).reshape(1, 1, 1, 1, 64)
    value = gaussian_mixture_inverse(parameters, uniform)
    reconstructed = gaussian_mixture_cdf(parameters, value)
    assert torch.max(torch.abs(reconstructed - uniform)).item() <= 2.0e-6
    assert torch.isfinite(gaussian_mixture_log_probability(parameters, value)).all()
    for index in range(64):
        fixed = parameters[..., index : index + 1].expand(1, 15, 1, 1, 16)
        ranks = torch.linspace(0.01, 0.99, 16).reshape(1, 1, 1, 1, 16)
        samples = gaussian_mixture_inverse(fixed, ranks)
        assert torch.all(torch.diff(samples.reshape(-1)) > 0)


def test_all_finite_gaussian_parameters_have_finite_physical_moments() -> None:
    parameters = torch.zeros(2, 15, 2, 2, 2)
    parameters[:, 5:10] = torch.linspace(-10.0, 10.0, 5).reshape(1, 5, 1, 1, 1)
    parameters[:, 10:15] = 20.0
    proof = _finite_physical_log_moments(parameters, 0.09877202271987233)
    assert set(proof) == {
        "maximum_standardized_residual_log_moment_order_1",
        "maximum_standardized_residual_log_moment_order_2",
    }
    assert np.isfinite(list(proof.values())).all()


def test_classification_branches() -> None:
    assert classify(True, True, True, True, False)[0] == "identifiable_query_local_gaussian_mixture_copula_sufficient"
    assert (
        classify(False, True, True, False, False)[0]
        == "finite_moment_local_marginal_is_calibrated_but_empirical_rank_copula_limits_morphology"
    )
    assert (
        classify(False, False, True, True, False)[0]
        == "finite_moment_gaussian_mixture_body_is_supported_but_extreme_calibration_is_insufficient"
    )
    assert (
        classify(False, False, False, False, True)[0]
        == "query_local_parameter_alignment_is_not_causal"
    )
    assert (
        classify(False, False, False, False, False)[0]
        == "identifiable_query_local_gaussian_mixture_copula_is_not_a_common_domain_repair"
    )
