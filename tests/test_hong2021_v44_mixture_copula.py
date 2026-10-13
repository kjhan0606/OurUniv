import hashlib
from pathlib import Path

import numpy as np
import torch

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v44_development_gate import classify
from hong2021_v44_network import (
    LocalMixtureUNet,
    logistic_mixture_cdf,
    logistic_mixture_inverse,
    logistic_mixture_log_probability,
    parameter_count,
)
from hong2021_v44_sample import ENSEMBLE_SCHEMA
from hong2021_v44_train import PARAMETERS, PROGRAM_SHA256


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_firewall_and_evaluator_schema() -> None:
    path = REPO / "config/hong2021_v44_local_mixture_copula_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"spatial_rank_transport": false' in text
    assert '"global_residual_scaling": false' in text
    assert '"posthoc_Ak": false' in text
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS


def test_network_shape_parameter_count_and_zero_output() -> None:
    model = LocalMixtureUNet()
    assert parameter_count(model) == PARAMETERS
    with torch.no_grad():
        output = model(torch.zeros(1, 7, 16, 16, 16))
    assert output.shape == (1, 15, 16, 16, 16)
    assert torch.count_nonzero(output).item() == 0


def test_mixture_likelihood_inverse_is_accurate_and_monotone() -> None:
    generator = torch.Generator().manual_seed(44)
    parameters = torch.randn(1, 15, 1, 1, 64, generator=generator)
    uniform = torch.linspace(1.0e-4, 1.0 - 1.0e-4, 64).reshape(1, 1, 1, 1, 64)
    value = logistic_mixture_inverse(parameters, uniform)
    reconstructed = logistic_mixture_cdf(parameters, value)
    assert torch.max(torch.abs(reconstructed - uniform)).item() <= 2.0e-6
    assert torch.isfinite(logistic_mixture_log_probability(parameters, value)).all()
    for index in range(64):
        fixed = parameters[..., index : index + 1].expand(1, 15, 1, 1, 16)
        ranks = torch.linspace(0.01, 0.99, 16).reshape(1, 1, 1, 1, 16)
        samples = logistic_mixture_inverse(fixed, ranks)
        assert torch.all(torch.diff(samples.reshape(-1)) > 0)


def test_classification_branches() -> None:
    assert classify(True, True, True, True, False)[0] == "query_local_mixture_copula_sufficient"
    assert (
        classify(False, True, True, False, False)[0]
        == "local_marginal_likelihood_is_calibrated_but_empirical_rank_copula_limits_morphology"
    )
    assert (
        classify(False, False, True, True, False)[0]
        == "local_mixture_body_is_supported_but_extreme_likelihood_is_insufficient"
    )
    assert (
        classify(False, False, False, False, True)[0]
        == "query_local_parameter_alignment_is_not_causal"
    )
    assert (
        classify(False, False, False, False, False)[0]
        == "query_local_mixture_copula_is_not_a_common_domain_repair"
    )
