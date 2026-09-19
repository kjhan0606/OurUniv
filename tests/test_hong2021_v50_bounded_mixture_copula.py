import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v50_development_gate import classify
from hong2021_v50_network import (
    INITIAL_BIASES,
    LOWER_SUPPORT,
    QUADRATURE_INITIAL_STANDARDIZED_MEAN,
    QUADRATURE_INITIAL_STANDARDIZED_VARIANCE,
    UPPER_SUPPORT,
    LocalMixtureUNet,
    bounded_mixture_cdf,
    bounded_mixture_inverse,
    bounded_mixture_log_probability,
    bounded_to_latent,
    initial_standardized_quadrature,
    latent_to_bounded,
    parameter_count,
)
from hong2021_v50_sample import ENSEMBLE_SCHEMA
from hong2021_v50_train import PARAMETERS, PROGRAM_SHA256


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_firewall_and_evaluator_schema() -> None:
    path = REPO / "config/hong2021_v50_bounded_logit_mixture_copula_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"sample_clipping": false' in text
    assert '"component_scale_cap": false' in text
    assert '"posthoc_Ak": false' in text
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS


def test_network_shape_count_and_frozen_biases() -> None:
    model = LocalMixtureUNet()
    assert parameter_count(model) == PARAMETERS
    with torch.no_grad():
        output = model(torch.zeros(1, 7, 16, 16, 16))
    expected = torch.tensor(INITIAL_BIASES).reshape(1, 15, 1, 1, 1)
    assert output.shape == (1, 15, 16, 16, 16)
    assert torch.count_nonzero(model.output.weight).item() == 0
    assert torch.max(torch.abs(output - expected)).item() <= 1.0e-7


def test_support_transform_roundtrip_and_hard_boundary() -> None:
    value = torch.linspace(LOWER_SUPPORT + 0.01, UPPER_SUPPORT - 0.01, 1000)
    reconstructed = latent_to_bounded(bounded_to_latent(value))
    assert torch.max(torch.abs(reconstructed - value)).item() < 2.0e-6
    with pytest.raises(ValueError):
        bounded_to_latent(torch.tensor([LOWER_SUPPORT], dtype=torch.float64))
    with pytest.raises(ValueError):
        bounded_to_latent(torch.tensor([UPPER_SUPPORT], dtype=torch.float64))


def test_initial_quadrature_matches_frozen_values() -> None:
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    mean, variance = initial_standardized_quadrature(
        torch.from_numpy(nodes), torch.from_numpy(weights)
    )
    assert abs(mean - QUADRATURE_INITIAL_STANDARDIZED_MEAN) < 1.0e-6
    assert abs(variance - QUADRATURE_INITIAL_STANDARDIZED_VARIANCE) < 1.0e-6


def test_bounded_likelihood_inverse_is_accurate_monotone_and_interior() -> None:
    generator = torch.Generator().manual_seed(50)
    parameters = torch.randn(1, 15, 1, 1, 1, generator=generator).expand(
        1, 15, 1, 1, 64
    )
    uniform = torch.linspace(1.0e-4, 1.0 - 1.0e-4, 64).reshape(1, 1, 1, 1, 64)
    value = bounded_mixture_inverse(parameters, uniform)
    reconstructed = bounded_mixture_cdf(parameters, value)
    assert torch.max(torch.abs(reconstructed - uniform)).item() <= 2.0e-6
    assert torch.all((value > LOWER_SUPPORT) & (value < UPPER_SUPPORT))
    assert torch.isfinite(bounded_mixture_log_probability(parameters, value)).all()
    assert torch.all(torch.diff(value.reshape(-1)) > 0)


def test_classification_branches() -> None:
    assert classify(True, True, True, True, False)[0] == (
        "bounded_query_local_mixture_copula_sufficient"
    )
    assert classify(False, True, True, False, False)[0] == (
        "bounded_local_marginal_is_calibrated_but_empirical_rank_copula_limits_morphology"
    )
    assert classify(False, False, True, True, False)[0] == (
        "bounded_marginal_support_preserves_the_field_body_but_is_not_sufficient_for_extreme_calibration"
    )
    assert classify(False, False, False, False, True)[0] == (
        "query_local_parameter_alignment_is_not_causal"
    )
    assert classify(False, False, False, False, False)[0] == (
        "bounded_query_local_mixture_copula_is_not_a_common_domain_repair"
    )
