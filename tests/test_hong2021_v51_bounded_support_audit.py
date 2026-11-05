import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from hong2021_v51_bounded_support_audit import (
    PROGRAM_SHA256,
    _bounded_mixture_cdf64,
    _quadrature_object,
    _strata_summary,
    _support_threshold,
    classify,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall() -> None:
    path = REPO / "config/hong2021_v51_bounded_support_calibration_audit_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"training_or_refit": false' in text
    assert '"support_change": false' in text
    assert '"development_array_access": "forbidden"' in text
    assert '"historical_EAGLE_access": "forbidden"' in text
    assert '"independent_gate_locked": true' in text


def test_audit_source_has_no_json_boolean_names() -> None:
    path = REPO / "src/hong2021_v51_bounded_support_audit.py"
    names = {
        node.id
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Name)
    }
    assert "false" not in names
    assert "true" not in names


def test_fixed_classification_precedence() -> None:
    assert classify(True, True, True, True, True, True)[0] == (
        "train_unoccupied_fixed_support_margin_dominates_the_bounded_physical_tail"
    )
    assert classify(False, True, True, True, True, True)[0] == (
        "structure_risk_conditioning_amplifies_the_bounded_extreme_tail"
    )
    assert classify(False, False, True, True, True, True)[0] == (
        "bounded_logit_mixture_is_overdispersed_in_the_train_upper_tail"
    )
    assert classify(False, False, False, True, True, True)[0] == (
        "V50_bounded_latent_mixture_has_effective_component_failure"
    )
    assert classify(False, False, False, False, True, True)[0] == (
        "bounded_voxel_log_score_calibrates_train_ranks_but_not_the_physical_second_moment"
    )
    assert classify(False, False, False, False, False, True)[0] == (
        "train_bounded_marginal_is_calibrated_but_empirical_rank_copula_or_query_shift_breaks_development_extremes"
    )
    assert classify(False, False, False, False, False, False)[0] == (
        "bounded_support_extreme_failure_is_mixed_or_not_identified"
    )


def test_support_thresholds_are_symmetric_and_interior() -> None:
    lower = _support_threshold(0.01, False)
    upper = _support_threshold(0.01, True)
    outer_lower = _support_threshold(0.001, False)
    outer_upper = _support_threshold(0.001, True)
    assert outer_lower < lower < upper < outer_upper
    assert math.isclose(
        lower + upper,
        _support_threshold(0.05, False) + _support_threshold(0.05, True),
    )


def test_float64_bounded_cdf_is_symmetric_for_one_standard_normal() -> None:
    parameters = torch.zeros((1, 15, 1, 1, 3), dtype=torch.float64)
    parameters[:, 1:5] = -100.0
    parameters[:, 5:10] = 0.0
    parameters[:, 10:15] = math.log(math.expm1(0.99))
    midpoint = 0.5 * (
        _support_threshold(0.01, False) + _support_threshold(0.01, True)
    )
    values = torch.full((1, 1, 1, 1, 3), midpoint, dtype=torch.float64)
    cdf = _bounded_mixture_cdf64(parameters, values)
    assert torch.max(torch.abs(cdf - 0.5)).item() < 1.0e-12


def test_bounded_quadrature_is_finite_and_converged_for_narrow_component() -> None:
    weights = torch.ones((1, 4), dtype=torch.float64)
    locations = torch.zeros((1, 4), dtype=torch.float64)
    scales = torch.full((1, 4), 0.05, dtype=torch.float64)
    base = torch.zeros(4, dtype=torch.float64)
    node64, weight64 = np.polynomial.hermite.hermgauss(64)
    node32, weight32 = np.polynomial.hermite.hermgauss(32)
    primary = _quadrature_object(
        weights,
        locations,
        scales,
        base,
        0.01,
        torch.from_numpy(node64),
        torch.from_numpy(weight64),
        9.0,
    )
    control = _quadrature_object(
        weights,
        locations,
        scales,
        base,
        0.01,
        torch.from_numpy(node32),
        torch.from_numpy(weight32),
        9.0,
    )
    for key in ("first", "second", "delta_squared"):
        assert np.isfinite(primary[key]).all()
        assert np.all(primary[key] > 0.0)
        assert np.allclose(primary[key], control[key], rtol=1.0e-10, atol=0.0)


def test_strata_summary_preserves_all_rows() -> None:
    variable = np.arange(10_000, dtype=np.float64)
    truth = 1.0 + variable / 10_000.0
    predicted = 2.0 * truth
    probability = np.full_like(variable, 1.0e-6)
    boundary = np.full_like(variable, 1.0e-7)
    result = _strata_summary(variable, truth, predicted, probability, boundary)
    assert sum(row["count"] for row in result["strata"].values()) == len(variable)
    for row in result["strata"].values():
        assert math.isclose(row["quadrature_over_truth_mean_delta_squared"], 2.0)
