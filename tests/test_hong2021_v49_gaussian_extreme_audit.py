import ast
import hashlib
import math
from pathlib import Path

import torch

from hong2021_v49_gaussian_extreme_audit import (
    PROGRAM_SHA256,
    _gaussian_component_moments,
    classify,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall() -> None:
    path = REPO / "config/hong2021_v49_gaussian_extreme_calibration_audit_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"training_or_refit": false' in text
    assert '"posthoc_scale_or_clipping": false' in text
    assert '"development_array_access": "forbidden"' in text
    assert '"independent_gate_locked": true' in text


def test_audit_source_has_no_json_boolean_names() -> None:
    path = REPO / "src/hong2021_v49_gaussian_extreme_audit.py"
    names = {
        node.id
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Name)
    }
    assert "false" not in names
    assert "true" not in names


def test_gaussian_component_physical_moments_match_closed_form() -> None:
    weights = torch.tensor([[[[[0.25]]], [[[0.75]]]]], dtype=torch.float64)
    locations = torch.tensor([[[[[-0.2]]], [[[0.4]]]]], dtype=torch.float64)
    scales = torch.tensor([[[[[0.3]]], [[[0.7]]]]], dtype=torch.float64)
    base = torch.tensor([[[[[0.1]]]]], dtype=torch.float64)
    target_std = 0.08
    first, second, delta_squared = _gaussian_component_moments(
        weights, locations, scales, base, target_std
    )
    a = 4.5 * math.log(10.0)
    for component in range(2):
        weight = float(weights[0, component, 0, 0, 0])
        location = float(locations[0, component, 0, 0, 0])
        scale = float(scales[0, component, 0, 0, 0])
        mean = 0.1 + target_std * location
        sigma = target_std * scale
        expected_first = weight * math.exp(a * mean + 0.5 * a * a * sigma * sigma)
        expected_second = weight * math.exp(
            2.0 * a * mean + 2.0 * a * a * sigma * sigma
        )
        expected_delta_squared = expected_second - 2.0 * expected_first + weight
        assert abs(float(first[0, component, 0, 0, 0]) - expected_first) < 1.0e-12
        assert abs(float(second[0, component, 0, 0, 0]) - expected_second) < 1.0e-12
        assert (
            abs(
                float(delta_squared[0, component, 0, 0, 0])
                - expected_delta_squared
            )
            < 1.0e-12
        )


def test_fixed_classification_precedence() -> None:
    assert classify(True, True, True, True, True)[0] == (
        "unsupported_Gaussian_component_mass_dominates_the_physical_train_tail"
    )
    assert classify(False, True, True, True, True)[0] == (
        "Gaussian_mixture_is_globally_overdispersed_in_the_train_extreme_tail"
    )
    assert classify(False, False, True, True, True)[0] == (
        "V48_Gaussian_mixture_has_effective_component_collapse"
    )
    assert classify(False, False, False, True, True)[0] == (
        "voxel_log_score_calibrates_extreme_quantiles_but_not_the_physical_second_moment"
    )
    assert classify(False, False, False, False, True)[0] == (
        "train_Gaussian_tail_is_calibrated_but_the_empirical_rank_copula_breaks_development_extremes"
    )
    assert classify(False, False, False, False, False)[0] == (
        "Gaussian_train_extreme_failure_is_mixed_or_not_identified"
    )
