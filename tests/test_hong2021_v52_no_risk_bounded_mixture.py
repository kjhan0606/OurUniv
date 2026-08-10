import ast
import hashlib
import math
from pathlib import Path

import numpy as np

import hong2021_v52_train as train_module
from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v52_development_gate import _extreme_comparison, classify
from hong2021_v52_sample import ENSEMBLE_SCHEMA
from hong2021_v52_train import PROGRAM_SHA256, _learning_rate, no_risk_condition_cube


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_firewall_and_evaluator_schema() -> None:
    path = REPO / "config/hong2021_v52_no_risk_bounded_mixture_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"risk_channel_nonzero_anywhere_in_V52": false' in text
    assert '"support_changed": false' in text
    assert '"posthoc_Ak": false' in text
    assert '"independent_gate_locked": true' in text
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS


def test_sources_have_no_json_boolean_names() -> None:
    for name in (
        "src/hong2021_v52_train.py",
        "src/hong2021_v52_sample.py",
        "src/hong2021_v52_development_gate.py",
    ):
        names = {
            node.id
            for node in ast.walk(ast.parse((REPO / name).read_text()))
            if isinstance(node, ast.Name)
        }
        assert "false" not in names
        assert "true" not in names


def test_no_risk_wrapper_changes_only_channel_five(monkeypatch) -> None:
    original = np.arange(7 * 8, dtype=np.float32).reshape(7, 2, 2, 2)
    target = np.ones((1, 2, 2, 2), dtype=np.float32)
    backbone = np.zeros_like(target)

    def fake_condition_cube(*args, risk_ablation=False, **kwargs):
        condition = original.copy()
        if risk_ablation:
            condition[5] = 0.0
        return condition, target.copy(), backbone.copy()

    monkeypatch.setattr(train_module, "condition_cube", fake_condition_cube)
    condition, actual_target, actual_backbone = no_risk_condition_cube(
        None, None, None, "TNG100", "train", 0
    )
    assert np.count_nonzero(condition[5]) == 0
    assert np.array_equal(condition[[0, 1, 2, 3, 4, 6]], original[[0, 1, 2, 3, 4, 6]])
    assert np.array_equal(actual_target, target)
    assert np.array_equal(actual_backbone, backbone)


def test_schedule_is_the_frozen_v50_schedule() -> None:
    assert math.isclose(_learning_rate(12_000), 2.0e-5)
    assert 2.0e-5 < _learning_rate(6_000) < 2.0e-4
    assert _learning_rate(1) < 2.0e-4


def test_extreme_comparison_uses_absolute_q_and_log_moment_distance() -> None:
    candidate = {
        "delta_q99_999_dex": -0.1,
        "generated_max_above_truth_max_dex": 0.2,
        "generated_over_truth_mean_delta_squared": 0.8,
    }
    reference = {
        "delta_q99_999_dex": 0.2,
        "generated_max_above_truth_max_dex": 0.3,
        "generated_over_truth_mean_delta_squared": 2.0,
    }
    row = _extreme_comparison(candidate, reference)
    assert row["candidate_strictly_improves_all_three"]
    assert not row["reference_equals_or_improves_all_three"]


def test_fixed_classification_precedence() -> None:
    assert classify(True, True, True, True, True, True, True, True)[0] == (
        "matched_no_risk_bounded_mixture_is_development_sufficient"
    )
    assert classify(False, True, True, True, True, False, True, True)[0] == (
        "no_risk_marginal_is_calibrated_but_empirical_rank_copula_limits_morphology"
    )
    assert classify(False, False, False, True, True, False, True, True)[0] == (
        "structure_risk_is_a_causal_amplifier_but_removal_is_not_sufficient"
    )
    assert classify(False, False, False, False, True, False, False, True)[0] == (
        "structure_risk_removal_reduces_extremes_but_damages_the_stochastic_field_body"
    )
    assert classify(False, False, False, False, False, True, True, True)[0] == (
        "inference_only_risk_ablation_did_not_survive_matched_retraining"
    )
    assert classify(False, False, False, False, False, True, False, True)[0] == (
        "no_risk_query_local_parameter_alignment_is_not_causal"
    )
    assert classify(False, False, False, False, False, True, False, False)[0] == (
        "matched_no_risk_result_is_mixed_or_not_a_common_domain_repair"
    )
