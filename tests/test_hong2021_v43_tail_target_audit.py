import hashlib
from pathlib import Path

import numpy as np

from hong2021_v41_two_stage import log10_mean_delta_squared
from hong2021_v43_tail_target_audit import (
    PROGRAM_SHA256,
    batch_log10_mean_delta_squared,
    classify,
    endpoint_diagnostics,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_diagnostic_firewall() -> None:
    path = REPO / "config/hong2021_v43_tail_threshold_target_audit_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"fit_or_sampling": false' in text
    assert '"new_field_generation": false' in text
    assert '"threshold_or_branch_tuning_after_audit": false' in text


def test_batch_amplitude_matches_scalar_definition() -> None:
    rng = np.random.default_rng(43)
    fields = rng.normal(0.0, 0.03, size=(3, 1, 8, 8, 8))
    expected = np.asarray([log10_mean_delta_squared(field) for field in fields])
    np.testing.assert_allclose(batch_log10_mean_delta_squared(fields), expected)


def test_endpoint_diagnostics_preserve_body_and_find_bracket() -> None:
    coordinate = np.linspace(-1.0, 1.0, 64**3).reshape(1, 64, 64, 64)
    backbone = 0.01 * coordinate
    residual = np.stack(
        (0.03 * np.sin(4.0 * coordinate), 0.04 * np.cos(3.0 * coordinate))
    ).reshape(2, 1, 64, 64, 64)
    threshold = np.full_like(backbone, 0.015)
    first = endpoint_diagnostics(backbone, residual, threshold, np.zeros(2))
    target = 0.5 * (first["zero_amplitude"] + first["unit_amplitude"])
    result = endpoint_diagnostics(backbone, residual, threshold, target)
    assert np.all(result["target_bracketed"])
    assert np.all((result["tail_fraction"] > 0) & (result["tail_fraction"] < 1))
    assert np.all((result["top26_recall"] >= 0) & (result["top26_recall"] <= 1))
    assert np.all((result["top3_recall"] >= 0) & (result["top3_recall"] <= 1))
    restored_body = residual - np.maximum(residual - threshold[None], 0.0)
    centered = restored_body - restored_body.mean(axis=(-3, -2, -1), keepdims=True)
    np.testing.assert_allclose(result["body"], centered)


def test_classification_branches() -> None:
    assert (
        classify(True, True, True)[0]
        == "v42_fixed_lambda_solver_or_implementation_is_inconsistent_with_attainable_support"
    )
    assert (
        classify(True, False, True)[0]
        == "object_target_is_supported_but_q99_9_tail_support_is_too_narrow"
    )
    assert (
        classify(True, False, False)[0]
        == "object_target_is_supported_but_transported_body_and_backbone_are_incompatible"
    )
    assert (
        classify(False, True, True)[0]
        == "v41_object_amplitude_target_is_not_cross_domain_supported"
    )
