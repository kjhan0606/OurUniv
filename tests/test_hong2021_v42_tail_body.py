import hashlib
from pathlib import Path

import numpy as np

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v42_development_gate import classify
from hong2021_v42_tail_body import (
    BLOCK,
    BLOCKS,
    ENSEMBLE_SCHEMA,
    LAMBDA_KNOTS,
    PROGRAM_SHA256,
    local_permutation,
    nested_transport,
    tail_calibrate,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_firewall_and_evaluator_schema() -> None:
    path = REPO / "config/hong2021_v42_within_block_tail_body_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"global_residual_scaling": false' in text
    assert '"hard_density_or_residual_clipping": false' in text
    assert '"posthoc_Ak": false' in text
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS


def test_local_and_nested_transport_are_exact_permutations() -> None:
    rng = np.random.default_rng(42)
    mapping = local_permutation(rng.normal(size=(BLOCK,) * 3), rng.normal(size=(BLOCK,) * 3), 2)
    np.testing.assert_array_equal(np.sort(mapping), np.arange(BLOCK**3))

    rank = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    block_risk = rng.normal(size=(16, 16, 16))
    native_risk = rng.normal(size=(BLOCKS, BLOCK, BLOCK, BLOCK))
    transported, block_map, local_maps, diagnostics = nested_transport(
        rank, block_risk, native_risk, 11, 2, native_mode="full"
    )
    np.testing.assert_array_equal(np.sort(transported.reshape(-1)), rank.reshape(-1))
    np.testing.assert_array_equal(np.sort(block_map), np.arange(BLOCKS))
    np.testing.assert_array_equal(
        np.sort(local_maps, axis=-1),
        np.broadcast_to(np.arange(BLOCK**3), local_maps.shape),
    )
    assert diagnostics["nonidentity_blocks"] <= 22
    assert diagnostics["native_modified_blocks"] == 11


def test_tail_calibration_preserves_non_tail_body_after_dc() -> None:
    coordinate = np.linspace(-1.0, 1.0, 8**3).reshape(1, 8, 8, 8)
    backbone = 0.01 * coordinate
    residual = 0.04 * np.sin(4.0 * coordinate)
    threshold = np.full_like(residual, 0.02)
    calibrated, diagnostics = tail_calibrate(
        backbone, residual, threshold, target=-0.5, enabled=True
    )
    restored = calibrated + diagnostics["tail_DC_projection"]
    np.testing.assert_allclose(
        restored[residual <= threshold], residual[residual <= threshold], atol=1e-14
    )
    assert LAMBDA_KNOTS[0] <= diagnostics["tail_lambda"] <= LAMBDA_KNOTS[-1]
    assert diagnostics["maximum_non_tail_error_after_undoing_DC"] <= 1e-14
    assert diagnostics["maximum_absolute_residual_dc"] <= 1e-14

    disabled, disabled_diagnostics = tail_calibrate(
        backbone, residual, threshold, target=-0.5, enabled=False
    )
    assert disabled_diagnostics["tail_lambda"] == 1.0
    np.testing.assert_allclose(disabled, residual - residual.mean(), atol=1e-14)


def test_classification_branches() -> None:
    assert classify(True, True, True, True)[0] == "within_block_tail_body_model_sufficient"
    assert (
        classify(False, True, True, True)[0]
        == "tail_and_body_repaired_remaining_failure_is_field_morphology_or_calibration"
    )
    assert (
        classify(False, True, True, False)[0]
        == "tail_only_calibration_still_damages_stochastic_body"
    )
    assert classify(False, False, True, True)[0] == "native_extreme_location_is_still_insufficient"
    assert (
        classify(False, True, False, True)[0]
        == "body_preserving_tail_intervention_is_not_a_common_domain_repair"
    )
