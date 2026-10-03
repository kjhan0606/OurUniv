import hashlib
from pathlib import Path

import numpy as np

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v41_development_gate import classify
from hong2021_v41_two_stage import (
    BLOCKS,
    ENSEMBLE_SCHEMA,
    PROGRAM_SHA256,
    amplitude_scale,
    blocks_to_cube,
    cube_to_blocks,
    log10_mean_delta_squared,
    seed_permutation,
    transport_blocks,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_firewall_and_evaluator_schema() -> None:
    path = REPO / "config/hong2021_v41_two_stage_structure_amplitude_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"donor_reselection": false' in text
    assert '"density_field_clipping": false' in text
    assert '"posthoc_Ak": false' in text
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS


def test_factor4_block_roundtrip_and_sparse_bijection() -> None:
    cube = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    np.testing.assert_array_equal(blocks_to_cube(cube_to_blocks(cube)), cube)
    rng = np.random.default_rng(41)
    risk = rng.normal(size=(16, 16, 16))
    carrier = rng.normal(size=BLOCKS)
    mapping, diagnostics = seed_permutation(risk, carrier, 11)
    assert np.array_equal(np.sort(mapping), np.arange(BLOCKS))
    assert diagnostics["nonidentity_blocks"] <= 22
    transported = transport_blocks(cube, mapping)
    np.testing.assert_array_equal(np.sort(transported.reshape(-1)), cube.reshape(-1))


def test_log_amplitude_interpolation_is_bounded_and_improves_knots() -> None:
    coordinate = np.linspace(-1.0, 1.0, 8**3).reshape(1, 8, 8, 8)
    backbone = 0.02 * coordinate
    residual = 0.04 * np.sin(3.0 * coordinate)
    target = log10_mean_delta_squared(backbone + 0.7 * residual)
    scale, achieved, levels = amplitude_scale(backbone, residual, target)
    assert 0.0 <= scale <= 2.0
    assert abs(achieved - target) <= min(abs(np.asarray(levels) - target))


def test_classification_branches() -> None:
    assert classify(True, True, True)[0] == "two_stage_structure_amplitude_model_sufficient"
    assert classify(False, True, True)[0] == "supervised_tails_repaired_but_sparse_block_transport_limits_morphology"
    assert classify(False, False, True)[0] == "object_amplitude_calibration_supported_but_structure_seeding_is_insufficient"
    assert classify(False, True, False)[0] == "structure_seeding_supported_but_object_amplitude_calibration_is_insufficient"
    assert classify(False, False, False)[0] == "two_stage_supervised_intervention_is_not_a_common_domain_repair"
