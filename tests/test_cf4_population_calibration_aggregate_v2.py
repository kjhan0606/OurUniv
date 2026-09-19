import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_population_calibration_aggregate_v2 as aggregate  # noqa: E402


def test_frozen_domain_support_encodes_three_density_only_nyquist_modes():
    np.testing.assert_array_equal(aggregate.EXPECTED_DELTA_BIN_IDS, np.arange(12))
    np.testing.assert_array_equal(aggregate.EXPECTED_THETA_BIN_IDS, np.arange(11))
    assert set(aggregate.EXPECTED_DELTA_BIN_IDS) - set(
        aggregate.EXPECTED_THETA_BIN_IDS
    ) == {11}


def test_absent_theta_bin_is_expanded_fail_closed():
    domain_bins = np.arange(11)
    domain_gate = np.ones(11, dtype=bool)
    union = np.arange(12)
    expanded, available = aggregate.expand_gate_to_union(domain_bins, domain_gate, union)
    np.testing.assert_array_equal(available, [True] * 11 + [False])
    np.testing.assert_array_equal(expanded, [True] * 11 + [False])


def test_domain_gate_mapping_rejects_missing_or_nonboolean_metadata():
    with pytest.raises(aggregate.base.CalibrationError):
        aggregate.expand_gate_to_union(
            np.array([0, 2]), np.array([True, True]), np.array([0, 1])
        )
    with pytest.raises(aggregate.base.CalibrationError):
        aggregate.expand_gate_to_union(
            np.array([0, 1]), np.array([1, 1]), np.array([0, 1])
        )


def test_corrected_artifact_schema_keeps_science_claim_separate():
    assert aggregate.RESULT_SCHEMA.endswith("result-v2")
    assert "DEVELOPMENT" in aggregate.RESULT_STATUS
    assert "NO_SCIENCE_CLAIM" in aggregate.RESULT_STATUS
    names = aggregate._expected_array_names()
    assert {"delta_bin_ids", "theta_bin_ids"} <= names
    assert {"delta_available_on_union", "theta_available_on_union"} <= names
    assert {"delta_strict_gate_on_union", "theta_strict_gate_on_union"} <= names
