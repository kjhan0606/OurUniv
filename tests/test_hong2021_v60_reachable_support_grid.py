import json
from pathlib import Path

import numpy as np

from hong2021_v18_init import sha256_file
from hong2021_v50_network import LOWER_SUPPORT, UPPER_SUPPORT
from hong2021_v60_reachable_support_grid import PROGRAM_SHA256, extended_grid


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v60_reachable_support_grid_program.json"


def test_program_is_frozen_hash_bound_and_keeps_firewall_closed():
    row = json.loads(PROGRAM.read_text())
    parent = row["parent_evidence"]
    record = json.loads((ROOT / parent["v59_record"]).read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(ROOT / parent["v59_record"]) == parent["v59_record_sha256"]
    assert row["status"] == "frozen_before_grid_implementation_or_materialization"
    assert record["audit"]["classification"] == parent["required_classification"]
    assert row["reachable_support_definition"][
        "unchanged_open_standardized_residual_support"
    ] == [LOWER_SUPPORT, UPPER_SUPPORT]
    assert row["grid_extension"]["no_threshold_removal_or_movement"] is True
    assert row["firewall"]["training_or_refit"] == "forbidden"
    assert row["firewall"]["development_access"] == "forbidden"
    assert row["firewall"]["independent_gate_locked"] is True


def test_grid_extension_preserves_existing_thresholds_and_hits_upper_exactly():
    existing = np.asarray([1.0, 2.0])
    thresholds, weights = extended_grid(0.0, existing, 1.0, 4.5)
    assert np.array_equal(thresholds[:2], existing)
    assert thresholds.tolist() == [1.0, 2.0, 3.0, 4.0, 4.5]
    assert thresholds[-1] == 4.5
    assert np.all(weights > 0.0)
    assert abs(float(weights.sum()) - 1.0) < 1e-12


def test_grid_extension_rejects_nonextension_and_nonmonotone_inputs():
    with np.testing.assert_raises(ValueError):
        extended_grid(0.0, np.asarray([1.0, 2.0]), 1.0, 2.0)
    with np.testing.assert_raises(ValueError):
        extended_grid(0.0, np.asarray([1.0, 1.0]), 1.0, 3.0)
