import ast
import hashlib
from pathlib import Path

import numpy as np

from hong2021_v46_tail_occupancy_audit import (
    PROBE_VOXELS,
    PROGRAM_SHA256,
    _effective,
    _probe_indices,
    classify,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall() -> None:
    path = REPO / "config/hong2021_v46_mixture_tail_occupancy_audit_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"training_or_refit": false' in text
    assert '"new_development_sampling": false' in text
    assert '"validation_access": "forbidden"' in text
    assert '"independent_gate_locked": true' in text


def test_audit_source_has_no_json_boolean_names() -> None:
    path = REPO / "src/hong2021_v46_tail_occupancy_audit.py"
    names = {
        node.id
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Name)
    }
    assert "false" not in names
    assert "true" not in names


def test_probe_indices_are_fixed_unique_and_object_specific() -> None:
    first = _probe_indices(0, 0)
    assert np.array_equal(first, _probe_indices(0, 0))
    assert len(first) == PROBE_VOXELS
    assert len(np.unique(first)) == PROBE_VOXELS
    assert first.min() >= 0 and first.max() < 64**3
    assert not np.array_equal(first, _probe_indices(0, 1))
    assert not np.array_equal(first, _probe_indices(1, 0))


def test_effective_component_count() -> None:
    assert abs(_effective(np.ones(5)) - 5.0) < 1.0e-12
    assert 1.0 <= _effective(np.array([1.0, 0.0, 0.0, 0.0, 0.0])) <= 1.0 + 1.0e-12


def test_fixed_classification_precedence() -> None:
    assert classify(True, True, True, True)[0] == (
        "unsupported_low_responsibility_component_mass_drives_the_train_tail"
    )
    assert classify(False, True, True, True)[0] == (
        "mixture_likelihood_is_globally_overdispersed_in_the_train_upper_tail"
    )
    assert classify(False, False, True, True)[0] == (
        "identifiable_initialization_did_not_prevent_component_collapse"
    )
    assert classify(False, False, False, True)[0] == (
        "train_mixture_tail_is_calibrated_but_empirical_rank_copula_breaks_development_extremes"
    )
    assert classify(False, False, False, False)[0] == (
        "mixture_tail_failure_is_mixed_or_not_identified"
    )
