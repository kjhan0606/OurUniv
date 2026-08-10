import ast
import hashlib
from pathlib import Path

import numpy as np

from hong2021_v53_domainwise_audit import (
    PROGRAM_SHA256,
    _bootstrap_mean_difference,
    _bootstrap_ratio,
    _stratum_masks,
    classify,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall() -> None:
    path = REPO / "config/hong2021_v53_v50_v52_domainwise_audit_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"new_development_sampling": false' in text
    assert '"historical_EAGLE_access": "forbidden"' in text
    assert '"independent_gate_locked": true' in text


def test_source_has_no_json_boolean_names() -> None:
    names = {
        node.id
        for node in ast.walk(
            ast.parse((REPO / "src/hong2021_v53_domainwise_audit.py").read_text())
        )
        if isinstance(node, ast.Name)
    }
    assert "false" not in names
    assert "true" not in names


def test_paired_bootstrap_helpers() -> None:
    indices = np.tile(np.arange(4, dtype=np.int16), (32, 1))
    first = np.asarray([2.0, 4.0, 6.0, 8.0])
    second = np.asarray([1.0, 2.0, 3.0, 4.0])
    difference = _bootstrap_mean_difference(first, second, indices)
    ratio = _bootstrap_ratio(first, second, indices)
    assert difference["paired_mean_difference"] == 2.5
    assert difference["paired_object_bootstrap_95"] == [2.5, 2.5]
    assert ratio["ratio"] == 2.0
    assert ratio["paired_object_bootstrap_95"] == [2.0, 2.0]
    assert ratio["interval_excludes_one"] is True


def test_strata_are_exhaustive_and_disjoint() -> None:
    value = np.arange(10, dtype=np.float64)
    masks = _stratum_masks(value, np.asarray([3.0, 6.0, 8.0]))
    total = np.sum(np.stack(masks), axis=0)
    assert np.array_equal(total, np.ones_like(value))
    assert [int(mask.sum()) for mask in masks] == [3, 3, 2, 2]


def test_fixed_classification_precedence() -> None:
    assert classify(True, True, True, True, True, True)[0] == (
        "risk_choice_does_not_remove_common_train_high_backbone_physical_tail_miscalibration"
    )
    assert classify(False, True, True, True, True, False)[0] == (
        "structure_risk_has_domain_dependent_high_backbone_utility"
    )
    assert classify(False, False, True, True, False, False)[0] == (
        "calibrated_train_marginals_leave_empirical_rank_copula_two_point_failure"
    )
    assert classify(False, False, False, True, True, True)[0] == (
        "sixteen_object_development_sample_does_not_resolve_domain_dependent_risk_utility"
    )
    assert classify(False, False, False, True, False, False)[0] == (
        "V50_V52_mixed_domain_failure_not_explained_by_risk_backbone_or_two_point_audits"
    )
