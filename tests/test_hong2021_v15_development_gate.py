from __future__ import annotations

import copy

from hong2021_v15_development_gate import canonical_digest, select_candidate_rows


def test_v15_selects_earliest_complete_three_domain_pass() -> None:
    rows = [
        {"step": 5000, "all_three_pass": True},
        {"step": 10000, "all_three_pass": True},
    ]
    assert select_candidate_rows(rows) == rows[0]
    assert select_candidate_rows([{**rows[0], "all_three_pass": False}]) is None


def test_decision_digest_detects_mutation_and_ignores_its_own_field() -> None:
    report = {"schema": "test", "development_pass": False, "candidates": [1, 2]}
    digest = canonical_digest(report)
    signed = {**report, "decision_digest_sha256": digest}
    assert canonical_digest(signed) == digest
    changed = copy.deepcopy(signed)
    changed["development_pass"] = True
    assert canonical_digest(changed) != digest
