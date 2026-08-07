from __future__ import annotations

from hong2021_v25_development_gate import _failure_classification


def _domain(maximum: float, checks: dict[str, bool]) -> dict:
    return {
        "field_gate": {"checks": checks, "pass": all(checks.values())},
        "mechanism_Q3_Q4": {
            "generated_max_above_truth_max_dex": maximum,
        },
    }


def _baseline() -> dict:
    checks = {"high_k": True, "peaks": True}
    return {
        "candidates": [{}, {}, {"domains": {
            "tng": _domain(0.7, checks),
            "simba_dev": _domain(0.6, checks),
            "swift_dev": _domain(0.2, checks),
        }}]
    }


def test_v25_failure_classification_for_calibration_morphology_tradeoff() -> None:
    final = {"domains": {
        "tng": _domain(0.4, {"high_k": False, "peaks": True}),
        "simba_dev": _domain(0.3, {"high_k": True, "peaks": True}),
        "swift_dev": _domain(0.2, {"high_k": True, "peaks": True}),
    }}
    result = _failure_classification(final, _baseline())
    assert result["class"] == "tail_reweighting_created_a_calibration_morphology_tradeoff"
    assert result["newly_lost_field_checks_vs_v24"] == {"tng": ["high_k"]}


def test_v25_failure_classification_when_q3_does_not_improve_both_domains() -> None:
    final = {"domains": {
        "tng": _domain(0.8, {"high_k": True, "peaks": True}),
        "simba_dev": _domain(0.3, {"high_k": True, "peaks": True}),
        "swift_dev": _domain(0.2, {"high_k": True, "peaks": True}),
    }}
    result = _failure_classification(final, _baseline())
    assert result["class"] == "target_tail_reweighting_not_the_primary_remaining_mechanism"
