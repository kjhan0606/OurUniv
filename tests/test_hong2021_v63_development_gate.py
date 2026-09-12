import json
from pathlib import Path

from hong2021_v63_development_gate import classify


REPO = Path(__file__).resolve().parents[1]


def test_v63_primary_branch_requires_sealing_and_explicit_approval() -> None:
    classification, next_step = classify(True, True, True, False, False, False)
    assert classification == (
        "conditional_log_physical_moment_objective_is_development_sufficient"
    )
    assert next_step == (
        "seal_V63_and_await_explicit_approval_before_independent_EAGLE_gate"
    )


def test_v63_copula_branch_is_fixed_before_evaluation() -> None:
    classification, next_step = classify(False, True, True, False, False, False)
    assert "rank_copula" in classification
    assert "without_changing_the_V63_likelihood_or_objective" in next_step


def test_v63_nontransfer_branch_remains_independent_gate_locked() -> None:
    classification, next_step = classify(False, False, False, False, True, False)
    assert "does_not_transfer" in classification
    assert "stop_before_independent_EAGLE" in next_step


def test_v63_result_selects_no_refit_sampler_alignment_audit() -> None:
    record = json.loads((REPO / "config/hong2021_v63_result_record.json").read_text())
    assert record["status"] == (
        "complete_development_failure_selected_sampler_alignment_audit"
    )
    assert record["development_decision"]["development_pass"] is False
    assert record["development_decision"][
        "V63_strictly_improves_all_three_over_both_V50_and_V52_every_domain"
    ] is True
    assert record["selected_next_step"]["action"] == (
        "freeze and run a no-refit train-only empirical-rank sampler-alignment audit"
    )
    assert record["firewall"]["independent_gate_locked"] is True
