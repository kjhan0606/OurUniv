from hong2021_v63_development_gate import classify


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

