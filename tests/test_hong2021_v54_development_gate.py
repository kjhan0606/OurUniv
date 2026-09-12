from hong2021_v54_development_gate import classify


def test_fixed_v54_development_classification() -> None:
    assert classify(True, True, True, True, True, True)[0] == "train_only_proper_tail_score_is_development_sufficient"
    assert classify(False, True, True, True, False, False)[0] == "proper_tail_marginal_is_calibrated_but_empirical_rank_copula_limits_morphology"
    assert classify(False, False, False, True, False, False)[0] == "proper_tail_score_improves_all_extremes_but_is_not_development_sufficient"
    assert classify(False, False, False, False, True, False)[0] == "proper_tail_score_does_not_transfer_from_train_mechanism_to_development"
    assert classify(False, False, False, False, False, True)[0] == "V54_query_local_tail_parameters_are_not_causal"
