import cf4_b1_arm_isolation_diagnosis_v1 as isolation


def test_fresh_seed_block_is_disjoint_from_all_sealed_ranges():
    isolation._assert_disjoint_seed_firewall()
    assert isolation.common_seed_schedule(0)["truth"] == 2026084000
    assert isolation.common_seed_schedule(15)["truth"] == 2026084015


def test_controls_are_explicit_and_do_not_change_the_mean_model_contract():
    assert isolation.D_CONTROLS == ("native", "no_phi", "no_discrepancy", "no_phi_no_discrepancy")
    assert isolation.REPLICATES == 16
