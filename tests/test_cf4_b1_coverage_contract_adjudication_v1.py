import cf4_b1_coverage_contract_adjudication_v1 as adjudication


def test_legacy_failure_counts_have_explicit_64_member_denominator():
    result = adjudication.summarize()
    assert result["member_count"] == 64
    assert result["failure_counts"] == {"coverage68": 47, "coverage95": 47, "union": 56}
    assert result["failure_fractions"] == {"coverage68": 47 / 64, "coverage95": 47 / 64, "union": 56 / 64}
    assert result["contingency"] == {"coverage68_only": 9, "coverage95_only": 9, "both": 38, "neither": 8}


def test_checker_does_not_change_science_boundary():
    result = adjudication.summarize()
    assert result["validation_opened"] is False
    assert result["B2_IC_FORWARD"] == "NOT_STARTED"
