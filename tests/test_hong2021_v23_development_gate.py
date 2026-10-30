from __future__ import annotations

from hong2021_v20_development_gate import Q5_CHECKS
from hong2021_v23_development_gate import (
    Q6_THRESHOLD_KEYS,
    _failure_classification,
    _mechanism_pass,
)


def _domain(*, q3=0.05, q4=1.0, q6_std=True, latent_mean=0.04):
    return {
        "mechanism_Q3_Q4": {
            "delta_q99_999_dex": q3,
            "generated_max_above_truth_max_dex": 0.1,
            "generated_over_truth_mean_delta_squared": q4,
        },
        "field_gate": {
            "pass": True,
            "checks": {name: True for name in Q5_CHECKS},
        },
        "conditional_Q6_std_no_harm": {"pass": q6_std},
        "conditional_Q6_latent": {
            "maximum_absolute_generated_minus_truth_mean": latent_mean
        },
    }


def test_q6_std_no_harm_is_independently_selection_blocking() -> None:
    assert Q6_THRESHOLD_KEYS == {
        "tng": "tng100_dev",
        "simba_dev": "simba_dev",
        "swift_dev": "swift_dev",
    }
    domains = {name: _domain() for name in ("tng", "simba_dev", "swift_dev")}
    assert _mechanism_pass(domains) == (True, True, True, True)
    domains["swift_dev"]["conditional_Q6_std_no_harm"]["pass"] = False
    assert _mechanism_pass(domains) == (True, True, True, False)


def test_failure_classification_uses_frozen_conditional_mean_boundaries() -> None:
    controlled = {
        "domains": {
            name: _domain(q4=2.0, latent_mean=0.05)
            for name in ("tng", "simba_dev", "swift_dev")
        },
        "Q3_all_domains": True,
        "Q4_all_domains": False,
        "Q6_std_no_harm_all_domains": True,
    }
    assert (
        _failure_classification(controlled)["class"]
        == "conditional_mean_controlled_but_insufficient"
    )
    uncontrolled = {
        **controlled,
        "domains": {
            "tng": _domain(latent_mean=0.10),
            "simba_dev": _domain(latent_mean=0.02),
            "swift_dev": _domain(latent_mean=0.02),
        },
    }
    assert (
        _failure_classification(uncontrolled)["class"]
        == "conditional_mean_penalty_failed_to_control_mean"
    )
    intermediate = {
        **controlled,
        "domains": {
            name: _domain(latent_mean=0.075)
            for name in ("tng", "simba_dev", "swift_dev")
        },
    }
    assert (
        _failure_classification(intermediate)["class"]
        == "intermediate_conditional_mean_response"
    )
