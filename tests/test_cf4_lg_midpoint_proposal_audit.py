import json
from pathlib import Path

import numpy as np

from cf4_lg_midpoint_proposal_audit import (
    diagonal_normal_logpdf,
    importance_ess_fraction,
    mixture_logpdf,
    verify_defensive_component,
)


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/p2_lg_z0_forward_likelihood_v8_proposal_audit.json"


def test_mixture_logpdf_matches_identical_components() -> None:
    normal = {"mean_mpc_h": [0.0, 0.0, 0.0], "sigma_mpc_h": [1.0, 1.0, 1.0]}
    components = [dict(normal, weight=0.25), dict(normal, weight=0.75)]
    value = np.array([0.2, -0.4, 0.7])
    assert np.isclose(mixture_logpdf(value, components), diagonal_normal_logpdf(value, normal))


def test_defensive_component_bounds_target_over_proposal() -> None:
    prior = {"mean_mpc_h": [0.0, 0.0, 0.0], "sigma_mpc_h": [3.0, 3.0, 3.0]}
    components = [
        dict(prior, weight=0.5),
        {"weight": 0.5, "mean_mpc_h": [1.0, 1.0, 1.0], "sigma_mpc_h": [1.0, 1.0, 1.0]},
    ]
    assert verify_defensive_component(prior, components, 0.5) == 2.0


def test_importance_ess_fraction_is_one_for_constant_weights() -> None:
    likelihood = np.ones(20)
    proposal_over_prior = np.ones(20)
    assert np.isclose(importance_ess_fraction(likelihood, proposal_over_prior), 1.0)


def test_audit_contract_opens_no_fresh_data() -> None:
    program = json.loads(PROGRAM.read_text())
    assert program["status"] == "frozen_before_conditional_proposal_audit"
    assert program["firewall"]["fresh_phase_or_catalog_accessed"] is False
    assert program["bank_size_selection"]["candidate_sizes"] == [64, 128, 256]
