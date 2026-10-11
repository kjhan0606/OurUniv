import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_same_truth_information_budget as information  # noqa: E402


PROGRAM = ROOT / "config/cf4_same_truth_information_budget_program_v1.json"
SOURCE = ROOT / "src/cf4_same_truth_information_budget.py"


def test_program_freezes_covariance_only_paired_scenarios_and_firewalls():
    program = json.loads(PROGRAM.read_text())
    assert tuple(program["design"]["scenario_order"]) == information.SCENARIOS
    assert program["design"]["known_nuisance_noise_standard_deviation_scales"] == {
        "known_s1": 1.0,
        "known_s0p3": 0.3,
        "known_s0p1": 0.1,
    }
    assert program["design"]["new_truth_seed_count"] == 0
    assert program["design"]["new_random_seed_count"] == 0
    assert program["design"]["covariance_only"] is True
    authorization = program["authorization"]
    for key in (
        "truth_array_generation_or_deserialization",
        "likelihood_datum_consumed_by_inference",
        "untouched_256_mock_validation",
        "resolution_increase",
        "ML_training",
        "frontier_promotion",
        "IC_PM_HOP_RAMSES",
    ):
        assert authorization[key] is False


def test_program_binds_every_repository_and_source_input():
    program = json.loads(PROGRAM.read_text())
    for collection in ("repository_bindings", "source_bindings"):
        for record in program[collection].values():
            assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record[
                "sha256"
            ]


def test_source_never_deserializes_truth_or_consumes_likelihood_datum():
    source = SOURCE.read_text()
    for forbidden in (
        'fields["truth_white"]',
        'fields["truth_delta"]',
        'fields["truth_theta"]',
        'fields["truth_velocity"]',
        'design["vobs"]',
        'seeds["truth"]',
        "2026083064",
        "2026083320",
    ):
        assert forbidden not in source
    assert 'd_fields["mock_true_position"]' in source
    assert 'base.seed_schedule(mock_index)' in source
    assert 'rng.standard_normal(4)' in source


def test_information_spectrum_recovers_known_covariance_fraction():
    rng = np.random.default_rng(12)
    # Posterior variance is 0.25 of the unit prior in every real component.
    draws = 0.5 * (
        rng.standard_normal((64, 16, 4))
        + 1j * rng.standard_normal((64, 16, 4))
    ) / np.sqrt(2.0)
    draws[:, :, 0] = 0.5 * rng.standard_normal((64, 16))
    bootstrap = information.base._bootstrap_indices(
        64, 100, information.base.BOOTSTRAP_SEED
    )
    metrics, arrays = information.posterior_information_spectrum(
        draws=draws,
        prior_variance=np.ones(4),
        assignment=np.array([0, 0, 1, 1]),
        self_conjugate=np.array([True, False, False, False]),
        bin_ids=np.array([0, 1]),
        bootstrap_indices=bootstrap,
        gates={
            "response_min_inclusive": 0.7,
            "correlation_r_min_inclusive": 0.7,
            "residual_power_ratio_max_inclusive": 0.5,
            "robust_information_lower_min_inclusive": 0.7,
        },
    )
    np.testing.assert_allclose(
        arrays["recovered_information_fraction"],
        1.0 - arrays["posterior_prior_trace_fraction"],
    )
    np.testing.assert_allclose(
        arrays["expected_correlation_r"] ** 2,
        arrays["recovered_information_fraction"],
    )
    assert np.all(np.abs(arrays["posterior_prior_trace_fraction"] - 0.25) < 0.04)
    assert metrics["real_degree_of_freedom_count"] == [3, 4]


def test_preregistered_decision_tree_prefers_finite_ceiling_failure():
    assert information.classify_information_budget(
        {
            "marginalized_s1": False,
            "known_s1": False,
            "known_s0p3": False,
            "known_s0p1": False,
        }
    ) == "FINITE_LOW_NOISE_CEILING_INSUFFICIENT_ADD_INDEPENDENT_Z0_DENSITY_TRACERS"
    assert information.classify_information_budget(
        {
            "marginalized_s1": False,
            "known_s1": False,
            "known_s0p3": True,
            "known_s0p1": True,
        }
    ) == "MEASUREMENT_ERROR_DOMINANT_IMPROVE_VELOCITY_LIKELIHOOD"
    assert information.classify_information_budget(
        {
            "marginalized_s1": False,
            "known_s1": True,
            "known_s0p3": True,
            "known_s0p1": True,
        }
    ) == "NUISANCE_MARGINALIZATION_DOMINANT"


def test_memory_requests_have_twenty_percent_expected_headroom():
    execution = json.loads(PROGRAM.read_text())["execution"]
    assert execution["member_requested_memory_MiB"] >= 1.2 * execution[
        "member_expected_peak_memory_MiB"
    ]
    assert execution["aggregate_requested_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_memory_MiB"
    ]
