import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_population_calibration as calibration  # noqa: E402


def _basis(reference_cz):
    count = 128
    longitude = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    sin_latitude = np.linspace(-0.8, 0.8, count)
    cosine = np.sqrt(1.0 - sin_latitude**2)
    return {
        "dist": np.linspace(8.0, 100.0, count),
        "e_dm": np.linspace(0.08, 0.3, count),
        "nhat": np.column_stack(
            (cosine * np.cos(longitude), cosine * np.sin(longitude), sin_latitude)
        ),
        "reference_cz": np.asarray(reference_cz, dtype=float),
        "H0": np.array(74.6),
        "d_min": np.array(1.0),
        "d_max": np.array(120.0),
    }


def _generator():
    return {
        "proposal_oversampling_factor": 4,
        "radial_histogram_bins": 12,
        "longitude_histogram_bins": 12,
        "sin_latitude_histogram_bins": 8,
        "edm_floor_mag": 0.04343,
        "radial_fraction": 0.95,
        "log_gaussian_density_bias": 1.0,
        "fidelity_longitude_bins": 12,
        "fidelity_sin_latitude_bins": 6,
    }


def test_seed_schedule_is_disjoint_and_keeps_untouched_validation_truth_range():
    all_nontruth = set()
    truth = []
    for index in range(calibration.MOCK_COUNT):
        schedule = calibration.seed_schedule(index)
        truth.append(schedule["truth"])
        current = {
            schedule["population"],
            schedule["distance_noise"],
            schedule["nuisance_truth"],
            schedule["preconditioner"],
            schedule["adjoint"],
            schedule["heldout_bootstrap"],
            *schedule["posterior_draws"],
        }
        assert len(current) == 22
        assert not (current & all_nontruth)
        all_nontruth |= current
    assert truth == list(range(2026083000, 2026083064))
    assert not (all_nontruth & set(range(2026083064, 2026083320)))
    with pytest.raises(calibration.CalibrationError):
        calibration.seed_schedule(64)


def test_population_generation_never_uses_observed_reference_velocity():
    first_basis = _basis(np.linspace(100.0, 20000.0, 128))
    second_basis = _basis(np.linspace(-9.0e8, 9.0e8, 128))
    delta = np.zeros((8, 8, 8))
    velocity = np.zeros((3, 8, 8, 8))
    first = calibration.generate_population_catalog(
        first_basis, delta, velocity, 3, _generator()
    )
    second = calibration.generate_population_catalog(
        second_basis, delta, velocity, 3, _generator()
    )
    for key in first["catalog"]:
        np.testing.assert_array_equal(first["catalog"][key], second["catalog"][key])
    for key in (
        "true_distance",
        "true_position",
        "true_radial_velocity",
        "nuisance_truth",
        "distance_log_error",
        "local_truth_delta",
        "empirical_source_index",
    ):
        np.testing.assert_array_equal(first[key], second[key])


def test_population_generator_is_count_conditioned_density_coupled_and_physical():
    basis = _basis(np.linspace(100.0, 20000.0, 128))
    delta = np.zeros((8, 8, 8))
    delta[:4] = 1.0
    velocity = np.zeros((3, 8, 8, 8))
    velocity[0] = 25.0
    result = calibration.generate_population_catalog(basis, delta, velocity, 7, _generator())
    catalog = result["catalog"]
    assert catalog["dist"].shape == (128,)
    assert catalog["nhat"].shape == (128, 3)
    np.testing.assert_allclose(np.linalg.norm(catalog["nhat"], axis=1), 1.0)
    assert np.all(result["true_distance"] > 0.0)
    expected_cz = (
        74.6 * result["true_distance"]
        + result["true_radial_velocity"]
        + catalog["nhat"] @ result["nuisance_truth"][:3]
        - result["true_distance"] * result["nuisance_truth"][3]
    )
    np.testing.assert_allclose(catalog["v3k"], expected_cz)
    assert np.unique(result["local_truth_delta"]).size > 1


def test_fidelity_gates_report_failure_without_relabeling_it_as_success():
    fidelity = {
        "conditioned_clean_group_count": 22136,
        "generated_clean_group_count": 22136,
        "BGc_selected_group_count": 19000,
        "observed_distance_KS": 0.02,
        "distance_error_mag_KS": 0.03,
        "redshift_velocity_KS": 0.12,
        "angular_histogram_total_variation": 0.04,
    }
    thresholds = {
        "conditioned_clean_group_count": 22136,
        "BGc_selected_group_count_min": 15450,
        "BGc_selected_group_count_max": 23176,
        "observed_distance_KS_max": 0.10,
        "distance_error_mag_KS_max": 0.05,
        "redshift_velocity_KS_max": 0.10,
        "angular_histogram_total_variation_max": 0.15,
    }
    result = calibration.population_fidelity_gates(fidelity, thresholds)
    assert result["redshift_velocity_KS_pass"] is False
    assert result["all_pass"] is False


def test_cluster_bootstrap_and_phase_null_are_deterministic():
    values = np.arange(64 * 3, dtype=float).reshape(64, 3)
    indices = calibration._bootstrap_indices(64, 100, calibration.BOOTSTRAP_SEED)
    first = calibration._bootstrap_interval(values, indices, statistic="mean")
    second = calibration._bootstrap_interval(values, indices, statistic="mean")
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    truth = np.ones((64, 4), dtype=np.complex128)
    mean = truth.copy()
    assignment = np.array([0, 0, 1, 1])
    bins = np.array([0, 1])
    p_first, null_first = calibration._phase_null(
        truth, mean, assignment, bins, replicate_count=256, seed=11
    )
    p_second, null_second = calibration._phase_null(
        truth, mean, assignment, bins, replicate_count=256, seed=11
    )
    np.testing.assert_array_equal(p_first, p_second)
    np.testing.assert_array_equal(null_first, null_second)
    assert np.all(p_first <= 0.01)


def test_complex_coverage_uses_two_components_except_self_conjugate():
    rng = np.random.default_rng(15)
    truth = rng.normal(size=(64, 2)) + 1j * rng.normal(size=(64, 2))
    truth[:, 0] = truth[:, 0].real
    mean = truth.copy()
    draws = mean[:, None, :] + (
        rng.normal(scale=0.2, size=(64, 16, 2))
        + 1j * rng.normal(scale=0.2, size=(64, 16, 2))
    )
    draws[:, :, 0] = draws[:, :, 0].real
    coverage = calibration._coverage_by_mock_bin(
        truth,
        mean,
        draws,
        np.array([0, 0]),
        np.array([True, False]),
        np.array([0]),
        1.0,
    )
    assert coverage.shape == (64, 1)
    np.testing.assert_array_equal(coverage, np.ones((64, 1)))


def test_domain_calibration_returns_every_preregistered_gate_array():
    rng = np.random.default_rng(21)
    truth = rng.normal(size=(64, 4)) + 1j * rng.normal(size=(64, 4))
    truth[:, 0] = truth[:, 0].real
    mean = 0.8 * truth
    draws = mean[:, None, :] + (
        rng.normal(scale=0.6, size=(64, 16, 4))
        + 1j * rng.normal(scale=0.6, size=(64, 16, 4))
    )
    draws[:, :, 0] = draws[:, :, 0].real
    indices = calibration._bootstrap_indices(64, 50, calibration.BOOTSTRAP_SEED)
    gates = {
        "phase_null_replicates": 32,
        "coverage68_nominal": 0.6826894921370859,
        "coverage95_nominal": 0.9544997361036416,
        "variance_bootstrap_upper_max_exclusive": 0.8,
        "phase_null_p_max_inclusive": 0.01,
        "coverage68_abs_error_max": 0.05,
        "coverage95_abs_error_max": 0.025,
    }
    metrics, arrays = calibration.compute_domain_calibration(
        domain_id="synthetic",
        truth=truth,
        mean=mean,
        draws=draws,
        prior_variance=np.ones(4),
        assignment=np.array([0, 0, 1, 1]),
        self_conjugate=np.array([True, False, False, False]),
        bin_ids=np.array([0, 1]),
        heldout_pass=np.array([True, False]),
        bootstrap_indices=indices,
        gates=gates,
        phase_seed=4,
    )
    assert metrics["domain_id"] == "synthetic"
    for name in (
        "response",
        "correlation_r",
        "residual_power_ratio",
        "phase_null_p_value",
        "coverage68",
        "coverage95",
        "strict_gate",
    ):
        assert arrays[name].shape == (2,)
    assert arrays["per_mock_variance_ratio_median"].shape == (64, 2)
    assert arrays["per_mock_coverage68"].shape == (64, 2)
    assert arrays["per_mock_coverage95"].shape == (64, 2)
    assert arrays["phase_null_cross"].shape == (32, 2)


def test_program_declares_development_only_and_no_0p3_claim():
    path = ROOT / "config/cf4_bgc_population_calibration_program_v1.json"
    if not path.exists():
        pytest.skip("program is materialized after source hashing")
    program = json.loads(path.read_text())
    assert program["development"]["mock_count"] == 64
    assert program["development"]["posterior_draw_count"] == 16
    assert program["authorization"]["development_64_mock_calibration"] is True
    assert program["authorization"]["untouched_256_mock_validation"] is False
    assert program["authorization"]["frontier_promotion"] is False
    assert program["authorization"]["IC_PM_HOP_RAMSES"] is False
    assert program["resolution_semantics"]["cell_size_cMpc_h"] == 12.0
    assert program["resolution_semantics"]["target_0p3_cMpc_h_reached"] is False
