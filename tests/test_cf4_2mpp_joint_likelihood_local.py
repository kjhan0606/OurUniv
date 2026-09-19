from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import gammaln

from cf4_2mpp_boundary_stress import boundary_stress_case
from cf4_2mpp_joint_likelihood_local import (
    LikelihoodInputError,
    joint_log_likelihood,
    observer_centred_spherical_rsd,
    poisson_log_likelihood,
    poisson_log_likelihood_from_log_intensity,
    population_masses_from_eta,
    population_log_masses_from_eta,
    predict_selected_intensity,
    shared_redshift_log_likelihood,
    tsc_deposit,
    validate_factor_ownership,
    gaussian_hermite_rule,
    quadrature_convergence_gate,
    QUADRATURE_HIGH_ORDERS,
    QUADRATURE_LOW_ORDER,
    QUADRATURE_RELATIVE_L1_TOLERANCE,
    QUADRATURE_STRESS_CASE_ID,
    VELOCITY_CONVENTION,
)


def _sources(count=7):
    positions = np.array(
        [[0.1, 0.2, 0.3], [1.9, 0.2, 0.4], [3.2, 3.1, 0.5],
         [5.9, 0.1, 5.8], [2.4, 4.8, 1.1], [4.2, 2.2, 5.5],
         [0.6, 5.7, 3.4]], dtype=np.float64
    )[:count]
    velocities = np.array(
        [[10.0, -20.0, 30.0], [-15.0, 4.0, 8.0], [2.0, 3.0, -7.0],
         [40.0, 0.0, -20.0], [-3.0, 11.0, 2.0], [12.0, -6.0, 4.0],
         [-8.0, 5.0, 9.0]], dtype=np.float64
    )[:count]
    return positions, velocities


def test_tsc_is_periodic_and_conserves_mass():
    positions = np.array([[0.0, 0.0, 0.0], [5.999999, 5.999999, 5.999999], [2.3, 4.7, 1.2]])
    masses = np.array([2.0, 3.5, 0.25], dtype=np.float64)
    deposited = tsc_deposit(positions, masses, 8, 6.0)
    assert deposited.shape == (8, 8, 8)
    assert np.all(deposited >= 0.0)
    assert np.sum(deposited) == pytest.approx(np.sum(masses), rel=0.0, abs=1e-14)
    shifted = tsc_deposit(positions + 6.0, masses, 8, 6.0)
    np.testing.assert_allclose(deposited, shifted, rtol=0.0, atol=1e-14)


def test_spherical_rsd_uses_local_line_of_sight_and_wraps():
    positions = np.array([[4.0, 3.0, 3.0], [2.0, 3.0, 3.0]], dtype=np.float64)
    velocities = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]], dtype=np.float64)
    result = observer_centred_spherical_rsd(
        positions, velocities, [3.0, 3.0, 3.0], 6.0, 100.0,
        little_h=0.5, scale_factor=1.0,
    )
    np.testing.assert_allclose(result.coherent_displacement_cMpc_h, [0.5, 0.5])
    np.testing.assert_allclose(result.positions, [[4.5, 3.0, 3.0], [1.5, 3.0, 3.0]])
    earlier = observer_centred_spherical_rsd(
        positions, velocities, [3.0, 3.0, 3.0], 6.0, 100.0,
        little_h=0.5, scale_factor=0.5,
    )
    np.testing.assert_allclose(earlier.coherent_displacement_cMpc_h, [1.0, 1.0])


def test_population_masses_are_positive_and_not_count_normalized():
    eta = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    masses = population_masses_from_eta(eta, np.zeros(6), np.log(np.arange(1.0, 7.0)))
    assert masses.shape == (6, 3)
    assert np.all(masses > 0.0)
    assert np.sum(masses[0]) != pytest.approx(3.0)
    np.testing.assert_allclose(
        np.exp(population_log_masses_from_eta(eta, np.zeros(6), np.log(np.arange(1.0, 7.0)))),
        masses,
    )


def test_selected_intensity_is_positive_and_scales_with_raw_exposure():
    positions, velocities = _sources()
    masses = np.ones((6, positions.shape[0]), dtype=np.float64)
    exposure = np.ones((6, 8, 8, 8), dtype=np.float64)
    kwargs = dict(
        observer=np.array([3.0, 3.0, 3.0]), box_size_cMpc_h=6.0,
        hubble_km_s_Mpc=100.0, little_h=0.746, scale_factor=1.0,
        sigma_fog_km_s=np.full(6, 20.0),
        sigma_redshift_km_s=np.full(6, 10.0),
    )
    first = predict_selected_intensity(positions, velocities, masses, exposure, **kwargs)
    second = predict_selected_intensity(positions, velocities, masses, 0.25 * exposure, **kwargs)
    assert first.shape == exposure.shape
    assert np.all(np.isfinite(first)) and np.all(first >= 0.0)
    np.testing.assert_allclose(second, 0.25 * first, rtol=0.0, atol=1e-14)
    assert np.sum(first[0]) == pytest.approx(np.sum(masses[0]), rel=0.0, abs=1e-12)


def test_unit_and_cubic_exposure_contracts_are_fail_closed():
    positions, velocities = _sources(2)
    masses = np.ones((6, 2), dtype=np.float64)
    exposure = np.ones((6, 4, 4, 3), dtype=np.float64)
    common = dict(
        observer=np.array([3.0, 3.0, 3.0]), box_size_cMpc_h=6.0,
        hubble_km_s_Mpc=100.0, little_h=0.746, scale_factor=1.0,
        sigma_fog_km_s=np.full(6, 20.0), sigma_redshift_km_s=np.full(6, 10.0),
    )
    with pytest.raises(LikelihoodInputError, match="cubic grid"):
        predict_selected_intensity(positions, velocities, masses, exposure, **common)
    with pytest.raises(LikelihoodInputError, match="little_h"):
        observer_centred_spherical_rsd(
            positions, velocities, [3.0, 3.0, 3.0], 6.0, 100.0,
            little_h=0.0, scale_factor=1.0,
        )
    with pytest.raises(LikelihoodInputError, match="scale_factor"):
        observer_centred_spherical_rsd(
            positions, velocities, [3.0, 3.0, 3.0], 6.0, 100.0,
            little_h=0.746, scale_factor=0.0,
        )
    with pytest.raises(LikelihoodInputError, match="velocity_convention"):
        observer_centred_spherical_rsd(
            positions, velocities, [3.0, 3.0, 3.0], 6.0, 100.0,
            little_h=0.746, scale_factor=1.0,
            velocity_convention="physical_total_velocity",
        )


def test_quadrature_contract_is_odd_order_and_normalized():
    for order in (3, 7, 9):
        nodes, weights = gaussian_hermite_rule(order)
        assert nodes.shape == weights.shape == (order,)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-14)
        assert np.sum(weights * nodes) == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(LikelihoodInputError, match="odd integer"):
        gaussian_hermite_rule(4)


def test_quadrature_convergence_gate_is_numeric_only_and_fail_closed():
    reference = np.ones((6, 2, 2, 2), dtype=np.float64)
    result = quadrature_convergence_gate(
        reference,
        reference,
        low_order=QUADRATURE_LOW_ORDER,
        high_order=QUADRATURE_HIGH_ORDERS[0],
        stress_case_id=QUADRATURE_STRESS_CASE_ID,
        relative_l1_tolerance=QUADRATURE_RELATIVE_L1_TOLERANCE,
    )
    assert result["status"] == "PASS"
    assert result["science_claim_authorized"] is False
    shifted = reference.copy()
    shifted[0, 0, 0, 0] *= 2.0
    failed = quadrature_convergence_gate(
        reference,
        shifted,
        low_order=QUADRATURE_LOW_ORDER,
        high_order=QUADRATURE_HIGH_ORDERS[0],
        stress_case_id=QUADRATURE_STRESS_CASE_ID,
        relative_l1_tolerance=QUADRATURE_RELATIVE_L1_TOLERANCE,
    )
    assert failed["status"] == "FAIL"
    with pytest.raises(LikelihoodInputError, match="preregistered"):
        quadrature_convergence_gate(
            reference,
            reference,
            low_order=QUADRATURE_LOW_ORDER,
            high_order=QUADRATURE_HIGH_ORDERS[0],
            stress_case_id=QUADRATURE_STRESS_CASE_ID,
            relative_l1_tolerance=0.0,
        )
    with pytest.raises(LikelihoodInputError, match="GH3"):
        quadrature_convergence_gate(
            reference,
            reference,
            low_order=3,
            high_order=5,
            stress_case_id=QUADRATURE_STRESS_CASE_ID,
            relative_l1_tolerance=QUADRATURE_RELATIVE_L1_TOLERANCE,
        )


def test_preregistered_boundary_stress_case_records_current_quadrature_failure():
    case = boundary_stress_case()
    fields = {
        order: predict_selected_intensity(
            case["positions"],
            case["velocities"],
            case["masses"],
            case["exposure"],
            quadrature_order=order,
            **case["kwargs"],
        )
        for order in (3, 7, 9)
    }
    gate_7 = quadrature_convergence_gate(
        fields[3], fields[7], low_order=3, high_order=7,
        stress_case_id=QUADRATURE_STRESS_CASE_ID,
        relative_l1_tolerance=QUADRATURE_RELATIVE_L1_TOLERANCE,
    )
    gate_9 = quadrature_convergence_gate(
        fields[3], fields[9], low_order=3, high_order=9,
        stress_case_id=QUADRATURE_STRESS_CASE_ID,
        relative_l1_tolerance=QUADRATURE_RELATIVE_L1_TOLERANCE,
    )
    assert gate_7["status"] == gate_9["status"] == "FAIL"
    assert gate_7["relative_l1"] > 0.6
    assert gate_9["relative_l1"] > 0.6


def test_poisson_factor_matches_direct_formula_and_rejects_zero_support():
    counts = np.zeros((6, 2, 2, 2), dtype=np.int64)
    counts[0, 0, 0, 0] = 2
    intensity = np.full(counts.shape, 0.75, dtype=np.float64)
    actual = poisson_log_likelihood(counts, intensity)
    expected = float(np.sum(counts * np.log(intensity) - intensity - gammaln(counts + 1.0)))
    assert actual == pytest.approx(expected)
    np.testing.assert_allclose(
        poisson_log_likelihood_from_log_intensity(counts, np.log(intensity)), expected
    )
    intensity[0, 0, 0, 0] = 0.0
    with pytest.raises(LikelihoodInputError, match="positive observed count"):
        poisson_log_likelihood(counts, intensity)


def test_shared_redshift_factor_matches_explicit_covariance():
    observed = np.array([10.0, 12.0, -4.0], dtype=np.float64)
    predicted = np.array([8.0, 11.0, -3.0], dtype=np.float64)
    sigma = np.array([2.0, 3.0, 1.5], dtype=np.float64)
    groups = np.array([0, 0, 1], dtype=np.int64)
    tau = np.array([1.25, 0.5], dtype=np.float64)
    actual = shared_redshift_log_likelihood(
        observed, predicted, sigma, groups, tau,
        secure_object_ids=["A", "B", "C"],
    )
    residual = observed - predicted
    covariance = np.diag(sigma**2)
    covariance[:2, :2] += tau[0] ** 2
    covariance[2, 2] += tau[1] ** 2
    sign, logdet = np.linalg.slogdet(covariance)
    expected = -0.5 * (residual @ np.linalg.solve(covariance, residual) + logdet + 3 * math.log(2.0 * math.pi))
    assert sign > 0.0
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_shared_redshift_manifest_dimension_is_exact_when_declared():
    observed = np.array([10.0, 12.0], dtype=np.float64)
    predicted = np.array([8.0, 11.0], dtype=np.float64)
    sigma = np.array([2.0, 3.0], dtype=np.float64)
    groups = np.array([0, 1], dtype=np.int64)
    tau = np.array([1.25, 0.5], dtype=np.float64)
    value = shared_redshift_log_likelihood(
        observed, predicted, sigma, groups, tau,
        secure_object_ids=["A", "B"], expected_group_count=2,
    )
    assert np.isfinite(value)
    with pytest.raises(LikelihoodInputError, match="manifest group count"):
        shared_redshift_log_likelihood(
            observed, predicted, sigma, groups, np.append(tau, 0.25),
            secure_object_ids=["A", "B"], expected_group_count=2,
        )
    with pytest.raises(LikelihoodInputError, match="all manifest group indices"):
        shared_redshift_log_likelihood(
            observed, predicted, sigma, np.array([0, 0], dtype=np.int64), tau,
            secure_object_ids=["A", "B"], expected_group_count=2,
        )


def test_joint_factor_is_exactly_count_plus_one_shared_redshift_factor():
    counts = np.zeros((6, 2, 2, 2), dtype=np.int64)
    intensity = np.ones_like(counts, dtype=np.float64)
    args = (
        np.array([1.0, 2.0]), np.array([0.0, 0.5]),
        np.array([1.0, 2.0]), np.array([0, 0], dtype=np.int64), np.array([0.75]),
    )
    assert joint_log_likelihood(
        counts, intensity, *args, secure_object_ids=["A", "B"]
    ) == pytest.approx(
        poisson_log_likelihood(counts, intensity) + shared_redshift_log_likelihood(
            *args, secure_object_ids=["A", "B"]
        )
    )


def test_factor_ownership_rejects_independent_redshift_factor_and_duplicate_ids():
    ownership = validate_factor_ownership(["A", "B"], np.array([0, 0], dtype=np.int64))
    assert ownership["count_factor_owner"] == "2Mpp_grid_counts"
    assert ownership["redshift_factor_owner"] == "CF4_group_marks_shared_redshift"
    with pytest.raises(LikelihoodInputError, match="unique"):
        validate_factor_ownership(["A", "A"], np.array([0, 0], dtype=np.int64))
    with pytest.raises(LikelihoodInputError, match=r"independent 2M\+\+ redshift factor"):
        validate_factor_ownership(
            ["A", "B"], np.array([0, 0], dtype=np.int64),
            independent_twompp_redshift_ids=["Z"],
        )


@pytest.mark.parametrize("bad", [np.ones((5, 2, 2, 2)), np.ones((6, 2, 2, 2), dtype=np.float32)])
def test_count_contract_is_strict(bad):
    with pytest.raises(LikelihoodInputError):
        poisson_log_likelihood(bad, np.ones((6, 2, 2, 2), dtype=np.float64))
