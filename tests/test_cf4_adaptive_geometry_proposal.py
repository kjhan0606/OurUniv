import math

import numpy as np

from cf4_adaptive_geometry_proposal import (
    LOG_UNIFORM_S2,
    adaptive_component_logpdf,
    antipodal_second_moment,
    antipodal_vmf_logpdf,
    canonical_axis,
    defensive_proposal_logpdf,
    draw_defensive_geometry,
    fit_adaptive_mixture,
    fit_cross_validated_mixture,
    run_synthetic_validation,
    sample_adaptive_component,
    solve_kappa,
    target_geometry_logpdf,
)


PRIOR = {
    "distribution": "diagonal_normal",
    "mean_mpc_h": [0.0, -6.0, 4.0],
    "sigma_mpc_h": [3.0, 3.0, 3.0],
}


PARAMETERS = {
    "alpha": [0.20, 0.30, 0.25, 0.25],
    "mean_mpc_h": [
        [-4.0, -8.0, 2.0],
        [4.0, -8.0, 2.0],
        [-4.0, -3.0, 6.0],
        [4.0, -3.0, 6.0],
    ],
    "covariance_mpc_h_squared": [
        [[2.25, 0.20, 0.00], [0.20, 1.44, 0.10], [0.00, 0.10, 1.00]],
        [[1.44, 0.00, 0.10], [0.00, 2.25, 0.20], [0.10, 0.20, 1.00]],
        [[1.00, 0.10, 0.00], [0.10, 1.44, 0.20], [0.00, 0.20, 2.25]],
        [[2.25, 0.00, 0.20], [0.00, 1.00, 0.10], [0.20, 0.10, 1.44]],
    ],
    "axis_direction": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    "axis_kappa": [0.0, 1.0, 5.0, 20.0],
}


def test_antipodal_density_is_even_and_normalized():
    rng = np.random.default_rng(3)
    axes = rng.normal(size=(1000, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    nodes, weights = np.polynomial.legendre.leggauss(256)
    quadrature_axes = np.column_stack((
        np.sqrt(1.0 - nodes**2), np.zeros_like(nodes), nodes
    ))
    for kappa in (0.0, 1.0, 20.0):
        positive = antipodal_vmf_logpdf(axes, [0.0, 0.0, 1.0], kappa)
        negative = antipodal_vmf_logpdf(-axes, [0.0, 0.0, 1.0], kappa)
        np.testing.assert_allclose(positive, negative, rtol=0.0, atol=1e-12)
        density = np.exp(antipodal_vmf_logpdf(
            quadrature_axes, [0.0, 0.0, 1.0], kappa
        ))
        integral = 2.0 * math.pi * (weights @ density)
        np.testing.assert_allclose(integral, 1.0, rtol=0.0, atol=5e-12)
    np.testing.assert_allclose(
        antipodal_vmf_logpdf(axes, [1.0, 0.0, 0.0], 0.0),
        LOG_UNIFORM_S2,
        rtol=0.0,
        atol=0.0,
    )


def test_canonical_axis_and_kappa_solver_are_deterministic():
    np.testing.assert_array_equal(canonical_axis([-2.0, 1.0, 0.0]), [2 / np.sqrt(5), -1 / np.sqrt(5), 0])
    for kappa in (0.0, 0.1, 1.0, 10.0, 20.0):
        if kappa == 0.0:
            resultant = 0.0
        else:
            resultant = 1.0 / np.tanh(kappa) - 1.0 / kappa
        np.testing.assert_allclose(solve_kappa(resultant), kappa, rtol=0.0, atol=1e-11)


def test_defensive_density_has_exact_analytic_bound_and_seed_reproducibility():
    first = [draw_defensive_geometry(PRIOR, PARAMETERS, 2026082003, i) for i in range(100)]
    second = [draw_defensive_geometry(PRIOR, PARAMETERS, 2026082003, i) for i in range(100)]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left["midpoint_offset_mpc_h"], right["midpoint_offset_mpc_h"])
        np.testing.assert_array_equal(left["axis"], right["axis"])
        assert left["proposal_branch"] == right["proposal_branch"]
        assert left["proposal_component"] == right["proposal_component"]
    midpoint = np.asarray([row["midpoint_offset_mpc_h"] for row in first])
    axes = np.asarray([row["axis"] for row in first])
    log_ratio = target_geometry_logpdf(midpoint, PRIOR) - defensive_proposal_logpdf(
        midpoint, axes, PRIOR, PARAMETERS
    )
    assert np.max(log_ratio) <= math.log(2.0) + 1e-12


def test_antipodal_sampler_matches_analytic_axis_second_moment():
    rng = np.random.Generator(np.random.PCG64DXSM(43))
    _, axes, component = sample_adaptive_component(rng, PARAMETERS, 150000)
    for index in range(4):
        selected = axes[component == index]
        actual = np.einsum("ni,nj->ij", selected, selected) / len(selected)
        expected = antipodal_second_moment(
            PARAMETERS["axis_direction"][index], PARAMETERS["axis_kappa"][index]
        )
        assert np.max(np.abs(actual - expected)) < 0.015


def test_weighted_em_recovers_a_finite_normalized_four_component_fit():
    rng = np.random.Generator(np.random.PCG64DXSM(51))
    midpoint, axes, _ = sample_adaptive_component(rng, PARAMETERS, 2400)
    result = fit_adaptive_mixture(
        midpoint, axes, np.ones(len(midpoint)), fold_id=4, master_seed=2026082002
    )
    fitted = result["parameters"]
    assert result["selected_restart"] in range(8)
    np.testing.assert_allclose(np.sum(fitted["alpha"]), 1.0, rtol=0.0, atol=1e-12)
    assert np.all(np.isfinite(adaptive_component_logpdf(midpoint, axes, fitted)))
    for covariance in fitted["covariance_mpc_h_squared"]:
        eigenvalues = np.linalg.eigvalsh(covariance)
        assert np.min(eigenvalues) >= 0.75**2 - 1e-12
        assert np.max(eigenvalues) <= 6.0**2 + 1e-12
    assert min(result["component_effective_membership"]) >= 4.0


def test_four_fold_cross_validation_is_deterministic_and_positive_on_h_samples():
    rng = np.random.Generator(np.random.PCG64DXSM(919))
    midpoint, axes, _ = sample_adaptive_component(rng, PARAMETERS, 1600)
    target = target_geometry_logpdf(midpoint, PRIOR)
    first = fit_cross_validated_mixture(
        midpoint, axes, np.zeros(len(midpoint)), target, 2026082002
    )
    second = fit_cross_validated_mixture(
        midpoint, axes, np.zeros(len(midpoint)), target, 2026082002
    )
    first_delta = [row["holdout_delta"] for row in first["folds"]]
    second_delta = [row["holdout_delta"] for row in second["folds"]]
    np.testing.assert_array_equal(first_delta, second_delta)
    assert first["all_holdout_delta_nonnegative"] is True
    assert min(first_delta) > 0.0
    assert first["full_fit"]["parameters"] == second["full_fit"]["parameters"]


def test_reduced_synthetic_validation_contract_passes():
    result = run_synthetic_validation(
        PRIOR, PARAMETERS, master_seed=2026082004, sampling_draw_count=50000
    )
    assert result["all_pass"], result
