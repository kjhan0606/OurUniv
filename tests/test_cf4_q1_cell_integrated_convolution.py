from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import ndtr

from cf4_2mpp_boundary_stress import boundary_stress_case, nonboundary_stress_case
from cf4_2mpp_joint_likelihood_local import (
    observer_centred_spherical_rsd,
    predict_selected_intensity,
    tsc_deposit,
)
from cf4_q1_cell_integrated_convolution import (
    Q1_CANDIDATE_RELATIVE_L1_TOLERANCE,
    Q1_DEFAULT_TAIL_CUTOFF,
    _axis_breakpoints,
    _integrate_particle_weights,
    cell_integrated_tsc_deposit,
    exposure_weighted_totals,
    gaussian_tail_probability,
    mass_conservation_gate,
    predict_selected_intensity_cell_integrated,
    q1_candidate_oracle_gate,
)


def _rhat(positions, observer, box_size):
    rel = (positions - observer + box_size / 2.0) % box_size
    rel -= box_size / 2.0
    return rel / np.linalg.norm(rel, axis=1)[:, None]


def test_gaussian_tail_is_below_frozen_tolerance():
    assert gaussian_tail_probability() < 1.0e-8


def test_zero_displacement_matches_conservative_point_tsc():
    positions = np.array([[0.13, 1.27, 3.41], [5.99, 2.03, 0.77]], dtype=np.float64)
    masses = np.array([2.0, 0.75], dtype=np.float64)
    los = _rhat(positions, np.array([3.0, 3.0, 3.0]), 6.0)
    q1 = cell_integrated_tsc_deposit(
        positions, masses, los, np.zeros(2), 8, 6.0
    )
    direct = tsc_deposit(positions, masses, 8, 6.0)
    np.testing.assert_allclose(q1, direct, rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    "fixture, source_index",
    [(boundary_stress_case, 0), (nonboundary_stress_case, 1)],
)
def test_oracle_agrees_with_independent_adaptive_reference(fixture, source_index):
    case = fixture()
    position = case["positions"][source_index]
    kwargs = case["kwargs"]
    rsd = observer_centred_spherical_rsd(
        case["positions"], case["velocities"], kwargs["observer"],
        kwargs["box_size_cMpc_h"], kwargs["hubble_km_s_Mpc"],
        little_h=kwargs["little_h"], scale_factor=kwargs["scale_factor"],
    )
    rhat = _rhat(rsd.positions, kwargs["observer"], kwargs["box_size_cMpc_h"])[source_index]
    scale = kwargs["little_h"] * math.hypot(
        kwargs["sigma_fog_km_s"][0], kwargs["sigma_redshift_km_s"][0]
    ) / (kwargs["scale_factor"] * kwargs["hubble_km_s_Mpc"])
    displacement = scale * rhat
    oracle, oracle_diagnostics = _integrate_particle_weights(position=rsd.positions[source_index], displacement_vector=displacement, grid_size=8, box_size=6.0, tail_cutoff=8.0, return_diagnostics=True)
    assert abs(oracle_diagnostics["pre_renormalization_mass_defect"]) < 1.0e-10

    def integrand(epsilon):
        density = math.exp(-0.5 * epsilon * epsilon) / math.sqrt(2.0 * math.pi)
        return tsc_deposit(
            (rsd.positions[source_index] + epsilon * displacement)[None, :],
            np.array([1.0], dtype=np.float64),
            8,
            6.0,
        ) * density

    boundaries = np.unique(np.asarray(
        [-8.0, 8.0]
        + sum(
            [_axis_breakpoints(rsd.positions[source_index][axis], displacement[axis], 6.0 / 8.0, 8.0) for axis in range(3)],
            [],
        ),
        dtype=np.float64,
    ))
    reference = np.zeros((8, 8, 8), dtype=np.float64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        value, _error = quad_vec(integrand, float(left), float(right), epsabs=1.0e-13, epsrel=1.0e-11)
        reference += value
    reference /= ndtr(8.0) - ndtr(-8.0)
    np.testing.assert_allclose(oracle, reference, rtol=0.0, atol=2.0e-11)


@pytest.mark.parametrize("fixture", [boundary_stress_case, nonboundary_stress_case])
def test_predictor_is_finite_and_mass_conservative(fixture):
    case = fixture()
    kwargs = case["kwargs"]
    intensity = predict_selected_intensity_cell_integrated(
        case["positions"], case["velocities"], case["masses"],
        np.ones_like(case["exposure"]), **kwargs
    )
    assert np.all(np.isfinite(intensity))
    assert np.all(intensity >= 0.0)
    report = mass_conservation_gate(intensity, case["masses"])
    assert report["status"] == "PASS"
    assert report["max_absolute_error"] <= 1.0e-12


def test_selection_is_applied_after_convolution_and_totals_are_diagnostic():
    case = nonboundary_stress_case()
    kwargs = case["kwargs"]
    first = predict_selected_intensity_cell_integrated(
        case["positions"], case["velocities"], case["masses"],
        case["exposure"], **kwargs
    )
    second = predict_selected_intensity_cell_integrated(
        case["positions"], case["velocities"], case["masses"],
        0.5 * case["exposure"], **kwargs
    )
    np.testing.assert_allclose(second, 0.5 * first, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        exposure_weighted_totals(first, np.ones_like(first)),
        np.sum(first, axis=(1, 2, 3)),
        rtol=0.0,
        atol=2.0e-14,
    )


def test_pre_renormalization_and_sliver_bounds_are_recorded():
    case = boundary_stress_case()
    positions = case["positions"][:3]
    observer = case["kwargs"]["observer"]
    box = case["kwargs"]["box_size_cMpc_h"]
    los = _rhat(positions, observer, box)
    field, diagnostics = cell_integrated_tsc_deposit(
        positions,
        np.ones(positions.shape[0], dtype=np.float64),
        los,
        np.full(positions.shape[0], 2.0, dtype=np.float64),
        8,
        box,
        return_diagnostics=True,
    )
    assert np.all(np.isfinite(field))
    assert diagnostics["renormalization_is_applied"] is True
    assert diagnostics["max_pre_renormalization_mass_defect"] <= 1.0e-10
    assert diagnostics["max_dropped_sliver_probability"] <= 1.0e-8
    assert diagnostics["max_clipped_negative_mass"] <= 1.0e-13


def test_tail_cutoff_6_8_10_converges_against_frozen_tail_gate():
    case = nonboundary_stress_case()
    position = case["positions"][:1]
    los = _rhat(position, case["kwargs"]["observer"], case["kwargs"]["box_size_cMpc_h"])
    fields = {
        cutoff: cell_integrated_tsc_deposit(
            position,
            np.ones(1, dtype=np.float64),
            los,
            np.full(1, 2.0, dtype=np.float64),
            8,
            case["kwargs"]["box_size_cMpc_h"],
            tail_cutoff=cutoff,
        )
        for cutoff in (6.0, 8.0, 10.0)
    }
    for lower, upper in ((6.0, 8.0), (8.0, 10.0)):
        error = np.sum(np.abs(fields[lower] - fields[upper])) / np.sum(np.abs(fields[upper]))
        assert error <= 1.0e-8


def test_dedicated_sigma_disp_2L_multiwrap_matches_independent_reference():
    box = 6.0
    grid_size = 4
    position = np.array([0.731, 1.367, 2.413], dtype=np.float64)
    los = np.array([1.0, math.sqrt(2.0), math.pi], dtype=np.float64)
    los /= np.linalg.norm(los)
    sigma_disp = 2.0 * box
    oracle, diagnostics = cell_integrated_tsc_deposit(
        position[None, :],
        np.ones(1, dtype=np.float64),
        los[None, :],
        np.array([sigma_disp], dtype=np.float64),
        grid_size,
        box,
        return_diagnostics=True,
    )
    assert sigma_disp > box
    assert diagnostics["tail_probability"] < 1.0e-8

    # Independent breakpoint construction: this intentionally does not call
    # the oracle's _axis_breakpoints helper.
    spacing = box / grid_size
    tail = 8.0
    displacement = sigma_disp * los
    boundaries = [-tail, tail]
    for axis in range(3):
        lo = position[axis] - abs(displacement[axis]) * tail
        hi = position[axis] + abs(displacement[axis]) * tail
        for cell_boundary in range(math.floor(lo / spacing) - 1, math.ceil(hi / spacing) + 2):
            epsilon = (cell_boundary * spacing - position[axis]) / displacement[axis]
            if -tail < epsilon < tail:
                boundaries.append(float(epsilon))
    breaks = np.unique(np.asarray(boundaries, dtype=np.float64))
    reference = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

    def integrand(epsilon):
        density = math.exp(-0.5 * epsilon * epsilon) / math.sqrt(2.0 * math.pi)
        shifted = (position + epsilon * displacement)[None, :]
        return tsc_deposit(shifted, np.array([1.0], dtype=np.float64), grid_size, box) * density

    for left, right in zip(breaks[:-1], breaks[1:]):
        value, _error = quad_vec(integrand, float(left), float(right), epsabs=1.0e-12, epsrel=1.0e-10)
        reference += value
    reference /= ndtr(tail) - ndtr(-tail)
    np.testing.assert_allclose(oracle, reference, rtol=0.0, atol=5.0e-10)


@pytest.mark.parametrize("fixture", [boundary_stress_case, nonboundary_stress_case])
def test_all_populations_and_three_sources_have_adaptive_reference_checks(fixture):
    """Close the Q1 audit gap for every population and representative source."""

    case = fixture()
    kwargs = case["kwargs"]
    box = kwargs["box_size_cMpc_h"]
    grid_size = 4
    rsd = observer_centred_spherical_rsd(
        case["positions"], case["velocities"], kwargs["observer"],
        box, kwargs["hubble_km_s_Mpc"], little_h=kwargs["little_h"],
        scale_factor=kwargs["scale_factor"],
    )
    rhat = _rhat(rsd.positions, kwargs["observer"], box)
    spacing = box / grid_size
    for source_index in range(3):
        for population in range(6):
            sigma_disp = kwargs["little_h"] * math.hypot(
                kwargs["sigma_fog_km_s"][population],
                kwargs["sigma_redshift_km_s"][population],
            ) / (kwargs["scale_factor"] * kwargs["hubble_km_s_Mpc"])
            displacement = sigma_disp * rhat[source_index]
            oracle = cell_integrated_tsc_deposit(
                rsd.positions[source_index : source_index + 1],
                np.ones(1, dtype=np.float64),
                rhat[source_index : source_index + 1],
                np.array([sigma_disp], dtype=np.float64),
                grid_size,
                box,
            )
            tail = 8.0
            boundaries = [-tail, tail]
            for axis in range(3):
                lo = rsd.positions[source_index, axis] - abs(displacement[axis]) * tail
                hi = rsd.positions[source_index, axis] + abs(displacement[axis]) * tail
                for cell_boundary in range(math.floor(lo / spacing) - 1, math.ceil(hi / spacing) + 2):
                    epsilon = (cell_boundary * spacing - rsd.positions[source_index, axis]) / displacement[axis]
                    if -tail < epsilon < tail:
                        boundaries.append(float(epsilon))
            breaks = np.unique(np.asarray(boundaries, dtype=np.float64))
            reference = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

            def integrand(epsilon):
                density = math.exp(-0.5 * epsilon * epsilon) / math.sqrt(2.0 * math.pi)
                shifted = (rsd.positions[source_index] + epsilon * displacement)[None, :]
                return tsc_deposit(shifted, np.array([1.0], dtype=np.float64), grid_size, box) * density

            for left, right in zip(breaks[:-1], breaks[1:]):
                value, _error = quad_vec(
                    integrand, float(left), float(right), epsabs=1.0e-11, epsrel=1.0e-9
                )
                reference += value
            reference /= ndtr(tail) - ndtr(-tail)
            np.testing.assert_allclose(oracle, reference, rtol=0.0, atol=2.0e-8)


def test_candidate_oracle_gate_is_new_and_fail_closed():
    oracle = np.ones((6, 2, 2, 2), dtype=np.float64)
    assert q1_candidate_oracle_gate(oracle, oracle)["status"] == "PASS"
    candidate = oracle.copy()
    candidate[0] *= 1.01
    result = q1_candidate_oracle_gate(candidate, oracle)
    assert result["status"] == "FAIL"
    assert result["science_claim_authorized"] is False
    assert result["relative_l1_tolerance"] == Q1_CANDIDATE_RELATIVE_L1_TOLERANCE


@pytest.mark.parametrize("fixture", [boundary_stress_case, nonboundary_stress_case])
def test_actual_legacy_gh_candidates_are_recorded_as_oracle_failures(fixture):
    case = fixture()
    kwargs = case["kwargs"]
    exposure = np.ones_like(case["exposure"])
    oracle = predict_selected_intensity_cell_integrated(
        case["positions"], case["velocities"], case["masses"], exposure, **kwargs
    )
    for order in (3, 7, 9, 15):
        candidate = predict_selected_intensity(
            case["positions"], case["velocities"], case["masses"], exposure,
            quadrature_order=order, **kwargs
        )
        gate = q1_candidate_oracle_gate(candidate, oracle)
        assert gate["status"] == "FAIL"
        assert gate["science_claim_authorized"] is False
