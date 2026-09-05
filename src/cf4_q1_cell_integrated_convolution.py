"""Q1 NumPy oracle for a periodic, cell-integrated Gaussian LOS response.

The existing likelihood deposits a point-shifted particle with a finite-order
Gauss--Hermite rule.  That is useful as a diagnostic, but it is not a reliable
observation operator at cell boundaries.  This module evaluates the frozen Q1
operator directly: on every interval where the periodic TSC neighbour stencil
is fixed, the product of the three one-dimensional TSC polynomials is integrated
against the standard-normal displacement analytically through degree six.

The implementation is an in-memory development oracle.  It performs no file
I/O, does not access GPFS or Slurm, and carries no permission to run inference
or make a resolution claim.  Gaussian tails outside the fixed cutoff are
renormalised per particle; the omitted probability is reported and is below the
frozen tail tolerance for the default cutoff.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr

from cf4_2mpp_joint_likelihood_local import (
    LikelihoodInputError,
    POPULATIONS,
    VELOCITY_CONVENTION,
    observer_centred_spherical_rsd,
    tsc_deposit,
)


Q1_CANDIDATE_RELATIVE_L1_TOLERANCE = 5.0e-3
Q1_CANDIDATE_ABSOLUTE_L1_FLOOR = 1.0e-12
Q1_ORACLE_INTERNAL_RELATIVE_L1_TOLERANCE = 5.0e-5
Q1_JAX_RELATIVE_L1_TOLERANCE = 1.0e-6
Q1_JAX_ABSOLUTE_L1_FLOOR = 1.0e-10
Q1_FALLBACK_RELATIVE_L1_TOLERANCE = 1.0e-8
Q1_TAIL_TRUNCATION_RELATIVE_L1_TOLERANCE = 1.0e-8
Q1_MASS_RELATIVE_TOLERANCE = 1.0e-10
Q1_MASS_ABSOLUTE_TOLERANCE = 1.0e-12
Q1_GRADIENT_RELATIVE_L1_TOLERANCE = 1.0e-5
Q1_GRADIENT_ABSOLUTE_L1_FLOOR = 1.0e-8
Q1_DEFAULT_TAIL_CUTOFF = 8.0
Q1_ORACLE_NEGATIVE_CLIP_TOLERANCE = 1.0e-13
Q1_NORMAL_MOMENT_TAIL_SWITCH = 5.0
Q1_SLIVER_EPSILON = 1.0e-10
Q1_SLIVER_PROBABILITY_TOLERANCE = 1.0e-8
Q1_OPERATOR_NAME = "periodic_torus_expected_cell_response"


def _validate_source_arrays(
    positions: Iterable[Iterable[float]],
    masses: Iterable[float],
    los_unit_vectors: Iterable[Iterable[float]],
    displacement_scales: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_array = np.asarray(positions, dtype=np.float64)
    mass_array = np.asarray(masses, dtype=np.float64)
    los_array = np.asarray(los_unit_vectors, dtype=np.float64)
    scale_array = np.asarray(displacement_scales, dtype=np.float64)
    if position_array.ndim != 2 or position_array.shape[1] != 3 or position_array.shape[0] == 0:
        raise LikelihoodInputError("positions must have shape (M, 3)")
    count = position_array.shape[0]
    if mass_array.shape != (count,):
        raise LikelihoodInputError("masses must have shape (M,)")
    if los_array.shape != (count, 3):
        raise LikelihoodInputError("los_unit_vectors must have shape (M, 3)")
    if scale_array.shape != (count,):
        raise LikelihoodInputError("displacement_scales must have shape (M,)")
    if (
        not np.all(np.isfinite(position_array))
        or not np.all(np.isfinite(mass_array))
        or not np.all(np.isfinite(los_array))
        or not np.all(np.isfinite(scale_array))
    ):
        raise LikelihoodInputError("source arrays must be finite")
    if np.any(mass_array < 0.0) or np.any(scale_array < 0.0):
        raise LikelihoodInputError("masses and displacement_scales must be non-negative")
    norms = np.linalg.norm(los_array, axis=1)
    if np.any(norms <= 0.0) or not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-12):
        raise LikelihoodInputError("los_unit_vectors must be finite unit vectors")
    return position_array, mass_array, los_array, scale_array


def gaussian_tail_probability(tail_cutoff: float = Q1_DEFAULT_TAIL_CUTOFF) -> float:
    """Return the omitted two-sided standard-normal tail probability."""

    if not math.isfinite(tail_cutoff) or tail_cutoff <= 0.0:
        raise LikelihoodInputError("tail_cutoff must be positive and finite")
    return float(2.0 * ndtr(-tail_cutoff))


def _normal_moments(
    left: float, right: float, degree: int, *, center: float = 0.0
) -> np.ndarray:
    """Return Gaussian moments in the local coordinate ``y=epsilon-center``."""

    if not left < right or degree < 0 or not math.isfinite(center):
        raise LikelihoodInputError("normal-moment interval must be non-empty")
    # Global powers of epsilon are ill-conditioned at several sigma.  Evaluate
    # all local moments together; the local polynomial coefficients stay O(1).
    if center != 0.0:
        left_local = left - center
        right_local = right - center

        def local_integrand(value: float) -> np.ndarray:
            density = math.exp(-0.5 * (center + value) ** 2) / math.sqrt(2.0 * math.pi)
            return np.asarray([density * value**order for order in range(degree + 1)])

        moments, _error = quad_vec(
            local_integrand,
            left_local,
            right_local,
            epsabs=1.0e-20,
            epsrel=1.0e-12,
        )
        return np.asarray(moments, dtype=np.float64)
    # At center zero, use the closed standard-normal recurrence in the central
    # region and a vector integral in the far tail to avoid CDF cancellation.
    if max(abs(left), abs(right)) > Q1_NORMAL_MOMENT_TAIL_SWITCH:
        def integrand(value: float) -> np.ndarray:
            density = math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)
            return np.asarray([density * value**order for order in range(degree + 1)])

        moments, _error = quad_vec(
            integrand, left, right, epsabs=1.0e-20, epsrel=1.0e-12
        )
        return np.asarray(moments, dtype=np.float64)
    moments = np.zeros(degree + 1, dtype=np.float64)
    phi_left = math.exp(-0.5 * left * left) / math.sqrt(2.0 * math.pi)
    phi_right = math.exp(-0.5 * right * right) / math.sqrt(2.0 * math.pi)
    moments[0] = float(ndtr(right) - ndtr(left))
    if degree == 0:
        return moments
    moments[1] = phi_left - phi_right
    for order in range(2, degree + 1):
        moments[order] = (
            (order - 1) * moments[order - 2]
            + (left ** (order - 1)) * phi_left
            - (right ** (order - 1)) * phi_right
        )
    return moments


def _tsc_polynomial(offset_slope: float, offset_intercept: float, neighbour: int) -> np.ndarray:
    """Return ascending polynomial coefficients in epsilon for one TSC weight."""

    slope = offset_slope
    intercept = offset_intercept
    if neighbour == -1:
        return np.array(
            [0.125 - 0.5 * intercept + 0.5 * intercept**2,
             -0.5 * slope + intercept * slope,
             0.5 * slope**2],
            dtype=np.float64,
        )
    if neighbour == 0:
        return np.array(
            [0.75 - intercept**2, -2.0 * intercept * slope, -slope**2],
            dtype=np.float64,
        )
    if neighbour == 1:
        return np.array(
            [0.125 + 0.5 * intercept + 0.5 * intercept**2,
             0.5 * slope + intercept * slope,
             0.5 * slope**2],
            dtype=np.float64,
        )
    raise LikelihoodInputError("TSC neighbour must be -1, 0, or 1")


def _axis_breakpoints(
    position: float,
    displacement: float,
    spacing: float,
    tail_cutoff: float,
) -> list[float]:
    """Find epsilon values where a periodic one-dimensional TSC stencil changes."""

    if displacement == 0.0:
        return []
    left = position - abs(displacement) * tail_cutoff
    right = position + abs(displacement) * tail_cutoff
    first = math.floor(left / spacing) - 1
    last = math.ceil(right / spacing) + 1
    values: list[float] = []
    for cell_boundary in range(first, last + 1):
        epsilon = (cell_boundary * spacing - position) / displacement
        if -tail_cutoff < epsilon < tail_cutoff:
            values.append(float(epsilon))
    return values


def _interval_stencil(
    positions: np.ndarray,
    displacement_vector: np.ndarray,
    midpoint: float,
    spacing: float,
    box_size: float,
    grid_size: int,
) -> tuple[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return fixed nearest nodes and the three epsilon polynomials per axis."""

    nearest_nodes: list[int] = []
    polynomials: list[np.ndarray] = []
    for axis in range(3):
        wrapped = float((positions[axis] + displacement_vector[axis] * midpoint) % box_size)
        nearest = int(math.floor(wrapped / spacing)) % grid_size
        offset_midpoint = wrapped / spacing - 0.5 - nearest
        slope = displacement_vector[axis] / spacing
        nearest_nodes.append(nearest)
        polynomials.append(
            np.stack(
                # The polynomial variable is y = epsilon - midpoint, so the
                # intercept is the offset at the midpoint rather than a large
                # global-epsilon intercept.
                [_tsc_polynomial(slope, offset_midpoint, neighbour) for neighbour in (-1, 0, 1)],
                axis=0,
            )
        )
    return tuple(nearest_nodes), tuple(polynomials)  # type: ignore[return-value]


def _integrate_particle_weights(
    position: np.ndarray,
    displacement_vector: np.ndarray,
    grid_size: int,
    box_size: float,
    tail_cutoff: float,
    *,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """Integrate one particle's 27 periodic TSC weights over Gaussian epsilon."""

    spacing = box_size / grid_size
    boundaries = [-tail_cutoff, tail_cutoff]
    for axis in range(3):
        boundaries.extend(
            _axis_breakpoints(position[axis], displacement_vector[axis], spacing, tail_cutoff)
        )
    raw_breaks = np.unique(np.asarray(boundaries, dtype=np.float64))
    # Distinct coordinate planes can intersect the same ray at nearly the
    # same epsilon.  Avoid ill-conditioned sliver intervals whose polynomial
    # moment subtraction is below float64 resolution; their total probability
    # is far below the absolute Q1 gate and the final per-particle
    # renormalisation preserves conservation.
    breaks = [float(raw_breaks[0])]
    dropped_sliver_probability = 0.0
    for boundary in raw_breaks[1:]:
        gap = float(boundary) - breaks[-1]
        if gap > Q1_SLIVER_EPSILON:
            breaks.append(float(boundary))
        else:
            dropped_sliver_probability += float(ndtr(boundary) - ndtr(breaks[-1]))
    deposit = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    clipped_negative_mass = 0.0
    for left, right in zip(breaks[:-1], breaks[1:]):
        if not right > left:
            continue
        midpoint = 0.5 * (left + right)
        nearest, axis_polynomials = _interval_stencil(
            position, displacement_vector, midpoint, spacing, box_size, grid_size
        )
        moments = _normal_moments(float(left), float(right), 6, center=midpoint)
        for ix, dx in enumerate((-1, 0, 1)):
            for iy, dy in enumerate((-1, 0, 1)):
                for iz, dz in enumerate((-1, 0, 1)):
                    polynomial = np.polynomial.polynomial.polymul(
                        np.polynomial.polynomial.polymul(
                            axis_polynomials[0][ix], axis_polynomials[1][iy]
                        ),
                        axis_polynomials[2][iz],
                    )
                    value = float(np.dot(polynomial, moments[: polynomial.size]))
                    # Recurrence cancellation in the probability moments is
                    # possible in the last ~8-sigma tail intervals.  The
                    # bound is far below the release L1 gate; larger negative
                    # values indicate a genuine polynomial/orientation bug.
                    if value < -Q1_ORACLE_NEGATIVE_CLIP_TOLERANCE:
                        raise LikelihoodInputError(
                            "cell-integrated TSC oracle produced a negative weight"
                        )
                    if value < 0.0:
                        clipped_negative_mass += -value
                    value = max(0.0, value)
                    deposit[
                        (nearest[0] + dx) % grid_size,
                        (nearest[1] + dy) % grid_size,
                        (nearest[2] + dz) % grid_size,
                    ] += value
    total = float(np.sum(deposit))
    if not math.isfinite(total) or total <= 0.0:
        raise LikelihoodInputError("cell-integrated TSC oracle has zero or non-finite mass")
    truncation_probability = float(ndtr(tail_cutoff) - ndtr(-tail_cutoff))
    diagnostics = {
        "pre_renormalization_total": total,
        "pre_renormalization_mass_defect": total - truncation_probability,
        "dropped_sliver_probability": dropped_sliver_probability,
        "clipped_negative_mass": clipped_negative_mass,
        "tail_probability": gaussian_tail_probability(tail_cutoff),
    }
    # The finite cutoff omits only the declared Gaussian tail.  Renormalisation
    # makes the conservative mass contract exact while the omitted probability
    # remains available to the caller through gaussian_tail_probability().
    deposit /= total
    if not return_diagnostics:
        return deposit
    if isinstance(deposit, np.ndarray):
        return deposit, diagnostics
    raise LikelihoodInputError("internal cell-integrated deposit type error")


def cell_integrated_tsc_deposit(
    positions: Iterable[Iterable[float]],
    masses: Iterable[float],
    los_unit_vectors: Iterable[Iterable[float]],
    displacement_scales: Iterable[float],
    grid_size: int,
    box_size_cMpc_h: float,
    *,
    tail_cutoff: float = Q1_DEFAULT_TAIL_CUTOFF,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Return the mass-preserving periodic cell-integrated Gaussian TSC field.

    ``displacement_scales[p]`` is the one-sigma displacement in cMpc/h and
    ``los_unit_vectors[p]`` is the observer-centred direction.  The stochastic
    displacement is ``displacement_scales[p] * epsilon * los_unit_vectors[p]``
    with epsilon standard normal.  This is the frozen Q1 torus expectation
    operator, evaluated without finite-order point-deposit quadrature.
    """

    if not isinstance(grid_size, int) or grid_size < 2:
        raise LikelihoodInputError("grid_size must be an integer >= 2")
    if not math.isfinite(box_size_cMpc_h) or box_size_cMpc_h <= 0.0:
        raise LikelihoodInputError("box_size_cMpc_h must be positive and finite")
    if not math.isfinite(tail_cutoff) or tail_cutoff <= 0.0:
        raise LikelihoodInputError("tail_cutoff must be positive and finite")
    positions_array, masses_array, los_array, scale_array = _validate_source_arrays(
        positions, masses, los_unit_vectors, displacement_scales
    )
    if np.any(positions_array < 0.0) or np.any(positions_array >= box_size_cMpc_h):
        raise LikelihoodInputError("positions must lie in [0, box_size)")
    result = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    particle_diagnostics: list[dict[str, float]] = []
    for particle in range(positions_array.shape[0]):
        mass = masses_array[particle]
        if mass == 0.0:
            particle_diagnostics.append(
                {
                    "pre_renormalization_total": 0.0,
                    "pre_renormalization_mass_defect": 0.0,
                    "dropped_sliver_probability": 0.0,
                    "clipped_negative_mass": 0.0,
                    "tail_probability": gaussian_tail_probability(tail_cutoff),
                }
            )
            continue
        if scale_array[particle] == 0.0:
            result += mass * tsc_deposit(
                positions_array[particle : particle + 1],
                np.asarray([1.0], dtype=np.float64),
                grid_size,
                box_size_cMpc_h,
            )
            particle_diagnostics.append(
                {
                    "pre_renormalization_total": 1.0,
                    "pre_renormalization_mass_defect": 0.0,
                    "dropped_sliver_probability": 0.0,
                    "clipped_negative_mass": 0.0,
                    "tail_probability": 0.0,
                }
            )
            continue
        displacement = scale_array[particle] * los_array[particle]
        particle_field, diagnostics = _integrate_particle_weights(
            positions_array[particle], displacement, grid_size, box_size_cMpc_h, tail_cutoff,
            return_diagnostics=True,
        )
        result += mass * particle_field
        particle_diagnostics.append(diagnostics)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise LikelihoodInputError("cell-integrated TSC result is not finite and non-negative")
    if not return_diagnostics:
        return result
    diagnostics = {
        "operator": Q1_OPERATOR_NAME,
        "tail_cutoff": float(tail_cutoff),
        "tail_probability": gaussian_tail_probability(tail_cutoff),
        "max_pre_renormalization_mass_defect": float(
            max(abs(item["pre_renormalization_mass_defect"]) for item in particle_diagnostics)
        ),
        "max_dropped_sliver_probability": float(
            max(item["dropped_sliver_probability"] for item in particle_diagnostics)
        ),
        "max_clipped_negative_mass": float(
            max(item["clipped_negative_mass"] for item in particle_diagnostics)
        ),
        "particle_diagnostics": particle_diagnostics,
        "renormalization_is_applied": True,
    }
    return result, diagnostics


def _observer_rhat(positions: np.ndarray, observer: Iterable[float], box_size: float) -> np.ndarray:
    observer_array = np.asarray(observer, dtype=np.float64)
    if observer_array.shape != (3,) or not np.all(np.isfinite(observer_array)):
        raise LikelihoodInputError("observer must be a finite vector with shape (3,)")
    relative = (positions - observer_array + box_size / 2.0) % box_size
    relative -= box_size / 2.0
    radius = np.linalg.norm(relative, axis=1)
    if np.any(radius <= 0.0):
        raise LikelihoodInputError("observer-centred line of sight is undefined at observer")
    return relative / radius[:, None]


def predict_selected_intensity_cell_integrated(
    source_positions: Iterable[Iterable[float]],
    source_velocities_km_s: Iterable[Iterable[float]],
    population_masses: Iterable[Iterable[float]],
    selection_exposure: Iterable[Iterable[Iterable[Iterable[float]]]],
    *,
    observer: Iterable[float],
    box_size_cMpc_h: float,
    hubble_km_s_Mpc: float,
    little_h: float,
    scale_factor: float,
    sigma_fog_km_s: Iterable[float],
    sigma_redshift_km_s: Iterable[float],
    velocity_convention: str = VELOCITY_CONVENTION,
    tail_cutoff: float = Q1_DEFAULT_TAIL_CUTOFF,
) -> np.ndarray:
    """Return six selected fields using the Q1 NumPy cell-integrated oracle."""

    exposure = np.asarray(selection_exposure, dtype=np.float64)
    if exposure.ndim != 4 or exposure.shape[0] != POPULATIONS:
        raise LikelihoodInputError("selection_exposure must have shape (6, N, N, N)")
    if len(set(exposure.shape[1:])) != 1 or exposure.shape[1] < 2:
        raise LikelihoodInputError("selection_exposure spatial dimensions must be one cubic grid")
    if not np.all(np.isfinite(exposure)) or np.any(exposure < 0.0):
        raise LikelihoodInputError("selection_exposure must be finite and non-negative")
    positions = np.asarray(source_positions, dtype=np.float64)
    velocities = np.asarray(source_velocities_km_s, dtype=np.float64)
    masses = np.asarray(population_masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise LikelihoodInputError("source_positions must have shape (M, 3)")
    if velocities.shape != positions.shape:
        raise LikelihoodInputError("source_velocities_km_s must match source_positions")
    if masses.shape != (POPULATIONS, positions.shape[0]):
        raise LikelihoodInputError("population_masses must have shape (6, M)")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise LikelihoodInputError("population_masses must be finite and non-negative")
    sigma_fog = np.asarray(sigma_fog_km_s, dtype=np.float64)
    sigma_redshift = np.asarray(sigma_redshift_km_s, dtype=np.float64)
    if sigma_fog.shape != (POPULATIONS,) or sigma_redshift.shape != (POPULATIONS,):
        raise LikelihoodInputError("sigma vectors must have shape (6,)")
    if (
        not np.all(np.isfinite(sigma_fog))
        or not np.all(np.isfinite(sigma_redshift))
        or np.any(sigma_fog < 0.0)
        or np.any(sigma_redshift < 0.0)
    ):
        raise LikelihoodInputError("sigma vectors must be finite and non-negative")
    rsd = observer_centred_spherical_rsd(
        positions,
        velocities,
        observer,
        box_size_cMpc_h,
        hubble_km_s_Mpc,
        little_h=little_h,
        scale_factor=scale_factor,
        velocity_convention=velocity_convention,
    )
    # The stochastic LOS is defined from the coherent-RSD position, before the
    # Gaussian displacement, as frozen in the Q1 plan.
    rhat = _observer_rhat(rsd.positions, observer, box_size_cMpc_h)
    total_sigma = np.hypot(sigma_fog, sigma_redshift)
    displacement_scales = little_h * total_sigma / (scale_factor * hubble_km_s_Mpc)
    raw = np.empty_like(exposure)
    for population in range(POPULATIONS):
        raw[population] = cell_integrated_tsc_deposit(
            rsd.positions,
            masses[population],
            rhat,
            np.full(positions.shape[0], displacement_scales[population], dtype=np.float64),
            exposure.shape[1],
            box_size_cMpc_h,
            tail_cutoff=tail_cutoff,
        )
    intensity = exposure * raw
    if not np.all(np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise LikelihoodInputError("predicted Q1 intensity is not finite and non-negative")
    return intensity


def q1_candidate_oracle_gate(
    candidate: Iterable[Iterable[Iterable[Iterable[float]]]],
    oracle: Iterable[Iterable[Iterable[Iterable[float]]]],
    *,
    relative_l1_tolerance: float = Q1_CANDIDATE_RELATIVE_L1_TOLERANCE,
    absolute_l1_floor: float = Q1_CANDIDATE_ABSOLUTE_L1_FLOOR,
) -> dict[str, object]:
    """Compare a candidate observation operator to the frozen Q1 oracle."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    oracle_array = np.asarray(oracle, dtype=np.float64)
    if candidate_array.shape != oracle_array.shape or candidate_array.ndim != 4:
        raise LikelihoodInputError("candidate and oracle must share shape (6, N, N, N)")
    if candidate_array.shape[0] != POPULATIONS:
        raise LikelihoodInputError("candidate and oracle must have six populations")
    if (
        not np.all(np.isfinite(candidate_array))
        or not np.all(np.isfinite(oracle_array))
        or np.any(candidate_array < 0.0)
        or np.any(oracle_array < 0.0)
    ):
        raise LikelihoodInputError("candidate and oracle must be finite and non-negative")
    if not math.isfinite(relative_l1_tolerance) or relative_l1_tolerance <= 0.0:
        raise LikelihoodInputError("relative_l1_tolerance must be positive and finite")
    if not math.isfinite(absolute_l1_floor) or absolute_l1_floor < 0.0:
        raise LikelihoodInputError("absolute_l1_floor must be finite and non-negative")
    absolute_error = np.sum(np.abs(candidate_array - oracle_array), axis=(1, 2, 3))
    oracle_norm = np.sum(np.abs(oracle_array), axis=(1, 2, 3))
    relative_error = np.divide(
        absolute_error,
        oracle_norm,
        out=np.zeros_like(absolute_error),
        where=oracle_norm > 0.0,
    )
    allowed = absolute_l1_floor + relative_l1_tolerance * oracle_norm
    per_population_pass = absolute_error <= allowed
    total_absolute = float(np.sum(absolute_error))
    total_norm = float(np.sum(oracle_norm))
    total_relative = total_absolute / total_norm if total_norm > 0.0 else total_absolute
    return {
        "status": "PASS" if bool(np.all(per_population_pass)) and total_absolute <= absolute_l1_floor + relative_l1_tolerance * total_norm else "FAIL",
        "relative_l1": total_relative,
        "absolute_l1": total_absolute,
        "per_population_relative_l1": relative_error.tolist(),
        "per_population_absolute_l1": absolute_error.tolist(),
        "relative_l1_tolerance": float(relative_l1_tolerance),
        "absolute_l1_floor": float(absolute_l1_floor),
        "science_claim_authorized": False,
    }


def mass_conservation_gate(
    deposited: Iterable[Iterable[Iterable[Iterable[float]]]],
    population_masses: Iterable[Iterable[float]],
    *,
    relative_tolerance: float = Q1_MASS_RELATIVE_TOLERANCE,
    absolute_tolerance: float = Q1_MASS_ABSOLUTE_TOLERANCE,
) -> dict[str, object]:
    """Check exposure-free per-population and total periodic mass conservation."""

    field = np.asarray(deposited, dtype=np.float64)
    masses = np.asarray(population_masses, dtype=np.float64)
    if field.ndim != 4 or field.shape[0] != POPULATIONS:
        raise LikelihoodInputError("deposited must have shape (6, N, N, N)")
    if masses.ndim != 2 or masses.shape[0] != POPULATIONS:
        raise LikelihoodInputError("population_masses must have shape (6, M)")
    if not np.all(np.isfinite(field)) or np.any(field < 0.0) or not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise LikelihoodInputError("mass inputs must be finite and non-negative")
    expected = np.sum(masses, axis=1)
    actual = np.sum(field, axis=(1, 2, 3))
    absolute_error = np.abs(actual - expected)
    relative_error = np.divide(
        absolute_error,
        expected,
        out=np.zeros_like(absolute_error),
        where=expected > 0.0,
    )
    allowed = absolute_tolerance + relative_tolerance * expected
    per_population_pass = absolute_error <= allowed
    return {
        "status": "PASS" if bool(np.all(per_population_pass)) else "FAIL",
        "expected_per_population": expected.tolist(),
        "actual_per_population": actual.tolist(),
        "absolute_error_per_population": absolute_error.tolist(),
        "relative_error_per_population": relative_error.tolist(),
        "max_absolute_error": float(np.max(absolute_error)),
        "max_relative_error": float(np.max(relative_error)),
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
        "exposure_free": True,
    }


def exposure_weighted_totals(
    deposited: Iterable[Iterable[Iterable[Iterable[float]]]],
    exposure: Iterable[Iterable[Iterable[Iterable[float]]]],
) -> np.ndarray:
    """Return selected totals as a diagnostic; selection does not conserve mass."""

    field = np.asarray(deposited, dtype=np.float64)
    exposure_array = np.asarray(exposure, dtype=np.float64)
    if field.shape != exposure_array.shape or field.ndim != 4 or field.shape[0] != POPULATIONS:
        raise LikelihoodInputError("deposited and exposure must share shape (6, N, N, N)")
    if not np.all(np.isfinite(field)) or not np.all(np.isfinite(exposure_array)) or np.any(field < 0.0) or np.any(exposure_array < 0.0):
        raise LikelihoodInputError("deposited and exposure must be finite and non-negative")
    return np.sum(field * exposure_array, axis=(1, 2, 3))
