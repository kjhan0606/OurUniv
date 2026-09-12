"""Pure in-memory six-tracer CF4/2M++ likelihood primitives.

This module is deliberately independent of GPFS, Slurm, catalog paths, and
publication.  It provides the numerical kernel needed by the KF-DESIGN
contract: a positive selected count intensity with observer-centred spherical
RSD, population-dependent FoG/redshift broadening, and a CF4 shared-redshift
Gaussian factor.  Callers must provide already-calibrated selection exposure
and PMWD source particles; no exposure or count normalization is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.special import gammaln


POPULATIONS = 6
VELOCITY_CONVENTION = "physical_peculiar_km_s_observer_subtracted"
SELECTION_EXPOSURE_UNITS = "dimensionless_expected_detected_count_multiplier_per_cell"
_LOG_MAX_FLOAT64 = math.log(np.finfo(np.float64).max)
_LOG_MIN_FLOAT64 = math.log(np.nextafter(0.0, 1.0))
QUADRATURE_LOW_ORDER = 3
QUADRATURE_HIGH_ORDERS = (7, 9)
QUADRATURE_STRESS_CASE_ID = "RSD_FOG_boundary_v1"
QUADRATURE_RELATIVE_L1_TOLERANCE = 5.0e-3


class LikelihoodInputError(ValueError):
    """A fail-closed likelihood input or contract violation."""


@dataclass(frozen=True)
class RSDResult:
    """Observer-centred coherent RSD positions and radial displacements."""

    positions: np.ndarray
    coherent_displacement_cMpc_h: np.ndarray


def gaussian_hermite_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return standard-normal Gauss--Hermite nodes and normalized weights."""

    if not isinstance(order, int) or order < 3 or order % 2 == 0 or order > 15:
        raise LikelihoodInputError("quadrature order must be an odd integer in [3, 15]")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return np.sqrt(2.0) * nodes, weights / math.sqrt(math.pi)


def quadrature_convergence_gate(
    low_order_intensity: Iterable[Iterable[Iterable[Iterable[float]]]],
    high_order_intensity: Iterable[Iterable[Iterable[Iterable[float]]]],
    *,
    low_order: int,
    high_order: int,
    stress_case_id: str,
    relative_l1_tolerance: float,
) -> dict[str, float | bool | str]:
    """Compare two GH intensity evaluations without claiming science convergence.

    The caller supplies the same boundary-stress mock inputs evaluated at two
    declared quadrature orders (normally GH3 and GH7/GH9).  A zero high-order
    field is handled explicitly.  The result is a numerical gate only; it does
    not authorize inference, KF-EXPAND, or an observational resolution claim.
    """

    if low_order != QUADRATURE_LOW_ORDER or high_order not in QUADRATURE_HIGH_ORDERS:
        raise LikelihoodInputError("quadrature comparison must be GH3 versus GH7 or GH9")
    if stress_case_id != QUADRATURE_STRESS_CASE_ID:
        raise LikelihoodInputError("quadrature stress_case_id is not preregistered")
    if relative_l1_tolerance != QUADRATURE_RELATIVE_L1_TOLERANCE:
        raise LikelihoodInputError("quadrature tolerance is not the preregistered value")
    low = np.asarray(low_order_intensity, dtype=np.float64)
    high = np.asarray(high_order_intensity, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 4 or low.shape[0] != POPULATIONS:
        raise LikelihoodInputError("quadrature intensities must share shape (6, N, N, N)")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise LikelihoodInputError("quadrature intensities must be finite")
    if np.any(low < 0.0) or np.any(high < 0.0):
        raise LikelihoodInputError("quadrature intensities must be non-negative")
    if not math.isfinite(relative_l1_tolerance) or relative_l1_tolerance <= 0.0:
        raise LikelihoodInputError("relative_l1_tolerance must be positive and finite")
    numerator = float(np.sum(np.abs(low - high)))
    denominator = float(np.sum(np.abs(high)))
    relative_l1 = numerator / denominator if denominator > 0.0 else numerator
    return {
        "status": "PASS" if relative_l1 <= relative_l1_tolerance else "FAIL",
        "relative_l1": relative_l1,
        "low_order": low_order,
        "high_order": high_order,
        "stress_case_id": stress_case_id,
        "relative_l1_tolerance": float(relative_l1_tolerance),
        "science_claim_authorized": False,
    }


def _array(name: str, value: Iterable[float], shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise LikelihoodInputError(f"{name} shape {array.shape} != {shape}")
    if array.dtype != dtype:
        raise LikelihoodInputError(f"{name} dtype {array.dtype} != {dtype}")
    if not np.all(np.isfinite(array)):
        raise LikelihoodInputError(f"{name} contains non-finite values")
    return array


def _positive_vector(name: str, value: Iterable[float], size: int, *, strict: bool = False) -> np.ndarray:
    array = _array(name, value, (size,), np.dtype(np.float64))
    if np.any(array < 0.0) or (strict and np.any(array <= 0.0)):
        relation = "> 0" if strict else ">= 0"
        raise LikelihoodInputError(f"{name} must be {relation}")
    return array


def population_log_masses_from_eta(
    eta_at_sources: Iterable[float],
    alpha: Iterable[float],
    log_bias: Iterable[float],
) -> np.ndarray:
    """Return stable log source masses without count renormalization."""

    eta = np.asarray(eta_at_sources, dtype=np.float64)
    if eta.ndim != 1 or eta.size == 0 or not np.all(np.isfinite(eta)):
        raise LikelihoodInputError("eta_at_sources must be a finite non-empty float64 vector")
    alpha_array = _array("alpha", alpha, (POPULATIONS,), np.dtype(np.float64))
    log_bias_array = _array("log_bias", log_bias, (POPULATIONS,), np.dtype(np.float64))
    if not np.all(np.isfinite(alpha_array)) or not np.all(np.isfinite(log_bias_array)):
        raise LikelihoodInputError("population nuisance parameters must be finite")
    bias = np.exp(log_bias_array)
    if not np.all(np.isfinite(bias)) or np.any(bias <= 0.0):
        raise LikelihoodInputError("population bias must be finite and positive")
    log_masses = alpha_array[:, None] + bias[:, None] * eta[None, :]
    if not np.all(np.isfinite(log_masses)):
        raise LikelihoodInputError("population log source masses must be finite")
    if np.any(log_masses > _LOG_MAX_FLOAT64) or np.any(log_masses < _LOG_MIN_FLOAT64):
        raise LikelihoodInputError("population source-mass exponent is outside float64 range")
    return log_masses


def population_masses_from_eta(
    eta_at_sources: Iterable[float],
    alpha: Iterable[float],
    log_bias: Iterable[float],
) -> np.ndarray:
    """Return positive six-population source masses without count renormalization."""

    return np.exp(population_log_masses_from_eta(eta_at_sources, alpha, log_bias))


def observer_centred_spherical_rsd(
    positions: Iterable[Iterable[float]],
    velocities_km_s: Iterable[Iterable[float]],
    observer: Iterable[float],
    box_size_cMpc_h: float,
    hubble_km_s_Mpc: float,
    *,
    little_h: float,
    scale_factor: float,
    velocity_convention: str = VELOCITY_CONVENTION,
) -> RSDResult:
    """Apply coherent radial displacement using the local spherical line of sight.

    Positions are cMpc/h and velocities are physical peculiar km/s.  Therefore
    the comoving displacement in cMpc/h is
    ``little_h * v_r / (a * H[a])``; the explicit factors prevent an accidental
    Mpc-versus-Mpc/h mix-up.
    """

    positions_array = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities_km_s, dtype=np.float64)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3 or positions_array.shape[0] == 0:
        raise LikelihoodInputError("positions must have shape (M, 3)")
    if velocities.shape != positions_array.shape:
        raise LikelihoodInputError("velocities must have the same shape as positions")
    if not np.all(np.isfinite(positions_array)) or not np.all(np.isfinite(velocities)):
        raise LikelihoodInputError("positions and velocities must be finite")
    observer_array = _array("observer", observer, (3,), np.dtype(np.float64))
    if not math.isfinite(box_size_cMpc_h) or box_size_cMpc_h <= 0.0:
        raise LikelihoodInputError("box_size_cMpc_h must be positive and finite")
    if not math.isfinite(hubble_km_s_Mpc) or hubble_km_s_Mpc <= 0.0:
        raise LikelihoodInputError("hubble_km_s_Mpc must be positive and finite")
    if not math.isfinite(little_h) or not 0.0 < little_h < 2.0:
        raise LikelihoodInputError("little_h must be finite and lie in (0, 2)")
    if not math.isfinite(scale_factor) or not 0.0 < scale_factor <= 1.0:
        raise LikelihoodInputError("scale_factor must be finite and lie in (0, 1]")
    if velocity_convention != VELOCITY_CONVENTION:
        raise LikelihoodInputError("velocity_convention must be observer-subtracted physical peculiar km/s")
    if np.any(positions_array < 0.0) or np.any(positions_array >= box_size_cMpc_h):
        raise LikelihoodInputError("positions must lie in [0, box_size)")
    relative = (positions_array - observer_array + box_size_cMpc_h / 2.0) % box_size_cMpc_h
    relative -= box_size_cMpc_h / 2.0
    radius = np.linalg.norm(relative, axis=1)
    if np.any(radius <= 0.0):
        raise LikelihoodInputError("observer-centred line of sight is undefined at the observer")
    rhat = relative / radius[:, None]
    radial_velocity = np.sum(velocities * rhat, axis=1)
    displacement = little_h * radial_velocity / (scale_factor * hubble_km_s_Mpc)
    shifted = (positions_array + displacement[:, None] * rhat) % box_size_cMpc_h
    return RSDResult(shifted, displacement)


def tsc_deposit(
    positions: Iterable[Iterable[float]],
    masses: Iterable[float],
    grid_size: int,
    box_size_cMpc_h: float,
) -> np.ndarray:
    """Deposit particle masses with periodic, conservative TSC weights."""

    positions_array = np.asarray(positions, dtype=np.float64)
    masses_array = np.asarray(masses, dtype=np.float64)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3 or positions_array.shape[0] == 0:
        raise LikelihoodInputError("positions must have shape (M, 3)")
    if masses_array.shape != (positions_array.shape[0],):
        raise LikelihoodInputError("masses must have shape (M,)")
    if not np.all(np.isfinite(positions_array)) or not np.all(np.isfinite(masses_array)):
        raise LikelihoodInputError("positions and masses must be finite")
    if np.any(masses_array < 0.0):
        raise LikelihoodInputError("masses must be non-negative")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise LikelihoodInputError("grid_size must be an integer >= 2")
    if not math.isfinite(box_size_cMpc_h) or box_size_cMpc_h <= 0.0:
        raise LikelihoodInputError("box_size_cMpc_h must be positive and finite")
    spacing = box_size_cMpc_h / grid_size
    wrapped = positions_array % box_size_cMpc_h
    cell = wrapped / spacing - 0.5
    nearest = np.floor(cell + 0.5).astype(np.int64)
    offset = cell - nearest

    def weights(component: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            0.5 * (0.5 - component) ** 2,
            0.75 - component**2,
            0.5 * (0.5 + component) ** 2,
        )

    wx, wy, wz = (weights(offset[:, axis]) for axis in range(3))
    result = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    for ix, dx in enumerate((-1, 0, 1)):
        for iy, dy in enumerate((-1, 0, 1)):
            for iz, dz in enumerate((-1, 0, 1)):
                weight = wx[ix] * wy[iy] * wz[iz]
                np.add.at(
                    result,
                    ((nearest[:, 0] + dx) % grid_size,
                     (nearest[:, 1] + dy) % grid_size,
                     (nearest[:, 2] + dz) % grid_size),
                    masses_array * weight,
                )
    return result


def predict_selected_intensity(
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
    quadrature_order: int = 3,
) -> np.ndarray:
    """Return the six positive selected count-intensity grids.

    The three-node Gaussian-Hermite rule approximates the population-specific
    line-of-sight FoG plus redshift-error convolution.  The selection exposure
    is applied after RSD and is never rescaled to observed counts.
    """

    exposure = np.asarray(selection_exposure, dtype=np.float64)
    if exposure.ndim != 4 or exposure.shape[0] != POPULATIONS:
        raise LikelihoodInputError("selection_exposure must have shape (6, N, N, N)")
    if len(set(exposure.shape[1:])) != 1 or exposure.shape[1] < 2:
        raise LikelihoodInputError("selection_exposure spatial dimensions must be one cubic grid")
    if not np.all(np.isfinite(exposure)) or np.any(exposure < 0.0):
        raise LikelihoodInputError("selection_exposure must be finite and non-negative")
    masses = np.asarray(population_masses, dtype=np.float64)
    positions = np.asarray(source_positions, dtype=np.float64)
    velocities = np.asarray(source_velocities_km_s, dtype=np.float64)
    if masses.shape != (POPULATIONS, positions.shape[0]):
        raise LikelihoodInputError("population_masses must have shape (6, M)")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise LikelihoodInputError("population_masses must be finite and non-negative")
    sigma_fog = _positive_vector("sigma_fog_km_s", sigma_fog_km_s, POPULATIONS)
    sigma_redshift = _positive_vector("sigma_redshift_km_s", sigma_redshift_km_s, POPULATIONS)
    rsd = observer_centred_spherical_rsd(
        positions, velocities, observer, box_size_cMpc_h, hubble_km_s_Mpc,
        little_h=little_h, scale_factor=scale_factor,
        velocity_convention=velocity_convention,
    )
    observer_array = np.asarray(observer, dtype=np.float64)
    relative = (positions - observer_array + box_size_cMpc_h / 2.0) % box_size_cMpc_h
    relative -= box_size_cMpc_h / 2.0
    rhat = relative / np.linalg.norm(relative, axis=1)[:, None]
    grid_size = exposure.shape[1]
    intensity = np.empty_like(exposure)
    total_sigma = np.hypot(sigma_fog, sigma_redshift)
    quadrature_nodes, quadrature_weights = gaussian_hermite_rule(quadrature_order)
    for population in range(POPULATIONS):
        deposited = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
        for node, weight in zip(quadrature_nodes, quadrature_weights):
            extra = node * little_h * total_sigma[population] / (scale_factor * hubble_km_s_Mpc)
            positions_node = (rsd.positions + extra * rhat) % box_size_cMpc_h
            deposited += weight * tsc_deposit(
                positions_node, masses[population], grid_size, box_size_cMpc_h
            )
        intensity[population] = exposure[population] * deposited
    if not np.all(np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise LikelihoodInputError("predicted intensity is not finite and non-negative")
    return intensity


def poisson_log_likelihood(counts: Iterable[Iterable[Iterable[Iterable[int]]]], intensity: Iterable[Iterable[Iterable[Iterable[float]]]]) -> float:
    """Evaluate the exact six-population Poisson log likelihood."""

    observed = np.asarray(counts)
    expected = np.asarray(intensity, dtype=np.float64)
    if observed.ndim != 4 or observed.shape[0] != POPULATIONS:
        raise LikelihoodInputError("counts must have shape (6, N, N, N)")
    if observed.dtype != np.dtype(np.int64):
        raise LikelihoodInputError("counts must have exact int64 dtype")
    if expected.shape != observed.shape:
        raise LikelihoodInputError("intensity shape must match counts")
    if np.any(observed < 0) or not np.all(np.isfinite(expected)) or np.any(expected < 0.0):
        raise LikelihoodInputError("counts/intensity violate non-negative finite contract")
    positive = observed > 0
    if np.any(positive & (expected <= 0.0)):
        raise LikelihoodInputError("positive observed count has non-positive intensity")
    log_expected = np.full(expected.shape, -np.inf, dtype=np.float64)
    positive_expected = expected > 0.0
    log_expected[positive_expected] = np.log(expected[positive_expected])
    return poisson_log_likelihood_from_log_intensity(observed, log_expected)


def poisson_log_likelihood_from_log_intensity(
    counts: Iterable[Iterable[Iterable[Iterable[int]]]],
    log_intensity: Iterable[Iterable[Iterable[Iterable[float]]]],
) -> float:
    """Evaluate Poisson likelihood in log space, allowing zero intensity only for zero counts."""

    observed = np.asarray(counts)
    log_expected = np.asarray(log_intensity, dtype=np.float64)
    if observed.ndim != 4 or observed.shape[0] != POPULATIONS:
        raise LikelihoodInputError("counts must have shape (6, N, N, N)")
    if observed.dtype != np.dtype(np.int64):
        raise LikelihoodInputError("counts must have exact int64 dtype")
    if log_expected.shape != observed.shape:
        raise LikelihoodInputError("log_intensity shape must match counts")
    if np.any(observed < 0) or np.any(np.isnan(log_expected)) or np.any(np.isposinf(log_expected)):
        raise LikelihoodInputError("counts/log_intensity violate finite support contract")
    positive = observed > 0
    if np.any(positive & np.isneginf(log_expected)):
        raise LikelihoodInputError("positive observed count has zero log-intensity support")
    expected = np.where(np.isneginf(log_expected), 0.0, np.exp(log_expected))
    count_term = np.zeros(log_expected.shape, dtype=np.float64)
    count_term[positive] = observed[positive] * log_expected[positive]
    value = count_term - expected - gammaln(observed + 1.0)
    return float(np.sum(value))


def _identity_tuple(name: str, values: Iterable[object]) -> tuple[str, ...]:
    try:
        result = tuple(str(value).strip() for value in values)
    except TypeError as exc:
        raise LikelihoodInputError(f"{name} must be an iterable of object identities") from exc
    if not result or any(not value for value in result):
        raise LikelihoodInputError(f"{name} must contain non-empty identities")
    if len(set(result)) != len(result):
        raise LikelihoodInputError(f"{name} must contain unique identities")
    return result


def validate_factor_ownership(
    secure_object_ids: Iterable[object],
    group_ids: Iterable[int],
    *,
    independent_twompp_redshift_ids: Iterable[object] = (),
) -> dict[str, object]:
    """Bind each secure object to exactly one redshift factor owner.

    The count factor owns the 2M++ grid, while the CF4 group-mark factor owns
    every secure object's redshift datum through one shared group latent.  An
    independent 2M++ redshift term is forbidden here, rather than relying on a
    caller to remember the no-double-counting rule.
    """

    objects = _identity_tuple("secure_object_ids", secure_object_ids)
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.size != len(objects) or groups.dtype.kind not in "iu":
        raise LikelihoodInputError("group_ids must be an integer vector aligned with secure_object_ids")
    independent = tuple(str(value).strip() for value in independent_twompp_redshift_ids)
    if independent:
        if any(not value for value in independent):
            raise LikelihoodInputError("independent 2M++ redshift identities must be non-empty")
        overlap = set(objects).intersection(independent)
        if overlap:
            raise LikelihoodInputError("secure objects cannot have an independent 2M++ redshift factor")
        raise LikelihoodInputError("independent 2M++ redshift factor is not part of this joint kernel")
    return {
        "count_factor_owner": "2Mpp_grid_counts",
        "redshift_factor_owner": "CF4_group_marks_shared_redshift",
        "secure_object_ids": objects,
        "independent_twompp_redshift_factor": False,
    }


def shared_redshift_log_likelihood(
    observed_km_s: Iterable[float],
    predicted_km_s: Iterable[float],
    measurement_sigma_km_s: Iterable[float],
    group_ids: Iterable[int],
    shared_sigma_km_s: Iterable[float],
    *,
    secure_object_ids: Iterable[object],
    independent_twompp_redshift_ids: Iterable[object] = (),
    expected_group_count: int | None = None,
) -> float:
    """Marginalize one shared redshift latent per CF4 group.

    For group ``g``, the covariance is ``diag(sigma_i^2) + tau_g^2 11^T``.
    This emits one joint factor; callers must not add independent Vcmb terms
    for the same securely crossmatched objects.
    """

    observed = np.asarray(observed_km_s, dtype=np.float64)
    predicted = np.asarray(predicted_km_s, dtype=np.float64)
    sigma = np.asarray(measurement_sigma_km_s, dtype=np.float64)
    groups = np.asarray(group_ids)
    tau = np.asarray(shared_sigma_km_s, dtype=np.float64)
    if observed.ndim != 1 or observed.size == 0:
        raise LikelihoodInputError("observed_km_s must be a non-empty vector")
    if predicted.shape != observed.shape or sigma.shape != observed.shape or groups.shape != observed.shape:
        raise LikelihoodInputError("observed, predicted, sigma, and group_ids must have one common shape")
    if groups.dtype.kind not in "iu" or tau.ndim != 1:
        raise LikelihoodInputError("group_ids must be integer and shared_sigma must be one-dimensional")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(tau)):
        raise LikelihoodInputError("redshift likelihood inputs must be finite")
    if np.any(sigma <= 0.0) or np.any(tau < 0.0):
        raise LikelihoodInputError("measurement sigma must be positive and shared sigma non-negative")
    if np.any(groups < 0) or np.any(groups >= tau.size):
        raise LikelihoodInputError("group_ids index shared_sigma exactly")
    if expected_group_count is not None:
        if not isinstance(expected_group_count, int) or expected_group_count <= 0:
            raise LikelihoodInputError("expected_group_count must be a positive integer")
        if tau.size != expected_group_count:
            raise LikelihoodInputError("shared_sigma size must equal the manifest group count")
        used_groups = np.unique(groups)
        if not np.array_equal(used_groups, np.arange(expected_group_count, dtype=groups.dtype)):
            raise LikelihoodInputError("all manifest group indices must be used exactly")
    validate_factor_ownership(
        secure_object_ids,
        groups,
        independent_twompp_redshift_ids=independent_twompp_redshift_ids,
    )
    residual = observed - predicted
    result = 0.0
    for group in np.unique(groups):
        mask = groups == group
        variance = sigma[mask] ** 2
        shared_variance = tau[int(group)] ** 2
        inv_diag = 1.0 / variance
        contraction = float(np.sum(inv_diag))
        weighted = float(np.sum(residual[mask] * inv_diag))
        quadratic = float(np.sum(residual[mask] ** 2 * inv_diag))
        denominator = 1.0 + shared_variance * contraction
        quadratic -= shared_variance * weighted**2 / denominator
        logdet = float(np.sum(np.log(variance)) + np.log(denominator))
        result += -0.5 * (quadratic + logdet + mask.sum() * math.log(2.0 * math.pi))
    return float(result)


def joint_log_likelihood(
    counts: Iterable[Iterable[Iterable[Iterable[int]]]],
    intensity: Iterable[Iterable[Iterable[Iterable[float]]]],
    observed_km_s: Iterable[float],
    predicted_km_s: Iterable[float],
    measurement_sigma_km_s: Iterable[float],
    group_ids: Iterable[int],
    shared_sigma_km_s: Iterable[float],
    *,
    secure_object_ids: Iterable[object],
    independent_twompp_redshift_ids: Iterable[object] = (),
    expected_group_count: int | None = None,
) -> float:
    """Combine count and one shared-redshift factor without double counting."""

    return poisson_log_likelihood(counts, intensity) + shared_redshift_log_likelihood(
        observed_km_s,
        predicted_km_s,
        measurement_sigma_km_s,
        group_ids,
        shared_sigma_km_s,
        secure_object_ids=secure_object_ids,
        independent_twompp_redshift_ids=independent_twompp_redshift_ids,
        expected_group_count=expected_group_count,
    )
