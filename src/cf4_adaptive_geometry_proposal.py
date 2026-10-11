#!/usr/bin/env python3
"""Frozen defensive midpoint/axis proposal for CF4 peak-evidence integration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from cf4_lg_midpoint_proposal import (
    diagonal_normal_logpdf,
    draw_diagonal_normal_mixture,
    mixture_logpdf,
)


LOG_UNIFORM_S2 = -math.log(4.0 * math.pi)


def logsumexp_axis(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)),
        axis=axis,
    )


def normalized_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("log weights must be a finite nonempty vector")
    shifted = np.exp(values - np.max(values))
    return shifted / np.sum(shifted)


def canonical_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("axis must be a finite nonzero three-vector")
    result = axis / norm
    index = int(np.argmax(np.abs(result)))
    if result[index] < 0.0:
        result = -result
    return result


def normalize_axes(axes: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=np.float64)
    if axes.ndim == 1:
        axes = axes[None, :]
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError("axes must have shape (n,3)")
    norm = np.linalg.norm(axes, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm == 0.0):
        raise ValueError("axes must be finite and nonzero")
    return axes / norm[:, None]


def _log_cosh(value: np.ndarray) -> np.ndarray:
    absolute = np.abs(np.asarray(value, dtype=np.float64))
    return absolute + np.log1p(np.exp(-2.0 * absolute)) - math.log(2.0)


def _log_sinh_positive(value: float) -> float:
    if not value > 0.0:
        raise ValueError("log sinh requires a positive value")
    if value < 1.0:
        return math.log(math.sinh(value))
    return value + math.log1p(-math.exp(-2.0 * value)) - math.log(2.0)


def antipodal_vmf_logpdf(
    axes: np.ndarray,
    direction: np.ndarray,
    kappa: float,
) -> np.ndarray:
    axes = normalize_axes(axes)
    direction = canonical_axis(direction)
    kappa = float(kappa)
    if not np.isfinite(kappa) or not 0.0 <= kappa <= 20.0:
        raise ValueError("kappa must lie in [0,20]")
    if kappa == 0.0:
        return np.full(len(axes), LOG_UNIFORM_S2)
    dot = axes @ direction
    normalization = (
        math.log(kappa) - math.log(4.0 * math.pi) - _log_sinh_positive(kappa)
    )
    return normalization + _log_cosh(kappa * dot)


def _orthogonal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction, dtype=np.float64)
    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(direction)))] = 1.0
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def sample_antipodal_vmf(
    rng: np.random.Generator,
    direction: np.ndarray,
    kappa: float,
    count: int,
) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be nonnegative")
    direction = canonical_axis(direction)
    kappa = float(kappa)
    if not 0.0 <= kappa <= 20.0:
        raise ValueError("kappa must lie in [0,20]")
    signs = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    centres = signs[:, None] * direction[None, :]
    uniform = rng.random(count)
    if kappa == 0.0:
        cosine = 2.0 * uniform - 1.0
    else:
        cosine = 1.0 + np.log(
            uniform + (1.0 - uniform) * math.exp(-2.0 * kappa)
        ) / kappa
    azimuth = 2.0 * math.pi * rng.random(count)
    first, second = _orthogonal_basis(direction)
    signed_first = signs[:, None] * first[None, :]
    signed_second = second[None, :]
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    result = (
        cosine[:, None] * centres
        + sine[:, None] * (
            np.cos(azimuth)[:, None] * signed_first
            + np.sin(azimuth)[:, None] * signed_second
        )
    )
    return result / np.linalg.norm(result, axis=1)[:, None]


def sample_isotropic_axes(rng: np.random.Generator, count: int) -> np.ndarray:
    cosine = 2.0 * rng.random(count) - 1.0
    azimuth = 2.0 * math.pi * rng.random(count)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    return np.column_stack((
        sine * np.cos(azimuth),
        sine * np.sin(azimuth),
        cosine,
    ))


def _validate_parameters(parameters: dict[str, Any]) -> dict[str, np.ndarray]:
    alpha = np.asarray(parameters["alpha"], dtype=np.float64)
    mean = np.asarray(parameters["mean_mpc_h"], dtype=np.float64)
    covariance = np.asarray(parameters["covariance_mpc_h_squared"], dtype=np.float64)
    direction = np.asarray(parameters["axis_direction"], dtype=np.float64)
    kappa = np.asarray(parameters["axis_kappa"], dtype=np.float64)
    component_count = len(alpha)
    if component_count != 4:
        raise ValueError("the frozen proposal requires exactly four components")
    if mean.shape != (component_count, 3):
        raise ValueError("mean shape mismatch")
    if covariance.shape != (component_count, 3, 3):
        raise ValueError("covariance shape mismatch")
    if direction.shape != (component_count, 3) or kappa.shape != (component_count,):
        raise ValueError("axis parameter shape mismatch")
    if np.any(alpha <= 0.0) or not np.isclose(alpha.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("component weights must be positive and normalized")
    if np.any(kappa < 0.0) or np.any(kappa > 20.0):
        raise ValueError("axis concentration is outside the frozen range")
    canonical = np.vstack([canonical_axis(value) for value in direction])
    clipped_covariance = []
    for matrix in covariance:
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.min(eigenvalues) < 0.75**2 - 1e-12 \
                or np.max(eigenvalues) > 6.0**2 + 1e-12:
            raise ValueError("covariance eigenvalue is outside the frozen range")
        clipped_covariance.append(matrix)
    return {
        "alpha": alpha,
        "mean": mean,
        "covariance": np.asarray(clipped_covariance),
        "direction": canonical,
        "kappa": kappa,
    }


def gaussian_component_logpdf(
    midpoint: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    midpoint = np.asarray(midpoint, dtype=np.float64)
    if midpoint.ndim == 1:
        midpoint = midpoint[None, :]
    factor = np.linalg.cholesky(covariance)
    residual = midpoint - mean
    whitened = np.linalg.solve(factor, residual.T).T
    logdet = 2.0 * np.sum(np.log(np.diag(factor)))
    return -0.5 * (
        3.0 * math.log(2.0 * math.pi)
        + logdet
        + np.einsum("ni,ni->n", whitened, whitened)
    )


def adaptive_component_logpdf(
    midpoint: np.ndarray,
    axes: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    midpoint = np.asarray(midpoint, dtype=np.float64)
    if midpoint.ndim == 1:
        midpoint = midpoint[None, :]
    axes = normalize_axes(axes)
    if len(midpoint) != len(axes):
        raise ValueError("midpoint and axis row counts differ")
    values = _validate_parameters(parameters)
    terms = []
    for index in range(4):
        terms.append(
            math.log(values["alpha"][index])
            + gaussian_component_logpdf(
                midpoint, values["mean"][index], values["covariance"][index]
            )
            + antipodal_vmf_logpdf(
                axes, values["direction"][index], values["kappa"][index]
            )
        )
    return logsumexp_axis(np.stack(terms, axis=1), axis=1)


def midpoint_prior_logpdf(midpoint: np.ndarray, prior: dict[str, Any]) -> np.ndarray:
    midpoint = np.asarray(midpoint, dtype=np.float64)
    if midpoint.ndim == 1:
        midpoint = midpoint[None, :]
    return np.asarray([diagonal_normal_logpdf(row, prior) for row in midpoint])


def target_geometry_logpdf(midpoint: np.ndarray, prior: dict[str, Any]) -> np.ndarray:
    return midpoint_prior_logpdf(midpoint, prior) + LOG_UNIFORM_S2


def defensive_proposal_logpdf(
    midpoint: np.ndarray,
    axes: np.ndarray,
    prior: dict[str, Any],
    parameters: dict[str, Any],
) -> np.ndarray:
    base = target_geometry_logpdf(midpoint, prior)
    adaptive = adaptive_component_logpdf(midpoint, axes, parameters)
    return logsumexp_axis(np.column_stack((
        math.log(0.5) + base,
        math.log(0.5) + adaptive,
    )), axis=1)


def draw_adaptation_geometry(
    peak: dict[str, Any],
    master_seed: int,
    draw_index: int,
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64DXSM(
        np.random.SeedSequence(int(master_seed), spawn_key=(int(draw_index),))
    ))
    prior = peak["protohalo_midpoint_prior"]
    proposal = peak["protohalo_midpoint_sampling_proposal"]
    midpoint, component = draw_diagonal_normal_mixture(rng, proposal["components"])
    axis = sample_isotropic_axes(rng, 1)[0]
    log_target = diagonal_normal_logpdf(midpoint, prior) + LOG_UNIFORM_S2
    log_proposal = mixture_logpdf(midpoint, proposal["components"]) + LOG_UNIFORM_S2
    return {
        "midpoint_offset_mpc_h": midpoint,
        "axis": axis,
        "proposal_branch": 0,
        "proposal_component": int(component),
        "log_target_geometry_density": log_target,
        "log_sampling_geometry_density": log_proposal,
        "log_target_over_proposal": log_target - log_proposal,
    }


def draw_defensive_geometry(
    prior: dict[str, Any],
    parameters: dict[str, Any],
    master_seed: int,
    draw_index: int,
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64DXSM(
        np.random.SeedSequence(int(master_seed), spawn_key=(int(draw_index),))
    ))
    values = _validate_parameters(parameters)
    if rng.random() < 0.5:
        mean = np.asarray(prior["mean_mpc_h"], dtype=np.float64)
        sigma = np.broadcast_to(np.asarray(prior["sigma_mpc_h"], dtype=np.float64), (3,))
        midpoint = rng.normal(mean, sigma)
        axis = sample_isotropic_axes(rng, 1)[0]
        branch = 0
        component = -1
    else:
        component = int(rng.choice(4, p=values["alpha"]))
        midpoint = rng.multivariate_normal(
            values["mean"][component], values["covariance"][component]
        )
        axis = sample_antipodal_vmf(
            rng, values["direction"][component], values["kappa"][component], 1
        )[0]
        branch = 1
    log_target = float(target_geometry_logpdf(midpoint, prior)[0])
    log_proposal = float(defensive_proposal_logpdf(
        midpoint, axis, prior, parameters
    )[0])
    return {
        "midpoint_offset_mpc_h": midpoint,
        "axis": axis,
        "proposal_branch": branch,
        "proposal_component": component,
        "log_target_geometry_density": log_target,
        "log_sampling_geometry_density": log_proposal,
        "log_target_over_proposal": log_target - log_proposal,
    }


def sample_adaptive_component(
    rng: np.random.Generator,
    parameters: dict[str, Any],
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _validate_parameters(parameters)
    component = rng.choice(4, size=count, p=values["alpha"]).astype(np.int8)
    midpoint = np.empty((count, 3), dtype=np.float64)
    axes = np.empty((count, 3), dtype=np.float64)
    for index in range(4):
        selected = np.flatnonzero(component == index)
        midpoint[selected] = rng.multivariate_normal(
            values["mean"][index], values["covariance"][index], len(selected)
        )
        axes[selected] = sample_antipodal_vmf(
            rng,
            values["direction"][index],
            values["kappa"][index],
            len(selected),
        )
    return midpoint, axes, component


def sample_defensive_proposal(
    rng: np.random.Generator,
    prior: dict[str, Any],
    parameters: dict[str, Any],
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    branch = (rng.random(count) >= 0.5).astype(np.int8)
    component = np.full(count, -1, dtype=np.int8)
    midpoint = np.empty((count, 3), dtype=np.float64)
    axes = np.empty((count, 3), dtype=np.float64)
    base = np.flatnonzero(branch == 0)
    mean = np.asarray(prior["mean_mpc_h"], dtype=np.float64)
    sigma = np.broadcast_to(
        np.asarray(prior["sigma_mpc_h"], dtype=np.float64), (3,)
    )
    midpoint[base] = rng.normal(mean, sigma, size=(len(base), 3))
    axes[base] = sample_isotropic_axes(rng, len(base))
    adaptive = np.flatnonzero(branch == 1)
    q_adaptive, a_adaptive, k_adaptive = sample_adaptive_component(
        rng, parameters, len(adaptive)
    )
    midpoint[adaptive] = q_adaptive
    axes[adaptive] = a_adaptive
    component[adaptive] = k_adaptive
    return midpoint, axes, branch, component


def _clip_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.75**2, 6.0**2)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _axis_mean_resultant(kappa: float) -> float:
    if kappa < 1e-5:
        return kappa / 3.0 - kappa**3 / 45.0 + 2.0 * kappa**5 / 945.0
    return 1.0 / math.tanh(kappa) - 1.0 / kappa


def solve_kappa(resultant: float) -> float:
    resultant = float(resultant)
    if not np.isfinite(resultant) or resultant < 0.0:
        raise ValueError("resultant must be finite and nonnegative")
    if resultant == 0.0:
        return 0.0
    if resultant >= _axis_mean_resultant(20.0):
        return 20.0
    lower, upper = 0.0, 20.0
    while upper - lower > 1e-12:
        midpoint = 0.5 * (lower + upper)
        if _axis_mean_resultant(midpoint) < resultant:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _component_log_terms(
    midpoint: np.ndarray,
    axes: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    values = _validate_parameters(parameters)
    terms = []
    for index in range(4):
        terms.append(
            math.log(values["alpha"][index])
            + gaussian_component_logpdf(
                midpoint, values["mean"][index], values["covariance"][index]
            )
            + antipodal_vmf_logpdf(
                axes, values["direction"][index], values["kappa"][index]
            )
        )
    return np.stack(terms, axis=1)


def _objective_and_responsibility(
    midpoint: np.ndarray,
    axes: np.ndarray,
    weights: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[float, np.ndarray]:
    terms = _component_log_terms(midpoint, axes, parameters)
    total = logsumexp_axis(terms, axis=1)
    responsibility = np.exp(terms - total[:, None])
    return float(weights @ total), responsibility


def _weighted_kmeans_initialization(
    midpoint: np.ndarray,
    axes: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected: list[int] = []
    selected.append(int(rng.choice(len(weights), p=weights)))
    nearest = np.full(len(weights), np.inf)
    for _ in range(1, 4):
        centre = selected[-1]
        distance = np.sum(((midpoint - midpoint[centre]) / 3.0) ** 2, axis=1)
        distance += 2.0 * np.maximum(
            0.0, 1.0 - (axes @ axes[centre]) ** 2
        )
        nearest = np.minimum(nearest, distance)
        probability = weights * nearest
        total = float(np.sum(probability))
        if not total > 0.0:
            raise RuntimeError("weighted k-means++ exhausted distinct centres")
        selected.append(int(rng.choice(len(weights), p=probability / total)))
    distance = []
    for centre in selected:
        value = np.sum(((midpoint - midpoint[centre]) / 3.0) ** 2, axis=1)
        value += 2.0 * np.maximum(
            0.0, 1.0 - (axes @ axes[centre]) ** 2
        )
        distance.append(value)
    assignment = np.argmin(np.stack(distance, axis=1), axis=1)
    alpha, mean, covariance, direction = [], [], [], []
    for component in range(4):
        membership = weights * (assignment == component)
        mass = float(np.sum(membership))
        if mass < 1e-4:
            raise RuntimeError("initial component has insufficient mass")
        effective = mass**2 / float(np.sum(membership**2))
        if effective < 4.0:
            raise RuntimeError("initial component has insufficient effective membership")
        component_mean = np.sum(membership[:, None] * midpoint, axis=0) / mass
        residual = midpoint - component_mean
        component_covariance = np.einsum(
            "n,ni,nj->ij", membership, residual, residual
        ) / mass
        dyad = np.einsum("n,ni,nj->ij", membership, axes, axes)
        eigenvalues, eigenvectors = np.linalg.eigh(dyad)
        alpha.append(mass)
        mean.append(component_mean)
        covariance.append(_clip_covariance(component_covariance))
        direction.append(canonical_axis(eigenvectors[:, int(np.argmax(eigenvalues))]))
    alpha = np.asarray(alpha) / np.sum(alpha)
    return {
        "alpha": alpha,
        "mean_mpc_h": np.asarray(mean),
        "covariance_mpc_h_squared": np.asarray(covariance),
        "axis_direction": np.asarray(direction),
        "axis_kappa": np.ones(4),
    }


def _m_step(
    midpoint: np.ndarray,
    axes: np.ndarray,
    weights: np.ndarray,
    responsibility: np.ndarray,
    previous: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    old = _validate_parameters(previous)
    alpha, mean, covariance, direction, kappa, effective = [], [], [], [], [], []
    for component in range(4):
        membership = weights * responsibility[:, component]
        mass = float(np.sum(membership))
        denominator = float(np.sum(membership**2))
        if not np.isfinite(denominator) or denominator == 0.0:
            raise RuntimeError("component effective-membership denominator is invalid")
        component_effective = mass**2 / denominator
        if mass < 1e-4 or component_effective < 4.0:
            raise RuntimeError("component failed the frozen membership gate")
        component_mean = np.sum(membership[:, None] * midpoint, axis=0) / mass
        residual = midpoint - component_mean
        component_covariance = np.einsum(
            "n,ni,nj->ij", membership, residual, residual
        ) / mass
        latent_sign = np.tanh(
            old["kappa"][component] * (axes @ old["direction"][component])
        )
        resultant_vector = np.sum(
            (membership * latent_sign)[:, None] * axes, axis=0
        ) / mass
        resultant = float(np.linalg.norm(resultant_vector))
        if resultant == 0.0:
            component_direction = np.asarray([1.0, 0.0, 0.0])
            component_kappa = 0.0
        else:
            component_direction = canonical_axis(resultant_vector)
            component_kappa = solve_kappa(resultant)
        alpha.append(mass)
        mean.append(component_mean)
        covariance.append(_clip_covariance(component_covariance))
        direction.append(component_direction)
        kappa.append(component_kappa)
        effective.append(component_effective)
    alpha = np.asarray(alpha) / np.sum(alpha)
    return {
        "alpha": alpha,
        "mean_mpc_h": np.asarray(mean),
        "covariance_mpc_h_squared": np.asarray(covariance),
        "axis_direction": np.asarray(direction),
        "axis_kappa": np.asarray(kappa),
    }, np.asarray(effective)


def _json_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.asarray(value).tolist() for key, value in parameters.items()
    }


def fit_adaptive_mixture(
    midpoint: np.ndarray,
    axes: np.ndarray,
    weights: np.ndarray,
    fold_id: int,
    master_seed: int = 2026082002,
) -> dict[str, Any]:
    midpoint = np.asarray(midpoint, dtype=np.float64)
    axes = normalize_axes(axes)
    weights = np.asarray(weights, dtype=np.float64)
    if midpoint.shape != axes.shape or weights.shape != (len(midpoint),):
        raise ValueError("fit arrays have incompatible shapes")
    if np.any(weights < 0.0) or not np.sum(weights) > 0.0:
        raise ValueError("fit weights must be nonnegative with positive mass")
    weights = weights / np.sum(weights)
    outcomes = []
    for restart in range(8):
        rng = np.random.Generator(np.random.PCG64DXSM(
            np.random.SeedSequence(
                int(master_seed), spawn_key=(int(fold_id), int(restart))
            )
        ))
        try:
            parameters = _weighted_kmeans_initialization(midpoint, axes, weights, rng)
            objective, responsibility = _objective_and_responsibility(
                midpoint, axes, weights, parameters
            )
            stable_iterations = 0
            converged = False
            effective = np.full(4, np.nan)
            for iteration in range(1, 201):
                updated, effective = _m_step(
                    midpoint, axes, weights, responsibility, parameters
                )
                new_objective, new_responsibility = _objective_and_responsibility(
                    midpoint, axes, weights, updated
                )
                change = new_objective - objective
                tolerance = 1e-10 * max(1.0, abs(objective))
                if change < -tolerance:
                    raise RuntimeError("weighted EM objective decreased")
                relative = max(0.0, change) / max(1.0, abs(objective))
                stable_iterations = stable_iterations + 1 if relative <= 1e-8 else 0
                parameters = updated
                objective = new_objective
                responsibility = new_responsibility
                if stable_iterations >= 5:
                    converged = True
                    break
            if not converged:
                raise RuntimeError("weighted EM did not converge in 200 iterations")
            outcomes.append({
                "restart": restart,
                "converged": True,
                "iterations": iteration,
                "objective": objective,
                "component_effective_membership": effective.tolist(),
                "parameters": _json_parameters(parameters),
            })
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            outcomes.append({
                "restart": restart,
                "converged": False,
                "failure": str(error),
            })
    successful = [row for row in outcomes if row["converged"]]
    if not successful:
        raise RuntimeError("all eight frozen weighted-EM restarts failed")
    best_objective = max(row["objective"] for row in successful)
    tied = [
        row for row in successful if best_objective - row["objective"] <= 1e-12
    ]
    selected = min(tied, key=lambda row: row["restart"])
    return {
        "fold_id": int(fold_id),
        "selected_restart": selected["restart"],
        "objective": selected["objective"],
        "iterations": selected["iterations"],
        "component_effective_membership": selected[
            "component_effective_membership"
        ],
        "parameters": selected["parameters"],
        "restart_outcomes": outcomes,
    }


def fit_cross_validated_mixture(
    midpoint: np.ndarray,
    axes: np.ndarray,
    log_weights: np.ndarray,
    target_log_density: np.ndarray,
    master_seed: int = 2026082002,
) -> dict[str, Any]:
    midpoint = np.asarray(midpoint, dtype=np.float64)
    axes = normalize_axes(axes)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    target_log_density = np.asarray(target_log_density, dtype=np.float64)
    if target_log_density.shape != (len(midpoint),):
        raise ValueError("target log-density shape mismatch")
    weights = normalized_log_weights(log_weights)
    folds = []
    index = np.arange(len(midpoint))
    for fold in range(4):
        holdout = index % 4 == fold
        training = ~holdout
        fit = fit_adaptive_mixture(
            midpoint[training], axes[training], weights[training], fold, master_seed
        )
        holdout_weights = weights[holdout] / np.sum(weights[holdout])
        delta = float(holdout_weights @ (
            adaptive_component_logpdf(
                midpoint[holdout], axes[holdout], fit["parameters"]
            ) - target_log_density[holdout]
        ))
        folds.append({
            "fold": fold,
            "holdout_delta": delta,
            "pass": bool(delta >= 0.0),
            "fit": fit,
        })
    full = fit_adaptive_mixture(midpoint, axes, weights, 4, master_seed)
    return {
        "folds": folds,
        "all_holdout_delta_nonnegative": bool(all(row["pass"] for row in folds)),
        "full_fit": full,
    }


def antipodal_second_moment(direction: np.ndarray, kappa: float) -> np.ndarray:
    direction = canonical_axis(direction)
    if kappa == 0.0:
        parallel = 1.0 / 3.0
    else:
        parallel = 1.0 - 2.0 * _axis_mean_resultant(float(kappa)) / float(kappa)
    transverse = 0.5 * (1.0 - parallel)
    return (
        transverse * np.eye(3)
        + (parallel - transverse) * np.outer(direction, direction)
    )


def _maximum_standardized_error(
    samples: np.ndarray,
    expected: np.ndarray,
) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    error = np.abs(np.mean(samples, axis=0) - expected)
    standard_error = np.std(samples, axis=0, ddof=1) / math.sqrt(len(samples))
    if np.any((standard_error == 0.0) & (error != 0.0)):
        return math.inf
    return float(np.max(np.divide(
        error,
        standard_error,
        out=np.zeros_like(error),
        where=standard_error > 0.0,
    )))


def run_synthetic_validation(
    prior: dict[str, Any],
    parameters: dict[str, Any],
    master_seed: int = 2026082004,
    sampling_draw_count: int = 200000,
) -> dict[str, Any]:
    values = _validate_parameters(parameters)
    rng = np.random.Generator(np.random.PCG64DXSM(int(master_seed)))

    test_axes = sample_isotropic_axes(rng, 1000)
    antipodal_errors = []
    for component in range(4):
        positive = antipodal_vmf_logpdf(
            test_axes, values["direction"][component], values["kappa"][component]
        )
        negative = antipodal_vmf_logpdf(
            -test_axes, values["direction"][component], values["kappa"][component]
        )
        antipodal_errors.append(float(np.max(np.abs(positive - negative))))
    maximum_antipodal_error = max(antipodal_errors)

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(256)
    normalization_errors = {}
    reference_direction = np.asarray([0.0, 0.0, 1.0])
    quadrature_axes = np.column_stack((
        np.sqrt(np.maximum(0.0, 1.0 - nodes**2)),
        np.zeros_like(nodes),
        nodes,
    ))
    for kappa in (0.0, 1.0, 20.0):
        density = np.exp(antipodal_vmf_logpdf(
            quadrature_axes, reference_direction, kappa
        ))
        integral = float(2.0 * math.pi * (quadrature_weights @ density))
        normalization_errors[str(kappa)] = abs(integral - 1.0)

    h_midpoint, h_axes, _ = sample_adaptive_component(
        rng, parameters, sampling_draw_count
    )
    mixture_mean = np.sum(values["alpha"][:, None] * values["mean"], axis=0)
    mixture_covariance = np.zeros((3, 3))
    axis_second = np.zeros((3, 3))
    for component in range(4):
        displacement = values["mean"][component] - mixture_mean
        mixture_covariance += values["alpha"][component] * (
            values["covariance"][component] + np.outer(displacement, displacement)
        )
        axis_second += values["alpha"][component] * antipodal_second_moment(
            values["direction"][component], values["kappa"][component]
        )
    midpoint_mean_z = _maximum_standardized_error(h_midpoint, mixture_mean)
    centred = h_midpoint - mixture_mean
    covariance_samples = np.einsum("ni,nj->nij", centred, centred)
    midpoint_covariance_z = _maximum_standardized_error(
        covariance_samples, mixture_covariance
    )
    axis_samples = np.einsum("ni,nj->nij", h_axes, h_axes)
    axis_second_z = _maximum_standardized_error(axis_samples, axis_second)

    proposal_midpoint, proposal_axes, branch, component = sample_defensive_proposal(
        rng, prior, parameters, sampling_draw_count
    )
    target_log = target_geometry_logpdf(proposal_midpoint, prior)
    proposal_log = defensive_proposal_logpdf(
        proposal_midpoint, proposal_axes, prior, parameters
    )
    importance = np.exp(target_log - proposal_log)
    importance_mean = float(np.mean(importance))
    importance_standard_error = float(
        np.std(importance, ddof=1) / math.sqrt(len(importance))
    )
    importance_z = abs(importance_mean - 1.0) / importance_standard_error
    maximum_log_bound_excess = float(np.max(
        target_log - proposal_log - math.log(2.0)
    ))

    scalar_count = min(1000, sampling_draw_count)
    scalar = np.asarray([
        defensive_proposal_logpdf(
            proposal_midpoint[index], proposal_axes[index], prior, parameters
        )[0]
        for index in range(scalar_count)
    ])
    scalar_vectorized_difference = float(np.max(np.abs(
        scalar - proposal_log[:scalar_count]
    )))

    reproducibility_rng_a = np.random.Generator(np.random.PCG64DXSM(int(master_seed)))
    reproducibility_rng_b = np.random.Generator(np.random.PCG64DXSM(int(master_seed)))
    repeated_a = sample_defensive_proposal(
        reproducibility_rng_a, prior, parameters, 1024
    )
    repeated_b = sample_defensive_proposal(
        reproducibility_rng_b, prior, parameters, 1024
    )
    bitwise_identical = bool(all(
        np.array_equal(first, second)
        for first, second in zip(repeated_a, repeated_b)
    ))

    checks = {
        "antipodal_log_density": maximum_antipodal_error <= 1e-12,
        "sphere_normalization": max(normalization_errors.values()) <= 5e-12,
        "midpoint_mean_moments": midpoint_mean_z <= 5.0,
        "midpoint_covariance_moments": midpoint_covariance_z <= 5.0,
        "axis_second_moments": axis_second_z <= 5.0,
        "unit_likelihood_importance_mean": (
            importance_z <= 5.0 and abs(importance_mean - 1.0) <= 0.01
        ),
        "defensive_bound": maximum_log_bound_excess <= 1e-12,
        "scalar_vectorized_log_density": scalar_vectorized_difference <= 1e-10,
        "same_seed_bitwise_reproducibility": bitwise_identical,
    }
    return {
        "master_seed": int(master_seed),
        "sampling_draw_count": int(sampling_draw_count),
        "maximum_antipodal_log_density_difference": maximum_antipodal_error,
        "sphere_normalization_absolute_errors": normalization_errors,
        "midpoint_mean_max_standardized_error": midpoint_mean_z,
        "midpoint_covariance_max_standardized_error": midpoint_covariance_z,
        "axis_second_moment_max_standardized_error": axis_second_z,
        "unit_likelihood_importance_mean": importance_mean,
        "unit_likelihood_importance_standard_error": importance_standard_error,
        "unit_likelihood_importance_standardized_error": importance_z,
        "maximum_defensive_log_bound_excess": maximum_log_bound_excess,
        "scalar_vectorized_log_density_max_difference": scalar_vectorized_difference,
        "branch_counts": np.bincount(branch, minlength=2).tolist(),
        "adaptive_component_counts": np.bincount(
            component[component >= 0], minlength=4
        ).tolist(),
        "checks": checks,
        "all_pass": bool(all(checks.values())),
    }
