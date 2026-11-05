#!/usr/bin/env python3
"""Diagonal-normal midpoint proposals shared by LG generation and audits."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _parameters(specification: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(specification["mean_mpc_h"], dtype=np.float64)
    sigma = np.broadcast_to(
        np.asarray(specification["sigma_mpc_h"], dtype=np.float64), (3,)
    )
    if mean.shape != (3,) or np.any(~np.isfinite(mean)):
        raise ValueError("midpoint mean_mpc_h must contain three finite values")
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("midpoint sigma_mpc_h must be finite and positive")
    return mean, sigma


def diagonal_normal_logpdf(
    value: np.ndarray, specification: dict[str, Any]
) -> float:
    """Normalized log density of a three-dimensional diagonal Gaussian."""
    value = np.asarray(value, dtype=np.float64)
    mean, sigma = _parameters(specification)
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise ValueError("midpoint value must contain three finite values")
    residual = (value - mean) / sigma
    return float(np.sum(
        -0.5 * residual**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    ))


def mixture_logpdf(
    value: np.ndarray, components: list[dict[str, Any]]
) -> float:
    """Normalized log density of a finite diagonal-normal mixture."""
    if not components:
        raise ValueError("midpoint proposal mixture must have components")
    weights = np.asarray([float(row["weight"]) for row in components])
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("midpoint mixture weights must be finite and positive")
    if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("midpoint mixture weights must sum to one")
    terms = np.asarray([
        math.log(float(component["weight"]))
        + diagonal_normal_logpdf(value, component)
        for component in components
    ])
    maximum = float(np.max(terms))
    return maximum + math.log(float(np.exp(terms - maximum).sum()))


def draw_diagonal_normal_mixture(
    rng: np.random.Generator, components: list[dict[str, Any]]
) -> tuple[np.ndarray, int]:
    """Draw from a validated mixture and return the component index."""
    # mixture_logpdf performs all component and weight validation.
    mixture_logpdf(np.zeros(3, dtype=np.float64), components)
    weights = np.asarray([float(row["weight"]) for row in components])
    component_index = int(rng.choice(len(components), p=weights))
    mean, sigma = _parameters(components[component_index])
    return rng.normal(mean, sigma), component_index


def verify_defensive_component(
    prior: dict[str, Any],
    components: list[dict[str, Any]],
    minimum_weight: float,
) -> float:
    """Return the analytic p/g bound implied by an exact prior component."""
    prior_mean, prior_sigma = _parameters(prior)
    matches = []
    for component in components:
        mean, sigma = _parameters(component)
        if np.array_equal(mean, prior_mean) and np.array_equal(sigma, prior_sigma):
            matches.append(component)
    if len(matches) != 1:
        raise RuntimeError("proposal must contain exactly one target-prior component")
    # Also validates every component and the normalization of the weights.
    mixture_logpdf(prior_mean, components)
    weight = float(matches[0]["weight"])
    if weight < minimum_weight:
        raise RuntimeError("defensive target-prior component is too small")
    return 1.0 / weight
