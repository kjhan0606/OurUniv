#!/usr/bin/env python3
"""Exact small/control engine for peak evidence in the projection null space.

This module implements the probability model required by the independent-CF4
parent architecture.  It deliberately uses full unitary FFTs and exact point
templates; it is the correctness reference for a later scalable N576 engine,
not yet the production all-parent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from cf4_projection_contract import (
    add_restriction_adjoint,
    normalized_errors,
    prolong_white_field,
    restrict_spectrum,
    restrict_white_field,
    restriction_adjoint_spectrum,
)


def _real_ifft(spectrum: np.ndarray, label: str) -> np.ndarray:
    value = np.fft.ifftn(spectrum, norm="ortho")
    real_rms = float(np.sqrt(np.mean(value.real ** 2)))
    imaginary_rms = float(np.sqrt(np.mean(value.imag ** 2)))
    if imaginary_rms / max(real_rms, np.finfo(float).tiny) > 1e-12:
        raise RuntimeError(f"{label} broke Hermitian symmetry")
    return value.real


def apply_filter(field: np.ndarray, filter_full: np.ndarray) -> np.ndarray:
    """Apply a real self-adjoint Fourier multiplier with unitary FFTs."""
    field = np.asarray(field, dtype=np.float64)
    filter_full = np.asarray(filter_full)
    if field.shape != filter_full.shape or field.ndim != 3:
        raise ValueError("field and filter_full must have one cubic 3-D shape")
    spectrum = np.fft.fftn(field, norm="ortho") * filter_full
    return _real_ifft(spectrum, "filter")


def minimum_norm_fine_mean(coarse: np.ndarray, fine_n: int) -> np.ndarray:
    """Return R* y in spatial coordinates."""
    coarse = np.asarray(coarse, dtype=np.float64)
    if coarse.ndim != 3 or not (
        coarse.shape[0] == coarse.shape[1] == coarse.shape[2]
    ):
        raise ValueError("coarse must be cubic")
    coarse_fft = np.fft.fftn(coarse, norm="ortho")
    fine_fft = restriction_adjoint_spectrum(coarse_fft, fine_n)
    return _real_ifft(fine_fft, "minimum-norm fine mean")


def null_project_field(field: np.ndarray, coarse_n: int) -> np.ndarray:
    """Apply Q=I-R*R to a fine spatial field."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or not (
        field.shape[0] == field.shape[1] == field.shape[2]
    ):
        raise ValueError("field must be cubic")
    spectrum = np.fft.fftn(field, norm="ortho")
    coarse_component = restrict_spectrum(spectrum, coarse_n)
    projected = add_restriction_adjoint(spectrum, -coarse_component)
    return _real_ifft(projected, "null projection")


def normalized_gaussian_logpdf(
    value: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, dict[str, float]]:
    value = np.asarray(value, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if value.ndim != 1 or mean.shape != value.shape:
        raise ValueError("value and mean must be aligned vectors")
    if covariance.shape != (value.size, value.size):
        raise ValueError("covariance shape mismatch")
    cholesky = np.linalg.cholesky(covariance)
    residual = value - mean
    whitened = np.linalg.solve(cholesky, residual)
    quadratic = float(whitened @ whitened)
    log_determinant = float(2.0 * np.sum(np.log(np.diag(cholesky))))
    logpdf = -0.5 * (
        value.size * math.log(2.0 * math.pi) + log_determinant + quadratic
    )
    return logpdf, {
        "quadratic": quadratic,
        "log_determinant": log_determinant,
        "normalization_dimension": int(value.size),
    }


@dataclass
class ExactPeakOperator:
    fine_n: int
    coarse_n: int
    filter_full: np.ndarray
    points: np.ndarray
    sigma: np.ndarray
    null_templates: np.ndarray
    signal_covariance: np.ndarray
    observation_covariance: np.ndarray

    def predict_parent(self, coarse: np.ndarray) -> np.ndarray:
        mean_fine = minimum_norm_fine_mean(coarse, self.fine_n)
        smoothed = apply_filter(mean_fine, self.filter_full)
        return smoothed[tuple(self.points.T)]

    def predict_field(self, field: np.ndarray) -> np.ndarray:
        smoothed = apply_filter(field, self.filter_full)
        return smoothed[tuple(self.points.T)]

    def log_evidence(
        self,
        coarse: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        mean = self.predict_parent(coarse)
        logpdf, terms = normalized_gaussian_logpdf(
            targets, mean, self.observation_covariance
        )
        return logpdf, {"predicted_mean": mean, **terms}

    def conditional_sample(
        self,
        coarse: np.ndarray,
        targets: np.ndarray,
        fine_seed: int,
        noise_seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        targets = np.asarray(targets, dtype=np.float64)
        base = prolong_white_field(coarse, self.fine_n, fine_seed)
        predicted = self.predict_field(base)
        rng = np.random.Generator(np.random.PCG64DXSM(int(noise_seed)))
        mock_noise = rng.normal(0.0, self.sigma)
        weights = np.linalg.solve(
            self.observation_covariance,
            targets - predicted - mock_noise,
        )
        correction = np.tensordot(weights, self.null_templates, axes=(0, 0))
        conditioned = (base.astype(np.float64) + correction).astype(np.float32)
        achieved = self.predict_field(conditioned)
        roundtrip = normalized_errors(
            restrict_white_field(conditioned, self.coarse_n), coarse
        )
        log_evidence, evidence_terms = self.log_evidence(coarse, targets)
        return conditioned, {
            "predicted_before": predicted,
            "achieved_after": achieved,
            "mock_noise": mock_noise,
            "weights": weights,
            "roundtrip": roundtrip,
            "normalized_log_evidence": log_evidence,
            "evidence_terms": evidence_terms,
        }


def prepare_exact_peak_operator(
    filter_full: np.ndarray,
    coarse_n: int,
    points: np.ndarray,
    sigma: float | np.ndarray,
) -> ExactPeakOperator:
    """Build exact Q-projected templates and their Gaussian covariance."""
    filter_full = np.asarray(filter_full)
    if filter_full.ndim != 3 or not (
        filter_full.shape[0] == filter_full.shape[1] == filter_full.shape[2]
    ):
        raise ValueError("filter_full must be cubic")
    fine_n = filter_full.shape[0]
    points = np.mod(np.asarray(points, dtype=np.int64), fine_n)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (m,3)")
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (len(points),))
    if np.any(~np.isfinite(sig)) or np.any(sig <= 0.0):
        raise ValueError("sigma must be finite and positive")

    templates = []
    for point in points:
        impulse = np.zeros((fine_n, fine_n, fine_n), dtype=np.float64)
        impulse[tuple(point)] = 1.0
        adjoint_template = apply_filter(impulse, filter_full)
        templates.append(null_project_field(adjoint_template, coarse_n))
    null_templates = np.asarray(templates)
    covariance = np.empty((len(points), len(points)), dtype=np.float64)
    for column, template in enumerate(null_templates):
        response = apply_filter(template, filter_full)
        covariance[:, column] = response[tuple(points.T)]
    covariance = 0.5 * (covariance + covariance.T)
    observation_covariance = covariance + np.diag(sig ** 2)
    np.linalg.cholesky(observation_covariance)
    return ExactPeakOperator(
        fine_n=fine_n,
        coarse_n=int(coarse_n),
        filter_full=filter_full.astype(np.complex128, copy=False),
        points=points,
        sigma=sig,
        null_templates=null_templates,
        signal_covariance=covariance,
        observation_covariance=observation_covariance,
    )


def engineering_metadata() -> dict[str, Any]:
    return {
        "role": "exact small/control reference, not scalable N576 implementation",
        "null_projector": "Q=I-R*R",
        "evidence": "normalized multivariate Gaussian including log determinant",
        "conditional": "Matheron draw in ker(R)",
        "FFT": "NumPy unitary full FFT; no workers=-1",
        "candidate_generation_authorized": False,
    }
