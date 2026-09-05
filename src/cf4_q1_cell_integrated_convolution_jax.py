"""Differentiable contraction counterpart for the frozen Q1 NumPy oracle.

Q1's periodic, cell-integrated response is the authoritative geometry oracle.
This module does not reimplement its dynamic interval integration.  Instead it
accepts a finite, source-major response basis produced by that oracle and
contracts it with state-dependent source masses in JAX.  The representation is
deliberately small and explicit: it is a development-fixture bridge for value
and mass-gradient checks, not a claim that the full PM source stream fits in a
single dense basis.  Q2's resource preflight keeps the direct sourcewise route
closed for production inference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

try:  # JAX is optional in the lightweight test environment.
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover - exercised by the skip path in tests.
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


Q1_NUMPY_SOURCE_SHA256 = "74ae1bb12171a2baac76c8052d592b4dc5098043bf7c11bca6ffb9eea852d6b2"
Q1_JAX_CONTRACTION_NAME = "q1_cell_integrated_response_basis_contraction"


class JaxUnavailable(RuntimeError):
    """Raised when the optional JAX dependency is not installed."""


class JaxOperatorInputError(ValueError):
    """Raised when a response basis violates the static-shape contract."""


def require_jax() -> None:
    if jax is None or jnp is None:
        raise JaxUnavailable("JAX is required for the differentiable Q1 contraction path")


def assert_q1_numpy_provenance(expected_sha256: str = Q1_NUMPY_SOURCE_SHA256) -> str:
    """Verify that the contraction is still bound to the sealed Q1 oracle."""

    root = Path(__file__).resolve().parents[1]
    actual = hashlib.sha256((root / "src" / "cf4_q1_cell_integrated_convolution.py").read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise JaxOperatorInputError(f"Q1 NumPy provenance mismatch: expected {expected_sha256}, got {actual}")
    return actual


def _validate_response_basis(response_basis: Iterable[object]) -> np.ndarray:
    response = np.asarray(response_basis, dtype=np.float64)
    if response.ndim != 4 or response.shape[0] == 0:
        raise JaxOperatorInputError("response_basis must have shape (M, N, N, N) with M > 0")
    if len(set(response.shape[1:])) != 1 or response.shape[1] < 2:
        raise JaxOperatorInputError("response_basis must be a cubic grid with N >= 2")
    if not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise JaxOperatorInputError("response_basis must be finite and non-negative")
    totals = np.sum(response, axis=(1, 2, 3))
    if not np.allclose(totals, 1.0, rtol=0.0, atol=1.0e-10):
        raise JaxOperatorInputError("each response basis element must conserve unit mass")
    return np.ascontiguousarray(response, dtype=np.float64)


def contract_mass_field(response_basis: Iterable[object], masses: Iterable[float]):
    """Return ``sum_m masses[m] * response_basis[m]`` as a JAX array."""

    require_jax()
    response = _validate_response_basis(response_basis)
    mass = np.asarray(masses, dtype=np.float64)
    if mass.shape != (response.shape[0],):
        raise JaxOperatorInputError("masses must have one entry per response basis element")
    if not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
        raise JaxOperatorInputError("masses must be finite and non-negative")
    return jnp.einsum("mxyz,m->xyz", jnp.asarray(response), jnp.asarray(mass), precision=jax.lax.Precision.HIGHEST)


def contract_mass_field_from_jax(response_basis, masses):
    """JAX-traceable contraction for gradients/JVPs after host validation."""

    require_jax()
    response = jnp.asarray(response_basis, dtype=jnp.float64)
    mass = jnp.asarray(masses, dtype=jnp.float64)
    return jnp.einsum("mxyz,m->xyz", response, mass, precision=jax.lax.Precision.HIGHEST)


def mass_gradient_basis(response_basis: Iterable[object]):
    """Return the exact field Jacobian with respect to source masses."""

    require_jax()
    response = _validate_response_basis(response_basis)
    return jnp.asarray(response)

