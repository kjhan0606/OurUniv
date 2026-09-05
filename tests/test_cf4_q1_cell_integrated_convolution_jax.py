"""Small-fixture value/gradient checks for the Q1 JAX contraction bridge."""

from __future__ import annotations

import numpy as np
import pytest

from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit
from cf4_q1_cell_integrated_convolution_jax import (
    JaxOperatorInputError,
    assert_q1_numpy_provenance,
    contract_mass_field,
    contract_mass_field_from_jax,
    jax,
    jnp,
    mass_gradient_basis,
)


pytestmark = pytest.mark.skipif(jax is None, reason="JAX is not installed")


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray([[0.37, 1.22, 2.31], [5.71, 0.42, 3.66], [2.08, 4.90, 5.43]], dtype=np.float64)
    los = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [1.0, 2.0, 2.0]], dtype=np.float64)
    los /= np.linalg.norm(los, axis=1)[:, None]
    scales = np.asarray([0.31, 0.77, 1.05], dtype=np.float64)
    masses = np.asarray([0.7, 1.2, 0.3], dtype=np.float64)
    return positions, los, scales, masses


def _response_basis() -> tuple[np.ndarray, np.ndarray]:
    positions, los, scales, masses = _fixture()
    basis = np.stack(
        [
            cell_integrated_tsc_deposit(
                positions,
                np.eye(3, dtype=np.float64)[source],
                los,
                scales,
                8,
                6.0,
            )
            for source in range(3)
        ],
        axis=0,
    )
    oracle = cell_integrated_tsc_deposit(positions, masses, los, scales, 8, 6.0)
    return basis, oracle


def test_provenance_is_bound_to_sealed_q1_oracle() -> None:
    assert assert_q1_numpy_provenance().startswith("74ae1bb1")


def test_jax_contraction_matches_q1_numpy_value() -> None:
    basis, oracle = _response_basis()
    candidate = np.asarray(contract_mass_field(basis, _fixture()[-1]))
    np.testing.assert_allclose(candidate, oracle, rtol=0.0, atol=2.0e-12)


def test_jax_mass_jacobian_is_the_response_basis() -> None:
    basis, _oracle = _response_basis()
    masses = jnp.asarray(_fixture()[-1])
    jacobian = jax.jacfwd(lambda m: contract_mass_field_from_jax(basis, m))(masses)
    np.testing.assert_allclose(np.asarray(jacobian), basis, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(np.asarray(mass_gradient_basis(basis)), basis, rtol=0.0, atol=0.0)


def test_basis_rejects_nonconservative_response() -> None:
    basis, _oracle = _response_basis()
    broken = basis.copy()
    broken[0, 0, 0, 0] += 1.0e-4
    with pytest.raises(JaxOperatorInputError, match="conserve unit mass"):
        contract_mass_field(broken, _fixture()[-1])
