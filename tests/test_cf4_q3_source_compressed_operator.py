from __future__ import annotations

import numpy as np
import pytest

from cf4_2mpp_joint_likelihood_local import LikelihoodInputError
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit, q1_candidate_oracle_gate
from cf4_q3_source_compressed_operator import (
    aggregate_mass_basis,
    aggregate_population_masses,
    candidate_metadata,
    evaluate_grouped_q1_operator,
    exact_geometry_compress,
)


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.25, 1.5, 7.0], [4.25, 1.5, 7.0]],
        dtype=np.float64,
    )
    los = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    scale = np.asarray([0.25, 0.25, 0.75, 0.75], dtype=np.float64)
    return positions, los, scale


def test_exact_grouping_is_deterministic_and_lossless_for_masses() -> None:
    positions, los, scale = _geometry()
    compressed = exact_geometry_compress(positions, los, scale)
    masses = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    grouped = aggregate_population_masses(compressed, masses)
    assert compressed.group_count == 2
    assert compressed.source_to_group.tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(grouped, [[3.0, 7.0], [7.0, 3.0]])
    assert candidate_metadata(compressed)["compression_ratio"] == 2.0


def test_mass_basis_and_gradients_are_summed_without_state_shortcut() -> None:
    positions, los, scale = _geometry()
    compressed = exact_geometry_compress(positions, los, scale)
    basis = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    grouped = aggregate_mass_basis(compressed, basis)
    np.testing.assert_allclose(grouped[:, 0, :], basis[:, 0, :] + basis[:, 1, :])
    np.testing.assert_allclose(grouped[:, 1, :], basis[:, 2, :] + basis[:, 3, :])


def test_grouped_operator_matches_uncompressed_q1_oracle() -> None:
    positions, los, scale = _geometry()
    masses = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [0.5, 1.5, 2.5, 3.5],
            [3.5, 2.5, 1.5, 0.5],
            [1.25, 1.25, 2.25, 2.25],
            [2.25, 2.25, 1.25, 1.25],
        ]
    )
    compressed = exact_geometry_compress(positions, los, scale)
    grouped = evaluate_grouped_q1_operator(compressed, masses, 8, 8.0)
    reference = np.stack(
        [cell_integrated_tsc_deposit(positions, masses[p], los, scale, 8, 8.0) for p in range(6)]
    )
    gate = q1_candidate_oracle_gate(grouped, reference)
    assert gate["status"] == "PASS"
    np.testing.assert_allclose(grouped, reference, rtol=0.0, atol=2.0e-13)


def test_distinct_bitwise_geometry_is_not_merged() -> None:
    positions, los, scale = _geometry()
    positions = positions.copy()
    positions[1, 0] = np.nextafter(positions[1, 0], np.inf)
    compressed = exact_geometry_compress(positions, los, scale)
    assert compressed.group_count == 3
    assert compressed.compression_ratio == pytest.approx(4.0 / 3.0)


def test_invalid_geometry_fails_closed() -> None:
    positions, los, scale = _geometry()
    bad_los = los.copy()
    bad_los[0] *= 2.0
    with pytest.raises(LikelihoodInputError):
        exact_geometry_compress(positions, bad_los, scale)
