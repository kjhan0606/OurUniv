from __future__ import annotations

import numpy as np
import pytest

from cf4_2mpp_joint_likelihood_local import LikelihoodInputError
from cf4_q1_cell_integrated_convolution import (
    cell_integrated_tsc_deposit,
    q1_candidate_oracle_gate,
)
from cf4_q6_knot_aligned_operator import (
    aggregate_mass_basis,
    aggregate_population_masses,
    candidate_metadata,
    exact_knot_compress,
    evaluate_grouped_q1_operator,
    scatter_group_cotangent,
)


def _development_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],  # exact duplicate: safe merge
            [1.0 + 2.0 ** -30, 2.0, 3.0],  # near-coincident: must not merge
            [7.999999999, 0.25, 7.75],  # periodic seam
            [3.125, 6.25, 1.75],
            [5.5, 4.5, 2.5],
        ],
        dtype=np.float64,
    )
    los = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.6, 0.8],
            [1.0 / np.sqrt(3.0)] * 3,
        ],
        dtype=np.float64,
    )
    scales = np.asarray([0.25, 0.25, 0.25, 0.75, 0.5, 0.0], dtype=np.float64)
    return positions, los, scales


def test_exact_knot_contract_matches_q1_with_non_degenerate_geometry() -> None:
    positions, los, scales = _development_geometry()
    compressed = exact_knot_compress(positions, los, scales, 8, 8.0)
    masses = np.asarray(
        [
            [1.0, 2.0, 0.5, 3.0, 4.0, 1.5],
            [0.5, 1.0, 2.0, 2.5, 3.5, 0.25],
            [4.0, 3.0, 2.0, 1.0, 0.75, 0.5],
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            [3.0, 2.0, 1.0, 0.5, 0.25, 0.125],
        ],
        dtype=np.float64,
    )
    grouped = evaluate_grouped_q1_operator(compressed, masses)
    reference = np.stack(
        [
            cell_integrated_tsc_deposit(
                positions, masses[p], los, scales, 8, 8.0
            )
            for p in range(masses.shape[0])
        ]
    )
    gate = q1_candidate_oracle_gate(grouped, reference)
    assert gate["status"] == "PASS"
    np.testing.assert_allclose(grouped, reference, rtol=0.0, atol=3.0e-13)
    assert compressed.source_count == 6
    assert compressed.group_count == 5
    assert compressed.compression_ratio == pytest.approx(6.0 / 5.0)


def test_signature_is_not_position_only_and_exposes_frozen_contract() -> None:
    positions, los, scales = _development_geometry()
    compressed = exact_knot_compress(positions, los, scales, 8, 8.0)
    metadata = candidate_metadata(compressed)
    assert metadata["approximation"] is False
    assert metadata["arbitrary_cotangent_transpose_preserved"] is True
    assert "absolute_target_cell_and_27_stencil_mapping" in metadata["signature_components"]
    assert "ordered_breakpoints_and_midpoints" in metadata["signature_components"]
    assert metadata["interval_count_max"] >= metadata["interval_count_min"]
    # The near-coincident source differs by one bit and cannot be merged.
    assert compressed.source_to_group[0] != compressed.source_to_group[2]


def test_group_transpose_is_exact_for_arbitrary_cotangent() -> None:
    positions, los, scales = _development_geometry()
    compressed = exact_knot_compress(positions, los, scales, 8, 8.0)
    rng = np.random.default_rng(20260904)
    basis = rng.normal(size=(6, compressed.source_count, 4))
    grouped = aggregate_mass_basis(compressed, basis)
    cotangent = rng.normal(size=(compressed.group_count, 4))
    scattered = scatter_group_cotangent(compressed, cotangent)
    np.testing.assert_allclose(
        np.sum(grouped * cotangent), np.sum(basis * scattered), rtol=0.0, atol=2.0e-13
    )


def test_population_mass_aggregation_preserves_all_populations() -> None:
    positions, los, scales = _development_geometry()
    compressed = exact_knot_compress(positions, los, scales, 8, 8.0)
    masses = np.arange(6 * compressed.source_count, dtype=np.float64).reshape(6, -1)
    grouped = aggregate_population_masses(compressed, masses)
    np.testing.assert_allclose(grouped[:, 0], masses[:, 0] + masses[:, 1])
    np.testing.assert_allclose(grouped[:, 1], masses[:, 2])


def test_invalid_inputs_fail_closed() -> None:
    positions, los, scales = _development_geometry()
    with pytest.raises(LikelihoodInputError):
        exact_knot_compress(positions, los * 2.0, scales, 8, 8.0)
    compressed = exact_knot_compress(positions, los, scales, 8, 8.0)
    with pytest.raises(LikelihoodInputError):
        scatter_group_cotangent(compressed, np.zeros(compressed.group_count + 1))
