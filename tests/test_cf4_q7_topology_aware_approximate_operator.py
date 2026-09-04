from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from cf4_q7_topology_aware_approximate_operator import (
    Q7_DERIVATIVE_DIRECTIONS,
    FROZEN_DIRECTION_NAMES,
    Q7_SUMMARY_COUNT,
    build_topology_aware_atlas,
    candidate_metadata,
    compare_to_q1,
    evaluate_atlas,
    induced_summary_l1_bounds,
    sourcewise_q1_fields,
)
from scripts.cf4_q7_development_fixture import (
    FAMILIES,
    fixture_sha256,
    generate_fixture,
)
from cf4_2mpp_joint_likelihood_local import LikelihoodInputError


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0 + 2.0 ** -30, 2.0, 3.0],
            [7.999999999, 0.25, 7.75],
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


def _basis_metadata(basis: np.ndarray) -> dict[str, str | tuple[str, ...]]:
    labels = tuple(FROZEN_DIRECTION_NAMES)
    return {
        "directional_basis_labels": labels,
        "directional_basis_sha256": hashlib.sha256(
            json.dumps(labels, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "directional_basis_content_sha256": hashlib.sha256(
            np.asarray(basis, dtype="<f8", order="C").tobytes()
        ).hexdigest(),
        "directional_basis_registry_id": "q7-development-fixture-v1",
    }


def test_preregistered_q7_fixtures_are_reproducible() -> None:
    expected = {
        "same_knot_different_coefficient": "c3549762ee86ef2e01ea733dc4396327323f88368bdc759cabb8a3e7096a6fe2",
        "translated_cell": "7d70a8e3805587acd7d8d93c4d81e06386359bca5efc64e16cc2602516b3d16d",
        "near_coincident": "b6e9219293af3391e267113898a813bcc0851514774bb4ea20c6c0f24e48fee7",
        "interval_permutation": "bc7dd434cd4d3c2b02701fb2c63e51e943929fcbc54096e84231b6f1389eca7d",
        "seam_sliver_clip": "8816bb32b81bcecbc446c0cb69686c112920b48374c7e459478f5cf37a7b1b6f",
        "los_sigma_variation": "e31366c9fdf791a2289bab54b3b9d7f38aef002e461aa171b1a89501e6e5021a",
        "cold_zero_near_zero": "c48b28bb340e8dc9107b7d40dfd86d72aa485e513f89cd1c5dc50ff472e6322b",
    }
    for family, _seed, _count in FAMILIES:
        assert fixture_sha256(*generate_fixture(family=family)) == expected[family]


def test_default_route_falls_back_without_continuous_certificate() -> None:
    positions, los, scales = _geometry()
    atlas = build_topology_aware_atlas(positions, los, scales, 8, 8.0)
    masses = np.arange(6 * positions.shape[0], dtype=np.float64).reshape(6, -1) + 0.25
    basis = np.ones((Q7_DERIVATIVE_DIRECTIONS, positions.shape[0]), dtype=np.float64)
    result = evaluate_atlas(atlas, masses, directional_mass_basis=basis, **_basis_metadata(basis))
    assert result.overflow_source_count < positions.shape[0]
    assert result.certificate_status in {"CERTIFIED", "CERTIFIED_WITH_SOURCEWISE_OVERFLOW"}
    assert not result.used_uncertified_finite_enclosure
    assert result.gradients.shape == (Q7_DERIVATIVE_DIRECTIONS, 8, 8, 8)


def test_finite_enclosure_is_explicitly_non_promotable_but_preserves_state_basis() -> None:
    positions, los, scales, masses = generate_fixture(
        family="same_knot_different_coefficient"
    )
    atlas = build_topology_aware_atlas(positions, los, scales, 8, 8.0)
    basis = np.random.default_rng(20260904).normal(size=(6, 32, Q7_DERIVATIVE_DIRECTIONS))
    with pytest.raises(LikelihoodInputError):
        evaluate_atlas(
            atlas,
            masses,
            directional_mass_basis=basis,
            allow_uncertified_finite_enclosure=True,
            **_basis_metadata(basis),
        )
    result = evaluate_atlas(atlas, masses, directional_mass_basis=basis, **_basis_metadata(basis))
    assert atlas.compression_ratio == pytest.approx(2.0)
    assert result.overflow_source_count == 0
    assert not result.used_uncertified_finite_enclosure
    assert result.certificate_status == "CERTIFIED"
    assert np.all(np.isfinite(result.certified_value_l1_per_population))
    assert np.all(np.isfinite(result.finite_gradient_l1))
    assert result.gradients.shape == (6, Q7_DERIVATIVE_DIRECTIONS, 8, 8, 8)


def test_declared_continuous_enclosure_is_checked_against_fixture_members() -> None:
    positions, los, scales = _geometry()
    finite = build_topology_aware_atlas(positions, los, scales, 8, 8.0)
    bounds = {
        index: (finite.lower_enclosures[index], finite.upper_enclosures[index])
        for index in range(finite.bin_count)
    }
    with pytest.raises(LikelihoodInputError):
        build_topology_aware_atlas(positions, los, scales, 8, 8.0, continuous_enclosures=bounds)
    metadata = candidate_metadata(finite)
    assert metadata["numeric_knot_coefficients_in_key"] is False
    assert metadata["continuous_certificate_required_for_promotion"] is True
    assert metadata["summary_bound_count"] == Q7_SUMMARY_COUNT


def test_invalid_continuous_enclosure_fails_closed() -> None:
    positions, los, scales = _geometry()
    finite = build_topology_aware_atlas(positions, los, scales, 8, 8.0)
    bad = {
        0: (
            np.zeros((8, 8, 8), dtype=np.float64),
            np.zeros((8, 8, 8), dtype=np.float64),
        )
    }
    # Unless bin zero happens to be exactly zero everywhere, the member check
    # must reject this purported continuous certificate.
    with pytest.raises(LikelihoodInputError):
        build_topology_aware_atlas(
            positions, los, scales, 8, 8.0, continuous_enclosures=bad
        )
    measured = compare_to_q1(
        finite.representative_fields[:1],
        finite.representative_fields[:1],
    )
    assert measured["value_l1"] == 0.0


def test_zero_scale_subcell_positions_are_not_merged() -> None:
    positions = np.asarray([[1.10, 2.0, 3.0], [1.20, 2.0, 3.0]], dtype=np.float64)
    los = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    scales = np.zeros(2, dtype=np.float64)
    atlas = build_topology_aware_atlas(positions, los, scales, 8, 8.0)
    assert atlas.bin_count == 2
    assert all(not item.continuous_certified for item in atlas.bins)


def test_summary_bound_api_requires_175_fixed_maps() -> None:
    per_bin = np.asarray([0.5, 1.0], dtype=np.float64)
    norms = np.ones((Q7_SUMMARY_COUNT, 2), dtype=np.float64)
    result = induced_summary_l1_bounds(per_bin, norms)
    assert result.shape == (Q7_SUMMARY_COUNT,)
    assert np.all(result == 1.5)
    with pytest.raises(LikelihoodInputError):
        induced_summary_l1_bounds(per_bin, np.ones((174, 2)))
