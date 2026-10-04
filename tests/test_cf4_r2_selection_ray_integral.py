import importlib.util
from pathlib import Path

import healpy as hp
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cf4_r2_selection_ray_integral.py"
SPEC = importlib.util.spec_from_file_location("cf4_r2_selection_ray_integral", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def mock_physics():
    rtab = np.linspace(MODULE.R_INNER, MODULE.R_OUTER, 4097)
    coefficients = np.arange(1., 7.)
    cumulative = coefficients[:, None] * (rtab[None, :] ** 3 - MODULE.R_INNER ** 3) / 3.
    cumulative_edges = np.asarray([
        np.interp(MODULE.RADIAL_EDGES, rtab, cumulative[p]) for p in range(6)
    ])
    ones = np.ones(hp.nside2npix(512), dtype=np.float64)
    return {
        "maps": [ones, ones.copy()],
        "rotation": np.eye(3),
        "rtab": rtab,
        "cumulative": cumulative,
        "cumulative_edges": cumulative_edges,
    }


def test_radial_edges_split_intervals_without_double_counting():
    lo, hi = MODULE._shell_indices(
        np.array([29., 30., 31., 59.]),
        np.array([31., 31., 32., 61.]))
    np.testing.assert_array_equal(lo, [0, 1, 1, 1])
    np.testing.assert_array_equal(hi, [1, 1, 1, 2])


def test_chunk_sizes_and_one_piece_agree_for_max_cap_geometry():
    physics = mock_physics()
    low = np.zeros(3)
    high = np.full(3, 3.)
    center = (low + high) / 2.
    axis, cap = MODULE.candidate_cap(center, low, high, np.eye(3))
    candidates = hp.query_disc(64, axis, cap, inclusive=True, nest=False)
    results = [MODULE.integrate_candidate_chunks(
        candidates, 64, low, high, physics, size)
        for size in (7, 113, len(candidates))]
    for current in results[1:]:
        np.testing.assert_allclose(current["exposure"], results[0]["exposure"],
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(current["geometry"], results[0]["geometry"],
                                   rtol=1e-12, atol=1e-14)
    assert results[0]["shell_hit_rays"] > 0


def test_outer_shell_tangent_has_no_positive_length_shell_rays():
    physics = mock_physics()
    low = np.array([MODULE.R_OUTER, 0., 0.])
    high = low + MODULE.DX
    center = (low + high) / 2.
    axis, cap = MODULE.candidate_cap(center, low, high, np.eye(3))
    candidates = hp.query_disc(64, axis, cap, inclusive=True, nest=False)
    result = MODULE.integrate_candidate_chunks(
        candidates, 64, low, high, physics, 127)
    assert result["shell_hit_rays"] == 0
    np.testing.assert_array_equal(result["exposure"], np.zeros((6, 6)))
    np.testing.assert_array_equal(result["geometry"], np.zeros(6))


def test_full_geometry_census_records_cross_tab_without_assuming_set_equality():
    census = MODULE.exact_geometry_census()
    counts = census["counts"]
    assert counts["active_shell_intersection"] == 938_128
    assert counts["inner_boundary_partial"] == 56
    assert counts["outer_boundary_partial"] == 67_544
    assert counts["active_cells_over_cap_estimate_limit"] == 56
    assert len(census["maximum_cap_cells"]) == 8
    assert (counts["inner_partial_and_cap_over_limit"]
            + counts["inner_partial_and_cap_not_over_limit"]
            == counts["inner_boundary_partial"])
    assert (counts["inner_partial_and_cap_over_limit"]
            + counts["not_inner_partial_and_cap_over_limit"]
            == counts["active_cells_over_cap_estimate_limit"])
    crosstab = census["joint_status_crosstab"]
    assert len(crosstab) == 8
    assert sum(row["active_cells"] for row in crosstab) == counts[
        "active_shell_intersection"]
    assert sum(row["active_cells"] for row in crosstab
               if row["inner_boundary_partial"]
               and row["cap_estimate_over_limit"]) == counts[
                   "inner_partial_and_cap_over_limit"]
    assert sum(row["active_cells"] for row in crosstab
               if row["outer_boundary_partial"]) == counts[
                   "outer_boundary_partial"]
