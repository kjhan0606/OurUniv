import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cf4_r2_shell_geometry_census.py"
SPEC = importlib.util.spec_from_file_location("cf4_r2_shell_geometry_census", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_origin_corner_voxel_is_partial_inner_shell_and_has_large_cap():
    low = np.zeros(3)
    high = np.full(3, 3.)
    rmin, rmax = MODULE.interval_distance_bounds(low, high)
    assert rmin == 0.
    assert np.isclose(rmax, 3. * np.sqrt(3.))
    assert rmin < MODULE.R_INNER < rmax
    assert np.isclose(MODULE.cell_cap((low + high) / 2.),
                      np.arccos(1. / np.sqrt(3.)) + 1e-6)


def test_shell_boundary_tangency_is_not_positive_volume_support():
    low = np.array([MODULE.R_OUTER, 0., 0.])
    high = low + MODULE.DX
    rmin, _ = MODULE.interval_distance_bounds(low, high)
    assert rmin == MODULE.R_OUTER
    assert not (rmin < MODULE.R_OUTER)


def test_max_cap_control_corner_directions_are_contained():
    low = np.zeros(3)
    high = np.full(3, 3.)
    center = (low + high) / 2.
    cap = MODULE.cell_cap(center)
    for corner in MODULE.corners_of_box(low, high):
        norm = np.linalg.norm(corner)
        if norm == 0.:
            continue
        angle = np.arccos(np.clip(np.dot(corner, center) /
                                  (norm * np.linalg.norm(center)), -1., 1.))
        assert angle <= cap


def test_ray_box_reference_detects_partial_shell_ray():
    direction = np.ones((3, 1), dtype=float) / np.sqrt(3.)
    enter, leave, hits = MODULE.ray_box_intervals(
        direction, np.zeros(3), np.full(3, 3.))
    assert hits[0]
    assert enter[0] == 0.
    assert np.isclose(leave[0], 3. * np.sqrt(3.))
    assert min(leave[0], MODULE.R_OUTER) > max(enter[0], MODULE.R_INNER)
