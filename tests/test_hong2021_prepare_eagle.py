from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_prepare_eagle import (
    BOX_MPC_H,
    CELL_MPC_H,
    CUBE_GRID,
    REGULAR_GRID,
    accumulate_coordinates,
    farthest_point_subset,
    galaxy_input_grid,
    geometry_safe_mask,
    snapshot_members,
)


def test_snapshot_members_are_sorted_numerically() -> None:
    members = []
    for index in reversed(range(256)):
        member = tarfile.TarInfo(
            "RefL0100N1504/snapshot_028_z000p000/"
            f"snap_028_z000p000.{index}.hdf5"
        )
        member.type = tarfile.REGTYPE
        members.append(member)
    result = snapshot_members(members)
    assert result[0].name.endswith(".0.hdf5")
    assert result[-1].name.endswith(".255.hdf5")


def test_geometry_safe_mask_requires_an_exact_full_cube() -> None:
    lower_safe_center = np.array([(CUBE_GRID // 2) * CELL_MPC_H] * 3)
    upper_safe_cell = REGULAR_GRID - CUBE_GRID // 2 - 1
    upper_safe_center = np.array([(upper_safe_cell + 0.5) * CELL_MPC_H] * 3)
    unsafe_lower = lower_safe_center.copy()
    unsafe_lower[0] -= CELL_MPC_H
    unsafe_upper = upper_safe_center.copy()
    unsafe_upper[2] += 2 * CELL_MPC_H
    mask = geometry_safe_mask(
        np.vstack([lower_safe_center, upper_safe_center, unsafe_lower, unsafe_upper])
    )
    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_accumulate_coordinates_excludes_only_regular_fringe() -> None:
    counts = np.zeros((REGULAR_GRID,) * 3, dtype=np.uint64)
    coordinates = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [CELL_MPC_H + 0.01, 0.01, 0.01],
            [REGULAR_GRID * CELL_MPC_H + 0.01, 0.01, 0.01],
            [BOX_MPC_H - 1.0e-6, BOX_MPC_H - 1.0e-6, BOX_MPC_H - 1.0e-6],
        ]
    )
    read, binned = accumulate_coordinates(counts, coordinates)
    assert read == 5
    assert binned == 3
    assert counts.sum() == 3
    assert counts[0, 0, 0] == 2
    assert counts[1, 0, 0] == 1


def test_farthest_point_subset_is_deterministic_and_unique() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [9.0, 0.0, 0.0]]
    )
    identifiers = np.array([20, 10, 40, 30])
    first = farthest_point_subset(positions, identifiers, 3)
    second = farthest_point_subset(positions, identifiers, 3)
    np.testing.assert_array_equal(first, second)
    assert first[0] == 1
    assert len(np.unique(first)) == 3


def test_galaxy_input_uses_center_relative_radial_velocity_and_mask() -> None:
    center = np.array([20.0, 20.0, 20.0])
    center_velocity = np.array([10.0, 20.0, 30.0])
    origin = np.floor(center / CELL_MPC_H).astype(np.int64) - CUBE_GRID // 2
    positions = np.array(
        [
            center,
            center + [0.0, 0.0, CELL_MPC_H],
            center + [CELL_MPC_H, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [center_velocity, center_velocity + [0.0, 0.0, 40.0], center_velocity + [50.0, 0.0, 0.0]]
    )
    count, velocity, kept = galaxy_input_grid(
        center, center_velocity, origin, positions, velocities
    )
    # The center is removed (r=0); the x-axis object is removed by |b|<10.
    assert kept == 1
    assert count.sum() == 1
    np.testing.assert_allclose(velocity[count > 0], 40.0)
