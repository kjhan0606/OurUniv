import hashlib
from pathlib import Path

import numpy as np

from hong2021_v33_kinematic_data import (
    PROGRAM_SHA256,
    galaxy_input_grid_with_dispersion,
)


REPO = Path(__file__).resolve().parents[1]


def test_v33_program_hash_and_observational_boundary():
    path = REPO / "config/hong2021_v33_intrinsic_velocity_moment_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"posthoc_Ak": false' in text
    assert '"Astrid_access": "forbidden"' in text
    assert '"historical_EAGLE_access": "forbidden"' in text
    assert "no observational error is fabricated in V33" in text


def test_velocity_moments_are_unbiased_and_zero_below_two_objects():
    position = np.asarray(
        [
            [1.00, 0.00, 1.00],
            [1.01, 0.00, 1.01],
            [3.00, 0.00, 3.00],
        ],
        dtype=np.float64,
    )
    radial_hat = position / np.linalg.norm(position, axis=1)[:, None]
    radial_speed = np.asarray([0.0, 10.0, 23.0])
    velocity = radial_hat * radial_speed[:, None]
    cell = np.asarray([[5, 5, 5], [5, 5, 5], [8, 8, 8]])
    count, mean, dispersion, kept = galaxy_input_grid_with_dispersion(
        np.zeros(3),
        np.zeros(3),
        np.zeros(3, dtype=np.int64),
        position,
        velocity,
        cell,
        simulation_box_mpc_h=20.0,
        full_grid=64,
    )
    assert kept == 3
    assert count[5, 5, 5] == 2
    assert mean[5, 5, 5] == 5
    assert np.isclose(dispersion[5, 5, 5], np.sqrt(50.0))
    assert count[8, 8, 8] == 1
    assert mean[8, 8, 8] == 23
    assert dispersion[8, 8, 8] == 0
    assert np.count_nonzero(dispersion) == 1


def test_latitude_mask_and_periodic_cube_are_preserved():
    position = np.asarray([[1.0, 0.0, 0.0], [9.8, 0.0, 2.0]])
    velocity = np.zeros_like(position)
    cell = np.asarray([[1, 1, 1], [63, 1, 1]])
    count, mean, dispersion, kept = galaxy_input_grid_with_dispersion(
        np.zeros(3),
        np.zeros(3),
        np.asarray([-1, 0, 0]),
        position,
        velocity,
        cell,
        simulation_box_mpc_h=20.0,
        full_grid=64,
    )
    assert kept == 1
    assert count.sum() == 1
    assert np.count_nonzero(mean) == 0
    assert np.count_nonzero(dispersion) == 0
