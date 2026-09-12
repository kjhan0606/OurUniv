from types import SimpleNamespace

import numpy as np

from src.cf4_p2_screen import (
    _deep_merge,
    extract_central_arrays,
    extract_central_particles,
)


def test_extract_central_particles_reconstructs_periodic_positions_in_chunks():
    particles = SimpleNamespace(
        pmid=np.array([[2, 2, 2], [0, 0, 0], [3, 3, 3]], dtype=np.int16),
        disp=np.array([[0.1, -0.1, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        vel=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        conf=SimpleNamespace(
            cell_size=np.array([1.0, 1.0, 1.0]),
            box_size=np.array([4.0, 4.0, 4.0]),
        ),
    )
    pos, vel = extract_central_particles(
        particles,
        np.array([2.0, 2.0, 2.0]),
        0.25,
        velocity_unit=100.0,
        chunk_size=1,
    )
    np.testing.assert_allclose(pos, [[2.1, 1.9, 2.0]], atol=1e-6)
    np.testing.assert_allclose(vel, [[100.0, 200.0, 300.0]])


def test_deep_merge_preserves_frozen_nested_fields():
    base = {"screen": {"mesh_size": 576, "cut": 3.0}, "parent_seeds": [1]}
    merged = _deep_merge(base, {"screen": {"mesh_size": 512}, "parent_seeds": [2]})
    assert merged == {"screen": {"mesh_size": 512, "cut": 3.0}, "parent_seeds": [2]}


def test_extract_central_arrays_selects_matching_velocities():
    pos = np.array([[2.1, 1.9, 2.0], [3.0, 3.0, 3.0]], np.float32)
    vel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
    got_pos, got_vel = extract_central_arrays(
        pos, vel, np.array([2.0, 2.0, 2.0]), 0.25, velocity_unit=100.0)
    np.testing.assert_allclose(got_pos, pos[:1])
    np.testing.assert_allclose(got_vel, vel[:1] * 100.0)
