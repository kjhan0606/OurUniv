from types import SimpleNamespace

import numpy as np

from src.cf4_p2_trace import (
    select_candidate,
    trace_periodic_sphere,
    trace_position_array,
)
from src.cf4_make_ic import (
    embed_ic,
    embed_ic_projected,
    fourier_resample_white_field,
)


def test_trace_periodic_sphere_returns_stable_ids_and_wraps():
    particles = SimpleNamespace(
        pmid=np.array([[0, 0, 0], [2, 2, 2], [3, 3, 3]], dtype=np.int16),
        disp=np.array([[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        conf=SimpleNamespace(
            cell_size=np.array([1.0, 1.0, 1.0]),
            box_size=np.array([4.0, 4.0, 4.0]),
        ),
    )
    ids, lagrangian, final = trace_periodic_sphere(
        particles, np.array([0.0, 0.0, 0.0]), 0.2, chunk_size=1)
    np.testing.assert_array_equal(ids, [1])
    np.testing.assert_allclose(lagrangian, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(final, [[3.9, 0.0, 0.0]], atol=1e-6)


def test_select_candidate_uses_lowest_frozen_ranking_score():
    result = {
        "results": [
            {"parent_seed": 3, "small_scale_seed": 1, "screen_pass": True,
             "best_pair": {"ranking_score": 2.0}},
            {"parent_seed": 3, "small_scale_seed": 2, "screen_pass": True,
             "best_pair": {"ranking_score": 1.0}},
            {"parent_seed": 3, "small_scale_seed": 3, "screen_pass": False,
             "best_pair": None},
        ]
    }
    assert select_candidate(result, None, None)["small_scale_seed"] == 2
    assert select_candidate(result, 3, 1)["small_scale_seed"] == 1


def test_trace_position_array_reconstructs_regular_grid_ids():
    final = np.array(
        [[3.9, 0.0, 0.0], [0.0, 0.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=np.float32,
    )
    ids, lagrangian, selected = trace_position_array(
        final, np.zeros(3), 0.2, mesh_size=2, spacing=2.0,
        box_size=4.0, chunk_size=1)
    np.testing.assert_array_equal(ids, [1])
    np.testing.assert_allclose(lagrangian, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(selected, [[3.9, 0.0, 0.0]])


def test_fourier_resample_preserves_shared_non_nyquist_modes():
    rng = np.random.default_rng(7)
    source = rng.standard_normal((12, 12, 12))
    target = fourier_resample_white_field(source, 8)
    source_fft = np.fft.rfftn(source)
    target_fft = np.fft.rfftn(target)
    scale = (8.0 / 12.0) ** 1.5
    # Interior modes avoid the real-FFT Nyquist convention on each axis.
    np.testing.assert_allclose(target_fft[1:4, 1:4, 1:4],
                               source_fft[1:4, 1:4, 1:4] * scale,
                               rtol=2e-6, atol=2e-6)


def test_direct_projected_embedding_matches_two_step_definition():
    rng = np.random.default_rng(5)
    coarse = rng.standard_normal((4, 4, 4))
    two_step = fourier_resample_white_field(embed_ic(coarse, 12, 91), 8)
    direct = embed_ic_projected(coarse, 12, 8, 91)
    np.testing.assert_allclose(direct, two_step, rtol=2e-6, atol=2e-6)
