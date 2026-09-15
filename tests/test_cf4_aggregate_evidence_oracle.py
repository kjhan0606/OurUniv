import hashlib
import json
import math

import numpy as np

from cf4_aggregate_evidence_oracle import (
    AggregateEvidenceControllerOracle,
    AtlasBounds,
    ExactCovarianceCache,
    atlas_point_indices,
    canonical_axis,
    canonical_axis_offset,
    covariance_key,
    covariance_terms_for_keys,
    evaluate_log_z_from_atlases,
    extract_response_atlas,
    geometry_key,
    geometry_key_from_grid_axis,
    logmeanexp_parent,
    load_verified_atlas_manifest,
    lookup_response_atlas,
    parent_response_grid,
    points_from_geometry_key,
    response_atlas_bounds,
    target_vector,
    vectorized_log_evidence,
)
from cf4_lg_peak_cr import two_peak_points
from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    parent_mean_at_point_sets,
)


def test_geometry_key_uses_ties_to_even_and_unwrapped_q():
    dx = 2.0 / 3.0
    q = dx * np.asarray([0.5, 1.5, -2.5])
    key = geometry_key(q, [1.0, 0.0, 0.0], dx_mpc_h=dx)
    assert key[:3] == (288, 290, 286)
    shifted = geometry_key(
        q + np.asarray([576 * dx, 0.0, 0.0]),
        [1.0, 0.0, 0.0],
        dx_mpc_h=dx,
    )
    assert shifted == key


def test_axis_canonicalization_uses_lowest_maximum_index():
    axis = np.asarray([-1.0, 1.0, 0.0]) / math.sqrt(2.0)
    expected = -axis
    np.testing.assert_allclose(canonical_axis(axis), expected, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(canonical_axis(-axis), expected, rtol=0.0, atol=2e-16)
    np.testing.assert_array_equal(
        canonical_axis_offset(axis), canonical_axis_offset(-axis)
    )
    assert tuple(canonical_axis_offset(axis)) == (2, -2, 0)


def test_antipodal_axes_have_identical_keys_and_point_sets():
    midpoint = np.asarray([287, 280, 295])
    axis = np.asarray([0.21, -0.73, 0.65])
    left = geometry_key_from_grid_axis(midpoint, axis)
    right = geometry_key_from_grid_axis(midpoint, -axis)
    assert left == right
    np.testing.assert_array_equal(
        points_from_geometry_key(left), points_from_geometry_key(right)
    )


def test_canonical_key_points_match_legacy_points_up_to_centre_permutation():
    midpoint = np.asarray([283, 277, 301])
    axis = np.asarray([-0.8, 0.25, 0.54])
    key = geometry_key_from_grid_axis(midpoint, axis)
    actual = points_from_geometry_key(key)
    expected, _ = two_peak_points(576, midpoint, axis, 6, 2)
    actual_blocks = sorted(tuple(map(tuple, actual[i:i + 7])) for i in (0, 7))
    expected_blocks = sorted(tuple(map(tuple, expected[i:i + 7])) for i in (0, 7))
    assert actual_blocks == expected_blocks
    assert covariance_key(key)[:3] == tuple(np.mod(midpoint, 3))


def test_logmeanexp_parent_is_stable_with_parents_on_last_axis():
    values = np.asarray([
        [-1001.0, -1000.0, -1200.0],
        [-4.0, -4.0, -4.0],
    ])
    actual = logmeanexp_parent(values)
    expected = np.asarray([
        -1000.0 + math.log(1.0 + math.exp(-1.0) + math.exp(-200.0))
        - math.log(3.0),
        -4.0,
    ])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)


def test_frozen_response_atlas_bounds_have_five_cell_padding():
    bounds = response_atlas_bounds(
        [0.0, -6.0, 4.0], [3.0, 3.0, 3.0], dx_mpc_h=2.0 / 3.0
    )
    assert bounds.relative_min == (-45, -54, -39)
    assert bounds.relative_max == (45, 36, 51)
    assert bounds.padded_min == (-50, -59, -44)
    assert bounds.padded_max == (50, 41, 56)
    assert bounds.shape == (101, 101, 101)


def test_atlas_extract_lookup_is_periodic_and_rejects_outside_points():
    n = 12
    grid = np.arange(n**3, dtype=np.float64).reshape(n, n, n)
    bounds = AtlasBounds((-1, -1, -1), (1, 1, 1), (-2, -2, -2), (2, 2, 2))
    atlas = extract_response_atlas(grid, bounds)
    points = np.asarray([[4, 5, 6], [8, 7, 6]])
    actual = lookup_response_atlas(atlas, points, bounds, fine_n=n)
    np.testing.assert_array_equal(actual, grid[tuple(points.T)])
    indices, inside = atlas_point_indices(
        np.asarray([[3, 6, 6], [9, 6, 6]]), bounds, fine_n=n
    )
    assert indices.shape == (2, 3)
    assert inside.tolist() == [False, False]
    with np.testing.assert_raises(KeyError):
        lookup_response_atlas(
            atlas, np.asarray([[3, 6, 6]]), bounds, fine_n=n
        )


def test_parent_response_grid_matches_existing_exact_point_evaluator():
    rng = np.random.default_rng(73)
    coarse = rng.normal(size=(4, 4, 4))
    kernel = rng.normal(size=(12, 12, 12))
    filter_full = np.fft.fftn(kernel, norm="ortho")
    points = rng.integers(0, 12, size=(37, 3))
    grid = parent_response_grid(coarse, filter_full)
    expected = parent_mean_at_point_sets(coarse, filter_full, [points])[0]
    np.testing.assert_allclose(
        grid[tuple(points.T)], expected, rtol=0.0, atol=0.0
    )


def test_covariance_terms_reuse_exact_phase_and_offset_key():
    rng = np.random.default_rng(82)
    kernel = rng.normal(size=(12, 12, 12))
    filter_full = np.fft.fftn(kernel, norm="ortho")
    first = (6, 6, 6, 3, 0, 0)
    second = (9, 6, 6, 3, 0, 0)
    cholesky, logdet, diagnostics = covariance_terms_for_keys(
        filter_full, [second, first], coarse_n=4, fine_n=12
    )
    assert diagnostics["geometry_key_count"] == 2
    assert diagnostics["unique_covariance_key_count"] == 1
    np.testing.assert_array_equal(cholesky[0], cholesky[1])
    np.testing.assert_array_equal(logdet[0], logdet[1])


def test_exact_covariance_cache_reuses_terms_across_calls():
    rng = np.random.default_rng(182)
    filter_full = np.fft.fftn(rng.normal(size=(12, 12, 12)), norm="ortho")
    cache = ExactCovarianceCache(filter_full, coarse_n=4, fine_n=12)
    first = (6, 6, 6, 3, 0, 0)
    equivalent = (9, 6, 6, 3, 0, 0)
    first_cholesky, first_logdet, first_diagnostic = cache.terms([first])
    second_cholesky, second_logdet, second_diagnostic = cache.terms([equivalent])
    assert first_diagnostic["new_covariance_key_count"] == 1
    assert second_diagnostic["new_covariance_key_count"] == 0
    assert cache.evaluation_batches == 1
    assert cache.evaluated_covariance_keys == 1
    np.testing.assert_array_equal(first_cholesky, second_cholesky)
    np.testing.assert_array_equal(first_logdet, second_logdet)


def test_atlas_log_z_matches_existing_exact_evidence(tmp_path):
    rng = np.random.default_rng(281)
    kernel = rng.normal(size=(12, 12, 12))
    filter_full = np.fft.fftn(kernel, norm="ortho")
    bounds = AtlasBounds(
        relative_min=(-5, -5, -5),
        relative_max=(6, 6, 6),
        padded_min=(-5, -5, -5),
        padded_max=(6, 6, 6),
    )
    entries = []
    coarse_fields = []
    for parent, seed in enumerate((3193, 3194)):
        coarse = rng.normal(size=(4, 4, 4)).astype(np.float32)
        coarse_fields.append(coarse)
        response = parent_response_grid(coarse, filter_full)
        atlas = extract_response_atlas(response, bounds)
        atlas_path = tmp_path / f"atlas_{seed}.npy"
        np.save(atlas_path, atlas, allow_pickle=False)
        parent_path = tmp_path / f"parent_{seed}.npz"
        np.savez(parent_path, sample_seed=seed, s_out=coarse)
        entries.append({
            "seed": seed,
            "atlas": str(atlas_path),
            "atlas_sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
            "parent_field": str(parent_path),
            "parent_field_sha256": hashlib.sha256(
                parent_path.read_bytes()
            ).hexdigest(),
        })
    keys = [
        (6, 6, 6, 3, 0, 0),
        (7, 5, 6, 2, -2, 0),
        (5, 7, 6, 2, 0, -2),
    ]
    targets = target_vector(1.2, 0.3)
    actual_keys, actual, diagnostics = evaluate_log_z_from_atlases(
        keys,
        entries,
        bounds,
        filter_full,
        targets,
        coarse_n=4,
        fine_n=12,
        sigma_delta=0.25,
        covariance_cache=ExactCovarianceCache(
            filter_full, coarse_n=4, fine_n=12
        ),
    )
    assert actual_keys == sorted(keys)
    assert diagnostics["outside_atlas_key_count"] == 0
    points = [points_from_geometry_key(key, fine_n=12) for key in actual_keys]
    covariance, _ = covariance_for_point_sets(filter_full, 4, points)
    observation = covariance + np.eye(14)[None] * 0.25**2
    cholesky = np.linalg.cholesky(observation)
    logdet = 2.0 * np.sum(
        np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1
    )
    expected = np.empty_like(actual)
    for parent, coarse in enumerate(coarse_fields):
        means = np.asarray(parent_mean_at_point_sets(coarse, filter_full, points))
        expected[:, parent] = vectorized_log_evidence(
            means, np.broadcast_to(targets, means.shape), cholesky, logdet
        )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)


def test_outside_atlas_slow_path_is_exact_and_requires_parent_hash(tmp_path):
    rng = np.random.default_rng(381)
    filter_full = np.fft.fftn(
        rng.normal(size=(12, 12, 12)), norm="ortho"
    )
    coarse = rng.normal(size=(4, 4, 4)).astype(np.float32)
    parent_path = tmp_path / "parent.npz"
    np.savez(parent_path, sample_seed=3193, s_out=coarse)
    bounds = AtlasBounds(
        relative_min=(-1, -1, -1),
        relative_max=(1, 1, 1),
        padded_min=(-2, -2, -2),
        padded_max=(2, 2, 2),
    )
    response = parent_response_grid(coarse, filter_full)
    atlas = extract_response_atlas(response, bounds)
    atlas_path = tmp_path / "atlas.npy"
    np.save(atlas_path, atlas, allow_pickle=False)
    entry = {
        "seed": 3193,
        "atlas": str(atlas_path),
        "atlas_sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
        "parent_field": str(parent_path),
        "parent_field_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    }
    key = (6, 6, 6, 3, 0, 0)
    target = target_vector(1.1, 0.2)
    _, actual, diagnostic = evaluate_log_z_from_atlases(
        [key],
        [entry],
        bounds,
        filter_full,
        target,
        coarse_n=4,
        fine_n=12,
        covariance_cache=ExactCovarianceCache(
            filter_full, coarse_n=4, fine_n=12
        ),
    )
    assert diagnostic["outside_atlas_key_count"] == 1
    points = [points_from_geometry_key(key, fine_n=12)]
    means = np.asarray(parent_mean_at_point_sets(coarse, filter_full, points))
    covariance, _ = covariance_for_point_sets(filter_full, 4, points)
    cholesky = np.linalg.cholesky(covariance + np.eye(14)[None] * 0.25**2)
    logdet = 2.0 * np.sum(
        np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1
    )
    expected = vectorized_log_evidence(
        means, target[None], cholesky, logdet
    )
    np.testing.assert_allclose(actual[:, 0], expected, rtol=0.0, atol=2e-12)
    missing_hash = dict(entry)
    missing_hash.pop("parent_field_sha256")
    with np.testing.assert_raises_regex(RuntimeError, "mandatory parent hash"):
        evaluate_log_z_from_atlases(
            [key],
            [missing_hash],
            bounds,
            filter_full,
            target,
            coarse_n=4,
            fine_n=12,
            covariance_cache=ExactCovarianceCache(
                filter_full, coarse_n=4, fine_n=12
            ),
        )


def test_controller_requires_four_exact_terminal_histories_then_seals():
    calls = []

    def evaluator(keys):
        calls.append(keys)
        values = np.asarray([
            np.arange(256, dtype=np.float64) + sum(key) for key in keys
        ])
        return keys, values

    oracle = AggregateEvidenceControllerOracle(evaluator)
    q = np.asarray([[0.0, -6.0, 4.0], [0.0, -6.0, 4.0]])
    axis = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    keys, aggregate = oracle.evaluate(q, axis)
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert keys.shape == (2, 6)
    assert aggregate[0] == aggregate[1]
    oracle.evaluate(q, axis)
    assert len(calls) == 1
    terminal = np.tile(keys[:1], (2048, 1))
    with np.testing.assert_raises_regex(RuntimeError, "all four"):
        oracle.seal_terminal_histories()
    with np.testing.assert_raises_regex(ValueError, "four frozen seeds"):
        oracle.register_terminal_history(17, terminal)
    with np.testing.assert_raises_regex(TypeError, "exact int16"):
        oracle.register_terminal_history(2026082301, terminal.astype(float) + 0.5)
    overflow = terminal.astype(np.int64)
    overflow[0, 0] = 40000
    with np.testing.assert_raises_regex(TypeError, "exact int16"):
        oracle.register_terminal_history(2026082301, overflow)
    outside = terminal.copy()
    outside[0, 0] = 576
    with np.testing.assert_raises_regex(ValueError, "outside"):
        oracle.register_terminal_history(2026082301, outside)
    noncanonical = terminal.copy()
    noncanonical[0, 3:] *= -1
    with np.testing.assert_raises_regex(ValueError, "not canonical"):
        oracle.register_terminal_history(2026082301, noncanonical)
    oracle.register_terminal_history(2026082301, terminal)
    with np.testing.assert_raises_regex(RuntimeError, "all histories are sealed"):
        oracle.terminal_parent_log_z(2026082301, terminal)
    with np.testing.assert_raises_regex(RuntimeError, "duplicate"):
        oracle.register_terminal_history(2026082301, terminal)
    for seed in (2026082302, 2026082303, 2026082304):
        oracle.register_terminal_history(seed, terminal)
    oracle.seal_terminal_histories()
    with np.testing.assert_raises_regex(RuntimeError, "closed after terminal seal"):
        oracle.evaluate(q, axis)
    with np.testing.assert_raises_regex(TypeError, "exact int16"):
        oracle.terminal_parent_log_z(2026082301, terminal.astype(float) + 0.9)
    with np.testing.assert_raises_regex(RuntimeError, "registered history"):
        wrong = terminal.copy()
        wrong[0, 0] = (int(wrong[0, 0]) + 1) % 576
        oracle.terminal_parent_log_z(2026082301, wrong)
    parent = oracle.terminal_parent_log_z(2026082301, terminal)
    assert parent.shape == (2048, 256)


def test_production_manifest_loader_rejects_hash_status_and_parent_contract(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {
        "schema": "ouruniv-cf4-parent-response-atlas-manifest-v1",
        "status": "complete_exact_parent_response_atlas",
        "parent_count": 255,
        "dtype": "float64",
    }
    path.write_text(json.dumps(manifest))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.testing.assert_raises_regex(RuntimeError, "manifest hash"):
        load_verified_atlas_manifest(path, "0" * 64)
    with np.testing.assert_raises_regex(RuntimeError, "parent count"):
        load_verified_atlas_manifest(path, sha)
    manifest.update({
        "parent_count": 256,
        "bounds": {
            "relative_min": [-45, -54, -39],
            "relative_max": [45, 36, 51],
            "padded_min": [-50, -59, -44],
            "padded_max": [50, 41, 56],
        },
        "entries": [],
    })
    path.write_text(json.dumps(manifest))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.testing.assert_raises_regex(RuntimeError, "seeds"):
        load_verified_atlas_manifest(path, sha)
