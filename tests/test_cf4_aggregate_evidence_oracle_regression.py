import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

import cf4_aggregate_evidence_oracle as oracle_module
import cf4_aggregate_evidence_oracle_regression as regression_module
from cf4_aggregate_evidence_oracle import (
    AtlasBounds,
    ExactCovarianceCache,
    covariance_key,
    evaluate_log_z_from_atlases,
    extract_response_atlas,
    geometry_key_from_grid_axis,
    logmeanexp_parent,
    parent_response_grid,
    points_from_geometry_key,
    response_atlas_bounds,
    target_vector,
    vectorized_log_evidence,
)
from cf4_aggregate_evidence_oracle_regression import (
    FROZEN_DESIGN,
    FROZEN_DESIGN_SHA256,
    array_sha256,
    atomic_json,
    atomic_npz,
    dense_phase_control,
    deterministic_atlas_keys,
    explicit_dense_covariance,
    historical_bank_regression,
    _gate_results,
    reassemble_parent_blocks,
    reconstruct_historical_keys,
    sha256_file,
    unique_keys_and_inverse,
    validate_dense_phase_coverage,
    validate_frozen_design,
    validate_program,
)


def test_frozen_design_sha_lineage_and_authorization_are_exact():
    design = json.loads(FROZEN_DESIGN.read_text())
    assert sha256_file(FROZEN_DESIGN) == FROZEN_DESIGN_SHA256
    validate_frozen_design(design, FROZEN_DESIGN)
    assert design["authorization"]["implementation_and_tests_authorized"] is True
    for key in (
        "regression_execution_authorized",
        "production_SMC_authorized",
        "conditional_field_bank_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
    ):
        assert design["authorization"][key] is False


def test_production_selection_has_bitwise_golden_keys_and_candidate_indices():
    bounds = response_atlas_bounds([0.0, -6.0, 4.0], [3.0, 3.0, 3.0])
    first = deterministic_atlas_keys(bounds)
    second = deterministic_atlas_keys(bounds)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert array_sha256(
        first["inside_keys"], first["inside_candidate_index"]
    ) == "d91f20f27fb2f43df781f79b2e9f89bd9c29f802a41a107ca69ab5d375955b13"
    assert array_sha256(
        first["outside_keys"], first["outside_candidate_index"]
    ) == "3b5ab76f91b008a8671b33a46f33f36f4788251bebe53e35ce784cabc031aefc"
    assert first["inside_keys"].dtype == np.int16
    assert first["outside_keys"].dtype == np.int16
    assert first["inside_candidate_index"].dtype == np.int32
    assert first["outside_candidate_index"].dtype == np.int32
    assert np.all(first["inside_candidate_index"] <= 1023)
    assert np.all(first["outside_candidate_index"] <= 63)


def _synthetic_entries(tmp_path, filter_full, bounds):
    rng = np.random.default_rng(8401)
    entries = []
    coarse_fields = []
    for seed in (3193, 3429):
        coarse = rng.normal(size=(4, 4, 4)).astype(np.float32)
        coarse_fields.append(coarse)
        parent_path = tmp_path / f"parent_{seed}.npz"
        np.savez(parent_path, sample_seed=seed, s_out=coarse)
        response = parent_response_grid(coarse, filter_full)
        atlas = extract_response_atlas(response, bounds)
        atlas_path = tmp_path / f"atlas_{seed}.npy"
        np.save(atlas_path, atlas, allow_pickle=False)
        entries.append({
            "seed": seed,
            "parent_field": str(parent_path),
            "parent_field_sha256": sha256_file(parent_path),
            "atlas": str(atlas_path),
            "atlas_sha256": sha256_file(atlas_path),
            "shape": list(bounds.shape),
            "dtype": "float64",
        })
    return entries, coarse_fields


def test_n12_inside_and_outside_slow_paths_match_full_response(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(8402)
    filter_full = np.fft.fftn(rng.normal(size=(12, 12, 12)), norm="ortho")
    bounds = AtlasBounds(
        relative_min=(-1, -1, -1),
        relative_max=(1, 1, 1),
        padded_min=(-5, -5, -5),
        padded_max=(5, 5, 5),
    )
    entries, coarse_fields = _synthetic_entries(tmp_path, filter_full, bounds)
    inside = (6, 6, 6, 3, 0, 0)
    outside = (3, 6, 6, 3, 0, 0)
    target = target_vector(1.2, 0.3)
    cache = ExactCovarianceCache(filter_full, coarse_n=4, fine_n=12)
    slow_path_calls = []
    original_parent_response = oracle_module.parent_response_grid

    def counted_parent_response(*args, **kwargs):
        slow_path_calls.append(1)
        return original_parent_response(*args, **kwargs)

    monkeypatch.setattr(
        oracle_module, "parent_response_grid", counted_parent_response
    )
    _, inside_log_z, inside_diagnostic = evaluate_log_z_from_atlases(
        [inside], entries, bounds, filter_full, target,
        coarse_n=4, fine_n=12, covariance_cache=cache,
    )
    assert len(slow_path_calls) == 0
    _, outside_log_z, outside_diagnostic = evaluate_log_z_from_atlases(
        [outside], entries, bounds, filter_full, target,
        coarse_n=4, fine_n=12, covariance_cache=cache,
    )
    assert len(slow_path_calls) == 2
    assert inside_diagnostic["inside_atlas_key_count"] == 1
    assert inside_diagnostic["outside_atlas_key_count"] == 0
    assert outside_diagnostic["inside_atlas_key_count"] == 0
    assert outside_diagnostic["outside_atlas_key_count"] == 1

    for key, actual in ((inside, inside_log_z), (outside, outside_log_z)):
        cholesky, logdet, _ = cache.terms([key])
        points = points_from_geometry_key(key, fine_n=12)
        direct = np.empty((1, 2), dtype=np.float64)
        for parent, coarse in enumerate(coarse_fields):
            response = parent_response_grid(coarse, filter_full)
            mean = response[tuple(points.T)][None]
            direct[:, parent] = vectorized_log_evidence(
                mean, target[None], cholesky, logdet
            )
        np.testing.assert_allclose(actual, direct, rtol=0.0, atol=2e-12)

    missing_parent_hash = dict(entries[0])
    missing_parent_hash.pop("parent_field_sha256")
    with np.testing.assert_raises_regex(RuntimeError, "mandatory parent hash"):
        evaluate_log_z_from_atlases(
            [outside], [missing_parent_hash], bounds, filter_full, target,
            coarse_n=4, fine_n=12,
            covariance_cache=ExactCovarianceCache(
                filter_full, coarse_n=4, fine_n=12
            ),
        )


def test_n12_explicit_all_27_phase_matrices_match_cache(tmp_path):
    rng = np.random.default_rng(8403)
    filter_full = np.fft.fftn(rng.normal(size=(12, 12, 12)), norm="ortho")
    bounds = AtlasBounds(
        relative_min=(-5, -5, -5),
        relative_max=(5, 5, 5),
        padded_min=(-5, -5, -5),
        padded_max=(5, 5, 5),
    )
    entries, _ = _synthetic_entries(tmp_path, filter_full, bounds)
    arrays, metrics = dense_phase_control(
        filter_full,
        target_vector(1.2, 0.3),
        entries,
        coarse_n=4,
        fine_n=12,
    )
    assert metrics["phase_count"] == 27
    assert metrics["unique_covariance_key_count"] == 27
    assert metrics["response_grids_held_simultaneously"] == 1
    assert metrics["signal_covariance_max_abs_difference"] <= 1e-12
    assert metrics["signal_covariance_relative_Frobenius_difference"] <= 1e-10
    assert metrics["cholesky_max_abs_difference"] <= 1e-12
    assert metrics["logdet_max_abs_difference"] <= 1e-10
    assert metrics["normalized_log_Z_max_abs_difference"] <= 1e-10
    assert arrays["dense_phase"].shape == (27, 3)
    assert arrays["dense_signal_direct"].shape == (27, 14, 14)
    expected_phase = np.asarray(
        list(np.ndindex(3, 3, 3)), dtype=np.int8
    )
    expected_keys = np.column_stack((
        6 + expected_phase.astype(np.int16),
        np.tile(np.asarray([3, 0, 0], dtype=np.int16), (27, 1)),
    ))
    np.testing.assert_array_equal(arrays["dense_phase"], expected_phase)
    np.testing.assert_array_equal(arrays["dense_keys"], expected_keys)


def test_dense_phase_coverage_rejects_ratio_missing_and_duplicate_assignment():
    phases = list(np.ndindex(3, 3, 3))
    assignment = np.ones((27, 14, 14), dtype=np.int16)
    validate_dense_phase_coverage(phases, assignment, ratio=3)
    with np.testing.assert_raises_regex(RuntimeError, "ratio three"):
        validate_dense_phase_coverage(phases, assignment, ratio=2)
    with np.testing.assert_raises_regex(RuntimeError, "set or lexicographic"):
        validate_dense_phase_coverage(phases[:-1], assignment, ratio=3)
    with np.testing.assert_raises_regex(RuntimeError, "exactly once"):
        validate_dense_phase_coverage(
            phases, np.ones((26, 14, 14), dtype=np.int16), ratio=3
        )
    duplicate = assignment.copy()
    duplicate[0, 0, 0] = 2
    with np.testing.assert_raises_regex(RuntimeError, "exactly once"):
        validate_dense_phase_coverage(phases, duplicate, ratio=3)


def test_dense_phase_rejects_finite_response_with_nan_diagnostic(monkeypatch):
    filter_full = np.ones((12, 12, 12), dtype=np.complex128)

    def nan_diagnostic(_filter, _coarse_n, _phase):
        return np.zeros((12, 12, 12), dtype=np.float64), {
            "imaginary_relative_RMS": np.nan,
            "maximum_absolute_imaginary": 0.0,
        }

    monkeypatch.setattr(
        regression_module,
        "_phase_response_grid_with_diagnostics",
        nan_diagnostic,
    )
    phases = np.asarray(list(np.ndindex(3, 3, 3)), dtype=np.int16)
    keys = np.column_stack((
        6 + phases,
        np.tile(np.asarray([3, 0, 0], dtype=np.int16), (27, 1)),
    ))
    with np.testing.assert_raises_regex(RuntimeError, "nonfinite"):
        explicit_dense_covariance(filter_full, 4, keys)


def test_reconstruct_keys_accepts_antipodal_peak_block_permutation():
    midpoint = np.asarray([[288, 288, 288], [289, 287, 290]], dtype=np.int16)
    axis = np.asarray([[1.0, 0.0, 0.0], [-0.2, -0.9, 0.3]])
    offset = (midpoint.astype(np.float64) - 288.0) * (2.0 / 3.0)
    keys = np.asarray([
        geometry_key_from_grid_axis(q, a) for q, a in zip(midpoint, axis)
    ], dtype=np.int16)
    points = np.stack([
        points_from_geometry_key(key) for key in keys
    ]).astype(np.int16)
    points[1] = np.concatenate((points[1, 7:], points[1, :7]), axis=0)
    kinds = np.tile(
        np.asarray([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        (2, 1),
    )
    actual = reconstruct_historical_keys(offset, axis, midpoint, points, kinds)
    np.testing.assert_array_equal(actual, keys)
    fractional_midpoint = midpoint.astype(np.float64)
    fractional_midpoint[0, 0] += 0.5
    with np.testing.assert_raises_regex(RuntimeError, "not aligned"):
        reconstruct_historical_keys(
            offset, axis, fractional_midpoint, points, kinds
        )
    nonfinite_axis = axis.copy()
    nonfinite_axis[0, 0] = np.nan
    with np.testing.assert_raises_regex(RuntimeError, "nonfinite"):
        reconstruct_historical_keys(
            offset, nonfinite_axis, midpoint, points, kinds
        )


class _FakeCovarianceCache:
    def __init__(self):
        self.evaluated_covariance_keys = 0
        self._seen = set()

    @staticmethod
    def _logdet(key):
        return float(sum(key)) * 0.01

    def terms(self, keys):
        unique = sorted(set(tuple(map(int, row)) for row in keys))
        for key in unique:
            self._seen.add(covariance_key(key))
        self.evaluated_covariance_keys = len(self._seen)
        return (
            np.tile(np.eye(14), (len(unique), 1, 1)),
            np.asarray([self._logdet(key) for key in unique]),
            {},
        )


class _FakeEvaluator:
    def __init__(self):
        self.covariance_cache = _FakeCovarianceCache()

    def __call__(self, keys):
        unique = sorted(set(tuple(map(int, row)) for row in keys))
        self.covariance_cache.terms(unique)
        parent = np.arange(256, dtype=np.float64)
        values = np.stack([
            0.1 * sum(key) + 0.001 * parent for key in unique
        ])
        return unique, values


def test_historical_duplicate_inverse_and_parent_orientation_are_exact(tmp_path):
    midpoint = np.asarray([
        [288, 288, 288],
        [289, 288, 288],
        [288, 288, 288],
        [290, 287, 289],
    ], dtype=np.int16)
    axis = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.2, -0.8, 0.5],
    ])
    offset = (midpoint.astype(np.float64) - 288.0) * (2.0 / 3.0)
    keys = np.asarray([
        geometry_key_from_grid_axis(q, a) for q, a in zip(midpoint, axis)
    ], dtype=np.int16)
    points = np.stack([
        points_from_geometry_key(key) for key in keys
    ]).astype(np.int16)
    points[2] = np.concatenate((points[2, 7:], points[2, :7]), axis=0)
    kinds = np.tile(
        np.asarray([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        (4, 1),
    )
    evaluator = _FakeEvaluator()
    unique, inverse = unique_keys_and_inverse(keys)
    unique_values = evaluator(unique)[1]
    historical_log_z = unique_values[inverse].T
    historical_log_z_bar = logmeanexp_parent(unique_values[inverse])
    unique_logdet = np.asarray([
        evaluator.covariance_cache._logdet(tuple(map(int, row))) for row in unique
    ])
    historical_logdet = unique_logdet[inverse]
    path = tmp_path / "historical.npz"
    np.savez(
        path,
        parent_seed=np.arange(3193, 3449, dtype=np.int32),
        log_Z_peak=historical_log_z,
        midpoint_offset_mpc_h=offset,
        axis=axis,
        midpoint_grid=midpoint,
        points=points,
        point_kinds=kinds,
        logmean_parent_Z_by_geometry=historical_log_z_bar,
        covariance_log_determinant=historical_logdet,
    )
    arrays, metrics = historical_bank_regression(
        "bank_test",
        path,
        evaluator,
        draw_count=4,
        unique_geometry_count=len(unique),
        unique_covariance_count=len({covariance_key(row) for row in unique}),
        batch_size=2,
    )
    np.testing.assert_array_equal(arrays["bank_test_keys"], keys)
    np.testing.assert_array_equal(arrays["bank_test_new_log_Z"], historical_log_z)
    assert metrics["historical_log_Z_max_abs_difference"] == 0.0
    assert metrics["historical_log_Z_bar_max_abs_difference"] == 0.0
    assert metrics["historical_covariance_logdet_max_abs_difference"] == 0.0
    assert metrics["cache"]["second_evaluation_bitwise_identical"] is True
    assert metrics["cache"]["new_covariance_keys_second_evaluation"] == 0

    wrong_dtype = tmp_path / "historical_wrong_dtype.npz"
    np.savez(
        wrong_dtype,
        parent_seed=np.arange(3193, 3449, dtype=np.int32),
        log_Z_peak=historical_log_z,
        midpoint_offset_mpc_h=offset,
        axis=axis,
        midpoint_grid=midpoint,
        points=points,
        point_kinds=kinds,
        logmean_parent_Z_by_geometry=historical_log_z_bar.astype(np.int64),
        covariance_log_determinant=historical_logdet,
    )
    with np.testing.assert_raises_regex(RuntimeError, "dtype or shape"):
        historical_bank_regression(
            "bank_test",
            wrong_dtype,
            _FakeEvaluator(),
            draw_count=4,
            unique_geometry_count=len(unique),
            unique_covariance_count=len({covariance_key(row) for row in unique}),
            batch_size=2,
        )


def test_nonfinite_metric_has_priority_over_scientific_mismatch():
    design = json.loads(FROZEN_DESIGN.read_text())
    dense_gate = design["dense_27_phase_control"]["gates"]
    dense = {
        "phase_count": 27,
        "phases": [list(phase) for phase in np.ndindex(3, 3, 3)],
        "unique_covariance_key_count": 27,
        "response_grids_held_simultaneously": 1,
        "maximum_phase_response_imaginary_relative_RMS": 0.0,
        "maximum_phase_response_absolute_imaginary": 0.0,
        "maximum_pre_symmetrization_asymmetry": 0.0,
        "signal_covariance_max_abs_difference": 0.0,
        "signal_covariance_relative_Frobenius_difference": 0.0,
        "cholesky_max_abs_difference": 0.0,
        "logdet_max_abs_difference": np.nan,
        "normalized_log_Z_max_abs_difference": 0.0,
    }
    assert dense_gate["logdet_max_abs_difference"] == 1e-10
    atlas = {
        "inside_diagnostics": {
            "inside_atlas_key_count": 1024,
            "outside_atlas_key_count": 0,
        },
        "outside_diagnostics": {
            "inside_atlas_key_count": 0,
            "outside_atlas_key_count": 64,
        },
        "outside_slow_path_full_response_parent_evaluations": 256,
        "direct_full_response_parent_evaluations": 256,
        "inside_log_Z_max_abs_difference": 0.0,
        "outside_log_Z_max_abs_difference": 0.0,
    }
    bank = {
        "historical_log_Z_max_abs_difference": 0.0,
        "historical_log_Z_bar_max_abs_difference": 0.0,
        "historical_covariance_logdet_max_abs_difference": 0.0,
        "cache": {
            "second_evaluation_bitwise_identical": True,
            "new_covariance_keys_second_evaluation": 0,
        },
    }
    gates, failure = _gate_results(
        design, dense, atlas, {"bank_2048": bank, "bank_8192": bank}
    )
    assert gates["all_values_finite"] is False
    assert gates["oracle_regression_pass"] is False
    assert failure == "nonfinite_or_numerical_failure"


def test_parent_blocks_reassemble_by_absolute_start_and_reject_missing_block():
    blocks = [
        (2, np.full((3, 2), 2.0), 2),
        (0, np.full((3, 2), 0.0), 2),
    ]
    actual, evaluations = reassemble_parent_blocks(
        blocks, geometry_count=3, parent_count=4, parent_block_size=2
    )
    np.testing.assert_array_equal(
        actual, np.concatenate((blocks[1][1], blocks[0][1]), axis=1)
    )
    assert evaluations == 4
    with np.testing.assert_raises_regex(RuntimeError, "out of order"):
        reassemble_parent_blocks(
            blocks[:1], geometry_count=3, parent_count=4, parent_block_size=2
        )


def test_atomic_regression_artifacts_are_exclusive(tmp_path):
    arrays_path = tmp_path / "arrays.npz"
    result_path = tmp_path / "result.json"
    atomic_npz(arrays_path, {"x": np.arange(3, dtype=np.int16)})
    atomic_json(result_path, {"status": "complete_fail_exact_oracle_regression"})
    with np.testing.assert_raises(FileExistsError):
        atomic_npz(arrays_path, {"x": np.arange(3, dtype=np.int16)})
    with np.testing.assert_raises(FileExistsError):
        atomic_json(result_path, {"status": "changed"})
    with np.load(arrays_path, allow_pickle=False) as item:
        np.testing.assert_array_equal(item["x"], np.arange(3, dtype=np.int16))
    assert json.loads(result_path.read_text())["status"] == (
        "complete_fail_exact_oracle_regression"
    )


def _program_record(program_path, *, implementation_sha, authorized):
    design = json.loads(FROZEN_DESIGN.read_text())
    storage = design["storage"]
    data = Path(storage["data_directory"])
    return {
        "schema": "ouruniv-cf4-aggregate-evidence-oracle-regression-program-v1",
        "status": "frozen_before_exact_oracle_regression",
        "design": {
            "path": "config/cf4_aggregate_evidence_oracle_regression_design.json",
            "sha256": FROZEN_DESIGN_SHA256,
        },
        "implementation": {
            "path": "src/cf4_aggregate_evidence_oracle_regression.py",
            "sha256": implementation_sha,
        },
        "execution": {
            "host": "LagEunha",
            "device": "CPU",
            "worker_processes": 8,
            "threads_per_worker": 1,
            "multiprocessing_start_method": "fork",
            "parent_block_size": 32,
            "geometry_batch_size": 256,
            "process_table_polling": False,
            "automatic_retry_or_scaling": False,
        },
        "storage": {
            "program": str(program_path.resolve()),
            "data_directory": str(data),
            "state_directory": storage["state_directory"],
            "result": str(data / storage["result"]),
            "arrays": str(data / storage["arrays"]),
            "manifest": str(data / storage["manifest"]),
            "exclusive_create_and_atomic_publication": True,
        },
        "decision": {
            "regression_execution_authorized": authorized,
            "production_SMC_authorized": False,
            "conditional_field_bank_authorized": False,
            "parent_or_seed_selection_authorized": False,
            "PM_or_halo_finder_authorized": False,
            "RAMSES_authorized": False,
            "automatic_follow_on": False,
        },
    }


def test_public_execution_rejects_noncanonical_missing_unauthorized_and_wrong_sha(
    tmp_path, monkeypatch
):
    calls = []
    canonical_data = Path(
        json.loads(FROZEN_DESIGN.read_text())["storage"]["data_directory"]
    )
    data_existed_before = canonical_data.exists()

    def forbidden_core(*args, **kwargs):
        calls.append(1)
        raise AssertionError("core must remain unreachable")

    monkeypatch.setattr(regression_module, "_run_regression_core", forbidden_core)
    arbitrary = tmp_path / "not-canonical.json"
    arbitrary.write_text("{}")
    with np.testing.assert_raises_regex(PermissionError, "not canonical"):
        regression_module.execute_program(arbitrary)

    canonical = tmp_path / "canonical.json"
    monkeypatch.setattr(regression_module, "CANONICAL_PROGRAM", canonical)
    with np.testing.assert_raises(FileNotFoundError):
        regression_module.execute_program(canonical)

    source_sha = sha256_file(Path(regression_module.__file__))
    canonical.write_text(json.dumps(_program_record(
        canonical, implementation_sha=source_sha, authorized=False
    )))
    with np.testing.assert_raises(PermissionError):
        regression_module.execute_program(canonical)

    canonical.write_text(json.dumps(_program_record(
        canonical, implementation_sha="0" * 64, authorized=True
    )))
    with np.testing.assert_raises_regex(RuntimeError, "implementation hash"):
        regression_module.execute_program(canonical)
    assert calls == []
    assert canonical_data.exists() is data_existed_before


def test_canonical_program_passes_preflight_and_opens_only_regression():
    path = (
        Path(__file__).resolve().parents[1]
        / "config/cf4_aggregate_evidence_oracle_regression_program.json"
    )
    program = json.loads(path.read_text())
    validate_program(program, path)
    assert program["source_commit"] == "05b2cc373c94ba719492929f10409383257100b1"
    assert program["execution"]["worker_processes"] == 8
    assert program["execution"]["parent_block_size"] == 32
    assert program["execution"]["geometry_batch_size"] == 256
    assert program["decision"]["regression_execution_authorized"] is True
    for key in (
        "production_SMC_authorized",
        "conditional_field_bank_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
        "automatic_follow_on",
    ):
        assert program["decision"][key] is False


def test_canonical_scripts_are_hash_pinned_marker_only_and_no_follow_on():
    root = Path(__file__).resolve().parents[1]
    runner = (
        root / "scripts/run_cf4_aggregate_evidence_oracle_regression_lageunha.sh"
    ).read_text()
    launcher = (
        root / "scripts/launch_cf4_aggregate_evidence_oracle_regression_lageunha.sh"
    ).read_text()
    status = (
        root / "scripts/status_cf4_aggregate_evidence_oracle_regression.sh"
    ).read_text()
    combined = "\n".join((runner, launcher, status)).lower()
    assert "expected_program_sha=" in runner
    assert "expected_design_sha=" in runner
    assert "expected_implementation_sha=" in runner
    assert "expected_tests_sha=" in runner
    assert "merge-base --is-ancestor" in runner
    assert "git -C \"$repo\" diff --quiet HEAD" in runner
    assert "flock -n" in runner
    assert "complete_pass_exact_oracle_regression" in runner
    assert "complete_fail_exact_oracle_regression" in runner
    assert "nonfinite_or_numerical_failure" not in runner.split(
        "allowed_failure =", 1
    )[1].split("}", 1)[0]
    assert "production_SMC_execution_authorized=false" in runner
    assert "conditional_field_bank_authorized=false" in runner
    assert "invalid_state_no_marker" in status
    assert "invalid_state_conflicting_markers" in status
    assert "invalid_state_empty_marker" in status
    assert "invalid_data_without_lifecycle_state" in status
    assert "invalid_complete_artifacts" in status
    assert "validate_program_and_pins" in runner
    assert "postflight source hash mismatch" in runner
    assert "changed during execution" in runner
    assert "result lineage mismatch" in runner
    assert "RUNNING" in runner and "COMPLETE" in runner and "FAILED" in runner
    assert "pgrep" not in combined
    assert "postgres" not in combined
    assert "while " not in combined
    assert "sleep " not in combined


def _status_result(script, state, data):
    environment = os.environ.copy()
    environment["CF4_ORACLE_REGRESSION_STATUS_STATE"] = str(state)
    environment["CF4_ORACLE_REGRESSION_STATUS_DATA"] = str(data)
    return subprocess.run(
        [str(script)],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )


def test_status_lifecycle_is_fail_closed_for_orphans_empty_and_conflicts(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/status_cf4_aggregate_evidence_oracle_regression.sh"
    )

    absent = _status_result(script, tmp_path / "state0", tmp_path / "data0")
    assert absent.returncode == 0
    assert absent.stdout.strip() == "status=not_started"

    orphan_data = tmp_path / "data1"
    orphan_data.mkdir()
    orphan = _status_result(script, tmp_path / "state1", orphan_data)
    assert orphan.returncode == 65
    assert orphan.stdout.strip() == "status=invalid_data_without_lifecycle_state"

    markerless_state = tmp_path / "state2"
    markerless_state.mkdir()
    markerless = _status_result(script, markerless_state, tmp_path / "data2")
    assert markerless.returncode == 65
    assert markerless.stdout.strip() == "status=invalid_state_no_marker"

    empty_state = tmp_path / "state3"
    empty_state.mkdir()
    (empty_state / "COMPLETE").touch()
    empty = _status_result(script, empty_state, tmp_path / "data3")
    assert empty.returncode == 65
    assert empty.stdout.strip() == "status=invalid_state_empty_marker"

    conflict_state = tmp_path / "state4"
    conflict_state.mkdir()
    (conflict_state / "COMPLETE").touch()
    (conflict_state / "FAILED").write_text("status=failed\n")
    conflict = _status_result(script, conflict_state, tmp_path / "data4")
    assert conflict.returncode == 65
    assert conflict.stdout.strip() == "status=invalid_state_conflicting_markers"

    incomplete_state = tmp_path / "state5"
    incomplete_data = tmp_path / "data5"
    incomplete_state.mkdir()
    incomplete_data.mkdir()
    (incomplete_state / "COMPLETE").write_text("status=complete\n")
    incomplete = _status_result(script, incomplete_state, incomplete_data)
    assert incomplete.returncode == 65
    assert incomplete.stdout.strip() == "status=invalid_complete_artifacts"

    complete_state = tmp_path / "state6"
    complete_data = tmp_path / "data6"
    complete_state.mkdir()
    complete_data.mkdir()
    marker_text = "status=complete\nscience_status=complete_pass_exact_oracle_regression\n"
    (complete_state / "COMPLETE").write_text(marker_text)
    for name in ("arrays.npz", "result.json", "manifest.json"):
        (complete_data / name).write_bytes(b"sealed")
    complete = _status_result(script, complete_state, complete_data)
    assert complete.returncode == 0
    assert complete.stdout == marker_text
