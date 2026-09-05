import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cf4_lg_highk_streaming_forward import (
    SCHEMA,
    _highk_gates,
    _mode_indices,
    _valid_completed_row,
    run_streaming_forward,
    validate_production_run_manifest,
)


def test_pilot_indices_and_production_batch_coverage_are_fixed() -> None:
    program = {"technical_pilot": {"schedule_indices": [0, 64, 128, 192]}}
    np.testing.assert_array_equal(_mode_indices(program, "pilot", None), [0, 64, 128, 192])
    np.testing.assert_array_equal(_mode_indices(program, "production", 15), np.arange(240, 256))
    with pytest.raises(ValueError, match="requires --batch-index"):
        _mode_indices(program, "production", None)


def test_completed_row_resume_requires_full_identity_and_halo_hash(tmp_path: Path) -> None:
    row = tmp_path / "row_000"
    row.mkdir()
    np.savez(row / "halos.npz", halo_mass=np.asarray([1.0], dtype=np.float32))
    identity = {
        "schedule_index": 0, "schedule_sha256": "schedule", "program_sha256": "program",
        "covariance_cache_sha256": "cache", "parent_seed": 3193, "group_id": 0,
        "geometry_key": [6, 6, 6, 2, 0, 0], "fine_field_seed": 1,
        "likelihood_noise_seed": 2,
    }
    result = {
        "schema": SCHEMA, "status": "complete", **identity,
        "halo_catalogue": "halos.npz",
        "halo_catalogue_sha256": hashlib.sha256((row / "halos.npz").read_bytes()).hexdigest(),
    }
    (row / "result.json").write_text(json.dumps(result))
    assert _valid_completed_row(row, identity=identity)
    assert not _valid_completed_row(row, identity={**identity, "fine_field_seed": 99})
    np.savez(row / "halos.npz", halo_mass=np.asarray([2.0], dtype=np.float32))
    assert not _valid_completed_row(row, identity=identity)


def test_manifest_validator_rejects_nonexact_batch_coverage(tmp_path: Path) -> None:
    program, cache = tmp_path / "program.json", tmp_path / "cache.npz"
    program.write_text("{}")
    np.savez(cache, value=np.asarray([1]))
    manifest = {
        "schema": "ouruniv-cf4-lg-highk-streaming-forward-run-manifest-v1",
        "status": "prepared",
        "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "covariance_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "batch_count": 16, "rows_per_batch": 16,
        "batches": [list(range(number * 16, (number + 1) * 16)) for number in range(16)],
    }
    validate_production_run_manifest(manifest, program_path=program, cache_path=cache)
    manifest["batches"][15] = list(range(239, 255))
    with pytest.raises(ValueError, match="exact 16x16"):
        validate_production_run_manifest(manifest, program_path=program, cache_path=cache)


def test_highk_gates_use_fixed_model_program_block() -> None:
    program = {"fixed_model": {"highk_numerical_gates": {
        "coarse_roundtrip_relative_RMS_max": 2e-6,
        "correction_restriction_relative_RMS_max": 2e-6,
        "maximum_response_identity_error_max": 2e-5,
        "null_subspace_mean_square_range": [0.95, 1.05],
        "absolute_global_field_mean_max": 0.005,
        "maximum_field_imaginary_relative_RMS": 2e-5,
    }}}
    diagnostics = {
        "coarse_roundtrip_relative_RMS": 1e-8,
        "correction_restriction_relative_RMS": 1e-8,
        "maximum_response_identity_error": 1e-8,
        "null_subspace_mean_square": 1.0,
        "field_mean": 0.001,
        "field_imaginary_relative_RMS": 1e-8,
        "peak_evidence_reapplied": False,
    }
    assert all(_highk_gates(diagnostics, program).values())


def test_frozen_program_refuses_unauthorized_execution(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    program = json.loads(
        (root / "config/cf4_lg_highk_streaming_forward_program_v1.json").read_text()
    )
    program["authorization"]["integrated_four_row_PM_pilot_execution"] = False
    temporary = tmp_path / "unauthorized-program.json"
    temporary.write_text(json.dumps(program))
    with pytest.raises(PermissionError, match="not authorized"):
        try:
            run_streaming_forward(
                program_path=temporary,
                mode="pilot", output_root=root / "unused-test-output",
            )
        finally:
            temporary.unlink()
