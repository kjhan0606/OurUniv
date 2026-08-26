import json
from pathlib import Path

import numpy as np
import pytest

from scripts.cf4_lg_highk_recovery_gpu_guard import (
    parse_memory_totals,
    validate_single_visible_gpu,
)
from scripts.check_cf4_lg_highk_streaming_pm_production_v1 import (
    RESULT_KEYS,
    validate_canonical_rows,
)
from cf4_lg_highk_covariance_cache import sha256_file
from cf4_lg_highk_streaming_forward import SCHEMA


SCHEDULE_SHA = "schedule-sha"
PROGRAM_SHA = "program-sha"
CACHE_SHA = "cache-sha"
PROGRAM = {
    "fixed_model": {
        "fine_mesh": 576,
        "constraint_count": 14,
        "FFT_workers": 1,
        "highk_numerical_gates": {
            "coarse_roundtrip_relative_RMS_max": 2e-6,
            "correction_restriction_relative_RMS_max": 2e-6,
            "maximum_response_identity_error_max": 2e-5,
            "null_subspace_mean_square_range": [0.95, 1.05],
            "absolute_global_field_mean_max": 0.005,
            "maximum_field_imaginary_relative_RMS": 2e-5,
        },
    }
}


def _schedule() -> dict[str, np.ndarray]:
    return {
        "keys": np.arange(256 * 6, dtype=np.int64).reshape(256, 6),
        "parent_seed": np.arange(1000, 1256, dtype=np.int64),
        "group_id": np.arange(256, dtype=np.int64) % 4,
        "fine_field_seed": np.arange(2000, 2256, dtype=np.int64),
        "likelihood_noise_seed": np.arange(3000, 3256, dtype=np.int64),
        "posterior_weight": np.full(256, 1.0 / 256.0, dtype=np.float64),
    }


def _identity(schedule: dict[str, np.ndarray], index: int) -> dict[str, object]:
    return {
        "schedule_index": index,
        "schedule_sha256": SCHEDULE_SHA,
        "program_sha256": PROGRAM_SHA,
        "covariance_cache_sha256": CACHE_SHA,
        "parent_seed": int(schedule["parent_seed"][index]),
        "group_id": int(schedule["group_id"][index]),
        "geometry_key": schedule["keys"][index].tolist(),
        "fine_field_seed": int(schedule["fine_field_seed"][index]),
        "likelihood_noise_seed": int(schedule["likelihood_noise_seed"][index]),
    }


def _write_valid_row(
    root: Path, schedule: dict[str, np.ndarray], index: int, *, n_halos: int = 2
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text("{}\n")
    row = root / f"row_{index:03d}"
    row.mkdir()
    halo_mass = np.ones((n_halos,), dtype=np.float32)
    if n_halos >= 2:
        halo_mass[:2] = [2.0, 1.0]
    np.savez(
        row / "halos.npz",
        halo_pos=np.zeros((n_halos, 3), dtype=np.float32),
        halo_vel=np.zeros((n_halos, 3), dtype=np.float32),
        halo_mass=halo_mass,
        particle_mass=np.float64(1.0),
        box_size=np.float64(384.0),
    )
    diagnostics = {
        "fine_seed": int(schedule["fine_field_seed"][index]),
        "noise_seed": int(schedule["likelihood_noise_seed"][index]),
        "predicted_before": [0.0] * 14,
        "achieved_after": [0.0] * 14,
        "targets": [0.0] * 14,
        "sigma": [1.0] * 14,
        "mock_noise": [0.0] * 14,
        "weights": [0.0] * 14,
        "coarse_roundtrip_relative_RMS": 0.0,
        "correction_restriction_absolute_RMS": 0.0,
        "correction_restriction_relative_RMS": 0.0,
        "maximum_response_identity_error": 0.0,
        "null_subspace_mean_square": 1.0,
        "field_mean": 0.0,
        "field_RMS": 1.0,
        "field_imaginary_relative_RMS": 0.0,
        "FFT_workers": 1,
        "peak_evidence_reapplied": False,
    }
    gates = {
        "coarse_roundtrip": True,
        "correction_in_null_space": True,
        "response_identity": True,
        "null_power": True,
        "global_mean": True,
        "field_imaginary": True,
        "peak_evidence_not_reapplied": True,
    }
    hard_pair = {
        "halo_i": 0,
        "halo_j": 1,
        "m1_fof_msun_h": 2.0,
        "m2_fof_msun_h": 1.0,
        "mass_ratio": 2.0,
        "separation_mpc_h": 1.0,
        "midpoint_mpc_h": [0.5, 0.0, 0.0],
        "midpoint_offset_mpc_h": 0.5,
        "isolation_mpc_h": 99.0,
        "peculiar_radial_velocity_km_s": 0.0,
        "total_radial_velocity_km_s": 100.0,
        "tangential_velocity_km_s": 0.0,
        "m33_candidate": None,
        "ranking_score": 1.0,
    }
    components = {
        "member_log10_mass": -1.0,
        "separation": -1.0,
        "midpoint": -1.0,
        "total_radial_velocity": -1.0,
        "tangential_speed": -1.0,
        "isolation": -1.0,
    }
    z0_pair = {
        "halo_i": 0,
        "halo_j": 1,
        "masses_msun_h": [2.0, 1.0],
        "mass_ratio": 2.0,
        "separation_mpc_h": 1.0,
        "midpoint_mpc_h": [0.5, 0.0, 0.0],
        "midpoint_offset_vector_mpc_h": [0.5, 0.0, 0.0],
        "midpoint_offset_mpc_h": 0.5,
        "isolation_mpc_h": 99.0,
        "peculiar_radial_velocity_km_s": 0.0,
        "total_radial_velocity_km_s": 100.0,
        "tangential_velocity_km_s": 0.0,
        "log_likelihood": -6.0,
        "log_likelihood_components": components,
    }
    hard_pairs = [hard_pair] if n_halos >= 2 else []
    z0_pairs = [z0_pair] if n_halos >= 2 else []
    result = {
        "schema": SCHEMA,
        "status": "complete",
        **_identity(schedule, index),
        "halo_catalogue": "halos.npz",
        "halo_catalogue_sha256": sha256_file(row / "halos.npz"),
        "posterior_weight": float(schedule["posterior_weight"][index]),
        "field_sha256": "a" * 64,
        "field_persisted": False,
        "full_pm_particle_state_persisted": False,
        "parent_centered_P1_evaluated": False,
        "field_diagnostics": diagnostics,
        "highk_numerical_gates": gates,
        "highk_numerical_pass": True,
        "n_central_particles": max(40, n_halos),
        "n_halos": n_halos,
        "hard_p2_pairs": hard_pairs,
        "hard_p2_pass": bool(hard_pairs),
        "z0_likelihood": {
            "n_candidate_pairs": len(z0_pairs),
            "log_likelihood": -6.0 if z0_pairs else -np.inf,
            "best_pair": z0_pairs[0] if z0_pairs else None,
            "candidate_pairs": z0_pairs,
        },
        "seconds": 1.0,
    }
    (row / "result.json").write_text(json.dumps(result))
    return row


def _audit(root: Path, schedule: dict[str, np.ndarray], *, complete: bool = False):
    return validate_canonical_rows(
        output_root=root,
        schedule=schedule,
        program=PROGRAM,
        schedule_sha256=SCHEDULE_SHA,
        program_sha256=PROGRAM_SHA,
        cache_sha256=CACHE_SHA,
        require_complete=complete,
    )


def _rewrite_result(row: Path, **updates: object) -> None:
    result_path = row / "result.json"
    result = json.loads(result_path.read_text())
    result.update(updates)
    result_path.write_text(json.dumps(result))


def _mutate_result(row: Path, mutation) -> None:
    result_path = row / "result.json"
    result = json.loads(result_path.read_text())
    mutation(result)
    result_path.write_text(json.dumps(result))


def _refresh_halo_hash(row: Path) -> None:
    _rewrite_result(row, halo_catalogue_sha256=sha256_file(row / "halos.npz"))


def test_fake_incomplete_root_with_valid_row_passes_read_only_audit(tmp_path: Path):
    schedule = _schedule()
    _write_valid_row(tmp_path, schedule, 0)
    result = _audit(tmp_path, schedule)
    assert result["status"] == "valid_incomplete"
    assert result["complete_indices"] == [0]
    assert result["missing_count"] == 255


@pytest.mark.parametrize("mutation", ["identity", "hash", "false_highk"])
def test_result_identity_hash_and_highk_mutations_are_rejected(
    tmp_path: Path, mutation: str
):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0)
    if mutation == "identity":
        _rewrite_result(row, fine_field_seed=999999)
    elif mutation == "hash":
        _rewrite_result(row, halo_catalogue_sha256="0" * 64)
    else:
        _rewrite_result(row, highk_numerical_pass=False)
    with pytest.raises(RuntimeError):
        _audit(tmp_path, schedule)


@pytest.mark.parametrize("missing_key", sorted(RESULT_KEYS))
def test_every_frozen_result_key_is_required(tmp_path: Path, missing_key: str):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0)
    _mutate_result(row, lambda result: result.pop(missing_key))
    with pytest.raises(RuntimeError, match="frozen result keyset"):
        _audit(tmp_path, schedule)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("posterior_weight", 0.5),
        ("parent_centered_P1_evaluated", True),
        ("field_sha256", "A" * 64),
        ("field_sha256", "a" * 63),
        ("n_central_particles", -1),
        ("n_central_particles", 576**3 + 1),
        ("n_central_particles", "40"),
        ("seconds", 0.0),
        ("seconds", float("nan")),
        ("seconds", "1.0"),
    ],
)
def test_scalar_result_contract_mutations_are_rejected(
    tmp_path: Path, field: str, value: object
):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0)
    _rewrite_result(row, **{field: value})
    with pytest.raises(RuntimeError):
        _audit(tmp_path, schedule)


def test_recomputed_highk_gates_must_match_diagnostics_and_stored_flags(tmp_path: Path):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0)
    _mutate_result(
        row, lambda result: result["field_diagnostics"].update(field_mean=1.0)
    )
    with pytest.raises(RuntimeError, match="recomputed gates"):
        _audit(tmp_path, schedule)

    row.joinpath("result.json").unlink()
    row.joinpath("halos.npz").unlink()
    row.rmdir()
    row = _write_valid_row(tmp_path, schedule, 0)

    def consistently_failed_gate(result):
        result["field_diagnostics"]["field_mean"] = 1.0
        result["highk_numerical_gates"]["global_mean"] = False
        result["highk_numerical_pass"] = False

    _mutate_result(row, consistently_failed_gate)
    with pytest.raises(RuntimeError, match="did not pass all"):
        _audit(tmp_path, schedule)

    row.joinpath("result.json").unlink()
    row.joinpath("halos.npz").unlink()
    row.rmdir()
    row = _write_valid_row(tmp_path, schedule, 0)
    _mutate_result(
        row,
        lambda result: result["highk_numerical_gates"].update(global_mean=False),
    )
    with pytest.raises(RuntimeError, match="recomputed gates"):
        _audit(tmp_path, schedule)


@pytest.mark.parametrize("mutation", ["pass", "schema", "finite", "identity", "mass"])
def test_hard_p2_pair_contract_mutations_are_rejected(tmp_path: Path, mutation: str):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0)

    def change(result):
        pair = result["hard_p2_pairs"][0]
        if mutation == "pass":
            result["hard_p2_pass"] = False
        elif mutation == "schema":
            pair.pop("ranking_score")
        elif mutation == "finite":
            pair["ranking_score"] = float("nan")
        elif mutation == "identity":
            pair["halo_j"] = 0
        else:
            pair["m1_fof_msun_h"] = 3.0

    _mutate_result(row, change)
    with pytest.raises(RuntimeError):
        _audit(tmp_path, schedule)


@pytest.mark.parametrize(
    "mutation", ["outer_schema", "count", "best", "mixture", "pair_schema", "components", "identity"]
)
def test_z0_likelihood_contract_mutations_are_rejected(tmp_path: Path, mutation: str):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0)

    def change(result):
        likelihood = result["z0_likelihood"]
        pair = likelihood["candidate_pairs"][0]
        if mutation == "outer_schema":
            likelihood["unexpected"] = 1
        elif mutation == "count":
            likelihood["n_candidate_pairs"] = 2
        elif mutation == "best":
            likelihood["best_pair"] = None
        elif mutation == "mixture":
            likelihood["log_likelihood"] = -5.0
        elif mutation == "pair_schema":
            pair.pop("mass_ratio")
        elif mutation == "components":
            pair["log_likelihood_components"]["isolation"] = -2.0
            likelihood["best_pair"] = pair
        else:
            pair["halo_j"] = 0
            likelihood["best_pair"] = pair

    _mutate_result(row, change)
    with pytest.raises(RuntimeError):
        _audit(tmp_path, schedule)


@pytest.mark.parametrize(
    "updates",
    [
        {"field_persisted": True},
        {"full_pm_particle_state_persisted": True},
        {"n_halos": 3},
    ],
)
def test_result_streaming_flags_and_halo_count_are_rejected(tmp_path: Path, updates):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0, n_halos=1)
    _rewrite_result(row, **updates)
    with pytest.raises(RuntimeError):
        _audit(tmp_path, schedule)


@pytest.mark.parametrize(
    "mutation", ["malformed", "dtype", "shape", "nonfinite", "extra_entry"]
)
def test_malformed_or_contract_violating_npz_is_rejected(tmp_path: Path, mutation: str):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0)
    halo = row / "halos.npz"
    if mutation == "malformed":
        halo.write_bytes(b"not an npz")
    else:
        position = np.zeros((1, 3), dtype=np.float32)
        mass = np.ones((1,), dtype=np.float32)
        if mutation == "dtype":
            position = position.astype(np.float64)
        elif mutation == "shape":
            position = np.zeros((1, 2), dtype=np.float32)
        elif mutation == "nonfinite":
            mass[0] = np.nan
        arrays = {
            "halo_pos": position,
            "halo_vel": np.zeros((1, 3), dtype=np.float32),
            "halo_mass": mass,
            "particle_mass": np.float64(1.0),
            "box_size": np.float64(384.0),
        }
        if mutation == "extra_entry":
            arrays["unexpected"] = np.asarray(1, dtype=np.int64)
        np.savez(halo, **arrays)
    _refresh_halo_hash(row)
    with pytest.raises(RuntimeError):
        _audit(tmp_path, schedule)


@pytest.mark.parametrize("kind", ["row", "result", "halos"])
def test_canonical_row_and_files_must_not_be_symlinks(tmp_path: Path, kind: str):
    schedule = _schedule()
    if kind == "row":
        real_root = tmp_path / "real"
        real_row = _write_valid_row(real_root, schedule, 0)
        output = tmp_path / "output"
        output.mkdir()
        (output / "run_manifest.json").write_text("{}\n")
        (output / "row_000").symlink_to(real_row, target_is_directory=True)
    else:
        output = tmp_path / "output"
        row = _write_valid_row(output, schedule, 0)
        target = row / ("result.json" if kind == "result" else "halos.npz")
        external = tmp_path / f"external-{target.name}"
        target.replace(external)
        target.symlink_to(external)
    with pytest.raises(RuntimeError, match="symlink|regular file"):
        _audit(output, schedule)


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_extra_row_or_root_entry_is_rejected(tmp_path: Path, entry_kind: str):
    schedule = _schedule()
    row = _write_valid_row(tmp_path, schedule, 0)
    extra_row = row / "extra"
    extra_row.write_bytes(b"x") if entry_kind == "file" else extra_row.mkdir()
    with pytest.raises(RuntimeError, match="unexpected entries"):
        _audit(tmp_path, schedule)
    extra_row.unlink() if entry_kind == "file" else extra_row.rmdir()
    extra_root = tmp_path / "unexpected"
    extra_root.write_text("x") if entry_kind == "file" else extra_root.mkdir()
    with pytest.raises(RuntimeError, match="production-root entries"):
        _audit(tmp_path, schedule)


def test_require_complete_rejects_leftover_staging_before_missing_rows(tmp_path: Path):
    schedule = _schedule()
    _write_valid_row(tmp_path, schedule, 0)
    (tmp_path / f".row_001.123.{'a' * 32}.staging").mkdir()
    with pytest.raises(RuntimeError, match="retains staging"):
        _audit(tmp_path, schedule, complete=True)


def test_require_complete_rejects_fewer_than_256_rows(tmp_path: Path):
    schedule = _schedule()
    _write_valid_row(tmp_path, schedule, 0)
    with pytest.raises(RuntimeError, match="exactly 256 rows; found 1"):
        _audit(tmp_path, schedule, complete=True)


def test_require_complete_accepts_exactly_256_valid_rows(tmp_path: Path):
    schedule = _schedule()
    for index in range(256):
        _write_valid_row(tmp_path, schedule, index, n_halos=0)
    result = _audit(tmp_path, schedule, complete=True)
    assert result["status"] == "complete"
    assert result["complete_count"] == 256
    assert result["staging_directories"] == []


def test_gpu_guard_requires_exactly_one_visible_gpu_with_115000_mib():
    assert parse_memory_totals("143771\n") == [143771]
    assert validate_single_visible_gpu([143771]) == 143771
    with pytest.raises(RuntimeError, match="exactly one"):
        validate_single_visible_gpu([])
    with pytest.raises(RuntimeError, match="exactly one"):
        validate_single_visible_gpu([143771, 143771])
    with pytest.raises(RuntimeError, match="at least 115000"):
        validate_single_visible_gpu([95830])
    with pytest.raises(RuntimeError, match="invalid nvidia-smi"):
        parse_memory_totals("143771 MiB\n")
