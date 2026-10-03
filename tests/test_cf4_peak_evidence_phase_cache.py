import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_peak_evidence import prepare_exact_peak_operator
from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    full_spectrum_from_rfft,
    impulse_spectrum,
    parent_mean_at_point_sets,
    phase_cache_metadata,
    phase_response_grid,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian_filter(n, radius=1.4):
    frequency = 2.0 * np.pi * np.fft.fftfreq(n)
    k2 = (
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    result = np.exp(-0.5 * radius ** 2 * k2)
    result[0, 0, 0] = 0.0
    return result


def test_analytic_impulse_spectrum_matches_unitary_fft():
    n = 12
    point = np.asarray([7, 2, 11])
    impulse = np.zeros((n, n, n))
    impulse[tuple(point)] = 1.0
    np.testing.assert_allclose(
        impulse_spectrum(n, point),
        np.fft.fftn(impulse, norm="ortho"),
        rtol=2e-14, atol=2e-14,
    )


def test_full_spectrum_expansion_matches_real_field_fft():
    rng = np.random.default_rng(69)
    field = rng.standard_normal((12, 12, 12))
    expanded = full_spectrum_from_rfft(np.fft.rfftn(field, norm="ortho"))
    np.testing.assert_allclose(
        expanded, np.fft.fftn(field, norm="ortho"),
        rtol=2e-14, atol=2e-14,
    )


def test_phase_response_grid_matches_exact_operator_column():
    n, coarse_n = 12, 4
    filt = gaussian_filter(n)
    phase = np.asarray([2, 1, 0])
    points = np.asarray([
        phase,
        [5, 7, 9],
        [11, 2, 4],
        [1, 10, 8],
    ])
    exact = prepare_exact_peak_operator(filt, coarse_n, points, 0.25)
    grid = phase_response_grid(filt, coarse_n, phase)
    expected_column = grid[tuple(points.T)]
    np.testing.assert_allclose(
        expected_column, exact.signal_covariance[:, 0],
        rtol=2e-12, atol=2e-14,
    )


def test_phase_cache_matches_exact_covariance_for_multiple_geometries():
    n, coarse_n = 12, 4
    filt = gaussian_filter(n)
    point_sets = [
        np.asarray([[0, 0, 0], [1, 2, 3], [11, 7, 5], [4, 8, 2]]),
        np.asarray([[5, 5, 5], [8, 2, 11], [3, 9, 6], [10, 1, 4]]),
    ]
    cached, metadata = covariance_for_point_sets(filt, coarse_n, point_sets)
    for points, covariance in zip(point_sets, cached):
        exact = prepare_exact_peak_operator(filt, coarse_n, points, 0.3)
        np.testing.assert_allclose(
            covariance, exact.signal_covariance, rtol=3e-12, atol=3e-14
        )
    assert metadata["refinement_ratio"] == 3
    assert 1 < metadata["phase_count_used"] <= 27
    assert metadata["response_grids_held_simultaneously"] == 1
    assert metadata["maximum_pre_symmetrization_asymmetry"] < 1e-12


def test_parent_mean_uses_one_exact_field_for_all_point_sets():
    rng = np.random.default_rng(67)
    n, coarse_n = 12, 4
    filt = gaussian_filter(n)
    coarse = rng.standard_normal((coarse_n, coarse_n, coarse_n))
    point_sets = [
        np.asarray([[0, 0, 0], [1, 2, 3], [11, 7, 5]]),
        np.asarray([[5, 5, 5], [8, 2, 11], [3, 9, 6]]),
    ]
    cached = parent_mean_at_point_sets(coarse, filt, point_sets)
    for points, mean in zip(point_sets, cached):
        exact = prepare_exact_peak_operator(filt, coarse_n, points, 0.2)
        np.testing.assert_allclose(
            mean, exact.predict_parent(coarse), rtol=2e-12, atol=2e-14
        )


def test_phase_cache_is_exact_but_not_yet_full_size_authorized():
    metadata = phase_cache_metadata()
    assert metadata["covariance"] == "exact AQA*; no stationary approximation"
    assert metadata["production_phase_count"] == 27
    assert metadata["memory_policy"] == "one Nfine response grid at a time"
    assert "no workers=-1" in metadata["FFT_workers"]
    assert metadata["all_parent_evidence_authorized"] is False


def test_full_size_phase_control_is_hash_pinned_and_firewalled():
    program = json.loads((
        ROOT / "config/cf4_peak_evidence_phase_control_v2_program.json"
    ).read_text())
    for key in ("implementation", "phase_cache", "projection_contract"):
        item = program[key]
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    assert sha256_file(
        ROOT / program["authorization"]["architecture_design"]
    ) == program["authorization"]["architecture_design_sha256"]
    peak = program["peak_geometry_implementation"]
    assert sha256_file(ROOT / peak["path"]) == peak["sha256"]
    model = program["Local_Group_model"]
    assert sha256_file(ROOT / model["source_program"]) == (
        model["source_program_sha256"]
    )
    firewall = program["information_firewall"]
    assert firewall["all_parent_weights_computed"] is False
    assert firewall["candidate_field_generated"] is False
    assert firewall["PM_or_halo_finder_run"] is False
    assert firewall["parent_or_seed_selection_allowed"] is False
    assert firewall["RAMSES_authorized"] is False
    extends = program["extends"]
    assert sha256_file(ROOT / extends["v1_program"]) == extends["v1_program_sha256"]
    assert sha256_file(ROOT / extends["v1_failure_record"]) == (
        extends["v1_failure_record_sha256"]
    )
    assert program["gates"]["phase_response_imaginary_relative_RMS_max"] == 1e-10
    assert program["gates"]["phase_response_absolute_imaginary_max"] == 1e-15


def test_phase_control_lifecycle_is_single_shot_without_process_polling():
    paths = [
        ROOT / "scripts/run_cf4_peak_evidence_phase_control_v2_lageunha.sh",
        ROOT / "scripts/launch_cf4_peak_evidence_phase_control_v2_lageunha.sh",
        ROOT / "scripts/status_cf4_peak_evidence_phase_control_v2.sh",
    ]
    for path in paths:
        text = path.read_text()
        assert "pgrep" not in text
        assert "while " not in text
        assert "sleep " not in text
        assert "workers=-1" not in text


def test_v1_failure_is_preserved_as_tolerance_not_science_failure():
    record = json.loads((
        ROOT / "config/cf4_peak_evidence_phase_control_failure_record.json"
    ).read_text())
    assert record["status"] == "failed_numerical_tolerance_before_scientific_gate"
    assert record["failure"]["scientific_gate_opened"] is False
    assert record["failure"]["parent_evidence_computed"] is False
    assert record["decision"]["reuse_failed_output_as_pass"] is False
    assert record["decision"]["delete_or_overwrite_failed_state"] is False
    assert record["decision"]["v2_control_authorized"] is True
    assert record["decision"]["all_parent_evidence_authorized"] is False


def test_v2_result_authorizes_evidence_program_but_not_candidates():
    record = json.loads((
        ROOT / "config/cf4_peak_evidence_phase_control_v2_result_record.json"
    ).read_text())
    assert record["status"] == "complete_pass_exact_N576_phase_cache"
    assert sha256_file(ROOT / record["lineage"]["program"]) == (
        record["lineage"]["program_sha256"]
    )
    assert sha256_file(ROOT / record["lineage"]["phase_cache_implementation"]) == (
        record["lineage"]["phase_cache_implementation_sha256"]
    )
    assert record["gates"]["phase_cache_pass"] is True
    decision = record["decision"]
    assert decision["freeze_all_256_parent_evidence_program_authorized"] is True
    assert decision["stationary_covariance_approximation_authorized"] is False
    assert decision["candidate_generation_authorized"] is False
    assert decision["parent_or_seed_selection_authorized"] is False
    assert decision["PM_or_RAMSES_authorized"] is False
