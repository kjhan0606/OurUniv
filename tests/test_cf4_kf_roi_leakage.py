import hashlib
import itertools
import json
import math
import sys
from copy import deepcopy
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cf4_kf_roi_leakage import (  # noqa: E402
    FROZEN_DESIGN_SHA256,
    LeakageError,
    NativeBin,
    WindowSpec,
    build_native_bins,
    calculate,
    canonical_json_bytes,
    compute_mixing_matrix,
    deterministic_npz_bytes,
    load_frozen_design,
    mode_counts_by_native_bin,
    orientation_averaged_structure_factor,
    publish_artifacts,
    self_conjugate_squared_radius_histogram,
    single_sphere_window_moment,
    sphere_radial_transform,
    squared_radius_histogram_fft,
    supported_suffix,
    validate_preflight_result,
)
from check_cf4_kf_roi_leakage_v1 import audit_directory  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "config" / "cf4_kf_bin_manifest_design_v1.json"
GRANT_PATH = ROOT / "config" / "cf4_kf_roi_leakage_execution_v1.json"
COMMIT = "a" * 40


def _small_bins(grid_size):
    half = grid_size // 2
    ratio = 2.0**0.25
    edges = [1.0]
    while edges[-1] * ratio <= half:
        edges.append(edges[-1] * ratio)
    if edges[-1] != half:
        edges.append(float(half))
    return [
        NativeBin(index, lower, upper, math.sqrt(lower * upper), index == len(edges) - 2)
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]))
    ]


def _direct_squared_radius_histogram(grid_size):
    half = grid_size // 2
    frequencies = range(-half, half)
    return Counter(
        x * x + y * y + z * z
        for x, y, z in itertools.product(frequencies, repeat=3)
    )


def _direct_native_bin(q, bins):
    for item in bins:
        if item.lower * item.lower <= q and (
            q < item.upper * item.upper
            or (item.terminal and q <= item.upper * item.upper)
        ):
            return item.index
    raise AssertionError(f"q={q} was not assigned")


def _canonical_conjugate(vector, grid_size):
    half = grid_size // 2
    conjugate = []
    for component in vector:
        value = (-component) % grid_size
        conjugate.append(value - grid_size if value >= half else value)
    return tuple(conjugate)


def test_fft_squared_radius_counts_match_exact_small_lattice_enumeration():
    grid_size = 8
    histogram, audit = squared_radius_histogram_fft(grid_size)
    direct = _direct_squared_radius_histogram(grid_size)

    assert audit["total_count_assertion_pass"]
    assert audit["full_lattice_count"] == grid_size**3
    assert audit["max_abs_rounding_error"] <= audit["rounding_abs_tolerance"]
    assert {index: int(value) for index, value in enumerate(histogram) if value} == dict(
        direct
    )


def test_native_mode_counts_handle_conjugates_self_modes_and_dc_exactly():
    grid_size = 8
    half = grid_size // 2
    bins = _small_bins(grid_size)
    full, independent, audit = mode_counts_by_native_bin(
        grid_size, bins, fundamental=1.0
    )
    direct = _direct_squared_radius_histogram(grid_size)
    full_sphere_without_dc = sum(
        count for q, count in direct.items() if 0 < q <= half * half
    )
    direct_full_bins = np.zeros(len(bins), dtype=np.int64)
    direct_independent_bins = np.zeros(len(bins), dtype=np.int64)
    frequencies = range(-half, half)
    for vector in itertools.product(frequencies, repeat=3):
        q = sum(component * component for component in vector)
        if q == 0 or q > half * half:
            continue
        index = _direct_native_bin(q, bins)
        direct_full_bins[index] += 1
        if vector <= _canonical_conjugate(vector, grid_size):
            direct_independent_bins[index] += 1

    assert self_conjugate_squared_radius_histogram(grid_size) == {half * half: 3}
    assert np.array_equal(full, direct_full_bins)
    assert np.array_equal(independent, direct_independent_bins)
    assert int(full.sum()) == full_sphere_without_dc
    assert int(independent.sum()) == (full_sphere_without_dc + 3) // 2
    assert audit["DC_excluded"]
    assert audit["self_conjugate_vectors_in_analysis_sphere"] == 3


def test_frozen_native_edges_have_37_complete_plus_terminal_bin():
    design, digest = load_frozen_design(DESIGN_PATH)
    bins = build_native_bins(design)

    assert digest == FROZEN_DESIGN_SHA256
    assert len(bins) == 38
    assert sum(not item.terminal for item in bins) == 37
    assert bins[-1].terminal
    assert bins[-1].upper == math.pi / 0.3
    assert all(bins[index].upper == bins[index + 1].lower for index in range(37))


def test_sphere_transform_zero_equals_window_volume_moment():
    radius = 5.0
    order = 48
    transform_zero = sphere_radial_transform(np.array([0.0]), radius, order)[0]
    volume_moment = single_sphere_window_moment(radius, 1, order)

    assert transform_zero == pytest.approx(volume_moment, rel=1e-14, abs=1e-14)


def test_disjoint_union_structure_factor_at_zero_is_multiplicity_squared():
    specification = WindowSpec(
        key="union",
        radius=1.0,
        centers=((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0)),
        box_size=20.0,
    )

    factor = orientation_averaged_structure_factor(np.array([0.0]), specification)

    assert factor[0] == pytest.approx(9.0)


def test_small_mixing_matrix_is_finite_nonnegative_and_parseval_normalized():
    bins = [
        NativeBin(0, 0.1, 0.3, math.sqrt(0.03), False),
        NativeBin(1, 0.3, 0.8, math.sqrt(0.24), False),
        NativeBin(2, 0.8, 2.0, math.sqrt(1.6), True),
    ]
    specification = WindowSpec("sphere", 2.0, (), 20.0)

    matrix, moments, audit = compute_mixing_matrix(
        bins,
        specification,
        radial_order=32,
        shell_order=4,
        mu_order=6,
    )

    assert matrix.shape == (3, 3)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0.0)
    assert moments["int_W2_dV"] > 0.0
    assert moments["V_eff"] > 0.0
    assert audit["parseval_denominator"] == pytest.approx(
        (2.0 * math.pi) ** 3 * moments["int_W2_dV"]
    )
    assert audit["sphere_A0_relative_error"] < 1e-14


@pytest.mark.parametrize(
    ("mask", "expected_pass", "reason"),
    [
        ([False, False, True, True], True, "contiguous_supported_suffix_through_Nyquist"),
        ([False, True, False, True], False, "supported_suffix_hole_fail_closed"),
        ([False, False, False], False, "no_supported_native_bin"),
    ],
)
def test_supported_suffix_is_fail_closed_on_holes(mask, expected_pass, reason):
    result = supported_suffix(np.asarray(mask, dtype=np.bool_))

    assert result["pass"] is expected_pass
    assert result["reason"] == reason


def test_publish_refuses_overwrite_before_creating_staging(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        publish_artifacts(output, {}, {}, {})

    assert list(tmp_path.iterdir()) == [output]


def test_production_refuses_absent_or_bad_preflight_before_heavy_work(tmp_path):
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))

    with pytest.raises(LeakageError, match="production requires"):
        calculate(design, FROZEN_DESIGN_SHA256, "production", COMMIT)

    bad = tmp_path / "preflight.json"
    bad.write_bytes(canonical_json_bytes({"status": "PRECHECK"}))
    with pytest.raises(LeakageError, match="SHA256 mismatch"):
        validate_preflight_result(
            bad, "0" * 64, FROZEN_DESIGN_SHA256, COMMIT
        )


def test_preflight_validation_requires_pass_and_exact_commit(tmp_path):
    payload = {
        "status": "PRECHECK",
        "mode": "preflight",
        "design_raw_sha256": FROZEN_DESIGN_SHA256,
        "implementation_commit": COMMIT,
        "numerical_convergence": {"status": "FAIL"},
    }
    path = tmp_path / "result.json"
    encoded = canonical_json_bytes(payload)
    path.write_bytes(encoded)

    with pytest.raises(LeakageError, match="numerical preflight PASS"):
        validate_preflight_result(
            path, hashlib.sha256(encoded).hexdigest(), FROZEN_DESIGN_SHA256, COMMIT
        )


def test_npz_and_published_payload_hashes_are_deterministic(tmp_path):
    arrays = {
        "z": np.arange(6, dtype=np.float64).reshape(2, 3),
        "a": np.array([1, 2, 3], dtype=np.int64),
    }
    assert deterministic_npz_bytes(arrays) == deterministic_npz_bytes(
        dict(reversed(list(arrays.items())))
    )

    result = {
        "status": "PRECHECK",
        "mode": "preflight",
        "design_raw_sha256": FROZEN_DESIGN_SHA256,
        "implementation_commit": COMMIT,
    }
    counts = {"schema": "test", "counts": [1, 2, 3]}
    first = tmp_path / "first"
    second = tmp_path / "second"
    publish_artifacts(first, result, counts, arrays)
    publish_artifacts(second, result, counts, dict(reversed(list(arrays.items()))))

    for filename in (
        "result.json",
        "mode_counts.json",
        "mixing_matrices.npz",
        "manifest.json",
        "COMPLETE",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_checker_validates_hash_bound_preflight_directory(tmp_path):
    output = tmp_path / "preflight"
    implementation = ROOT / "src" / "cf4_kf_roi_leakage.py"
    result = {
        "schema": "ouruniv-cf4-kf-roi-leakage-result-v1",
        "status": "PRECHECK",
        "mode": "preflight",
        "design_raw_sha256": FROZEN_DESIGN_SHA256,
        "implementation_path": "src/cf4_kf_roi_leakage.py",
        "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "implementation_commit": COMMIT,
        "truth_or_candidate_data_consumed": False,
        "final_manifest_materialized": False,
        "k_boundary_claim_created": False,
        "overall_leakage_gate_pass": False,
        "scientific_disposition": "PRECHECK_only_not_scientific",
        "numerical_convergence": {"status": "PASS"},
    }
    counts = {
        "schema": "ouruniv-cf4-kf-roi-leakage-mode-counts-v1",
        "design_raw_sha256": FROZEN_DESIGN_SHA256,
        "native_bins": [{"index": index} for index in range(38)],
        "count_audit": {"total_count_assertion_pass": True},
    }
    publish_artifacts(
        output,
        result,
        counts,
        {"fine__sphere_R8": np.zeros((38, 38), dtype=np.float64)},
    )
    grant = deepcopy(
        json.loads(
            (ROOT / "config" / "cf4_kf_roi_leakage_execution_v1.json").read_text(
                encoding="utf-8"
            )
        )
    )
    grant["scope"]["preflight_output"] = str(output)
    grant_path = tmp_path / "grant.json"
    grant_path.write_bytes(canonical_json_bytes(grant))

    audited = audit_directory(
        output, DESIGN_PATH, grant_path, "preflight", COMMIT
    )

    assert audited["status"] == "PASS"
    assert audited["overall_leakage_gate_pass"] is False


def test_execution_grant_is_narrow_cpu_only_and_keeps_downstream_blocked():
    grant = json.loads(GRANT_PATH.read_text(encoding="utf-8"))

    assert grant["status"] == "user_approved_narrow_execution_grant"
    assert grant["scope"]["design_raw_sha256"] == FROZEN_DESIGN_SHA256
    authorization = grant["authorization"]
    assert authorization["Slurm_preflight_authorized"]
    assert authorization["maximum_preflight_submissions"] == 1
    assert authorization["Slurm_production_authorized_conditionally"]
    assert authorization["maximum_production_submissions"] == 1
    for key in (
        "network_access_authorized",
        "final_manifest_materialization_authorized",
        "KF_EXPAND_authorized",
        "all_D_mock_execution_authorized",
        "production_science_inference_authorized",
        "retry_authorized",
        "numeric_retuning_authorized",
        "replacement_run_authorized",
    ):
        assert authorization[key] is False
    slurm = grant["Slurm"]
    assert slurm["partition"] == "a10"
    assert slurm["CPU_only"]
    assert slurm["GPU_count"] == 0
    assert slurm["cpus_per_task"] == 16
    assert slurm["preflight_memory_MiB"] == 4096
    assert slurm["production_memory_rule"] == (
        "max(1024 MiB, ceil(preflight MaxRSS MiB * 1.2))"
    )
    assert grant["numerical_contract"]["convergence_abs_tolerance"] == 0.0005
