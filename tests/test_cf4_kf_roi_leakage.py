import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cf4_kf_roi_leakage as leakage  # noqa: E402
from cf4_kf_roi_leakage import (  # noqa: E402
    FINE_NUMERICS,
    FROZEN_DESIGN_SHA256,
    LeakageError,
    MixingEvaluation,
    NativeBin,
    QPowerCumulative,
    WindowSpec,
    _compare_evaluations,
    _window_frequency_scale,
    _window_moments,
    _window_specs,
    build_native_bins,
    canonical_json_bytes,
    compute_mixing_matrix,
    deterministic_npz_bytes,
    evaluate_support,
    load_execution_grant,
    load_frozen_design,
    maximal_contiguous_runs,
    mode_counts_by_native_bin,
    orientation_averaged_structure_factor,
    parseval_audit,
    publish_artifacts,
    raised_cosine_window,
    self_conjugate_squared_radius_histogram,
    sphere_radial_transform,
    squared_radius_histogram_fft,
    uv_shell_integral,
    uv_v_segments,
)
from check_cf4_kf_roi_leakage_v3 import audit_directory  # noqa: E402


DESIGN_PATH = ROOT / "config" / "cf4_kf_bin_manifest_design_v2.json"
GRANT_PATH = ROOT / "config" / "cf4_kf_roi_leakage_execution_v3.json"
DESIGN_V1_PATH = ROOT / "config" / "cf4_kf_bin_manifest_design_v1.json"
GRANT_V1_PATH = ROOT / "config" / "cf4_kf_roi_leakage_execution_v1.json"
GRANT_V2_PATH = ROOT / "config" / "cf4_kf_roi_leakage_execution_v2.json"
COMMIT = "a" * 40
TEST_NUMERICS = {
    "moment_order": 64,
    "u_order": 10,
    "v_period_samples": 32,
    "v_panel_order": 8,
    "q_period_samples": 32,
    "q_panel_order": 8,
    "parseval_tail_x": (256.0, 512.0),
}


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
    result = []
    for component in vector:
        value = (-component) % grid_size
        result.append(value - grid_size if value >= half else value)
    return tuple(result)


def _split_real_space_transform(q, radius, order=600):
    nodes, weights = leggauss(order)
    total = np.zeros_like(np.asarray(q, dtype=float))
    for lower, upper in ((0.0, 0.75 * radius), (0.75 * radius, radius)):
        r = lower + 0.5 * (upper - lower) * (nodes + 1.0)
        wr = 0.5 * (upper - lower) * weights
        total += 4.0 * math.pi * np.sum(
            wr[None, :]
            * r[None, :] ** 2
            * raised_cosine_window(r, radius)[None, :]
            * np.sinc(np.asarray(q)[:, None] * r[None, :] / math.pi),
            axis=1,
        )
    return total


def _tensor_shell_reference(a, b, c, d, cumulative, order=220):
    nodes, weights = leggauss(order)
    ko = a + 0.5 * (b - a) * (nodes + 1.0)
    ki = c + 0.5 * (d - c) * (nodes + 1.0)
    wo = 0.5 * (b - a) * weights
    wi = 0.5 * (d - c) * weights
    u = ko[:, None] + ki[None, :]
    v = ko[:, None] - ki[None, :]
    return float(
        np.sum(
            wo[:, None]
            * wi[None, :]
            * ko[:, None]
            * ki[None, :]
            * (cumulative(u) - cumulative(np.abs(v)))
        )
    )


def test_fft_squared_radius_counts_match_exact_small_lattice_enumeration():
    histogram, audit = squared_radius_histogram_fft(8)
    direct = _direct_squared_radius_histogram(8)
    assert audit["total_count_assertion_pass"]
    assert audit["full_lattice_count"] == 8**3
    assert {i: int(v) for i, v in enumerate(histogram) if v} == dict(direct)


def test_native_mode_counts_handle_conjugates_self_modes_and_dc_exactly():
    grid_size = 8
    half = grid_size // 2
    bins = _small_bins(grid_size)
    full, independent, audit = mode_counts_by_native_bin(grid_size, bins, 1.0)
    direct_full = np.zeros(len(bins), dtype=np.int64)
    direct_independent = np.zeros(len(bins), dtype=np.int64)
    for vector in itertools.product(range(-half, half), repeat=3):
        q = sum(component * component for component in vector)
        if q == 0 or q > half * half:
            continue
        index = _direct_native_bin(q, bins)
        direct_full[index] += 1
        if vector <= _canonical_conjugate(vector, grid_size):
            direct_independent[index] += 1
    assert self_conjugate_squared_radius_histogram(grid_size) == {16: 3}
    assert np.array_equal(full, direct_full)
    assert np.array_equal(independent, direct_independent)
    assert audit["DC_excluded"]


def test_design_v2_preserves_lattice_bins_and_predecessor_binding():
    design, digest = load_frozen_design(DESIGN_PATH)
    bins = build_native_bins(design)
    assert digest == FROZEN_DESIGN_SHA256
    assert len(bins) == 38 and sum(not item.terminal for item in bins) == 37
    assert bins[-1].upper == math.pi / 0.3
    assert design["predecessor"]["design_raw_sha256"] == hashlib.sha256(
        DESIGN_V1_PATH.read_bytes()
    ).hexdigest()
    assert design["diagnostic_output_domain"]["creates_DFT_lattice_modes"] is False
    assert design["native_bin_support_and_proposal"]["supported_suffix_through_terminal_required"] is False


def test_analytic_transform_matches_independent_split_quadrature_broad_qR():
    radius = 5.0
    beta = 4.0 * math.pi
    x = np.array([0.0, 1e-10, 1e-3, 0.1, 1.0, beta - 1e-8, beta, beta + 1e-8, 100.0, 650.0])
    q = x / radius
    analytic = sphere_radial_transform(q, radius)
    reference = _split_real_space_transform(q, radius)
    assert np.max(np.abs(analytic - reference)) / (4.0 * math.pi * radius**3) < 2e-12


def test_q_angular_integral_matches_independent_mu_reference():
    specification = WindowSpec(
        "union", 1.0, ((0.0, 0.0, 0.0), (3.2, 0.0, 0.0), (0.0, 4.1, 0.0)), 100.0
    )
    cumulative = QPowerCumulative(specification, 4.0, period_samples=48, panel_order=10)
    ko = np.array([[0.7], [1.0], [1.3]])
    ki = np.array([[0.7, 0.9, 1.4]])
    actual = cumulative.angular_integral(ko, ki)
    mu, weights = leggauss(500)
    q = np.sqrt(ko[:, :, None] ** 2 + ki[:, :, None] ** 2 - 2.0 * ko[:, :, None] * ki[:, :, None] * mu)
    power = sphere_radial_transform(q, specification.radius) ** 2
    power *= orientation_averaged_structure_factor(q, specification)
    reference = np.sum(weights * power, axis=2)
    assert actual == pytest.approx(reference, rel=2e-10, abs=2e-10)


def test_parseval_retained_for_sphere_and_union():
    specifications = (
        WindowSpec("sphere", 5.0, (), 384.0),
        WindowSpec("union", 2.0, ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)), 384.0),
    )
    for specification in specifications:
        audit = parseval_audit(specification, _window_moments(specification, 180), FINE_NUMERICS)
        assert audit["pass"]
        assert audit["relative_parseval_error"] < 1e-8
        assert audit["relative_finite_tail_increment"] < 1e-7


@pytest.mark.parametrize(
    ("intervals", "required_points"),
    [
        ((1.0, 2.0, 1.0, 2.0), {0.0}),
        ((0.0, 1.0, 1.0, 2.0), {-1.0}),
        ((1.0, 3.0, 2.0, 4.0), {-1.0}),
        ((2.0, 4.0, 1.0, 3.0), {1.0}),
    ],
)
def test_uv_branch_splits_cover_domain_without_gap_or_overlap(intervals, required_points):
    segments = uv_v_segments(*intervals)
    lower = intervals[0] - intervals[3]
    upper = intervals[1] - intervals[2]
    assert segments[0][0] == lower and segments[-1][1] == upper
    assert sum(second - first for first, second in segments) == pytest.approx(upper - lower)
    assert all(segments[index][1] == segments[index + 1][0] for index in range(len(segments) - 1))
    boundaries = {value for segment in segments for value in segment}
    assert required_points <= boundaries


def test_uv_mapping_constant_power_matches_analytic_and_tensor_reference():
    a, b, c, d = 0.4, 1.3, 0.7, 1.5
    cumulative = lambda q: np.asarray(q, dtype=float) ** 2 / 2.0
    actual = uv_shell_integral(
        a, b, c, d, cumulative, frequency_scale=2.0, numerics=TEST_NUMERICS
    )
    analytic = 2.0 * (b**3 - a**3) / 3.0 * (d**3 - c**3) / 3.0
    reference = _tensor_shell_reference(a, b, c, d, cumulative)
    assert actual == pytest.approx(analytic, rel=2e-13, abs=2e-13)
    assert actual == pytest.approx(reference, rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("window_key", ["sphere_R31", "union_R6_M4"])
def test_uv_mapping_narrow_or_oscillatory_window_matches_tensor_reference(window_key):
    design, _ = load_frozen_design(DESIGN_PATH)
    specifications, _ = _window_specs(design)
    specification = specifications[window_key]
    cumulative = QPowerCumulative(
        specification, 3.0, period_samples=48, panel_order=10
    )
    a, b, c, d = 0.85, 1.12, 0.91, 1.18
    actual = uv_shell_integral(
        a,
        b,
        c,
        d,
        cumulative,
        frequency_scale=_window_frequency_scale(specification),
        numerics=TEST_NUMERICS,
    )
    reference = _tensor_shell_reference(a, b, c, d, cumulative, order=320)
    assert actual == pytest.approx(reference, rel=2e-8, abs=1e-10)


def test_uv_shell_reciprocity_for_unequal_shells():
    specification = WindowSpec("sphere", 8.0, (), 100.0)
    cumulative = QPowerCumulative(specification, 4.0, period_samples=40, panel_order=10)
    first = uv_shell_integral(0.4, 0.9, 1.1, 1.7, cumulative, frequency_scale=16.0, numerics=TEST_NUMERICS)
    second = uv_shell_integral(1.1, 1.7, 0.4, 0.9, cumulative, frequency_scale=16.0, numerics=TEST_NUMERICS)
    assert first == pytest.approx(second, rel=2e-11, abs=1e-12)


def test_small_mixing_matrix_has_guards_and_reciprocity():
    bins = [
        NativeBin(0, 0.1, 0.3, math.sqrt(0.03), False),
        NativeBin(1, 0.3, 0.8, math.sqrt(0.24), False),
        NativeBin(2, 0.8, 2.0, math.sqrt(1.6), True),
    ]
    matrix, lower, upper, moments, audit = compute_mixing_matrix(
        bins, WindowSpec("sphere", 2.0, (), 20.0), numerics=TEST_NUMERICS
    )
    assert matrix.shape == (3, 3) and lower.shape == upper.shape == (3,)
    assert np.all(np.isfinite(matrix)) and np.all(matrix >= 0.0)
    assert audit["q_cumulative_max_h_Mpc"] == 6.0
    assert audit["analysis_reciprocity_pass"]
    assert audit["parseval_denominator"] == pytest.approx((2 * math.pi) ** 3 * moments["int_W2_dV"])


def test_guard_decomposition_and_highest_contiguous_interior_run():
    bins = [
        NativeBin(0, 1.0, 2.0, math.sqrt(2.0), False),
        NativeBin(1, 2.0, 3.0, math.sqrt(6.0), False),
        NativeBin(2, 3.0, 4.0, math.sqrt(12.0), True),
    ]
    matrix = np.diag([0.995, 0.995, 0.80])
    lower = np.array([0.002, 0.002, 0.05])
    upper = np.array([0.002, 0.002, 0.05])
    result = evaluate_support(matrix, lower, upper, [100, 100, 100], bins, {"V_eff": 1.0}, 1.0)
    containment, column, outside, total, far, decomposition, _, valid, supported, proposal = result
    assert np.allclose(outside, lower + upper + far)
    assert np.allclose(total, column + lower + upper)
    assert np.max(np.abs(decomposition)) < 1e-15 and np.all(valid)
    assert supported.tolist() == [True, True, False]
    assert proposal["deterministic_proposal"]["end_native_bin"] == 1
    assert proposal["terminal_failure_allowed"] is True


def test_run_selection_tiebreaks_more_bins_then_summed_modes():
    bins_more = [
        NativeBin(0, 1.0, 2.0, 1.4, False),
        NativeBin(1, 2.0, 5.0, 3.1, False),
        NativeBin(2, 4.0, 5.0, 4.5, True),
    ]
    proposal = maximal_contiguous_runs(np.array([True, True, False]), bins_more, [1, 1, 9])
    assert proposal["deterministic_proposal"]["native_bin_count"] == 2
    bins_modes = [
        NativeBin(0, 1.0, 5.0, 2.2, False),
        NativeBin(1, 2.0, 4.0, 2.8, False),
        NativeBin(2, 4.0, 5.0, 4.5, True),
    ]
    proposal = maximal_contiguous_runs(np.array([True, False, True]), bins_modes, [3, 0, 9])
    assert proposal["deterministic_proposal"]["start_native_bin"] == 2


def _synthetic_terminal_failure_evaluation():
    bins = [
        NativeBin(0, 1.0, 2.0, 1.4, False),
        NativeBin(1, 2.0, 3.0, 2.4, False),
        NativeBin(2, 3.0, 4.0, 3.4, True),
    ]
    matrix = np.diag([0.995, 0.995, 0.8])
    lower = np.array([0.002, 0.002, 0.05])
    upper = lower.copy()
    values = evaluate_support(matrix, lower, upper, [100] * 3, bins, {"V_eff": 1.0}, 1.0)
    containment, column, outside, total, far, _, neff, valid, supported, proposal = values
    audit = {
        "maximum_total_through_guard": float(np.max(total)),
        "sphere_A0_relative_error": 0.0,
        "parseval_q_space": {"relative_parseval_error": 0.0, "relative_finite_tail_increment": 0.0},
        "analysis_reciprocity_max_relative_unnormalized": 0.0,
        "guard_decomposition_max_abs_residual": 0.0,
    }
    return MixingEvaluation(matrix, lower, upper, far, total, containment, column, outside, neff, valid, supported, proposal, {"int_W1_dV": 1.0, "int_W2_dV": 1.0, "int_W4_dV": 1.0, "V_eff": 1.0}, audit)


def test_terminal_support_failure_does_not_invalidate_numerical_pass():
    evaluation = _synthetic_terminal_failure_evaluation()
    assert not evaluation.raw_supported[-1]
    comparison = _compare_evaluations({"window": evaluation}, {"window": evaluation})
    assert comparison["status"] == "PASS"
    assert comparison["contiguous_run_proposal_identical"]


def test_precheck_fail_prints_diagnostics_and_publishes_nothing(tmp_path, monkeypatch, capsys):
    output = tmp_path / "preflight"
    monkeypatch.setattr(leakage, "load_frozen_design", lambda _: ({}, FROZEN_DESIGN_SHA256))
    monkeypatch.setattr(leakage, "load_execution_grant", lambda *_: ({"scope": {"preflight_output": str(output)}}, "b" * 64))
    failed = {"schema": "ouruniv-cf4-kf-roi-leakage-result-v3", "status": "PRECHECK_FAIL", "mode": "preflight", "numerical_convergence": {"status": "FAIL", "per_numeric_window": {"sphere_R31": {"fine_far_tail": [-0.1]}}}}
    monkeypatch.setattr(leakage, "calculate", lambda *args, **kwargs: (failed, {}, {}))
    assert leakage.main(["--design", str(DESIGN_PATH), "--execution-grant", str(GRANT_PATH), "--output", str(output), "--mode", "preflight", "--implementation-commit", COMMIT]) == 2
    captured = capsys.readouterr()
    assert '"PRECHECK_FAIL"' in captured.out and '"PRECHECK_FAIL"' in captured.err
    assert not output.exists()


def _published_result(grant_sha):
    roi_template = {
        "analysis_column_sum": [0.98] * 38,
        "signed_outside_analysis_residual": [0.02] * 38,
        "lower_guard": [0.002] * 38,
        "upper_guard": [0.003] * 38,
        "far_tail": [0.015] * 38,
        "total_through_upper_guard": [0.985] * 38,
        "normalization_valid": [True] * 38,
        "contiguous_run_geometry_proposal": {"terminal_failure_allowed": True, "proposal_semantics": "geometry_window_only_not_scientific_frontier_or_observational_resolution"},
    }
    return {
        "schema": "ouruniv-cf4-kf-roi-leakage-result-v3",
        "status": "PRECHECK_PASS",
        "mode": "preflight",
        "design_raw_sha256": FROZEN_DESIGN_SHA256,
        "execution_grant_raw_sha256": grant_sha,
        "implementation_path": "src/cf4_kf_roi_leakage.py",
        "implementation_sha256": hashlib.sha256((ROOT / "src/cf4_kf_roi_leakage.py").read_bytes()).hexdigest(),
        "implementation_commit": COMMIT,
        "truth_or_candidate_data_consumed": False,
        "geometry_window_proposals_are_scientific_claims": False,
        "scientific_leakage_decision_authorized": False,
        "final_manifest_materialized": False,
        "k_boundary_claim_created": False,
        "frozen_coarse_numerics": json.loads(canonical_json_bytes(leakage.COARSE_NUMERICS)),
        "frozen_fine_numerics": json.loads(canonical_json_bytes(leakage.FINE_NUMERICS)),
        "numerical_convergence": {"status": "PASS", "all_analysis_column_normalizations_valid": True, "native_classification_identical": True, "contiguous_run_proposal_identical": True, "threshold_margin_safety_pass": True, "max_analysis_reciprocity_relative_error": 0.0, "max_guard_decomposition_abs_residual": 0.0},
        "ROI_results": [
            {**deepcopy(roi_template), "numeric_product_key": key}
            for key in (
                "sphere_R8",
                "sphere_R5",
                "sphere_R8",
                "union_R6_M4",
                "sphere_R31",
                "sphere_R8",
            )
        ],
    }


def test_deterministic_publication_and_v3_checker(tmp_path):
    grant = json.loads(GRANT_PATH.read_text())
    output = tmp_path / "preflight"
    grant["scope"]["preflight_output"] = str(output)
    grant_path = tmp_path / "grant.json"
    grant_bytes = canonical_json_bytes(grant)
    grant_path.write_bytes(grant_bytes)
    result = _published_result(hashlib.sha256(grant_bytes).hexdigest())
    counts = {
        "schema": "ouruniv-cf4-kf-roi-leakage-mode-counts-v3",
        "design_raw_sha256": FROZEN_DESIGN_SHA256,
        "native_bins": [
            {"lower_h_Mpc": float(index + 1), "upper_h_Mpc": float(index + 2)}
            for index in range(38)
        ],
        "count_audit": {"total_count_assertion_pass": True},
    }
    arrays = {}
    for key in ("sphere_R5", "sphere_R8", "sphere_R31", "union_R6_M4"):
        arrays[f"coarse__{key}"] = 0.98 * np.eye(38)
        arrays[f"fine__{key}"] = 0.98 * np.eye(38)
        for order in ("coarse", "fine"):
            arrays[f"{order}_lower_guard__{key}"] = np.full(38, 0.002)
            arrays[f"{order}_upper_guard__{key}"] = np.full(38, 0.003)
            arrays[f"{order}_far_tail__{key}"] = np.full(38, 0.015)
    assert deterministic_npz_bytes(arrays) == deterministic_npz_bytes(dict(reversed(list(arrays.items()))))
    publish_artifacts(output, result, counts, arrays)
    audit = audit_directory(output, DESIGN_PATH, grant_path, COMMIT)
    assert audit["precheck_status"] == "PRECHECK_PASS"


def test_design_and_grant_tamper_fail_closed(tmp_path):
    design = json.loads(DESIGN_PATH.read_text())
    design["analysis_lattice"]["grid_spacing_cMpc_h"] = 0.31
    design_path = tmp_path / "design.json"
    design_path.write_bytes(canonical_json_bytes(design))
    with pytest.raises(LeakageError, match="frozen design SHA256 mismatch"):
        load_frozen_design(design_path)
    grant = json.loads(GRANT_PATH.read_text())
    grant["numerical_contract"]["fine"]["v_period_samples"] = 33
    grant_path = tmp_path / "grant.json"
    grant_path.write_bytes(canonical_json_bytes(grant))
    with pytest.raises(LeakageError, match="frozen fine numerics mismatch"):
        load_execution_grant(grant_path, FROZEN_DESIGN_SHA256)


def test_v3_grant_is_single_preflight_and_prior_files_unchanged():
    grant, digest = load_execution_grant(GRANT_PATH, FROZEN_DESIGN_SHA256)
    assert len(digest) == 64
    assert grant["authorization"]["maximum_preflight_submissions"] == 1
    assert grant["authorization"]["Slurm_production_authorized"] is False
    assert grant["scope"]["output_root"] == "/gpfs/kjhan/CF4/kf_design/roi_leakage_v3"
    assert hashlib.sha256(DESIGN_V1_PATH.read_bytes()).hexdigest() == "76b71a482a1d92b146e335e231c5b4430f06df009566f22ce1efb739c5c96da9"
    assert hashlib.sha256(GRANT_V1_PATH.read_bytes()).hexdigest() == "4511916c16b77e39985b7bfc22230d7773e02799cc8cccc08159d0f8f39586a9"
    assert hashlib.sha256(GRANT_V2_PATH.read_bytes()).hexdigest() == "86c39590be609c47f0333026d9eedcfcb71ba33a4b2309caca9e6b687b6d1607"
