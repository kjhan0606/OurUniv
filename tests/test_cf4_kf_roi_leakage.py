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
    NativeBin,
    QPowerCumulative,
    WindowSpec,
    _window_moments,
    build_native_bins,
    canonical_json_bytes,
    compute_mixing_matrix,
    deterministic_npz_bytes,
    evaluate_support,
    load_execution_grant,
    load_frozen_design,
    mode_counts_by_native_bin,
    orientation_averaged_structure_factor,
    parseval_audit,
    publish_artifacts,
    raised_cosine_window,
    self_conjugate_squared_radius_histogram,
    sphere_radial_transform,
    squared_radius_histogram_fft,
    supported_suffix,
)
from check_cf4_kf_roi_leakage_v2 import audit_directory  # noqa: E402


DESIGN_PATH = ROOT / "config" / "cf4_kf_bin_manifest_design_v1.json"
GRANT_PATH = ROOT / "config" / "cf4_kf_roi_leakage_execution_v2.json"
V1_GRANT_PATH = ROOT / "config" / "cf4_kf_roi_leakage_execution_v1.json"
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
    result = []
    for component in vector:
        value = (-component) % grid_size
        result.append(value - grid_size if value >= half else value)
    return tuple(result)


def _split_real_space_transform(q, radius, order=700):
    nodes, weights = leggauss(order)
    total = np.zeros_like(np.asarray(q, dtype=float))
    for lower, upper in ((0.0, 0.75 * radius), (0.75 * radius, radius)):
        r = lower + 0.5 * (upper - lower) * (nodes + 1.0)
        wr = 0.5 * (upper - lower) * weights
        window = raised_cosine_window(r, radius)
        total += 4.0 * math.pi * np.sum(
            wr[None, :] * r[None, :] ** 2 * window[None, :]
            * np.sinc(np.asarray(q)[:, None] * r[None, :] / math.pi),
            axis=1,
        )
    return total


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


def test_frozen_native_edges_have_37_complete_plus_terminal_bin():
    design, digest = load_frozen_design(DESIGN_PATH)
    bins = build_native_bins(design)
    assert digest == FROZEN_DESIGN_SHA256
    assert len(bins) == 38
    assert sum(not item.terminal for item in bins) == 37
    assert bins[-1].terminal and bins[-1].upper == math.pi / 0.3


def test_analytic_transform_matches_independent_split_quadrature_broad_qR():
    radius = 5.0
    beta = 4.0 * math.pi
    x = np.array(
        [0.0, 1e-10, 1e-3, 0.1, 1.0, beta - 1e-8, beta, beta + 1e-8, 100.0, 650.0]
    )
    q = x / radius
    analytic = sphere_radial_transform(q, radius)
    reference = _split_real_space_transform(q, radius)
    volume_scale = 4.0 * math.pi * radius**3
    assert np.max(np.abs(analytic - reference)) / volume_scale < 2e-12


def test_disjoint_union_structure_factor_at_zero_is_multiplicity_squared():
    specification = WindowSpec(
        "union", 1.0, ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0)), 20.0
    )
    assert orientation_averaged_structure_factor(np.array([0.0]), specification)[0] == pytest.approx(9.0)


@pytest.mark.parametrize(
    "specification",
    [
        WindowSpec("sphere", 2.0, (), 100.0),
        WindowSpec(
            "union",
            1.0,
            ((0.0, 0.0, 0.0), (3.2, 0.0, 0.0), (0.0, 4.1, 0.0)),
            100.0,
        ),
    ],
)
def test_q_variable_angular_integral_matches_independent_mu_reference(specification):
    cumulative = QPowerCumulative(
        specification, 4.0, period_samples=48, panel_order=10
    )
    ko = np.array([[0.7], [1.0], [1.3]])
    ki = np.array([[0.7, 0.9, 1.4]])
    actual = cumulative.angular_integral(ko, ki)
    mu, weights = leggauss(500)
    q = np.sqrt(
        ko[:, :, None] ** 2
        + ki[:, :, None] ** 2
        - 2.0 * ko[:, :, None] * ki[:, :, None] * mu[None, None, :]
    )
    power = sphere_radial_transform(q, specification.radius) ** 2
    power *= orientation_averaged_structure_factor(q, specification)
    reference = np.sum(weights[None, None, :] * power, axis=2)
    assert actual == pytest.approx(reference, rel=2e-10, abs=2e-10)


@pytest.mark.parametrize(
    "specification",
    [
        WindowSpec("sphere", 5.0, (), 384.0),
        WindowSpec(
            "union", 2.0, ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)), 384.0
        ),
    ],
)
def test_q_space_parseval_matches_independent_real_space_moment(specification):
    moments = _window_moments(specification, 180)
    audit = parseval_audit(specification, moments, FINE_NUMERICS)
    assert audit["pass"]
    assert audit["relative_parseval_error"] < 1e-8
    assert audit["relative_finite_tail_increment"] < 1e-7


def test_small_mixing_matrix_is_finite_and_uses_q_angular_normalization():
    bins = [
        NativeBin(0, 0.1, 0.3, math.sqrt(0.03), False),
        NativeBin(1, 0.3, 0.8, math.sqrt(0.24), False),
        NativeBin(2, 0.8, 2.0, math.sqrt(1.6), True),
    ]
    matrix, moments, audit = compute_mixing_matrix(
        bins,
        WindowSpec("sphere", 2.0, (), 20.0),
        numerics={
            "moment_order": 64,
            "shell_order": 6,
            "q_period_samples": 24,
            "q_panel_order": 8,
            "parseval_tail_x": (256.0, 512.0),
        },
    )
    assert matrix.shape == (3, 3)
    assert np.all(np.isfinite(matrix)) and np.all(matrix >= 0.0)
    assert audit["analytic_radial_transform"]
    assert audit["parseval_denominator"] == pytest.approx((2 * math.pi) ** 3 * moments["int_W2_dV"])
    assert audit["parseval_q_space"]["pass"]


def test_over_unity_column_is_diagnosed_without_clipping_or_early_abort():
    matrix = np.diag([1.01, 0.95])
    result = evaluate_support(matrix, [100, 100], {"V_eff": 1000.0}, 10.0)
    containment, column_sum, residual, _, valid, supported, _ = result
    assert containment[0] == pytest.approx(1.01)
    assert column_sum[0] == pytest.approx(1.01)
    assert residual[0] == pytest.approx(-0.01)
    assert valid.tolist() == [False, True]
    assert not supported[0]


@pytest.mark.parametrize(
    ("mask", "expected", "reason"),
    [
        ([False, False, True, True], True, "contiguous_supported_suffix_through_Nyquist"),
        ([False, True, False, True], False, "supported_suffix_hole_fail_closed"),
        ([False, False, False], False, "no_supported_native_bin"),
    ],
)
def test_supported_suffix_is_fail_closed_on_holes(mask, expected, reason):
    result = supported_suffix(np.asarray(mask, dtype=np.bool_))
    assert result["pass"] is expected
    assert result["reason"] == reason


def test_precheck_fail_prints_diagnostics_and_publishes_nothing(tmp_path, monkeypatch, capsys):
    output = tmp_path / "preflight"
    monkeypatch.setattr(leakage, "load_frozen_design", lambda _: ({}, FROZEN_DESIGN_SHA256))
    monkeypatch.setattr(
        leakage,
        "load_execution_grant",
        lambda *_: ({"scope": {"preflight_output": str(output)}}, "b" * 64),
    )
    failed = {
        "schema": "ouruniv-cf4-kf-roi-leakage-result-v2",
        "status": "PRECHECK_FAIL",
        "mode": "preflight",
        "numerical_convergence": {
            "status": "FAIL",
            "per_numeric_window": {"sphere_R5": {"fine_column_sum": [1.01]}},
        },
    }
    monkeypatch.setattr(leakage, "calculate", lambda *args, **kwargs: (failed, {}, {}))
    assert leakage.main(
        [
            "--design", str(DESIGN_PATH), "--execution-grant", str(GRANT_PATH),
            "--output", str(output), "--mode", "preflight",
            "--implementation-commit", COMMIT,
        ]
    ) == 2
    captured = capsys.readouterr()
    assert '"PRECHECK_FAIL"' in captured.out and '"PRECHECK_FAIL"' in captured.err
    assert not output.exists()


def test_publish_refuses_overwrite_and_rejects_failed_precheck(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(Exception, match="passing|overwrite"):
        publish_artifacts(output, {"status": "PRECHECK_FAIL"}, {}, {})
    assert list(tmp_path.iterdir()) == [output]


def _published_result(grant_sha):
    roi = {
        "analysis_column_sum": [0.99] * 38,
        "signed_outside_analysis_residual": [0.01] * 38,
        "normalization_valid": [True] * 38,
        "numerical_audit": {"parseval_q_space": {"pass": True}},
    }
    return {
        "schema": "ouruniv-cf4-kf-roi-leakage-result-v2",
        "status": "PRECHECK_PASS",
        "mode": "preflight",
        "design_raw_sha256": FROZEN_DESIGN_SHA256,
        "execution_grant_raw_sha256": grant_sha,
        "implementation_path": "src/cf4_kf_roi_leakage.py",
        "implementation_sha256": hashlib.sha256((ROOT / "src/cf4_kf_roi_leakage.py").read_bytes()).hexdigest(),
        "implementation_commit": COMMIT,
        "truth_or_candidate_data_consumed": False,
        "scientific_leakage_decision_authorized": False,
        "final_manifest_materialized": False,
        "k_boundary_claim_created": False,
        "overall_leakage_gate_pass": False,
        "scientific_disposition": "numerical_PRECHECK_only_no_scientific_leakage_decision_authorized",
        "numerical_convergence": {
            "status": "PASS",
            "all_analysis_column_normalizations_valid": True,
        },
        "ROI_results": [deepcopy(roi) for _ in range(6)],
    }


def test_deterministic_publication_and_v2_checker(tmp_path):
    grant = json.loads(GRANT_PATH.read_text())
    output = tmp_path / "preflight"
    grant["scope"]["preflight_output"] = str(output)
    grant_path = tmp_path / "grant.json"
    grant_bytes = canonical_json_bytes(grant)
    grant_path.write_bytes(grant_bytes)
    result = _published_result(hashlib.sha256(grant_bytes).hexdigest())
    counts = {
        "schema": "ouruniv-cf4-kf-roi-leakage-mode-counts-v2",
        "design_raw_sha256": FROZEN_DESIGN_SHA256,
        "native_bins": [{} for _ in range(38)],
        "count_audit": {"total_count_assertion_pass": True},
    }
    arrays = {"fine__sphere": np.eye(38), "coarse__sphere": np.eye(38)}
    assert deterministic_npz_bytes(arrays) == deterministic_npz_bytes(dict(reversed(list(arrays.items()))))
    publish_artifacts(output, result, counts, arrays)
    audit = audit_directory(output, DESIGN_PATH, grant_path, COMMIT)
    assert audit["precheck_status"] == "PRECHECK_PASS"
    assert audit["scientific_leakage_decision_authorized"] is False

    authority_failures = (
        ("Slurm_preflight_authorized", False, "does not authorize the v2 Slurm preflight"),
        ("maximum_preflight_submissions", 2, "exactly one preflight"),
        ("scientific_leakage_decision_authorized", True, "forbidden authority"),
    )
    for field, value, message in authority_failures:
        altered = deepcopy(grant)
        altered["authorization"][field] = value
        grant_path.write_bytes(canonical_json_bytes(altered))
        with pytest.raises(Exception, match=message):
            audit_directory(output, DESIGN_PATH, grant_path, COMMIT)


def test_v2_grant_is_single_preflight_only_and_preserves_v1():
    grant, digest = load_execution_grant(GRANT_PATH, FROZEN_DESIGN_SHA256)
    assert len(digest) == 64
    assert grant["failed_predecessor"]["Slurm_job_id"] == 327817
    assert grant["failed_predecessor"]["canonical_output_or_COMPLETE_published"] is False
    assert grant["authorization"]["maximum_preflight_submissions"] == 1
    assert grant["authorization"]["Slurm_production_authorized"] is False
    assert grant["scope"]["output_root"] == "/gpfs/kjhan/CF4/kf_design/roi_leakage_v2"
    assert hashlib.sha256(V1_GRANT_PATH.read_bytes()).hexdigest() == "4511916c16b77e39985b7bfc22230d7773e02799cc8cccc08159d0f8f39586a9"


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("coarse", "q_period_samples", 17, "frozen coarse numerics mismatch"),
        ("fine", "parseval_tail_x", [512.0, 1025.0], "frozen fine numerics mismatch"),
    ],
)
def test_grant_loader_rejects_any_frozen_numerical_change(
    tmp_path, section, field, value, message
):
    grant = json.loads(GRANT_PATH.read_text())
    grant["numerical_contract"][section][field] = value
    path = tmp_path / "altered_grant.json"
    path.write_bytes(canonical_json_bytes(grant))
    with pytest.raises(LeakageError, match=message):
        load_execution_grant(path, FROZEN_DESIGN_SHA256)


def test_grant_loader_requires_scientific_decision_authority_exact_false(tmp_path):
    grant = json.loads(GRANT_PATH.read_text())
    grant["authorization"]["scientific_leakage_decision_authorized"] = True
    path = tmp_path / "altered_grant.json"
    path.write_bytes(canonical_json_bytes(grant))
    with pytest.raises(LeakageError, match="forbidden downstream authority"):
        load_execution_grant(path, FROZEN_DESIGN_SHA256)
