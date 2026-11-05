import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

from cf4_adaptive_geometry_proposal import (
    draw_adaptation_geometry,
    fit_cross_validated_mixture,
)
from cf4_peak_evidence_adaptation import (
    adaptation_failure_classification,
    atomic_npz,
    logsumexp_over_parents,
    validate_program_contract,
    vectorized_log_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program():
    return json.loads((
        ROOT / "config/cf4_peak_evidence_adaptation_program.json"
    ).read_text())


def test_vectorized_log_evidence_matches_scalar_cholesky_solves():
    rng = np.random.default_rng(11)
    draws, dimension = 17, 6
    matrix = rng.normal(size=(draws, dimension, dimension))
    covariance = matrix @ np.swapaxes(matrix, 1, 2) + np.eye(dimension)[None] * 0.4
    cholesky = np.linalg.cholesky(covariance)
    logdet = 2.0 * np.sum(
        np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1
    )
    means = rng.normal(size=(draws, dimension))
    targets = rng.normal(size=(draws, dimension))
    actual, quadratic = vectorized_log_evidence(
        means, targets, cholesky, logdet
    )
    expected = []
    expected_quadratic = []
    for index in range(draws):
        residual = targets[index] - means[index]
        whitened = np.linalg.solve(cholesky[index], residual)
        value = float(whitened @ whitened)
        expected_quadratic.append(value)
        expected.append(-0.5 * (
            dimension * math.log(2.0 * math.pi) + logdet[index] + value
        ))
    np.testing.assert_allclose(quadratic, expected_quadratic, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_parent_logmeanexp_is_stable_and_parent_uniform():
    values = np.asarray([
        [-1001.0, -10.0, -4.0],
        [-1000.0, -11.0, -4.0],
        [-1200.0, -12.0, -4.0],
    ])
    actual = logsumexp_over_parents(values) - math.log(3.0)
    expected = np.asarray([
        -1000.0 + math.log(1.0 + math.exp(-1.0) + math.exp(-200.0)) - math.log(3.0),
        math.log((math.exp(-10.0) + math.exp(-11.0) + math.exp(-12.0)) / 3.0),
        -4.0,
    ])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)


def test_adaptation_failure_priority_blocks_fallback_on_numerical_failure():
    assert adaptation_failure_classification(
        True, True, False, False, False, False, False
    ) == "invalid_numerical_or_lineage"
    assert adaptation_failure_classification(
        True, True, True, False, False, False, False
    ) == "insufficient_adaptation_support_fallback_authorized"
    assert adaptation_failure_classification(
        True, True, True, True, False, False, False
    ) == "proposal_family_fit_or_validation_failure"
    assert adaptation_failure_classification(
        True, True, True, True, True, True, True
    ) is None


def test_adaptation_draw_uses_individual_pcg64dxsm_seed_sequences():
    v8 = json.loads((ROOT / "config/p2_lg_z0_forward_importance_v8.json").read_text())
    peak = v8["peak_constraints"]
    first = [draw_adaptation_geometry(peak, 2026082001, i) for i in range(20)]
    second = [draw_adaptation_geometry(peak, 2026082001, i) for i in range(20)]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(
            left["midpoint_offset_mpc_h"], right["midpoint_offset_mpc_h"]
        )
        np.testing.assert_array_equal(left["axis"], right["axis"])
        assert left["proposal_component"] == right["proposal_component"]
        assert left["log_target_over_proposal"] == right["log_target_over_proposal"]
    assert not np.array_equal(first[0]["axis"], first[1]["axis"])


def test_compact_npz_is_exclusive_and_roundtrips(tmp_path):
    path = tmp_path / "arrays.npz"
    arrays = {
        "log_Z_peak": np.arange(24, dtype=np.float64).reshape(3, 8),
        "axis": np.eye(3),
    }
    atomic_npz(path, arrays)
    with np.load(path, allow_pickle=False) as actual:
        np.testing.assert_array_equal(actual["log_Z_peak"], arrays["log_Z_peak"])
        np.testing.assert_array_equal(actual["axis"], arrays["axis"])
    with np.testing.assert_raises(FileExistsError):
        atomic_npz(path, arrays)


def test_adaptation_program_is_hash_pinned_and_canonical():
    program = load_program()
    for item in program["pinned_local_files"]:
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    storage = program["storage"]
    validate_program_contract(
        program,
        Path(storage["canonical_output"]),
        Path(storage["canonical_arrays"]),
        Path(storage["canonical_proposal"]),
    )
    with np.testing.assert_raises_regex(RuntimeError, "not canonical"):
        validate_program_contract(
            program,
            Path(storage["canonical_output"]).with_name("other.json"),
            Path(storage["canonical_arrays"]),
            Path(storage["canonical_proposal"]),
        )


def test_fitter_api_cannot_receive_parent_identity_or_cf4_metrics():
    parameters = set(inspect.signature(fit_cross_validated_mixture).parameters)
    assert parameters == {
        "midpoint", "axes", "log_weights", "target_log_density", "master_seed"
    }
    program = load_program()
    firewall = program["information_firewall"]
    assert firewall["parent_seed_or_identity_passed_to_fitter"] is False
    assert firewall["parent_specific_Z_passed_to_fitter"] is False
    assert firewall["CF4_deviance_or_rank_loaded_for_fitting"] is False


def test_adaptation_program_freezes_support_and_downstream_gates():
    program = load_program()
    assert program["adaptation_bank"]["draw_count"] == 2048
    assert program["adaptation_bank"]["master_seed"] == 2026082001
    gates = program["gates"]
    assert gates["all_524288_log_Z_finite"] is True
    assert gates["real_scalar_vectorized_log_Z_and_log_weight_max_difference"] == 1e-10
    assert gates["geometry_marginal_ESS_min"] == 128.0
    assert gates["maximum_normalized_geometry_weight_max"] == 0.025
    decision = program["decision"]
    assert decision["conditional_field_bank_authorized"] is False
    assert decision["candidate_generation_authorized"] is False
    assert decision["PM_or_RAMSES_authorized"] is False


def test_adaptation_scripts_are_hash_pinned_marker_driven_and_bounded():
    runner = (ROOT / "scripts/run_cf4_peak_evidence_adaptation_lageunha.sh").read_text()
    launcher = (ROOT / "scripts/launch_cf4_peak_evidence_adaptation_lageunha.sh").read_text()
    status = (ROOT / "scripts/status_cf4_peak_evidence_adaptation.sh").read_text()
    combined = "\n".join((runner, launcher, status)).lower()
    assert "expected_program_sha=" in runner
    assert "expected_implementation_sha=" in runner
    assert "expected_proposal_implementation_sha=" in runner
    assert 'scripts/run_cf4_peak_evidence_adaptation_lageunha.sh' in runner
    assert "runner_sha256=%s" in runner
    assert '"${host_short,,}" != "$expected_host"' in runner
    assert "unexpected adaptation result schema" in runner
    assert "adaptation opened a forbidden downstream authorization" in runner
    assert "worker_processes=8" in runner
    assert "threads_per_worker=1" in runner
    assert "pgrep" not in combined
    assert "postgres" not in combined
    assert "while " not in combined
    assert "sleep " not in combined
