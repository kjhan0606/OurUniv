import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from cf4_all_parent_peak_evidence import (
    effective_sample_size,
    evidence_from_means,
    failure_classification,
    joint_importance_diagnostics,
    logsumexp,
    normalized_weights,
    one_sided_weighted_ks_statistic,
    prepare_parent_cases,
    validate_program_contract,
    validate_reference_calibration,
    weighted_ks_permutation_test,
    weighted_quantile,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_logsumexp_and_weights_remain_stable_for_extreme_evidence():
    values = np.asarray([-1200.0, -1000.0, -1001.0])
    actual = logsumexp(values)
    expected = -1000.0 + np.log(1.0 + np.exp(-1.0) + np.exp(-200.0))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)
    weights = normalized_weights(values)
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-14)
    assert np.argmax(weights) == 1
    assert effective_sample_size(weights) < 2.0


def test_effective_sample_size_is_exact_for_uniform_and_delta_weights():
    assert effective_sample_size(np.ones(8)) == 8.0
    delta = np.zeros(8)
    delta[3] = 1.0
    assert effective_sample_size(delta) == 1.0


def test_joint_support_records_global_and_each_parent_geometry_ess():
    values = np.zeros((4, 8))
    result = joint_importance_diagnostics(values)
    assert result["joint_parent_geometry_ESS"] == 32.0
    assert result["maximum_joint_parent_geometry_weight"] == 1.0 / 32.0
    np.testing.assert_array_equal(result["parent_geometry_ESS"], np.full(4, 8.0))


def test_weighted_quantile_uses_normalized_cumulative_mass():
    values = np.asarray([4.0, 1.0, 3.0, 2.0])
    weights = np.asarray([0.1, 0.6, 0.2, 0.1])
    assert weighted_quantile(values, weights, 0.50) == 1.0
    assert weighted_quantile(values, weights, 0.65) == 2.0
    assert weighted_quantile(values, weights, 0.90) == 3.0
    assert weighted_quantile(values, weights, 1.00) == 4.0


def test_one_sided_weighted_ks_detects_high_deviance_shift_and_is_reproducible():
    values = np.arange(8, dtype=float)
    uniform = np.ones(8)
    high_shift = np.asarray([0.01, 0.01, 0.01, 0.01, 0.04, 0.08, 0.24, 0.60])
    assert one_sided_weighted_ks_statistic(values, uniform) == 0.0
    assert one_sided_weighted_ks_statistic(values, high_shift) > 0.5
    first = weighted_ks_permutation_test(values, high_shift, 1000, 2026081901)
    second = weighted_ks_permutation_test(values, high_shift, 1000, 2026081901)
    assert first == second
    assert first["permutation_pvalue"] < 0.05


def test_failure_classification_prioritizes_invalid_and_mc_before_tension():
    assert failure_classification(False, True, True, True, False, False) \
        == "invalid_numerical_or_lineage"
    assert failure_classification(True, False, True, True, False, False) \
        == "invalid_numerical_or_lineage"
    assert failure_classification(True, True, False, True, False, False) \
        == "Monte_Carlo_or_proposal_instability"
    assert failure_classification(True, True, True, False, False, False) \
        == "Monte_Carlo_or_proposal_instability"
    assert failure_classification(True, True, True, True, False, True) \
        == "parent_support_or_CF4_compatibility"
    assert failure_classification(True, True, True, True, True, True) is None


def test_evidence_from_means_includes_normalization_and_midpoint_ratio():
    means = np.asarray([[0.1, -0.2], [0.3, 0.4]])
    geometries = [
        {
            "targets": np.asarray([0.5, -0.1]),
            "log_midpoint_target_over_proposal": -0.7,
        },
        {
            "targets": np.asarray([0.2, 0.9]),
            "log_midpoint_target_over_proposal": 0.2,
        },
    ]
    covariance = [
        np.asarray([[0.8, 0.1], [0.1, 0.6]]),
        np.asarray([[0.5, 0.0], [0.0, 0.9]]),
    ]
    cholesky = [np.linalg.cholesky(value) for value in covariance]
    logdet = np.asarray([
        np.linalg.slogdet(value)[1] for value in covariance
    ])
    log_importance, rows = evidence_from_means(
        means, geometries, cholesky, logdet
    )
    for index in range(2):
        residual = geometries[index]["targets"] - means[index]
        expected_log_z = -0.5 * (
            2 * np.log(2.0 * np.pi)
            + logdet[index]
            + residual @ np.linalg.solve(covariance[index], residual)
        )
        np.testing.assert_allclose(rows[index]["log_Z_peak"], expected_log_z)
        np.testing.assert_allclose(
            log_importance[index],
            expected_log_z
            + geometries[index]["log_midpoint_target_over_proposal"],
        )


def test_parent_lineage_requires_every_seed_once_and_in_order():
    calibration = {
        "reference_field_hashes": [
            {"seed": 3, "path": "three", "sha256": "c"},
            {"seed": 1, "path": "one", "sha256": "a"},
            {"seed": 2, "path": "two", "sha256": "b"},
        ]
    }
    program = {
        "parents": {"seed_range_inclusive": [1, 3], "count": 3}
    }
    rows = prepare_parent_cases(calibration, program)
    assert [row["seed"] for row in rows] == [1, 2, 3]


def test_prerequisite_status_authorization_and_canonical_output_are_hard_gates():
    program = json.loads((
        ROOT / "config/cf4_all_parent_peak_evidence_program.json"
    ).read_text())
    canonical = Path(program["storage"]["canonical_output"])
    validate_program_contract(program, canonical)
    bad_status = deepcopy(program)
    bad_status["status"] = "draft"
    with np.testing.assert_raises_regex(RuntimeError, "not in the frozen status"):
        validate_program_contract(bad_status, canonical)
    with np.testing.assert_raises_regex(RuntimeError, "not the frozen canonical"):
        validate_program_contract(program, canonical.with_name("other.json"))

    calibration_path = Path(program["reference_calibration"]["path"])
    calibration = json.loads(calibration_path.read_text())
    validate_reference_calibration(calibration, program)
    bad_calibration = deepcopy(calibration)
    bad_calibration["two_chain_audit"]["all_pass"] = False
    with np.testing.assert_raises_regex(RuntimeError, "two-chain audit"):
        validate_reference_calibration(bad_calibration, program)


def test_all_parent_program_is_hash_pinned_and_requires_every_parent():
    program = json.loads((
        ROOT / "config/cf4_all_parent_peak_evidence_program.json"
    ).read_text())
    for item in program["pinned_local_files"]:
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    assert program["parents"]["seed_range_inclusive"] == [3193, 3448]
    assert program["parents"]["count"] == 256
    assert program["parents"]["subset_or_result_dependent_replication_allowed"] is False
    integration = program["integration"]
    assert integration["draw_count"] == 64
    assert integration["common_random_numbers"] is True
    assert integration["reuse_identical_draws_for_every_parent"] is True
    assert program["execution"]["worker_processes"] == 8
    assert program["execution"]["threads_per_worker"] == 1


def test_all_parent_program_keeps_fields_and_simulations_closed():
    program = json.loads((
        ROOT / "config/cf4_all_parent_peak_evidence_program.json"
    ).read_text())
    firewall = program["information_firewall"]
    assert firewall["evidence_only"] is True
    assert firewall["new_N576_null_field_drawn"] is False
    assert firewall["conditional_or_candidate_field_generated"] is False
    assert firewall["PM_or_halo_finder_run"] is False
    assert firewall["parent_or_seed_selection_allowed"] is False
    assert firewall["RAMSES_authorized"] is False
    gates = program["gates"]
    assert gates["parent_ESS_min"] == 32.0
    assert gates["maximum_parent_weight_max"] == 0.10
    assert gates["joint_parent_geometry_ESS_min"] == 128.0
    assert gates["maximum_joint_parent_geometry_weight_max"] == 0.025
    assert gates["weighted_CF4_one_sided_KS_permutation_p_min"] == 0.01
    assert gates["pass_requires_all"] is True


def test_v1_result_record_seals_failure_without_opening_fields():
    record = json.loads((
        ROOT / "config/cf4_all_parent_peak_evidence_v1_result_record.json"
    ).read_text())
    assert record["status"] == "complete_fail_Monte_Carlo_or_proposal_instability"
    lineage = record["lineage"]
    assert sha256_file(ROOT / lineage["program"]) == lineage["program_sha256"]
    assert sha256_file(ROOT / lineage["implementation"]) \
        == lineage["implementation_sha256"]
    decision = record["decision"]
    assert decision["conditional_field_bank_authorized"] is False
    assert decision["candidate_generation_authorized"] is False
    assert decision["parent_or_seed_selection_authorized"] is False
    assert decision["PM_or_halo_finder_authorized"] is False
    assert decision["RAMSES_authorized"] is False
    canonical = Path(lineage["canonical_result"])
    if canonical.exists():
        assert sha256_file(canonical) == lineage["canonical_result_sha256"]


def test_runner_is_marker_driven_hash_pinned_and_resource_bounded():
    runner = (ROOT / "scripts/run_cf4_all_parent_peak_evidence_lageunha.sh").read_text()
    launcher = (ROOT / "scripts/launch_cf4_all_parent_peak_evidence_lageunha.sh").read_text()
    status = (ROOT / "scripts/status_cf4_all_parent_peak_evidence.sh").read_text()
    combined = "\n".join((runner, launcher, status)).lower()
    assert "expected_program_sha=" in runner
    assert "expected_implementation_sha=" in runner
    assert "git -c \"$repo\" diff --quiet head" in runner.lower()
    assert "unexpected result schema" in runner
    assert "authorization flags violate the frozen firewall" in runner
    assert '"${host_short,,}" != "$expected_host"' in runner
    assert "worker_processes=8" in runner
    assert "threads_per_worker=1" in runner
    assert "process_table_polling" not in combined
    assert "pgrep" not in combined
    assert "postgres" not in combined
    assert "while " not in combined
    assert "sleep " not in combined
