import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "config/cf4_datum_bearing_z0_phaseb_smoke_v3_result_record.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    return json.loads(RECORD.read_text())


def test_phase_b_pass_is_technical_only():
    record = load()
    assert record["status"] == "PASS_PHASE_B_TECHNICAL_SMOKE_STOP_BEFORE_PHASE_C"
    disposition = record["scientific_disposition"]
    assert disposition["Phase_B"] == "PASS_TECHNICAL_ONLY"
    assert disposition["present_day_observational_density_posterior"] == "NOT_CREATED"
    assert disposition["present_day_observational_velocity_posterior"] == "NOT_CREATED"
    assert disposition["observational_frontier_or_resolution_claim"] == "NOT_ALLOWED"
    assert disposition["target_0p3_cMpc_h_claim"] == "NOT_ALLOWED"
    assert disposition["Phase_C"] == "BLOCKED_PENDING_SEPARATE_USER_APPROVAL"


def test_no_gate_was_relaxed_and_all_v3_gates_passed():
    record = load()
    assert record["failure_lineage"]["gate_relaxation_after_failure"] is False
    gate = record["gate_result"]
    assert gate["passed_gate_count"] == gate["total_gate_count"] == 13
    assert gate["failed_gates"] == []
    assert gate["RSD_mass_relative_error"] <= gate["RSD_mass_relative_error_max"]
    assert gate["RSD_adjoint_relative_error"] <= gate["RSD_adjoint_relative_error_max"]
    assert max(gate["directional_gradient_best_relative_errors"]) <= gate[
        "directional_gradient_relative_error_max"
    ]
    assert gate["radial_forward_relative_error"] <= gate["radial_forward_relative_error_max"]
    assert gate["RSD_max_abs_displacement_cMpc_h"] < gate["boundary_strict_max_cMpc_h"]


def test_mock_and_validation_firewalls_remained_closed():
    firewall = load()["mock_firewall"]
    assert firewall["truth_seed_reused"] == 2026083000
    assert firewall["new_truth_seed_consumed"] is False
    assert firewall["validation_seed_consumed"] is False
    assert firewall["actual_2Mpp_count_arrays_read"] is False
    assert firewall["actual_CF4_velocity_datum_used"] is False
    assert firewall["seed_or_parent_ranking"] is False


def test_mechanics_are_not_misreported_as_convergence():
    mechanics = load()["mechanics_diagnostics"]
    assert mechanics["optimizer_iterations"] == 32
    assert mechanics["optimizer_status"] == 1
    assert mechanics["optimizer_convergence_claim"] is False
    assert mechanics["HMC_transition_count"] == 4
    assert mechanics["HMC_convergence_or_efficiency_claim"] is False


def test_selection_and_prior_proxy_limitations_are_explicit():
    diagnostics = load()["model_and_selection_diagnostics"]
    assert diagnostics["selection_order4_to_order6_truth_lambda_relative_L1"] > 0.01
    assert diagnostics["selection_convergence_claim"] is False
    assert diagnostics["prior_dominated_proxy_fraction"] == 26384 / 32768
    assert diagnostics["prior_dominated_proxy_is_science_frontier"] is False


def test_published_artifact_hashes_match():
    record = load()
    directory = Path(record["published_artifacts"]["directory"])
    assert directory.is_dir()
    for name, metadata in record["published_artifacts"].items():
        if name == "directory":
            continue
        assert sha256(directory / name) == metadata["sha256"]


def test_bound_program_implementation_and_runner_match():
    lineage = load()["implementation_lineage"]
    for key in ("program", "implementation", "runner"):
        metadata = lineage[key]
        path = ROOT / metadata["path"]
        assert path.stat().st_size == metadata["bytes"]
        assert sha256(path) == metadata["sha256"]


def test_phase_c_requires_new_approval_and_no_automatic_follow_on():
    record = load()
    assert record["next_recommended"]["requires_user_approval"] is True
    assert record["scientific_disposition"]["automatic_follow_on"] is False
