import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "config/cf4_twompp_exact_joint_covariance_pilot_v1_result_record.json"
PROGRAM = ROOT / "config/cf4_twompp_exact_joint_covariance_pilot_program_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_result_record_binds_completed_artifacts_and_preserves_firewall() -> None:
    record = _load(RECORD)
    assert record["status"] == "COMPLETE_EXACT_GEOMETRY0_MATERIAL_GATE_FAIL_NO_ROUTE_REJECTION"
    assert record["execution"]["Slurm_job_id"] == 329618
    assert record["execution"]["state"] == "COMPLETED"
    assert record["execution"]["exit_code"] == "0:0"
    assert record["execution"]["automatic_follow_on_executed"] is False
    for binding in record["published_artifacts"].values():
        if not isinstance(binding, dict):
            continue
        path = Path(binding["path"])
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    firewall = record["geometry_and_firewall_audit"]
    assert firewall["truth_field_array_generated_or_deserialized"] is False
    assert firewall["likelihood_datum_consumed_by_inference"] is False
    assert firewall["galaxy_positions_consumed_as_field_likelihood_datum"] is False
    assert firewall["untouched_256_mock_validation_executed"] is False
    assert firewall["present_density_posterior_created"] is False
    assert firewall["IC_inference_executed"] is False


def test_recorded_lowest_bin_metrics_match_published_exact_result() -> None:
    record = _load(RECORD)
    published = _load(Path(record["published_artifacts"]["result"]["path"]))
    for scenario_name, record_key in (
        (
            "velocity_only_exact_marginalized_nuisance",
            "velocity_only_exact_marginalized_nuisance",
        ),
        ("known_selection_reference_bias", "known_selection_reference_bias"),
        (
            "normalization_marginalized_reference_bias",
            "normalization_marginalized_reference_bias",
        ),
    ):
        source = published["scenarios"][scenario_name]
        recorded = record["lowest_joint_bin_information"][record_key]
        for domain in ("delta", "theta"):
            assert source["domains"][domain]["recovered_information_fraction"][0] == recorded[
                "delta_and_theta_recovered_information_fraction"
            ]
            assert source["domains"][domain][
                "recovered_information_numerical_95_lower"
            ][0] == recorded["delta_and_theta_numerical_95_lower"]
            assert source["domains"][domain]["material_gate"][0] is False
            assert source["domains"][domain]["strong_stretch_gate"][0] is False
    assert published["decision"]["known_selection_lowest_joint_material_pass"] is False
    assert published["decision"][
        "normalization_marginalized_lowest_joint_material_pass"
    ] is False


def test_gate_is_not_lowered_and_single_geometry_is_not_route_rejection() -> None:
    record = _load(RECORD)
    program = _load(PROGRAM)
    material = record["lowest_joint_bin_information"]["material_gate"]
    assert material["information_point_and_numerical_lower_min_inclusive"] == 0.5
    assert (
        program["design"]["information_gates"]["material"]
        ["information_numerical_95_lower_min_inclusive"]
        == 0.5
    )
    known = record["lowest_joint_bin_information"]["known_selection_reference_bias"]
    velocity = record["lowest_joint_bin_information"][
        "velocity_only_exact_marginalized_nuisance"
    ]
    assert math.isclose(
        known["information_gain_over_exact_velocity_only"],
        known["delta_and_theta_recovered_information_fraction"]
        - velocity["delta_and_theta_recovered_information_fraction"],
    )
    disposition = record["scientific_disposition"]
    assert disposition["exact_64_geometry_production_for_same_linear_design"] == "NO_GO"
    assert disposition["2Mpp_density_tracer_route_level_rejection"] == "NOT_ALLOWED"
    assert record["recommended_next_stage_requiring_user_approval"][
        "new_code_or_Slurm_authorized_by_this_record"
    ] is False
