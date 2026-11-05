from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/cf4_bundle_b1_likelihood_contract_amendment_v1.json"


def _load() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_active_factorization_is_disjoint_and_no_double_counts_redshifts():
    value = _load()
    assert value["bundle"] == "B1-Z0-POSTERIOR"
    assert value["status"] == "DESIGN_COMPLETE_CALIBRATION_AND_EXECUTION_BLOCKED"
    factors = value["active_B1_factorization"]
    assert factors["density_factor"]["owner"] == "2Mpp_grid_counts"
    assert factors["velocity_factor"]["owner"] == "CF4_group_marks_shared_redshift"
    assert factors["density_factor"]["object_overlap_policy"].startswith(
        "exclude the 17,007"
    )
    assert factors["density_factor"]["independent_twompp_redshift_factor"] is False
    assert factors["velocity_factor"]["independent_cf4_galaxy_factor"] is False
    assert "share the latent z=0 field" in factors["dependence_policy"]


def test_selection_rsd_and_nuisance_contracts_are_explicit_and_positive():
    value = _load()
    selection = value["selection_contract"]
    assert len(selection["population_order"]) == 6
    assert selection["angular_response"]["empty_sky_required"] is True
    assert selection["angular_response"]["raw_exposure_only"] is True
    assert selection["voxel_integration"]["grid"] == [6, 32, 32, 32]
    assert selection["voxel_integration"]["subpoints_per_voxel"] == 8
    rsd = value["redshift_space_and_deposition_contract"]
    assert rsd["line_of_sight"].startswith("observer-centred spherical")
    assert rsd["coherent_displacement_cMpc_h"] == "h*v_r/(a*H(a))"
    assert rsd["quadrature"]["comparison_orders"] == [7, 9]
    assert rsd["predicted_intensity"]["deposition"].startswith(
        "periodic conservative differentiable TSC"
    )
    nuisance = value["nuisance_and_discrepancy_contract"]
    assert nuisance["population_normalization"]["parameters"] == 6
    assert nuisance["luminosity_bias"]["parameters"] == 6
    assert "universal scalar" in nuisance["luminosity_bias"]["prior"]
    assert nuisance["baseline_count_model"].startswith("independent Poisson")
    assert nuisance["model_discrepancy"]["constraints"][2] == "no arbitrary per-voxel freedom"


def test_covariance_structure_scope_and_mock_firewall_fail_closed():
    value = _load()
    covariance = value["cf4_covariance_and_overlap_contract"]
    assert covariance["secure_mapping"]["mapping_sha256"] == (
        "64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf"
    )
    assert covariance["secure_mapping"]["secure_cf4_groups"] == 11610
    assert covariance["secure_mapping"]["excluded_twompp_targets"] == 17007
    structure = value["structure_summary_scope"]
    assert structure["B1_active_likelihood"] is False
    assert structure["targets"] == [
        "Local_Group", "Virgo", "Coma", "Local_Void", "Bootes_Void", "observer_environment"
    ]
    firewall = value["independent_mock_firewall"]
    assert firewall["development"] == {
        "count": 64,
        "seed_start_inclusive": 2026083000,
        "seed_stop_exclusive": 2026083064,
        "assignment": "16 mocks per arm A-D, each seed exactly once",
        "purpose": "calibration and model-choice only",
    }
    assert firewall["untouched_validation"]["count"] == 256
    assert firewall["untouched_validation"]["seed_start_inclusive"] == 2026083064
    assert firewall["untouched_validation"]["seed_stop_exclusive"] == 2026083320
    assert [arm["id"] for arm in firewall["arms"]] == ["A", "B", "C", "D"]
    assert firewall["execution_state"] == "bindings frozen, mocks not executed in this amendment"


def test_no_external_execution_or_premature_promotion_is_authorized():
    value = _load()
    auth = value["authorization"]
    for key in (
        "read_actual_catalog_or_GPFS",
        "run_observational_inference",
        "run_development_mocks",
        "open_untouched_validation_seeds",
        "KF_EXPAND",
        "Slurm_submission",
        "GPFS_read",
        "GPFS_write",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    ):
        assert auth[key] is False
    decision = value["closure_decision"]
    assert decision["contract_amendment"] == "COMPLETE"
    assert decision["actual_z0_posterior"] == "NOT_CREATED"
    assert decision["KF_EXPAND"] == "BLOCKED_PENDING_DIAGNOSTIC_REPAIR"
    assert decision["B2_IC_FORWARD"] == "NOT_STARTED"
    assert decision["requires_new_user_approval"] is True
