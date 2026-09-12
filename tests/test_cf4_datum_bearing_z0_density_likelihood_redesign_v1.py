import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config" / "cf4_datum_bearing_z0_density_likelihood_redesign_v1.json"


def _load():
    return json.loads(DESIGN.read_text(encoding="utf-8"))


def test_audit_is_z0_first_and_execution_free():
    design = _load()
    assert design["status"] == (
        "AUDIT_COMPLETE_PHASE_A_DATUM_BUILDER_RECOMMENDED_APPROVAL_REQUIRED"
    )
    assert design["scope"]["primary_product"].startswith("posterior samples of the z=0")
    assert design["scope"]["field_inference_executed"] is False
    assert design["scope"]["Slurm_job_submitted"] is False
    assert design["authorization"]["implement_Phase_A_datum_builder"] is False
    assert design["authorization"]["submit_Phase_A_Slurm"] is False
    assert design["authorization"]["infer_IC"] is False
    assert design["scientific_route_decision"]["IC_directness_claim"] == (
        "NOT_ALLOWED_BEFORE_Z0_MOCK_PASS"
    )


def test_actual_datum_is_six_integer_ngp_count_grids():
    datum = _load()["actual_datum_contract"]
    assert datum["source_rows"] == 36635
    assert datum["population_counts_exact"] == [9617, 3463, 527, 15671, 6197, 1160]
    assert sum(datum["population_counts_exact"]) == datum["source_rows"]
    assert len(datum["population_order"]) == 6
    assert datum["voxel_assignment"].startswith("NGP/")
    assert datum["count_dtype"] == "int64"
    assert datum["CIC_fractional_counts_allowed"] is False
    assert datum["pilot_grid"] == {
        "N": 32,
        "box_size_cMpc_h": 384.0,
        "cell_size_cMpc_h": 12.0,
        "radial_min_cMpc_h": 5.0,
        "radial_max_cMpc_h": 180.0,
    }
    assert datum["holdout"]["split_before_voxelization"] is True
    assert datum["holdout"]["unit"] == "catalog row"
    assert datum["holdout"]["adjacent_voxel_random_holdout_allowed"] is False


def test_likelihood_is_positive_and_selection_is_not_count_normalized():
    design = _load()
    selection = design["selection_response_contract"]
    likelihood = design["positive_count_likelihood"]
    assert selection["response_semantics"] == "raw dimensionless exposure/support only"
    assert selection["forbidden_normalization"].startswith(
        "Do not normalize the selection field"
    )
    assert likelihood["baseline_distribution"].startswith("independent Poisson")
    assert "exp(alpha_p + b_p eta_z0(x))" in likelihood["intensity_equation"]
    assert likelihood["real_space_substitution_allowed"] is False
    assert "integral ds K_p" in likelihood["RSD_kernel_normalization"]
    nuisance = design["nuisance_hierarchy"]
    assert nuisance["population_normalization"].startswith("six independent")
    assert nuisance["galaxy_bias"].startswith("six positive")
    assert "no universal scalar is allowed" in nuisance["galaxy_bias"]


def test_mock_firewall_and_stages_are_fail_closed():
    design = _load()
    firewall = design["development_mock_contract"]["firewall"]
    assert firewall["development"]["count"] == 64
    assert firewall["development"]["seed_start_inclusive"] == 2026083000
    assert firewall["development"]["seed_stop_exclusive"] == 2026083064
    assert firewall["untouched_validation"]["count"] == 256
    assert firewall["untouched_validation"]["seed_start_inclusive"] == 2026083064
    assert firewall["untouched_validation"]["seed_stop_exclusive"] == 2026083320
    mocks = design["development_mock_contract"]
    assert mocks["new_truth_seeds_allowed"] is False
    assert mocks["seed_or_parent_ranking_allowed"] is False
    assert [arm["id"] for arm in mocks["stress_arms"]] == ["A", "B", "C", "D"]

    stages = {item["phase"]: item for item in design["staged_program"]}
    assert stages["AUDIT"]["status"] == "COMPLETE"
    assert stages["A"]["status"] == "RECOMMENDED_APPROVAL_REQUIRED"
    assert stages["A"]["field_inference"] is False
    assert stages["A"]["mock_seed_consumption"] == 0
    assert stages["A"]["automatic_follow_on"] is False
    assert stages["B"]["status"].startswith("BLOCKED_PENDING_PHASE_A")
    assert stages["C"]["status"].startswith("BLOCKED_PENDING_PHASE_B")
    assert stages["E"]["status"] == "LOCKED"
    assert stages["F"]["actual_observational_field_inference"] is True
    assert stages["F"]["IC_inference"] is False
    assert stages["G"]["IC_inference"] is True


def test_resolution_frontier_is_calibrated_not_arbitrary():
    design = _load()
    policy = design["resolution_and_low_high_boundary_policy"]
    assert policy["technical_pilot_resolution_cMpc_h"] == 12.0
    assert policy["technical_pilot_science_resolution_claim_allowed"] is False
    assert policy["target_numerical_cell_size_cMpc_h_max"] == 0.3
    assert policy["numerical_and_observational_resolution_are_distinct"] is True
    assert policy["boundary_can_move"] is True
    assert policy["arbitrary_manual_boundary_allowed"] is False
    assert policy["skip_failed_k_bin_allowed"] is False
    assert policy["maximize_frontier_subject_to_calibration"] is True
    gates = design["diagnostics_and_gates"]
    assert gates["no_single_information_cutoff"] is True
    assert gates["old_Fisher_I_0p5_reused_as_likelihood_gate"] is False
    assert gates["old_historical_0p8_response_bundle_reused_without_reconciliation"] is False


def test_phase_a_memory_headroom_and_next_approval_boundary():
    design = _load()
    phase_a = next(item for item in design["staged_program"] if item["phase"] == "A")
    resources = phase_a["prospective_resources"]
    assert resources["controller"] == "syntax_submit_and_audit_only"
    assert resources["output_root"].startswith("/gpfs/kjhan/CF4/")
    assert resources["requested_memory_MiB"] >= 1.2 * resources["expected_peak_memory_MiB"]
    assert resources["manual_numerical_execution_on_syntax_or_syn101_allowed"] is False
    assert resources["renameat2_required"] is False
    assert design["next_action"]["requires_new_user_approval"] is True
    assert design["next_action"]["implementation_before_approval_allowed"] is False
    assert design["next_action"]["Slurm_before_approval_allowed"] is False
    assert design["next_action"]["automatic_Phase_B_after_Phase_A_allowed"] is False
