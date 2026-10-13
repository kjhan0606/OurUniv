import hashlib
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "config" / "cf4_kf_bin_manifest_design_v1.json"
KF_PATH = ROOT / "config" / "cf4_kf_design_v1.json"
ROI_SOURCE_PATH = ROOT / "config" / "p1_targets_v2_observer.json"
ROI_SOURCE_SHA256 = (
    "34bf11486b084312d6a7e476f469e6973cfbf3a9392c7c7eab9b1a6b9a124e0a"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _derived_center(observer, sgl_deg, sgb_deg, distance_mpc_h):
    longitude = math.radians(sgl_deg)
    latitude = math.radians(sgb_deg)
    relative = (
        distance_mpc_h * math.cos(latitude) * math.cos(longitude),
        distance_mpc_h * math.cos(latitude) * math.sin(longitude),
        distance_mpc_h * math.sin(latitude),
    )
    return [origin + offset for origin, offset in zip(observer, relative)]


def test_design_freeze_has_no_execution_authority():
    design = _load(DESIGN_PATH)

    assert design["schema"] == "ouruniv-cf4-kf-bin-manifest-design-v1"
    assert design["date"] == "2026-08-30"
    assert design["status"] == (
        "user_approved_design_frozen_ROI_leakage_and_manifest_materialization_pending"
    )
    assert design["stage"] == "KF-DESIGN"
    authority = design["authority"]
    assert authority["authorization_basis"] == "current_user_approval"
    assert authority["design_freeze_recorded"]
    assert authority["ROI_geometry_approved"]
    assert authority["ROI_geometry_frozen"]
    for key in (
        "ROI_leakage_execution_authorized",
        "final_manifest_materialization_authorized",
        "KF_EXPAND_authorized",
        "all_D_mock_execution_authorized",
        "production_compute_authorized",
        "Slurm_submission_authorized",
        "GPFS_read_authorized",
        "GPFS_write_authorized",
        "network_access_authorized",
    ):
        assert authority[key] is False


def test_lattice_and_native_bin_design_are_exact_and_have_no_science_cutoff():
    design = _load(DESIGN_PATH)
    lattice = design["analysis_lattice"]

    assert lattice["cells_per_axis_N"] == 1280
    assert lattice["grid_spacing_cMpc_h"] == 0.3
    assert lattice["box_size_cMpc_h"] == 384.0
    assert lattice["cells_per_axis_N"] * lattice["grid_spacing_cMpc_h"] == lattice[
        "box_size_cMpc_h"
    ]
    assert lattice["fundamental_h_Mpc"] == 2.0 * math.pi / 384.0
    assert lattice["isotropic_analysis_Nyquist_h_Mpc"] == math.pi / 0.3
    assert lattice["isotropic_analysis_Nyquist_h_Mpc"] == 10.471975511965978
    assert lattice["DFT_integer_frequency_convention_per_axis"] == {
        "minimum_inclusive": -640,
        "maximum_inclusive": 639,
    }
    assert lattice["DC_mode_included"] is False
    assert lattice["cube_corner_modes_included"] is False
    assert lattice["user_supplied_low_or_high_k_cutoff_allowed"] is False

    bins = design["native_bin_design"]
    assert bins["first_lower_edge"] == "k_f"
    assert bins["edge_ratio"] == "2^(1/4)"
    assert bins["delta_log2_k"] == 0.25
    assert "terminal truncated bin ending at k_Ny" in bins["construction"]
    assert bins["complete_native_bin_count_for_frozen_lattice"] == 37
    assert bins["terminal_truncated_bin_count_for_frozen_lattice"] == 1
    assert bins["complete_bin_interval"] == "lower inclusive and upper exclusive"
    assert bins["terminal_bin_interval"] == "lower inclusive and k_Ny upper inclusive"
    assert bins["record_every_native_bin"]
    assert bins["record_zero_mode_bins"]
    assert bins["record_failed_bins"]
    for forbidden in (
        "prefix_omission_allowed",
        "suffix_omission_allowed",
        "failed_bin_omission_allowed",
        "bin_reordering_allowed",
        "shortened_Nyquist_allowed",
        "user_science_cutoff_allowed",
    ):
        assert bins[forbidden] is False


def test_exact_real_mode_count_and_manifest_envelope_are_frozen_fail_closed():
    design = _load(DESIGN_PATH)
    modes = design["independent_real_mode_count"]

    assert modes["algorithm_status"] == "frozen"
    assert "integer squared radius" in modes["histogram_key"]
    assert "count the pair once" in modes["pair_count_rule"]
    assert "count that real self-conjugate mode once" in modes[
        "self_conjugate_rule"
    ]
    assert modes["floating_point_catalog_enumeration_allowed"] is False
    assert modes["truth_or_candidate_input_allowed"] is False
    merge = modes["merged_bin_rule"]
    assert merge["minimum_independent_real_modes"] == 32
    assert merge["record_native_memberships"]
    assert merge["record_native_counts"]
    assert merge["record_merged_counts"]
    assert merge["truth_blind"]

    envelope = design["manifest_envelope_contract"]
    assert envelope["future_final_manifest_path"] == (
        "config/cf4_kf_bin_manifest_v1.json"
    )
    assert envelope["final_manifest_status_now"] == "absent_blocking"
    assert envelope["body_hash_algorithm"] == "SHA256"
    assert envelope["body_hash_key_outside_body"] == "manifest_body_sha256"
    assert envelope["self_hash_recursion_allowed"] is False
    canonical = envelope["body_canonical_serialization"]
    assert canonical["JSON_object_keys"] == "sorted lexicographically"
    assert canonical["encoding"] == "UTF-8"
    assert canonical["termination"] == "exactly one newline"
    assert envelope["k_boundary_claim_allowed_now"] is False


def test_frozen_roi_geometry_matches_bound_source_and_overlap_contract():
    design = _load(DESIGN_PATH)
    source = _load(ROI_SOURCE_PATH)
    geometry = design["ROI_geometry"]

    assert hashlib.sha256(ROI_SOURCE_PATH.read_bytes()).hexdigest() == (
        ROI_SOURCE_SHA256
    )
    assert geometry["source"]["raw_sha256"] == ROI_SOURCE_SHA256
    assert geometry["source"]["cosmology_h"] == source["cosmology_h"] == 0.746
    observer = geometry["observer_box_center_cMpc_h"]
    assert observer == [192.0, 192.0, 192.0]
    rois = {item["id"]: item for item in geometry["ROIs"]}
    assert list(rois) == [
        "Local_Group",
        "Virgo",
        "Coma",
        "Local_Void",
        "Bootes_Void",
        "observer_environment",
    ]
    assert rois["Local_Group"]["center_cMpc_h"] == observer
    assert rois["Local_Group"]["radius_cMpc_h"] == 8.0
    assert rois["observer_environment"]["center_cMpc_h"] == observer
    assert rois["observer_environment"]["radius_cMpc_h"] == 8.0
    assert rois["Virgo"]["radius_cMpc_h"] == 5.0
    assert rois["Coma"]["radius_cMpc_h"] == 8.0
    assert rois["Bootes_Void"]["radius_cMpc_h"] == 31.0
    assert rois["Local_Void"]["component_radius_cMpc_h"] == 6.0
    assert rois["Local_Void"]["geometry"] == "union_max_of_spheres"

    for roi_id, source_section in (("Virgo", "Virgo"), ("Coma", "Coma")):
        roi = rois[roi_id]
        source_roi = source["clusters"][source_section]
        expected = _derived_center(
            observer,
            source_roi["sgl_deg"],
            source_roi["sgb_deg"],
            source_roi["distance_mpc"] * source["cosmology_h"],
        )
        assert roi["center_cMpc_h"] == pytest.approx(expected, abs=1e-12)

    bootes_source = source["bootes_void"]
    bootes_expected = _derived_center(
        observer,
        bootes_source["sgl_deg"],
        bootes_source["sgb_deg"],
        bootes_source["distance_mpc_h"],
    )
    assert rois["Bootes_Void"]["center_cMpc_h"] == pytest.approx(
        bootes_expected, abs=1e-12
    )
    local_void = rois["Local_Void"][
        "component_centers_from_observer_plus_source_offsets_cMpc_h"
    ]
    for component in local_void:
        source_offset = source["local_void"]["probes"][component["id"]]
        assert component["source_offset_cMpc_h"] == source_offset
        assert component["center_cMpc_h"] == [
            origin + offset for origin, offset in zip(observer, source_offset)
        ]

    overlap = geometry["overlap_contract"]
    assert overlap[
        "Local_Group_and_observer_environment_windows_are_spatially_identical"
    ]
    assert overlap["semantic_scores_are_separate"]
    assert overlap["scores_may_be_summed"] is False
    assert overlap["overlap_must_be_reported"]


def test_leakage_support_requires_strict_suffix_and_future_route_is_not_authority():
    design = _load(DESIGN_PATH)
    leakage = design["ROI_window_and_leakage_design"]

    assert leakage["status"] == "frozen_not_executed"
    window = leakage["window"]
    assert window["family"] == "compact_C1_raised_cosine_sphere"
    assert window["radial_definition"] == {
        "r_le_0.75R": "w(r)=1",
        "0.75R_lt_r_le_R": "w(r)=0.5*[1+cos(pi*(r-0.75R)/(0.25R))]",
        "r_gt_R": "w(r)=0",
    }
    gate = leakage["native_bin_support_gate"]
    assert gate["response_contained_in_b_minus_1_b_b_plus_1_min_inclusive"] == 0.9
    assert gate["alias_or_outside_analysis_fraction_max_inclusive"] == 0.01
    assert gate["localized_effective_independent_mode_count_min_inclusive"] == 32
    suffix = leakage["supported_suffix_rule"]
    assert "unbroken supported suffix" in suffix["rule"]
    assert suffix["failure_after_first_supported_semantics"] == (
        "final_manifest_materialization_fail_closed"
    )
    assert suffix["all_supported_and_failed_bins_retained_in_leakage_artifact"]
    assert suffix["post_result_threshold_or_geometry_change_allowed"] is False
    assert leakage["pass_results_declared_now"] is False
    assert leakage["supported_lowest_scales_declared_now"] is False
    assert leakage["final_manifest_materialization_allowed_now"] is False

    route = design["future_execution_route"]
    assert route["authorization_status"] == "not_authorized_design_only"
    assert route["heavy_execution_mode_if_separately_authorized"] == "Slurm_only"
    assert route["controller"] == "syntax_controller"
    assert route["controller_allowed_actions"] == ["submit", "audit"]
    assert route["manual_syn101_allowed"] is False
    assert route["manual_syntax_allowed"] is False
    assert route["output_root"] == "/gpfs/kjhan/CF4/kf_design/roi_leakage_v1"
    assert route["refuse_overwrite"]
    assert "COMPLETE" in route["publication_order"]
    assert route["current_execution_authorized"] is False


def test_kf_contract_binds_frozen_design_without_changing_gates_or_seeds():
    kf = _load(KF_PATH)
    contract = kf["declared_bin_manifest_contract"]

    assert contract["status"] == "design_frozen_ROI_leakage_pending_blocking"
    assert contract["design_path"] == (
        "config/cf4_kf_bin_manifest_design_v1.json"
    )
    assert contract["ROI_geometry_approved_and_frozen"]
    assert contract["future_manifest_path"] == "config/cf4_kf_bin_manifest_v1.json"
    assert contract["future_execution_route_is_authorization"] is False
    assert contract["future_execution_route"]["current_execution_authorized"] is False
    assert kf["authority"]["ROI_leakage_execution_authorized"] is False
    assert kf["authority"]["final_bin_manifest_materialization_authorized"] is False
    assert kf["authority"]["KF_EXPAND_authorized"] is False
    assert kf["authority"]["Slurm_submission_authorized"] is False
    assert kf["authority"]["GPFS_read_authorized"] is False
    assert kf["authority"]["GPFS_write_authorized"] is False
    assert kf["authority"]["network_access_authorized"] is False

    for index, roi in enumerate(kf["domains"]["ROIs"]):
        assert roi["geometry_status"] == "approved_frozen"
        assert roi["geometry_design_binding"] == (
            f"config/cf4_kf_bin_manifest_design_v1.json#/ROI_geometry/ROIs/{index}"
        )

    gates = kf["strict_bin_gates"]
    assert gates["response"] == {
        "minimum_inclusive": 0.8,
        "maximum_inclusive": 1.2,
    }
    assert gates["r_of_k"]["minimum_inclusive"] == 0.7
    assert gates["residual_power_ratio"]["maximum_inclusive"] == 0.5
    assert gates["phase_coherence"]["familywise_p_max_inclusive"] == 0.05
    assert gates["prior_to_posterior_variance_reduction"][
        "posterior_to_prior_variance_ratio_median_max_inclusive"
    ] == 0.8
    assert gates["coverage"]["minimum_untouched_validation_mocks"] == 256
    assert gates["held_out_prediction"][
        "mock_bootstrap_95_percent_lower_bound_min_exclusive"
    ] == 0.0

    firewall = kf["independent_mock_firewall"]
    assert firewall["development"]["seed_start_inclusive"] == 2026083000
    assert firewall["development"]["seed_stop_exclusive"] == 2026083064
    assert firewall["untouched_validation"]["seed_start_inclusive"] == 2026083064
    assert firewall["untouched_validation"]["seed_stop_exclusive"] == 2026083320
