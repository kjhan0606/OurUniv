import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOINT_PATH = ROOT / "config" / "cf4_2mpp_joint_likelihood_v1.json"
RESULT_PATH = ROOT / "config" / "cf4_2mpp_crossmatch_v1_result.json"
KF_PATH = ROOT / "config" / "cf4_kf_design_v1.json"

MAPPING_SHA256 = (
    "64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf"
)
RESULT_SHA256 = (
    "3e2e5841d62e9581c7437a28f07d6f5c3423b023749f99a678546d1d7d29752a"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_complete_result_hash_counts_and_partition_are_bound():
    joint = _load(JOINT_PATH)
    result_bytes = RESULT_PATH.read_bytes()
    result = json.loads(result_bytes)

    assert hashlib.sha256(result_bytes).hexdigest() == RESULT_SHA256
    assert result["status"] == "COMPLETE"
    assert result["mapping"]["sha256"] == MAPPING_SHA256
    assert sum(result["mapping"]["class_counts"].values()) == result["mapping"][
        "rows"
    ] == 55877
    assert result["inputs"]["twompp_catalog"]["real_rows_eligible"] == 69160
    assert result["mapping"]["class_counts"] == {
        "coordinate_redshift_conflict": 188,
        "extended_review_candidate": 273,
        "nonreciprocal_collision": 5,
        "secure_joint_mark": 16584,
        "unmatched": 38827,
    }
    assert result["mapping"]["unique_cf4_1PGC_secure"] == 11610

    artifacts = joint["crossmatch_contract"]["artifacts"]
    assert artifacts["mapping"]["sha256"] == MAPPING_SHA256
    assert artifacts["result"]["raw_sha256"] == RESULT_SHA256
    assert artifacts["result"]["mapping_sha256_binding"] == MAPPING_SHA256
    counts = joint["crossmatch_contract"]["verified_actual_diagnostic_counts"]
    assert counts == {
        "CF4_individual_rows": 55877,
        "2Mpp_real_rows": 69160,
        "secure_mutual_rows": 16584,
        "nonreciprocal_collision_rows": 5,
        "coordinate_redshift_conflict_rows": 188,
        "extended_review_candidate_rows": 273,
        "unmatched_rows": 38827,
        "secure_unique_CF4_1PGC_groups": 11610,
    }


def test_joint_contract_forbids_double_counting_and_keeps_execution_blocked():
    joint = _load(JOINT_PATH)

    assert joint["status"] == "crossmatch_complete_likelihood_blocked"
    authority = joint["authority"]
    assert authority["crossmatch_artifact_publication_authorized"]
    assert (
        authority["crossmatch_artifact_publication_authorization_basis"]
        == "current_user_approval"
    )
    for key in (
        "joint_likelihood_execution_authorized",
        "KF_EXPAND_authorized",
        "all_D_mock_execution_authorized",
        "production_compute_authorized",
        "Slurm_submission_authorized",
        "GPFS_read_authorized",
        "GPFS_write_authorized",
        "network_access_authorized",
    ):
        assert authority[key] is False

    factorization = joint["joint_factorization"]
    assert factorization["CF4_group_factor_is_canonical"]
    assert not factorization[
        "CF4_individual_positions_or_counts_are_an_independent_density_factor"
    ]
    assert not factorization["independent_Vcmb_double_counting_allowed"]
    assert "one shared redshift datum or latent" in factorization[
        "shared_secure_object_rule"
    ]

    policy = joint["imputed_and_latent_policy"]
    assert policy["ZoA_fake_rows"]["observation_status"] == "excluded"
    assert not policy["ZoA_fake_rows"]["allowed_as_independent_observation"]
    assert policy["Cln_rows"]["radial_redshift_status"] == "cloned_imputed_latent"
    assert not policy["Cln_rows"][
        "radial_redshift_allowed_as_independent_datum"
    ]
    assert all("not been published" not in item for item in joint["blockers"])
    assert joint["stage_transition"]["all_D_likelihood_ready"] is False


def test_kf_inventory_binds_crossmatch_but_retains_likelihood_and_manifest_blocks():
    kf = _load(KF_PATH)
    external = next(
        item
        for item in kf["data_inventory"]
        if item["id"] == "D_galaxy_density_external"
    )

    assert "crossmatch_COMPLETE" in external["status"]
    mapping = external["artifacts"]["CF4_crossmatch_mapping"]
    result = external["artifacts"]["CF4_crossmatch_result"]
    assert mapping["sha256"] == MAPPING_SHA256
    assert result["raw_sha256"] == RESULT_SHA256
    assert result["mapping_sha256_binding"] == MAPPING_SHA256
    assert external["verified_CF4_crossmatch"] == {
        "CF4_individual_rows": 55877,
        "2Mpp_real_rows": 69160,
        "secure_joint_mark_rows": 16584,
        "coordinate_redshift_conflict_rows": 188,
        "nonreciprocal_collision_rows": 5,
        "extended_review_candidate_rows": 273,
        "unmatched_rows": 38827,
        "secure_unique_CF4_1PGC_groups": 11610,
        "quarantine_auto_promotion_allowed": False,
    }

    authority = kf["authority"]
    for key in (
        "KF_EXPAND_authorized",
        "all_D_mock_execution_authorized",
        "production_compute_authorized",
        "Slurm_submission_authorized",
        "GPFS_read_authorized",
        "GPFS_write_authorized",
        "network_access_authorized",
    ):
        assert authority[key] is False
    blockers = " ".join(authority["all_D_mock_blockers"])
    for required in (
        "angular empty-sky completeness",
        "K/evolution/extinction/LF",
        "RSD/FoG",
        "luminosity-bias/HOD/model-discrepancy",
        "CF4 group-member shared-redshift covariance",
    ):
        assert required in blockers
    assert kf["declared_bin_manifest_contract"]["status"] == (
        "design_frozen_ROI_leakage_pending_blocking"
    )
    assert kf["declared_bin_manifest_contract"][
        "evaluation_inputs_must_bind_to_manifest_SHA256"
    ] is True


def test_frozen_threshold_seed_and_roi_contract_values_are_unchanged():
    kf = _load(KF_PATH)
    gates = kf["strict_bin_gates"]

    assert gates["response"] == {
        "minimum_inclusive": 0.8,
        "maximum_inclusive": 1.2,
    }
    assert gates["r_of_k"]["minimum_inclusive"] == 0.7
    assert gates["residual_power_ratio"]["maximum_inclusive"] == 0.5
    assert gates["phase_coherence"]["familywise_p_max_inclusive"] == 0.05
    variance = gates["prior_to_posterior_variance_reduction"]
    assert variance["posterior_to_prior_variance_ratio_median_max_inclusive"] == 0.8
    assert variance["mock_bootstrap_95_percent_upper_bound_max_exclusive"] == 1.0
    coverage = gates["coverage"]
    assert coverage["minimum_untouched_validation_mocks"] == 256
    assert coverage["absolute_error_max_inclusive"] == {"68": 0.05, "95": 0.025}
    assert gates["held_out_prediction"][
        "mock_bootstrap_95_percent_lower_bound_min_exclusive"
    ] == 0.0

    firewall = kf["independent_mock_firewall"]
    assert firewall["development"] == {
        "count": 64,
        "seed_start_inclusive": 2026083000,
        "seed_stop_exclusive": 2026083064,
        "purpose": "method development only",
    }
    assert firewall["untouched_validation"] == {
        "count": 256,
        "seed_start_inclusive": 2026083064,
        "seed_stop_exclusive": 2026083320,
        "purpose": "one-time validation only",
    }

    rois = kf["domains"]["ROIs"]
    assert [item["id"] for item in rois] == [
        "Local_Group",
        "Virgo",
        "Coma",
        "Local_Void",
        "Bootes_Void",
        "observer_environment",
    ]
    assert all(
        item["geometry_status"] == "approved_frozen"
        for item in rois
    )
    assert rois[0]["geometry_source"] == [
        "config/p1_targets_v2_observer.json observer-centred radii",
        "existing Local Group target configs/references",
    ]
    assert [item["geometry_source"] for item in rois[1:]] == [
        "config/p1_targets_v2_observer.json#/clusters/Virgo",
        "config/p1_targets_v2_observer.json#/clusters/Coma",
        "config/p1_targets_v2_observer.json#/local_void",
        "config/p1_targets_v2_observer.json#/bootes_void",
        "config/p1_targets_v2_observer.json#/observer_environment",
    ]
    assert kf["domains"]["ROI_freeze_time"] == "before truth or candidate inspection"
    assert kf["frozen_contract"]["material_thresholds"]["frozen"]
    assert kf["frozen_contract"]["mock_seed_ranges"]["frozen"]
    assert kf["frozen_contract"]["ROI_sources"]["frozen"]
