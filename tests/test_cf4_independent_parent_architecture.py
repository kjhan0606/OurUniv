import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_design():
    return json.loads((
        ROOT / "config/cf4_independent_parent_architecture_design.json"
    ).read_text())


def test_design_is_authorized_by_sealed_projection_and_failure_records():
    design = load_design()
    for item in design["authorizations"].values():
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    source = design["unchanged_Local_Group_model"]
    assert sha256_file(ROOT / source["source_program"]) == (
        source["source_program_sha256"]
    )


def test_design_uses_every_independent_parent_and_exact_projection_kernel():
    design = load_design()
    bank = design["sealed_CF4_parent_bank"]
    assert bank["parent_seed_range_inclusive"] == [3193, 3448]
    assert bank["parent_count"] == 256
    assert "all 256" in bank["parent_proposal"]
    target = design["hierarchical_target"]
    assert "CF4_parent_index_j" in target["state"]
    assert "ker(R)" in target["peak_conditioning_subspace"]
    assert "R x=y_j" in target["fine_prior"]


def test_design_importance_weight_includes_peak_evidence_and_midpoint_ratio():
    target = load_design()["hierarchical_target"]
    assert "log Z_peak" in target["pre_z0_log_weight"]
    assert "log p(q) - log g(q)" in target["pre_z0_log_weight"]
    assert "log-mean-exp" in target["post_z0_log_weight"]
    assert "may not be dropped" in target["post_z0_log_weight"]


def test_evidence_feasibility_is_common_random_number_and_all_parent():
    stage = load_design()["stage_1_all_parent_evidence_feasibility"]
    integration = stage["integration"]
    assert integration["common_random_numbers"] is True
    assert integration["midpoint_axis_draws_per_parent"] == 64
    assert integration["reuse_same_64_draws_for_every_parent"] is True
    assert integration["fine_null_space_field_drawn"] is False
    assert integration["PM_or_halo_finder_run"] is False
    gates = stage["gates"]
    assert gates["parent_effective_sample_size_min"] == 32.0
    assert gates["maximum_single_parent_weight_max"] == 0.10
    assert gates["pass_requires_all"] is True


def test_design_does_not_authorize_generation_or_simulation():
    firewall = load_design()["information_firewall"]
    assert firewall["design_only"] is True
    assert firewall["all_256_parents_required"] is True
    assert firewall["single_parent_or_survivor_subset_forbidden"] is True
    assert firewall["no_candidate_field_opened"] is True
    assert firewall["candidate_generation_authorized"] is False
    assert firewall["PM_authorized"] is False
    assert firewall["seed_selection_authorized"] is False
    assert firewall["RAMSES_authorized"] is False
