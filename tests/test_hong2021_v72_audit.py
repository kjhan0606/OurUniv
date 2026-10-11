import json
from pathlib import Path

from hong2021_v18_init import sha256_file


REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "config/hong2021_v72_feasibility_and_fresh_partition_audit.json"


def test_v72_audit_binds_v71_and_reserves_two_fresh_stages() -> None:
    audit = json.loads(AUDIT.read_text())
    assert audit["status"] == (
        "complete_metadata_only_two_fresh_stages_available_candidate_not_yet_frozen"
    )
    parent = audit["parent_evidence"]
    assert sha256_file(REPO / parent["v71_result_record"]) == parent[
        "v71_result_record_sha256"
    ]
    assert sha256_file(REPO / parent["v35_development_definition"]) == parent[
        "v35_development_definition_sha256"
    ]
    consumed = audit["historical_development_consumption"]
    screen = audit["fresh_stage_A_screen"]
    confirmatory = audit["fresh_stage_B_confirmatory"]
    for domain in ("TNG100", "SIMBA", "Swift"):
        old = set(consumed[domain])
        first = set(screen[domain])
        second = set(confirmatory[domain])
        assert len(old) == len(first) == len(second) == 16
        assert not old & first
        assert not old & second
        assert not first & second
    assert audit["partition_checks"][
        "both_stages_disjoint_from_all_historical_development_each_domain"
    ] is True


def test_v72_audit_does_not_reinterpret_v71_or_use_target_payload() -> None:
    audit = json.loads(AUDIT.read_text())
    assert audit["metadata_access"]["validation_input_voxels_read"] is False
    assert audit["metadata_access"]["validation_target_voxels_read"] is False
    validity = audit["global_maximum_gate_validity"]
    assert validity["sample_size_ratio_generated_over_truth"] == 16
    assert validity["V71_still_fails_without_this_row"] is True
    assert audit["V71_failure_reinterpretation_for_new_design_only"][
        "V71_verdict_changed"
    ] is False
    candidate = audit["recommended_V72_candidate"]
    assert candidate["abbreviation"] == "SQT"
    assert candidate["training_required"] is False
    authorization = audit["authorization"]
    assert authorization["V72_candidate_program_frozen"] is False
    assert authorization["stage_A_payload_access_authorized"] is False
    assert authorization["stage_B_payload_access_authorized"] is False
    assert authorization["independent_EAGLE_access_authorized"] is False
