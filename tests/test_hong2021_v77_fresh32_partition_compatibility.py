import hashlib
import json
from pathlib import Path

import numpy as np

import hong2021_v77_fresh32_partition_compatibility as audit


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v77_fresh32_partition_compatibility_audit_program.json"


def test_program_is_byte_bound_and_target_free() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == audit.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == audit.PROGRAM_SCHEMA
    assert program["status"] == audit.PROGRAM_STATUS
    assert program["scope_limits"]["validation_input_or_target_payload_access"] is False
    assert program["scope_limits"]["physical_morphology_threshold_relaxation"] is False
    assert program["authorization"]["construct_or_sample_a_candidate"] is False


def _groups(counts: list[int]) -> np.ndarray:
    return np.concatenate([np.full(count, group) for group, count in enumerate(counts)])


def test_compatible_query_design_has_exact_quota_multisets() -> None:
    rng = np.random.default_rng(770010)
    cases = [
        ("TNG100", _groups([30] * 4), [6, 8, 9, 9], 4),
        ("SIMBA", _groups([30] * 8), [7, 10, 15], 3),
        ("Swift", _groups([30] * 20), [4, 4, 4, 5, 5, 5, 5], 7),
    ]
    for domain, groups, quotas, occupied in cases:
        selected = audit.sample_compatible_queries(domain, groups, rng)
        labels, counts = np.unique(groups[selected], return_counts=True)
        assert len(selected) == len(np.unique(selected)) == 32
        assert len(labels) == occupied
        assert sorted(counts.tolist()) == quotas


def test_query_design_keeps_an_oracle_donor() -> None:
    groups = _groups([6, 20, 20, 20])
    try:
        audit.sample_compatible_queries("TNG100", groups, np.random.default_rng(1))
    except ValueError as error:
        assert "oracle donor" in str(error)
    else:
        raise AssertionError("V77 accepted a fully depleted fit-train group")


def test_conservative_complete_lower_uses_union_bound() -> None:
    row = audit.conservative_complete_lower(0.916, 0.051, 0.05)
    assert np.isclose(row["complete_gate_pass_lower"], 0.815)
    assert row["pass"] is True
    failed = audit.conservative_complete_lower(0.90, 0.06, 0.05)
    assert failed["pass"] is False


def test_frozen_metadata_selection_is_unique_and_32_per_domain() -> None:
    program = json.loads(PROGRAM.read_text())
    for domain in audit.DOMAIN_ORDER:
        selected = program["metadata_only_selection"][domain]["selected_indices"]
        assert len(selected) == len(set(selected)) == 32
        assert sum(program["metadata_only_selection"][domain]["quota"].values()) == 32


def test_source_and_runner_preserve_firewalls() -> None:
    source = (REPO / "src/hong2021_v77_fresh32_partition_compatibility.py").read_text()
    assert 'validation_input_or_target_payload_accessed": False' in source
    assert 'candidate_or_fresh_payload_execution_authorized": False' in source
    assert 'training_or_model_sampling_performed": False' in source
    assert 'handle["input"][:]' not in source
    assert 'handle["target"][:]' not in source
    runner = (
        REPO / "scripts/hong2021_v77_fresh32_partition_compatibility_lageunha.sh"
    ).read_text()
    assert "taskset -c 64 nice -n 15" in runner
    assert 'CUDA_VISIBLE_DEVICES=""' in runner
