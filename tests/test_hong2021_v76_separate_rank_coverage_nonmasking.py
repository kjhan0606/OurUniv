import hashlib
import json
from pathlib import Path

import numpy as np

import hong2021_v76_separate_rank_coverage_nonmasking as audit


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v76_separate_rank_coverage_nonmasking_audit_program.json"


def test_program_is_byte_bound_and_preserves_firewall() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == audit.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == audit.PROGRAM_SCHEMA
    assert program["status"] == audit.PROGRAM_STATUS
    assert program["scientific_rule"]["number_of_tests"] == 6
    assert program["scientific_rule"]["per_test_alpha"] == 1 / 120
    assert program["scope_limits"]["V72_ensemble_or_array_access"] is False
    assert program["scope_limits"]["new_candidate_or_complete_gate_authorized"] is False


def test_program_load_touches_only_bound_parents(monkeypatch) -> None:
    visited: list[Path] = []
    original = audit.sha256_file

    def traced(path: str | Path) -> str:
        resolved = Path(path).resolve()
        visited.append(resolved)
        return original(resolved)

    monkeypatch.setattr(audit, "sha256_file", traced)
    _, bindings = audit.load_program(PROGRAM, REPO)
    allowed = set(bindings.values()) | {PROGRAM.resolve()}
    assert set(visited) == allowed


def test_separate_rule_is_strict_at_frozen_threshold() -> None:
    alpha = 1 / 120
    assert audit.separate_pass(alpha, 1.0) is False
    assert audit.separate_pass(1.0, alpha) is False
    assert audit.separate_pass(np.nextafter(alpha, 1.0), 1.0) is True
    assert audit.separate_pass(1.0, np.nextafter(alpha, 1.0)) is True


def test_nonmasking_audit_enumerates_all_grid_pairs() -> None:
    program = json.loads(PROGRAM.read_text())
    result = audit.nonmasking_audit(program)
    assert result["comparisons"] == 36
    assert result["mismatches"] == []
    assert result["rank_only_signal_detected"] is True
    assert result["coverage_only_signal_detected"] is True
    assert result["nonmasking_invariant_pass"] is True


def test_probability_row_uses_wilson_interval() -> None:
    row = audit.probability_row(np.asarray([True] * 8 + [False] * 2))
    assert row["successes"] == 8
    assert row["trials"] == 10
    assert row["Wilson_95"][0] < 0.8 < row["Wilson_95"][1]


def test_consumed_diagnostic_identifies_rank_masking() -> None:
    v75 = {
        "consumed_V72_diagnostic": {
            arm: {
                domain: {
                    "conditional_p_values": {
                        "rank_tv": 0.00001 if domain == "TNG100" else 0.5,
                        "coverage_deviation": 0.9,
                        "composite": 0.9,
                    }
                }
                for domain in audit.DOMAIN_ORDER
            }
            for arm in ("candidate", "control")
        }
    }
    result = audit.consumed_v75_diagnostic(v75)
    assert result["candidate"]["TNG100"]["scalar_masking_detected"] is True
    assert result["candidate"]["SIMBA"]["scalar_masking_detected"] is False
    assert result["selection_role"] is False
    assert result["V72_verdict_changed"] is False


def test_power_detection_is_union_not_scalar_rescaling() -> None:
    rank_p = np.asarray([0.001, 0.5, 0.5])
    coverage_p = np.asarray([0.5, 0.001, 0.5])
    detected = (rank_p <= audit.PER_TEST_ALPHA) | (
        coverage_p <= audit.PER_TEST_ALPHA
    )
    assert detected.tolist() == [True, True, False]


def test_source_and_runner_preserve_firewalls() -> None:
    source = (
        REPO / "src/hong2021_v76_separate_rank_coverage_nonmasking.py"
    ).read_text()
    assert 'V72_ensemble_or_array_accessed": False' in source
    assert 'complete_gate_or_new_candidate_execution_authorized": False' in source
    assert 'training_or_model_sampling_performed": False' in source
    runner = (
        REPO / "scripts/hong2021_v76_separate_rank_coverage_nonmasking_lageunha.sh"
    ).read_text()
    assert "taskset -c 64 nice -n 15" in runner
    assert 'CUDA_VISIBLE_DEVICES=""' in runner
