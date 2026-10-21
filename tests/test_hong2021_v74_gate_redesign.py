import hashlib
import json
from pathlib import Path

import numpy as np

import hong2021_v74_gate_redesign as audit


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v74_query_count_energy_gate_redesign_program.json"


def test_program_is_byte_bound_and_prospective_only() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == audit.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == audit.PROGRAM_SCHEMA
    assert program["status"] == audit.PROGRAM_STATUS
    assert program["candidate_query_counts"] == [16, 32]
    assert program["scope_limits"]["this_is_not_a_generator"] is True
    assert program["scope_limits"]["validation_or_fresh_partition_access"] is False
    assert program["scope_limits"]["physical_threshold_relaxation"] is False
    assert program["predeclared_outcomes"]["full_prospective_gate_complete"] is False


def test_program_parent_bindings_load_without_raw_truth(monkeypatch) -> None:
    visited: list[Path] = []
    original = audit.sha256_file

    def traced(path: str | Path) -> str:
        resolved = Path(path).resolve()
        visited.append(resolved)
        return original(resolved)

    monkeypatch.setattr(audit, "sha256_file", traced)
    program = audit.load_program(PROGRAM, REPO)
    assert program["parent_evidence"]["V73_summary_cache_sha256"]
    allowed_gpfs = {
        Path(program["parent_evidence"][key]).resolve()
        for key in ("V73_audit_result", "V73_summary_record", "V73_summary_cache")
    }
    assert all(path.is_relative_to(REPO) or path in allowed_gpfs for path in visited)


def test_wilson_interval_contains_observed_probability() -> None:
    lower, upper = audit.wilson_interval(16000, 20000)
    assert lower < 0.8 < upper
    assert audit.wilson_interval(0, 10)[0] == 0.0
    assert np.isclose(audit.wilson_interval(10, 10)[1], 1.0)


def _assert_query_design(
    domain: str, groups: np.ndarray, query_count: int, expected_counts: list[int]
) -> None:
    selected = audit.sample_queries_count(
        domain, groups, query_count, np.random.default_rng(740010 + query_count)
    )
    assert len(selected) == len(np.unique(selected)) == query_count
    observed = np.unique(groups[selected], return_counts=True)[1]
    assert sorted(observed.tolist()) == sorted(expected_counts)


def test_query_designs_have_frozen_group_quotas() -> None:
    _assert_query_design("TNG100", np.repeat(np.arange(4), 40), 16, [4] * 4)
    _assert_query_design("TNG100", np.repeat(np.arange(4), 40), 32, [8] * 4)
    _assert_query_design("SIMBA", np.repeat(np.arange(8), 20), 16, [2] * 8)
    _assert_query_design("SIMBA", np.repeat(np.arange(8), 20), 32, [4] * 8)
    _assert_query_design("Swift", np.repeat(np.arange(20), 10), 16, [1] * 16)
    _assert_query_design("Swift", np.repeat(np.arange(20), 10), 32, [1] * 8 + [2] * 12)


def test_energy_delta_is_zero_for_identical_oracles() -> None:
    summary = {"truth_max": np.linspace(1.0, 3.0, 80)}
    queries = np.arange(16)
    oracle = np.arange(16, 32).reshape(16, 1).repeat(16, axis=1)
    assert audit.energy_delta(summary, queries, oracle, oracle) == 0.0


def _phase(absolute: float, joint: float, trials: int = 20000):
    output = {}
    for query_count in audit.QUERY_COUNTS:
        output[query_count] = {}
        for domain in audit.DOMAIN_ORDER:
            absolute_rows = np.zeros(trials, dtype=bool)
            joint_rows = np.zeros(trials, dtype=bool)
            absolute_rows[: int(absolute * trials)] = True
            joint_rows[: int(joint * trials)] = True
            output[query_count][domain] = {
                "absolute_core": absolute_rows,
                "joint": joint_rows,
            }
    return output


def test_query_selection_uses_wilson_lower_bound() -> None:
    selected, rows = audit.select_query_count(_phase(0.9, 0.9), True)
    assert selected == 16
    assert rows["16"]["attainable"] is True
    selected, _ = audit.select_query_count(_phase(0.8, 0.8), True)
    assert selected is None
    selected, _ = audit.select_query_count(_phase(0.95, 0.95), False)
    assert selected is None


def test_energy_calibration_requires_point_and_wilson_limits() -> None:
    good = {
        "family_wise_false_rejection": {
            "probability": 0.05,
            "Wilson_95": [0.047, 0.053],
        }
    }
    assert audit.energy_calibrated(good) is True
    good["family_wise_false_rejection"]["Wilson_95"][1] = 0.07
    assert audit.energy_calibrated(good) is False


def test_source_and_runner_preserve_firewalls() -> None:
    source = (REPO / "src/hong2021_v74_gate_redesign.py").read_text()
    assert "raw_train_truth_reread\": False" in source
    assert "validation_or_fresh_payload_accessed\": False" in source
    assert "full_prospective_gate_complete\": False" in source
    runner = (REPO / "scripts/hong2021_v74_gate_redesign_lageunha.sh").read_text()
    assert "taskset -c 64 nice -n 15" in runner
    assert "CUDA_VISIBLE_DEVICES=\"\"" in runner
