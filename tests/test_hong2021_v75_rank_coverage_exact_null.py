import hashlib
import json
from pathlib import Path

import numpy as np

import hong2021_v75_rank_coverage_exact_null as audit


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v75_rank_coverage_exact_null_audit_program.json"


def test_program_is_byte_bound_and_keeps_the_firewall() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == audit.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == audit.PROGRAM_SCHEMA
    assert program["status"] == audit.PROGRAM_STATUS
    assert program["scope_limits"]["this_is_not_a_generator"] is True
    assert program["scope_limits"]["validation_or_fresh_partition_access"] is False
    assert program["scope_limits"]["V72_stage_B_access"] is False
    assert program["prospective_exact_label_test"]["domain_alpha"] == 1 / 60
    assert program["predeclared_decision"][
        "complete_gate_or_candidate_execution_still_requires_explicit_user_approval"
    ] is True


def test_program_load_reads_only_bound_parents_and_consumed_artifacts(monkeypatch) -> None:
    visited: list[Path] = []
    original = audit.sha256_file

    def traced(path: str | Path) -> str:
        resolved = Path(path).resolve()
        visited.append(resolved)
        return original(resolved)

    monkeypatch.setattr(audit, "sha256_file", traced)
    program = audit.load_program(PROGRAM, REPO)
    allowed_gpfs = {
        Path(program["consumed_V72_diagnostic"][arm][domain]["path"]).resolve()
        for arm in ("candidate", "control")
        for domain in audit.DOMAIN_ORDER
    }
    assert all(path.is_relative_to(REPO) or path in allowed_gpfs for path in visited)


def test_query_label_table_has_exact_ranks_for_ordered_scalar_fields() -> None:
    fields = np.arange(audit.FIELD_COUNT, dtype=np.float64)[:, None]
    table = audit.query_label_table(fields, np.random.default_rng(75))
    assert table["voxels"] == 1
    assert np.array_equal(table["histogram"], np.eye(17, dtype=np.int64))
    assert table["adjacent_tie_pairs"] == 0


def test_query_label_table_is_field_permutation_equivariant_without_ties() -> None:
    rng = np.random.default_rng(750010)
    fields = rng.normal(size=(17, 128))
    first = audit.query_label_table(fields, np.random.default_rng(1))
    permutation = rng.permutation(17)
    second = audit.query_label_table(fields[permutation], np.random.default_rng(2))
    inverse = np.argsort(permutation)
    assert np.array_equal(first["histogram"], second["histogram"][inverse])
    assert np.array_equal(first["coverage68"], second["coverage68"][inverse])
    assert np.array_equal(first["coverage95"], second["coverage95"][inverse])


def test_tie_breaking_is_deterministic_and_retains_uniform_rank_counts() -> None:
    fields = np.ones((17, 257), dtype=np.float64)
    first = audit.query_label_table(fields, np.random.default_rng(750011))
    second = audit.query_label_table(fields, np.random.default_rng(750011))
    assert np.array_equal(first["histogram"], second["histogram"])
    assert first["histogram"].sum() == 17 * 257
    assert np.array_equal(first["histogram"].sum(axis=0), np.full(17, 257))
    assert first["adjacent_tie_fraction"] == 1.0


def test_assignment_statistics_matches_direct_query_selection() -> None:
    rng = np.random.default_rng(750012)
    fields = rng.normal(size=(3, 17, 200))
    table = audit.build_table_from_fields(fields, 750013)
    labels = np.asarray([[0, 1, 2], [16, 16, 16]], dtype=np.int8)
    measured = audit.assignment_statistics(table, labels)
    for row, assignment in enumerate(labels):
        histogram = sum(table["histogram"][q, assignment[q]] for q in range(3))
        expected = 3 * 200 / 17
        tv = 0.5 * np.abs(histogram - expected).sum() / (3 * 200)
        assert np.isclose(measured["rank_tv"][row], tv)


def test_conditional_p_value_uses_plus_one_and_upper_tail() -> None:
    observed = {"rank_tv": 3.0, "coverage_deviation": 2.0, "composite": 1.0}
    null = {
        "rank_tv": np.asarray([1.0, 2.0, 3.0]),
        "coverage_deviation": np.asarray([1.0, 2.0, 3.0]),
        "composite": np.asarray([0.0, 1.0, 2.0]),
    }
    p = audit.conditional_p_values(observed, null)
    assert p["rank_tv"] == 0.5
    assert p["coverage_deviation"] == 0.75
    assert p["composite"] == 0.75


def test_perfect_spatial_dependence_exposes_old_rank_threshold() -> None:
    fields = audit._synthetic_fields(
        "perfect_within_field_dependence",
        queries=32,
        voxels=64,
        rng=np.random.default_rng(750014),
    )
    table = audit.build_table_from_fields(fields, 750015)
    labels = np.random.default_rng(750016).integers(0, 17, size=(2000, 32))
    values = audit.assignment_statistics(table, labels)
    assert np.mean(values["rank_tv"] <= 0.05) < 0.1


def test_source_and_runner_forbid_new_scientific_access() -> None:
    source = (REPO / "src/hong2021_v75_rank_coverage_exact_null.py").read_text()
    assert "validation_or_fresh_payload_accessed\": False" in source
    assert "V72_verdict_changed\": False" in source
    assert "complete_gate_or_new_candidate_execution_authorized\": False" in source
    runner = (
        REPO / "scripts/hong2021_v75_rank_coverage_exact_null_lageunha.sh"
    ).read_text()
    assert "taskset -c 64 nice -n 15" in runner
    assert "CUDA_VISIBLE_DEVICES=\"\"" in runner
