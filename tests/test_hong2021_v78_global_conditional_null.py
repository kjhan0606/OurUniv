import hashlib
import json
from pathlib import Path

import numpy as np

import hong2021_v78_global_conditional_null as audit


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v78_global_conditional_null_redesign_audit_program.json"


def test_program_is_byte_bound_and_preserves_firewall() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == audit.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == audit.PROGRAM_SCHEMA
    assert program["status"] == audit.PROGRAM_STATUS
    assert program["complete_global_rule"]["global_alpha"] == 0.05
    assert program["scope_limits"]["validation_input_or_target_payload_access"] is False
    assert program["authorization"]["construct_or_sample_a_candidate"] is False


def test_upper_tail_is_inclusive_and_add_one() -> None:
    reference = np.asarray([1.0, 2.0, 3.0])
    measured = audit.upper_tail_p(reference, np.asarray([0.0, 2.0, 4.0]))
    assert np.array_equal(measured, np.asarray([1.0, 0.75, 0.25]))


def test_sparse_statistic_parentheses_select_most_anomalous_family() -> None:
    family_p = np.asarray([[0.5, 0.01, 0.9], [0.2, 0.3, 0.4]])
    sparse = (-np.log(family_p)).max(axis=1)
    assert np.allclose(sparse, [-np.log(0.01), -np.log(0.2)])
    assert not np.array_equal(sparse, -np.log(family_p).max(axis=1))


def test_global_p_has_one_half_budget_per_block() -> None:
    pe = np.asarray([0.025, 0.5, 0.01])
    rc = np.asarray([0.5, 0.025, 0.5])
    assert np.allclose(audit.global_p_value(pe, rc), [0.05, 0.05, 0.02])


def test_mahalanobis_score_is_zero_at_center() -> None:
    rng = np.random.default_rng(780010)
    values = rng.normal(size=(1000, 3))
    model = audit.mahalanobis_model(values)
    score = audit.mahalanobis_score(model["center"][None], model)
    assert score.shape == (1,)
    assert score[0] == 0.0
    assert model["ridge"] > 0


def test_family_matrices_apply_frozen_transforms() -> None:
    phase = {}
    for domain in audit.DOMAIN_ORDER:
        phase[f"{domain}__q_delta"] = np.asarray([0.0, 0.1])
        phase[f"{domain}__q4_ratio"] = np.asarray([1.0, 2.0])
    q = audit.family_matrix(phase, "q99_999", "coherent_q99_999_bias")
    q4 = audit.family_matrix(phase, "Q4", "coherent_Q4_excess")
    assert np.allclose(q[:, 0], [0.15, 0.25])
    assert np.allclose(q4[:, 0], [np.log(1.5), np.log(3.0)])


def test_rank_coverage_block_preserves_pairs_and_adjusts_six(monkeypatch) -> None:
    class Arrays(dict):
        pass

    arrays = Arrays(
        {
            "null__x__rank_tv_p": np.linspace(0.001, 1.0, 100000),
            "null__x__coverage_deviation_p": np.linspace(1.0, 0.001, 100000),
        }
    )
    value = audit.rank_coverage_block_samples(
        arrays, "x", 100, np.random.default_rng(780011)
    )
    assert value.shape == (100,)
    assert np.all((value >= 0) & (value <= 1))


def test_source_and_runner_preserve_firewalls() -> None:
    source = (REPO / "src/hong2021_v78_global_conditional_null.py").read_text()
    assert 'validation_input_or_target_payload_accessed": False' in source
    assert 'candidate_or_fresh_payload_execution_authorized": False' in source
    assert 'training_or_model_sampling_performed": False' in source
    assert "(-np.log(family_p)).max(axis=1)" in source
    runner = (
        REPO / "scripts/hong2021_v78_global_conditional_null_lageunha.sh"
    ).read_text()
    assert "taskset -c 64 nice -n 15" in runner
    assert 'CUDA_VISIBLE_DEVICES=""' in runner
