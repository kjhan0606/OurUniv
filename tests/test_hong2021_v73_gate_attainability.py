import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

import hong2021_v73_gate_attainability as audit
from hong2021_residual_evaluate import density_statistics


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v73_train_truth_gate_attainability_audit_program.json"


def test_program_is_byte_bound_and_preserves_the_data_firewall() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == audit.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == audit.PROGRAM_SCHEMA
    assert program["status"] == audit.PROGRAM_STATUS
    assert program["scope_limits"]["this_is_not_a_V73_generator"] is True
    assert program["scope_limits"]["no_fresh_partition_consumed"] is True
    assert program["resource_and_firewall"]["device"] == "CPU only"
    assert program["resource_and_firewall"]["V72_stage_B_access"] == "forbidden"
    assert program["resource_and_firewall"]["Astrid_access"] == "forbidden"
    assert program["resource_and_firewall"]["historical_or_independent_EAGLE_access"] == "forbidden"
    assert set(program["already_consumed_train_inputs"]) == set(audit.DOMAIN_ORDER)


def test_program_parent_and_measurement_bindings_load_without_train_payload() -> None:
    program = audit.load_program(PROGRAM, REPO)
    assert program["parent_evidence"]["v72_result_record_sha256"]
    assert program["frozen_measurement_sources"]["field_gate_sha256"]


def test_environment_field_order_matches_the_frozen_implementation() -> None:
    assert tuple(density_statistics(np.ones((3, 3, 3))).keys()) == (
        audit.ALL_ENVIRONMENT_FIELDS
    )


def test_pooled_upper_quantile_is_exact_with_repeated_cubes() -> None:
    generator = np.random.default_rng(730000)
    cubes = generator.normal(size=(5, 40))
    top = np.sort(cubes, axis=1)[:, ::-1]
    selected = np.asarray([0, 0, 2, 4, 4, 4], dtype=np.int64)
    pooled = cubes[selected].reshape(-1)
    for quantile in (0.8, 0.9, 0.99999):
        observed = audit.pooled_quantile_from_top(
            top, selected, quantile=quantile, voxels_per_cube=cubes.shape[1]
        )
        assert np.isclose(observed, np.quantile(pooled, quantile), rtol=0.0, atol=1e-14)


def test_pooled_upper_quantile_rejects_an_insufficient_top_cache() -> None:
    top = np.asarray([[4.0], [3.0]])
    selected = np.asarray([0, 1])
    try:
        audit.pooled_quantile_from_top(
            top, selected, quantile=0.6, voxels_per_cube=10
        )
    except ValueError as error:
        assert "insufficient" in str(error)
    else:
        raise AssertionError("an insufficient order-statistic cache was accepted")


def test_fast_ks_matches_scipy_with_ties() -> None:
    generator = np.random.default_rng(730004)
    first = generator.integers(-3, 4, size=31)
    second = generator.integers(-3, 4, size=57)
    observed = audit.ks_statistic_fast(first, second)
    expected = ks_2samp(first, second, method="exact").statistic
    assert observed == expected


def test_grouped_query_and_oracle_sampling_is_disjoint() -> None:
    groups = np.repeat(np.arange(4), 30)
    generator = np.random.default_rng(730005)
    queries = audit.sample_queries("TNG100", groups, generator)
    donors = audit.sample_same_group_oracle(groups, queries, generator)
    assert queries.shape == (16,)
    assert donors.shape == (16, 16)
    assert np.array_equal(np.unique(groups[queries], return_counts=True)[1], [4] * 4)
    assert not set(queries.tolist()) & set(donors.reshape(-1).tolist())
    assert np.all(groups[donors] == groups[queries, None])


def test_trial_metric_full_and_absolute_only_paths() -> None:
    objects = 40
    bins = 32
    histogram = np.zeros((objects, 400), dtype=np.int64)
    histogram[:, 200] = 64**3
    deterministic_histogram = np.zeros_like(histogram)
    deterministic_histogram[:, 100] = 64**3
    environment = np.zeros((objects, len(audit.ALL_ENVIRONMENT_FIELDS)))
    summary = {
        "truth_top": np.tile(np.linspace(4.0, 2.0, audit.TOP_VALUES), (objects, 1)),
        "truth_max": np.full(objects, 4.0),
        "truth_delta2": np.ones(objects),
        "truth_power": np.ones((objects, bins)),
        "truth_2pcf": np.zeros((objects, bins)),
        "truth_hist": histogram,
        "truth_env": environment,
        "residual_ms": np.ones(objects),
        "det_2pcf": np.ones((objects, bins)),
        "det_hist": deterministic_histogram,
        "det_env": np.ones_like(environment),
    }
    queries = np.arange(16)
    donors = np.arange(16, 32).reshape(16, 1).repeat(16, axis=1)
    k = np.linspace(0.1, 10.0, bins)
    count = np.ones(bins, dtype=np.int64)
    radius = (np.arange(bins) + 0.5) * 0.3125
    full = audit.trial_metrics(
        summary, queries, summary, donors, k, count, radius, donors
    )
    assert full["absolute_core"] is True
    assert full["morphology_core"] is True
    assert full["joint"] is True
    assert not bool(full["energy_A_better_B"])
    absolute = audit.trial_metrics(
        summary,
        queries,
        summary,
        donors,
        k,
        count,
        radius,
        absolute_only=True,
    )
    assert absolute["absolute_core"] is True
    assert "joint" not in absolute


def _boolean_trials(value: bool, trials: int = 10) -> dict[str, np.ndarray]:
    return {
        "absolute_core": np.full(trials, value, dtype=bool),
        "joint": np.full(trials, value, dtype=bool),
    }


def test_predeclared_decision_prefers_gate_redesign() -> None:
    same = {domain: _boolean_trials(True) for domain in audit.DOMAIN_ORDER}
    same["SIMBA"] = _boolean_trials(False)
    cross = {"TNG100_to_SIMBA": {"absolute_core": np.ones(10, dtype=bool)}}
    energy = {
        domain: {"underpowered_interval_contains_zero": True}
        for domain in audit.DOMAIN_ORDER
    }
    result = audit.decide(same, cross, energy, {"material": False})
    assert result["gate_redesign_required"] is True
    assert result["classification"] == (
        "sampling_sensitive_V72_gate_requires_null_calibrated_redesign"
    )
    assert result["next"].startswith("stop_new_Hong_candidates")


def test_runner_keeps_cross_domain_stress_absolute_only() -> None:
    source = (REPO / "src/hong2021_v73_gate_attainability.py").read_text()
    cross = source[source.index("def run_cross_domain") : source.index("def _absolute_band")]
    assert "absolute_only=True" in cross
    assert "training_or_model_sampling_performed\": False" in source
    assert "validation_or_fresh_payload_accessed\": False" in source
