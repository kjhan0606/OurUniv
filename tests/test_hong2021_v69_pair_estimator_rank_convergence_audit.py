import json
from pathlib import Path

import numpy as np

from hong2021_v69_pair_estimator_rank_convergence import (
    classify,
    population_log_ratios,
)


REPO = Path(__file__).resolve().parents[1]


def test_v69_population_ratio_aggregates_objects_before_log() -> None:
    predicted = np.full((3, 16, 3), 2.0)
    truth = np.ones_like(predicted)
    assert np.allclose(population_log_ratios(predicted, truth), np.log(2.0))


def test_v69_classification_requires_adjacent_and_independent_convergence() -> None:
    selected = classify(True, True, True)
    adjacent = classify(True, False, True)
    independent = classify(True, True, False)
    failed = classify(False, True, True)
    assert selected[2] is True and "rank64" in selected[1]
    assert adjacent[2] is False
    assert independent[2] is False
    assert failed[2] is False


def test_v69_result_closes_pair_objective_and_selects_joint_model_preflight() -> None:
    record = json.loads((REPO / "config/hong2021_v69_result_record.json").read_text())
    assert record["audit"]["candidate_selected"] is False
    assert record["rank_convergence"]["rank64_estimator_selected"] is False
    assert record["selected_next_step"]["action"] == (
        "freeze a train-only query-aligned joint spatial likelihood model preflight with no pair loss"
    )
    assert record["firewall"]["gradient_computed"] is False
    assert record["firewall"]["independent_gate_locked"] is True
