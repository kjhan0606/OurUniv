import json
from pathlib import Path

import numpy as np

from hong2021_v68_population_pair_objective_audit import classify, population_row


REPO = Path(__file__).resolve().parents[1]


def test_v68_population_gradient_aggregates_moments_before_ratio() -> None:
    predicted = np.full((3, 16, 3), 2.0)
    truth = np.ones_like(predicted)
    jacobian = np.ones((3, 16, 3, 495))
    nll = np.full((3, 16, 495), 0.5)
    row = population_row([0, 4, 8, 12], predicted, truth, jacobian, nll)
    assert np.allclose(row["population_log_ratio"], np.log(2.0))
    assert row["pair_gradient_L2"] > 0.0
    assert row["bounded_NLL_gradient_L2"] > 0.0


def test_v68_classification_requires_stability_coherence_and_safety() -> None:
    selected = classify(True, True, True, True)
    unstable = classify(True, False, True, True)
    incoherent = classify(True, True, False, True)
    failed = classify(False, True, True, True)
    assert selected[2] is True and "population_pair" in selected[1]
    assert unstable[2] is False
    assert incoherent[2] is False
    assert failed[2] is False


def test_v68_result_rejects_pair_loss_and_selects_rank_convergence() -> None:
    record = json.loads((REPO / "config/hong2021_v68_result_record.json").read_text())
    assert record["audit"]["candidate_selected"] is False
    assert record["population_audit"]["population_value_stable"] is False
    assert record["population_audit"]["optimization_safe"] is True
    assert record["selected_next_step"]["rank_levels"] == [8, 16, 32, 64]
    assert record["firewall"]["training_or_refit_performed"] is False
    assert record["firewall"]["independent_gate_locked"] is True
