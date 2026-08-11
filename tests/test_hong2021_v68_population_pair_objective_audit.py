import numpy as np

from hong2021_v68_population_pair_objective_audit import classify, population_row


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
