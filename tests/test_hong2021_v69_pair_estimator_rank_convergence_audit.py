import numpy as np

from hong2021_v69_pair_estimator_rank_convergence import (
    classify,
    population_log_ratios,
)


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
