import numpy as np

from hong2021_v67_nonlocal_context_predictability_audit import (
    classify,
    leave_one_domain_out,
    pooled_metrics,
)


def test_v67_leave_one_domain_out_probe_uses_only_training_fold_scaling() -> None:
    generator = np.random.default_rng(167)
    features = generator.normal(size=(48, 33))
    response = 2.0 * features[:, 0] - 0.5 * features[:, 1]
    prediction, reference, folds = leave_one_domain_out(
        features, response, 10.0, record_fits=True
    )
    metrics = pooled_metrics(prediction, response, reference)
    assert tuple(folds) == ("TNG100", "SIMBA", "Swift")
    assert all(row["training_objects"] == 32 for row in folds.values())
    assert metrics["Ridge_RMSE"] < metrics["constant_reference_RMSE"]
    assert metrics["Pearson_prediction_response"] > 0.8


def test_v67_classification_requires_predictability_and_permutation() -> None:
    selected = classify(True, True, True)
    nonsignificant = classify(True, True, False)
    unpredictable = classify(True, False, True)
    failed = classify(False, True, True)
    assert selected[2] is True and "nonlocal_context" in selected[1]
    assert nonsignificant[2] is False
    assert unpredictable[2] is False
    assert failed[2] is False
