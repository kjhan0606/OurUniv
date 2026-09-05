from __future__ import annotations

import numpy as np

from hong2021_v14_location_scale import _fit_cv, predict_location_scales


def test_three_source_cv_selects_a_finite_regularization() -> None:
    rng = np.random.default_rng(31)
    features = [rng.normal(loc=offset, size=(20 + index, 2)) for index, offset in enumerate((-1.0, 0.0, 1.0))]
    targets = [
        np.column_stack((0.1 + value[:, 0], -2.0 + 0.2 * value[:, 1], -3.0 - 0.1 * value[:, 0]))
        for value in features
    ]
    selected, rows = _fit_cv(features, targets, [0.0, 1.0e-3, 1.0], 5, 7)
    assert len(selected) == 3
    assert all(value in (0.0, 1.0e-3, 1.0) for value in selected)
    assert len(rows) == 3


def test_location_scale_prediction_uses_constants_and_positive_ridges() -> None:
    model = {
        "feature_mean": [1.0, 2.0],
        "feature_std": [2.0, 4.0],
        "location": {"coefficients": [0.5, 2.0, -3.0]},
        "bands": [
            {"kind": "constant_log_rms", "log_scale": np.log(0.1)},
            {"kind": "constant_log_rms", "log_scale": np.log(0.2)},
            {"kind": "ridge_log_rms", "coefficients": [np.log(0.3), 0.0, 0.0]},
            {"kind": "ridge_log_rms", "coefficients": [np.log(0.4), 0.0, 0.0]},
        ],
    }
    location, scales = predict_location_scales(np.array([[3.0, 6.0]]), model)
    np.testing.assert_allclose(location, [-0.5])
    np.testing.assert_allclose(scales, [[0.1, 0.2, 0.3, 0.4]])
