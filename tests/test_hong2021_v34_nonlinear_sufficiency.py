import hashlib
from pathlib import Path

import numpy as np

from hong2021_v34_nonlinear_sufficiency import (
    MODEL_FEATURE_COUNTS,
    PATCH_OFFSETS,
    PROGRAM_SHA256,
    StreamingMetrics,
    multiscale_features,
    periodic_oriented_patch,
    repeat_parent_patch,
)


REPO = Path(__file__).resolve().parents[1]


def test_v34_program_hash_and_firewall():
    path = REPO / "config/hong2021_v34_oriented_nonlinear_sufficiency_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"posthoc_Ak": false' in text
    assert '"Astrid_access": "forbidden"' in text
    assert '"historical_EAGLE_access": "forbidden"' in text
    assert '"validation_used_for_fit_or_early_stopping": false' in text


def test_oriented_patch_retains_lexicographic_neighbor_identity():
    field = np.zeros((4, 4, 4), dtype=np.float64)
    field[1, 2, 3] = 7
    patch = periodic_oriented_patch(field)
    center = (0, 2, 0)
    offset = (1, 0, -1)
    assert patch[center][PATCH_OFFSETS.index(offset)] == 7
    assert np.count_nonzero(patch[center]) == 1


def test_parent_patch_is_mapped_to_each_containing_two_cube():
    parent = np.arange(2**3 * 27).reshape(2, 2, 2, 27)
    repeated = repeat_parent_patch(parent)
    assert repeated.shape == (4, 4, 4, 27)
    np.testing.assert_array_equal(repeated[2, 0, 3], parent[1, 0, 1])


def test_multiscale_feature_shape_and_exact_sig_v():
    count = np.zeros((64, 64, 64), dtype=np.float64)
    velocity = np.zeros_like(count)
    dispersion = np.zeros_like(count)
    count[0, 0, 0] = 2
    velocity[0, 0, 0] = 5
    dispersion[0, 0, 0] = np.sqrt(50.0)
    backbone = np.zeros_like(count)
    truth = np.ones_like(count) * 0.2
    feature, target = multiscale_features(
        count, velocity, dispersion, backbone, truth, 4
    )
    assert feature.shape == (16, 16, 16, MODEL_FEATURE_COUNTS["nonlinear_oriented_multiscale"])
    assert target.shape == (16, 16, 16)
    assert np.isclose(feature[0, 0, 0, 2], 5.0)
    assert np.allclose(target, 0.2)


def test_streaming_metrics_matches_direct_computation():
    target = np.asarray([0.0, 1.0, 2.0, 3.0])
    prediction = np.asarray([0.5, 0.5, 2.5, 2.5])
    metric = StreamingMetrics()
    metric.add(prediction[:2], target[:2])
    metric.add(prediction[2:], target[2:])
    result = metric.result()
    assert result["rows"] == 4
    assert np.isclose(result["rmse"], 0.5)
    assert np.isclose(
        result["pearson_prediction_target"], np.corrcoef(prediction, target)[0, 1]
    )
