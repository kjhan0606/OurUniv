import hashlib
from pathlib import Path

import h5py
import numpy as np

import hong2021_v36_local_tail as v36

from hong2021_v36_local_tail import (
    FULL_FEATURES,
    PROGRAM_SHA256,
    local_features_targets,
    pinball_loss,
    selected_coordinates,
)


REPO = Path(__file__).resolve().parents[1]


def test_v36_program_hash_and_firewall():
    path = REPO / "config/hong2021_v36_local_tail_sufficiency_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"field_clipping": false' in text
    assert '"posthoc_Ak": false' in text
    assert '"Astrid_access": "forbidden"' in text


def test_lattice_and_native_selected_coordinate_mapping():
    selected = np.asarray([0, 1, 4095, 4096 + 16])
    first = selected_coordinates(selected, 0, 4096, lattice=True)
    np.testing.assert_array_equal(first[0], [0, 0, 0])
    np.testing.assert_array_equal(first[-1], [60, 60, 60])
    second = selected_coordinates(selected, 1, 4096, lattice=True)
    np.testing.assert_array_equal(second[0], [1, 4, 0])
    native = selected_coordinates(np.asarray([0, 64**2 + 2 * 64 + 3]), 0, 64**3, lattice=False)
    np.testing.assert_array_equal(native[1], [1, 2, 3])


def test_pinball_loss_is_zero_for_exact_prediction():
    value = np.asarray([-1.0, 0.0, 2.0])
    assert pinball_loss(value, value, 0.999) == 0


def test_tail_rule_requires_registered_coverage():
    ratios = {
        domain: {
            str(q): {"candidate": 0.95 if q in (0.001, 0.999) else 0.89}
            for q in v36.QUANTILES
        }
        for domain in v36.DOMAIN_ORDER
    }
    validation = {
        domain: {
            str(q): {"model": {"coverage_error": 0.0}}
            for q in v36.QUANTILES
        }
        for domain in v36.DOMAIN_ORDER
    }
    assert v36._tail_rule(ratios, "candidate", validation, "model")
    validation["Swift"]["0.01"]["model"]["coverage_error"] = 0.006
    assert not v36._tail_rule(ratios, "candidate", validation, "model")


def test_local_feature_shape_on_synthetic_hdf5(tmp_path):
    data_path = tmp_path / "data.h5"
    cache_path = tmp_path / "cache.h5"
    with h5py.File(data_path, "w") as data:
        x = data.create_dataset("input", shape=(1, 3, 64, 64, 64), dtype="f4")
        y = data.create_dataset("target", shape=(1, 1, 64, 64, 64), dtype="f4")
        x[0, 0, 0, 0, 0] = 2
        x[0, 1, 0, 0, 0] = 5
        x[0, 2, 0, 0, 0] = 7
        y[0, 0] = 0.2
    with h5py.File(cache_path, "w") as cache:
        cache.create_dataset("conditional_mean", data=np.zeros((1, 1, 64, 64, 64), dtype=np.float32))
        cache.create_dataset("predicted_residual_dc", data=np.zeros(1, dtype=np.float32))
    with h5py.File(data_path, "r") as data, h5py.File(cache_path, "r") as cache:
        feature, target = local_features_targets(
            data, cache, 0, np.asarray([[0, 0, 0], [1, 1, 1]])
        )
    assert feature.shape == (2, FULL_FEATURES)
    assert target.shape == (2,)
    assert np.allclose(target, 0.2)
