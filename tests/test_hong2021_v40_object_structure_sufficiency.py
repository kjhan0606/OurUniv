import hashlib
from pathlib import Path

import numpy as np

from hong2021_v40_object_structure_sufficiency import (
    BACKBONE_COLUMNS,
    PROGRAM_SHA256,
    block_max,
    classify,
    component_recall,
    object_features,
    top_count_mask,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall() -> None:
    path = REPO / "config/hong2021_v40_object_structure_sufficiency_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"new_generator_fit_or_sampling": false' in text
    assert '"posthoc_Ak": false' in text
    assert '"Astrid_access": "forbidden"' in text
    assert len(BACKBONE_COLUMNS) == 57


def test_block_max_and_deterministic_top_count() -> None:
    cube = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    maximum = block_max(cube)
    assert maximum.shape == (16, 16, 16)
    assert maximum[0, 0, 0] == cube[3, 3, 3]
    score = np.zeros((16, 16, 16), dtype=np.float32)
    selected = top_count_mask(score, 3)
    np.testing.assert_array_equal(np.flatnonzero(selected), [0, 1, 2])


def test_component_recall_counts_and_weights_native_components() -> None:
    positive = np.zeros((64, 64, 64), dtype=bool)
    positive[1:3, 1:3, 1:3] = True
    positive[40, 40, 40] = True
    mass = np.zeros_like(positive, dtype=np.float64)
    mass[1:3, 1:3, 1:3] = 1.0
    mass[40, 40, 40] = 8.0
    selected = np.zeros((16, 16, 16), dtype=bool)
    selected[0, 0, 0] = True
    result = component_recall(positive, mass, selected)
    assert result["components"] == 2
    assert result["hit_components"] == 1
    assert result["mass"] == 16.0
    assert result["hit_mass"] == 8.0


def test_object_feature_order_and_classification() -> None:
    fields = {
        "log1p_block_count": np.ones((16, 16, 16)),
        "block_mean_velocity_kms": np.full((16, 16, 16), 2.0),
        "exact_population_velocity_dispersion_kms": np.full((16, 16, 16), 3.0),
        "backbone_mean_y": np.full((16, 16, 16), 4.0),
    }
    feature = object_features(fields)
    assert feature.shape == (28,)
    np.testing.assert_array_equal(feature[[0, 7, 14, 21]], [1.0, 2.0, 3.0, 4.0])
    assert classify(True, True)[0] == "object_amplitude_and_structure_location_are_transferably_observable"
    assert classify(False, False)[0] == "backbone_observables_are_insufficient_for_rare_structure_reconstruction"
