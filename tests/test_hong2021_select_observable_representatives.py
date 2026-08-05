from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_select_observable_representatives import farthest_feature_subset


def test_farthest_feature_subset_is_deterministic_and_spread() -> None:
    features = np.asarray([[0.0], [1.0], [4.0], [10.0], [5.0]])
    first = farthest_feature_subset(features, 4)
    second = farthest_feature_subset(features, 4)
    np.testing.assert_array_equal(first, [0, 3, 4, 1])
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 4
