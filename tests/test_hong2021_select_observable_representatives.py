from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import h5py
import json


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_select_observable_representatives import (
    V14_AUDIT_SCHEMA,
    balanced_audit_feature_fit,
    farthest_feature_subset,
)


def test_farthest_feature_subset_is_deterministic_and_spread() -> None:
    features = np.asarray([[0.0], [1.0], [4.0], [10.0], [5.0]])
    first = farthest_feature_subset(features, 4)
    second = farthest_feature_subset(features, 4)
    np.testing.assert_array_equal(first, [0, 3, 4, 1])
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 4


def test_balanced_audit_fit_gives_sources_equal_weight(tmp_path: Path) -> None:
    paths = []
    for index, (samples, value) in enumerate(((20, 0.0), (3, 1.0), (7, 2.0))):
        path = tmp_path / f"source_{index}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "observable_context_features",
                data=np.full((samples, len(FEATURE_NAMES)), value),
            )
            handle.attrs["schema"] = V14_AUDIT_SCHEMA
            handle.attrs["domain"] = f"domain-{index}"
            handle.attrs["feature_uses_target"] = False
            handle.attrs["feature_names"] = json.dumps(list(FEATURE_NAMES))
        paths.append(path)
    fit = balanced_audit_feature_fit(paths)
    np.testing.assert_allclose(fit["mean"], 1.0)
    np.testing.assert_allclose(fit["std"], np.sqrt(2.0 / 3.0))
    assert fit["uses_density_truth"] is False
