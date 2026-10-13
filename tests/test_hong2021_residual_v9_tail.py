from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_residual_v9_tail import (
    balanced_tail_weights,
    voxel_tail_weights,
)


def test_balanced_tail_weights_have_unit_equal_source_expectation() -> None:
    tng = np.asarray([50, 300, 100, 40, 10])
    simba = np.asarray([100, 250, 100, 45, 5])
    result = balanced_tail_weights(tng, simba)
    probability = np.asarray(result["probability"]["equal_source"])
    weights = np.asarray(result["weights"])
    np.testing.assert_allclose(np.sum(probability * weights), 1.0)
    assert weights[-1] > weights[2]
    assert weights[0] > weights[1]


def test_voxel_tail_weights_follow_density_boundaries() -> None:
    truth = torch.tensor([-2.0, -1.0, -0.1, 0.0, 1.5, 2.5])[None, None, :, None, None] / 4.5
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    actual = voxel_tail_weights(truth, weights).flatten()
    torch.testing.assert_close(actual, torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0, 5.0]))
