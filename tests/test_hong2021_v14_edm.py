from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hong2021_v14_edm import (
    source_balanced_feature_standardization,
    source_balanced_sigma_data,
)
from hong2021_v14_mean_correction import DOMAINS


@dataclass
class _RMS:
    cache_rms: float


class _FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, value: float, samples: int) -> None:
        self.value = value
        self.samples = samples

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        condition = torch.zeros(4, 4, 4, 4)
        condition[0] = self.value
        condition[1] = self.value / 2
        condition[2] = self.value * 2
        field = torch.zeros(1, 4, 4, 4)
        return condition, field, field, field


def test_sigma_data_uses_equal_source_rms_not_sample_counts() -> None:
    datasets = dict(zip(DOMAINS, (_RMS(1.0), _RMS(2.0), _RMS(3.0)), strict=True))
    actual = source_balanced_sigma_data(datasets)  # type: ignore[arg-type]
    assert actual == np.sqrt((1.0 + 4.0 + 9.0) / 3.0)


def test_feature_standardization_gives_three_sources_equal_weight() -> None:
    datasets = {
        "TNG100": _FeatureDataset(1.0, 20),
        "SIMBA": _FeatureDataset(2.0, 3),
        "Swift-EAGLE": _FeatureDataset(3.0, 7),
    }
    fit = source_balanced_feature_standardization(datasets, batch_size=4)
    assert fit["source_weight"] == 1.0 / 3.0
    assert fit["source_samples"] == {"TNG100": 20, "SIMBA": 3, "Swift-EAGLE": 7}
    assert fit["uses_density_truth"] is False
