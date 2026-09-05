from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hong2021_v14_edm import V17_E5_SCHEMA, decoder_upsampling_for_schema
from hong2021_v17_edm import _validate_exact_experiment


def _experiment() -> dict:
    registry = json.loads(
        Path("config/hong2021_v17_development_program.json").read_text()
    )
    return registry["e5_band_balanced_denoising"]


def test_v17_registry_exposes_only_nearest_equal_band_loss() -> None:
    experiment = _experiment()
    _validate_exact_experiment(experiment)
    assert decoder_upsampling_for_schema(V17_E5_SCHEMA) == "nearest"
    assert experiment["sampling_seeds"] == {
        "TNG100": 45777, "SIMBA": 46777, "Swift": 47777
    }


def test_v17_registry_rejects_loss_coefficient_mutation() -> None:
    experiment = copy.deepcopy(_experiment())
    experiment["loss"]["coefficients"]["B"] = {
        "numerator": 1, "denominator": 2
    }
    with pytest.raises(ValueError, match="loss coefficients"):
        _validate_exact_experiment(experiment)


def test_v17_registry_rejects_fft_convention_mutation() -> None:
    experiment = copy.deepcopy(_experiment())
    experiment["loss"]["fft"]["norm"] = "backward"
    with pytest.raises(ValueError, match="FFT convention"):
        _validate_exact_experiment(experiment)
