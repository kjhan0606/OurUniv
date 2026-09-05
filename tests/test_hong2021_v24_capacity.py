from __future__ import annotations

from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import SUPPORTED_CHECKPOINT_SCHEMAS, V24_E12_SCHEMA
from hong2021_v24_edm import PARAMETERS


def test_v24_schema_and_base48_parameter_count_are_frozen() -> None:
    assert V24_E12_SCHEMA in SUPPORTED_CHECKPOINT_SCHEMAS
    model = ObservableContextUNet(base_channels=48)
    assert sum(value.numel() for value in model.parameters()) == PARAMETERS == 8133361
