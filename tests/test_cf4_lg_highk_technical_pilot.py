import json
from pathlib import Path

import numpy as np
import pytest

from cf4_lg_highk_technical_pilot import _load_pilot_rows


def test_program_is_frozen_but_execution_is_not_authorized():
    root = Path(__file__).parents[1]
    program = json.loads(
        (root / "config/cf4_lg_highk_technical_pilot_program_v1.json").read_text()
    )
    assert program["pilot_rows"] == [
        {"schedule_index": 0, "group_id": 0, "parent_seed": 3423},
        {"schedule_index": 64, "group_id": 1, "parent_seed": 3405},
        {"schedule_index": 128, "group_id": 2, "parent_seed": 3195},
        {"schedule_index": 192, "group_id": 3, "parent_seed": 3367},
    ]
    authorization = program["authorization"]
    assert authorization["implementation_complete"] is True
    assert authorization["technical_pilot_execution_authorized"] is False
    assert authorization["PM_authorized"] is False
    assert program["execution"]["memory"] == "36G"


def test_pilot_rows_are_one_per_group(tmp_path: Path):
    path = tmp_path / "schedule.npz"
    count = 256
    np.savez(
        path,
        schedule_index=np.arange(count),
        group_id=np.repeat(np.arange(4), 64),
        parent_seed=np.full(count, 3193),
        keys=np.tile(np.asarray([6, 6, 6, 2, 0, 0]), (count, 1)),
        fine_field_seed=np.arange(count) + 1000,
        likelihood_noise_seed=np.arange(count) + 2000,
        posterior_weight=np.full(count, 1.0 / count),
    )
    rows = _load_pilot_rows(path)
    np.testing.assert_array_equal(rows["schedule_index"], [0, 64, 128, 192])
    np.testing.assert_array_equal(rows["group_id"], [0, 1, 2, 3])


def test_pilot_rows_reject_changed_group_layout(tmp_path: Path):
    path = tmp_path / "schedule.npz"
    count = 256
    np.savez(
        path,
        schedule_index=np.arange(count),
        group_id=np.zeros(count),
        parent_seed=np.full(count, 3193),
        keys=np.tile(np.asarray([6, 6, 6, 2, 0, 0]), (count, 1)),
        fine_field_seed=np.arange(count) + 1000,
        likelihood_noise_seed=np.arange(count) + 2000,
        posterior_weight=np.full(count, 1.0 / count),
    )
    with pytest.raises(ValueError, match="one row per bridge group"):
        _load_pilot_rows(path)
