from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hong2021_v84c0_unique_cell_tail_audit import (
    cube_cells,
    partition_digest,
    prospective_partition,
    tail_shape,
    weighted_quantile,
)


REPO = Path(__file__).resolve().parents[1]


def test_cube_cells_wrap_periodically_and_are_unique() -> None:
    cells = cube_cells(np.asarray([-1, 78, 79]), 80)
    assert len(cells) == 64**3
    assert len(np.unique(cells)) == 64**3
    assert cells.min() >= 0
    assert cells.max() < 80**3


def test_weighted_quantile_respects_cell_multiplicity_weights() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 100.0])
    weights = np.asarray([1.0, 1.0, 1.0, 0.01])
    assert weighted_quantile(values, weights, 0.5) == 1.0


def test_exponential_tail_has_equal_near_and_far_slopes() -> None:
    probability = (np.arange(1, 1_000_001) - 0.5) / 1_000_000
    excess = -0.5 * np.log1p(-probability)
    row = tail_shape(excess, np.ones_like(excess), 2_000_000.0)
    assert abs(row["far_over_near_scale"] - 1.0) < 2.0e-5


def test_prospective_partition_reserves_outer_payload_and_zero_overlap() -> None:
    v35 = json.loads(
        (REPO / "config/hong2021_v35_residual_spectrum_phase_program.json").read_text()
    )
    partition = prospective_partition(v35)
    assert {key: len(partition["TNG100"][key]) for key in ("inner", "outer", "embargo")} == {
        "inner": 139,
        "outer": 32,
        "embargo": 190,
    }
    assert partition["TNG100"]["inner_outer_target_cell_intersection"] == 0
    assert partition["SIMBA"]["inner_groups"] == [16, 17, 18, 23]
    assert partition["SIMBA"]["outer_groups"] == [19, 22]
    assert partition["Swift"]["inner_groups"] == [1, 2, 3, 5, 7, 10, 11, 12, 13, 14, 15, 17, 18, 19]
    assert partition["Swift"]["outer_groups"] == [0, 4, 16]
    assert partition_digest(partition) == partition_digest(prospective_partition(v35))
