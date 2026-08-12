from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import h5py

from hong2021_v84a_group_tail_attribution import (
    PopulationAccumulator,
    direct_tail_failure,
    group_leakage,
    periodic_nearest_distance,
)


REPO = Path(__file__).resolve().parents[1]


def test_periodic_nearest_distance_wraps_box() -> None:
    query = np.asarray([[0.2, 0.0, 0.0], [5.0, 5.0, 5.0]])
    reference = np.asarray([[9.9, 0.0, 0.0]])
    observed = periodic_nearest_distance(query, reference, 10.0)
    assert np.allclose(observed, [0.3, np.sqrt(74.01)])


def test_population_tail_failure_uses_both_sides_and_frozen_interval() -> None:
    passing = {
        "direct_PIT": {
            "tail_exceedance": {
                "0.001": {"lower_over_expected": 1.0, "upper_over_expected": 1.0},
                "0.0001": {"lower_over_expected": 0.9, "upper_over_expected": 1.2},
            }
        }
    }
    assert not direct_tail_failure(passing)
    passing["direct_PIT"]["tail_exceedance"]["0.0001"]["upper_over_expected"] = 1.3
    assert direct_tail_failure(passing)


def test_population_accumulator_reports_uniform_PIT() -> None:
    positions = np.arange(8)
    accumulator = PopulationAccumulator(positions, True)
    condition = np.zeros((7, 2, 2, 2), dtype=np.float32)
    condition[3] = np.arange(8).reshape(2, 2, 2)
    target = np.zeros((1, 2, 2, 2), dtype=np.float32)
    uniform = (np.arange(8, dtype=np.float64) + 0.5).reshape(1, 2, 2, 2) / 8
    accumulator.add_probe(condition, target)
    accumulator.add_pit(uniform, condition, target)
    row = accumulator.summary()
    assert row["direct_PIT"]["mean"] == 0.5
    assert row["direct_PIT"]["voxels"] == 8
    assert row["direct_PIT"]["central_coverage"]["50"] == 0.5


def test_group_leakage_uses_only_consumed_validation_selection(tmp_path) -> None:
    train_path = tmp_path / "train.h5"
    validation_path = tmp_path / "validation.h5"
    with h5py.File(train_path, "w") as train:
        train.create_dataset(
            "center_position_mpc_h",
            data=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        train.create_dataset("realization", data=np.asarray([1, 1]))
    with h5py.File(validation_path, "w") as validation:
        validation.create_dataset(
            "center_position_mpc_h",
            data=np.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        )
        validation.create_dataset("realization", data=np.asarray([1, 2]))
    with h5py.File(train_path) as train, h5py.File(validation_path) as validation:
        row = group_leakage("SIMBA", train, validation, [0], [1], [1])
    assert row["consumed_validation_groups"] == [2]
    assert row["fit_consumed_group_intersection"] == []
    assert row["consumed_nearest_fit_center_within_same_group"]["available"] is False


def test_amended_program_freezes_validation_object_counts_and_new_output() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v84a_group_tail_attribution_program.json").read_text()
    )
    assert program["amendment"]["revision"] == 1
    assert {
        domain: row["validation_objects"]
        for domain, row in program["domains"].items()
    } == {"TNG100": 93, "SIMBA": 64, "Swift": 148}
    assert "v84a1_group_tail_attribution" in program["output"]
