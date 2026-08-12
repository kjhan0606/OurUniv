from __future__ import annotations

import json
from pathlib import Path

from hong2021_v84b_contract import group_partition, partition_digest


REPO = Path(__file__).resolve().parents[1]


def test_group_partition_is_disjoint_and_expected() -> None:
    v35 = json.loads(
        (REPO / "config/hong2021_v35_residual_spectrum_phase_program.json").read_text()
    )
    partition = group_partition(v35)
    assert {key: len(partition["TNG100"][key]) for key in ("fit", "holdout", "embargo")} == {
        "fit": 361,
        "holdout": 44,
        "embargo": 27,
    }
    assert partition["TNG100"]["minimum_holdout_fit_center_distance_mpc_h"] >= 10.0
    assert partition["SIMBA"]["holdout_groups"] == [20, 21]
    assert partition["Swift"]["holdout_groups"] == [6, 8, 9]
    assert len(partition["SIMBA"]["fit"]) == 146
    assert len(partition["SIMBA"]["holdout"]) == 56
    assert len(partition["Swift"]["fit"]) == 348
    assert len(partition["Swift"]["holdout"]) == 61
    for row in partition.values():
        assert not (set(row["fit"]) & set(row["holdout"]))


def test_group_partition_digest_is_deterministic() -> None:
    v35 = json.loads(
        (REPO / "config/hong2021_v35_residual_spectrum_phase_program.json").read_text()
    )
    assert partition_digest(group_partition(v35)) == partition_digest(group_partition(v35))
