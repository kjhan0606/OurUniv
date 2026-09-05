from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_seal_eagle_gate import build_seal


def test_build_seal_partitions_objects_and_hashes_inputs(tmp_path: Path) -> None:
    data = tmp_path / "data.h5"
    with h5py.File(data, "w") as handle:
        handle.attrs["schema"] = "test-independent"
        handle.attrs["independent_test_only"] = True
        handle.create_dataset("center_galaxy_id", data=[100, 101, 102, 103])
        handle.create_dataset(
            "center_position_mpc_h", data=np.arange(12).reshape(4, 3)
        )
    representative = tmp_path / "representative.json"
    representative.write_text(
        json.dumps({"indices": [2, 0], "galaxy_ids": [102, 100]})
    )
    other = []
    for name in ("ensemble.h5", "metrics.json", "decision.json", "model.pt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        other.append(path)

    seal = build_seal(data, representative, *other)

    assert seal["counts"] == {
        "all": 4,
        "used_v6_test": 2,
        "reserved_confirmation": 2,
    }
    assert seal["used_v6_test"]["indices"] == [2, 0]
    assert seal["reserved_confirmation"]["indices"] == [1, 3]
    assert len(seal["files"]["prepared_data"]["sha256"]) == 64


def test_build_seal_rejects_mismatched_ids(tmp_path: Path) -> None:
    data = tmp_path / "data.h5"
    with h5py.File(data, "w") as handle:
        handle.attrs["schema"] = "test-independent"
        handle.attrs["independent_test_only"] = True
        handle.create_dataset("center_galaxy_id", data=[10])
        handle.create_dataset("center_position_mpc_h", data=[[0.0, 0.0, 0.0]])
    representative = tmp_path / "representative.json"
    representative.write_text(json.dumps({"indices": [0], "galaxy_ids": [11]}))
    files = []
    for index in range(4):
        path = tmp_path / f"file{index}"
        path.write_bytes(b"x")
        files.append(path)

    try:
        build_seal(data, representative, *files)
    except ValueError as error:
        assert "GalaxyIDs" in str(error)
    else:
        raise AssertionError("mismatched IDs should fail")
