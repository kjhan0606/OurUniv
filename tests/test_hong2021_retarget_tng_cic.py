import json

import h5py
import numpy as np

from hong2021_retarget_tng_cic import retarget


def test_retarget_changes_only_density_target(tmp_path):
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs["density_scale"] = 4.5
        handle.create_dataset("input", data=np.arange(2 * 4**3, dtype=np.float32).reshape(1, 2, 4, 4, 4))
        handle.create_dataset("target", data=np.zeros((1, 1, 4, 4, 4), dtype=np.float32), chunks=(1, 1, 4, 4, 4))
        handle.create_dataset("cube_origin_cell", data=np.array([[-1, 2, 3]], dtype=np.int16))
        handle.create_dataset("center_subhalo_id", data=np.array([17]))

    grid_path = tmp_path / "grid.npy"
    coordinate = np.arange(240, dtype=np.float32)
    grid = 10.0 + coordinate[:, None, None] + 0.01 * coordinate[None, :, None]
    grid = np.broadcast_to(grid, (240, 240, 240)).copy()
    np.save(grid_path, grid)
    grid_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "hong2021-periodic-dm-particle-grid-v1",
                "complete": True,
                "assignment": "cic",
            }
        )
    )
    destination = tmp_path / "retargeted.h5"
    result = retarget(source, grid_path, destination)
    assert result["samples"] == 1
    with h5py.File(source, "r") as old, h5py.File(destination, "r") as new:
        np.testing.assert_array_equal(old["input"][:], new["input"][:])
        np.testing.assert_array_equal(old["cube_origin_cell"][:], new["cube_origin_cell"][:])
        np.testing.assert_array_equal(old["center_subhalo_id"][:], new["center_subhalo_id"][:])
        assert np.isfinite(new["target"][:]).all()
        assert not np.array_equal(old["target"][:], new["target"][:])
        assert not bool(new.attrs["input_and_split_changed"])
