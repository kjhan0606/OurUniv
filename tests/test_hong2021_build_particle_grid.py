from pathlib import Path

import h5py
import numpy as np
import pytest

from hong2021_build_particle_grid import build_grid, expand_sources


def write_snapshot(path: Path, coordinates: np.ndarray, total: int) -> None:
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 1000.0
        header.attrs["Redshift"] = 0.0
        header.attrs["MassTable"] = np.array([0, 2.5, 0, 0, 0, 0])
        number = np.zeros(6, dtype=np.uint32)
        number[1] = total
        header.attrs["NumPart_Total"] = number
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        particles = handle.create_group("PartType1")
        particles.create_dataset("Coordinates", data=coordinates)


def test_build_grid_from_snapshot_pieces(tmp_path):
    write_snapshot(
        tmp_path / "snap.0.hdf5",
        np.array([[0, 0, 0], [125, 125, 125]], dtype=np.float32),
        3,
    )
    write_snapshot(
        tmp_path / "snap.1.hdf5",
        np.array([[999, 999, 999]], dtype=np.float32),
        3,
    )
    destination = tmp_path / "grid.npy"
    result = build_grid(
        expand_sources(str(tmp_path / "snap.*.hdf5")),
        destination=destination,
        grid=4,
        box_mpc_h=1.0,
        coordinate_scale_to_mpc_h=0.001,
        assignment="cic",
        block_particles=2,
    )
    grid = np.load(destination)
    assert result["dm_particles"] == 3
    assert result["source_file_count"] == 2
    assert grid.dtype == np.float32
    assert grid.sum(dtype=np.float64) == pytest.approx(3.0, abs=1e-6)
    assert destination.with_suffix(".json").is_file()


def test_build_grid_refuses_overwrite(tmp_path):
    source = tmp_path / "snapshot.hdf5"
    write_snapshot(source, np.array([[0, 0, 0]], dtype=np.float32), 1)
    destination = tmp_path / "grid.npy"
    destination.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="overwrite"):
        build_grid(
            [source],
            destination=destination,
            grid=2,
            box_mpc_h=1,
            coordinate_scale_to_mpc_h=0.001,
            assignment="ngp",
            block_particles=1,
        )
