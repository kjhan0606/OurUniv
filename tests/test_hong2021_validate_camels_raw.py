from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from hong2021_validate_camels_raw import snapshot_geometry


def _snapshot(path: Path, suite: str) -> None:
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        particles = handle.create_group("PartType1")
        coordinates = particles.create_dataset("Coordinates", shape=(1, 3), dtype="f8")
        if suite == "SIMBA":
            header.attrs["BoxSize"] = 25_000.0
            header.attrs["HubbleParam"] = 0.6711
            header.attrs["Omega0"] = 0.3
        else:
            header.attrs["BoxSize"] = np.full(3, 25.0 / 0.6711)
            cosmology = handle.create_group("Cosmology")
            cosmology.attrs["h"] = np.array([0.6711])
            cosmology.attrs["Omega_m"] = np.array([0.3])
            units = handle.create_group("Units")
            units.attrs["Unit length in cgs (U_L)"] = np.array([3.0856775814913673e24])
            coordinates.attrs["h-scale exponent"] = np.array([0.0])


@pytest.mark.parametrize(
    ("suite", "expected_scale", "expected_schema"),
    (("SIMBA", 0.001, "gadget_kpc_h"), ("Swift-EAGLE", 0.6711, "swift_mpc_times_h")),
)
def test_snapshot_geometry_maps_both_suites_to_common_box(
    tmp_path: Path, suite: str, expected_scale: float, expected_schema: str
) -> None:
    path = tmp_path / "snapshot.h5"
    _snapshot(path, suite)
    with h5py.File(path, "r") as handle:
        result = snapshot_geometry(handle, suite)
    assert result["coordinate_scale_to_mpc_h"] == expected_scale
    assert result["snapshot_coordinate_schema"] == expected_schema
    assert result["converted_box_mpc_h"] == pytest.approx(25.0, abs=1.0e-12)


def test_swift_geometry_rejects_gadget_scale_assumption(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.h5"
    _snapshot(path, "Swift-EAGLE")
    with h5py.File(path, "r+") as handle:
        handle["Cosmology"].attrs.modify("h", np.array([0.001]))
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="converts"):
        snapshot_geometry(handle, "Swift-EAGLE")
