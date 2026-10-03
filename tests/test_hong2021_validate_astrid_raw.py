from pathlib import Path

import h5py
import numpy as np

import hong2021_validate_astrid_raw as validator


def write_snapshot(path: Path, particles: int) -> None:
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        totals = np.zeros(6, dtype=np.uint32)
        totals[1] = particles
        header.attrs["NumPart_Total"] = totals
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["BoxSize"] = 25_000.0
        header.attrs["Redshift"] = 0.0
        header.attrs["HubbleParam"] = 0.6711
        header.attrs["Omega0"] = 0.3
        dm = handle.create_group("PartType1")
        dm.create_dataset(
            "Coordinates",
            data=np.linspace(0.0, 24_999.0, particles * 3).reshape(particles, 3),
        )


def write_catalog(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 25_000.0
        header.attrs["Redshift"] = 0.0
        subhalo = handle.create_group("Subhalo")
        subhalo.create_dataset(
            "SubhaloPos", data=np.asarray([[1000, 2000, 3000], [4000, 5000, 6000]])
        )
        subhalo.create_dataset("SubhaloVel", data=np.zeros((2, 3)))
        mass = np.zeros((2, 6))
        mass[:, 4] = [5.0, 8.0]
        subhalo.create_dataset("SubhaloMassType", data=mass)


def test_astrid_gadget_units_and_frozen_observer_are_validated(tmp_path, monkeypatch):
    root = tmp_path / "Astrid" / "L25n256"
    particles = 8
    monkeypatch.setattr(validator, "EXPECTED_REALIZATIONS", (0,))
    monkeypatch.setattr(validator, "EXPECTED_DM_PARTICLES", particles)
    write_snapshot(root / "raw/CV_0/snapshot_090.hdf5", particles)
    write_catalog(root / "CV/CV_0/groups_090.hdf5")
    report = validator.validate(root, (0,))
    assert report["coordinate_scale_to_mpc_h"] == 0.001
    assert report["box_mpc_h"] == 25.0
    assert report["rows"][0]["catalog"]["observer_candidates"] == 2
    assert report["rows"][0]["catalog"]["frozen_observer_subhalo_index"] == 0
