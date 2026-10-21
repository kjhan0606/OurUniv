#!/usr/bin/env python
"""Validate downloaded raw CAMELS development inputs and their length units.

Gadget-format SIMBA snapshots store coordinates in kpc/h, while public
Swift-EAGLE snapshots store coordinates in Mpc with an explicit cosmological
``h``.  This validator accepts only those two predeclared schemas, proves that
both map to the frozen 25 Mpc/h box, and records the exact coordinate scale
needed by the common CIC builder.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA = "hong2021-v14-camels-raw-development-download-v2"
ALLOWED_SUITES = ("SIMBA", "Swift-EAGLE")
EXPECTED_DM_PARTICLES = 256**3
EXPECTED_BOX_MPC_H = 25.0
MPC_CM = 3.0856775814913673e24


def scalar(value: Any, *, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one value, got {array.shape}")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def constant_vector(value: Any, *, name: str, size: int) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    if not np.all(array == array[0]):
        raise ValueError(f"{name} is not an isotropic box")
    return float(array[0])


def snapshot_geometry(handle: h5py.File, suite: str) -> dict[str, Any]:
    header = handle["Header"].attrs
    if suite == "SIMBA":
        raw_box = scalar(header["BoxSize"], name="Header/BoxSize")
        hubble = scalar(header["HubbleParam"], name="Header/HubbleParam")
        omega_m = scalar(header["Omega0"], name="Header/Omega0")
        scale = 0.001
        schema = "gadget_kpc_h"
        if not np.isclose(raw_box, 25_000.0, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"unexpected SIMBA raw box size {raw_box}")
    elif suite == "Swift-EAGLE":
        raw_box = constant_vector(
            header["BoxSize"], name="Header/BoxSize", size=3
        )
        cosmology = handle["Cosmology"].attrs
        hubble = scalar(cosmology["h"], name="Cosmology/h")
        omega_m = scalar(cosmology["Omega_m"], name="Cosmology/Omega_m")
        coordinate = handle["PartType1/Coordinates"]
        units = handle["Units"].attrs
        unit_length_cm = scalar(
            units["Unit length in cgs (U_L)"], name="Units/U_L"
        )
        h_exponent = scalar(
            coordinate.attrs["h-scale exponent"],
            name="PartType1/Coordinates h-scale exponent",
        )
        if not np.isclose(unit_length_cm, MPC_CM, rtol=1.0e-8, atol=0.0):
            raise ValueError("Swift coordinate unit is not one Mpc")
        if h_exponent != 0.0:
            raise ValueError("Swift coordinates unexpectedly include an h factor")
        scale = hubble
        schema = "swift_mpc_times_h"
    else:
        raise ValueError(f"unsupported CAMELS development suite: {suite}")
    converted_box = raw_box * scale
    if not np.isclose(
        converted_box, EXPECTED_BOX_MPC_H, rtol=0.0, atol=1.0e-7
    ):
        raise ValueError(
            f"snapshot box converts to {converted_box}, not {EXPECTED_BOX_MPC_H} Mpc/h"
        )
    return {
        "snapshot_coordinate_schema": schema,
        "raw_box_size": raw_box,
        "coordinate_scale_to_mpc_h": scale,
        "converted_box_mpc_h": converted_box,
        "hubble_param": hubble,
        "omega_m": omega_m,
    }


def validate_suite(
    suite: str,
    root: str | Path,
    first: int,
    last: int,
) -> dict[str, Any]:
    if suite not in ALLOWED_SUITES:
        raise ValueError(f"unsupported CAMELS development suite: {suite}")
    if first < 0 or last < first:
        raise ValueError("invalid realization interval")
    root = Path(root)
    rows = []
    scales = []
    for realization in range(first, last + 1):
        snapshot = root / "raw" / f"CV_{realization}" / "snapshot_090.hdf5"
        catalog = root / "CV" / f"CV_{realization}" / "groups_090.hdf5"
        with h5py.File(snapshot, "r") as handle:
            header = handle["Header"].attrs
            coordinates = handle["PartType1/Coordinates"]
            shape = tuple(coordinates.shape)
            redshift = scalar(header["Redshift"], name="Header/Redshift")
            if abs(redshift) > 1.0e-6:
                raise ValueError(f"snapshot is not z=0: {snapshot}")
            if shape != (EXPECTED_DM_PARTICLES, 3):
                raise ValueError(f"unexpected DM coordinate shape {shape}: {snapshot}")
            geometry = snapshot_geometry(handle, suite)
            scales.append(geometry["coordinate_scale_to_mpc_h"])
            snapshot_row = {
                "path": str(snapshot.resolve()),
                "bytes": snapshot.stat().st_size,
                "dm_coordinate_shape": list(shape),
                **geometry,
            }
        with h5py.File(catalog, "r") as handle:
            header = handle["Header"].attrs
            required = (
                "Subhalo/SubhaloPos",
                "Subhalo/SubhaloVel",
                "Subhalo/SubhaloMassType",
            )
            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError(f"catalog missing {missing}: {catalog}")
            catalog_box = scalar(header["BoxSize"], name="catalog Header/BoxSize")
            catalog_redshift = scalar(
                header["Redshift"], name="catalog Header/Redshift"
            )
            if not np.isclose(catalog_box, 25_000.0, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"unexpected catalog BoxSize: {catalog}")
            if abs(catalog_redshift) > 1.0e-6:
                raise ValueError(f"catalog is not z=0: {catalog}")
            subhalos = int(handle["Subhalo/SubhaloPos"].shape[0])
            if handle["Subhalo/SubhaloVel"].shape != (subhalos, 3):
                raise ValueError(f"invalid SubhaloVel shape: {catalog}")
            if handle["Subhalo/SubhaloMassType"].shape[0] != subhalos:
                raise ValueError(f"invalid SubhaloMassType shape: {catalog}")
        rows.append(
            {
                "realization": realization,
                "snapshot": snapshot_row,
                "catalog": {
                    "path": str(catalog.resolve()),
                    "bytes": catalog.stat().st_size,
                    "subhalos": subhalos,
                    "coordinate_schema": "gadget_kpc_h",
                    "coordinate_scale_to_mpc_h": 0.001,
                },
            }
        )
    if not np.allclose(scales, scales[0], rtol=0.0, atol=0.0):
        raise ValueError("snapshot coordinate scale changes between realizations")
    return {
        "schema": SCHEMA,
        "suite": suite,
        "role": "development_only",
        "realizations": list(range(first, last + 1)),
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "common_cic_box_mpc_h": EXPECTED_BOX_MPC_H,
        "snapshot_coordinate_scale_to_mpc_h": scales[0],
        "catalog_coordinate_scale_to_mpc_h": 0.001,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=ALLOWED_SUITES, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = validate_suite(args.suite, args.root, args.first, args.last)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, destination)
    print(f"Validated {args.suite} raw V14 development data: {destination}")


if __name__ == "__main__":
    main()
