#!/usr/bin/env python
"""Validate the sealed CAMELS-Astrid CV0-26 raw one-shot inputs.

This module is intentionally separate from the development-data validator.
It may be invoked only by the committed one-shot runner after seal
verification.  Public CAMELS Gadget files store snapshot and catalogue
positions in kpc/h, hence the frozen conversion factor is exactly 0.001.
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

from hong2021_prepare_simba import (
    GALAXY_MASS_THRESHOLD,
    OBSERVER_MASS,
    choose_observer,
)
from hong2021_v14_freeze import verify_seal


SCHEMA = "hong2021-v14-astrid-raw-independent-download-v1"
EXPECTED_REALIZATIONS = tuple(range(27))
EXPECTED_DM_PARTICLES = 256**3
RAW_BOX_KPC_H = 25_000.0
COORDINATE_SCALE_TO_MPC_H = 0.001


def scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one scalar")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _header_total(header: h5py.AttributeManager) -> int:
    low = np.asarray(header["NumPart_Total"], dtype=np.uint64)
    high = np.asarray(
        header.get("NumPart_Total_HighWord", np.zeros_like(low)),
        dtype=np.uint64,
    )
    return int(low[1] + (high[1] << np.uint64(32)))


def _coordinate_range(dataset: h5py.Dataset) -> tuple[float, float]:
    minimum = np.inf
    maximum = -np.inf
    for begin in range(0, len(dataset), 2_000_000):
        value = np.asarray(dataset[begin : begin + 2_000_000], dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError("coordinate dataset contains a non-finite value")
        minimum = min(minimum, float(value.min()))
        maximum = max(maximum, float(value.max()))
    if minimum < -1.0e-6 or maximum > RAW_BOX_KPC_H + 1.0e-6:
        raise ValueError(f"coordinate range {minimum}..{maximum} leaves the box")
    return minimum, maximum


def validate(root: Path, realizations: tuple[int, ...]) -> dict[str, Any]:
    if realizations != EXPECTED_REALIZATIONS:
        raise ValueError("Astrid one-shot validation requires exactly CV0-26")
    rows = []
    for realization in realizations:
        snapshot = root / "raw" / f"CV_{realization}" / "snapshot_090.hdf5"
        catalog = root / "CV" / f"CV_{realization}" / "groups_090.hdf5"
        with h5py.File(snapshot, "r") as handle:
            header = handle["Header"].attrs
            box = scalar(header["BoxSize"], "snapshot Header/BoxSize")
            redshift = scalar(header["Redshift"], "snapshot Header/Redshift")
            hubble = scalar(header["HubbleParam"], "snapshot Header/HubbleParam")
            omega_m = scalar(header["Omega0"], "snapshot Header/Omega0")
            coordinates = handle["PartType1/Coordinates"]
            if coordinates.shape != (EXPECTED_DM_PARTICLES, 3):
                raise ValueError(f"unexpected Astrid DM shape: {snapshot}")
            if _header_total(header) != EXPECTED_DM_PARTICLES:
                raise ValueError(f"unexpected Astrid header DM total: {snapshot}")
            if not np.isclose(box, RAW_BOX_KPC_H, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"unexpected Astrid snapshot box: {snapshot}")
            if abs(redshift) > 1.0e-6:
                raise ValueError(f"Astrid snapshot is not z=0: {snapshot}")
            if not np.isclose(omega_m, 0.3, rtol=0.0, atol=1.0e-6):
                raise ValueError(f"Astrid CV Omega_m differs from frozen 0.3: {snapshot}")
            coordinate_min, coordinate_max = _coordinate_range(coordinates)
        with h5py.File(catalog, "r") as handle:
            header = handle["Header"].attrs
            catalog_box = scalar(header["BoxSize"], "catalog Header/BoxSize")
            catalog_redshift = scalar(header["Redshift"], "catalog Header/Redshift")
            required = (
                "Subhalo/SubhaloPos",
                "Subhalo/SubhaloVel",
                "Subhalo/SubhaloMassType",
            )
            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError(f"Astrid catalogue missing {missing}: {catalog}")
            position = np.asarray(handle["Subhalo/SubhaloPos"], dtype=np.float64)
            velocity = np.asarray(handle["Subhalo/SubhaloVel"], dtype=np.float64)
            mass_type = np.asarray(handle["Subhalo/SubhaloMassType"], dtype=np.float64)
            subhalos = len(position)
            if position.shape != (subhalos, 3) or velocity.shape != (subhalos, 3):
                raise ValueError(f"invalid Astrid subhalo vector shape: {catalog}")
            if mass_type.shape != (subhalos, 6):
                raise ValueError(f"invalid Astrid SubhaloMassType shape: {catalog}")
            if (
                not np.isfinite(position).all()
                or not np.isfinite(velocity).all()
                or not np.isfinite(mass_type).all()
                or np.any(mass_type < 0)
            ):
                raise ValueError(f"invalid Astrid catalogue values: {catalog}")
            if np.any(position < -1.0e-6) or np.any(position > RAW_BOX_KPC_H + 1.0e-6):
                raise ValueError(f"Astrid subhalo position leaves the box: {catalog}")
            if not np.isclose(catalog_box, RAW_BOX_KPC_H, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"unexpected Astrid catalogue box: {catalog}")
            if abs(catalog_redshift) > 1.0e-6:
                raise ValueError(f"Astrid catalogue is not z=0: {catalog}")
            stellar_mass = mass_type[:, 4] * 1.0e10
            observer = choose_observer(stellar_mass)
            observer_candidates = int(
                np.count_nonzero(
                    (stellar_mass > OBSERVER_MASS[0])
                    & (stellar_mass < OBSERVER_MASS[1])
                )
            )
            galaxies = int(np.count_nonzero(stellar_mass >= GALAXY_MASS_THRESHOLD))
        rows.append(
            {
                "realization": realization,
                "snapshot": {
                    "path": str(snapshot.resolve()),
                    "bytes": snapshot.stat().st_size,
                    "dm_particles": EXPECTED_DM_PARTICLES,
                    "coordinate_min_kpc_h": coordinate_min,
                    "coordinate_max_kpc_h": coordinate_max,
                    "hubble_param": hubble,
                    "omega_m": omega_m,
                },
                "catalog": {
                    "path": str(catalog.resolve()),
                    "bytes": catalog.stat().st_size,
                    "subhalos": subhalos,
                    "observer_candidates": observer_candidates,
                    "frozen_observer_subhalo_index": observer,
                    "frozen_observer_stellar_mass_m_sun": float(stellar_mass[observer]),
                    "galaxy_proxy_count": galaxies,
                },
            }
        )
        print(f"[Astrid raw validation] CV_{realization} complete", flush=True)
    return {
        "schema": SCHEMA,
        "suite": "CAMELS-Astrid",
        "role": "one_time_independent_gate",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "realizations": list(realizations),
        "snapshot_coordinate_schema": "gadget_kpc_h",
        "catalog_coordinate_schema": "gadget_kpc_h",
        "coordinate_scale_to_mpc_h": COORDINATE_SCALE_TO_MPC_H,
        "box_mpc_h": RAW_BOX_KPC_H * COORDINATE_SCALE_TO_MPC_H,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/kjhan/BACKUP/CF4"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify_seal(args.seal, repo=args.repo, require_committed=True)
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite Astrid manifest: {args.out}")
    report = validate(args.root, EXPECTED_REALIZATIONS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"validated": True, "manifest": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
