#!/usr/bin/env python
"""Audit whether a scientifically faithful Hong et al. (2021) run can start."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def csv_columns(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def any_file(root: Path, patterns: tuple[str, ...]) -> bool:
    if not root.is_dir():
        return False
    return any(any(root.rglob(pattern)) for pattern in patterns)


def build_report() -> dict[str, Any]:
    tng100_local = Path("/scratch/kjhan/IllustrisTNG/TNG100-1")
    tng300_local = Path(
        "/scratch/jhshin/02_illustris/08_illustrisTNG/TNG300-1"
    )
    cf4_columns = csv_columns(REPO / "data/cf4_galaxies.csv")
    raw_tng100_snapshot = any_file(
        tng100_local, ("snap_099.*.hdf5", "snapshot_099.*.hdf5")
    )
    raw_tng100_groupcat = any_file(
        tng100_local, ("fof_subhalo_tab_099.*.hdf5",)
    )
    raw_tng300_snapshot = any_file(
        tng300_local, ("snap_099.*.hdf5", "snapshot_099.*.hdf5")
    )
    processed_tng300_catalog = (
        tng300_local / "sav/galaxy_099.sav"
    ).is_file()
    required_cf4 = {
        "distance_modulus": "DM" in cf4_columns,
        "distance_modulus_error": "e_DM" in cf4_columns,
        "galactic_coordinates": {"GLON", "GLAT"}.issubset(cf4_columns),
        "B_band_absolute_magnitude": any(
            name.lower() in {"mb", "m_b", "bmag", "b_mag"} for name in cf4_columns
        ),
        "V_GSR": any(name.lower() in {"vgsr", "v_gsr"} for name in cf4_columns),
        "V_CMB_only": "Vcmb" in cf4_columns,
    }
    training_blockers = []
    if not raw_tng100_snapshot:
        training_blockers.append(
            "TNG100-1 snapshot 99 dark-matter coordinates are absent"
        )
    if not raw_tng100_groupcat:
        training_blockers.append("TNG100-1 group catalog 99 is absent")
    cf4_application_blockers = []
    if not required_cf4["B_band_absolute_magnitude"]:
        cf4_application_blockers.append(
            "CF4 table lacks the LEDA B-band magnitude used by the paper"
        )
    if not required_cf4["V_GSR"]:
        cf4_application_blockers.append(
            "CF4 table lacks V_GSR; it only supplies V_CMB"
        )
    report = {
        "paper_target": {
            "simulation": "TNG100-1 snapshot 99",
            "grid": 64,
            "box_mpc_h": 20.0,
            "voxel_mpc_h": 0.3125,
            "train_unaugmented": 432,
            "validation_unaugmented": 93,
        },
        "compute": {
            "pytorch_model_smoke_passed_separately": True,
            "paper_batch6_forward_backward_adam_passed": True,
            "paper_parameter_count": 461_024_955,
            "a10_peak_allocated_gib": 8.62,
            "a10_peak_reserved_gib": 12.22,
            "paper_reported_runtime": "73 hours on one 16-GB NVIDIA V100",
        },
        "local_simulation_data": {
            "tng100_root": str(tng100_local),
            "tng100_snapshot_099": raw_tng100_snapshot,
            "tng100_groupcat_099": raw_tng100_groupcat,
            "tng300_root": str(tng300_local),
            "tng300_snapshot_099": raw_tng300_snapshot,
            "tng300_processed_galaxy_catalog": processed_tng300_catalog,
            "note": (
                "The local TNG300 tree/offset/SAV products contain galaxy "
                "post-processing but not the particle coordinates required "
                "for the dark-matter target."
            ),
        },
        "credentials": {
            "TNG_API_KEY_environment_variable_present": bool(
                os.environ.get("TNG_API_KEY")
            ),
            "ILLUSTRIS_API_KEY_environment_variable_present": bool(
                os.environ.get("ILLUSTRIS_API_KEY")
            ),
            "values_recorded": False,
        },
        "cf4": {
            "path": str(REPO / "data/cf4_galaxies.csv"),
            "columns": cf4_columns,
            "paper_required_fields": required_cf4,
            "rows": 55_877,
        },
        "faithful_training_can_start": not training_blockers,
        "training_blockers": training_blockers,
        "faithful_cf4_application_can_start": not cf4_application_blockers,
        "cf4_application_blockers": cf4_application_blockers,
        "non_blocking_work_completed": [
            "paper architecture transcription",
            "paper augmentation and optimizer transcription",
            "uncertainty-aware velocity gridding",
            "training-file validator",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="recon/hong2021/audit_v1/report.json"
    )
    args = parser.parse_args()
    report = build_report()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
