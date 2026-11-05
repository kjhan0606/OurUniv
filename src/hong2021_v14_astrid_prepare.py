#!/usr/bin/env python
"""Prepare sealed Astrid input and V14 inference caches for the one-shot gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py

from hong2021_prepare_camels_raw import SCHEMA as INPUT_SCHEMA, prepare
from hong2021_v14_baseline_audit import (
    SCHEMA as BASELINE_SCHEMA,
    prepare_astrid_independent,
)
from hong2021_astrid_seal import verify_astrid_seal
from hong2021_v14_freeze import sha256
from hong2021_v14_residual_cache import (
    CORRECTED_SCHEMA,
    STANDARDIZED_SCHEMA,
    prepare_astrid_independent_corrected,
    prepare_standardized,
)
from hong2021_validate_astrid_raw import SCHEMA as RAW_MANIFEST_SCHEMA


PREPARATION_SCHEMA = "hong2021-v14-astrid-independent-preparation-v1"
REALIZATIONS = tuple(range(27))


def _complete_h5(path: Path, schema: str) -> bool:
    if not path.is_file() or not path.with_suffix(".json").is_file():
        return False
    with h5py.File(path, "r") as handle:
        return str(handle.attrs.get("schema")) == schema and bool(
            handle.attrs.get("complete", False)
        )


def _validate_raw_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != RAW_MANIFEST_SCHEMA:
        raise ValueError("not the sealed Astrid raw manifest")
    if value.get("realizations") != list(REALIZATIONS):
        raise ValueError("Astrid raw manifest does not contain exactly CV0-26")
    if value.get("coordinate_scale_to_mpc_h") != 0.001:
        raise ValueError("Astrid raw manifest coordinate conversion changed")
    return value


def prepare_input(
    *, seal: Path, repo: Path, root: Path, raw_manifest: Path, output: Path
) -> dict[str, Any]:
    verify_astrid_seal(seal, repo=repo, require_committed=True)
    _validate_raw_manifest(raw_manifest)
    report_path = output.with_suffix(".json")
    if _complete_h5(output, INPUT_SCHEMA):
        return json.loads(report_path.read_text())
    if output.exists() or report_path.exists():
        raise RuntimeError("incomplete Astrid prepared input exists; refusing ambiguity")
    args = SimpleNamespace(
        root=str(root),
        out=str(output),
        realizations=",".join(map(str, REALIZATIONS)),
        observers="single",
        role="one_time_independent_gate",
        suite="Astrid",
        grid_pattern=str(root / "derived/hong2021_v14/dm_cic_{realization}.npy"),
    )
    report = prepare(args)
    if report["samples"] != 27 or report["realizations"] != list(REALIZATIONS):
        raise RuntimeError("Astrid preparation did not produce one object per realization")
    return report


def prepare_model_cache(
    *, seal: Path, repo: Path, data: Path, output_root: Path, device: str
) -> dict[str, Any]:
    record = verify_astrid_seal(seal, repo=repo, require_committed=True)
    artifacts = {name: Path(row["path"]) for name, row in record["artifacts"].items()}
    with h5py.File(data, "r") as handle:
        if str(handle.attrs.get("schema")) != INPUT_SCHEMA or not bool(
            handle.attrs.get("complete", False)
        ):
            raise ValueError("not a complete sealed Astrid input")
        if tuple(handle["realization"][:]) != REALIZATIONS:
            raise ValueError("Astrid input order must be exactly CV0-26")
        if len(handle["input"]) != 27:
            raise ValueError("Astrid input must contain exactly 27 objects")
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = output_root / "astrid_baseline.h5"
    corrected = output_root / "astrid_corrected.h5"
    standardized = output_root / "astrid_standardized.h5"
    if not _complete_h5(baseline, BASELINE_SCHEMA):
        prepare_astrid_independent(
            SimpleNamespace(
                domain="CAMELS-Astrid", data=str(data),
                checkpoint=str(artifacts["deterministic_v4"]), out=str(baseline),
                batch=6, workers=1, device=device,
            )
        )
    if not _complete_h5(corrected, CORRECTED_SCHEMA):
        prepare_astrid_independent_corrected(
            SimpleNamespace(
                domain="CAMELS-Astrid", data=str(data), baseline_cache=str(baseline),
                correction_checkpoint=str(artifacts["mean_correction"]),
                out=str(corrected), batch=6, workers=1, device=device,
            )
        )
    if not _complete_h5(standardized, STANDARDIZED_SCHEMA):
        prepare_standardized(
            SimpleNamespace(
                corrected_cache=str(corrected),
                location_scale_model=str(artifacts["location_scale"]),
                out=str(standardized), chunk=4,
            )
        )
    report = {
        "schema": PREPARATION_SCHEMA,
        "seal": str(seal.resolve()),
        "source_data": str(data.resolve()),
        "objects": 27,
        "target_used_for_model_choice": False,
        "simulation_identity_feature": False,
        "posthoc_transfer": False,
        "caches": {
            label: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for label, path in (
                ("baseline", baseline), ("corrected", corrected),
                ("standardized", standardized),
            )
        },
        "complete": True,
    }
    destination = output_root / "preparation.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing != report:
            raise RuntimeError("existing Astrid preparation record differs")
    else:
        temporary = destination.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/kjhan/BACKUP/CF4"))
    sub = parser.add_subparsers(dest="mode", required=True)
    input_parser = sub.add_parser("input")
    input_parser.add_argument("--root", type=Path, required=True)
    input_parser.add_argument("--raw-manifest", type=Path, required=True)
    input_parser.add_argument("--out", type=Path, required=True)
    cache_parser = sub.add_parser("model-cache")
    cache_parser.add_argument("--data", type=Path, required=True)
    cache_parser.add_argument("--out", type=Path, required=True)
    cache_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mode == "input":
        report = prepare_input(
            seal=args.seal, repo=args.repo, root=args.root,
            raw_manifest=args.raw_manifest, output=args.out,
        )
    else:
        report = prepare_model_cache(
            seal=args.seal, repo=args.repo, data=args.data,
            output_root=args.out, device=args.device,
        )
    print(json.dumps({"complete": True, "schema": report["schema"]}, indent=2))


if __name__ == "__main__":
    main()
