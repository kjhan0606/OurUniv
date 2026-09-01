#!/usr/bin/env python3
"""Diagnose N32 2M++ raw-selection quadrature convergence after Phase-A gate failure.

The diagnostic rebuilds the full six-population raw exposure with four- and
six-point Gauss-Legendre quadrature per voxel axis.  It never alters exposure
based on observed occupancy, performs no inference, and publishes no datum.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROGRAM_SCHEMA = "ouruniv-cf4-twompp-selection-quadrature-diagnostic-program-v1"
RESULT_SCHEMA = "ouruniv-cf4-twompp-selection-quadrature-diagnostic-result-v1"
MANIFEST_SCHEMA = "ouruniv-cf4-twompp-selection-quadrature-diagnostic-manifest-v1"
COMPLETE_SCHEMA = "ouruniv-cf4-twompp-selection-quadrature-diagnostic-complete-v1"
STATUS = "COMPLETE_SELECTION_QUADRATURE_DIAGNOSTIC_NO_DATUM_PUBLICATION"
EXPECTED_FILES = {"selection_orders.npz", "result.json", "manifest.json", "COMPLETE"}


class DiagnosticError(ValueError):
    """Fail-closed selection-quadrature diagnostic error."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding["path"]))
    if not path.is_file():
        raise DiagnosticError(f"bound {label} is absent: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise DiagnosticError(f"bound {label} size changed")
    if sha256_file(path) != str(binding["sha256"]):
        raise DiagnosticError(f"bound {label} hash changed")
    return path


def load_program(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    program = json.loads(raw)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise DiagnosticError("unexpected selection diagnostic program schema")
    authorization = program.get("authorization", {})
    if not authorization.get("quadrature_diagnostic", False):
        raise DiagnosticError("selection quadrature diagnostic is not authorized")
    for forbidden in (
        "datum_publication",
        "field_inference",
        "mock_seed_access",
        "Phase_B_or_later",
        "automatic_follow_on",
    ):
        if authorization.get(forbidden, True):
            raise DiagnosticError(f"program improperly authorizes {forbidden}")
    for label, binding in program["bindings"].items():
        _verify_binding(binding, label)
    return program, hashlib.sha256(raw).hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"cannot load bound module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_raw_selection_order(
    base: Any,
    joint: Any,
    program: Mapping[str, Any],
    tracer_program: Mapping[str, Any],
    information_program: Mapping[str, Any],
    order: int,
) -> np.ndarray:
    """Integrate raw exposure with a global tensor Gauss-Legendre order."""

    import healpy as hp
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    if order < 2:
        raise DiagnosticError("quadrature order must be at least two")
    design = information_program["design"]
    tracer_design = tracer_program["tracer_design"]
    cosmology = tracer_program["cosmology"]
    grid = int(design["grid_N"])
    box = float(design["box_size_cMpc_h"])
    spacing = box / grid
    nodes, weights = np.polynomial.legendre.leggauss(order)
    offsets = 0.5 * spacing * nodes
    axis = (np.arange(grid, dtype=np.float64) + 0.5) * spacing - box / 2.0
    nside = int(design["HEALPix_NSIDE"])
    completeness11 = base.load_completeness_map(
        program["bindings"]["completeness_11_5"]["path"], nside
    )
    completeness12 = base.load_completeness_map(
        program["bindings"]["completeness_12_5"]["path"], nside
    )
    exposure = np.zeros((6, grid, grid, grid), dtype=np.float64)
    absolute_edges = np.asarray(design["absolute_K_edges"], dtype=np.float64)
    lf = design["Schechter"]
    radial_min = float(design["radial_min_cMpc_h"])
    radial_max = float(design["radial_max_cMpc_h"])
    for ix, ox in enumerate(offsets):
        for iy, oy in enumerate(offsets):
            for iz, oz in enumerate(offsets):
                coefficient = float(weights[ix] * weights[iy] * weights[iz] / 8.0)
                x, y, z = np.meshgrid(axis + ox, axis + oy, axis + oz, indexing="ij")
                radius = np.sqrt(x * x + y * y + z * z)
                active = (radius >= radial_min) & (radius <= radial_max)
                if not np.any(active):
                    continue
                lon = np.mod(np.arctan2(y[active], x[active]), 2.0 * np.pi)
                lat = np.arcsin(z[active] / radius[active])
                sg = SkyCoord(sgl=lon * u.rad, sgb=lat * u.rad, frame="supergalactic")
                pixels = hp.ang2pix(
                    nside,
                    0.5 * np.pi - sg.icrs.dec.rad,
                    np.mod(sg.icrs.ra.rad, 2.0 * np.pi),
                    nest=False,
                )
                luminosity_distance = joint._cosmology_distance_table(
                    radius[active], cosmology
                )
                active_flat = np.flatnonzero(active)
                for apparent in (0, 1):
                    angular = (completeness11 if apparent == 0 else completeness12)[pixels]
                    apparent_bright = (
                        None
                        if apparent == 0
                        else float(tracer_design["bright_apparent_K_max"])
                    )
                    apparent_faint = float(
                        tracer_design[
                            "bright_apparent_K_max"
                            if apparent == 0
                            else "faint_apparent_K_max"
                        ]
                    )
                    for absolute in range(3):
                        radial = joint.schechter_fraction(
                            luminosity_distance,
                            apparent_bright,
                            apparent_faint,
                            float(absolute_edges[absolute]),
                            float(absolute_edges[absolute + 1]),
                            float(lf["Mstar"]),
                            float(lf["alpha"]),
                        )
                        flat = exposure[3 * apparent + absolute].ravel()
                        flat[active_flat] += coefficient * angular * radial
    return exposure


def exposure_summary(counts: np.ndarray, exposure: np.ndarray) -> dict[str, Any]:
    if counts.shape != exposure.shape or counts.ndim != 4:
        raise DiagnosticError("count and exposure arrays are incompatible")
    population_count = counts.shape[0]
    occupied = counts > 0
    failed = occupied & (exposure <= 0.0)
    return {
        "positive_count_nonpositive_exposure_population_voxel_count": int(
            np.count_nonzero(failed)
        ),
        "galaxy_count_in_nonpositive_exposure_population_voxels": int(
            np.sum(counts[failed])
        ),
        "failed_population_voxel_count_by_population": np.count_nonzero(
            failed, axis=(1, 2, 3)
        ).tolist(),
        "failed_galaxy_count_by_population": [
            int(np.sum(counts[index][failed[index]]))
            for index in range(population_count)
        ],
        "support_sum": exposure.reshape(population_count, -1).sum(axis=1).tolist(),
        "positive_voxel_fraction": np.mean(
            exposure.reshape(population_count, -1) > 0.0, axis=1
        ).tolist(),
        "minimum": float(np.min(exposure)),
        "maximum": float(np.max(exposure)),
        "finite_unit_interval": bool(
            np.all(np.isfinite(exposure))
            and np.all(exposure >= 0.0)
            and np.all(exposure <= 1.0 + 2.0e-14)
        ),
    }


def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise DiagnosticError("selection comparison arrays are incompatible")
    population_count = reference.shape[0]
    delta = candidate - reference
    denominator = np.sum(np.abs(candidate), axis=(1, 2, 3))
    l1 = np.sum(np.abs(delta), axis=(1, 2, 3))
    support_reference = reference.reshape(population_count, -1).sum(axis=1)
    support_candidate = candidate.reshape(population_count, -1).sum(axis=1)
    return {
        "candidate_minus_reference_maximum_absolute": float(np.max(np.abs(delta))),
        "relative_L1_by_population": (l1 / denominator).tolist(),
        "relative_support_change_by_population": (
            (support_candidate - support_reference) / support_candidate
        ).tolist(),
        "positive_support_disagreement_fraction_by_population": np.mean(
            (reference > 0.0) != (candidate > 0.0), axis=(1, 2, 3)
        ).tolist(),
    }


def collect_diagnostic(
    program: Mapping[str, Any], program_sha256: str, implementation_commit: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise DiagnosticError("implementation commit must be lowercase 40-hex")
    base = _load_module(
        Path(program["bindings"]["tracer_implementation"]["path"]),
        "_cf4_selection_quadrature_tracer_v1",
    )
    joint = _load_module(
        Path(program["bindings"]["selection_implementation"]["path"]),
        "_cf4_selection_quadrature_selection_v1",
    )
    tracer_program = json.loads(
        Path(program["bindings"]["tracer_program"]["path"]).read_bytes()
    )
    information_program = json.loads(
        Path(program["bindings"]["selection_program"]["path"]).read_bytes()
    )
    with np.load(program["bindings"]["failed_datum"]["path"], allow_pickle=False) as archive:
        counts = np.asarray(archive["counts_all"], dtype=np.int64)
        order2 = np.asarray(archive["raw_selection_exposure"], dtype=np.float64)
    if counts.shape != (6, 32, 32, 32) or order2.shape != counts.shape:
        raise DiagnosticError("failed Phase-A arrays have unexpected shape")
    order4 = build_raw_selection_order(
        base, joint, program, tracer_program, information_program, 4
    )
    order6 = build_raw_selection_order(
        base, joint, program, tracer_program, information_program, 6
    )
    if order4.shape != counts.shape or order6.shape != counts.shape:
        raise DiagnosticError("diagnostic selection shape changed")
    zero2 = (counts > 0) & (order2 <= 0.0)
    cells = []
    for population, x, y, z in np.argwhere(zero2):
        cells.append(
            {
                "population": int(population),
                "voxel_index": [int(x), int(y), int(z)],
                "flat_voxel": int(np.ravel_multi_index((x, y, z), (32, 32, 32))),
                "observed_count": int(counts[population, x, y, z]),
                "exposure_order2": float(order2[population, x, y, z]),
                "exposure_order4": float(order4[population, x, y, z]),
                "exposure_order6": float(order6[population, x, y, z]),
            }
        )
    result = {
        "schema": RESULT_SCHEMA,
        "status": STATUS,
        "program_sha256": program_sha256,
        "implementation_commit": implementation_commit,
        "quadrature": {
            "orders_evaluated": [2, 4, 6],
            "subpoints_per_voxel": {"2": 8, "4": 64, "6": 216},
            "global_uniform_evaluation": True,
            "observed_count_dependent_exposure_modification": False,
        },
        "summaries": {
            "order2_failed_input": exposure_summary(counts, order2),
            "order4": exposure_summary(counts, order4),
            "order6": exposure_summary(counts, order6),
        },
        "comparisons": {
            "order2_reference_to_order4_candidate": comparison(order2, order4),
            "order4_reference_to_order6_candidate": comparison(order4, order6),
        },
        "order2_failed_cells": cells,
        "field_inference_executed": False,
        "datum_published": False,
        "mock_seed_accessed": False,
        "automatic_follow_on_executed": False,
        "selection_order_promotion_allowed_by_this_diagnostic": False,
    }
    return result, {"raw_selection_order4": order4, "raw_selection_order6": order6}


def publish_diagnostic(
    program_path: str | Path, output: str | Path, implementation_commit: str
) -> dict[str, Any]:
    program, program_sha = load_program(program_path)
    target = Path(output)
    stage = target.with_name(f".{target.name}.staging")
    if target.exists() or stage.exists():
        raise DiagnosticError("diagnostic output or staging path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    result, arrays = collect_diagnostic(program, program_sha, implementation_commit)
    np.savez_compressed(stage / "selection_orders.npz", **arrays)
    (stage / "result.json").write_bytes(canonical_json_bytes(result))
    artifacts = ("selection_orders.npz", "result.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "program_sha256": program_sha,
        "files": {
            name: {
                "bytes": (stage / name).stat().st_size,
                "sha256": sha256_file(stage / name),
            }
            for name in artifacts
        },
    }
    (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    complete = {
        "schema": COMPLETE_SCHEMA,
        "status": STATUS,
        "program_sha256": program_sha,
        "manifest_sha256": sha256_file(stage / "manifest.json"),
        "result_sha256": sha256_file(stage / "result.json"),
        "automatic_follow_on_executed": False,
    }
    (stage / "COMPLETE").write_bytes(canonical_json_bytes(complete))
    if {path.name for path in stage.iterdir()} != EXPECTED_FILES:
        raise DiagnosticError("diagnostic file set is not exact")
    stage.rename(target)
    return result


def validate_diagnostic(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise DiagnosticError("published diagnostic file set is not exact")
    result = json.loads((root / "result.json").read_bytes())
    manifest = json.loads((root / "manifest.json").read_bytes())
    complete = json.loads((root / "COMPLETE").read_bytes())
    if result.get("status") != STATUS or result.get("datum_published", True):
        raise DiagnosticError("published diagnostic status changed")
    for name in ("selection_orders.npz", "result.json"):
        expected = {
            "bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        if manifest["files"].get(name) != expected:
            raise DiagnosticError(f"manifest does not bind {name}")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise DiagnosticError("COMPLETE does not bind manifest")
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise DiagnosticError("COMPLETE does not bind result")
    with np.load(root / "selection_orders.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"raw_selection_order4", "raw_selection_order6"}:
            raise DiagnosticError("diagnostic NPZ key set changed")
        for name in archive.files:
            values = np.asarray(archive[name])
            if values.shape != (6, 32, 32, 32) or values.dtype != np.float64:
                raise DiagnosticError("diagnostic exposure array contract changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-diagnostic")
    run.add_argument("--program", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = sub.add_parser("validate-diagnostic")
    validate.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run-diagnostic":
        result = publish_diagnostic(args.program, args.output, args.implementation_commit)
    else:
        result = validate_diagnostic(args.directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
