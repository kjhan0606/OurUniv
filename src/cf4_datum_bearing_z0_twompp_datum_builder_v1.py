#!/usr/bin/env python3
"""Build the actual six-population N32 2M++ integer-count datum.

This Phase-A program performs no field inference and consumes no mock seed.  It
reproduces the frozen 36,635-row CF4-disjoint metadata-consistent subset, bins
each observed redshift-space position into exactly one NGP voxel, and builds
the raw angular-times-radial selection exposure without normalizing it to the
observed population totals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROGRAM_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-program-v1"
RESULT_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-result-v1"
MANIFEST_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-manifest-v1"
COMPLETE_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-complete-v1"
STATUS_PASS = "PASS_ACTUAL_36635_INTEGER_COUNT_DATUM_NO_FIELD_INFERENCE"
STATUS_FAIL = "FAIL_ACTUAL_COUNT_DATUM_NO_FIELD_INFERENCE"
EXPECTED_FILES = {"datum.npz", "row_manifest.csv", "result.json", "manifest.json", "COMPLETE"}


class DatumError(ValueError):
    """Fail-closed Phase-A datum-builder error."""


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
        raise DatumError(f"bound {label} is absent: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise DatumError(f"bound {label} size changed")
    if sha256_file(path) != str(binding["sha256"]):
        raise DatumError(f"bound {label} hash changed")
    return path


def load_program(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    program = json.loads(raw)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise DatumError("unexpected datum-builder program schema")
    authorization = program.get("authorization", {})
    if not authorization.get("Phase_A_datum_builder", False):
        raise DatumError("Phase-A datum builder is not authorized")
    for forbidden in (
        "field_inference",
        "mock_seed_access",
        "Phase_B_or_later",
        "automatic_follow_on",
    ):
        if authorization.get(forbidden, True):
            raise DatumError(f"program improperly authorizes {forbidden}")
    for label, binding in program["bindings"].items():
        _verify_binding(binding, label)
    return program, hashlib.sha256(raw).hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DatumError(f"cannot load bound module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_excluded_recnos(path: str | Path, expected_count: int) -> np.ndarray:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["recno", "reason", "parent_outliers_sha256"]:
            raise DatumError("excluded-recno manifest header changed")
        values = [int(row["recno"]) for row in reader]
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise DatumError("excluded-recno manifest count or uniqueness changed")
    return np.asarray(sorted(values), dtype=np.int64)


def hash_holdout_mask(
    recnos: np.ndarray, salt: str, numerator: int, denominator: int
) -> np.ndarray:
    """Return a deterministic pseudo-random row partition without float thresholds."""

    if not salt or numerator <= 0 or denominator <= numerator:
        raise DatumError("invalid holdout hash contract")
    limit = 1 << 64
    selected = []
    prefix = salt.encode("utf-8") + b":"
    for recno in np.asarray(recnos, dtype=np.int64):
        digest = hashlib.sha256(prefix + str(int(recno)).encode("ascii")).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        selected.append(denominator * value < numerator * limit)
    return np.asarray(selected, dtype=bool)


def integer_count_grids(
    populations: np.ndarray,
    flat_voxels: np.ndarray,
    holdout: np.ndarray,
    population_count: int,
    grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    populations = np.asarray(populations, dtype=np.int64)
    flat_voxels = np.asarray(flat_voxels, dtype=np.int64)
    holdout = np.asarray(holdout, dtype=bool)
    if not (populations.shape == flat_voxels.shape == holdout.shape):
        raise DatumError("count-grid row arrays have incompatible shapes")
    voxel_count = grid**3
    if np.any((populations < 0) | (populations >= population_count)):
        raise DatumError("population index is outside the frozen range")
    if np.any((flat_voxels < 0) | (flat_voxels >= voxel_count)):
        raise DatumError("voxel index is outside the frozen grid")

    joint = populations * voxel_count + flat_voxels

    def count(mask: np.ndarray) -> np.ndarray:
        values = np.bincount(joint[mask], minlength=population_count * voxel_count)
        return values.reshape((population_count, grid, grid, grid)).astype(np.int64)

    return count(np.ones(joint.size, dtype=bool)), count(~holdout), count(holdout)


def build_raw_selection(
    base: Any,
    joint: Any,
    program: Mapping[str, Any],
    tracer_program: Mapping[str, Any],
    information_program: Mapping[str, Any],
) -> np.ndarray:
    """Integrate the unnormalized six-population response in every N32 voxel."""

    import healpy as hp
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    design = information_program["design"]
    tracer_design = tracer_program["tracer_design"]
    cosmology = tracer_program["cosmology"]
    grid = int(design["grid_N"])
    box = float(design["box_size_cMpc_h"])
    spacing = box / grid
    if int(design["volume_quadrature_points_per_axis"]) != 2:
        raise DatumError("Phase A requires two-point Gauss quadrature per axis")
    nodes = np.asarray([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)])
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
    for ox in offsets:
        for oy in offsets:
            for oz in offsets:
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
                        flat[active_flat] += angular * radial / 8.0
    return exposure


def _row_manifest_bytes(
    recnos: np.ndarray,
    populations: np.ndarray,
    flat_voxels: np.ndarray,
    holdout: np.ndarray,
    radius: np.ndarray,
) -> bytes:
    import io

    handle = io.StringIO(newline="")
    fields = [
        "recno",
        "population_index",
        "voxel_flat_index",
        "split",
        "redshift_space_radius_cMpc_h",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    order = np.argsort(recnos)
    for index in order:
        writer.writerow(
            {
                "recno": int(recnos[index]),
                "population_index": int(populations[index]),
                "voxel_flat_index": int(flat_voxels[index]),
                "split": "holdout" if bool(holdout[index]) else "train",
                "redshift_space_radius_cMpc_h": format(float(radius[index]), ".17g"),
            }
        )
    return handle.getvalue().encode("utf-8")


def collect_datum(
    program: Mapping[str, Any], program_sha256: str, implementation_commit: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], bytes]:
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise DatumError("implementation commit must be lowercase 40-hex")
    base = _load_module(
        Path(program["bindings"]["tracer_implementation"]["path"]),
        "_cf4_datum_builder_tracer_v1",
    )
    joint = _load_module(
        Path(program["bindings"]["selection_implementation"]["path"]),
        "_cf4_datum_builder_selection_v1",
    )
    tracer_program = json.loads(
        Path(program["bindings"]["tracer_program"]["path"]).read_bytes()
    )
    information_program = json.loads(
        Path(program["bindings"]["selection_program"]["path"]).read_bytes()
    )
    frozen = program["frozen_subset"]
    design = information_program["design"]
    catalog = base.load_catalog(program["bindings"]["twompp_catalog"]["path"])
    exclusions, _ = base.read_crossmatch_exclusions(
        program["bindings"]["crossmatch"]["path"],
        int(program["crossmatch_policy"]["excluded_unique_targets_exact"]),
    )
    distance, absolute_magnitude = base.distance_and_absolute_magnitude(
        catalog["Vcmb"], catalog["Ksmag"], tracer_program["cosmology"]
    )
    eligible, partition, apparent_bin, absolute_bin = base.classify_disjoint_tracer(
        catalog,
        exclusions,
        distance,
        absolute_magnitude,
        tracer_program["tracer_design"],
    )
    excluded_recnos = read_excluded_recnos(
        program["bindings"]["excluded_recnos"]["path"],
        int(frozen["excluded_recno_count_exact"]),
    )
    metadata_excluded = np.isin(catalog["recno"], excluded_recnos)
    retained = eligible & ~metadata_excluded
    retained_indices = np.flatnonzero(retained)
    recnos = catalog["recno"][retained]
    populations = (3 * apparent_bin[retained] + absolute_bin[retained]).astype(np.int64)
    directions = base.supergalactic_unit_vectors(catalog["RA"], catalog["DEC"])[retained]
    radius = distance[retained]
    grid = int(design["grid_N"])
    box = float(design["box_size_cMpc_h"])
    spacing = box / grid
    positions = radius[:, None] * directions + box / 2.0
    if np.any(positions < 0.0) or np.any(positions >= box):
        raise DatumError("retained observed position lies outside the pilot box")
    voxel_indices = np.floor(positions / spacing).astype(np.int64)
    flat_voxels = np.ravel_multi_index(voxel_indices.T, (grid, grid, grid))
    split = program["split"]
    holdout = hash_holdout_mask(
        recnos,
        str(split["salt"]),
        int(split["holdout_fraction_numerator"]),
        int(split["holdout_fraction_denominator"]),
    )
    counts_all, counts_train, counts_holdout = integer_count_grids(
        populations, flat_voxels, holdout, 6, grid
    )
    raw_exposure = build_raw_selection(
        base, joint, program, tracer_program, information_program
    )
    arrays = {
        "counts_all": counts_all,
        "counts_train": counts_train,
        "counts_holdout": counts_holdout,
        "raw_selection_exposure": raw_exposure,
    }
    expected_populations = np.asarray(frozen["population_counts_exact"], dtype=np.int64)
    measured_populations = counts_all.reshape(6, -1).sum(axis=1)
    positive_count = counts_all > 0
    population_train = counts_train.reshape(6, -1).sum(axis=1)
    population_holdout = counts_holdout.reshape(6, -1).sum(axis=1)
    gates = {
        "retained_count_exact": recnos.size == int(frozen["retained_count_exact"]),
        "population_counts_exact": np.array_equal(measured_populations, expected_populations),
        "metadata_excluded_rows_were_parent_eligible": int(np.count_nonzero(metadata_excluded & eligible))
        == int(frozen["excluded_recno_count_exact"]),
        "retained_and_excluded_recnos_disjoint": not np.any(np.isin(recnos, excluded_recnos)),
        "retained_catalog_indices_unique": retained_indices.size == np.unique(retained_indices).size,
        "count_arrays_int64": all(
            arrays[name].dtype == np.int64
            for name in ("counts_all", "counts_train", "counts_holdout")
        ),
        "count_arrays_nonnegative": all(
            np.all(arrays[name] >= 0)
            for name in ("counts_all", "counts_train", "counts_holdout")
        ),
        "split_elementwise_exhaustive": np.array_equal(
            counts_all, counts_train + counts_holdout
        ),
        "each_population_has_train_and_holdout": bool(
            np.all(population_train > 0) and np.all(population_holdout > 0)
        ),
        "raw_exposure_shape_exact": raw_exposure.shape == (6, grid, grid, grid),
        "raw_exposure_finite_in_unit_interval": bool(
            np.all(np.isfinite(raw_exposure))
            and np.all(raw_exposure >= 0.0)
            and np.all(raw_exposure <= 1.0 + 8.0e-15)
        ),
        "each_population_has_positive_exposure_support": bool(
            np.all(raw_exposure.reshape(6, -1).sum(axis=1) > 0.0)
        ),
        "no_positive_count_in_nonpositive_exposure": not np.any(
            positive_count & (raw_exposure <= 0.0)
        ),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    result = {
        "schema": RESULT_SCHEMA,
        "status": STATUS_PASS if not failed else STATUS_FAIL,
        "program_sha256": program_sha256,
        "implementation_commit": implementation_commit,
        "catalog_partition_before_metadata_exclusion": partition,
        "counts": {
            "retained_total": int(recnos.size),
            "population_all": measured_populations.tolist(),
            "population_train": population_train.tolist(),
            "population_holdout": population_holdout.tolist(),
            "occupied_voxels_all": np.count_nonzero(counts_all, axis=(1, 2, 3)).tolist(),
        },
        "selection": {
            "normalization": "raw_exposure_not_normalized_to_observed_counts",
            "minimum": float(np.min(raw_exposure)),
            "maximum": float(np.max(raw_exposure)),
            "support_sum": raw_exposure.reshape(6, -1).sum(axis=1).tolist(),
            "positive_voxel_fraction": np.mean(
                raw_exposure.reshape(6, -1) > 0.0, axis=1
            ).tolist(),
            "positive_count_nonpositive_exposure_count": int(
                np.count_nonzero(positive_count & (raw_exposure <= 0.0))
            ),
            "quadrature_subpoints_per_voxel": 8,
        },
        "split": {
            "method": "SHA256_recno_deterministic_predictive_partition",
            "salt_sha256": hashlib.sha256(str(split["salt"]).encode("utf-8")).hexdigest(),
            "nominal_train_fraction": 1.0
            - int(split["holdout_fraction_numerator"])
            / int(split["holdout_fraction_denominator"]),
            "nominal_holdout_fraction": int(split["holdout_fraction_numerator"])
            / int(split["holdout_fraction_denominator"]),
            "realized_holdout_fraction": float(np.mean(holdout)),
            "conditional_independent_Poisson_claim": False,
        },
        "gates": gates,
        "failed_gates": failed,
        "field_inference_executed": False,
        "mock_seed_accessed": False,
        "observational_resolution_claim_allowed": False,
        "automatic_follow_on_executed": False,
    }
    rows = _row_manifest_bytes(recnos, populations, flat_voxels, holdout, radius)
    return result, arrays, rows


def publish_datum(
    program_path: str | Path, output: str | Path, implementation_commit: str
) -> dict[str, Any]:
    program, program_sha = load_program(program_path)
    target = Path(output)
    stage = target.with_name(f".{target.name}.staging")
    if target.exists() or stage.exists():
        raise DatumError("datum output or staging path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    result, arrays, rows = collect_datum(program, program_sha, implementation_commit)
    np.savez_compressed(stage / "datum.npz", **arrays)
    (stage / "row_manifest.csv").write_bytes(rows)
    (stage / "result.json").write_bytes(canonical_json_bytes(result))
    if result["failed_gates"]:
        failed = {
            "schema": "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-failed-v1",
            "status": STATUS_FAIL,
            "failed_gates": result["failed_gates"],
            "program_sha256": program_sha,
        }
        (stage / "FAILED").write_bytes(canonical_json_bytes(failed))
        raise DatumError(f"datum gates failed; diagnostics preserved at {stage}")
    artifact_names = ("datum.npz", "row_manifest.csv", "result.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "program_sha256": program_sha,
        "files": {
            name: {
                "bytes": (stage / name).stat().st_size,
                "sha256": sha256_file(stage / name),
            }
            for name in artifact_names
        },
    }
    (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    complete = {
        "schema": COMPLETE_SCHEMA,
        "status": STATUS_PASS,
        "program_sha256": program_sha,
        "manifest_sha256": sha256_file(stage / "manifest.json"),
        "result_sha256": sha256_file(stage / "result.json"),
        "field_inference_executed": False,
        "automatic_follow_on_executed": False,
    }
    (stage / "COMPLETE").write_bytes(canonical_json_bytes(complete))
    if {path.name for path in stage.iterdir()} != EXPECTED_FILES:
        raise DatumError("datum artifact file set is not exact")
    stage.rename(target)
    return result


def validate_datum(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise DatumError("published datum file set is not exact")
    raw = {name: (root / name).read_bytes() for name in EXPECTED_FILES if name != "datum.npz"}
    result = json.loads(raw["result.json"])
    manifest = json.loads(raw["manifest.json"])
    complete = json.loads(raw["COMPLETE"])
    if result.get("status") != STATUS_PASS or result.get("failed_gates"):
        raise DatumError("published datum result is not passing")
    if manifest.get("schema") != MANIFEST_SCHEMA or complete.get("schema") != COMPLETE_SCHEMA:
        raise DatumError("datum publication schema changed")
    for name in ("datum.npz", "row_manifest.csv", "result.json"):
        expected = {
            "bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        if manifest["files"].get(name) != expected:
            raise DatumError(f"manifest does not bind {name}")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise DatumError("COMPLETE does not bind the manifest")
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise DatumError("COMPLETE does not bind the result")
    with np.load(root / "datum.npz", allow_pickle=False) as archive:
        required = {
            "counts_all",
            "counts_train",
            "counts_holdout",
            "raw_selection_exposure",
        }
        if set(archive.files) != required:
            raise DatumError("datum NPZ key set changed")
        all_counts = np.asarray(archive["counts_all"])
        train = np.asarray(archive["counts_train"])
        holdout = np.asarray(archive["counts_holdout"])
        exposure = np.asarray(archive["raw_selection_exposure"])
    if all_counts.shape != (6, 32, 32, 32) or exposure.shape != all_counts.shape:
        raise DatumError("published datum shape changed")
    if all_counts.dtype != np.int64 or train.dtype != np.int64 or holdout.dtype != np.int64:
        raise DatumError("published count dtype changed")
    if not np.array_equal(all_counts, train + holdout):
        raise DatumError("published split is not exhaustive")
    if not np.all(np.isfinite(exposure)) or np.any(exposure < 0.0) or np.any(exposure > 1.0 + 8e-15):
        raise DatumError("published raw exposure is invalid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-datum-builder")
    run.add_argument("--program", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = sub.add_parser("validate-datum")
    validate.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run-datum-builder":
        result = publish_datum(args.program, args.output, args.implementation_commit)
    else:
        result = validate_datum(args.directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
