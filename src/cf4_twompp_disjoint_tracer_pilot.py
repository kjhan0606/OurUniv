#!/usr/bin/env python3
"""Technical pilot for a CF4-disjoint 2M++ density-tracer design.

This module does not perform field inference.  It validates the official
2M++ catalogue, the ARES/BORG angular-completeness maps, the conservative
CF4 crossmatch exclusion, and the published Carrick density cube before any
joint velocity-plus-density information calculation is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "ouruniv-cf4-twompp-disjoint-tracer-pilot-v1"
STATUS_PASS = "PASS_TECHNICAL_INPUT_GATE_NO_FIELD_INFERENCE"
PROGRAM_SCHEMA = "ouruniv-cf4-twompp-disjoint-tracer-pilot-program-v1"
EXPECTED_FILES = {"result.json", "manifest.json", "COMPLETE"}
LIGHT_SPEED_KMS = 299_792.458


class PilotError(ValueError):
    """Fail-closed technical-pilot error."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_program(path: str | Path) -> tuple[dict[str, Any], str]:
    program_path = Path(path)
    raw = program_path.read_bytes()
    program = json.loads(raw)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise PilotError("unexpected pilot program schema")
    if not program.get("authorization", {}).get("technical_pilot", False):
        raise PilotError("technical pilot is not authorized")
    return program, hashlib.sha256(raw).hexdigest()


def verify_file_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(binding["path"]))
    if not path.is_file():
        raise PilotError(f"bound input is absent: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(binding["bytes"]) or digest != binding["sha256"]:
        raise PilotError(f"bound input changed: {path}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def verify_ares_repository(binding: Mapping[str, Any]) -> dict[str, Any]:
    import subprocess

    root = Path(str(binding["path"]))
    if not (root / ".git").is_dir():
        raise PilotError("ARES repository is absent or not a git worktree")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != binding["commit"]:
        raise PilotError("ARES repository commit changed")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PilotError("ARES repository commit is malformed")
    return {"path": str(root), "commit": commit}


def read_crossmatch_exclusions(
    path: str | Path, expected_unique_targets: int
) -> tuple[set[int], dict[str, int]]:
    targets: set[int] = set()
    class_counts: dict[str, int] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "cf4_recno",
            "twompp_recno",
            "match_class",
        }
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise PilotError("crossmatch header changed")
        for row in reader:
            label = row["match_class"].strip()
            class_counts[label] = class_counts.get(label, 0) + 1
            target = row["twompp_recno"].strip()
            if label == "unmatched":
                if target:
                    raise PilotError("unmatched CF4 row unexpectedly has a 2M++ target")
                continue
            if not target:
                raise PilotError("non-unmatched CF4 row lacks a 2M++ target")
            targets.add(int(target))
    if len(targets) != expected_unique_targets:
        raise PilotError("unique crossmatch exclusion count changed")
    return targets, class_counts


def load_catalog(path: str | Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[Any]] = {
        "recno": [],
        "Ksmag": [],
        "Vcmb": [],
        "c11_5": [],
        "c12_5": [],
        "Cln": [],
        "Ref": [],
        "RA": [],
        "DEC": [],
    }
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(columns).issubset(reader.fieldnames):
            raise PilotError("2M++ catalogue header changed")
        for row in reader:
            columns["recno"].append(int(row["recno"]))
            columns["Ksmag"].append(float(row["Ksmag"]))
            columns["Vcmb"].append(float(row["Vcmb"]))
            columns["c11_5"].append(float(row["c11_5"]))
            columns["c12_5"].append(
                float(row["c12_5"]) if row["c12_5"].strip() else math.nan
            )
            columns["Cln"].append(int(row["Cln"]))
            columns["Ref"].append(row["Ref"].strip())
            columns["RA"].append(float(row["_RA"]))
            columns["DEC"].append(float(row["_DE"]))
    if len(set(columns["recno"])) != len(columns["recno"]):
        raise PilotError("2M++ recno is not unique")
    return {
        "recno": np.asarray(columns["recno"], dtype=np.int64),
        "Ksmag": np.asarray(columns["Ksmag"], dtype=np.float64),
        "Vcmb": np.asarray(columns["Vcmb"], dtype=np.float64),
        "c11_5": np.asarray(columns["c11_5"], dtype=np.float64),
        "c12_5": np.asarray(columns["c12_5"], dtype=np.float64),
        "Cln": np.asarray(columns["Cln"], dtype=np.int8),
        "Ref": np.asarray(columns["Ref"], dtype=str),
        "RA": np.asarray(columns["RA"], dtype=np.float64),
        "DEC": np.asarray(columns["DEC"], dtype=np.float64),
    }


def load_completeness_map(path: str | Path, expected_nside: int) -> np.ndarray:
    import healpy as hp

    values = np.asarray(
        hp.read_map(str(path), field=0, nest=False, dtype=np.float64, memmap=False),
        dtype=np.float64,
    )
    if values.shape != (12 * expected_nside**2,):
        raise PilotError("HEALPix completeness map size changed")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise PilotError("HEALPix completeness map is outside [0,1]")
    return values


def galactic_directions(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    coordinates = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    galactic = coordinates.galactic
    return galactic.l.rad.astype(np.float64), galactic.b.rad.astype(np.float64)


def supergalactic_unit_vectors(
    ra: np.ndarray, dec: np.ndarray
) -> np.ndarray:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    coordinates = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    sg = coordinates.supergalactic
    longitude = sg.sgl.rad
    latitude = sg.sgb.rad
    cosine = np.cos(latitude)
    return np.column_stack(
        (cosine * np.cos(longitude), cosine * np.sin(longitude), np.sin(latitude))
    ).astype(np.float64)


def map_catalog_agreement(
    values: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    catalog_values: np.ndarray,
    valid: np.ndarray,
    nside: int,
) -> dict[str, float | int]:
    import healpy as hp

    theta = 0.5 * np.pi - latitude[valid]
    phi = np.mod(longitude[valid], 2.0 * np.pi)
    pixels = hp.ang2pix(nside, theta, phi, nest=False)
    difference = np.abs(values[pixels] - catalog_values[valid])
    if difference.size == 0 or np.any(~np.isfinite(difference)):
        raise PilotError("no finite map/catalog completeness comparisons")
    return {
        "row_count": int(difference.size),
        "median_absolute_difference": float(np.median(difference)),
        "p95_absolute_difference": float(np.quantile(difference, 0.95)),
        "maximum_absolute_difference": float(np.max(difference)),
    }


def distance_and_absolute_magnitude(
    vcmb: np.ndarray, apparent_magnitude: np.ndarray, cosmology: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM

    model = FlatLambdaCDM(
        H0=float(cosmology["H0_km_s_Mpc"]) * u.km / u.s / u.Mpc,
        Om0=float(cosmology["Omega_m"]),
        Ob0=float(cosmology["Omega_b"]),
        Tcmb0=float(cosmology["Tcmb_K"]) * u.K,
    )
    redshift = vcmb / LIGHT_SPEED_KMS
    distance_hmpc = model.comoving_distance(redshift).to_value(u.Mpc) * float(
        cosmology["h"]
    )
    distance_modulus = model.distmod(redshift).value
    return distance_hmpc, apparent_magnitude - distance_modulus


def classify_disjoint_tracer(
    catalog: Mapping[str, np.ndarray],
    excluded_targets: set[int],
    distance_hmpc: np.ndarray,
    absolute_magnitude: np.ndarray,
    design: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, int], np.ndarray, np.ndarray]:
    count = catalog["recno"].size
    reason = np.full(count, "", dtype="U40")
    zoa = np.char.lower(np.char.strip(catalog["Ref"].astype(str))) == "zoa"
    reason[zoa] = "excluded_ZoA_imputation"
    mask = reason == ""
    cloned = catalog["Cln"] == 1
    reason[mask & cloned] = "excluded_cloned_redshift"
    mask = reason == ""
    finite = (
        np.isfinite(catalog["Ksmag"])
        & np.isfinite(catalog["Vcmb"])
        & np.isfinite(catalog["RA"])
        & np.isfinite(catalog["DEC"])
        & np.isfinite(distance_hmpc)
        & np.isfinite(absolute_magnitude)
        & (catalog["Vcmb"] > 0.0)
    )
    reason[mask & ~finite] = "excluded_invalid_observed_fields"
    mask = reason == ""
    magnitude_valid = (catalog["Ksmag"] <= float(design["faint_apparent_K_max"])) & (
        (catalog["Ksmag"] <= float(design["bright_apparent_K_max"]))
        | np.isfinite(catalog["c12_5"])
    )
    reason[mask & ~magnitude_valid] = "excluded_apparent_magnitude_or_mask"
    mask = reason == ""
    overlap = np.isin(catalog["recno"], np.fromiter(excluded_targets, dtype=np.int64))
    reason[mask & overlap] = "excluded_any_CF4_match_candidate"
    mask = reason == ""
    radial = (distance_hmpc >= float(design["radial_min_cMpc_h"])) & (
        distance_hmpc <= float(design["radial_max_cMpc_h"])
    )
    reason[mask & ~radial] = "excluded_radial_support"
    mask = reason == ""
    absolute = (absolute_magnitude > float(design["absolute_K_bright"])) & (
        absolute_magnitude < float(design["absolute_K_faint"])
    )
    reason[mask & ~absolute] = "excluded_absolute_magnitude"
    eligible = reason == ""
    reason[eligible] = "eligible_disjoint_tracer"
    counts = {label: int(np.count_nonzero(reason == label)) for label in np.unique(reason)}
    if sum(counts.values()) != count:
        raise PilotError("tracer classification is not an exact partition")

    apparent_bin = np.where(
        catalog["Ksmag"] <= float(design["bright_apparent_K_max"]), 0, 1
    ).astype(np.int8)
    edges = np.asarray(design["absolute_K_edges"], dtype=np.float64)
    absolute_bin = np.searchsorted(edges, absolute_magnitude, side="left") - 1
    # Edges ascend from the bright (more negative) to faint limit.
    absolute_bin = np.clip(absolute_bin, 0, edges.size - 2).astype(np.int8)
    return eligible, counts, apparent_bin, absolute_bin


def voxel_summary(
    unit_vectors: np.ndarray,
    distance_hmpc: np.ndarray,
    eligible: np.ndarray,
    apparent_bin: np.ndarray,
    absolute_bin: np.ndarray,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    grid = int(design["grid_N"])
    box = float(design["box_size_cMpc_h"])
    spacing = box / grid
    positions = distance_hmpc[eligible, None] * unit_vectors[eligible] + box / 2.0
    if np.any(positions < 0.0) or np.any(positions >= box):
        raise PilotError("eligible tracer lies outside the N32 box")
    indices = np.floor(positions / spacing).astype(np.int64)
    flat = np.ravel_multi_index(indices.T, (grid, grid, grid))
    occupancy = np.bincount(flat, minlength=grid**3)
    populations = {}
    for apparent in (0, 1):
        for absolute in range(len(design["absolute_K_edges"]) - 1):
            selected = eligible & (apparent_bin == apparent) & (absolute_bin == absolute)
            populations[f"apparent_{apparent}_absolute_{absolute}"] = int(np.count_nonzero(selected))
    occupied = occupancy[occupancy > 0]
    return {
        "spacing_cMpc_h": spacing,
        "eligible_row_count": int(np.count_nonzero(eligible)),
        "occupied_voxel_count": int(occupied.size),
        "occupied_voxel_fraction": float(occupied.size / occupancy.size),
        "count_per_occupied_voxel": {
            "minimum": int(occupied.min()) if occupied.size else 0,
            "median": float(np.median(occupied)) if occupied.size else 0.0,
            "p95": float(np.quantile(occupied, 0.95)) if occupied.size else 0.0,
            "maximum": int(occupied.max()) if occupied.size else 0,
        },
        "six_population_counts": populations,
    }


def carrick_cube_audit(path: str | Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    cube = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(cube.shape) != list(expected["shape"]) or str(cube.dtype) != expected["dtype"]:
        raise PilotError("Carrick density cube shape or dtype changed")
    values = np.asarray(cube)
    finite = np.isfinite(values)
    if not np.all(finite):
        raise PilotError("Carrick density cube contains nonfinite values")
    minimum = float(np.min(values))
    if minimum < float(expected["density_contrast_minimum_allowed"]):
        raise PilotError("Carrick density cube violates the density-contrast lower bound")
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "C_contiguous": bool(values.flags.c_contiguous),
        "finite_fraction": float(np.mean(finite)),
        "minimum": minimum,
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "central_voxel": float(values[128, 128, 128]),
        "zero_fraction": float(np.mean(values == 0.0)),
    }


def run_audit(program: Mapping[str, Any], program_sha256: str, commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise PilotError("implementation commit must be lowercase 40-hex")
    inputs = program["inputs"]
    bound = {
        name: verify_file_binding(inputs[name])
        for name in (
            "twompp_catalog",
            "cf4_twompp_crossmatch",
            "cf4_twompp_crossmatch_result",
            "completeness_11_5",
            "completeness_12_5",
            "carrick_density",
            "carrick_readme",
        )
    }
    ares = verify_ares_repository(inputs["ares_repository"])
    exclusions, class_counts = read_crossmatch_exclusions(
        inputs["cf4_twompp_crossmatch"]["path"],
        int(program["no_double_counting"]["expected_unique_2Mpp_targets_excluded"]),
    )
    catalog = load_catalog(inputs["twompp_catalog"]["path"])
    if catalog["recno"].size != int(program["catalog_gate"]["expected_rows"]):
        raise PilotError("2M++ row count changed")

    nside = int(program["angular_completeness_gate"]["HEALPix_NSIDE"])
    completeness11 = load_completeness_map(inputs["completeness_11_5"]["path"], nside)
    completeness12 = load_completeness_map(inputs["completeness_12_5"]["path"], nside)
    longitude, latitude = galactic_directions(catalog["RA"], catalog["DEC"])
    real = np.char.lower(np.char.strip(catalog["Ref"].astype(str))) != "zoa"
    agreement11 = map_catalog_agreement(
        completeness11,
        longitude,
        latitude,
        catalog["c11_5"],
        real & np.isfinite(catalog["c11_5"]),
        nside,
    )
    agreement12 = map_catalog_agreement(
        completeness12,
        longitude,
        latitude,
        catalog["c12_5"],
        real & np.isfinite(catalog["c12_5"]),
        nside,
    )

    distance, absolute_magnitude = distance_and_absolute_magnitude(
        catalog["Vcmb"], catalog["Ksmag"], program["cosmology"]
    )
    eligible, reason_counts, apparent_bin, absolute_bin = classify_disjoint_tracer(
        catalog, exclusions, distance, absolute_magnitude, program["tracer_design"]
    )
    unit_vectors = supergalactic_unit_vectors(catalog["RA"], catalog["DEC"])
    voxels = voxel_summary(
        unit_vectors,
        distance,
        eligible,
        apparent_bin,
        absolute_bin,
        program["tracer_design"],
    )
    carrick = carrick_cube_audit(
        inputs["carrick_density"]["path"], program["carrick_reference_gate"]
    )

    angular_gate = program["angular_completeness_gate"]
    gates = {
        "ARES_commit_exact": ares["commit"] == inputs["ares_repository"]["commit"],
        "crossmatch_unique_target_count_exact": len(exclusions)
        == int(program["no_double_counting"]["expected_unique_2Mpp_targets_excluded"]),
        "completeness_11_5_median_abs_difference": agreement11[
            "median_absolute_difference"
        ]
        <= float(angular_gate["median_absolute_difference_max_inclusive"]),
        "completeness_11_5_p95_abs_difference": agreement11[
            "p95_absolute_difference"
        ]
        <= float(angular_gate["p95_absolute_difference_max_inclusive"]),
        "completeness_12_5_median_abs_difference": agreement12[
            "median_absolute_difference"
        ]
        <= float(angular_gate["median_absolute_difference_max_inclusive"]),
        "completeness_12_5_p95_abs_difference": agreement12[
            "p95_absolute_difference"
        ]
        <= float(angular_gate["p95_absolute_difference_max_inclusive"]),
        "eligible_disjoint_tracer_minimum": voxels["eligible_row_count"]
        >= int(program["catalog_gate"]["eligible_disjoint_rows_min_inclusive"]),
        "all_six_populations_nonempty": all(
            value > 0 for value in voxels["six_population_counts"].values()
        ),
        "Carrick_cube_full_finite": carrick["finite_fraction"] == 1.0,
    }
    if not all(gates.values()):
        failed = sorted(key for key, value in gates.items() if not value)
        raise PilotError(f"technical pilot gate failed: {failed}")

    return {
        "schema": SCHEMA,
        "status": STATUS_PASS,
        "program_sha256": program_sha256,
        "implementation_commit": commit,
        "bound_inputs": bound,
        "ARES_repository": ares,
        "crossmatch": {
            "class_counts": class_counts,
            "unique_2Mpp_targets_excluded": len(exclusions),
            "policy": "exclude every non-unmatched 2M++ target before density-tracer use",
        },
        "angular_completeness": {
            "HEALPix_NSIDE": nside,
            "ordering": "RING",
            "map_11_5": {
                "minimum": float(np.min(completeness11)),
                "maximum": float(np.max(completeness11)),
                "positive_pixel_fraction": float(np.mean(completeness11 > 0.0)),
                "catalog_agreement": agreement11,
            },
            "map_12_5": {
                "minimum": float(np.min(completeness12)),
                "maximum": float(np.max(completeness12)),
                "positive_pixel_fraction": float(np.mean(completeness12 > 0.0)),
                "catalog_agreement": agreement12,
            },
        },
        "exclusive_catalog_classification": reason_counts,
        "tracer_voxel_summary": voxels,
        "Carrick_density_reference": carrick,
        "gates": gates,
        "field_inference_executed": False,
        "likelihood_datum_consumed_by_field_inference": False,
        "information_frontier_claim_allowed": False,
        "observational_resolution_claim_allowed": False,
        "next_action_requires_result_record_and_user_approval": True,
    }


def publish_pilot(
    program_path: str | Path, output: str | Path, implementation_commit: str
) -> dict[str, Any]:
    program, program_sha = load_program(program_path)
    target = Path(output)
    stage = target.with_name(f".{target.name}.staging")
    if target.exists() or stage.exists():
        raise PilotError("pilot output or staging path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    try:
        result = run_audit(program, program_sha, implementation_commit)
        result_bytes = canonical_json_bytes(result)
        (stage / "result.json").write_bytes(result_bytes)
        result_sha = hashlib.sha256(result_bytes).hexdigest()
        manifest = {
            "schema": "ouruniv-cf4-twompp-disjoint-tracer-pilot-manifest-v1",
            "files": {
                "result.json": {
                    "bytes": len(result_bytes),
                    "sha256": result_sha,
                }
            },
        }
        manifest_bytes = canonical_json_bytes(manifest)
        (stage / "manifest.json").write_bytes(manifest_bytes)
        complete = {
            "schema": "ouruniv-cf4-twompp-disjoint-tracer-pilot-complete-v1",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "result_sha256": result_sha,
            "status": STATUS_PASS,
        }
        (stage / "COMPLETE").write_bytes(canonical_json_bytes(complete))
        os.rename(stage, target)
    except BaseException:
        # Preserve failed staging content and traceback context for audit.
        raise
    return validate_pilot(target)


def validate_pilot(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise PilotError("pilot artifact file set is not exact")
    result_raw = (root / "result.json").read_bytes()
    manifest_raw = (root / "manifest.json").read_bytes()
    complete_raw = (root / "COMPLETE").read_bytes()
    result = json.loads(result_raw)
    manifest = json.loads(manifest_raw)
    complete = json.loads(complete_raw)
    if result.get("schema") != SCHEMA or result.get("status") != STATUS_PASS:
        raise PilotError("pilot result status changed")
    if not all(result.get("gates", {}).values()):
        raise PilotError("pilot result contains a failed gate")
    result_sha = hashlib.sha256(result_raw).hexdigest()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    declared = manifest.get("files", {}).get("result.json", {})
    if declared != {"bytes": len(result_raw), "sha256": result_sha}:
        raise PilotError("pilot manifest does not bind result.json")
    if complete.get("manifest_sha256") != manifest_sha:
        raise PilotError("pilot COMPLETE does not bind manifest.json")
    if complete.get("result_sha256") != result_sha or complete.get("status") != STATUS_PASS:
        raise PilotError("pilot COMPLETE does not bind the passing result")
    return {"status": "PASS", "result_sha256": result_sha, "gates": result["gates"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-pilot")
    run.add_argument("--program", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = subparsers.add_parser("validate-pilot")
    validate.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run-pilot":
        result = publish_pilot(args.program, args.output, args.implementation_commit)
    else:
        result = validate_pilot(args.directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
