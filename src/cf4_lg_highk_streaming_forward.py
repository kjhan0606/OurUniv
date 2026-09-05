#!/usr/bin/env python3
"""Stream frozen high-k fields through PM and FoF without retaining fields.

This is intentionally a *forward-all-rows* runner.  It has no P1 import and
therefore cannot perform the forbidden parent-centred P1 prefilter.  A later
pair-recentred P1 gate consumes the halo catalogues produced here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cf4_aggregate_evidence_oracle import target_vector
from cf4_lg_highk_conditional_field import conditional_field
from cf4_lg_highk_covariance_cache import (
    load_covariance_for_schedule_row,
    sha256_file,
    validate_covariance_cache,
)
from cf4_peak_evidence_phase_cache import full_spectrum_from_rfft
from cf4_p2_screen import (
    RHO_CRIT,
    VUNIT_KMS,
    extract_central_arrays,
    find_pairs,
    load_config,
    rank_score,
)
from cf4_lg_z0_likelihood import score_catalog


SCHEMA = "ouruniv-cf4-lg-highk-streaming-forward-result-v1"
RUN_MANIFEST_SCHEMA = "ouruniv-cf4-lg-highk-streaming-forward-run-manifest-v1"
PILOT_INDICES = np.asarray([0, 64, 128, 192], dtype=np.int64)
PRODUCTION_BATCHES = 16
ROWS_PER_BATCH = 16

ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_program(path: Path) -> tuple[dict[str, Any], Path]:
    path = Path(path)
    program = json.loads(path.read_text())
    if program.get("schema") != "ouruniv-cf4-lg-highk-streaming-forward-program-v1":
        raise ValueError("unexpected streaming-forward program schema")
    if "inputs" not in program:
        raise ValueError("streaming-forward program has no inputs")
    return program, path


def _pinned_input(program: Mapping[str, Any], name: str) -> tuple[Path, str]:
    try:
        spec = program["inputs"][name]
        path = _resolve_path(spec["path"])
        expected = str(spec["sha256"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"program input {name!r} must contain path and sha256") from error
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 changed: {actual} != {expected}")
    return path, expected


def validate_program_inputs(program: Mapping[str, Any]) -> dict[str, tuple[Path, str]]:
    """Validate every input consumed by this runner; P1 is explicitly absent."""
    required = (
        "schedule", "density_filter", "parent_manifest", "hard_p2_config",
        "z0_likelihood_source",
    )
    pins = {name: _pinned_input(program, name) for name in required}
    policy = program.get("probability_policy", {})
    if policy.get("parent_centered_P1_prefilter_authorized") is not False:
        raise ValueError("program must explicitly forbid parent-centred P1 prefilter")
    if policy.get("peak_evidence_reapplied") is not False:
        raise ValueError("program must explicitly prohibit peak-evidence reapplication")
    return pins


def _load_schedule(path: Path, program: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = {
        "schedule_index", "group_id", "parent_seed", "keys", "fine_field_seed",
        "likelihood_noise_seed", "posterior_weight",
    }
    with np.load(path, allow_pickle=False) as item:
        missing = required.difference(item.files)
        if missing:
            raise ValueError(f"schedule is missing arrays: {sorted(missing)}")
        schedule = {name: np.asarray(item[name]) for name in required}
    count = int(program["inputs"]["schedule"].get("row_count", 256))
    if count != 256 or len(schedule["schedule_index"]) != count \
            or not np.array_equal(schedule["schedule_index"], np.arange(count)):
        raise ValueError("frozen schedule must retain 256 contiguous rows")
    if schedule["keys"].shape != (count, 6):
        raise ValueError("frozen schedule geometry shape changed")
    unique_count = int(len(np.unique(schedule["keys"], axis=0)))
    expected_unique = int(program["inputs"]["schedule"].get("unique_geometry_key_count", 250))
    if unique_count != expected_unique:
        raise ValueError("frozen schedule unique geometry key count changed")
    groups = np.asarray(schedule["group_id"], dtype=np.int64)
    if not np.array_equal(np.bincount(groups, minlength=4), np.full(4, 64)):
        raise ValueError("frozen schedule group balance changed")
    if not np.allclose(schedule["posterior_weight"], 1.0 / 256.0, rtol=0.0, atol=1e-15):
        raise ValueError("frozen schedule posterior weights changed")
    return schedule


def _parent_entries(path: Path) -> dict[int, dict[str, Any]]:
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete_exact_parent_response_atlas" \
            or int(manifest.get("parent_count", -1)) != 256:
        raise ValueError("parent manifest is not the complete exact atlas")
    entries = {int(row["seed"]): row for row in manifest["entries"]}
    if len(entries) != 256:
        raise ValueError("parent manifest seed identity changed")
    return entries


def _output_row(output_root: Path, schedule_index: int) -> Path:
    return output_root / f"row_{int(schedule_index):03d}"


def _valid_completed_row(row_dir: Path, *, identity: Mapping[str, Any]) -> bool:
    """Return True only for a self-consistent completed canonical row.

    Any existing canonical row that does not satisfy this narrow predicate is a
    fail-closed condition; callers must never silently replay or overwrite it.
    """
    result_path, halo_path = row_dir / "result.json", row_dir / "halos.npz"
    if not result_path.is_file() or not halo_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        result.get("schema") == SCHEMA
        and result.get("status") == "complete"
        and all(result.get(name) == value for name, value in identity.items())
        and result.get("halo_catalogue") == "halos.npz"
        and result.get("halo_catalogue_sha256") == sha256_file(halo_path)
    )


def production_run_manifest(
    *, program_path: Path, program: Mapping[str, Any], cache_path: Path
) -> dict[str, Any]:
    """Create the immutable 16-by-16 production coverage contract in memory."""
    schedule_path, schedule_hash = _pinned_input(program, "schedule")
    cache_validation = validate_covariance_cache(
        cache_path, schedule_path=schedule_path,
        filter_path=_pinned_input(program, "density_filter")[0],
    )
    batches = [list(range(number * ROWS_PER_BATCH, (number + 1) * ROWS_PER_BATCH))
               for number in range(PRODUCTION_BATCHES)]
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "status": "prepared",
        "program": str(Path(program_path).resolve()),
        "program_sha256": sha256_file(Path(program_path)),
        "schedule": str(schedule_path.resolve()),
        "schedule_sha256": schedule_hash,
        "covariance_cache": str(Path(cache_path).resolve()),
        "covariance_cache_sha256": cache_validation["sha256"],
        "batch_count": PRODUCTION_BATCHES,
        "rows_per_batch": ROWS_PER_BATCH,
        "batches": batches,
    }


def validate_production_run_manifest(
    manifest: Mapping[str, Any], *, program_path: Path, cache_path: Path
) -> None:
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA or manifest.get("status") != "prepared":
        raise ValueError("production run manifest is not prepared")
    if manifest.get("program_sha256") != sha256_file(Path(program_path)):
        raise RuntimeError("production run manifest program pin changed")
    if manifest.get("covariance_cache_sha256") != sha256_file(Path(cache_path)):
        raise RuntimeError("production run manifest covariance-cache pin changed")
    batches = manifest.get("batches")
    expected = [list(range(batch * 16, (batch + 1) * 16)) for batch in range(16)]
    if manifest.get("batch_count") != 16 or manifest.get("rows_per_batch") != 16 \
            or batches != expected:
        raise ValueError("production run manifest does not provide exact 16x16 coverage")


def prepare_production_run(*, program_path: Path, output_root: Path, cache_path: Path) -> dict[str, Any]:
    program, _ = load_program(program_path)
    if program.get("authorization", {}).get("production_256_forward_execution") is not True:
        raise PermissionError("production preparation is not authorized by the frozen program")
    validate_program_inputs(program)
    _load_schedule(_pinned_input(program, "schedule")[0], program)
    manifest = production_run_manifest(
        program_path=program_path, program=program, cache_path=cache_path
    )
    output_root = Path(output_root)
    if output_root.exists() and not (output_root / "run_manifest.json").exists() \
            and any(output_root.iterdir()):
        raise RuntimeError("refusing to prepare production in a nonempty root without run_manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "run_manifest.json"
    if target.exists():
        existing = json.loads(target.read_text())
        validate_production_run_manifest(existing, program_path=program_path, cache_path=cache_path)
        if existing != manifest:
            raise RuntimeError("existing production run manifest differs from frozen contract")
        return existing
    _atomic_json(target, manifest)
    return manifest


def _mode_indices(program: Mapping[str, Any], mode: str, batch_index: int | None) -> np.ndarray:
    if mode == "pilot":
        indices = np.asarray(program["technical_pilot"]["schedule_indices"], dtype=np.int64)
        if not np.array_equal(indices, PILOT_INDICES):
            raise ValueError("technical pilot indices must be [0, 64, 128, 192]")
        if batch_index is not None:
            raise ValueError("--batch-index is only valid in production mode")
        return indices
    if mode != "production":
        raise ValueError("mode must be pilot or production")
    if batch_index is None or not 0 <= batch_index < PRODUCTION_BATCHES:
        raise ValueError("production requires --batch-index in [0, 15]")
    return np.arange(batch_index * ROWS_PER_BATCH, (batch_index + 1) * ROWS_PER_BATCH)


def _highk_gates(diagnostics: Mapping[str, Any], program: Mapping[str, Any]) -> dict[str, bool]:
    """Apply preregistered numerical gates, requiring an explicit gate block."""
    gates = program.get(
        "highk_numerical_gates",
        program.get("fixed_model", {}).get(
            "highk_numerical_gates",
            program.get("technical_pilot", {}).get("numerical_gates"),
        ),
    )
    if not isinstance(gates, Mapping):
        raise ValueError("program must provide explicit highk_numerical_gates")
    try:
        null_lo, null_hi = gates["null_subspace_mean_square_range"]
        return {
            "coarse_roundtrip": diagnostics["coarse_roundtrip_relative_RMS"] <= gates["coarse_roundtrip_relative_RMS_max"],
            "correction_in_null_space": diagnostics["correction_restriction_relative_RMS"] <= gates["correction_restriction_relative_RMS_max"],
            "response_identity": diagnostics["maximum_response_identity_error"] <= gates["maximum_response_identity_error_max"],
            "null_power": null_lo <= diagnostics["null_subspace_mean_square"] <= null_hi,
            "global_mean": abs(diagnostics["field_mean"]) <= gates["absolute_global_field_mean_max"],
            "field_imaginary": diagnostics["field_imaginary_relative_RMS"] <= gates["maximum_field_imaginary_relative_RMS"],
            "peak_evidence_not_reapplied": diagnostics["peak_evidence_reapplied"] is False,
        }
    except KeyError as error:
        raise ValueError(f"highk_numerical_gates is missing {error.args[0]}") from error


def _load_parent(entry: Mapping[str, Any], parent_seed: int) -> tuple[np.ndarray, dict[str, float]]:
    path = Path(entry["parent_field"])
    actual_hash = sha256_file(path)
    if actual_hash != entry["parent_field_sha256"]:
        raise RuntimeError(f"parent {parent_seed} SHA256 changed")
    with np.load(path, allow_pickle=False) as item:
        if int(item["sample_seed"]) != parent_seed:
            raise ValueError("parent internal seed changed")
        coarse = np.asarray(item["s_out"], dtype=np.float32)
        cosmology = {
            "Om": float(item["Om"]), "Ob": float(item["Ob"]),
            "h": float(item["hh"]), "A_s_1e9": float(item["A_s_1e9"]),
            "ns": float(item["ns"]),
            "box_size": float(item["L"]), "mesh": int(item["N"]),
        }
    return coarse, cosmology


def _run_row(
    *, schedule_index: int, schedule: Mapping[str, np.ndarray], parent_entries: Mapping[int, Mapping[str, Any]],
    filter_full: np.ndarray, cache_path: Path, hard_p2: Mapping[str, Any], z0_program: Mapping[str, Any],
    fixed: Mapping[str, Any], forward: Any, cosmology: Mapping[str, float], output_root: Path,
    row_identity: Mapping[str, Any], program: Mapping[str, Any],
) -> dict[str, Any]:
    row_dir = _output_row(output_root, schedule_index)
    if row_dir.exists():
        if _valid_completed_row(row_dir, identity=row_identity):
            saved = json.loads((row_dir / "result.json").read_text())
            return {
                "schedule_index": schedule_index, "resumed": True,
                "highk_numerical_pass": bool(saved.get("highk_numerical_pass")),
            }
        raise RuntimeError(f"canonical row {row_dir} is incomplete or does not verify; refusing overwrite")
    staging = output_root / f".row_{schedule_index:03d}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    field = final_pos = final_vel = central_pos = central_vel = halos = None
    try:
        key, points, covariance = load_covariance_for_schedule_row(cache_path, schedule_index)
        if not np.array_equal(key, schedule["keys"][schedule_index]):
            raise RuntimeError("validated covariance cache returned the wrong geometry key")
        parent_seed = int(schedule["parent_seed"][schedule_index])
        coarse, parent_cosmology = _load_parent(parent_entries[parent_seed], parent_seed)
        if parent_cosmology != cosmology:
            raise ValueError("parent cosmology changed within a compiled forward batch")
        field, field_diagnostics = conditional_field(
            coarse, filter_full, points,
            target_vector(float(fixed["centre_target_delta_linear"]), float(fixed["shell_target_delta_linear"])),
            float(fixed["likelihood_sigma_delta"]),
            fine_seed=int(schedule["fine_field_seed"][schedule_index]),
            noise_seed=int(schedule["likelihood_noise_seed"][schedule_index]),
            signal_covariance=covariance, float_dtype=np.float32,
            workers=int(fixed["FFT_workers"]),
        )
        field_sha256 = hashlib.sha256(memoryview(field).cast("B")).hexdigest()
        highk_gates = _highk_gates(field_diagnostics, program)
        if not all(highk_gates.values()):
            raise RuntimeError(
                f"schedule row {schedule_index} failed a high-k numerical gate"
            )
        import jax.numpy as jnp
        final_pos, final_vel = forward(jnp.asarray(field))
        # The full N576 field is intentionally released before host particle
        # extraction and never passed to an output routine.
        del field, coarse
        field = coarse = None
        fof_cfg = fixed["FoF"]
        centre = np.full(3, float(fixed["box_size_mpc_h"]) / 2.0)
        central_pos, central_vel = extract_central_arrays(
            final_pos, final_vel, centre, float(fof_cfg["central_half_width_mpc_h"]),
            velocity_unit=VUNIT_KMS,
        )
        del final_pos, final_vel
        final_pos = final_vel = None
        from fof import fof
        spacing = float(fixed["particle_spacing_mpc_h"])
        particle_mass = float(cosmology["Om"]) * RHO_CRIT * spacing**3
        halos = fof(
            central_pos, central_vel, L=float(fixed["box_size_mpc_h"]), mean_sep=spacing,
            b=float(fof_cfg["linking_length_b"]), n_min=int(fof_cfg["minimum_particle_count"]),
            m_particle=particle_mass, periodic=bool(fof_cfg["periodic"]), verbose=False,
        )
        pairs = find_pairs(halos, centre, hard_p2["screen"], hard_p2["m33_subpeak_gate"])
        for pair in pairs:
            pair["ranking_score"] = rank_score(pair, hard_p2["ranking"])
        pairs.sort(key=lambda pair: pair["ranking_score"])
        likelihood = score_catalog(
            halos["pos"], halos["vel"], halos["mass"], centre=centre,
            box_size=float(fixed["box_size_mpc_h"]), program=z0_program,
        )
        halo_path = staging / "halos.npz"
        _atomic_savez(
            halo_path,
            halo_pos=np.asarray(halos["pos"], dtype=np.float32),
            halo_vel=np.asarray(halos["vel"], dtype=np.float32),
            halo_mass=np.asarray(halos["mass"], dtype=np.float32),
            particle_mass=np.float64(particle_mass), box_size=np.float64(fixed["box_size_mpc_h"]),
        )
        result = {
            "schema": SCHEMA, "status": "complete", "schedule_index": int(schedule_index),
            **row_identity,
            "group_id": int(schedule["group_id"][schedule_index]),
            "geometry_key": key.tolist(), "posterior_weight": float(schedule["posterior_weight"][schedule_index]),
            "fine_field_seed": int(schedule["fine_field_seed"][schedule_index]),
            "likelihood_noise_seed": int(schedule["likelihood_noise_seed"][schedule_index]),
            "field_sha256": field_sha256,
            "halo_catalogue": "halos.npz", "halo_catalogue_sha256": sha256_file(halo_path),
            "field_persisted": False, "full_pm_particle_state_persisted": False,
            "parent_centered_P1_evaluated": False, "field_diagnostics": field_diagnostics,
            "highk_numerical_gates": highk_gates, "highk_numerical_pass": all(highk_gates.values()),
            "n_central_particles": int(central_pos.shape[0]), "n_halos": int(halos["mass"].size),
            "hard_p2_pairs": pairs, "hard_p2_pass": bool(pairs), "z0_likelihood": likelihood,
            "seconds": time.monotonic() - started,
        }
        _atomic_json(staging / "result.json", result)
        os.replace(staging, row_dir)
        return result
    except Exception:
        # This cleanup applies only to our unique, non-canonical staging dir.
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        del field, final_pos, final_vel, central_pos, central_vel, halos
        gc.collect()


def run_streaming_forward(
    *, program_path: Path, mode: str, output_root: Path, covariance_cache: Path | None = None,
    batch_index: int | None = None,
) -> list[dict[str, Any]]:
    """Run one four-row pilot or exactly one 16-row production batch."""
    program, _ = load_program(program_path)
    auth_key = "integrated_four_row_PM_pilot_execution" if mode == "pilot" else "production_256_forward_execution"
    if program.get("authorization", {}).get(auth_key) is not True:
        raise PermissionError(f"{mode} streaming execution is not authorized by the frozen program")
    pins = validate_program_inputs(program)
    schedule_path, schedule_sha = pins["schedule"]
    schedule = _load_schedule(schedule_path, program)
    cache_path = Path(covariance_cache) if covariance_cache else _resolve_path(
        Path(program["covariance_cache"]["artifact_root"]) / program["covariance_cache"]["cache_file"]
    )
    cache_info = validate_covariance_cache(
        cache_path, schedule_path=schedule_path, filter_path=pins["density_filter"][0]
    )
    if cache_info["diagnostics"]["unique_key_count"] != 250:
        raise ValueError("covariance cache must contain 250 unique geometry keys")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    indices = _mode_indices(program, mode, batch_index)
    if mode == "production":
        manifest_path = output_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("production requires prepared root run_manifest.json")
        validate_production_run_manifest(
            json.loads(manifest_path.read_text()), program_path=program_path, cache_path=cache_path
        )
    program_sha256 = sha256_file(Path(program_path))
    cache_sha256 = cache_info["sha256"]
    row_identities: dict[int, dict[str, Any]] = {}
    already_complete: list[dict[str, Any]] = []
    for index_value in indices:
        index = int(index_value)
        key = np.asarray(schedule["keys"][index], dtype=np.int64)
        identity = {
            "schedule_index": index, "schedule_sha256": schedule_sha,
            "program_sha256": program_sha256, "covariance_cache_sha256": cache_sha256,
            "parent_seed": int(schedule["parent_seed"][index]),
            "group_id": int(schedule["group_id"][index]), "geometry_key": key.tolist(),
            "fine_field_seed": int(schedule["fine_field_seed"][index]),
            "likelihood_noise_seed": int(schedule["likelihood_noise_seed"][index]),
        }
        row_identities[index] = identity
        row_dir = _output_row(output_root, index)
        if row_dir.exists():
            if not _valid_completed_row(row_dir, identity=identity):
                raise RuntimeError(f"canonical row {row_dir} is incomplete or does not verify; refusing replay")
            saved = json.loads((row_dir / "result.json").read_text())
            already_complete.append({
                "schedule_index": index, "resumed": True,
                "highk_numerical_pass": bool(saved.get("highk_numerical_pass")),
            })
    if len(already_complete) == len(indices):
        if mode == "pilot" and not all(row["highk_numerical_pass"] for row in already_complete):
            raise RuntimeError("resumed technical pilot contains a failed high-k numerical gate")
        return already_complete
    hard_p2 = load_config(pins["hard_p2_config"][0])
    if hard_p2.get("frozen_before_high_resolution_forwarding") is not True:
        raise ValueError("hard P2 configuration is not frozen")
    z0_program = json.loads(pins["z0_likelihood_source"][0].read_text())
    if not {"candidate_preselection", "z0_likelihood"}.issubset(z0_program):
        raise ValueError("z0 likelihood program lacks score_catalog inputs")
    filter_rfft = np.load(pins["density_filter"][0], allow_pickle=False)
    filter_full = full_spectrum_from_rfft(filter_rfft)
    del filter_rfft
    fixed = program["fixed_model"]
    screen = hard_p2["screen"]
    alignment = (
        int(fixed["fine_mesh"]) == int(screen["mesh_size"])
        and np.isclose(float(fixed["box_size_mpc_h"]), float(screen["box_size_mpc_h"]))
        and np.isclose(
            float(fixed["particle_spacing_mpc_h"]),
            float(screen["particle_spacing_mpc_h"]),
        )
        and np.isclose(
            float(fixed["FoF"]["central_half_width_mpc_h"]),
            float(screen["central_half_width_mpc_h"]),
        )
    )
    if not alignment:
        raise ValueError("fixed PM model and hard-P2 screen geometry differ")
    if filter_full.shape != (int(fixed["fine_mesh"]),) * 3:
        raise ValueError("density filter fine mesh changed")
    parent_entries = _parent_entries(pins["parent_manifest"][0])
    # Compile one PM forward per invocation.  Parent fields are checked against
    # the first selected row before compilation, then each row checks its own
    # file hash and cosmology again.
    _, cosmology = _load_parent(parent_entries[int(schedule["parent_seed"][indices[0]])], int(schedule["parent_seed"][indices[0]]))
    if cosmology["mesh"] != int(fixed["coarse_mesh"]) or not np.isclose(cosmology["box_size"], float(fixed["box_size_mpc_h"])):
        raise ValueError("parent mesh or box size differs from frozen model")
    import jax.numpy as jnp
    from mock_pipeline import make_forward
    _, _, forward = make_forward(
        int(fixed["fine_mesh"]), float(fixed["particle_spacing_mpc_h"]), jnp.float32,
        return_dens=False, cosmology={name: cosmology[name] for name in ("Om", "Ob", "h", "A_s_1e9", "ns")},
        return_particle_arrays=True,
    )
    results = []
    for index in indices:
        index = int(index)
        results.append(_run_row(
            schedule_index=index, schedule=schedule, parent_entries=parent_entries,
            filter_full=filter_full, cache_path=cache_path, hard_p2=hard_p2,
            z0_program=z0_program, fixed=fixed, forward=forward, cosmology=cosmology,
            output_root=output_root, row_identity=row_identities[index], program=program,
        ))
    if mode == "pilot" and any(not row.get("highk_numerical_pass", False) for row in results):
        raise RuntimeError("technical pilot completed but failed one or more high-k numerical gates")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "production"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--covariance-cache", type=Path)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--prepare-production", action="store_true")
    args = parser.parse_args()
    if args.prepare_production:
        if args.mode != "production":
            parser.error("--prepare-production requires --mode production")
        cache = args.covariance_cache
        if cache is None:
            program, _ = load_program(args.program)
            cache = _resolve_path(Path(program["covariance_cache"]["artifact_root"]) / program["covariance_cache"]["cache_file"])
        result = prepare_production_run(program_path=args.program, output_root=args.output_root, cache_path=cache)
    else:
        result = run_streaming_forward(
            program_path=args.program, mode=args.mode, output_root=args.output_root,
            covariance_cache=args.covariance_cache, batch_index=args.batch_index,
        )
    if isinstance(result, list):
        print(json.dumps({
            "mode": args.mode, "row_count": len(result),
            "resumed_count": sum(bool(row.get("resumed")) for row in result),
            "indices": [row["schedule_index"] for row in result],
        }, sort_keys=True))
    else:
        print(json.dumps({"status": result["status"], "batch_count": result["batch_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
