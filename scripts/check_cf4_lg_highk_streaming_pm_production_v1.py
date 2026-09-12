#!/usr/bin/env python3
"""Read-only identity and hash audit for the frozen 256-row PM production."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from cf4_lg_highk_covariance_cache import sha256_file, validate_covariance_cache
from cf4_lg_highk_streaming_forward import (
    PRODUCTION_BATCHES,
    ROWS_PER_BATCH,
    _load_schedule,
    _highk_gates,
    _pinned_input,
    _valid_completed_row,
    load_program,
    validate_production_run_manifest,
    validate_program_inputs,
)
from cf4_lg_z0_likelihood import logmeanexp


DEFAULT_REPO = Path("/home/kjhan/BACKUP/CF4")
DEFAULT_PROGRAM = DEFAULT_REPO / "config/cf4_lg_highk_streaming_forward_program_v1.json"
DEFAULT_CACHE = Path("/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_covariance_cache_v1/cache.npz")
DEFAULT_OUTPUT = Path("/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_production_v1")
HALO_ENTRY_NAMES = {
    "halo_pos", "halo_vel", "halo_mass", "particle_mass", "box_size",
}
STAGING_NAME = re.compile(r"\.row_([0-9]{3})\.[1-9][0-9]*\.[0-9a-f]{32}\.staging")
SHA256_LOWER_HEX = re.compile(r"[0-9a-f]{64}")
RESULT_KEYS = {
    "schema", "status", "schedule_index", "schedule_sha256",
    "program_sha256", "covariance_cache_sha256", "parent_seed", "group_id",
    "geometry_key", "fine_field_seed", "likelihood_noise_seed",
    "posterior_weight", "field_sha256", "halo_catalogue",
    "halo_catalogue_sha256", "field_persisted",
    "full_pm_particle_state_persisted", "parent_centered_P1_evaluated",
    "field_diagnostics", "highk_numerical_gates", "highk_numerical_pass",
    "n_central_particles", "n_halos", "hard_p2_pairs", "hard_p2_pass",
    "z0_likelihood", "seconds",
}
FIELD_DIAGNOSTIC_KEYS = {
    "fine_seed", "noise_seed", "predicted_before", "achieved_after", "targets",
    "sigma", "mock_noise", "weights", "coarse_roundtrip_relative_RMS",
    "correction_restriction_absolute_RMS",
    "correction_restriction_relative_RMS", "maximum_response_identity_error",
    "null_subspace_mean_square", "field_mean", "field_RMS",
    "field_imaginary_relative_RMS", "FFT_workers", "peak_evidence_reapplied",
}
HIGHK_GATE_KEYS = {
    "coarse_roundtrip", "correction_in_null_space", "response_identity",
    "null_power", "global_mean", "field_imaginary",
    "peak_evidence_not_reapplied",
}
HARD_PAIR_KEYS = {
    "halo_i", "halo_j", "m1_fof_msun_h", "m2_fof_msun_h", "mass_ratio",
    "separation_mpc_h", "midpoint_mpc_h", "midpoint_offset_mpc_h",
    "isolation_mpc_h", "peculiar_radial_velocity_km_s",
    "total_radial_velocity_km_s", "tangential_velocity_km_s",
    "m33_candidate", "ranking_score",
}
M33_KEYS = {
    "halo_index", "host_index", "mass_fof_msun_h", "host_separation_mpc_h",
}
Z0_KEYS = {"n_candidate_pairs", "log_likelihood", "best_pair", "candidate_pairs"}
Z0_PAIR_KEYS = {
    "halo_i", "halo_j", "masses_msun_h", "mass_ratio", "separation_mpc_h",
    "midpoint_mpc_h", "midpoint_offset_vector_mpc_h", "midpoint_offset_mpc_h",
    "isolation_mpc_h", "peculiar_radial_velocity_km_s",
    "total_radial_velocity_km_s", "tangential_velocity_km_s",
    "log_likelihood", "log_likelihood_components",
}
Z0_COMPONENT_KEYS = {
    "member_log10_mass", "separation", "midpoint", "total_radial_velocity",
    "tangential_speed", "isolation",
}


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _finite_vector(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length \
        and all(_finite_number(item) for item in value)


def _matches_saved_float32(value: object, saved: float) -> bool:
    return _finite_number(value) and math.isclose(
        float(value), saved, rel_tol=1e-6, abs_tol=0.0
    )


def _validate_halo_catalogue(
    halo_path: Path, *, expected_n_halos: int
) -> dict[str, np.ndarray]:
    try:
        with np.load(halo_path, allow_pickle=False) as item:
            if set(item.files) != HALO_ENTRY_NAMES:
                raise RuntimeError(
                    f"{halo_path} has unexpected NPZ entries: {sorted(item.files)}"
                )
            arrays = {name: np.asarray(item[name]) for name in HALO_ENTRY_NAMES}
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"malformed halo catalogue {halo_path}") from error

    expected = {
        "halo_pos": (np.dtype(np.float32), (expected_n_halos, 3)),
        "halo_vel": (np.dtype(np.float32), (expected_n_halos, 3)),
        "halo_mass": (np.dtype(np.float32), (expected_n_halos,)),
        "particle_mass": (np.dtype(np.float64), ()),
        "box_size": (np.dtype(np.float64), ()),
    }
    for name, (dtype, shape) in expected.items():
        array = arrays[name]
        if array.dtype != dtype or array.shape != shape:
            raise RuntimeError(
                f"{halo_path}:{name} has dtype/shape {array.dtype}/{array.shape}; "
                f"expected {dtype}/{shape}"
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{halo_path}:{name} contains nonfinite values")
    return arrays


def _validate_field_diagnostics(
    result: dict[str, object], *, identity: dict[str, object], program: dict[str, object]
) -> None:
    diagnostics = result.get("field_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != FIELD_DIAGNOSTIC_KEYS:
        raise RuntimeError("field_diagnostics does not have the frozen writer keyset")
    if type(diagnostics["fine_seed"]) is not int \
            or diagnostics["fine_seed"] != identity["fine_field_seed"] \
            or type(diagnostics["noise_seed"]) is not int \
            or diagnostics["noise_seed"] != identity["likelihood_noise_seed"]:
        raise RuntimeError("field diagnostic seeds do not match the schedule row")
    constraint_count = int(program["fixed_model"]["constraint_count"])
    for name in ("predicted_before", "achieved_after", "targets", "mock_noise", "weights"):
        if not _finite_vector(diagnostics[name], constraint_count):
            raise RuntimeError(f"field_diagnostics.{name} has invalid values")
    sigma = diagnostics["sigma"]
    if not _finite_vector(sigma, constraint_count) or any(value <= 0.0 for value in sigma):
        raise RuntimeError("field_diagnostics.sigma has invalid values")
    for name in (
        "coarse_roundtrip_relative_RMS", "correction_restriction_absolute_RMS",
        "correction_restriction_relative_RMS", "maximum_response_identity_error",
        "null_subspace_mean_square", "field_mean", "field_RMS",
        "field_imaginary_relative_RMS",
    ):
        if not _finite_number(diagnostics[name]):
            raise RuntimeError(f"field_diagnostics.{name} is not finite")
    if type(diagnostics["FFT_workers"]) is not int \
            or diagnostics["FFT_workers"] != int(program["fixed_model"]["FFT_workers"]):
        raise RuntimeError("field_diagnostics.FFT_workers changed")
    if diagnostics["peak_evidence_reapplied"] is not False:
        raise RuntimeError("field diagnostics report peak-evidence reapplication")
    stored_gates = result.get("highk_numerical_gates")
    if not isinstance(stored_gates, dict) or set(stored_gates) != HIGHK_GATE_KEYS \
            or any(type(value) is not bool for value in stored_gates.values()):
        raise RuntimeError("highk_numerical_gates does not have the frozen keyset")
    recomputed = _highk_gates(diagnostics, program)
    if stored_gates != recomputed \
            or result.get("highk_numerical_pass") is not all(recomputed.values()):
        raise RuntimeError("stored high-k gates/pass differ from recomputed gates")
    if not all(recomputed.values()):
        raise RuntimeError("canonical row did not pass all recomputed high-k gates")


def _validate_pair_identity(pair: dict[str, object], *, n_halos: int) -> tuple[int, int]:
    i, j = pair.get("halo_i"), pair.get("halo_j")
    if type(i) is not int or type(j) is not int or not (0 <= i < j < n_halos):
        raise RuntimeError("pair has invalid or noncanonical halo identity")
    return i, j


def _validate_hard_pairs(
    result: dict[str, object], *, halos: dict[str, np.ndarray], n_halos: int
) -> None:
    pairs = result.get("hard_p2_pairs")
    if not isinstance(pairs, list) or result.get("hard_p2_pass") is not bool(pairs):
        raise RuntimeError("hard-P2 pass flag and pair list differ")
    identities: set[tuple[int, int]] = set()
    previous_score = -math.inf
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != HARD_PAIR_KEYS:
            raise RuntimeError("hard-P2 pair does not have the frozen keyset")
        identity = _validate_pair_identity(pair, n_halos=n_halos)
        if identity in identities:
            raise RuntimeError("hard-P2 pair identity is duplicated")
        identities.add(identity)
        i, j = identity
        for name in (
            "m1_fof_msun_h", "m2_fof_msun_h", "mass_ratio",
            "separation_mpc_h", "midpoint_offset_mpc_h", "isolation_mpc_h",
            "peculiar_radial_velocity_km_s", "total_radial_velocity_km_s",
            "tangential_velocity_km_s", "ranking_score",
        ):
            if not _finite_number(pair[name]):
                raise RuntimeError(f"hard-P2 pair {name} is not finite")
        if not _finite_vector(pair["midpoint_mpc_h"], 3):
            raise RuntimeError("hard-P2 midpoint is invalid")
        m1, m2 = float(halos["halo_mass"][i]), float(halos["halo_mass"][j])
        if not _matches_saved_float32(pair["m1_fof_msun_h"], m1) \
                or not _matches_saved_float32(pair["m2_fof_msun_h"], m2) \
                or m1 <= 0.0 or m2 <= 0.0 \
                or not math.isclose(pair["mass_ratio"], max(m1, m2) / min(m1, m2), rel_tol=1e-6, abs_tol=0.0):
            raise RuntimeError("hard-P2 pair masses do not match halo identity")
        if not math.isclose(
            pair["total_radial_velocity_km_s"],
            pair["peculiar_radial_velocity_km_s"] + 100.0 * pair["separation_mpc_h"],
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise RuntimeError("hard-P2 radial velocities are inconsistent")
        score = pair["ranking_score"]
        if score < previous_score:
            raise RuntimeError("hard-P2 pairs are not ranking-score sorted")
        previous_score = score
        third = pair["m33_candidate"]
        if third is not None:
            if not isinstance(third, dict) or set(third) != M33_KEYS:
                raise RuntimeError("M33 candidate does not have the frozen keyset")
            k, host = third["halo_index"], third["host_index"]
            if type(k) is not int or type(host) is not int \
                    or not 0 <= k < n_halos or k in identity or host not in identity:
                raise RuntimeError("M33 candidate has invalid halo identity")
            if not _finite_number(third["mass_fof_msun_h"]) \
                    or not _finite_number(third["host_separation_mpc_h"]) \
                    or not _matches_saved_float32(
                        third["mass_fof_msun_h"], float(halos["halo_mass"][k])
                    ):
                raise RuntimeError("M33 candidate values do not match its halo")


def _validate_z0_likelihood(
    result: dict[str, object], *, halos: dict[str, np.ndarray], n_halos: int
) -> None:
    likelihood = result.get("z0_likelihood")
    if not isinstance(likelihood, dict) or set(likelihood) != Z0_KEYS:
        raise RuntimeError("z0_likelihood does not have the frozen keyset")
    pairs = likelihood["candidate_pairs"]
    count = likelihood["n_candidate_pairs"]
    if not isinstance(pairs, list) or type(count) is not int or count != len(pairs):
        raise RuntimeError("z0 candidate count and list differ")
    if likelihood["best_pair"] != (pairs[0] if pairs else None):
        raise RuntimeError("z0 best_pair and candidate ordering differ")
    identities: set[tuple[int, int]] = set()
    scores: list[float] = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != Z0_PAIR_KEYS:
            raise RuntimeError("z0 candidate pair does not have the frozen keyset")
        identity = _validate_pair_identity(pair, n_halos=n_halos)
        if identity in identities:
            raise RuntimeError("z0 candidate pair identity is duplicated")
        identities.add(identity)
        i, j = identity
        for name in (
            "mass_ratio", "separation_mpc_h", "midpoint_offset_mpc_h",
            "isolation_mpc_h", "peculiar_radial_velocity_km_s",
            "total_radial_velocity_km_s", "tangential_velocity_km_s",
            "log_likelihood",
        ):
            if not _finite_number(pair[name]):
                raise RuntimeError(f"z0 candidate {name} is not finite")
        if not _finite_vector(pair["masses_msun_h"], 2) \
                or not _finite_vector(pair["midpoint_mpc_h"], 3) \
                or not _finite_vector(pair["midpoint_offset_vector_mpc_h"], 3):
            raise RuntimeError("z0 candidate vector has invalid shape or values")
        expected_masses = sorted(
            [float(halos["halo_mass"][i]), float(halos["halo_mass"][j])], reverse=True
        )
        if not all(
            _matches_saved_float32(value, expected)
            for value, expected in zip(pair["masses_msun_h"], expected_masses)
        ) \
                or expected_masses[1] <= 0.0 \
                or not math.isclose(pair["mass_ratio"], expected_masses[0] / expected_masses[1], rel_tol=1e-6, abs_tol=0.0):
            raise RuntimeError("z0 candidate masses do not match halo identity")
        if not math.isclose(
            pair["total_radial_velocity_km_s"],
            pair["peculiar_radial_velocity_km_s"] + 100.0 * pair["separation_mpc_h"],
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise RuntimeError("z0 candidate radial velocities are inconsistent")
        components = pair["log_likelihood_components"]
        if not isinstance(components, dict) or set(components) != Z0_COMPONENT_KEYS \
                or not all(_finite_number(value) for value in components.values()) \
                or not math.isclose(pair["log_likelihood"], sum(components.values()), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("z0 likelihood components are inconsistent")
        scores.append(pair["log_likelihood"])
    if scores != sorted(scores, reverse=True):
        raise RuntimeError("z0 candidate pairs are not likelihood sorted")
    stored_mixture = likelihood["log_likelihood"]
    expected_mixture = logmeanexp(np.asarray(scores, dtype=np.float64))
    if pairs:
        if not _finite_number(stored_mixture) or not math.isclose(
            stored_mixture, expected_mixture, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("z0 catalogue log likelihood is inconsistent")
    elif type(stored_mixture) is not float or stored_mixture != -math.inf:
        raise RuntimeError("empty z0 catalogue must have negative-infinite likelihood")


def validate_canonical_rows(
    *, output_root: Path, schedule: dict[str, np.ndarray], program: dict[str, object],
    schedule_sha256: str, program_sha256: str, cache_sha256: str,
    require_complete: bool,
) -> dict[str, object]:
    """Validate exact root/row layout, identities, hashes, and halo arrays."""
    total_rows = PRODUCTION_BATCHES * ROWS_PER_BATCH
    expected_row_names = {f"row_{index:03d}" for index in range(total_rows)}
    staging: list[str] = []
    unexpected: list[str] = []
    for path in output_root.iterdir():
        if path.name == "run_manifest.json":
            if path.is_symlink() or not path.is_file():
                unexpected.append(path.name)
            continue
        if path.name in expected_row_names:
            continue
        staging_match = STAGING_NAME.fullmatch(path.name)
        if staging_match is not None and int(staging_match.group(1)) < total_rows \
                and not path.is_symlink() and path.is_dir():
            staging.append(path.name)
            continue
        unexpected.append(path.name)
    if unexpected:
        raise RuntimeError(f"unexpected production-root entries: {sorted(unexpected)}")
    staging.sort()
    if require_complete and staging:
        raise RuntimeError(f"terminal production root retains staging entries: {staging}")

    complete: list[int] = []
    missing: list[int] = []
    for index in range(total_rows):
        row_dir = output_root / f"row_{index:03d}"
        if row_dir.is_symlink():
            raise RuntimeError(f"canonical row must not be a symlink: {row_dir}")
        if not row_dir.exists():
            missing.append(index)
            continue
        if not row_dir.is_dir():
            raise RuntimeError(f"canonical row is not a directory: {row_dir}")
        entries = {path.name for path in row_dir.iterdir()}
        if entries != {"result.json", "halos.npz"}:
            raise RuntimeError(
                f"canonical row {row_dir} has unexpected entries: {sorted(entries)}"
            )
        result_path = row_dir / "result.json"
        halo_path = row_dir / "halos.npz"
        for path in (result_path, halo_path):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"canonical row entry must be a regular file: {path}")
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"malformed row result {result_path}") from error
        if not isinstance(result, dict):
            raise RuntimeError(f"row result must be a JSON object: {result_path}")
        if set(result) != RESULT_KEYS:
            raise RuntimeError(
                f"canonical row {row_dir} does not have the frozen result keyset"
            )

        key = np.asarray(schedule["keys"][index], dtype=np.int64)
        identity = {
            "schedule_index": index,
            "schedule_sha256": schedule_sha256,
            "program_sha256": program_sha256,
            "covariance_cache_sha256": cache_sha256,
            "parent_seed": int(schedule["parent_seed"][index]),
            "group_id": int(schedule["group_id"][index]),
            "geometry_key": key.tolist(),
            "fine_field_seed": int(schedule["fine_field_seed"][index]),
            "likelihood_noise_seed": int(schedule["likelihood_noise_seed"][index]),
        }
        if not _valid_completed_row(row_dir, identity=identity):
            raise RuntimeError(
                f"canonical row {row_dir} has invalid identity, status, or halo hash"
            )
        if result["posterior_weight"] != float(schedule["posterior_weight"][index]):
            raise RuntimeError(f"canonical row {row_dir} has changed posterior weight")
        if result["parent_centered_P1_evaluated"] is not False:
            raise RuntimeError(f"canonical row {row_dir} evaluated parent-centered P1")
        if type(result["field_sha256"]) is not str \
                or SHA256_LOWER_HEX.fullmatch(result["field_sha256"]) is None:
            raise RuntimeError(f"canonical row {row_dir} has invalid field SHA256")
        if result.get("field_persisted") is not False \
                or result.get("full_pm_particle_state_persisted") is not False:
            raise RuntimeError(f"canonical row {row_dir} violates streaming persistence")
        _validate_field_diagnostics(result, identity=identity, program=program)
        n_halos = result.get("n_halos")
        if type(n_halos) is not int or n_halos < 0:
            raise RuntimeError(f"canonical row {row_dir} has invalid n_halos")
        n_central = result.get("n_central_particles")
        maximum_particles = int(program["fixed_model"]["fine_mesh"]) ** 3
        if type(n_central) is not int or not n_halos <= n_central <= maximum_particles:
            raise RuntimeError(f"canonical row {row_dir} has invalid n_central_particles")
        halos = _validate_halo_catalogue(halo_path, expected_n_halos=n_halos)
        _validate_hard_pairs(result, halos=halos, n_halos=n_halos)
        _validate_z0_likelihood(result, halos=halos, n_halos=n_halos)
        if not _finite_number(result.get("seconds")) or result["seconds"] <= 0.0:
            raise RuntimeError(f"canonical row {row_dir} has invalid runtime seconds")
        complete.append(index)

    if require_complete and (len(complete) != total_rows or missing):
        raise RuntimeError(
            f"terminal production requires exactly {total_rows} rows; "
            f"found {len(complete)}"
        )
    missing_batches = sorted({index // ROWS_PER_BATCH for index in missing})
    return {
        "status": "complete" if not missing else "valid_incomplete",
        "complete_count": len(complete),
        "complete_indices": complete,
        "missing_count": len(missing),
        "missing_indices": missing,
        "missing_batches": missing_batches,
        "staging_directories": staging,
    }


def audit_production_root(
    *, program_path: Path, cache_path: Path, output_root: Path,
    require_complete: bool = False,
) -> dict[str, object]:
    """Validate every existing canonical row without creating or changing files."""
    program, _ = load_program(program_path)
    pins = validate_program_inputs(program)
    schedule_path, schedule_sha256 = pins["schedule"]
    schedule = _load_schedule(schedule_path, program)
    cache_sha256 = validate_covariance_cache(
        cache_path,
        schedule_path=schedule_path,
        filter_path=pins["density_filter"][0],
    )["sha256"]
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("production run_manifest.json is missing")
    validate_production_run_manifest(
        json.loads(manifest_path.read_text()),
        program_path=program_path,
        cache_path=cache_path,
    )

    return validate_canonical_rows(
        output_root=output_root,
        schedule=schedule,
        program=program,
        schedule_sha256=schedule_sha256,
        program_sha256=sha256_file(program_path),
        cache_sha256=cache_sha256,
        require_complete=require_complete,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--covariance-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit_production_root(
        program_path=args.program,
        cache_path=args.covariance_cache,
        output_root=args.output_root,
        require_complete=args.require_complete,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
