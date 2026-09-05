#!/usr/bin/env python3
"""Materialize the development-only N32 z=0 posterior from the frozen V6 HMC.

The V6 sampler-mechanics run deliberately stores probes only.  This module
reuses that exact PMWD, standardized model, and identity-HMC implementation,
but stores compact posterior field summaries for all eight development mocks.
It never reads the actual CF4/2M++ likelihood datum and never opens validation
seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_datum_bearing_z0_phasec_pilot as phasec
import cf4_phasec_sampler_mechanics_pilot_v6 as mechanics


SCHEMA = "ouruniv-cf4-datum-bearing-z0-phasec-posterior-v1"
TASK_SCHEMA = "ouruniv-cf4-datum-bearing-z0-phasec-posterior-task-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-datum-bearing-z0-phasec-posterior-aggregate-v1"
TASK_FILES = {"posterior_summary.npz", "diagnostics.npz", "result.json", "manifest.json", "COMPLETE"}
AGGREGATE_FILES = {"aggregate.json", "manifest.json", "COMPLETE"}
ASSIGNMENTS = [
    {"task_index": i, "mock_index": i, "seed": 2026083000 + i, "arm": "ABCD"[i // 2]}
    for i in range(8)
]


class PosteriorError(ValueError):
    """The frozen development-posterior contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def task_name(assignment: Mapping[str, object]) -> str:
    return (
        f"posterior_v1_{int(assignment['task_index']):02d}_mock_{int(assignment['mock_index']):02d}"
        f"_seed_{int(assignment['seed'])}_arm_{assignment['arm']}"
    )


def _verify_binding(binding: Mapping[str, object], label: str) -> Path:
    path = Path(str(binding["path"]))
    expected = str(binding["sha256"])
    if not path.is_file() or sha256_file(path) != expected:
        raise PosteriorError(f"{label} hash binding failed: {path}")
    return path


def _verify_gpfs_aggregate(binding: Mapping[str, object], label: str) -> dict[str, object]:
    root = _verify_binding(binding, label)
    result = json.loads(root.read_text())
    if result.get("status") != str(binding["required_status"]):
        raise PosteriorError(f"{label} status is not the frozen pass")
    return result


def load_program(path: str | Path) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    payload = Path(path).read_bytes()
    controller = json.loads(payload)
    if controller.get("schema") != SCHEMA:
        raise PosteriorError("posterior program schema mismatch")
    auth = controller.get("authorization", {})
    for key in ("development_posterior_materialization", "Slurm_GPU_array", "GPFS_read_bound_inputs", "GPFS_write_new_outputs"):
        if auth.get(key) is not True:
            raise PosteriorError(f"missing authorization: {key}")
    for key in ("actual_observational_field_inference", "actual_2Mpp_count_read", "actual_CF4_velocity_datum_used", "validation_seed_access", "IC_PM_HOP_RAMSES"):
        if auth.get(key) is not False:
            raise PosteriorError(f"forbidden scope enabled: {key}")
    if controller.get("assignments") != ASSIGNMENTS:
        raise PosteriorError("eight development assignments changed")
    source_binding = controller["source_program"]
    source_path = _verify_binding(source_binding, "V6 sampler program")
    v6_controller, source_sha, base = mechanics.load_program(source_path)
    if source_sha != str(source_binding["sha256"]):
        raise PosteriorError("V6 source program digest changed")
    expected_v6_assignments = [
        {"task_index": i, "mock_index": mock, "seed": 2026083000 + mock, "arm": "ABCD"[mock // 2]}
        for i, mock in enumerate([1, 2, 3, 4, 5, 7])
    ]
    if v6_controller.get("assignments") != expected_v6_assignments:
        raise PosteriorError("V6 mechanics assignment lineage changed")
    _verify_gpfs_aggregate(controller["generator_gate_aggregate"], "generator gate aggregate")
    _verify_gpfs_aggregate(controller["sampler_v5_aggregate"], "sampler v5 aggregate")
    _verify_gpfs_aggregate(controller["sampler_v6_aggregate"], "sampler v6 aggregate")
    effective = copy.deepcopy(v6_controller)
    effective["assignments"] = copy.deepcopy(ASSIGNMENTS)
    return controller, effective, base, hashlib.sha256(payload).hexdigest()


def _artifact_manifest(directory: Path, schema: str) -> dict[str, object]:
    files = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": schema, "files": files}


def _stream_k_shell_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    samples: np.memmap,
    transfer: np.ndarray,
    box_size: float,
    edges: np.ndarray,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Compute the field-spectrum summary without materializing all FFTs."""

    n = truth.shape[0]
    frequency = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    truth_k = np.fft.fftn(truth, norm="ortho")
    estimate_k = np.fft.fftn(estimate, norm="ortho")
    sample_count = int(samples.shape[0])
    sum_k = np.zeros((n, n, n), dtype=np.complex128)
    sum_abs2 = np.zeros((n, n, n), dtype=np.float64)
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        sample_k = np.fft.fftn(
            np.asarray(samples[start:stop], dtype=np.float64),
            axes=(1, 2, 3),
            norm="ortho",
        )
        sum_k += sample_k.sum(axis=0)
        sum_abs2 += np.sum(np.abs(sample_k) ** 2, axis=0)
    sample_mean = sum_k / sample_count
    sample_var = np.maximum(sum_abs2 / sample_count - np.abs(sample_mean) ** 2, 0.0)
    rows = {
        key: []
        for key in (
            "k_mean",
            "mode_count",
            "response",
            "cross_correlation",
            "residual_power",
            "posterior_variance_reduction",
        )
    }
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (kmag >= lo) & (kmag < hi)
        mode_count = int(np.count_nonzero(mask))
        rows["k_mean"].append(float(kmag[mask].mean()) if mode_count else np.nan)
        rows["mode_count"].append(mode_count)
        if mode_count == 0:
            for key in (
                "response",
                "cross_correlation",
                "residual_power",
                "posterior_variance_reduction",
            ):
                rows[key].append(np.nan)
            continue
        pt = float(np.mean(np.abs(truth_k[mask]) ** 2))
        pe = float(np.mean(np.abs(estimate_k[mask]) ** 2))
        cross = float(np.mean((estimate_k[mask] * np.conjugate(truth_k[mask])).real))
        residual = float(np.mean(np.abs(estimate_k[mask] - truth_k[mask]) ** 2))
        posterior_var = float(np.mean(sample_var[mask]))
        prior_var = float(np.mean(np.asarray(transfer)[mask] ** 2))
        rows["response"].append(cross / max(pt, np.finfo(float).tiny))
        rows["cross_correlation"].append(cross / max(math.sqrt(pt * pe), np.finfo(float).tiny))
        rows["residual_power"].append(residual)
        rows["posterior_variance_reduction"].append(
            1.0 - posterior_var / max(prior_var, np.finfo(float).tiny)
        )
    return {key: np.asarray(value) for key, value in rows.items()}


def _materialize_field_samples(
    draws: np.ndarray,
    model_meta: Mapping[str, object],
    workdir: Path,
    chunk_size: int = 16,
) -> tuple[np.memmap, np.memmap]:
    """Write N32 posterior fields to GPFS-backed temporary memmaps in chunks."""

    field_size = int(model_meta["field_size"])
    total = int(draws.shape[0] * draws.shape[1])
    n = phasec.fixed.N
    density_path = workdir / "density_samples.f32"
    velocity_path = workdir / "velocity_samples.f32"
    density = np.memmap(density_path, mode="w+", dtype="float32", shape=(total, n, n, n))
    velocity = np.memmap(velocity_path, mode="w+", dtype="float32", shape=(total, 3, n, n, n))
    flat_draws = draws.reshape((total, draws.shape[-1]))
    transfer = np.asarray(model_meta["transfer"])
    growth_rate = float(model_meta["growth_rate"])
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        # density_velocity_samples preserves the first two axes as
        # (chain, draw); represent this flat chunk as one synthetic chain.
        white = flat_draws[start:stop, :field_size].reshape((1, stop - start, n, n, n))
        density_chunk, velocity_chunk = phasec.density_velocity_samples(
            white, transfer, growth_rate, phasec.fixed.BOX_SIZE
        )
        density[start:stop] = density_chunk.reshape((-1, n, n, n))
        velocity[start:stop] = velocity_chunk.reshape((-1, 3, n, n, n))
    density.flush()
    velocity.flush()
    return density, velocity


def _summary_fields(
    draws: np.ndarray,
    truth: Mapping[str, object],
    model_meta: Mapping[str, object],
    base: Mapping[str, object],
    program: Mapping[str, object],
    workdir: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    field_size = int(model_meta["field_size"])
    density_flat, velocity_flat = _materialize_field_samples(draws, model_meta, workdir)
    density_mean = np.asarray(density_flat.mean(axis=0), dtype=np.float64)
    density_std = np.asarray(density_flat.std(axis=0), dtype=np.float64)
    density_quantiles = np.quantile(density_flat, [0.025, 0.16, 0.84, 0.975], axis=0).astype(np.float32)
    velocity_mean = np.asarray(velocity_flat.mean(axis=0), dtype=np.float64)
    velocity_std = np.asarray(velocity_flat.std(axis=0), dtype=np.float64)
    velocity_quantiles = np.stack(
        [np.quantile(velocity_flat[:, component], [0.025, 0.16, 0.84, 0.975], axis=0) for component in range(3)],
        axis=1,
    ).astype(np.float32)
    truth_density = np.asarray(truth["coarse_delta"], dtype=np.float64)
    truth_velocity = np.moveaxis(np.asarray(truth["coarse_velocity"], dtype=np.float64), -1, 0)
    roi_names, roi_weights, roi_effective = phasec.build_roi_weights(base)
    unit_weight = np.ones((phasec.fixed.N,) * 3, dtype=np.float64)
    density_coverage = {
        "global_68": phasec.weighted_coverage(truth_density, density_quantiles[1], density_quantiles[2], unit_weight),
        "global_95": phasec.weighted_coverage(truth_density, density_quantiles[0], density_quantiles[3], unit_weight),
        "ROI_68": {}, "ROI_95": {},
    }
    velocity_coverage = {"global_68": [], "global_95": [], "ROI_68": {}, "ROI_95": {}}
    for component in range(3):
        velocity_coverage["global_68"].append(phasec.weighted_coverage(truth_velocity[component], velocity_quantiles[1, component], velocity_quantiles[2, component], unit_weight))
        velocity_coverage["global_95"].append(phasec.weighted_coverage(truth_velocity[component], velocity_quantiles[0, component], velocity_quantiles[3, component], unit_weight))
    for index, name in enumerate(roi_names):
        weight = roi_weights[index]
        density_coverage["ROI_68"][name] = phasec.weighted_coverage(truth_density, density_quantiles[1], density_quantiles[2], weight)
        density_coverage["ROI_95"][name] = phasec.weighted_coverage(truth_density, density_quantiles[0], density_quantiles[3], weight)
        velocity_coverage["ROI_68"][name] = [phasec.weighted_coverage(truth_velocity[c], velocity_quantiles[1, c], velocity_quantiles[2, c], weight) for c in range(3)]
        velocity_coverage["ROI_95"][name] = [phasec.weighted_coverage(truth_velocity[c], velocity_quantiles[0, c], velocity_quantiles[3, c], weight) for c in range(3)]
    k_edges = np.asarray(program["diagnostics"]["k_edges_h_Mpc"], dtype=np.float64)
    shell = _stream_k_shell_metrics(
        truth_density,
        density_mean,
        density_flat,
        np.asarray(model_meta["transfer"]),
        phasec.fixed.BOX_SIZE,
        k_edges,
        chunk_size=16,
    )
    frequency = 2.0 * np.pi * np.fft.fftfreq(phasec.fixed.N, d=phasec.fixed.BOX_SIZE / phasec.fixed.N)
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    safe_k2 = np.where(k2 > 0.0, k2, 1.0)
    velocity_shells = []
    for component, kval in enumerate((kx, ky, kz)):
        transfer = np.asarray(model_meta["transfer"]) * 100.0 * float(model_meta["growth_rate"]) * np.abs(kval) / safe_k2
        transfer[0, 0, 0] = 0.0
        velocity_shells.append(
            _stream_k_shell_metrics(
                truth_velocity[component],
                velocity_mean[component],
                velocity_flat[:, component],
                transfer,
                phasec.fixed.BOX_SIZE,
                k_edges,
                chunk_size=16,
            )
        )
    summary = {
        "coverage": {"density": density_coverage, "velocity_components": velocity_coverage},
        "k_shells": {key: value.tolist() for key, value in shell.items()},
        "velocity_k_shells_by_component": [{key: value.tolist() for key, value in part.items()} for part in velocity_shells],
        "ROI": {"names": roi_names, "N32_effective_weighted_cell_count": roi_effective.tolist(), "underresolved_at_N32": True},
    }
    arrays = {
        "truth_coarse_density": truth_density.astype(np.float32),
        "truth_coarse_velocity": truth_velocity.astype(np.float32),
        "posterior_density_mean": density_mean.astype(np.float32),
        "posterior_density_std": density_std.astype(np.float32),
        "posterior_density_quantiles": density_quantiles,
        "posterior_velocity_mean": velocity_mean.astype(np.float32),
        "posterior_velocity_std": velocity_std.astype(np.float32),
        "posterior_velocity_quantiles": velocity_quantiles,
        "posterior_white_thinned": white[:, :: int(program["diagnostics"]["stored_white_thinning"]), :].astype(np.float32),
        "roi_names": np.asarray(roi_names),
        "roi_weights": roi_weights.astype(np.float32),
    }
    return summary, arrays


def run_task(program_path: str | Path, output_root: str | Path, task_index: int, implementation_commit: str, device_record_path: str | Path) -> None:
    controller, effective, base, program_sha = load_program(program_path)
    if not (0 <= task_index < 8) or len(implementation_commit) != 40:
        raise PosteriorError("invalid task index or implementation commit")
    assignment = effective["assignments"][task_index]
    output = Path(output_root) / task_name(assignment)
    staging = output.parent / f".{output.name}.staging"
    workdir = output.parent / f".{output.name}.work"
    if output.exists() or staging.exists() or workdir.exists():
        raise PosteriorError("posterior output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(mode=0o700)
    import jax
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise PosteriorError("posterior task requires one allocated GPU")
    device = mechanics.load_device_record(device_record_path, effective)
    response6, response4 = mechanics.v1.phasec_v5._load_selection(base)
    nbar, bias = mechanics.v1.phasec_v5._published_prior_arrays(base)
    args = mechanics.v1.phasec_v5.fixed.frozen_args(base["input_bindings"]["CF4_catalog"]["path"])
    design = mechanics.v1.linear.prepare_catalog(args)
    seed = int(assignment["seed"])
    fine_white, _coarse_white, nesting = mechanics.v1.phasec_v5.nested_white_fields(seed, int(base["grid"]["inference_N"]), int(base["grid"]["truth_N"]), int(effective["rng_tags"]["high_k_white"]))
    truth = mechanics.v1.build_scan_truth(fine_white, effective, base)
    truth_intensity, stress_meta = mechanics.v1.phasec_v5.truth_intensity(str(assignment["arm"]), truth, response6, response4, nbar, bias, base, seed)
    mock = mechanics.v1.phasec_v5.generate_mock_data(str(assignment["arm"]), seed, truth_intensity, truth, design, base)
    nlp, count_lambda, initial, model_meta = mechanics.v1.build_standardized_model(base, response6, mock, design)
    draws, sampler = mechanics.run_identity_hmc(nlp, initial, effective, seed)
    derived, derived_arrays = mechanics.v1.derived_intensity_audit(draws, count_lambda, float(effective["gates"]["derived_count_intensity_max"]))
    derived["expected_retained_draw_count"] = int(draws.shape[0] * draws.shape[1])
    model_meta["sampler_logdensity"] = sampler["logdensity"]
    convergence, projection_arrays = mechanics.v1.convergence_projections(draws, model_meta, base, effective)
    field_summary, posterior_arrays = _summary_fields(draws, truth, model_meta, base, base, workdir)
    predictive_summary, predictive_arrays = mechanics.v1.phasec_v5.predictive_diagnostics(draws, count_lambda, mock, response6, design, model_meta, base, seed)
    sampler_config = effective["mechanics"]["sampler"]
    sampling_divergence = float(np.mean(sampler["is_divergent"]))
    warmup_divergence = float(np.sum(sampler["warmup_divergence_count"]) / (draws.shape[0] * int(sampler_config["warmup_steps"])))
    checks = {
        "MAP_optimizer_success": bool(sampler["MAP_success"]),
        "MAP_gradient_RMS": float(sampler["MAP_gradient_RMS"]) <= float(effective["gates"]["MAP_gradient_RMS_max"]),
        "all_draws_finite": bool(np.all(np.isfinite(draws))),
        "all_sampling_energies_finite": bool(np.all(np.isfinite(sampler["energy"]))),
        "all_warmup_energies_finite": bool(np.all(sampler["warmup_energies_finite"])),
        "derived_intensity_gate": bool(derived["pass"]),
        "Rhat": float(convergence["max_Rhat"]) <= float(effective["gates"]["rank_normalized_split_Rhat_max"]),
        "bulk_ESS": float(convergence["min_bulk_ESS"]) >= float(effective["gates"]["bulk_ESS_min"]),
        "tail_ESS": float(convergence["min_tail_ESS"]) >= float(effective["gates"]["tail_ESS_min"]),
        "divergence_fraction": sampling_divergence <= float(effective["gates"]["divergence_fraction_max"]),
    }
    pilot_pass = bool(all(checks.values()))
    result = {
        "schema": TASK_SCHEMA,
        "status": "PASS_Z0_DEVELOPMENT_POSTERIOR_TASK" if pilot_pass else "NO_GO_Z0_DEVELOPMENT_POSTERIOR_TASK",
        "assignment": assignment,
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__), "commit": implementation_commit},
        "environment": {"jax_backend": jax.default_backend(), "jax_devices": [str(d) for d in jax.devices()], "jax_version": jax.__version__, "physical_GPU_record": device},
        "truth": {"nested_white": nesting, "a_nbody_maxstep": float(effective["truth_integrator"]["a_nbody_maxstep"]), "a_nbody_step_count": int(truth["a_nbody_step_count"]), "density_min": float(truth["density_min"]), "density_max": float(truth["density_max"]), "stress": stress_meta},
        "mock": {"counts_train_total": int(np.sum(mock["counts_train"])), "counts_holdout_total": int(np.sum(mock["counts_holdout"])), "CF4_geometry_row_count": int(np.asarray(design["pos"]).shape[0]), "actual_2Mpp_counts_read": False, "actual_CF4_velocity_datum_used": False, "validation_seed_read": False},
        "sampler": {"latent_dimension": int(draws.shape[-1]), "chain_count": int(draws.shape[0]), "draws_per_chain": int(draws.shape[1]), "MAP_value": float(sampler["MAP_value"]), "MAP_gradient_RMS": float(sampler["MAP_gradient_RMS"]), "MAP_iterations": int(sampler["MAP_iterations"]), "sampling_mean_acceptance": float(np.mean(sampler["acceptance_rate"])), "sampling_divergence_fraction": sampling_divergence, "warmup_divergence_fraction": warmup_divergence, "convergence": convergence},
        "derived_count_intensity": derived,
        "predictive": predictive_summary,
        **field_summary,
        "checks": checks,
        "pilot_pass": pilot_pass,
        "semantics": {"mock_only": True, "development_calibration_not_validation": True, "full_N32_field_summary_stored": True, "actual_present_day_posterior_created": False, "observational_resolution_or_frontier_claim_created": False, "target_0p3_cMpc_h_reached": False, "automatic_Phase_D_allowed": False},
    }
    staging.mkdir(mode=0o700)
    np.savez_compressed(staging / "posterior_summary.npz", **posterior_arrays)
    np.savez_compressed(staging / "diagnostics.npz", **projection_arrays, **derived_arrays, **predictive_arrays, sampler_acceptance_rate=np.asarray(sampler["acceptance_rate"]), sampler_is_divergent=np.asarray(sampler["is_divergent"]), sampler_energy=np.asarray(sampler["energy"]), sampler_logdensity=np.asarray(sampler["logdensity"]))
    (staging / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (staging / "manifest.json").write_text(json.dumps(_artifact_manifest(staging, "ouruniv-cf4-datum-bearing-z0-phasec-posterior-task-manifest-v1"), indent=2, sort_keys=True) + "\n")
    (staging / "COMPLETE").write_text(json.dumps({"schema": "ouruniv-cf4-datum-bearing-z0-phasec-posterior-task-complete-v1", "result_sha256": sha256_file(staging / "result.json"), "manifest_sha256": sha256_file(staging / "manifest.json"), "pilot_pass": pilot_pass}, sort_keys=True) + "\n")
    del posterior_arrays
    shutil.rmtree(workdir)
    os.replace(staging, output)


def validate_task(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {p.name for p in root.iterdir()} != TASK_FILES:
        raise PosteriorError("posterior task artifact set mismatch")
    result = json.loads((root / "result.json").read_text())
    complete = json.loads((root / "COMPLETE").read_text())
    if result.get("schema") != TASK_SCHEMA or complete.get("result_sha256") != sha256_file(root / "result.json") or complete.get("manifest_sha256") != sha256_file(root / "manifest.json") or complete.get("pilot_pass") != result.get("pilot_pass"):
        raise PosteriorError("posterior task integrity mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "ouruniv-cf4-datum-bearing-z0-phasec-posterior-task-manifest-v1":
        raise PosteriorError("posterior task manifest schema mismatch")
    declared = {row.get("name") for row in manifest.get("files", [])}
    actual = {p.name for p in root.iterdir() if p.name not in {"manifest.json", "COMPLETE"}}
    if declared != actual:
        raise PosteriorError("posterior task manifest file set mismatch")
    for row in manifest["files"]:
        p = root / str(row["name"])
        if p.stat().st_size != int(row["bytes"]) or sha256_file(p) != row["sha256"]:
            raise PosteriorError(f"posterior task artifact hash mismatch: {p.name}")
    return result


def aggregate(program_path: str | Path, output_root: str | Path, aggregate_output: str | Path, implementation_commit: str) -> None:
    controller, effective, _base, program_sha = load_program(program_path)
    if len(implementation_commit) != 40:
        raise PosteriorError("aggregate commit must be full hash")
    output = Path(aggregate_output)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise PosteriorError("posterior aggregate output already exists")
    outcomes = []
    for assignment in effective["assignments"]:
        task_dir = Path(output_root) / task_name(assignment)
        try:
            result = validate_task(task_dir)
            if result.get("assignment") != assignment or result.get("program", {}).get("sha256") != program_sha or result.get("implementation", {}).get("commit") != implementation_commit:
                raise PosteriorError("posterior task lineage mismatch")
            outcomes.append({"assignment": assignment, "artifact_status": "VALID", "pilot_pass": bool(result["pilot_pass"]), "result_sha256": sha256_file(task_dir / "result.json"), "coverage": result.get("coverage"), "checks": result.get("checks"), "sampler": result.get("sampler"), "predictive": result.get("predictive")})
        except Exception as exc:
            outcomes.append({"assignment": assignment, "artifact_status": "MISSING_OR_INVALID", "pilot_pass": False, "error": f"{type(exc).__name__}: {exc}"})
    all_pass = bool(len(outcomes) == 8 and all(row["pilot_pass"] for row in outcomes))
    global68 = [row["coverage"]["density"]["global_68"] for row in outcomes if row["artifact_status"] == "VALID" and row.get("coverage")]
    global95 = [row["coverage"]["density"]["global_95"] for row in outcomes if row["artifact_status"] == "VALID" and row.get("coverage")]
    result = {"schema": AGGREGATE_SCHEMA, "status": "PASS_BALANCED_DEVELOPMENT_Z0_POSTERIOR" if all_pass else "NO_GO_BALANCED_DEVELOPMENT_Z0_POSTERIOR", "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha}, "implementation_commit": implementation_commit, "task_count": len(outcomes), "valid_artifact_count": sum(row["artifact_status"] == "VALID" for row in outcomes), "passing_task_count": sum(row["pilot_pass"] for row in outcomes), "balanced_mock_count": 8, "outcomes": outcomes, "coverage_summary": {"density_global_68_mean": float(np.mean(global68)) if global68 else None, "density_global_68_min": float(np.min(global68)) if global68 else None, "density_global_95_mean": float(np.mean(global95)) if global95 else None, "density_global_95_min": float(np.min(global95)) if global95 else None}, "scientific_disposition": {"development_z0_posterior": "AVAILABLE_FOR_CALIBRATION" if all_pass else "NOT_AVAILABLE", "actual_CF4_posterior": "NOT_CREATED", "parent_posterior_promotion": "NO_GO", "IC_PM_HOP_RAMSES": "NOT_RUN", "observational_0p3_cMpc_h": "NOT_ALLOWED"}, "scope_firewall": {"actual_observational_field_inference": False, "actual_2Mpp_count_read": False, "actual_CF4_velocity_datum_used": False, "validation_seed_read": False, "IC_PM_HOP_RAMSES": False}}
    staging.mkdir(mode=0o700)
    (staging / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (staging / "manifest.json").write_text(json.dumps(_artifact_manifest(staging, "ouruniv-cf4-datum-bearing-z0-phasec-posterior-aggregate-manifest-v1"), indent=2, sort_keys=True) + "\n")
    (staging / "COMPLETE").write_text(json.dumps({"schema": "ouruniv-cf4-datum-bearing-z0-phasec-posterior-aggregate-complete-v1", "aggregate_sha256": sha256_file(staging / "aggregate.json"), "manifest_sha256": sha256_file(staging / "manifest.json"), "balanced_mock_count": 8, "all_tasks_pass": all_pass}, sort_keys=True) + "\n")
    os.replace(staging, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--program", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--implementation-commit", required=True)
    run.add_argument("--device-record", required=True)
    check = sub.add_parser("validate-task")
    check.add_argument("--directory", required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--program", required=True)
    agg.add_argument("--output-root", required=True)
    agg.add_argument("--aggregate-output", required=True)
    agg.add_argument("--implementation-commit", required=True)
    agg_check = sub.add_parser("validate-aggregate")
    agg_check.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_task(args.program, args.output_root, args.task_index, args.implementation_commit, args.device_record)
    elif args.command == "validate-task":
        validate_task(args.directory)
    elif args.command == "aggregate":
        aggregate(args.program, args.output_root, args.aggregate_output, args.implementation_commit)
    else:
        root = Path(args.directory)
        if not root.is_dir() or {p.name for p in root.iterdir()} != AGGREGATE_FILES:
            raise PosteriorError("posterior aggregate artifact set mismatch")
        result = json.loads((root / "aggregate.json").read_text())
        complete = json.loads((root / "COMPLETE").read_text())
        if complete.get("aggregate_sha256") != sha256_file(root / "aggregate.json") or complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
            raise PosteriorError("posterior aggregate hash mismatch")
        if result.get("schema") != AGGREGATE_SCHEMA:
            raise PosteriorError("posterior aggregate schema mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
