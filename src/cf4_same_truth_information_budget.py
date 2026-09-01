#!/usr/bin/env python3
"""Posterior-information budget for the failed CF4 same-truth arm A.

This covariance-only audit never generates or deserializes a truth array and
never consumes a likelihood datum in inference.  It reuses the completed arm-A posterior draws for the marginalized
baseline and computes three known-nuisance covariance scenarios on the same
true-position geometry and D-selected training rows, at finite measurement
standard-deviation scales 1, 0.3, and 0.1.  Per-bin recovered information is

    I_bin = 1 - Tr(P_post,bin) / Tr(P_prior,bin).

For the exact linear-Gaussian model this is also the expected posterior-mean
response, while the posterior trace fraction is the expected residual-power
ratio and sqrt(I_bin) is the expected field correlation.  The 0.1 case is a
finite low-noise ceiling, not a zero-noise geometry theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_population_calibration as base
import cf4_population_calibration_aggregate_v2 as corrected
import cf4_same_truth_ablation as prior
from cf4_kf_bin_manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_SCHEMA = "ouruniv-cf4-same-truth-information-budget-program-v1"
MEMBER_SCHEMA = "ouruniv-cf4-same-truth-information-budget-member-result-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-same-truth-information-budget-result-v1"
MEMBER_STATUS = "COMPLETE_COVARIANCE_ONLY_INFORMATION_MEMBER_NO_SCIENCE_CLAIM"
AGGREGATE_STATUS = "COMPLETE_COVARIANCE_ONLY_INFORMATION_BUDGET_NO_SCIENCE_CLAIM"
SCENARIOS = (
    "marginalized_s1",
    "known_s1",
    "known_s0p3",
    "known_s0p1",
)
NEW_SCENARIOS = ("known_s1", "known_s0p3", "known_s0p1")
NOISE_SCALES = {"known_s1": 1.0, "known_s0p3": 0.3, "known_s0p1": 0.1}
EXPECTED_MEMBER_FILES = {"fields.npz", "result.json", "manifest.json", "COMPLETE"}
EXPECTED_AGGREGATE_FILES = {"metrics.npz", "result.json", "manifest.json", "COMPLETE"}


class InformationError(ValueError):
    """The frozen posterior-information audit contract failed closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise InformationError("repository binding escapes the repository")
    return path


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InformationError("cannot parse information-budget program") from exc
    if program.get("schema") != PROGRAM_SCHEMA:
        raise InformationError("information-budget program schema mismatch")
    authorization = program.get("authorization", {})
    required_true = {
        "same_truth_information_budget_audit",
        "Slurm_member_array_submission",
        "Slurm_dependent_aggregation_submission",
        "GPFS_declared_read_write",
    }
    required_false = {
        "truth_array_generation_or_deserialization",
        "likelihood_datum_consumed_by_inference",
        "population_generator_retuning",
        "untouched_256_mock_validation",
        "resolution_increase",
        "ML_training",
        "frontier_promotion",
        "IC_PM_HOP_RAMSES",
        "automatic_retry",
        "automatic_follow_on_after_aggregate",
    }
    if any(authorization.get(key) is not True for key in required_true):
        raise InformationError("required information-budget authorization is absent")
    if any(authorization.get(key) is not False for key in required_false):
        raise InformationError("information-budget science firewall changed")
    design = program.get("design", {})
    if design.get("mock_count") != base.MOCK_COUNT:
        raise InformationError("information-budget mock count changed")
    if design.get("posterior_draw_count") != base.POSTERIOR_DRAW_COUNT:
        raise InformationError("information-budget posterior draw count changed")
    if tuple(design.get("scenario_order", ())) != SCENARIOS:
        raise InformationError("information-budget scenario order changed")
    if design.get("new_truth_seed_count") != 0 or design.get("new_random_seed_count") != 0:
        raise InformationError("information-budget introduced a new random stream")
    configured_scales = design.get("known_nuisance_noise_standard_deviation_scales", {})
    if configured_scales != NOISE_SCALES:
        raise InformationError("known-nuisance finite noise scales changed")
    for collection in ("repository_bindings", "source_bindings"):
        records = program.get(collection, {})
        if not isinstance(records, Mapping) or not records:
            raise InformationError(f"{collection} is absent")
        for record in records.values():
            if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
                raise InformationError(f"{collection} record is not exact")
            source = _repo_path(str(record["path"]))
            if sha256_file(source) != record["sha256"]:
                raise InformationError(f"SHA256 mismatch: {record['path']}")
    for label in ("prior_ablation", "completed_D"):
        completed = program.get(label, {})
        required = {
            "members_root",
            "aggregate_directory",
            "aggregate_result_sha256",
            "aggregate_metrics_sha256",
        }
        if label == "prior_ablation":
            required.add("member_implementation_commit")
        if set(completed) != required:
            raise InformationError(f"{label} binding set is not exact")
        aggregate = Path(str(completed["aggregate_directory"]))
        if sha256_file(aggregate / "result.json") != completed["aggregate_result_sha256"]:
            raise InformationError(f"{label} aggregate result binding changed")
        if sha256_file(aggregate / "metrics.npz") != completed["aggregate_metrics_sha256"]:
            raise InformationError(f"{label} aggregate metrics binding changed")
    resolution = program.get("resolution_semantics", {})
    if resolution.get("cell_size_cMpc_h") != base.fixed.BOX_SIZE / base.fixed.N:
        raise InformationError("information-budget development resolution changed")
    if resolution.get("target_0p3_cMpc_h_reached") is not False:
        raise InformationError("forbidden 0.3 cMpc/h claim")
    return program, hashlib.sha256(payload).hexdigest()


def _mode_plan() -> dict[str, np.ndarray]:
    body, _, _ = base.fixed.load_bin_manifest(ROOT / "config/cf4_kf_bin_manifest_v1.json")
    plan = base.fixed.global_merged_mode_plan(body)
    flat = np.asarray(plan["flat_independent_field_indices"], dtype=np.int64)
    assignment = np.asarray(plan["mode_merged_bin_index"], dtype=np.int64)
    grid_index = np.unravel_index(flat, (base.fixed.N,) * 3)
    theta_keep = np.logical_and.reduce([axis != base.fixed.N // 2 for axis in grid_index])
    return {
        "flat": flat,
        "assignment": assignment,
        "theta_keep": theta_keep,
        "theta_assignment": assignment[theta_keep],
    }


def solve_known_covariances(
    *,
    positions: np.ndarray,
    directions: np.ndarray,
    variance: np.ndarray,
    train: np.ndarray,
    args: argparse.Namespace,
    seeds: Mapping[str, object],
    transfer: np.ndarray,
    modes: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Sample three known-nuisance posterior covariances on one geometry."""

    import jax
    import jax.numpy as jnp

    forward_all, adjoint_all, _, dtype = base.linear.build_forward(positions, directions, args)
    train_index = np.flatnonzero(np.asarray(train, dtype=bool))
    if train_index.size == 0 or train_index.size == train.size:
        raise InformationError("information geometry lacks train or holdout rows")
    train_jax = jnp.asarray(train_index)
    A_train = jax.jit(lambda field: forward_all(field)[train_jax])

    @jax.jit
    def AT_train(values):
        expanded = jnp.zeros(train.size, dtype=dtype)
        return adjoint_all(expanded.at[train_jax].set(values))

    probe_rng = np.random.default_rng(int(seeds["preconditioner"]))
    raw_probe_response = []
    for _ in range(base.fixed.PRECONDITIONER_PROBES):
        probe = jnp.asarray(probe_rng.standard_normal((base.fixed.N,) * 3), dtype=dtype)
        raw_probe_response.append(np.asarray(A_train(probe), dtype=np.float64))
    raw_probe_response = np.stack(raw_probe_response)
    standard_deviation = np.sqrt(np.asarray(variance, dtype=np.float64)[train_index])
    if np.any(standard_deviation <= 0.0) or not np.all(np.isfinite(standard_deviation)):
        raise InformationError("information geometry noise scale is invalid")

    flat = np.asarray(modes["flat"], dtype=np.int64)
    theta_keep = np.asarray(modes["theta_keep"], dtype=bool)
    transfer_modes = np.asarray(transfer, dtype=np.float64).ravel()[flat]
    results: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    for scenario in NEW_SCENARIOS:
        noise_scale = NOISE_SCALES[scenario]
        scale = jnp.asarray(noise_scale * standard_deviation, dtype=dtype)
        An = jax.jit(lambda field: A_train(field) / scale)
        ATn = jax.jit(lambda values: AT_train(values / scale))

        @jax.jit
        def Cnorm(values):
            return values + An(ATn(values))

        adjoint_rng = np.random.default_rng(int(seeds["adjoint"]))
        sx = jnp.asarray(adjoint_rng.standard_normal((base.fixed.N,) * 3), dtype=dtype)
        dy = jnp.asarray(adjoint_rng.standard_normal(train_index.size), dtype=dtype)
        lhs = float(jnp.vdot(An(sx), dy))
        rhs = float(jnp.vdot(sx, ATn(dy)))
        adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)
        normalized_probes = raw_probe_response / (noise_scale * standard_deviation)[None, :]
        preconditioner = jnp.asarray(
            1.0 + np.mean(normalized_probes**2, axis=0), dtype=dtype
        )
        posterior_modes = []
        residuals = []
        for seed in seeds["posterior_draws"]:
            rng = np.random.default_rng(int(seed))
            xi = jnp.asarray(rng.standard_normal((base.fixed.N,) * 3), dtype=dtype)
            rng.standard_normal(4)  # Preserve the completed arm-A epsilon stream.
            epsilon = jnp.asarray(rng.standard_normal(train_index.size), dtype=dtype)
            alpha, relative, _ = base.linear.cg_solve(
                Cnorm, -An(xi) - epsilon, preconditioner, args
            )
            posterior = np.asarray(xi + ATn(alpha), dtype=np.float64)
            white_modes = np.fft.fftn(posterior, norm="ortho").ravel()[flat]
            posterior_modes.append(white_modes * transfer_modes)
            residuals.append(float(relative))
        draws_delta = np.stack(posterior_modes)
        numerical = {
            "adjoint_relative_error": adjoint_error,
            "adjoint_max_inclusive": base.fixed.ADJOINT_MAX,
            "adjoint_pass": bool(adjoint_error <= base.fixed.ADJOINT_MAX),
            "all_16_sample_cg_relative_residuals": residuals,
            "sample_cg_max_inclusive": base.fixed.CG_RESIDUAL_MAX,
        }
        numerical["all_16_sample_cg_pass"] = bool(
            np.all(np.asarray(residuals) <= base.fixed.CG_RESIDUAL_MAX)
        )
        numerical["all_pass"] = bool(
            numerical["adjoint_pass"] and numerical["all_16_sample_cg_pass"]
        )
        numerical["posterior_mean_solved"] = False
        if not numerical["all_pass"]:
            raise InformationError(f"{scenario} covariance numerical gate failed")
        arrays[f"scenario_{scenario}_posterior_draws_delta_modes"] = draws_delta
        arrays[f"scenario_{scenario}_posterior_draws_theta_modes"] = draws_delta[:, theta_keep]
        results[scenario] = {
            "nuisance": "known_and_subtracted",
            "noise_standard_deviation_scale": noise_scale,
            "noise_variance_scale": noise_scale**2,
            "numerical_gates": numerical,
        }
    return results, arrays


def solve_member(
    program: Mapping[str, object],
    program_sha256: str,
    mock_index: int,
    implementation_commit: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if not 0 <= mock_index < base.MOCK_COUNT:
        raise InformationError("mock index lies outside the 64 geometry members")
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise InformationError("implementation commit must be lowercase 40-hex")
    prior_member = Path(str(program["prior_ablation"]["members_root"])) / f"member-{mock_index:02d}"
    prior_aggregate_result = json.loads(
        (
            Path(str(program["prior_ablation"]["aggregate_directory"]))
            / "result.json"
        ).read_bytes()
    )
    expected_prior_validation = prior_aggregate_result["member_artifact_hashes"][mock_index]
    prior_validation = {
        "status": "PASS",
        "mock_index": mock_index,
        "result_sha256": sha256_file(prior_member / "result.json"),
        "fields_sha256": sha256_file(prior_member / "fields.npz"),
    }
    if prior_validation != expected_prior_validation:
        raise InformationError("prior ablation member is not the aggregate-bound artifact")
    prior_result = json.loads((prior_member / "result.json").read_bytes())
    if prior_result["implementation_commit"] != program["prior_ablation"]["member_implementation_commit"]:
        raise InformationError("prior ablation member implementation changed")
    d_member = Path(str(program["completed_D"]["members_root"])) / f"member-{mock_index:02d}"
    d_validation = {
        "status": "PASS",
        "mock_index": mock_index,
        "result_sha256": sha256_file(d_member / "result.json"),
        "fields_sha256": sha256_file(d_member / "fields.npz"),
    }
    if prior_result["completed_D_member"] != d_validation:
        raise InformationError("prior ablation and D member lineage differ")

    prior_names = {
        "arm_a_posterior_draws_delta_modes",
        "arm_a_posterior_draws_theta_modes",
        "delta_mode_bin_index",
        "theta_mode_bin_index",
        "delta_prior_variance",
        "theta_prior_variance",
        "delta_self_conjugate",
        "theta_self_conjugate",
        "selected_raw_idx",
        "train_raw_idx",
        "holdout_raw_idx",
    }
    with np.load(prior_member / "fields.npz", allow_pickle=False) as fields:
        prior_fields = {name: np.array(fields[name]) for name in prior_names}
    d_names = {
        "mock_cz",
        "mock_observed_distance",
        "mock_distance_error_mag",
        "mock_direction",
        "mock_true_position",
    }
    with np.load(d_member / "fields.npz", allow_pickle=False) as fields:
        d_fields = {name: np.array(fields[name]) for name in d_names}
    # Reconstruct D's variance and row selection without reading truth fields,
    # nuisance truth, any datum, or observational CF4 peculiar velocities.
    count = d_fields["mock_cz"].size
    catalog = {
        "H0": np.array(74.6),
        "v3k": d_fields["mock_cz"],
        "dist": d_fields["mock_observed_distance"],
        "e_dm": d_fields["mock_distance_error_mag"],
        "nhat": d_fields["mock_direction"],
        "pgc": np.arange(1, count + 1, dtype=np.int64),
    }
    args = base.fixed.frozen_args(ROOT / "data/cf4_clean.npz")
    design = base.linear.prepare_bgc_catalog(args, catalog)
    selected = np.asarray(design["raw_idx"], dtype=np.int64)
    train = ~np.asarray(design["holdout"], dtype=bool)
    if not np.array_equal(selected, prior_fields["selected_raw_idx"]):
        raise InformationError("information audit selected rows differ from arm A")
    if not np.array_equal(selected[train], prior_fields["train_raw_idx"]):
        raise InformationError("information audit training rows differ from arm A")
    if not np.array_equal(selected[~train], prior_fields["holdout_raw_idx"]):
        raise InformationError("information audit holdout rows differ from arm A")
    modes = _mode_plan()
    if not np.array_equal(modes["assignment"], prior_fields["delta_mode_bin_index"]):
        raise InformationError("information audit delta mode plan changed")
    if not np.array_equal(modes["theta_assignment"], prior_fields["theta_mode_bin_index"]):
        raise InformationError("information audit theta mode plan changed")
    transfer, _ = base.fixed.build_density_transfer(args)
    seeds = base.seed_schedule(mock_index)
    new_results, new_arrays = solve_known_covariances(
        positions=d_fields["mock_true_position"][selected],
        directions=d_fields["mock_direction"][selected],
        variance=np.asarray(design["variance"], dtype=np.float64),
        train=train,
        args=args,
        seeds=seeds,
        transfer=transfer,
        modes=modes,
    )
    arrays = {
        "delta_mode_bin_index": prior_fields["delta_mode_bin_index"],
        "theta_mode_bin_index": prior_fields["theta_mode_bin_index"],
        "delta_prior_variance": prior_fields["delta_prior_variance"],
        "theta_prior_variance": prior_fields["theta_prior_variance"],
        "delta_self_conjugate": prior_fields["delta_self_conjugate"],
        "theta_self_conjugate": prior_fields["theta_self_conjugate"],
        "selected_raw_idx": selected,
        "train_raw_idx": prior_fields["train_raw_idx"],
        "holdout_raw_idx": prior_fields["holdout_raw_idx"],
        "scenario_marginalized_s1_posterior_draws_delta_modes": prior_fields[
            "arm_a_posterior_draws_delta_modes"
        ],
        "scenario_marginalized_s1_posterior_draws_theta_modes": prior_fields[
            "arm_a_posterior_draws_theta_modes"
        ],
        **new_arrays,
    }
    result = {
        "schema": MEMBER_SCHEMA,
        "status": MEMBER_STATUS,
        "mock_index": mock_index,
        "program_sha256": program_sha256,
        "implementation_commit": implementation_commit,
        "implementation_source_sha256": sha256_file(__file__),
        "prior_ablation_member": prior_validation,
        "completed_D_member": d_validation,
        "posterior_and_numerical_seeds_reused": seeds,
        "new_random_seed_count": 0,
        "truth_array_generated_or_deserialized": False,
        "truth_seed_used": False,
        "likelihood_datum_consumed_by_inference": False,
        "catalog_redshift_and_distance_marks_read_for_variance_reconstruction": True,
        "covariance_only": True,
        "selected_rows_and_split_exactly_reused": True,
        "scenarios": {
            "marginalized_s1": {
                "source": "immutable_completed_arm_A_posterior_draws",
                "nuisance": "Gaussian_prior_marginalized",
                "noise_standard_deviation_scale": 1.0,
                "new_inference_executed": False,
            },
            **new_results,
        },
        "development_only": True,
        "untouched_256_mock_validation_executed": False,
        "frontier_or_science_claim_allowed": False,
        "target_0p3_cMpc_h_claim_allowed": False,
    }
    return result, arrays


def _member_array_names() -> set[str]:
    names = {
        "delta_mode_bin_index",
        "theta_mode_bin_index",
        "delta_prior_variance",
        "theta_prior_variance",
        "delta_self_conjugate",
        "theta_self_conjugate",
        "selected_raw_idx",
        "train_raw_idx",
        "holdout_raw_idx",
    }
    for scenario in SCENARIOS:
        names.add(f"scenario_{scenario}_posterior_draws_delta_modes")
        names.add(f"scenario_{scenario}_posterior_draws_theta_modes")
    return names


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_directory(
    output_path: str | Path,
    result: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    *,
    kind: str,
) -> None:
    if kind not in {"member", "aggregate"}:
        raise InformationError("publication kind must be member or aggregate")
    output = Path(output_path)
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing overwrite of {output}")
    if not output.parent.is_dir():
        raise InformationError("output parent must already exist")
    stage = output.parent / f".{output.name}.staging"
    try:
        stage.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"refusing existing staging directory {stage}") from None
    identity = (stage.stat().st_dev, stage.stat().st_ino)
    published = False
    try:
        array_name = "fields.npz" if kind == "member" else "metrics.npz"
        array_payload = base.fixed.deterministic_npz_bytes(arrays)
        result_payload = canonical_json_bytes(result)
        _write_exclusive(stage / array_name, array_payload)
        _write_exclusive(stage / "result.json", result_payload)
        manifest = {
            "schema": f"ouruniv-cf4-same-truth-information-budget-{kind}-artifact-manifest-v1",
            "status": result["status"],
            "payloads": {
                array_name: {
                    "sha256": hashlib.sha256(array_payload).hexdigest(),
                    "bytes": len(array_payload),
                },
                "result.json": {
                    "sha256": hashlib.sha256(result_payload).hexdigest(),
                    "bytes": len(result_payload),
                },
            },
        }
        manifest_payload = canonical_json_bytes(manifest)
        _write_exclusive(stage / "manifest.json", manifest_payload)
        complete = {
            "schema": f"ouruniv-cf4-same-truth-information-budget-{kind}-complete-v1",
            "status": result["status"],
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "COMPLETE_written_last": True,
        }
        _write_exclusive(stage / "COMPLETE", canonical_json_bytes(complete))
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(stage, output)
        published = True
    finally:
        if not published:
            try:
                current = stage.stat()
            except FileNotFoundError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == identity:
                shutil.rmtree(stage)


def validate_member(directory: str | Path, expected_index: int | None = None) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {item.name for item in root.iterdir()} != EXPECTED_MEMBER_FILES:
        raise InformationError("information member artifact set is not exact")
    result_payload = (root / "result.json").read_bytes()
    fields_payload = (root / "fields.npz").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    manifest = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise InformationError("information member result is not canonical JSON")
    if result.get("schema") != MEMBER_SCHEMA or result.get("status") != MEMBER_STATUS:
        raise InformationError("information member schema/status mismatch")
    if expected_index is not None and result.get("mock_index") != expected_index:
        raise InformationError("information member index mismatch")
    if result.get("new_random_seed_count") != 0 or result.get("truth_array_generated_or_deserialized") is not False:
        raise InformationError("information member crossed the truth/random firewall")
    if result.get("likelihood_datum_consumed_by_inference") is not False or result.get("covariance_only") is not True:
        raise InformationError("information member is not covariance-only")
    if result.get("untouched_256_mock_validation_executed") is not False or result.get(
        "frontier_or_science_claim_allowed"
    ) is not False:
        raise InformationError("information member crossed the science firewall")
    for scenario in NEW_SCENARIOS:
        if result.get("scenarios", {}).get(scenario, {}).get("numerical_gates", {}).get("all_pass") is not True:
            raise InformationError(f"information member {scenario} numerical gate failed")
    expected_payloads = {
        "fields.npz": {"sha256": hashlib.sha256(fields_payload).hexdigest(), "bytes": len(fields_payload)},
        "result.json": {"sha256": hashlib.sha256(result_payload).hexdigest(), "bytes": len(result_payload)},
    }
    if manifest.get("payloads") != expected_payloads:
        raise InformationError("information member payload binding mismatch")
    if manifest.get("schema") != "ouruniv-cf4-same-truth-information-budget-member-artifact-manifest-v1":
        raise InformationError("information member manifest schema mismatch")
    if complete.get("schema") != "ouruniv-cf4-same-truth-information-budget-member-complete-v1":
        raise InformationError("information member COMPLETE schema mismatch")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest() or complete.get(
        "COMPLETE_written_last"
    ) is not True:
        raise InformationError("information member COMPLETE binding mismatch")
    with np.load(io.BytesIO(fields_payload), allow_pickle=False) as fields:
        if set(fields.files) != _member_array_names():
            raise InformationError("information member array set is not exact")
        for scenario in SCENARIOS:
            for domain, mode_count in (("delta", 8538), ("theta", 8535)):
                name = f"scenario_{scenario}_posterior_draws_{domain}_modes"
                if fields[name].shape != (base.POSTERIOR_DRAW_COUNT, mode_count) or not np.all(
                    np.isfinite(fields[name])
                ):
                    raise InformationError(f"information member draws invalid: {name}")
        selected = np.asarray(fields["selected_raw_idx"])
        train = np.asarray(fields["train_raw_idx"])
        hold = np.asarray(fields["holdout_raw_idx"])
        if selected.ndim != 1 or train.size + hold.size != selected.size:
            raise InformationError("information member split arrays are invalid")
        if not np.array_equal(np.sort(np.concatenate((train, hold))), selected):
            raise InformationError("information member split does not partition selected rows")
    return {
        "status": "PASS",
        "mock_index": result["mock_index"],
        "result_sha256": hashlib.sha256(result_payload).hexdigest(),
        "fields_sha256": hashlib.sha256(fields_payload).hexdigest(),
    }


def posterior_information_spectrum(
    *,
    draws: np.ndarray,
    prior_variance: np.ndarray,
    assignment: np.ndarray,
    self_conjugate: np.ndarray,
    bin_ids: np.ndarray,
    bootstrap_indices: np.ndarray,
    gates: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Estimate prior-weighted posterior trace and recovered information."""

    draws = np.asarray(draws)
    prior_variance = np.asarray(prior_variance, dtype=np.float64)
    assignment = np.asarray(assignment, dtype=np.int64)
    self_conjugate = np.asarray(self_conjugate, dtype=bool)
    if draws.shape[:2] != (base.MOCK_COUNT, base.POSTERIOR_DRAW_COUNT):
        raise InformationError("information draw stack has wrong leading shape")
    if draws.shape[2:] != assignment.shape or prior_variance.shape != assignment.shape:
        raise InformationError("information mode metadata shape mismatch")
    if np.any(prior_variance <= 0.0) or not np.all(np.isfinite(draws)):
        raise InformationError("information draws or prior variance are invalid")
    sample_variance = np.var(draws, axis=1, ddof=1)
    per_mock_trace = np.empty((base.MOCK_COUNT, bin_ids.size), dtype=np.float64)
    real_dof = np.empty(bin_ids.size, dtype=np.int64)
    for column, bin_id in enumerate(bin_ids):
        mask = assignment == bin_id
        if not np.any(mask):
            raise InformationError("declared information bin has no modes")
        denominator = float(np.sum(prior_variance[mask]))
        per_mock_trace[:, column] = np.sum(sample_variance[:, mask], axis=1) / denominator
        real_dof[column] = int(np.sum(np.where(self_conjugate[mask], 1, 2)))
    trace_point, trace_lower, trace_upper = base._bootstrap_interval(
        per_mock_trace, bootstrap_indices, statistic="mean"
    )
    information_point = 1.0 - trace_point
    information_lower = 1.0 - trace_upper
    information_upper = 1.0 - trace_lower
    if np.any(information_point < -0.1) or np.any(information_point > 1.0 + 1.0e-10):
        raise InformationError("posterior information estimate is outside its physical range")
    expected_correlation = np.sqrt(np.clip(information_point, 0.0, 1.0))
    response_pass = information_point >= float(gates["response_min_inclusive"])
    correlation_pass = expected_correlation >= float(gates["correlation_r_min_inclusive"])
    residual_pass = trace_point <= float(gates["residual_power_ratio_max_inclusive"])
    robust_pass = information_lower >= float(gates["robust_information_lower_min_inclusive"])
    point_pass = response_pass & correlation_pass & residual_pass
    strict = point_pass & robust_pass
    equivalent_snr = np.divide(
        np.clip(information_point, 0.0, None),
        trace_point,
        out=np.zeros_like(trace_point),
        where=trace_point > 1.0e-15,
    )
    metrics = {
        "posterior_prior_trace_fraction": trace_point.tolist(),
        "posterior_prior_trace_bootstrap_95_interval": np.column_stack(
            (trace_lower, trace_upper)
        ).tolist(),
        "recovered_information_fraction": information_point.tolist(),
        "recovered_information_bootstrap_95_interval": np.column_stack(
            (information_lower, information_upper)
        ).tolist(),
        "expected_response": information_point.tolist(),
        "expected_correlation_r": expected_correlation.tolist(),
        "expected_residual_power_ratio": trace_point.tolist(),
        "equivalent_scalar_signal_to_noise_ratio": equivalent_snr.tolist(),
        "real_degree_of_freedom_count": real_dof.tolist(),
        "effective_prior_weighted_constrained_degree_count": (
            information_point * real_dof
        ).tolist(),
        "point_performance_gate": point_pass.tolist(),
        "robust_information_gate": robust_pass.tolist(),
        "strict_gate": strict.tolist(),
    }
    arrays = {
        "per_mock_posterior_prior_trace_fraction": per_mock_trace,
        "posterior_prior_trace_fraction": trace_point,
        "posterior_prior_trace_bootstrap_lower_2p5": trace_lower,
        "posterior_prior_trace_bootstrap_upper_97p5": trace_upper,
        "recovered_information_fraction": information_point,
        "recovered_information_bootstrap_lower_2p5": information_lower,
        "recovered_information_bootstrap_upper_97p5": information_upper,
        "expected_response": information_point,
        "expected_correlation_r": expected_correlation,
        "expected_residual_power_ratio": trace_point,
        "equivalent_scalar_signal_to_noise_ratio": equivalent_snr,
        "real_degree_of_freedom_count": real_dof,
        "effective_prior_weighted_constrained_degree_count": information_point * real_dof,
        "point_performance_gate": point_pass,
        "robust_information_gate": robust_pass,
        "strict_gate": strict,
    }
    return metrics, arrays


def classify_information_budget(lowest: Mapping[str, bool]) -> str:
    marginalized, known1, known03, known01 = (
        bool(lowest[scenario]) for scenario in SCENARIOS
    )
    if marginalized:
        return "BASELINE_INFORMATION_SUFFICIENT_PRIOR_FIELD_FAILURE_REQUIRES_METRIC_AUDIT"
    if known1 and not marginalized:
        return "NUISANCE_MARGINALIZATION_DOMINANT"
    if known01 and not known1:
        return "MEASUREMENT_ERROR_DOMINANT_IMPROVE_VELOCITY_LIKELIHOOD"
    if not known01:
        return "FINITE_LOW_NOISE_CEILING_INSUFFICIENT_ADD_INDEPENDENT_Z0_DENSITY_TRACERS"
    if known03 and not known1:
        return "MEASUREMENT_ERROR_DOMINANT_IMPROVE_VELOCITY_LIKELIHOOD"
    return "MIXED_INFORMATION_BUDGET_NO_SINGLE_DOMINANT_COMPONENT"


def aggregate_members(
    program: Mapping[str, object],
    program_sha256: str,
    members_root: str | Path,
    member_implementation_commit: str,
    aggregation_runtime_commit: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    for label, commit in (
        ("member implementation", member_implementation_commit),
        ("aggregation runtime", aggregation_runtime_commit),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise InformationError(f"{label} commit must be lowercase 40-hex")
    root = Path(members_root)
    expected = {f"member-{index:02d}" for index in range(base.MOCK_COUNT)}
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected:
        raise InformationError("information member directory set is not exact")
    store = {
        scenario: {domain: [] for domain in ("delta", "theta")}
        for scenario in SCENARIOS
    }
    metadata: dict[str, dict[str, np.ndarray]] = {}
    member_hashes = []
    for index in range(base.MOCK_COUNT):
        member = root / f"member-{index:02d}"
        validation = validate_member(member, index)
        result = json.loads((member / "result.json").read_bytes())
        if result["program_sha256"] != program_sha256 or result["implementation_commit"] != member_implementation_commit:
            raise InformationError("information member program/implementation binding mismatch")
        member_hashes.append(validation)
        with np.load(member / "fields.npz", allow_pickle=False) as fields:
            for domain in ("delta", "theta"):
                current = {
                    "assignment": np.array(fields[f"{domain}_mode_bin_index"]),
                    "prior_variance": np.array(fields[f"{domain}_prior_variance"]),
                    "self_conjugate": np.array(fields[f"{domain}_self_conjugate"]),
                }
                if domain not in metadata:
                    metadata[domain] = current
                elif any(not np.array_equal(value, metadata[domain][key]) for key, value in current.items()):
                    raise InformationError(f"information {domain} metadata changed")
                for scenario in SCENARIOS:
                    store[scenario][domain].append(
                        np.array(fields[f"scenario_{scenario}_posterior_draws_{domain}_modes"])
                    )
    domain_bins = {
        domain: np.unique(metadata[domain]["assignment"])
        for domain in ("delta", "theta")
    }
    if not np.array_equal(domain_bins["delta"], corrected.EXPECTED_DELTA_BIN_IDS):
        raise InformationError("information delta support changed")
    if not np.array_equal(domain_bins["theta"], corrected.EXPECTED_THETA_BIN_IDS):
        raise InformationError("information theta support changed")
    union_bins = np.union1d(domain_bins["delta"], domain_bins["theta"])
    gates = program["information_gates"]
    bootstrap_indices = base._bootstrap_indices(
        base.MOCK_COUNT, int(gates["mock_cluster_bootstrap_replicates"]), base.BOOTSTRAP_SEED
    )
    body, body_sha, _ = base.fixed.load_bin_manifest(ROOT / "config/cf4_kf_bin_manifest_v1.json")
    upper_edges = base._merged_upper_edges(body)
    upper_k = np.asarray([upper_edges[int(bin_id)] for bin_id in union_bins])
    arrays: dict[str, np.ndarray] = {
        "bin_ids": union_bins,
        "delta_bin_ids": domain_bins["delta"],
        "theta_bin_ids": domain_bins["theta"],
        "upper_k_h_Mpc": upper_k,
        "bootstrap_mock_indices": bootstrap_indices,
    }
    scenario_results = {}
    strict_union = {}
    lowest = {}
    for scenario in SCENARIOS:
        domain_results = {}
        strict_union[scenario] = {}
        for domain in ("delta", "theta"):
            bins = domain_bins[domain]
            metrics, metric_arrays = posterior_information_spectrum(
                draws=np.stack(store[scenario][domain]),
                prior_variance=metadata[domain]["prior_variance"],
                assignment=metadata[domain]["assignment"],
                self_conjugate=metadata[domain]["self_conjugate"],
                bin_ids=bins,
                bootstrap_indices=bootstrap_indices,
                gates=gates,
            )
            domain_results[domain] = metrics
            arrays.update(
                {
                    f"scenario_{scenario}_{domain}_{name}": value
                    for name, value in metric_arrays.items()
                }
            )
            expanded, available = corrected.expand_gate_to_union(
                bins, np.asarray(metrics["strict_gate"], dtype=bool), union_bins
            )
            strict_union[scenario][domain] = expanded
            arrays[f"scenario_{scenario}_{domain}_available_on_union"] = available
            arrays[f"scenario_{scenario}_{domain}_strict_gate_on_union"] = expanded
        frontier = base.frontier.evaluate_field_frontiers(
            upper_k,
            strict_union[scenario]["delta"],
            strict_union[scenario]["theta"],
        )
        lowest[scenario] = bool(
            strict_union[scenario]["delta"][0]
            and strict_union[scenario]["theta"][0]
        )
        scenario_results[scenario] = {
            "semantics": program["scenarios"][scenario],
            "domain_information_spectrum": domain_results,
            "information_frontier_diagnostic": {
                "density_delta": base._frontier_payload(frontier.density_delta),
                "velocity_divergence_theta": base._frontier_payload(
                    frontier.velocity_divergence_theta
                ),
                "joint": base._frontier_payload(frontier.joint),
            },
            "lowest_joint_bin_robust_pass": lowest[scenario],
        }
    contrasts = {}
    for domain in ("delta", "theta"):
        marginalized = arrays[
            f"scenario_marginalized_s1_{domain}_recovered_information_fraction"
        ]
        known1 = arrays[f"scenario_known_s1_{domain}_recovered_information_fraction"]
        known03 = arrays[f"scenario_known_s0p3_{domain}_recovered_information_fraction"]
        known01 = arrays[f"scenario_known_s0p1_{domain}_recovered_information_fraction"]
        nuisance_gain = known1 - marginalized
        noise_gain_03 = known03 - known1
        noise_gain_01 = known01 - known1
        arrays[f"contrast_{domain}_known_minus_marginalized_s1_information"] = nuisance_gain
        arrays[f"contrast_{domain}_known_s0p3_minus_s1_information"] = noise_gain_03
        arrays[f"contrast_{domain}_known_s0p1_minus_s1_information"] = noise_gain_01
        contrasts[domain] = {
            "known_minus_marginalized_s1_information": nuisance_gain.tolist(),
            "known_s0p3_minus_s1_information": noise_gain_03.tolist(),
            "known_s0p1_minus_s1_information": noise_gain_01.tolist(),
        }
    diagnosis = classify_information_budget(lowest)
    result = {
        "schema": AGGREGATE_SCHEMA,
        "status": AGGREGATE_STATUS,
        "program_sha256": program_sha256,
        "member_implementation_commit": member_implementation_commit,
        "aggregation_runtime_commit": aggregation_runtime_commit,
        "implementation_source_sha256": sha256_file(__file__),
        "member_count": base.MOCK_COUNT,
        "posterior_draw_count": base.POSTERIOR_DRAW_COUNT,
        "member_artifact_hashes": member_hashes,
        "bin_manifest_body_sha256": body_sha,
        "union_merged_bin_ids": union_bins.tolist(),
        "domain_available_merged_bin_ids": {
            "delta": domain_bins["delta"].tolist(),
            "theta": domain_bins["theta"].tolist(),
        },
        "cumulative_upper_k_h_Mpc": upper_k.tolist(),
        "scenarios": scenario_results,
        "information_contrasts": contrasts,
        "lowest_joint_bin_robust_pass": lowest,
        "preregistered_diagnostic_code": diagnosis,
        "finite_ceiling_limit": (
            "known_s0p1 reduces measurement standard deviations by ten and variances "
            "by one hundred; it is not a zero-noise radial-geometry theorem"
        ),
        "truth_array_generated_or_deserialized": False,
        "likelihood_datum_consumed_by_inference": False,
        "catalog_marks_read_for_variance_reconstruction": True,
        "covariance_only": True,
        "development_only": True,
        "untouched_256_mock_validation_executed": False,
        "frontier_or_science_claim_allowed": False,
        "target_0p3_cMpc_h_claim_allowed": False,
        "next_action_requires_user_approval": True,
    }
    return result, arrays


def _aggregate_array_names() -> set[str]:
    names = {
        "bin_ids",
        "delta_bin_ids",
        "theta_bin_ids",
        "upper_k_h_Mpc",
        "bootstrap_mock_indices",
    }
    suffixes = {
        "per_mock_posterior_prior_trace_fraction",
        "posterior_prior_trace_fraction",
        "posterior_prior_trace_bootstrap_lower_2p5",
        "posterior_prior_trace_bootstrap_upper_97p5",
        "recovered_information_fraction",
        "recovered_information_bootstrap_lower_2p5",
        "recovered_information_bootstrap_upper_97p5",
        "expected_response",
        "expected_correlation_r",
        "expected_residual_power_ratio",
        "equivalent_scalar_signal_to_noise_ratio",
        "real_degree_of_freedom_count",
        "effective_prior_weighted_constrained_degree_count",
        "point_performance_gate",
        "robust_information_gate",
        "strict_gate",
        "available_on_union",
        "strict_gate_on_union",
    }
    for scenario in SCENARIOS:
        for domain in ("delta", "theta"):
            names |= {f"scenario_{scenario}_{domain}_{suffix}" for suffix in suffixes}
    for domain in ("delta", "theta"):
        names |= {
            f"contrast_{domain}_known_minus_marginalized_s1_information",
            f"contrast_{domain}_known_s0p3_minus_s1_information",
            f"contrast_{domain}_known_s0p1_minus_s1_information",
        }
    return names


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {item.name for item in root.iterdir()} != EXPECTED_AGGREGATE_FILES:
        raise InformationError("information aggregate artifact set is not exact")
    result_payload = (root / "result.json").read_bytes()
    metrics_payload = (root / "metrics.npz").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    manifest = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise InformationError("information aggregate result is not canonical JSON")
    if result.get("schema") != AGGREGATE_SCHEMA or result.get("status") != AGGREGATE_STATUS:
        raise InformationError("information aggregate schema/status mismatch")
    if result.get("member_count") != base.MOCK_COUNT or set(result.get("scenarios", {})) != set(SCENARIOS):
        raise InformationError("information aggregate scenario/member contract mismatch")
    if result.get("truth_array_generated_or_deserialized") is not False or result.get("likelihood_datum_consumed_by_inference") is not False:
        raise InformationError("information aggregate crossed the data firewall")
    if result.get("untouched_256_mock_validation_executed") is not False or result.get(
        "frontier_or_science_claim_allowed"
    ) is not False:
        raise InformationError("information aggregate crossed the science firewall")
    expected_payloads = {
        "metrics.npz": {"sha256": hashlib.sha256(metrics_payload).hexdigest(), "bytes": len(metrics_payload)},
        "result.json": {"sha256": hashlib.sha256(result_payload).hexdigest(), "bytes": len(result_payload)},
    }
    if manifest.get("payloads") != expected_payloads:
        raise InformationError("information aggregate payload binding mismatch")
    if manifest.get("schema") != "ouruniv-cf4-same-truth-information-budget-aggregate-artifact-manifest-v1":
        raise InformationError("information aggregate manifest schema mismatch")
    if complete.get("schema") != "ouruniv-cf4-same-truth-information-budget-aggregate-complete-v1":
        raise InformationError("information aggregate COMPLETE schema mismatch")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest() or complete.get(
        "COMPLETE_written_last"
    ) is not True:
        raise InformationError("information aggregate COMPLETE binding mismatch")
    with np.load(io.BytesIO(metrics_payload), allow_pickle=False) as metrics:
        if set(metrics.files) != _aggregate_array_names():
            raise InformationError("information aggregate metric array set is not exact")
        if not np.array_equal(metrics["delta_bin_ids"], corrected.EXPECTED_DELTA_BIN_IDS):
            raise InformationError("information aggregate delta support changed")
        if not np.array_equal(metrics["theta_bin_ids"], corrected.EXPECTED_THETA_BIN_IDS):
            raise InformationError("information aggregate theta support changed")
        for scenario in SCENARIOS:
            if metrics[f"scenario_{scenario}_theta_available_on_union"].tolist() != [True] * 11 + [False]:
                raise InformationError(f"{scenario} theta support is not fail-closed")
            if bool(metrics[f"scenario_{scenario}_theta_strict_gate_on_union"][-1]):
                raise InformationError(f"{scenario} absent theta terminal bin passed")
        for name in metrics.files:
            if not np.all(np.isfinite(metrics[name])):
                raise InformationError(f"information aggregate metric is nonfinite: {name}")
    return {
        "status": "PASS",
        "member_count": base.MOCK_COUNT,
        "result_sha256": hashlib.sha256(result_payload).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    member = sub.add_parser("run-member")
    member.add_argument("--program", required=True, type=Path)
    member.add_argument("--mock-index", required=True, type=int)
    member.add_argument("--output", required=True, type=Path)
    member.add_argument("--implementation-commit", required=True)
    validate_member_parser = sub.add_parser("validate-member")
    validate_member_parser.add_argument("--directory", required=True, type=Path)
    validate_member_parser.add_argument("--expected-index", type=int)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--program", required=True, type=Path)
    aggregate.add_argument("--members-root", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--member-implementation-commit", required=True)
    aggregate.add_argument("--aggregation-runtime-commit", required=True)
    validate_aggregate_parser = sub.add_parser("validate-aggregate")
    validate_aggregate_parser.add_argument("--directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-member":
            program, program_sha = load_program(args.program)
            result, arrays = solve_member(
                program, program_sha, args.mock_index, args.implementation_commit
            )
            publish_directory(args.output, result, arrays, kind="member")
            report = validate_member(args.output, args.mock_index)
        elif args.command == "validate-member":
            report = validate_member(args.directory, args.expected_index)
        elif args.command == "aggregate":
            program, program_sha = load_program(args.program)
            result, arrays = aggregate_members(
                program,
                program_sha,
                args.members_root,
                args.member_implementation_commit,
                args.aggregation_runtime_commit,
            )
            publish_directory(args.output, result, arrays, kind="aggregate")
            report = validate_aggregate(args.output)
        else:
            report = validate_aggregate(args.directory)
    except (OSError, ValueError, InformationError, base.CalibrationError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
