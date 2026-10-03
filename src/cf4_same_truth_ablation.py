#!/usr/bin/env python3
"""Paired same-truth likelihood ablation for the CF4 population calibration.

The completed 64 population-calibration members are immutable inputs.  For
each member this program reconstructs the exact selected row set and split,
then runs three diagnostic likelihood arms on the stored truth and catalog:

* A: true positions and an exact Gaussian linear datum;
* B: observed-distance positions and a direct moment-matched lognormal datum;
* C: the completed BGc geometry/error design with an exact Gaussian datum.

Arm D is the already completed population-selection plus BGc chain and is
never rerun.  All arms therefore share the already realised population
selection; this experiment does not remove or validate survey selection.  It
is development-only and cannot promote a frontier or make a 0.3 cMpc/h claim.
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
from cf4_kf_bin_manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_SCHEMA = "ouruniv-cf4-same-truth-likelihood-ablation-program-v1"
MEMBER_SCHEMA = "ouruniv-cf4-same-truth-likelihood-ablation-member-result-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-same-truth-likelihood-ablation-result-v1"
MEMBER_STATUS = "COMPLETE_SAME_TRUTH_DEVELOPMENT_MEMBER_NO_SCIENCE_CLAIM"
AGGREGATE_STATUS = "COMPLETE_SAME_TRUTH_DEVELOPMENT_ABLATION_NO_SCIENCE_CLAIM"
ARM_IDS = ("a", "b", "c")
ALL_ARM_IDS = ("a", "b", "c", "d")
NOISE_SEED_START = 2026900000
EXPECTED_MEMBER_FILES = {"fields.npz", "result.json", "manifest.json", "COMPLETE"}
EXPECTED_AGGREGATE_FILES = {"metrics.npz", "result.json", "manifest.json", "COMPLETE"}
DOMAIN_SUFFIXES = (
    "response",
    "correlation_r",
    "residual_power_ratio",
    "per_mock_variance_ratio_median",
    "variance_ratio_median",
    "variance_bootstrap_lower_2p5",
    "variance_bootstrap_upper_97p5",
    "phase_null_p_value",
    "phase_null_cross",
    "per_mock_coverage68",
    "coverage68",
    "coverage68_bootstrap_lower_2p5",
    "coverage68_bootstrap_upper_97p5",
    "per_mock_coverage95",
    "coverage95",
    "coverage95_bootstrap_lower_2p5",
    "coverage95_bootstrap_upper_97p5",
    "strict_gate",
)


class AblationError(ValueError):
    """The frozen same-truth ablation contract failed closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise AblationError("repository binding escapes the repository")
    return path


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AblationError("cannot parse same-truth program") from exc
    if program.get("schema") != PROGRAM_SCHEMA:
        raise AblationError("same-truth program schema mismatch")
    authorization = program.get("authorization", {})
    required_true = {
        "same_truth_64_mock_ablation",
        "Slurm_member_array_submission",
        "Slurm_dependent_aggregation_submission",
        "GPFS_declared_read_write",
    }
    required_false = {
        "new_truth_generation",
        "population_generator_retuning",
        "untouched_256_mock_validation",
        "frontier_promotion",
        "observational_CF4_inference",
        "IC_PM_HOP_RAMSES",
        "automatic_retry",
        "automatic_follow_on_after_aggregate",
    }
    if any(authorization.get(key) is not True for key in required_true):
        raise AblationError("required same-truth authorization is absent")
    if any(authorization.get(key) is not False for key in required_false):
        raise AblationError("same-truth science firewall changed")
    development = program.get("development", {})
    if development.get("mock_count") != base.MOCK_COUNT:
        raise AblationError("same-truth mock count changed")
    if development.get("posterior_draw_count") != base.POSTERIOR_DRAW_COUNT:
        raise AblationError("same-truth posterior draw count changed")
    if development.get("ablation_noise_seed_start") != NOISE_SEED_START:
        raise AblationError("same-truth noise seed range changed")
    if development.get("ablation_noise_seed_stop_exclusive") != NOISE_SEED_START + 64:
        raise AblationError("same-truth noise seed stop changed")
    if development.get("truth_source") != "immutable_completed_D_member_artifacts":
        raise AblationError("same-truth source changed")
    arms = program.get("arms", {})
    if tuple(arms) != ALL_ARM_IDS or arms["d"].get("execution") != "read_only_completed_artifact":
        raise AblationError("ablation arm set or D immutability changed")
    for collection in ("repository_bindings", "source_bindings"):
        records = program.get(collection, {})
        if not isinstance(records, Mapping) or not records:
            raise AblationError(f"{collection} is absent")
        for record in records.values():
            if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
                raise AblationError(f"{collection} record is not exact")
            source = _resolve_repo_path(str(record["path"]))
            if sha256_file(source) != record["sha256"]:
                raise AblationError(f"SHA256 mismatch: {record['path']}")
    completed = program.get("completed_D", {})
    required_d = {
        "members_root",
        "member_implementation_commit",
        "aggregate_directory",
        "aggregate_result_sha256",
        "aggregate_metrics_sha256",
        "aggregate_manifest_sha256",
        "aggregate_complete_sha256",
    }
    if set(completed) != required_d:
        raise AblationError("completed D binding set is not exact")
    if re.fullmatch(r"[0-9a-f]{40}", str(completed["member_implementation_commit"])) is None:
        raise AblationError("completed D implementation commit is invalid")
    aggregate = Path(str(completed["aggregate_directory"]))
    external = {
        "result.json": "aggregate_result_sha256",
        "metrics.npz": "aggregate_metrics_sha256",
        "manifest.json": "aggregate_manifest_sha256",
        "COMPLETE": "aggregate_complete_sha256",
    }
    if not aggregate.is_dir() or {item.name for item in aggregate.iterdir()} != EXPECTED_AGGREGATE_FILES:
        raise AblationError("completed D aggregate artifact set changed")
    for name, key in external.items():
        if sha256_file(aggregate / name) != completed[key]:
            raise AblationError(f"completed D aggregate binding mismatch: {name}")
    resolution = program.get("resolution_semantics", {})
    if resolution.get("cell_size_cMpc_h") != base.fixed.BOX_SIZE / base.fixed.N:
        raise AblationError("development resolution changed")
    if resolution.get("target_0p3_cMpc_h_reached") is not False:
        raise AblationError("forbidden 0.3 cMpc/h claim")
    return program, hashlib.sha256(payload).hexdigest()


def reconstruct_catalog(fields: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reconstruct only the generated catalog stored by completed arm D."""

    count = np.asarray(fields["mock_cz"]).size
    if count != 22136:
        raise AblationError("completed D generated catalog row count changed")
    return {
        "H0": np.array(74.6, dtype=np.float64),
        "v3k": np.asarray(fields["mock_cz"], dtype=np.float64),
        "dist": np.asarray(fields["mock_observed_distance"], dtype=np.float64),
        "e_dm": np.asarray(fields["mock_distance_error_mag"], dtype=np.float64),
        "nhat": np.asarray(fields["mock_direction"], dtype=np.float64),
        "pgc": np.arange(1, count + 1, dtype=np.int64),
    }


def direct_lognormal_design(
    catalog: Mapping[str, np.ndarray],
    raw_idx: np.ndarray,
    holdout: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Arm-B Gaussian moment approximation to direct lognormal distances."""

    raw = np.asarray(raw_idx, dtype=np.int64)
    observed_distance = np.asarray(catalog["dist"], dtype=np.float64)[raw]
    error_mag = np.asarray(catalog["e_dm"], dtype=np.float64)[raw]
    direction = np.asarray(catalog["nhat"], dtype=np.float64)[raw]
    cz = np.asarray(catalog["v3k"], dtype=np.float64)[raw]
    h0 = float(np.asarray(catalog["H0"]))
    hcat = h0 / 100.0
    sigma_ln = np.maximum(error_mag, args.edm_floor) * np.log(10.0) / 5.0
    corrected_distance = observed_distance * np.exp(-0.5 * sigma_ln**2)
    sigma_distance = corrected_distance * np.sqrt(np.expm1(sigma_ln**2))
    variance = (args.error_scale * h0 * sigma_distance) ** 2 + args.sigma_nl**2
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise AblationError("arm-B moment variance is invalid")
    return {
        "raw_idx": raw,
        "pos": corrected_distance[:, None] * direction * hcat + args.box_size / 2.0,
        "rhat": direction,
        "vobs": cz - h0 * corrected_distance,
        "variance": variance,
        "B": np.column_stack((direction, -corrected_distance)),
        "q_std": np.array([args.bulk_prior] * 3 + [args.h0_prior], dtype=np.float64),
        "holdout": np.asarray(holdout, dtype=bool),
    }


def paired_arm_designs(
    fields: Mapping[str, np.ndarray],
    bgc_design: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    noise_seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Return paired A/B/C designs without evaluating any forward operator."""

    raw = np.asarray(bgc_design["raw_idx"], dtype=np.int64)
    holdout = np.asarray(bgc_design["holdout"], dtype=bool)
    catalog = reconstruct_catalog(fields)
    direction = np.asarray(catalog["nhat"])[raw]
    true_distance = np.asarray(fields["mock_true_distance"], dtype=np.float64)[raw]
    true_position = np.asarray(fields["mock_true_position"], dtype=np.float64)[raw]
    common_noise = np.random.default_rng(noise_seed).standard_normal(raw.size)
    common = {
        "raw_idx": raw,
        "rhat": direction,
        "variance": np.asarray(bgc_design["variance"], dtype=np.float64),
        "q_std": np.asarray(bgc_design["q_std"], dtype=np.float64),
        "holdout": holdout,
    }
    arm_a = {
        **common,
        "pos": true_position,
        "B": np.column_stack((direction, -true_distance)),
        "exact_gaussian": np.array(True),
    }
    arm_b = {
        **direct_lognormal_design(catalog, raw, holdout, args),
        "exact_gaussian": np.array(False),
    }
    arm_c = {
        **common,
        "pos": np.asarray(bgc_design["pos"], dtype=np.float64),
        "B": np.asarray(bgc_design["B"], dtype=np.float64),
        "exact_gaussian": np.array(True),
    }
    return {"a": arm_a, "b": arm_b, "c": arm_c}, common_noise


def _mode_plan(manifest_path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    body, body_sha, file_sha = base.fixed.load_bin_manifest(manifest_path)
    plan = base.fixed.global_merged_mode_plan(body)
    flat = np.asarray(plan["flat_independent_field_indices"], dtype=np.int64)
    assignment = np.asarray(plan["mode_merged_bin_index"], dtype=np.int64)
    grid_index = np.unravel_index(flat, (base.fixed.N,) * 3)
    theta_keep = np.logical_and.reduce([axis != base.fixed.N // 2 for axis in grid_index])
    return {
        "body": body,
        "body_sha": body_sha,
        "file_sha": file_sha,
        "upper_edges": base._merged_upper_edges(body),
    }, {
        "flat": flat,
        "assignment": assignment,
        "theta_flat": flat[theta_keep],
        "theta_assignment": assignment[theta_keep],
        "theta_keep": theta_keep,
    }


def infer_arm(
    *,
    arm_id: str,
    design: Mapping[str, np.ndarray],
    common_noise: np.ndarray,
    truth_white: np.ndarray,
    truth_nuisance_q: np.ndarray,
    stored_true_radial: np.ndarray,
    seeds: Mapping[str, object],
    args: argparse.Namespace,
    transfer: np.ndarray,
    growth_rate: float,
    manifest: Mapping[str, object],
    modes: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run one exact linear-Gaussian solve on a frozen paired design."""

    import jax
    import jax.numpy as jnp

    forward_all, adjoint_all, forward_growth, dtype = base.linear.build_forward(
        design["pos"], design["rhat"], args
    )
    if abs(forward_growth - growth_rate) > 1.0e-12:
        raise AblationError(f"arm {arm_id} forward growth changed")
    truth_signal = np.asarray(forward_all(truth_white), dtype=np.float64)
    radial_gate = None
    if arm_id == "a":
        difference = truth_signal - np.asarray(stored_true_radial, dtype=np.float64)
        relative = float(
            np.linalg.norm(difference) / max(np.linalg.norm(stored_true_radial), 1.0e-30)
        )
        radial_gate = {
            "relative_error": relative,
            "max_abs_error_km_s": float(np.max(np.abs(difference))),
            "maximum_inclusive": base.fixed.RADIAL_FORWARD_MAX_RELATIVE_ERROR,
            "pass": bool(relative <= base.fixed.RADIAL_FORWARD_MAX_RELATIVE_ERROR),
        }
        if not radial_gate["pass"]:
            raise AblationError("arm-A stored truth and exact forward operator disagree")
    if bool(np.asarray(design["exact_gaussian"])):
        observed = (
            truth_signal
            + np.asarray(design["B"]) @ np.asarray(truth_nuisance_q)
            + np.sqrt(np.asarray(design["variance"])) * common_noise
        )
    else:
        observed = np.asarray(design["vobs"], dtype=np.float64)

    train = ~np.asarray(design["holdout"], dtype=bool)
    hold = ~train
    if not np.any(train) or not np.any(hold):
        raise AblationError(f"arm {arm_id} lacks train or holdout rows")
    train_index = np.flatnonzero(train)
    train_index_jax = jnp.asarray(train_index)
    A_train = jax.jit(lambda field: forward_all(field)[train_index_jax])

    @jax.jit
    def AT_train(values):
        expanded = jnp.zeros(observed.size, dtype=dtype)
        return adjoint_all(expanded.at[train_index_jax].set(values))

    scale = jnp.asarray(np.sqrt(np.asarray(design["variance"])[train]), dtype=dtype)
    Bn = jnp.asarray(np.asarray(design["B"])[train], dtype=dtype) / scale[:, None]
    qvar = jnp.asarray(np.asarray(design["q_std"]) ** 2, dtype=dtype)
    datum = jnp.asarray(observed[train], dtype=dtype) / scale
    An = jax.jit(lambda field: A_train(field) / scale)
    ATn = jax.jit(lambda values: AT_train(values / scale))

    @jax.jit
    def Cnorm(values):
        return values + An(ATn(values)) + Bn @ (qvar * (Bn.T @ values))

    adjoint_rng = np.random.default_rng(int(seeds["adjoint"]))
    sx = jnp.asarray(adjoint_rng.standard_normal((base.fixed.N,) * 3), dtype=dtype)
    dy = jnp.asarray(adjoint_rng.standard_normal(train_index.size), dtype=dtype)
    lhs = float(jnp.vdot(An(sx), dy))
    rhs = float(jnp.vdot(sx, ATn(dy)))
    adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)

    probe_rng = np.random.default_rng(int(seeds["preconditioner"]))
    probe_power = np.zeros(train_index.size, dtype=np.float64)
    for _ in range(base.fixed.PRECONDITIONER_PROBES):
        probe = jnp.asarray(probe_rng.standard_normal((base.fixed.N,) * 3), dtype=dtype)
        probe_power += np.asarray(An(probe), dtype=np.float64) ** 2
    probe_power /= base.fixed.PRECONDITIONER_PROBES
    nuisance_diag = np.sum(
        np.asarray(Bn, dtype=np.float64) ** 2 * np.asarray(design["q_std"])[None, :] ** 2,
        axis=1,
    )
    preconditioner = jnp.asarray(1.0 + probe_power + nuisance_diag, dtype=dtype)
    alpha_mean, mean_relative, _ = base.linear.cg_solve(
        Cnorm, datum, preconditioner, args
    )
    mean_white = np.asarray(ATn(alpha_mean), dtype=np.float64)
    mean_q = np.asarray(qvar * (Bn.T @ alpha_mean), dtype=np.float64)

    posterior_white = []
    posterior_q = []
    prior_white = []
    prior_q = []
    sample_residuals = []
    for seed in seeds["posterior_draws"]:
        rng = np.random.default_rng(int(seed))
        xi = jnp.asarray(rng.standard_normal((base.fixed.N,) * 3), dtype=dtype)
        q0 = jnp.asarray(rng.standard_normal(4) * np.asarray(design["q_std"]), dtype=dtype)
        epsilon0 = jnp.asarray(rng.standard_normal(train_index.size), dtype=dtype)
        alpha, relative, _ = base.linear.cg_solve(
            Cnorm, datum - An(xi) - Bn @ q0 - epsilon0, preconditioner, args
        )
        prior_white.append(np.asarray(xi, dtype=np.float64))
        prior_q.append(np.asarray(q0, dtype=np.float64))
        posterior_white.append(np.asarray(xi + ATn(alpha), dtype=np.float64))
        posterior_q.append(np.asarray(q0 + qvar * (Bn.T @ alpha), dtype=np.float64))
        sample_residuals.append(float(relative))
    posterior_white_array = np.stack(posterior_white)
    posterior_q_array = np.stack(posterior_q)
    prior_white_array = np.stack(prior_white)
    prior_q_array = np.stack(prior_q)
    numerical = base.fixed._numerical_gate(
        adjoint_error, mean_relative, sample_residuals[:4]
    )
    numerical["all_16_sample_cg_relative_residuals"] = sample_residuals
    numerical["all_16_sample_cg_pass"] = bool(
        np.all(np.asarray(sample_residuals) <= base.fixed.CG_RESIDUAL_MAX)
    )
    numerical["all_pass"] = bool(
        numerical["adjoint_pass"]
        and numerical["mean_cg_pass"]
        and numerical["all_16_sample_cg_pass"]
    )
    if not numerical["all_pass"]:
        raise AblationError(f"arm {arm_id} numerical gate failed")

    mean_delta = base.fixed.white_to_delta(mean_white, transfer)
    mean_velocity = base.fixed.delta_to_velocity(mean_delta, growth_rate)
    mean_theta = base.fixed.velocity_to_normalized_divergence(mean_velocity, growth_rate)
    draw_delta = np.stack(
        [base.fixed.white_to_delta(draw, transfer) for draw in posterior_white_array]
    )
    draw_theta = np.stack(
        [
            base.fixed.velocity_to_normalized_divergence(
                base.fixed.delta_to_velocity(field, growth_rate), growth_rate
            )
            for field in draw_delta
        ]
    )
    consistency = [
        base.fixed.non_nyquist_delta_theta_relative_error(mean_delta, mean_theta),
        *[
            base.fixed.non_nyquist_delta_theta_relative_error(delta, theta)
            for delta, theta in zip(draw_delta, draw_theta)
        ],
    ]
    if max(consistency) > base.fixed.THETA_NON_NYQUIST_MAX_RELATIVE_ERROR:
        raise AblationError(f"arm {arm_id} delta/theta consistency failed")

    flat = np.asarray(modes["flat"], dtype=np.int64)
    theta_flat = np.asarray(modes["theta_flat"], dtype=np.int64)
    arrays = {
        "posterior_mean_nuisance_q": mean_q,
        "posterior_draws_nuisance_q": posterior_q_array,
        "posterior_mean_delta_modes": np.fft.fftn(mean_delta, norm="ortho").ravel()[flat],
        "posterior_draws_delta_modes": np.stack(
            [np.fft.fftn(field, norm="ortho").ravel()[flat] for field in draw_delta]
        ),
        "posterior_mean_theta_modes": np.fft.fftn(mean_theta, norm="ortho").ravel()[theta_flat],
        "posterior_draws_theta_modes": np.stack(
            [np.fft.fftn(field, norm="ortho").ravel()[theta_flat] for field in draw_theta]
        ),
    }

    B_hold = np.asarray(design["B"])[hold]
    observed_hold = observed[hold]
    noise_hold = np.asarray(design["variance"])[hold]
    prior_latent = np.stack(
        [
            np.asarray(forward_all(field))[hold] + B_hold @ q
            for field, q in zip(prior_white_array, prior_q_array)
        ]
    )
    prior_logp = base._predictive_log_density(
        observed_hold, np.zeros(observed_hold.size), prior_latent, noise_hold
    )
    heldout_rows = []
    for merged_id in np.unique(np.asarray(modes["assignment"], dtype=int)):
        upper = float(manifest["upper_edges"][int(merged_id)])
        mean_low = base._bandlimit(mean_white, upper, base.fixed.BOX_SIZE)
        latent_mean = np.asarray(forward_all(mean_low))[hold] + B_hold @ mean_q
        latent_draws = []
        for posterior, prior, q in zip(
            posterior_white_array, prior_white_array, posterior_q_array
        ):
            hybrid = prior + base._bandlimit(
                posterior - prior, upper, base.fixed.BOX_SIZE
            )
            latent_draws.append(np.asarray(forward_all(hybrid))[hold] + B_hold @ q)
        candidate_logp = base._predictive_log_density(
            observed_hold, latent_mean, np.stack(latent_draws), noise_hold
        )
        heldout_rows.append(
            {
                "merged_bin_index": int(merged_id),
                "cumulative_upper_k_h_Mpc": upper,
                "posterior_log_predictive_density": candidate_logp,
                "prior_log_predictive_density": prior_logp,
                "per_row_improvement": (candidate_logp - prior_logp) / observed_hold.size,
            }
        )
    result = {
        "arm_id": arm_id,
        "row_count": int(observed.size),
        "train_rows": int(np.count_nonzero(train)),
        "holdout_rows": int(np.count_nonzero(hold)),
        "likelihood": (
            "exact_linear_Gaussian"
            if bool(np.asarray(design["exact_gaussian"]))
            else "Gaussian_moment_approximation_to_direct_lognormal_distance_noise"
        ),
        "numerical_gates": numerical,
        "truth_forward_consistency_gate": radial_gate,
        "delta_theta_non_nyquist_max_relative_error": float(max(consistency)),
        "heldout_cumulative_prediction": heldout_rows,
    }
    return result, arrays


def solve_member(
    program: Mapping[str, object],
    program_sha256: str,
    mock_index: int,
    implementation_commit: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if not 0 <= mock_index < base.MOCK_COUNT:
        raise AblationError("mock index lies outside the 64 completed truths")
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise AblationError("implementation commit must be lowercase 40-hex")
    completed = program["completed_D"]
    d_member = Path(str(completed["members_root"])) / f"member-{mock_index:02d}"
    d_validation = base.validate_member(d_member, expected_index=mock_index)
    d_result = json.loads((d_member / "result.json").read_bytes())
    if d_result["implementation_commit"] != completed["member_implementation_commit"]:
        raise AblationError("completed D member implementation binding changed")
    d_aggregate_result = json.loads(
        (Path(str(completed["aggregate_directory"])) / "result.json").read_bytes()
    )
    if d_validation != d_aggregate_result["member_artifact_hashes"][mock_index]:
        raise AblationError("completed D member is not the aggregate-bound artifact")

    with np.load(d_member / "fields.npz", allow_pickle=False) as loaded:
        fields = {name: np.array(loaded[name]) for name in loaded.files}
    args = base.fixed.frozen_args(ROOT / "data/cf4_clean.npz")
    catalog = reconstruct_catalog(fields)
    bgc_design = base.linear.prepare_bgc_catalog(args, catalog)
    train = ~np.asarray(bgc_design["holdout"], dtype=bool)
    if not np.array_equal(bgc_design["raw_idx"][train], fields["train_raw_idx"]):
        raise AblationError("reconstructed D training rows changed")
    if not np.array_equal(bgc_design["raw_idx"][~train], fields["holdout_raw_idx"]):
        raise AblationError("reconstructed D holdout rows changed")

    manifest, modes = _mode_plan(ROOT / "config/cf4_kf_bin_manifest_v1.json")
    for domain, assignment in (
        ("delta", modes["assignment"]),
        ("theta", modes["theta_assignment"]),
    ):
        if not np.array_equal(assignment, fields[f"{domain}_mode_bin_index"]):
            raise AblationError(f"completed D {domain} mode plan changed")
    transfer, growth_rate = base.fixed.build_density_transfer(args)
    arm_designs, common_noise = paired_arm_designs(
        fields, bgc_design, args, NOISE_SEED_START + mock_index
    )
    raw = np.asarray(bgc_design["raw_idx"], dtype=np.int64)
    stored_true_radial = fields["mock_true_radial_velocity"][raw]
    seeds = base.seed_schedule(mock_index)

    arm_results = {}
    output_arrays: dict[str, np.ndarray] = {
        "truth_delta_modes": fields["truth_delta_modes"],
        "truth_theta_modes": fields["truth_theta_modes"],
        "delta_mode_bin_index": fields["delta_mode_bin_index"],
        "theta_mode_bin_index": fields["theta_mode_bin_index"],
        "delta_prior_variance": fields["delta_prior_variance"],
        "theta_prior_variance": fields["theta_prior_variance"],
        "delta_self_conjugate": fields["delta_self_conjugate"],
        "theta_self_conjugate": fields["theta_self_conjugate"],
        "selected_raw_idx": raw,
        "train_raw_idx": fields["train_raw_idx"],
        "holdout_raw_idx": fields["holdout_raw_idx"],
        "common_exact_standard_normal_noise": common_noise,
    }
    for arm_id in ARM_IDS:
        arm_result, arm_arrays = infer_arm(
            arm_id=arm_id,
            design=arm_designs[arm_id],
            common_noise=common_noise,
            truth_white=fields["truth_white"],
            truth_nuisance_q=fields["truth_nuisance_q"],
            stored_true_radial=stored_true_radial,
            seeds=seeds,
            args=args,
            transfer=transfer,
            growth_rate=growth_rate,
            manifest=manifest,
            modes=modes,
        )
        arm_results[arm_id] = arm_result
        output_arrays.update({f"arm_{arm_id}_{name}": value for name, value in arm_arrays.items()})

    result = {
        "schema": MEMBER_SCHEMA,
        "status": MEMBER_STATUS,
        "mock_index": mock_index,
        "program_sha256": program_sha256,
        "implementation_commit": implementation_commit,
        "implementation_source_sha256": sha256_file(__file__),
        "completed_D_member": d_validation,
        "completed_D_member_result_sha256": sha256_file(d_member / "result.json"),
        "truth_seed_reused_not_generated": int(d_result["truth_seed"]),
        "ablation_noise_seed": NOISE_SEED_START + mock_index,
        "posterior_and_numerical_seeds_reused_from_D": seeds,
        "selected_rows_and_split_exactly_reused": True,
        "population_selection_realization_shared_by_all_arms": True,
        "population_selection_removed_by_this_ablation": False,
        "observational_CF4_vpec_or_vobs_used": False,
        "arms": arm_results,
        "bin_manifest": {
            "file_sha256": manifest["file_sha"],
            "body_sha256": manifest["body_sha"],
        },
        "development_only": True,
        "untouched_256_mock_validation_executed": False,
        "frontier_or_science_claim_allowed": False,
        "target_0p3_cMpc_h_claim_allowed": False,
    }
    return result, output_arrays


def _member_array_names() -> set[str]:
    shared = {
        "truth_delta_modes",
        "truth_theta_modes",
        "delta_mode_bin_index",
        "theta_mode_bin_index",
        "delta_prior_variance",
        "theta_prior_variance",
        "delta_self_conjugate",
        "theta_self_conjugate",
        "selected_raw_idx",
        "train_raw_idx",
        "holdout_raw_idx",
        "common_exact_standard_normal_noise",
    }
    arm_suffixes = {
        "posterior_mean_nuisance_q",
        "posterior_draws_nuisance_q",
        "posterior_mean_delta_modes",
        "posterior_draws_delta_modes",
        "posterior_mean_theta_modes",
        "posterior_draws_theta_modes",
    }
    return shared | {f"arm_{arm}_{suffix}" for arm in ARM_IDS for suffix in arm_suffixes}


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
    """Publish a complete ablation artifact by one same-filesystem rename."""

    if kind not in {"member", "aggregate"}:
        raise AblationError("publication kind must be member or aggregate")
    output = Path(output_path)
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing overwrite of {output}")
    if not output.parent.is_dir():
        raise AblationError("output parent must already exist")
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
            "schema": f"ouruniv-cf4-same-truth-likelihood-ablation-{kind}-artifact-manifest-v1",
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
            "schema": f"ouruniv-cf4-same-truth-likelihood-ablation-{kind}-complete-v1",
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
        raise AblationError("ablation member artifact set is not exact")
    result_payload = (root / "result.json").read_bytes()
    fields_payload = (root / "fields.npz").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    manifest = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise AblationError("ablation member result is not canonical JSON")
    if result.get("schema") != MEMBER_SCHEMA or result.get("status") != MEMBER_STATUS:
        raise AblationError("ablation member schema/status mismatch")
    if expected_index is not None and result.get("mock_index") != expected_index:
        raise AblationError("ablation member index mismatch")
    if result.get("truth_seed_reused_not_generated") != base.DEVELOPMENT_TRUTH_SEED_START + result["mock_index"]:
        raise AblationError("ablation member truth binding changed")
    if result.get("ablation_noise_seed") != NOISE_SEED_START + result["mock_index"]:
        raise AblationError("ablation noise seed changed")
    if result.get("population_selection_removed_by_this_ablation") is not False:
        raise AblationError("member falsely claims removal of population selection")
    if result.get("untouched_256_mock_validation_executed") is not False or result.get(
        "frontier_or_science_claim_allowed"
    ) is not False:
        raise AblationError("member crossed the science firewall")
    for arm in ARM_IDS:
        if result.get("arms", {}).get(arm, {}).get("numerical_gates", {}).get("all_pass") is not True:
            raise AblationError(f"member arm {arm} numerical gate failed")
    if result["arms"]["a"].get("truth_forward_consistency_gate", {}).get("pass") is not True:
        raise AblationError("member arm-A truth-forward gate failed")
    expected_payloads = {
        "fields.npz": {"sha256": hashlib.sha256(fields_payload).hexdigest(), "bytes": len(fields_payload)},
        "result.json": {"sha256": hashlib.sha256(result_payload).hexdigest(), "bytes": len(result_payload)},
    }
    if manifest.get("payloads") != expected_payloads:
        raise AblationError("ablation member payload binding mismatch")
    if manifest.get("schema") != "ouruniv-cf4-same-truth-likelihood-ablation-member-artifact-manifest-v1":
        raise AblationError("ablation member manifest schema mismatch")
    if complete.get("schema") != "ouruniv-cf4-same-truth-likelihood-ablation-member-complete-v1":
        raise AblationError("ablation member COMPLETE schema mismatch")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest() or complete.get(
        "COMPLETE_written_last"
    ) is not True:
        raise AblationError("ablation member COMPLETE binding mismatch")
    with np.load(io.BytesIO(fields_payload), allow_pickle=False) as fields:
        if set(fields.files) != _member_array_names():
            raise AblationError("ablation member array set is not exact")
        shapes = {
            "truth_delta_modes": (8538,),
            "truth_theta_modes": (8535,),
            "delta_mode_bin_index": (8538,),
            "theta_mode_bin_index": (8535,),
            "delta_prior_variance": (8538,),
            "theta_prior_variance": (8535,),
            "delta_self_conjugate": (8538,),
            "theta_self_conjugate": (8535,),
        }
        for arm in ARM_IDS:
            shapes.update(
                {
                    f"arm_{arm}_posterior_mean_nuisance_q": (4,),
                    f"arm_{arm}_posterior_draws_nuisance_q": (base.POSTERIOR_DRAW_COUNT, 4),
                    f"arm_{arm}_posterior_mean_delta_modes": (8538,),
                    f"arm_{arm}_posterior_draws_delta_modes": (base.POSTERIOR_DRAW_COUNT, 8538),
                    f"arm_{arm}_posterior_mean_theta_modes": (8535,),
                    f"arm_{arm}_posterior_draws_theta_modes": (base.POSTERIOR_DRAW_COUNT, 8535),
                }
            )
        for name, shape in shapes.items():
            if fields[name].shape != shape or not np.all(np.isfinite(fields[name])):
                raise AblationError(f"ablation member array invalid: {name}")
        selected = np.asarray(fields["selected_raw_idx"])
        train = np.asarray(fields["train_raw_idx"])
        hold = np.asarray(fields["holdout_raw_idx"])
        noise = np.asarray(fields["common_exact_standard_normal_noise"])
        if selected.ndim != 1 or noise.shape != selected.shape or train.size + hold.size != selected.size:
            raise AblationError("ablation member selected-row arrays are invalid")
        if not np.array_equal(np.sort(np.concatenate((train, hold))), selected):
            raise AblationError("ablation member split does not partition selected rows")
    return {
        "status": "PASS",
        "mock_index": result["mock_index"],
        "result_sha256": hashlib.sha256(result_payload).hexdigest(),
        "fields_sha256": hashlib.sha256(fields_payload).hexdigest(),
    }


def _heldout_matrix(rows_by_mock: list[list[dict[str, object]]], bin_ids: np.ndarray) -> np.ndarray:
    matrix = np.empty((base.MOCK_COUNT, bin_ids.size), dtype=np.float64)
    for mock_index, rows in enumerate(rows_by_mock):
        row_map = {int(row["merged_bin_index"]): row for row in rows}
        if set(row_map) != set(bin_ids.tolist()):
            raise AblationError("arm heldout bin set changed")
        matrix[mock_index] = [row_map[int(bin_id)]["per_row_improvement"] for bin_id in bin_ids]
    return matrix


def classify_lowest_bin(pattern: Mapping[str, bool], generator_pass: bool) -> str:
    """Apply the preregistered coarse causal tree without retuning thresholds."""

    a, b, c, d = (bool(pattern[arm]) for arm in ALL_ARM_IDS)
    if not a:
        return "IDEAL_TRUE_POSITION_BASELINE_INSUFFICIENT"
    if not b and c and not d:
        return "MULTIPLE_DIRECT_DISTANCE_AND_BGC_DATUM_LOSSES"
    if b and c and not d:
        return "BGC_DATUM_OR_GENERATIVE_LIKELIHOOD_MISMATCH"
    if not b and c and d:
        return "DIRECT_LOGNORMAL_DISTANCE_APPROXIMATION_LOSS"
    if b and not c and d:
        return "FIXED_BGC_GEOMETRY_OR_ERROR_DESIGN_INSUFFICIENT"
    if a and b and c and d and not generator_pass:
        return "POPULATION_GENERATOR_FIDELITY_REMAINS_FAILED"
    if a and b and c and d and generator_pass:
        return "NO_LOWEST_BIN_LOSS_DETECTED"
    return "MIXED_NON_NESTED_PATTERN_NO_SINGLE_CAUSE"


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
            raise AblationError(f"{label} commit must be lowercase 40-hex")
    root = Path(members_root)
    expected = {f"member-{index:02d}" for index in range(base.MOCK_COUNT)}
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected:
        raise AblationError("ablation member directory set is not exact")

    store = {
        arm: {
            domain: {"truth": [], "mean": [], "draws": []}
            for domain in ("delta", "theta")
        }
        for arm in ARM_IDS
    }
    metadata: dict[str, dict[str, np.ndarray]] = {}
    heldout_rows = {arm: [] for arm in ARM_IDS}
    member_hashes = []
    for index in range(base.MOCK_COUNT):
        member = root / f"member-{index:02d}"
        validation = validate_member(member, expected_index=index)
        result = json.loads((member / "result.json").read_bytes())
        if result["program_sha256"] != program_sha256 or result["implementation_commit"] != member_implementation_commit:
            raise AblationError("ablation member program/implementation binding mismatch")
        member_hashes.append(validation)
        for arm in ARM_IDS:
            heldout_rows[arm].append(result["arms"][arm]["heldout_cumulative_prediction"])
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
                    raise AblationError(f"ablation {domain} metadata changed")
                truth = np.array(fields[f"truth_{domain}_modes"])
                for arm in ARM_IDS:
                    store[arm][domain]["truth"].append(truth)
                    store[arm][domain]["mean"].append(
                        np.array(fields[f"arm_{arm}_posterior_mean_{domain}_modes"])
                    )
                    store[arm][domain]["draws"].append(
                        np.array(fields[f"arm_{arm}_posterior_draws_{domain}_modes"])
                    )

    domain_bins = {domain: np.unique(metadata[domain]["assignment"]) for domain in ("delta", "theta")}
    if not np.array_equal(domain_bins["delta"], corrected.EXPECTED_DELTA_BIN_IDS):
        raise AblationError("ablation delta support changed")
    if not np.array_equal(domain_bins["theta"], corrected.EXPECTED_THETA_BIN_IDS):
        raise AblationError("ablation theta support changed")
    union_bins = np.union1d(domain_bins["delta"], domain_bins["theta"])
    gates = program["aggregate_gates"]
    bootstrap_indices = base._bootstrap_indices(
        base.MOCK_COUNT, int(gates["mock_cluster_bootstrap_replicates"]), base.BOOTSTRAP_SEED
    )
    manifest_body, manifest_sha, _ = base.fixed.load_bin_manifest(
        ROOT / "config/cf4_kf_bin_manifest_v1.json"
    )
    upper_edges = base._merged_upper_edges(manifest_body)
    upper_k = np.asarray([upper_edges[int(bin_id)] for bin_id in union_bins])
    arrays: dict[str, np.ndarray] = {
        "bin_ids": union_bins,
        "delta_bin_ids": domain_bins["delta"],
        "theta_bin_ids": domain_bins["theta"],
        "upper_k_h_Mpc": upper_k,
        "bootstrap_mock_indices": bootstrap_indices,
    }
    arm_results: dict[str, object] = {}
    lowest_pattern: dict[str, bool] = {}
    for arm_offset, arm in enumerate(ARM_IDS):
        heldout = _heldout_matrix(heldout_rows[arm], union_bins)
        heldout_point, heldout_lower, heldout_upper = base._bootstrap_interval(
            heldout, bootstrap_indices, statistic="mean"
        )
        heldout_pass_union = heldout_lower > float(
            gates["heldout_per_row_improvement_lower_min_exclusive"]
        )
        arrays.update(
            {
                f"arm_{arm}_heldout_per_mock_per_row_improvement": heldout,
                f"arm_{arm}_heldout_mean_per_row_improvement": heldout_point,
                f"arm_{arm}_heldout_bootstrap_lower_2p5": heldout_lower,
                f"arm_{arm}_heldout_bootstrap_upper_97p5": heldout_upper,
                f"arm_{arm}_heldout_pass": heldout_pass_union,
            }
        )
        domain_results = {}
        expanded_strict = {}
        for domain_offset, domain in enumerate(("delta", "theta")):
            bins = domain_bins[domain]
            domain_store = store[arm][domain]
            metrics, metric_arrays = base.compute_domain_calibration(
                domain_id=f"arm_{arm}_global_z0_{domain}",
                truth=np.stack(domain_store["truth"]),
                mean=np.stack(domain_store["mean"]),
                draws=np.stack(domain_store["draws"]),
                prior_variance=metadata[domain]["prior_variance"],
                assignment=metadata[domain]["assignment"],
                self_conjugate=metadata[domain]["self_conjugate"],
                bin_ids=bins,
                heldout_pass=heldout_pass_union[np.isin(union_bins, bins)],
                bootstrap_indices=bootstrap_indices,
                gates=gates,
                phase_seed=base.PHASE_NULL_SEED + domain_offset,
            )
            domain_results[domain] = metrics
            arrays.update(
                {f"arm_{arm}_{domain}_{name}": value for name, value in metric_arrays.items()}
            )
            expanded, available = corrected.expand_gate_to_union(
                bins, np.asarray(metrics["strict_gate"], dtype=bool), union_bins
            )
            expanded_strict[domain] = expanded
            arrays[f"arm_{arm}_{domain}_available_on_union"] = available
            arrays[f"arm_{arm}_{domain}_strict_gate_on_union"] = expanded
        frontiers = base.frontier.evaluate_field_frontiers(
            upper_k, expanded_strict["delta"], expanded_strict["theta"]
        )
        lowest_pattern[arm] = bool(
            expanded_strict["delta"][0] and expanded_strict["theta"][0]
        )
        arm_results[arm] = {
            "semantics": program["arms"][arm]["name"],
            "heldout_cumulative_prediction": {
                "mean_per_row_improvement": heldout_point.tolist(),
                "bootstrap_95_interval": np.column_stack((heldout_lower, heldout_upper)).tolist(),
                "pass": heldout_pass_union.tolist(),
            },
            "domain_metrics": domain_results,
            "frontier_diagnostic": {
                "density_delta": base._frontier_payload(frontiers.density_delta),
                "velocity_divergence_theta": base._frontier_payload(frontiers.velocity_divergence_theta),
                "joint": base._frontier_payload(frontiers.joint),
            },
            "lowest_joint_bin_pass": lowest_pattern[arm],
        }

    completed = program["completed_D"]
    d_dir = Path(str(completed["aggregate_directory"]))
    corrected.validate_aggregate(d_dir)
    d_result = json.loads((d_dir / "result.json").read_bytes())
    with np.load(d_dir / "metrics.npz", allow_pickle=False) as d_metrics:
        if not np.array_equal(d_metrics["bin_ids"], union_bins):
            raise AblationError("completed D union bin support changed")
        for key in (
            "heldout_per_mock_per_row_improvement",
            "heldout_mean_per_row_improvement",
            "heldout_bootstrap_lower_2p5",
            "heldout_bootstrap_upper_97p5",
            "heldout_pass",
        ):
            arrays[f"arm_d_{key}"] = np.array(d_metrics[key])
        for domain in ("delta", "theta"):
            for suffix in DOMAIN_SUFFIXES:
                arrays[f"arm_d_{domain}_{suffix}"] = np.array(d_metrics[f"{domain}_{suffix}"])
            for suffix in ("available_on_union", "strict_gate_on_union"):
                arrays[f"arm_d_{domain}_{suffix}"] = np.array(d_metrics[f"{domain}_{suffix}"])
        lowest_pattern["d"] = bool(
            d_metrics["delta_strict_gate_on_union"][0]
            and d_metrics["theta_strict_gate_on_union"][0]
        )
    arm_results["d"] = {
        "semantics": program["arms"]["d"]["name"],
        "execution": "immutable_completed_artifact_not_rerun",
        "result_sha256": completed["aggregate_result_sha256"],
        "metrics_sha256": completed["aggregate_metrics_sha256"],
        "population_generator_fidelity": d_result["population_generator_fidelity"],
        "domain_metrics": d_result["domain_metrics"],
        "frontier_diagnostic": d_result["development_strict_frontier_diagnostic"],
        "lowest_joint_bin_pass": lowest_pattern["d"],
    }
    generator_pass = bool(
        d_result["population_generator_fidelity"]["all_64_members_pass"]
    )
    diagnosis = classify_lowest_bin(lowest_pattern, generator_pass)
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
        "bin_manifest_body_sha256": manifest_sha,
        "union_merged_bin_ids": union_bins.tolist(),
        "domain_available_merged_bin_ids": {
            "delta": domain_bins["delta"].tolist(),
            "theta": domain_bins["theta"].tolist(),
        },
        "cumulative_upper_k_h_Mpc": upper_k.tolist(),
        "arms": arm_results,
        "lowest_joint_bin_pattern": lowest_pattern,
        "preregistered_diagnostic_code": diagnosis,
        "diagnostic_limit": (
            "A/B/C/D share one realised population selection. B and C are not a one-factor "
            "nested pair, so mixed contrasts are diagnostic and not a unique causal proof."
        ),
        "population_selection_removed_or_validated": False,
        "development_only": True,
        "untouched_256_mock_validation_executed": False,
        "frontier_or_science_claim_allowed": False,
        "target_0p3_cMpc_h_claim_allowed": False,
        "next_action_requires_user_approval": True,
    }
    return result, arrays


def _aggregate_array_names() -> set[str]:
    names = {"bin_ids", "delta_bin_ids", "theta_bin_ids", "upper_k_h_Mpc", "bootstrap_mock_indices"}
    for arm in ALL_ARM_IDS:
        names |= {
            f"arm_{arm}_heldout_per_mock_per_row_improvement",
            f"arm_{arm}_heldout_mean_per_row_improvement",
            f"arm_{arm}_heldout_bootstrap_lower_2p5",
            f"arm_{arm}_heldout_bootstrap_upper_97p5",
            f"arm_{arm}_heldout_pass",
        }
        for domain in ("delta", "theta"):
            names |= {f"arm_{arm}_{domain}_{suffix}" for suffix in DOMAIN_SUFFIXES}
            names |= {
                f"arm_{arm}_{domain}_available_on_union",
                f"arm_{arm}_{domain}_strict_gate_on_union",
            }
    return names


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {item.name for item in root.iterdir()} != EXPECTED_AGGREGATE_FILES:
        raise AblationError("ablation aggregate artifact set is not exact")
    result_payload = (root / "result.json").read_bytes()
    metrics_payload = (root / "metrics.npz").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    manifest = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise AblationError("ablation aggregate result is not canonical JSON")
    if result.get("schema") != AGGREGATE_SCHEMA or result.get("status") != AGGREGATE_STATUS:
        raise AblationError("ablation aggregate schema/status mismatch")
    if result.get("member_count") != base.MOCK_COUNT or set(result.get("arms", {})) != set(ALL_ARM_IDS):
        raise AblationError("ablation aggregate arm/member contract mismatch")
    if result.get("population_selection_removed_or_validated") is not False:
        raise AblationError("aggregate falsely claims selection removal")
    if result.get("untouched_256_mock_validation_executed") is not False or result.get(
        "frontier_or_science_claim_allowed"
    ) is not False:
        raise AblationError("aggregate crossed the science firewall")
    expected_payloads = {
        "metrics.npz": {"sha256": hashlib.sha256(metrics_payload).hexdigest(), "bytes": len(metrics_payload)},
        "result.json": {"sha256": hashlib.sha256(result_payload).hexdigest(), "bytes": len(result_payload)},
    }
    if manifest.get("payloads") != expected_payloads:
        raise AblationError("ablation aggregate payload binding mismatch")
    if manifest.get("schema") != "ouruniv-cf4-same-truth-likelihood-ablation-aggregate-artifact-manifest-v1":
        raise AblationError("ablation aggregate manifest schema mismatch")
    if complete.get("schema") != "ouruniv-cf4-same-truth-likelihood-ablation-aggregate-complete-v1":
        raise AblationError("ablation aggregate COMPLETE schema mismatch")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest() or complete.get(
        "COMPLETE_written_last"
    ) is not True:
        raise AblationError("ablation aggregate COMPLETE binding mismatch")
    with np.load(io.BytesIO(metrics_payload), allow_pickle=False) as metrics:
        if set(metrics.files) != _aggregate_array_names():
            raise AblationError("ablation aggregate metric array set is not exact")
        if not np.array_equal(metrics["delta_bin_ids"], corrected.EXPECTED_DELTA_BIN_IDS):
            raise AblationError("ablation aggregate delta support changed")
        if not np.array_equal(metrics["theta_bin_ids"], corrected.EXPECTED_THETA_BIN_IDS):
            raise AblationError("ablation aggregate theta support changed")
        for arm in ALL_ARM_IDS:
            if metrics[f"arm_{arm}_theta_available_on_union"].tolist() != [True] * 11 + [False]:
                raise AblationError(f"arm {arm} theta support is not fail-closed")
            if bool(metrics[f"arm_{arm}_theta_strict_gate_on_union"][-1]):
                raise AblationError(f"arm {arm} absent theta terminal bin passed")
        for name in metrics.files:
            if not np.all(np.isfinite(metrics[name])):
                raise AblationError(f"ablation aggregate metric is nonfinite: {name}")
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
    except (OSError, ValueError, AblationError, base.CalibrationError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
