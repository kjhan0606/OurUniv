#!/usr/bin/env python3
"""Grouped-CF4 empirical-selection truth mocks and 64-mock calibration.

The generator is conditional on the observed number of clean grouped CF4
constraints.  It uses only their distance, sky direction, and distance-error
marks as an empirical selection basis; observed peculiar velocities never
enter generation or inference.  Candidate group locations are coupled to an
independent LCDM truth density through a frozen log-Gaussian intensity, then
receive lognormal distance noise before the unchanged leakage-free BGc and
linear-CR likelihood are applied.

This is a development calibration, not the untouched 256-mock validation and
not an observational-frontier or 0.3 cMpc/h claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_bgc_fixed_design_smoke as fixed
import cf4_constraint_frontier as frontier
import cf4_linear_cr as linear
from cf4_kf_bin_manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_SCHEMA = "ouruniv-cf4-bgc-population-calibration-program-v1"
MEMBER_RESULT_SCHEMA = "ouruniv-cf4-bgc-population-calibration-member-result-v1"
AGGREGATE_RESULT_SCHEMA = "ouruniv-cf4-bgc-population-calibration-result-v1"
MOCK_COUNT = 64
POSTERIOR_DRAW_COUNT = 16
DEVELOPMENT_TRUTH_SEED_START = 2026083000
EXPECTED_MEMBER_FILES = {"fields.npz", "result.json", "manifest.json", "COMPLETE"}
EXPECTED_AGGREGATE_FILES = {"metrics.npz", "result.json", "manifest.json", "COMPLETE"}
BOOTSTRAP_SEED = 2026800000
PHASE_NULL_SEED = 2026800001


class CalibrationError(ValueError):
    """A generator, inference, calibration, or artifact contract failed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_schedule(mock_index: int) -> dict[str, object]:
    """Return disjoint frozen streams; validation truth seeds stay untouched."""

    if not isinstance(mock_index, int) or isinstance(mock_index, bool):
        raise CalibrationError("mock index must be an integer")
    if not 0 <= mock_index < MOCK_COUNT:
        raise CalibrationError("mock index lies outside the 64-mock development set")
    return {
        "truth": DEVELOPMENT_TRUTH_SEED_START + mock_index,
        "population": 2026100000 + mock_index,
        "distance_noise": 2026200000 + mock_index,
        "nuisance_truth": 2026300000 + mock_index,
        "preconditioner": 2026400000 + mock_index,
        "adjoint": 2026500000 + mock_index,
        "posterior_draws": [
            2026600000 + POSTERIOR_DRAW_COUNT * mock_index + draw
            for draw in range(POSTERIOR_DRAW_COUNT)
        ],
        "heldout_bootstrap": 2026700000 + mock_index,
    }


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CalibrationError("cannot parse calibration program") from exc
    if program.get("schema") != PROGRAM_SCHEMA:
        raise CalibrationError("calibration program schema mismatch")
    if program.get("development", {}).get("mock_count") != MOCK_COUNT:
        raise CalibrationError("program must freeze exactly 64 development mocks")
    if program.get("development", {}).get("posterior_draw_count") != POSTERIOR_DRAW_COUNT:
        raise CalibrationError("program must freeze exactly 16 posterior draws")
    if program.get("authorization", {}).get("untouched_256_mock_validation") is not False:
        raise CalibrationError("program must leave untouched validation unauthorized")
    if program.get("authorization", {}).get("development_64_mock_calibration") is not True:
        raise CalibrationError("64-mock development calibration is not authorized")
    for forbidden in ("frontier_promotion", "KF_EXPAND", "IC_PM_HOP_RAMSES"):
        if program.get("authorization", {}).get(forbidden) is not False:
            raise CalibrationError(f"program must leave {forbidden} unauthorized")
    if program.get("development", {}).get("truth_seed_start") != DEVELOPMENT_TRUTH_SEED_START:
        raise CalibrationError("development truth seed start changed")
    if program.get("development", {}).get("truth_seed_stop_exclusive") != 2026083064:
        raise CalibrationError("development truth seed range changed")
    if program.get("validation_firewall", {}).get("truth_seed_start") != 2026083064:
        raise CalibrationError("untouched validation truth seed start changed")
    if program.get("validation_firewall", {}).get("truth_seed_stop_exclusive") != 2026083320:
        raise CalibrationError("untouched validation truth seed range changed")
    if set(program.get("inputs", {})) != {"catalog", "bin_manifest"}:
        raise CalibrationError("program input binding set is not exact")
    for record in program["inputs"].values():
        if not isinstance(record, Mapping) or not {"path", "sha256"} <= set(record):
            raise CalibrationError("program input binding is incomplete")
        source = (ROOT / str(record["path"])).resolve()
        if ROOT.resolve() not in source.parents:
            raise CalibrationError("program input binding escapes the repository")
        if sha256_file(source) != record["sha256"]:
            raise CalibrationError(f"program input SHA256 mismatch: {record['path']}")
    expected_sources = {
        "population_calibration",
        "fixed_design_field_kernels",
        "constraint_frontier_gates",
        "linear_CR",
        "BGc_likelihood",
        "CF4_ingest",
    }
    if set(program.get("source_bindings", {})) != expected_sources:
        raise CalibrationError("program source binding set is not exact")
    for record in program["source_bindings"].values():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise CalibrationError("program source binding is not exact")
        source = (ROOT / str(record["path"])).resolve()
        if ROOT.resolve() not in source.parents:
            raise CalibrationError("program source binding escapes the repository")
        if sha256_file(source) != record["sha256"]:
            raise CalibrationError(f"program source SHA256 mismatch: {record['path']}")
    gates = program.get("aggregate_gates", {})
    exact_gate_values = {
        "mock_cluster_bootstrap_seed": BOOTSTRAP_SEED,
        "phase_null_seed_delta": PHASE_NULL_SEED,
        "phase_null_seed_theta": PHASE_NULL_SEED + 1,
        "response_min_inclusive": frontier.RESPONSE_MIN,
        "response_max_inclusive": frontier.RESPONSE_MAX,
        "correlation_r_min_inclusive": frontier.CORRELATION_MIN,
        "residual_power_ratio_max_inclusive": frontier.RESIDUAL_POWER_RATIO_MAX,
    }
    for key, expected in exact_gate_values.items():
        if gates.get(key) != expected:
            raise CalibrationError(f"program aggregate gate changed: {key}")
    resolution = program.get("resolution_semantics", {})
    if resolution.get("cell_size_cMpc_h") != fixed.BOX_SIZE / fixed.N:
        raise CalibrationError("program development cell size changed")
    if resolution.get("target_0p3_cMpc_h_reached") is not False:
        raise CalibrationError("program makes a forbidden 0.3 cMpc/h claim")
    return program, hashlib.sha256(payload).hexdigest()


def observed_selection_basis(catalog_path: str | Path) -> dict[str, np.ndarray]:
    """Load only selection/error marks; retain cz solely as a fidelity reference."""

    fixed.verify_frozen_provenance(catalog_path)
    with np.load(catalog_path, allow_pickle=False) as catalog:
        dist = np.asarray(catalog["dist"], dtype=np.float64)
        edm = np.asarray(catalog["e_dm"], dtype=np.float64)
        nhat = np.asarray(catalog["nhat"], dtype=np.float64)
        reference_cz = np.asarray(catalog["v3k"], dtype=np.float64)
        h0 = float(catalog["H0"])
        d_min = float(catalog["d_min"])
        d_max = float(catalog["d_max"])
    valid = (
        np.isfinite(dist)
        & np.isfinite(edm)
        & np.isfinite(reference_cz)
        & np.all(np.isfinite(nhat), axis=1)
        & (dist > d_min)
        & (dist <= d_max)
        & (edm > 0.0)
    )
    if np.count_nonzero(valid) != 22136:
        raise CalibrationError("frozen clean grouped-CF4 selection basis count changed")
    unit_error = np.max(np.abs(np.linalg.norm(nhat[valid], axis=1) - 1.0))
    if unit_error > 1.0e-10:
        raise CalibrationError("selection-basis directions are not unit vectors")
    return {
        "dist": dist[valid],
        "e_dm": edm[valid],
        "nhat": nhat[valid],
        "reference_cz": reference_cz[valid],
        "H0": np.array(h0),
        "d_min": np.array(d_min),
        "d_max": np.array(d_max),
    }


def _direction_to_angles(nhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.mod(np.arctan2(nhat[:, 1], nhat[:, 0]), 2.0 * np.pi)
    sin_latitude = np.clip(nhat[:, 2], -1.0, 1.0)
    return longitude, sin_latitude


def cic_sample_scalar(
    field: np.ndarray, positions: np.ndarray, box_size: float = fixed.BOX_SIZE
) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if field.ndim != 3 or len(set(field.shape)) != 1:
        raise CalibrationError("scalar CIC field must be cubic")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise CalibrationError("CIC positions must have shape (rows,3)")
    n = field.shape[0]
    spacing = box_size / n
    coordinate = (positions % box_size) / spacing
    base = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - base
    sampled = np.zeros(positions.shape[0], dtype=np.float64)
    for dx in (0, 1):
        wx = fraction[:, 0] if dx else 1.0 - fraction[:, 0]
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
            for dz in (0, 1):
                wz = fraction[:, 2] if dz else 1.0 - fraction[:, 2]
                sampled += (
                    wx
                    * wy
                    * wz
                    * field[
                        (base[:, 0] + dx) % n,
                        (base[:, 1] + dy) % n,
                        (base[:, 2] + dz) % n,
                    ]
                )
    return sampled


def _jitter_empirical_cells(
    basis: Mapping[str, np.ndarray],
    source_index: np.ndarray,
    rng: np.random.Generator,
    *,
    radial_bins: int,
    longitude_bins: int,
    sin_latitude_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram-kernel sample preserving joint distance/sky/error marks."""

    dist_source = np.asarray(basis["dist"])[source_index]
    nhat_source = np.asarray(basis["nhat"])[source_index]
    edm = np.asarray(basis["e_dm"])[source_index]
    d_min = float(np.asarray(basis["d_min"]))
    d_max = float(np.asarray(basis["d_max"]))
    radial_edges = np.linspace(d_min, d_max, radial_bins + 1)
    radial_index = np.clip(
        np.searchsorted(radial_edges, dist_source, side="right") - 1,
        0,
        radial_bins - 1,
    )
    lower = radial_edges[radial_index]
    upper = radial_edges[radial_index + 1]
    distance = np.cbrt(lower**3 + rng.random(source_index.size) * (upper**3 - lower**3))

    longitude, sin_latitude = _direction_to_angles(nhat_source)
    longitude_width = 2.0 * np.pi / longitude_bins
    longitude_index = np.minimum(
        (longitude / longitude_width).astype(np.int64), longitude_bins - 1
    )
    longitude_new = (longitude_index + rng.random(source_index.size)) * longitude_width
    sin_width = 2.0 / sin_latitude_bins
    sin_index = np.clip(
        ((sin_latitude + 1.0) / sin_width).astype(np.int64),
        0,
        sin_latitude_bins - 1,
    )
    sin_new = -1.0 + (sin_index + rng.random(source_index.size)) * sin_width
    cos_new = np.sqrt(np.maximum(1.0 - sin_new**2, 0.0))
    direction = np.column_stack(
        (cos_new * np.cos(longitude_new), cos_new * np.sin(longitude_new), sin_new)
    )
    return distance, direction, edm


def generate_population_catalog(
    basis: Mapping[str, np.ndarray],
    truth_delta: np.ndarray,
    truth_velocity: np.ndarray,
    mock_index: int,
    generator: Mapping[str, object],
) -> dict[str, object]:
    """Generate one empirical-selection, density-coupled grouped mock catalog."""

    seeds = seed_schedule(mock_index)
    population_rng = np.random.default_rng(int(seeds["population"]))
    noise_rng = np.random.default_rng(int(seeds["distance_noise"]))
    nuisance_rng = np.random.default_rng(int(seeds["nuisance_truth"]))
    count = np.asarray(basis["dist"]).size
    factor = int(generator["proposal_oversampling_factor"])
    candidate_count = factor * count
    source_index = population_rng.integers(0, count, size=candidate_count)
    observed_distance, direction, edm = _jitter_empirical_cells(
        basis,
        source_index,
        population_rng,
        radial_bins=int(generator["radial_histogram_bins"]),
        longitude_bins=int(generator["longitude_histogram_bins"]),
        sin_latitude_bins=int(generator["sin_latitude_histogram_bins"]),
    )
    sigma_ln = np.maximum(edm, float(generator["edm_floor_mag"])) * np.log(10.0) / 5.0
    epsilon_ln = noise_rng.normal(0.0, sigma_ln)
    true_distance = observed_distance * np.exp(-epsilon_ln)
    hcat = float(np.asarray(basis["H0"])) / 100.0
    radius_limit = fixed.BOX_SIZE / 2.0 * float(generator["radial_fraction"])
    valid = (
        np.isfinite(true_distance)
        & (true_distance > 0.0)
        & (true_distance * hcat < radius_limit)
    )
    if np.count_nonzero(valid) < count:
        raise CalibrationError("oversampled population has too few box-valid candidates")
    observed_distance = observed_distance[valid]
    true_distance = true_distance[valid]
    direction = direction[valid]
    edm = edm[valid]
    epsilon_ln = epsilon_ln[valid]
    source_index = source_index[valid]
    positions = true_distance[:, None] * direction * hcat + fixed.BOX_SIZE / 2.0
    local_delta = cic_sample_scalar(truth_delta, positions)
    bias = float(generator["log_gaussian_density_bias"])
    log_weight = bias * local_delta
    if not np.all(np.isfinite(log_weight)):
        raise CalibrationError("population log intensity is nonfinite")
    gumbel = population_rng.gumbel(size=log_weight.size)
    selected = np.argpartition(log_weight + gumbel, -count)[-count:]
    selected.sort()
    observed_distance = observed_distance[selected]
    true_distance = true_distance[selected]
    direction = direction[selected]
    edm = edm[selected]
    epsilon_ln = epsilon_ln[selected]
    positions = positions[selected]
    local_delta = local_delta[selected]
    source_index = source_index[selected]

    radial_velocity = fixed.cic_sample_radial_velocity(
        truth_velocity, positions, direction, fixed.BOX_SIZE
    )
    q_std = np.array([150.0, 150.0, 150.0, 3.0])
    nuisance_truth = nuisance_rng.standard_normal(4) * q_std
    nuisance_signal = direction @ nuisance_truth[:3] - true_distance * nuisance_truth[3]
    h0 = float(np.asarray(basis["H0"]))
    cz = h0 * true_distance + radial_velocity + nuisance_signal
    catalog = {
        "H0": np.array(h0),
        "v3k": cz,
        "dist": observed_distance,
        "e_dm": edm,
        "nhat": direction,
        "pgc": np.arange(1, count + 1, dtype=np.int64),
    }
    return {
        "catalog": catalog,
        "true_distance": true_distance,
        "true_position": positions,
        "true_radial_velocity": radial_velocity,
        "nuisance_truth": nuisance_truth,
        "distance_log_error": epsilon_ln,
        "local_truth_delta": local_delta,
        "empirical_source_index": source_index,
        "seeds": seeds,
    }


def _cdf_ks(first: np.ndarray, second: np.ndarray) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(first, second, method="asymp").statistic)


def _angular_total_variation(
    first: np.ndarray, second: np.ndarray, longitude_bins: int, sin_bins: int
) -> float:
    first_l, first_s = _direction_to_angles(first)
    second_l, second_s = _direction_to_angles(second)
    bins = (np.linspace(0.0, 2.0 * np.pi, longitude_bins + 1), np.linspace(-1.0, 1.0, sin_bins + 1))
    left = np.histogram2d(first_l, first_s, bins=bins)[0]
    right = np.histogram2d(second_l, second_s, bins=bins)[0]
    left /= left.sum()
    right /= right.sum()
    return float(0.5 * np.sum(np.abs(left - right)))


def population_fidelity(
    basis: Mapping[str, np.ndarray], generated_catalog: Mapping[str, np.ndarray], design: Mapping[str, np.ndarray], generator: Mapping[str, object]
) -> dict[str, object]:
    reference_target = (
        (np.asarray(basis["reference_cz"]) >= 1500.0)
        & (np.asarray(basis["reference_cz"]) <= 18000.0)
    )
    return {
        "conditioned_clean_group_count": int(np.asarray(basis["dist"]).size),
        "generated_clean_group_count": int(np.asarray(generated_catalog["dist"]).size),
        "BGc_selected_group_count": int(np.asarray(design["raw_idx"]).size),
        "observed_fixed_design_selected_group_count_reference": 19313,
        "observed_pre_BGc_target_count_reference": int(np.count_nonzero(reference_target)),
        "observed_distance_KS": _cdf_ks(
            np.asarray(basis["dist"]), np.asarray(generated_catalog["dist"])
        ),
        "distance_error_mag_KS": _cdf_ks(
            np.asarray(basis["e_dm"]), np.asarray(generated_catalog["e_dm"])
        ),
        "redshift_velocity_KS": _cdf_ks(
            np.asarray(basis["reference_cz"]), np.asarray(generated_catalog["v3k"])
        ),
        "angular_histogram_total_variation": _angular_total_variation(
            np.asarray(basis["nhat"]),
            np.asarray(generated_catalog["nhat"]),
            int(generator["fidelity_longitude_bins"]),
            int(generator["fidelity_sin_latitude_bins"]),
        ),
    }


def population_fidelity_gates(
    fidelity: Mapping[str, object], thresholds: Mapping[str, object]
) -> dict[str, object]:
    """Evaluate frozen catalog-level gates without suppressing failed mocks."""

    selected = int(fidelity["BGc_selected_group_count"])
    checks = {
        "clean_group_count_exact": (
            int(fidelity["conditioned_clean_group_count"])
            == int(fidelity["generated_clean_group_count"])
            == int(thresholds["conditioned_clean_group_count"])
        ),
        "BGc_selected_group_count_in_range": (
            int(thresholds["BGc_selected_group_count_min"])
            <= selected
            <= int(thresholds["BGc_selected_group_count_max"])
        ),
        "observed_distance_KS_pass": (
            float(fidelity["observed_distance_KS"])
            <= float(thresholds["observed_distance_KS_max"])
        ),
        "distance_error_mag_KS_pass": (
            float(fidelity["distance_error_mag_KS"])
            <= float(thresholds["distance_error_mag_KS_max"])
        ),
        "redshift_velocity_KS_pass": (
            float(fidelity["redshift_velocity_KS"])
            <= float(thresholds["redshift_velocity_KS_max"])
        ),
        "angular_histogram_total_variation_pass": (
            float(fidelity["angular_histogram_total_variation"])
            <= float(thresholds["angular_histogram_total_variation_max"])
        ),
    }
    return {**checks, "all_pass": bool(all(checks.values()))}


def _bandlimit(field: np.ndarray, upper_k: float, box_size: float) -> np.ndarray:
    n = field.shape[0]
    frequency = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    mask = (kx**2 + ky**2 + kz**2) <= upper_k**2 * (1.0 + 8.0e-15)
    return np.fft.ifftn(np.fft.fftn(field) * mask).real


def _predictive_log_density(
    datum: np.ndarray,
    mean: np.ndarray,
    latent_draws: np.ndarray,
    noise_variance: np.ndarray,
) -> float:
    variance = noise_variance + np.var(latent_draws, axis=0, ddof=1)
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise CalibrationError("predictive variance is invalid")
    residual = datum - mean
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * variance) + residual**2 / variance))


def _merged_upper_edges(manifest_body: Mapping[str, object]) -> dict[int, float]:
    native = {int(row["index"]): row for row in manifest_body["native_bins"]}
    return {
        int(row["merged_bin_index"]): max(
            float(native[index]["upper_h_Mpc"]) for index in row["native_bin_indices"]
        )
        for row in manifest_body["merged_bins"]
    }


def solve_member(
    program: Mapping[str, object], program_sha256: str, mock_index: int, implementation_commit: str
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Generate one population mock, infer its posterior, and retain diagnostics."""

    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise CalibrationError("implementation commit must be lowercase 40-hex")
    catalog_path = ROOT / program["inputs"]["catalog"]["path"]
    manifest_path = ROOT / program["inputs"]["bin_manifest"]["path"]
    basis = observed_selection_basis(catalog_path)
    args = fixed.frozen_args(catalog_path)
    seeds = seed_schedule(mock_index)
    truth_white = np.random.default_rng(int(seeds["truth"])).standard_normal((fixed.N,) * 3)
    transfer, growth_rate = fixed.build_density_transfer(args)
    truth_delta = fixed.white_to_delta(truth_white, transfer)
    truth_velocity = fixed.delta_to_velocity(truth_delta, growth_rate)
    population = generate_population_catalog(
        basis, truth_delta, truth_velocity, mock_index, program["generator"]
    )
    design = linear.prepare_bgc_catalog(args, population["catalog"])
    if np.count_nonzero(~design["holdout"]) == 0 or np.count_nonzero(design["holdout"]) == 0:
        raise CalibrationError("generated BGc design lacks train or holdout rows")
    fidelity = population_fidelity(
        basis, population["catalog"], design, program["generator"]
    )
    fidelity_gates = population_fidelity_gates(
        fidelity, program["population_fidelity_gates"]
    )

    import jax
    import jax.numpy as jnp

    forward_all, adjoint_all, forward_growth, dtype = linear.build_forward(
        design["pos"], design["rhat"], args
    )
    if abs(forward_growth - growth_rate) > 1.0e-12:
        raise CalibrationError("forward and retained-field growth rates disagree")
    train = ~design["holdout"]
    hold = design["holdout"]
    train_index = np.flatnonzero(train)
    train_index_jax = jnp.asarray(train_index)
    A_train = jax.jit(lambda field: forward_all(field)[train_index_jax])

    @jax.jit
    def AT_train(values):
        expanded = jnp.zeros(design["raw_idx"].size, dtype=dtype)
        return adjoint_all(expanded.at[train_index_jax].set(values))

    scale = jnp.asarray(np.sqrt(design["variance"][train]), dtype=dtype)
    Bn = jnp.asarray(design["B"][train], dtype=dtype) / scale[:, None]
    qvar = jnp.asarray(design["q_std"] ** 2, dtype=dtype)
    datum = jnp.asarray(design["vobs"][train], dtype=dtype) / scale
    An = jax.jit(lambda field: A_train(field) / scale)
    ATn = jax.jit(lambda values: AT_train(values / scale))

    @jax.jit
    def Cnorm(values):
        return values + An(ATn(values)) + Bn @ (qvar * (Bn.T @ values))

    adjoint_rng = np.random.default_rng(int(seeds["adjoint"]))
    sx = jnp.asarray(adjoint_rng.standard_normal((fixed.N,) * 3), dtype=dtype)
    dy = jnp.asarray(adjoint_rng.standard_normal(train_index.size), dtype=dtype)
    lhs = float(jnp.vdot(An(sx), dy))
    rhs = float(jnp.vdot(sx, ATn(dy)))
    adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)

    probe_rng = np.random.default_rng(int(seeds["preconditioner"]))
    probe_power = np.zeros(train_index.size, dtype=np.float64)
    for _ in range(fixed.PRECONDITIONER_PROBES):
        probe = jnp.asarray(probe_rng.standard_normal((fixed.N,) * 3), dtype=dtype)
        probe_power += np.asarray(An(probe), dtype=np.float64) ** 2
    probe_power /= fixed.PRECONDITIONER_PROBES
    nuisance_diag = np.sum(
        np.asarray(Bn, dtype=np.float64) ** 2 * design["q_std"][None, :] ** 2,
        axis=1,
    )
    preconditioner = jnp.asarray(1.0 + probe_power + nuisance_diag, dtype=dtype)
    alpha_mean, mean_rel, _ = linear.cg_solve(Cnorm, datum, preconditioner, args)
    posterior_mean_white = np.asarray(ATn(alpha_mean), dtype=np.float64)
    posterior_mean_q = np.asarray(qvar * (Bn.T @ alpha_mean), dtype=np.float64)

    posterior_draws = []
    posterior_q = []
    prior_draws = []
    prior_q = []
    sample_residuals = []
    for seed in seeds["posterior_draws"]:
        rng = np.random.default_rng(int(seed))
        xi = jnp.asarray(rng.standard_normal((fixed.N,) * 3), dtype=dtype)
        q0 = jnp.asarray(rng.standard_normal(4) * design["q_std"], dtype=dtype)
        epsilon0 = jnp.asarray(rng.standard_normal(train_index.size), dtype=dtype)
        alpha, relative, _ = linear.cg_solve(
            Cnorm, datum - An(xi) - Bn @ q0 - epsilon0, preconditioner, args
        )
        prior_draws.append(np.asarray(xi, dtype=np.float64))
        prior_q.append(np.asarray(q0, dtype=np.float64))
        posterior_draws.append(np.asarray(xi + ATn(alpha), dtype=np.float64))
        posterior_q.append(np.asarray(q0 + qvar * (Bn.T @ alpha), dtype=np.float64))
        sample_residuals.append(float(relative))
    posterior_draws_array = np.stack(posterior_draws)
    posterior_q_array = np.stack(posterior_q)
    prior_draws_array = np.stack(prior_draws)
    prior_q_array = np.stack(prior_q)
    gates = fixed._numerical_gate(adjoint_error, mean_rel, sample_residuals[:4])
    gates["all_16_sample_cg_relative_residuals"] = sample_residuals
    gates["all_16_sample_cg_pass"] = bool(
        np.all(np.asarray(sample_residuals) <= fixed.CG_RESIDUAL_MAX)
    )
    gates["all_pass"] = bool(
        gates["adjoint_pass"] and gates["mean_cg_pass"] and gates["all_16_sample_cg_pass"]
    )
    if not gates["all_pass"]:
        raise CalibrationError("member adjoint or CG gate failed")

    manifest_body, manifest_body_sha, manifest_file_sha = fixed.load_bin_manifest(manifest_path)
    plan = fixed.global_merged_mode_plan(manifest_body)
    flat = np.asarray(plan["flat_independent_field_indices"], dtype=np.int64)
    assignment = np.asarray(plan["mode_merged_bin_index"], dtype=np.int64)
    grid_index = np.unravel_index(flat, (fixed.N,) * 3)
    theta_keep = np.logical_and.reduce([axis != fixed.N // 2 for axis in grid_index])
    theta_flat = flat[theta_keep]
    theta_assignment = assignment[theta_keep]
    truth_theta = fixed.velocity_to_normalized_divergence(truth_velocity, growth_rate)
    mean_delta = fixed.white_to_delta(posterior_mean_white, transfer)
    mean_velocity = fixed.delta_to_velocity(mean_delta, growth_rate)
    mean_theta = fixed.velocity_to_normalized_divergence(mean_velocity, growth_rate)
    draw_delta = np.stack([fixed.white_to_delta(draw, transfer) for draw in posterior_draws_array])
    draw_velocity = np.stack([fixed.delta_to_velocity(field, growth_rate) for field in draw_delta])
    draw_theta = np.stack(
        [fixed.velocity_to_normalized_divergence(field, growth_rate) for field in draw_velocity]
    )
    delta_theta_errors = [
        fixed.non_nyquist_delta_theta_relative_error(truth_delta, truth_theta),
        fixed.non_nyquist_delta_theta_relative_error(mean_delta, mean_theta),
        *[
            fixed.non_nyquist_delta_theta_relative_error(delta, theta)
            for delta, theta in zip(draw_delta, draw_theta)
        ],
    ]
    if max(delta_theta_errors) > fixed.THETA_NON_NYQUIST_MAX_RELATIVE_ERROR:
        raise CalibrationError("member delta/theta consistency gate failed")

    delta_modes = {
        "truth": np.fft.fftn(truth_delta, norm="ortho").ravel()[flat],
        "mean": np.fft.fftn(mean_delta, norm="ortho").ravel()[flat],
        "draws": np.stack(
            [np.fft.fftn(field, norm="ortho").ravel()[flat] for field in draw_delta]
        ),
    }
    theta_modes = {
        "truth": np.fft.fftn(truth_theta, norm="ortho").ravel()[theta_flat],
        "mean": np.fft.fftn(mean_theta, norm="ortho").ravel()[theta_flat],
        "draws": np.stack(
            [np.fft.fftn(field, norm="ortho").ravel()[theta_flat] for field in draw_theta]
        ),
    }

    hold_index = np.flatnonzero(hold)
    B_hold = np.asarray(design["B"])[hold]
    observed_hold = np.asarray(design["vobs"])[hold]
    noise_hold = np.asarray(design["variance"])[hold]
    prior_latent = np.stack(
        [
            np.asarray(forward_all(field))[hold] + B_hold @ q
            for field, q in zip(prior_draws_array, prior_q_array)
        ]
    )
    prior_logp = _predictive_log_density(
        observed_hold, np.zeros(hold_index.size), prior_latent, noise_hold
    )
    upper_edges = _merged_upper_edges(manifest_body)
    heldout_rows = []
    for merged_id in np.asarray(plan["available_merged_bin_ids"], dtype=int):
        upper = upper_edges[int(merged_id)]
        mean_low = _bandlimit(posterior_mean_white, upper, fixed.BOX_SIZE)
        latent_mean = np.asarray(forward_all(mean_low))[hold] + B_hold @ posterior_mean_q
        latent_draws = []
        for posterior, prior, q in zip(
            posterior_draws_array, prior_draws_array, posterior_q_array
        ):
            hybrid = prior + _bandlimit(posterior - prior, upper, fixed.BOX_SIZE)
            latent_draws.append(np.asarray(forward_all(hybrid))[hold] + B_hold @ q)
        candidate_logp = _predictive_log_density(
            observed_hold, latent_mean, np.stack(latent_draws), noise_hold
        )
        heldout_rows.append(
            {
                "merged_bin_index": int(merged_id),
                "cumulative_upper_k_h_Mpc": upper,
                "posterior_log_predictive_density": candidate_logp,
                "prior_log_predictive_density": prior_logp,
                "per_row_improvement": (candidate_logp - prior_logp) / hold_index.size,
            }
        )

    result = {
        "schema": MEMBER_RESULT_SCHEMA,
        "status": "COMPLETE_DEVELOPMENT_MEMBER_NO_SCIENCE_CLAIM",
        "mock_index": mock_index,
        "program_sha256": program_sha256,
        "implementation_commit": implementation_commit,
        "implementation_source_sha256": sha256_file(__file__),
        "truth_seed": int(seeds["truth"]),
        "all_seeds": seeds,
        "selection_semantics": "empirical_grouped_CF4_selection_conditioned_on_clean_group_count",
        "observed_vpec_or_vobs_used": False,
        "observed_v3k_used_for_generation": False,
        "population_selection_mock_generated": True,
        "full_survey_selection_normalization_modeled": False,
        "catalog_fidelity": fidelity,
        "catalog_fidelity_gates": fidelity_gates,
        "catalog_design": {
            "selected_rows": int(design["raw_idx"].size),
            "train_rows": int(np.count_nonzero(train)),
            "holdout_rows": int(np.count_nonzero(hold)),
            "BGc_candidate_rows": int(design["bgc_candidate_n"]),
            "BGc_training_reference_pool_rows": int(design["bgc_reference_n"]),
        },
        "numerical_gates": gates,
        "delta_theta_non_nyquist_max_relative_error": float(max(delta_theta_errors)),
        "heldout_cumulative_prediction": heldout_rows,
        "bin_manifest": {
            "file_sha256": manifest_file_sha,
            "body_sha256": manifest_body_sha,
        },
        "development_only": True,
        "untouched_256_mock_validation_executed": False,
        "frontier_or_science_claim_allowed": False,
    }
    arrays = {
        "truth_white": truth_white,
        "posterior_mean_white": posterior_mean_white,
        "posterior_draws_white": posterior_draws_array,
        "prior_draws_white": prior_draws_array,
        "truth_delta": truth_delta,
        "truth_theta": truth_theta,
        "posterior_mean_delta": mean_delta,
        "posterior_mean_theta": mean_theta,
        "posterior_draws_delta": draw_delta,
        "posterior_draws_theta": draw_theta,
        "truth_velocity": truth_velocity,
        "posterior_mean_velocity": mean_velocity,
        "posterior_draws_velocity": draw_velocity,
        "truth_nuisance_q": population["nuisance_truth"],
        "posterior_mean_nuisance_q": posterior_mean_q,
        "posterior_draws_nuisance_q": posterior_q_array,
        "prior_draws_nuisance_q": prior_q_array,
        "mock_cz": population["catalog"]["v3k"],
        "mock_observed_distance": population["catalog"]["dist"],
        "mock_distance_error_mag": population["catalog"]["e_dm"],
        "mock_direction": population["catalog"]["nhat"],
        "mock_true_distance": population["true_distance"],
        "mock_true_position": population["true_position"],
        "mock_true_radial_velocity": population["true_radial_velocity"],
        "mock_distance_log_error": population["distance_log_error"],
        "mock_local_truth_delta": population["local_truth_delta"],
        "mock_empirical_source_index": population["empirical_source_index"],
        "truth_delta_modes": delta_modes["truth"],
        "posterior_mean_delta_modes": delta_modes["mean"],
        "posterior_draws_delta_modes": delta_modes["draws"],
        "truth_theta_modes": theta_modes["truth"],
        "posterior_mean_theta_modes": theta_modes["mean"],
        "posterior_draws_theta_modes": theta_modes["draws"],
        "delta_mode_bin_index": assignment,
        "theta_mode_bin_index": theta_assignment,
        "delta_prior_variance": transfer.ravel()[flat] ** 2,
        "theta_prior_variance": transfer.ravel()[theta_flat] ** 2,
        "delta_self_conjugate": np.logical_and.reduce(
            [axis == ((-axis) % fixed.N) for axis in grid_index]
        ),
        "theta_self_conjugate": np.logical_and.reduce(
            [axis[theta_keep] == ((-axis[theta_keep]) % fixed.N) for axis in grid_index]
        ),
        "train_raw_idx": design["raw_idx"][train],
        "holdout_raw_idx": design["raw_idx"][hold],
    }
    return result, arrays


def _bootstrap_indices(mock_count: int, replicate_count: int, seed: int) -> np.ndarray:
    if mock_count != MOCK_COUNT or replicate_count <= 0:
        raise CalibrationError("bootstrap shape violates the frozen development contract")
    return np.random.default_rng(seed).integers(
        0, mock_count, size=(replicate_count, mock_count), dtype=np.int16
    )


def _bootstrap_interval(
    values: np.ndarray, bootstrap_indices: np.ndarray, *, statistic: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != MOCK_COUNT:
        raise CalibrationError("bootstrap values must have shape (64,bins)")
    samples = values[np.asarray(bootstrap_indices, dtype=np.int64)]
    if statistic == "mean":
        estimates = np.mean(samples, axis=1)
        point = np.mean(values, axis=0)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
        point = np.median(values, axis=0)
    else:
        raise CalibrationError("unknown bootstrap statistic")
    lower, upper = np.quantile(estimates, [0.025, 0.975], axis=0, method="linear")
    return point, lower, upper


def _coverage_by_mock_bin(
    truth: np.ndarray,
    mean: np.ndarray,
    draws: np.ndarray,
    assignment: np.ndarray,
    self_conjugate: np.ndarray,
    bin_ids: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Central posterior coverage, clustered by truth mock.

    Canonical non-self Fourier coefficients contribute their independent real
    and imaginary components; self-conjugate coefficients contribute only the
    real component.  The Student-t multiplier accounts for estimating scale
    from exactly 16 posterior draws while the analytic posterior mean remains
    the interval centre.
    """

    standard = np.std(draws, axis=1, ddof=1)
    output = np.empty((truth.shape[0], bin_ids.size), dtype=np.float64)
    for column, bin_id in enumerate(bin_ids):
        mode_mask = assignment == bin_id
        if not np.any(mode_mask):
            raise CalibrationError(f"coverage bin has no modes: {int(bin_id)}")
        self_mask = self_conjugate[mode_mask]
        for mock in range(truth.shape[0]):
            truth_bin = truth[mock, mode_mask]
            mean_bin = mean[mock, mode_mask]
            sd_bin = standard[mock, mode_mask]
            draws_bin = draws[mock][:, mode_mask]
            real_sd = np.std(draws_bin.real, axis=0, ddof=1)
            real_hit = np.abs(truth_bin.real - mean_bin.real) <= multiplier * real_sd
            hits = [real_hit]
            if np.any(~self_mask):
                imag_sd = np.std(
                    draws_bin[:, ~self_mask].imag, axis=0, ddof=1
                )
                imag_hit = (
                    np.abs(truth_bin[~self_mask].imag - mean_bin[~self_mask].imag)
                    <= multiplier * imag_sd
                )
                hits.append(imag_hit)
            if (
                np.any(~np.isfinite(sd_bin))
                or np.any(real_sd <= 0.0)
                or (len(hits) == 2 and np.any(imag_sd <= 0.0))
            ):
                raise CalibrationError("posterior component scale is nonpositive")
            output[mock, column] = np.mean(np.concatenate(hits))
    return output


def _phase_null(
    truth: np.ndarray,
    mean: np.ndarray,
    assignment: np.ndarray,
    bin_ids: np.ndarray,
    *,
    replicate_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided random-sign phase null over canonical mock-mode units."""

    rng = np.random.default_rng(seed)
    null_cross = np.empty((replicate_count, bin_ids.size), dtype=np.float64)
    observed = np.empty(bin_ids.size, dtype=np.float64)
    for column, bin_id in enumerate(bin_ids):
        mask = assignment == bin_id
        contribution = np.real(np.conjugate(truth[:, mask]) * mean[:, mask])
        observed[column] = np.sum(contribution)
        for replicate in range(replicate_count):
            signs = rng.integers(0, 2, size=contribution.shape, dtype=np.int8) * 2 - 1
            null_cross[replicate, column] = np.sum(signs * contribution)
    p_value = (1.0 + np.sum(null_cross >= observed[None, :], axis=0)) / (
        replicate_count + 1.0
    )
    return p_value, null_cross


def compute_domain_calibration(
    *,
    domain_id: str,
    truth: np.ndarray,
    mean: np.ndarray,
    draws: np.ndarray,
    prior_variance: np.ndarray,
    assignment: np.ndarray,
    self_conjugate: np.ndarray,
    bin_ids: np.ndarray,
    heldout_pass: np.ndarray,
    bootstrap_indices: np.ndarray,
    gates: Mapping[str, object],
    phase_seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    truth = np.asarray(truth)
    mean = np.asarray(mean)
    draws = np.asarray(draws)
    if truth.shape != mean.shape or truth.ndim != 2:
        raise CalibrationError(f"{domain_id} truth/mean shape mismatch")
    if draws.shape != (MOCK_COUNT, POSTERIOR_DRAW_COUNT, truth.shape[1]):
        raise CalibrationError(f"{domain_id} posterior draw shape mismatch")
    if assignment.shape != (truth.shape[1],) or self_conjugate.shape != assignment.shape:
        raise CalibrationError(f"{domain_id} mode metadata shape mismatch")
    if self_conjugate.dtype != np.dtype(bool):
        raise CalibrationError(f"{domain_id} self-conjugate mask is not boolean")
    if prior_variance.shape != assignment.shape or np.any(prior_variance <= 0.0):
        raise CalibrationError(f"{domain_id} prior variance is invalid")
    if not all(np.all(np.isfinite(value)) for value in (truth, mean, draws, prior_variance)):
        raise CalibrationError(f"{domain_id} arrays contain nonfinite values")

    response = []
    correlation = []
    residual = []
    per_mock_variance = np.empty((MOCK_COUNT, bin_ids.size), dtype=np.float64)
    posterior_variance = np.var(draws, axis=1, ddof=1)
    mode_counts = []
    for column, bin_id in enumerate(bin_ids):
        mask = assignment == bin_id
        if not np.any(mask):
            raise CalibrationError(f"{domain_id} declared bin has no mode")
        truth_bin = truth[:, mask]
        mean_bin = mean[:, mask]
        truth_power = float(np.sum(np.abs(truth_bin) ** 2))
        mean_power = float(np.sum(np.abs(mean_bin) ** 2))
        cross = float(np.sum(np.real(np.conjugate(truth_bin) * mean_bin)))
        if truth_power <= 0.0 or mean_power <= 0.0:
            raise CalibrationError(f"{domain_id} zero field power")
        response.append(cross / truth_power)
        correlation.append(cross / math.sqrt(truth_power * mean_power))
        residual.append(float(np.sum(np.abs(mean_bin - truth_bin) ** 2)) / truth_power)
        per_mock_variance[:, column] = np.median(
            posterior_variance[:, mask] / prior_variance[None, mask], axis=1
        )
        mode_counts.append(int(np.count_nonzero(mask)))

    variance_point, variance_lower, variance_upper = _bootstrap_interval(
        per_mock_variance, bootstrap_indices, statistic="median"
    )
    phase_p, phase_null_cross = _phase_null(
        truth,
        mean,
        assignment,
        bin_ids,
        replicate_count=int(gates["phase_null_replicates"]),
        seed=phase_seed,
    )

    from scipy.stats import t as student_t

    nominal68 = float(gates["coverage68_nominal"])
    nominal95 = float(gates["coverage95_nominal"])
    multiplier68 = float(
        student_t.ppf((1.0 + nominal68) / 2.0, POSTERIOR_DRAW_COUNT - 1)
    )
    multiplier95 = float(
        student_t.ppf((1.0 + nominal95) / 2.0, POSTERIOR_DRAW_COUNT - 1)
    )
    per_mock_coverage68 = _coverage_by_mock_bin(
        truth, mean, draws, assignment, self_conjugate, bin_ids, multiplier68
    )
    per_mock_coverage95 = _coverage_by_mock_bin(
        truth, mean, draws, assignment, self_conjugate, bin_ids, multiplier95
    )
    coverage68, coverage68_lower, coverage68_upper = _bootstrap_interval(
        per_mock_coverage68, bootstrap_indices, statistic="mean"
    )
    coverage95, coverage95_lower, coverage95_upper = _bootstrap_interval(
        per_mock_coverage95, bootstrap_indices, statistic="mean"
    )

    response_array = np.asarray(response)
    correlation_array = np.clip(np.asarray(correlation), -1.0, 1.0)
    residual_array = np.asarray(residual)
    variance_pass = variance_upper < float(gates["variance_bootstrap_upper_max_exclusive"])
    phase_pass = phase_p <= float(gates["phase_null_p_max_inclusive"])
    coverage68_pass = (
        (np.abs(coverage68 - nominal68) <= float(gates["coverage68_abs_error_max"]))
        & (coverage68_lower <= nominal68)
        & (coverage68_upper >= nominal68)
    )
    coverage95_pass = (
        (np.abs(coverage95 - nominal95) <= float(gates["coverage95_abs_error_max"]))
        & (coverage95_lower <= nominal95)
        & (coverage95_upper >= nominal95)
    )
    strict = frontier.strict_gate_mask(
        response_array,
        correlation_array,
        residual_array,
        phase_pass,
        variance_pass,
        coverage68_pass,
        coverage95_pass,
        heldout_pass,
    )
    metrics = {
        "domain_id": domain_id,
        "mode_counts": mode_counts,
        "response": response_array.tolist(),
        "correlation_r": correlation_array.tolist(),
        "residual_power_ratio": residual_array.tolist(),
        "posterior_prior_variance_ratio_median": variance_point.tolist(),
        "variance_bootstrap_95_interval": np.column_stack(
            (variance_lower, variance_upper)
        ).tolist(),
        "variance_reduction_pass": variance_pass.tolist(),
        "phase_random_sign_null_p_value": phase_p.tolist(),
        "phase_null_pass": phase_pass.tolist(),
        "coverage68": coverage68.tolist(),
        "coverage68_bootstrap_95_interval": np.column_stack(
            (coverage68_lower, coverage68_upper)
        ).tolist(),
        "coverage68_pass": coverage68_pass.tolist(),
        "coverage95": coverage95.tolist(),
        "coverage95_bootstrap_95_interval": np.column_stack(
            (coverage95_lower, coverage95_upper)
        ).tolist(),
        "coverage95_pass": coverage95_pass.tolist(),
        "heldout_cumulative_improvement_pass": heldout_pass.tolist(),
        "strict_gate": strict.tolist(),
    }
    arrays = {
        "response": response_array,
        "correlation_r": correlation_array,
        "residual_power_ratio": residual_array,
        "per_mock_variance_ratio_median": per_mock_variance,
        "variance_ratio_median": variance_point,
        "variance_bootstrap_lower_2p5": variance_lower,
        "variance_bootstrap_upper_97p5": variance_upper,
        "phase_null_p_value": phase_p,
        "phase_null_cross": phase_null_cross,
        "per_mock_coverage68": per_mock_coverage68,
        "coverage68": coverage68,
        "coverage68_bootstrap_lower_2p5": coverage68_lower,
        "coverage68_bootstrap_upper_97p5": coverage68_upper,
        "per_mock_coverage95": per_mock_coverage95,
        "coverage95": coverage95,
        "coverage95_bootstrap_lower_2p5": coverage95_lower,
        "coverage95_bootstrap_upper_97p5": coverage95_upper,
        "strict_gate": strict,
    }
    return metrics, arrays


def _frontier_payload(value: frontier.FrontierResult) -> dict[str, object]:
    return {
        "upper_k_h_Mpc_or_null": value.k_eff,
        "prefix_bin_count": value.prefix_bin_count,
        "first_failed_index_or_null": value.first_failed_index,
        "ignored_passing_indices": list(value.ignored_passing_indices),
    }


def aggregate_members(
    program: Mapping[str, object],
    program_sha256: str,
    members_root: str | Path,
    implementation_commit: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Aggregate exactly 64 complete development members without promotion."""

    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise CalibrationError("implementation commit must be lowercase 40-hex")
    root = Path(members_root)
    if not root.is_dir():
        raise CalibrationError("member root is absent")
    expected_names = {f"member-{index:02d}" for index in range(MOCK_COUNT)}
    if {path.name for path in root.iterdir()} != expected_names:
        raise CalibrationError("member directory set is not exact")

    member_hashes = []
    domain_store = {
        "delta": {"truth": [], "mean": [], "draws": []},
        "theta": {"truth": [], "mean": [], "draws": []},
    }
    metadata: dict[str, dict[str, np.ndarray]] = {}
    heldout_rows = []
    fidelity_rows = []
    for index in range(MOCK_COUNT):
        member = root / f"member-{index:02d}"
        validation = validate_member(member, expected_index=index)
        result = json.loads((member / "result.json").read_bytes())
        if result["program_sha256"] != program_sha256:
            raise CalibrationError("member program binding mismatch")
        if result["implementation_commit"] != implementation_commit:
            raise CalibrationError("member implementation commit mismatch")
        member_hashes.append(validation)
        fidelity_rows.append(result["catalog_fidelity"])
        heldout_rows.append(result["heldout_cumulative_prediction"])
        with np.load(member / "fields.npz", allow_pickle=False) as fields:
            for domain in ("delta", "theta"):
                domain_store[domain]["truth"].append(np.array(fields[f"truth_{domain}_modes"]))
                domain_store[domain]["mean"].append(
                    np.array(fields[f"posterior_mean_{domain}_modes"])
                )
                domain_store[domain]["draws"].append(
                    np.array(fields[f"posterior_draws_{domain}_modes"])
                )
                current = {
                    "assignment": np.array(fields[f"{domain}_mode_bin_index"]),
                    "prior_variance": np.array(fields[f"{domain}_prior_variance"]),
                    "self_conjugate": np.array(fields[f"{domain}_self_conjugate"]),
                }
                if domain not in metadata:
                    metadata[domain] = current
                else:
                    for key in current:
                        if not np.array_equal(current[key], metadata[domain][key]):
                            raise CalibrationError(f"{domain} member metadata changed")

    delta_assignment = metadata["delta"]["assignment"]
    theta_assignment = metadata["theta"]["assignment"]
    bin_ids = np.unique(delta_assignment)
    if not np.array_equal(bin_ids, np.unique(theta_assignment)):
        raise CalibrationError("delta/theta available bin IDs differ")
    heldout = np.empty((MOCK_COUNT, bin_ids.size), dtype=np.float64)
    for mock, rows in enumerate(heldout_rows):
        row_map = {int(row["merged_bin_index"]): row for row in rows}
        if set(row_map) != set(bin_ids.tolist()):
            raise CalibrationError("heldout cumulative bin set changed")
        heldout[mock] = [row_map[int(bin_id)]["per_row_improvement"] for bin_id in bin_ids]

    aggregate_gates = program["aggregate_gates"]
    bootstrap_indices = _bootstrap_indices(
        MOCK_COUNT,
        int(aggregate_gates["mock_cluster_bootstrap_replicates"]),
        BOOTSTRAP_SEED,
    )
    heldout_point, heldout_lower, heldout_upper = _bootstrap_interval(
        heldout, bootstrap_indices, statistic="mean"
    )
    heldout_pass = heldout_lower > float(
        aggregate_gates["heldout_per_row_improvement_lower_min_exclusive"]
    )

    all_metrics: dict[str, object] = {}
    all_arrays: dict[str, np.ndarray] = {
        "bin_ids": bin_ids,
        "bootstrap_mock_indices": bootstrap_indices,
        "heldout_per_mock_per_row_improvement": heldout,
        "heldout_mean_per_row_improvement": heldout_point,
        "heldout_bootstrap_lower_2p5": heldout_lower,
        "heldout_bootstrap_upper_97p5": heldout_upper,
        "heldout_pass": heldout_pass,
    }
    for offset, domain in enumerate(("delta", "theta")):
        store = domain_store[domain]
        metrics, arrays = compute_domain_calibration(
            domain_id=(
                "global_z0_density_delta"
                if domain == "delta"
                else "global_discrete_normalized_velocity_divergence_theta"
            ),
            truth=np.stack(store["truth"]),
            mean=np.stack(store["mean"]),
            draws=np.stack(store["draws"]),
            prior_variance=metadata[domain]["prior_variance"],
            assignment=metadata[domain]["assignment"],
            self_conjugate=metadata[domain]["self_conjugate"],
            bin_ids=bin_ids,
            heldout_pass=heldout_pass,
            bootstrap_indices=bootstrap_indices,
            gates=aggregate_gates,
            phase_seed=PHASE_NULL_SEED + offset,
        )
        all_metrics[domain] = metrics
        all_arrays.update({f"{domain}_{key}": value for key, value in arrays.items()})

    manifest_path = ROOT / program["inputs"]["bin_manifest"]["path"]
    manifest_body, manifest_body_sha, _ = fixed.load_bin_manifest(manifest_path)
    upper_edges = _merged_upper_edges(manifest_body)
    upper_k = np.asarray([upper_edges[int(bin_id)] for bin_id in bin_ids])
    all_arrays["upper_k_h_Mpc"] = upper_k
    field_frontier = frontier.evaluate_field_frontiers(
        upper_k,
        np.asarray(all_metrics["delta"]["strict_gate"], dtype=bool),
        np.asarray(all_metrics["theta"]["strict_gate"], dtype=bool),
    )

    fidelity_names = (
        "BGc_selected_group_count",
        "observed_distance_KS",
        "distance_error_mag_KS",
        "redshift_velocity_KS",
        "angular_histogram_total_variation",
    )
    fidelity_arrays = {
        name: np.asarray([row[name] for row in fidelity_rows]) for name in fidelity_names
    }
    all_arrays.update({f"fidelity_{key}": value for key, value in fidelity_arrays.items()})
    fidelity_member_pass = np.asarray(
        [
            population_fidelity_gates(row, program["population_fidelity_gates"])[
                "all_pass"
            ]
            for row in fidelity_rows
        ],
        dtype=bool,
    )
    all_arrays["fidelity_member_pass"] = fidelity_member_pass
    generator_fidelity = {
        "all_64_members_pass": bool(np.all(fidelity_member_pass)),
        "passing_member_count": int(np.count_nonzero(fidelity_member_pass)),
        "BGc_selected_group_count_range": [
            int(np.min(fidelity_arrays["BGc_selected_group_count"])),
            int(np.max(fidelity_arrays["BGc_selected_group_count"])),
        ],
        "metric_maxima": {
            name: float(np.max(fidelity_arrays[name]))
            for name in fidelity_names
            if name != "BGc_selected_group_count"
        },
    }
    all_strict = bool(
        generator_fidelity["all_64_members_pass"]
        and np.all(all_arrays["delta_strict_gate"])
        and np.all(all_arrays["theta_strict_gate"])
    )
    result = {
        "schema": AGGREGATE_RESULT_SCHEMA,
        "status": "COMPLETE_64_MOCK_DEVELOPMENT_CALIBRATION_NO_SCIENCE_CLAIM",
        "program_sha256": program_sha256,
        "implementation_commit": implementation_commit,
        "implementation_source_sha256": sha256_file(__file__),
        "member_count": MOCK_COUNT,
        "posterior_draw_count": POSTERIOR_DRAW_COUNT,
        "member_artifact_hashes": member_hashes,
        "bin_manifest_body_sha256": manifest_body_sha,
        "available_merged_bin_ids": bin_ids.tolist(),
        "cumulative_upper_k_h_Mpc": upper_k.tolist(),
        "population_generator_fidelity": generator_fidelity,
        "heldout_cumulative_prediction": {
            "mean_per_row_improvement": heldout_point.tolist(),
            "bootstrap_95_interval": np.column_stack(
                (heldout_lower, heldout_upper)
            ).tolist(),
            "pass": heldout_pass.tolist(),
        },
        "domain_metrics": all_metrics,
        "development_strict_frontier_diagnostic": {
            "density_delta": _frontier_payload(field_frontier.density_delta),
            "velocity_divergence_theta": _frontier_payload(
                field_frontier.velocity_divergence_theta
            ),
            "joint": _frontier_payload(field_frontier.joint),
            "all_available_bins_and_generator_fidelity_pass": all_strict,
            "semantics": "development_diagnostic_only_not_a_promoted_constraint_frontier",
        },
        "selection_semantics": "empirical_grouped_CF4_selection_conditioned_on_clean_group_count",
        "full_survey_selection_normalization_modeled": False,
        "observed_vpec_or_vobs_used": False,
        "observed_v3k_used_for_generation": False,
        "development_only": True,
        "untouched_256_mock_validation_executed": False,
        "frontier_or_science_claim_allowed": False,
        "target_0p3_cMpc_h_claim_allowed": False,
        "next_action_requires_user_approval": True,
    }
    return result, all_arrays


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
        raise CalibrationError("publication kind must be member or aggregate")
    output = Path(output_path)
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing overwrite of {output}")
    if not output.parent.is_dir():
        raise CalibrationError("output parent must already exist")
    stage = output.parent / f".{output.name}.staging"
    try:
        stage.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"refusing existing staging directory {stage}") from None
    identity = (stage.stat().st_dev, stage.stat().st_ino)
    published = False
    try:
        fields_name = "fields.npz" if kind == "member" else "metrics.npz"
        fields_payload = fixed.deterministic_npz_bytes(arrays)
        result_payload = canonical_json_bytes(result)
        _write_exclusive(stage / fields_name, fields_payload)
        _write_exclusive(stage / "result.json", result_payload)
        manifest = {
            "schema": f"ouruniv-cf4-bgc-population-calibration-{kind}-artifact-manifest-v1",
            "status": result["status"],
            "payloads": {
                fields_name: {"sha256": hashlib.sha256(fields_payload).hexdigest(), "bytes": len(fields_payload)},
                "result.json": {"sha256": hashlib.sha256(result_payload).hexdigest(), "bytes": len(result_payload)},
            },
        }
        manifest_payload = canonical_json_bytes(manifest)
        _write_exclusive(stage / "manifest.json", manifest_payload)
        complete = {
            "schema": f"ouruniv-cf4-bgc-population-calibration-{kind}-complete-v1",
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


def validate_member(directory: str | Path, *, expected_index: int | None = None) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != EXPECTED_MEMBER_FILES:
        raise CalibrationError("member artifact file set is not exact")
    result_payload = (root / "result.json").read_bytes()
    fields_payload = (root / "fields.npz").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    manifest = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise CalibrationError("member result is not canonical JSON")
    if result.get("schema") != MEMBER_RESULT_SCHEMA or result.get("status") != "COMPLETE_DEVELOPMENT_MEMBER_NO_SCIENCE_CLAIM":
        raise CalibrationError("member result schema/status mismatch")
    if expected_index is not None and result.get("mock_index") != expected_index:
        raise CalibrationError("member mock index mismatch")
    if result.get("truth_seed") != DEVELOPMENT_TRUTH_SEED_START + result["mock_index"]:
        raise CalibrationError("member truth seed mismatch")
    if result.get("observed_vpec_or_vobs_used") is not False or result.get("observed_v3k_used_for_generation") is not False:
        raise CalibrationError("member used an observed velocity in generation")
    if result.get("numerical_gates", {}).get("all_pass") is not True:
        raise CalibrationError("member numerical gate failed")
    if result.get("frontier_or_science_claim_allowed") is not False:
        raise CalibrationError("member makes a forbidden science claim")
    fields_name = "fields.npz"
    expected_payloads = {
        fields_name: {"sha256": hashlib.sha256(fields_payload).hexdigest(), "bytes": len(fields_payload)},
        "result.json": {"sha256": hashlib.sha256(result_payload).hexdigest(), "bytes": len(result_payload)},
    }
    if manifest.get("payloads") != expected_payloads:
        raise CalibrationError("member artifact payload bindings mismatch")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest() or complete.get("COMPLETE_written_last") is not True:
        raise CalibrationError("member COMPLETE binding mismatch")
    with np.load(io.BytesIO(fields_payload), allow_pickle=False) as fields:
        required = {
            "truth_white": (fixed.N,) * 3,
            "posterior_mean_white": (fixed.N,) * 3,
            "posterior_draws_white": (POSTERIOR_DRAW_COUNT, fixed.N, fixed.N, fixed.N),
            "prior_draws_white": (POSTERIOR_DRAW_COUNT, fixed.N, fixed.N, fixed.N),
            "truth_delta": (fixed.N,) * 3,
            "truth_theta": (fixed.N,) * 3,
            "posterior_mean_delta": (fixed.N,) * 3,
            "posterior_mean_theta": (fixed.N,) * 3,
            "posterior_draws_delta": (POSTERIOR_DRAW_COUNT, fixed.N, fixed.N, fixed.N),
            "posterior_draws_theta": (POSTERIOR_DRAW_COUNT, fixed.N, fixed.N, fixed.N),
            "truth_velocity": (3, fixed.N, fixed.N, fixed.N),
            "posterior_mean_velocity": (3, fixed.N, fixed.N, fixed.N),
            "posterior_draws_velocity": (POSTERIOR_DRAW_COUNT, 3, fixed.N, fixed.N, fixed.N),
            "truth_nuisance_q": (4,),
            "posterior_mean_nuisance_q": (4,),
            "posterior_draws_nuisance_q": (POSTERIOR_DRAW_COUNT, 4),
            "prior_draws_nuisance_q": (POSTERIOR_DRAW_COUNT, 4),
            "mock_cz": (22136,),
            "mock_observed_distance": (22136,),
            "mock_distance_error_mag": (22136,),
            "mock_direction": (22136, 3),
            "mock_true_distance": (22136,),
            "mock_true_position": (22136, 3),
            "mock_true_radial_velocity": (22136,),
            "mock_distance_log_error": (22136,),
            "mock_local_truth_delta": (22136,),
            "mock_empirical_source_index": (22136,),
            "truth_delta_modes": (8538,),
            "posterior_mean_delta_modes": (8538,),
            "posterior_draws_delta_modes": (POSTERIOR_DRAW_COUNT, 8538),
            "truth_theta_modes": (8535,),
            "posterior_mean_theta_modes": (8535,),
            "posterior_draws_theta_modes": (POSTERIOR_DRAW_COUNT, 8535),
            "delta_mode_bin_index": (8538,),
            "theta_mode_bin_index": (8535,),
            "delta_prior_variance": (8538,),
            "theta_prior_variance": (8535,),
            "delta_self_conjugate": (8538,),
            "theta_self_conjugate": (8535,),
        }
        expected_names = set(required) | {"train_raw_idx", "holdout_raw_idx"}
        if set(fields.files) != expected_names:
            raise CalibrationError("member field array set is not exact")
        for name, shape in required.items():
            if name not in fields.files or fields[name].shape != shape or not np.all(np.isfinite(fields[name])):
                raise CalibrationError(f"member field missing, nonfinite, or wrong shape: {name}")
        if fields["delta_self_conjugate"].dtype != np.dtype(bool) or fields[
            "theta_self_conjugate"
        ].dtype != np.dtype(bool):
            raise CalibrationError("member self-conjugate masks are not boolean")
        for name in ("train_raw_idx", "holdout_raw_idx"):
            if fields[name].ndim != 1 or fields[name].size == 0:
                raise CalibrationError(f"member index array is empty or non-vector: {name}")
    return {
        "status": "PASS",
        "mock_index": result["mock_index"],
        "result_sha256": hashlib.sha256(result_payload).hexdigest(),
        "fields_sha256": hashlib.sha256(fields_payload).hexdigest(),
    }


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != EXPECTED_AGGREGATE_FILES:
        raise CalibrationError("aggregate artifact file set is not exact")
    result_payload = (root / "result.json").read_bytes()
    metrics_payload = (root / "metrics.npz").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    manifest = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise CalibrationError("aggregate result is not canonical JSON")
    if result.get("schema") != AGGREGATE_RESULT_SCHEMA or result.get("status") != (
        "COMPLETE_64_MOCK_DEVELOPMENT_CALIBRATION_NO_SCIENCE_CLAIM"
    ):
        raise CalibrationError("aggregate result schema/status mismatch")
    if result.get("member_count") != MOCK_COUNT or result.get("posterior_draw_count") != POSTERIOR_DRAW_COUNT:
        raise CalibrationError("aggregate count contract mismatch")
    if result.get("untouched_256_mock_validation_executed") is not False:
        raise CalibrationError("aggregate consumed forbidden validation mocks")
    if result.get("frontier_or_science_claim_allowed") is not False:
        raise CalibrationError("aggregate makes a forbidden science claim")
    expected_payloads = {
        "metrics.npz": {
            "sha256": hashlib.sha256(metrics_payload).hexdigest(),
            "bytes": len(metrics_payload),
        },
        "result.json": {
            "sha256": hashlib.sha256(result_payload).hexdigest(),
            "bytes": len(result_payload),
        },
    }
    if manifest.get("payloads") != expected_payloads:
        raise CalibrationError("aggregate artifact payload bindings mismatch")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest() or complete.get(
        "COMPLETE_written_last"
    ) is not True:
        raise CalibrationError("aggregate COMPLETE binding mismatch")
    with np.load(io.BytesIO(metrics_payload), allow_pickle=False) as metrics:
        bin_count = len(result["available_merged_bin_ids"])
        exact_names = {
            "bin_ids",
            "upper_k_h_Mpc",
            "bootstrap_mock_indices",
            "heldout_per_mock_per_row_improvement",
            "heldout_mean_per_row_improvement",
            "heldout_bootstrap_lower_2p5",
            "heldout_bootstrap_upper_97p5",
            "heldout_pass",
            "fidelity_BGc_selected_group_count",
            "fidelity_observed_distance_KS",
            "fidelity_distance_error_mag_KS",
            "fidelity_redshift_velocity_KS",
            "fidelity_angular_histogram_total_variation",
            "fidelity_member_pass",
        }
        domain_suffixes = {
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
        }
        exact_names |= {
            f"{domain}_{suffix}"
            for domain in ("delta", "theta")
            for suffix in domain_suffixes
        }
        if set(metrics.files) != exact_names:
            raise CalibrationError("aggregate metric array set is not exact")
        if metrics["bin_ids"].shape != (bin_count,) or metrics["upper_k_h_Mpc"].shape != (
            bin_count,
        ):
            raise CalibrationError("aggregate bin arrays have wrong shape")
        if metrics["bootstrap_mock_indices"].shape[1] != MOCK_COUNT:
            raise CalibrationError("aggregate bootstrap indices have wrong shape")
        for name in metrics.files:
            if not np.all(np.isfinite(metrics[name])):
                raise CalibrationError(f"aggregate metric contains nonfinite values: {name}")
    return {
        "status": "PASS",
        "member_count": MOCK_COUNT,
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
    validate = sub.add_parser("validate-member")
    validate.add_argument("--directory", required=True, type=Path)
    validate.add_argument("--expected-index", type=int)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--program", required=True, type=Path)
    aggregate.add_argument("--members-root", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--implementation-commit", required=True)
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
            report = validate_member(args.output, expected_index=args.mock_index)
        elif args.command == "validate-member":
            report = validate_member(args.directory, expected_index=args.expected_index)
        elif args.command == "aggregate":
            program, program_sha = load_program(args.program)
            result, arrays = aggregate_members(
                program, program_sha, args.members_root, args.implementation_commit
            )
            publish_directory(args.output, result, arrays, kind="aggregate")
            report = validate_aggregate(args.output)
        else:
            report = validate_aggregate(args.directory)
    except (OSError, ValueError, CalibrationError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
