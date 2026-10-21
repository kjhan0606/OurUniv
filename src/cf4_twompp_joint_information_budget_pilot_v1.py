#!/usr/bin/env python3
"""N32 velocity-plus-2M++ covariance information-budget pilot.

This is a linear, covariance-only feasibility calculation.  It combines an
isotropic per-bin surrogate for the completed CF4 velocity posterior with a
Gaussianized-Poisson Fisher operator built from the frozen metadata-consistent
2M++ subset.  It does not infer a density field, consume galaxy positions as a
field-likelihood datum, or establish an observational resolution.

The selection build and Fisher solve are separate commands because the former
needs Astropy/healpy while the latter needs the frozen PMWD environment.
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
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_SCHEMA = "ouruniv-cf4-twompp-joint-information-budget-pilot-program-v1"
RESULT_SCHEMA = "ouruniv-cf4-twompp-joint-information-budget-pilot-result-v1"
MANIFEST_SCHEMA = "ouruniv-cf4-twompp-joint-information-budget-pilot-manifest-v1"
COMPLETE_SCHEMA = "ouruniv-cf4-twompp-joint-information-budget-pilot-complete-v1"
EXPECTED_FILES = {
    "selection.npz",
    "metrics.npz",
    "result.json",
    "manifest.json",
    "COMPLETE",
}


class PilotError(ValueError):
    """Fail-closed joint-information-pilot error."""


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
        raise PilotError(f"bound {label} is absent: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise PilotError(f"bound {label} size changed")
    if sha256_file(path) != str(binding["sha256"]):
        raise PilotError(f"bound {label} hash changed")
    return path


def load_program(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    program = json.loads(raw)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise PilotError("unexpected joint information pilot program schema")
    authorization = program.get("authorization", {})
    if not authorization.get("joint_information_budget_technical_pilot", False):
        raise PilotError("joint information technical pilot is not authorized")
    if authorization.get("field_inference", True):
        raise PilotError("program improperly authorizes field inference")
    for label, binding in program["bindings"].items():
        _verify_binding(binding, label)
    return program, hashlib.sha256(raw).hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PilotError(f"cannot load bound module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def schechter_fraction(
    luminosity_distance_mpc: np.ndarray,
    apparent_bright: float | None,
    apparent_faint: float,
    absolute_bright: float,
    absolute_faint: float,
    mstar: float,
    alpha: float,
) -> np.ndarray:
    """ARES-equivalent LF fraction surviving apparent and absolute cuts."""

    from scipy.special import gammainc

    distance = np.asarray(luminosity_distance_mpc, dtype=np.float64)
    if np.any(distance <= 0.0) or not np.all(np.isfinite(distance)):
        raise PilotError("luminosity distances must be finite and positive")
    shape = alpha + 1.0
    if shape <= 0.0:
        raise PilotError("Schechter alpha must be greater than -1")
    modulus = 5.0 * np.log10(distance) + 25.0
    observed_faint = apparent_faint - modulus
    observed_bright = (
        np.full_like(distance, -np.inf)
        if apparent_bright is None
        else apparent_bright - modulus
    )
    overlap_bright = np.maximum(absolute_bright, observed_bright)
    overlap_faint = np.minimum(absolute_faint, observed_faint)
    valid = overlap_faint > overlap_bright
    x_low = np.power(10.0, 0.4 * (mstar - overlap_faint))
    x_high = np.power(10.0, 0.4 * (mstar - overlap_bright))
    denominator_low = 10.0 ** (0.4 * (mstar - absolute_faint))
    denominator_high = 10.0 ** (0.4 * (mstar - absolute_bright))
    denominator = gammainc(shape, denominator_high) - gammainc(
        shape, denominator_low
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise PilotError("Schechter population denominator is invalid")
    fraction = np.zeros_like(distance)
    fraction[valid] = (
        gammainc(shape, x_high[valid]) - gammainc(shape, x_low[valid])
    ) / denominator
    return np.clip(fraction, 0.0, 1.0)


def _retained_population(
    program: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any], Mapping[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    subset_source = Path(str(program["bindings"]["subset_implementation"]["path"]))
    subset = _load_module(subset_source, "_cf4_twompp_subset_for_joint_information_v1")
    subset_program_path = Path(str(program["bindings"]["subset_program"]["path"]))
    _, effective, _, _, _ = subset.load_program(subset_program_path)
    inputs = effective["inputs"]
    catalog = subset.v4.v3.load_catalog(inputs["twompp_catalog"]["path"])
    exclusions, _ = subset.base.read_crossmatch_exclusions(
        inputs["cf4_twompp_crossmatch"]["path"],
        int(effective["no_double_counting"]["expected_unique_2Mpp_targets_excluded"]),
    )
    distance, absolute_magnitude = subset.base.distance_and_absolute_magnitude(
        catalog["Vcmb"], catalog["Ksmag"], effective["cosmology"]
    )
    eligible, _, apparent_bin, absolute_bin = subset.base.classify_disjoint_tracer(
        catalog,
        exclusions,
        distance,
        absolute_magnitude,
        effective["tracer_design"],
    )
    manifest_path = Path(str(program["bindings"]["excluded_recnos"]["path"]))
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        recnos = np.asarray(
            [int(row["recno"]) for row in csv.DictReader(handle)], dtype=np.int64
        )
    if recnos.size != int(program["design"]["excluded_recno_count"]):
        raise PilotError("frozen exclusion count changed")
    retained = eligible & ~np.isin(catalog["recno"], recnos)
    counts = np.asarray(
        [
            np.count_nonzero(
                retained & (apparent_bin == apparent) & (absolute_bin == absolute)
            )
            for apparent in (0, 1)
            for absolute in range(3)
        ],
        dtype=np.int64,
    )
    expected = np.asarray(program["design"]["population_counts"], dtype=np.int64)
    if not np.array_equal(counts, expected) or int(counts.sum()) != 36635:
        raise PilotError("retained population counts changed")
    return subset, effective, catalog, retained, apparent_bin, absolute_bin


def _cosmology_distance_table(
    radius_cMpc_h: np.ndarray, cosmology: Mapping[str, Any]
) -> np.ndarray:
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM

    model = FlatLambdaCDM(
        H0=float(cosmology["H0_km_s_Mpc"]) * u.km / u.s / u.Mpc,
        Om0=float(cosmology["Omega_m"]),
        Ob0=float(cosmology["Omega_b"]),
        Tcmb0=float(cosmology["Tcmb_K"]) * u.K,
    )
    z_grid = np.linspace(0.0, 0.2, 20001, dtype=np.float64)
    comoving_h = model.comoving_distance(z_grid).to_value(u.Mpc) * float(
        cosmology["h"]
    )
    if float(np.max(radius_cMpc_h)) >= float(comoving_h[-1]):
        raise PilotError("redshift interpolation table does not cover selection volume")
    redshift = np.interp(radius_cMpc_h, comoving_h, z_grid)
    return model.luminosity_distance(redshift).to_value(u.Mpc)


def build_selection(program_path: str | Path, stage_path: str | Path) -> dict[str, Any]:
    """Build volume-averaged six-population expected N32 count arrays."""

    import healpy as hp
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    program, program_sha = load_program(program_path)
    stage = Path(stage_path)
    if stage.exists():
        raise PilotError("selection staging directory already exists")
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    subset, effective, catalog, retained, _, _ = _retained_population(program)
    design = program["design"]
    grid = int(design["grid_N"])
    box = float(design["box_size_cMpc_h"])
    spacing = box / grid
    quadrature = int(design["volume_quadrature_points_per_axis"])
    if quadrature != 2:
        raise PilotError("v1 freezes two-point Gauss quadrature per axis")
    nodes = np.asarray([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)])
    offsets = 0.5 * spacing * nodes
    axis = (np.arange(grid, dtype=np.float64) + 0.5) * spacing - box / 2.0
    completeness11 = subset.base.load_completeness_map(
        effective["inputs"]["completeness_11_5"]["path"],
        int(design["HEALPix_NSIDE"]),
    )
    completeness12 = subset.base.load_completeness_map(
        effective["inputs"]["completeness_12_5"]["path"],
        int(design["HEALPix_NSIDE"]),
    )
    selection = np.zeros((6, grid, grid, grid), dtype=np.float64)
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
                icrs = sg.icrs
                pixels = hp.ang2pix(
                    int(design["HEALPix_NSIDE"]),
                    0.5 * np.pi - icrs.dec.rad,
                    np.mod(icrs.ra.rad, 2.0 * np.pi),
                    nest=False,
                )
                luminosity_distance = _cosmology_distance_table(
                    radius[active], effective["cosmology"]
                )
                for apparent in (0, 1):
                    angular = (completeness11 if apparent == 0 else completeness12)[
                        pixels
                    ]
                    apparent_bright = (
                        None if apparent == 0 else float(design["bright_apparent_K_max"])
                    )
                    apparent_faint = float(
                        design[
                            "bright_apparent_K_max"
                            if apparent == 0
                            else "faint_apparent_K_max"
                        ]
                    )
                    for absolute in range(3):
                        radial = schechter_fraction(
                            luminosity_distance,
                            apparent_bright,
                            apparent_faint,
                            float(absolute_edges[absolute]),
                            float(absolute_edges[absolute + 1]),
                            float(lf["Mstar"]),
                            float(lf["alpha"]),
                        )
                        population = 3 * apparent + absolute
                        flat = selection[population].ravel()
                        flat[np.flatnonzero(active)] += angular * radial / 8.0
    population_counts = np.asarray(design["population_counts"], dtype=np.int64)
    support = selection.reshape(6, -1).sum(axis=1)
    if np.any(support <= 0.0) or not np.all(np.isfinite(selection)):
        raise PilotError("a population selection has no finite N32 support")
    expected_counts = (
        population_counts[:, None, None, None]
        * selection
        / support[:, None, None, None]
    )
    if not np.allclose(
        expected_counts.reshape(6, -1).sum(axis=1), population_counts, rtol=2e-13
    ):
        raise PilotError("expected-count normalization failed")
    bias = np.asarray(design["reference_bias_by_population"], dtype=np.float64)
    if bias.shape != (6,) or np.any(bias <= 0.0):
        raise PilotError("reference bias vector is invalid")
    selection_path = stage / "selection.npz"
    np.savez_compressed(
        selection_path,
        selection=selection,
        expected_counts=expected_counts,
        population_counts=population_counts,
        reference_bias=bias,
    )
    retained_magnitude = np.asarray(catalog["Ksmag"])[retained]
    return {
        "program_sha256": program_sha,
        "selection_sha256": sha256_file(selection_path),
        "selection_bytes": selection_path.stat().st_size,
        "retained_count": int(retained.sum()),
        "retained_Ksmag_minimum": float(retained_magnitude.min()),
        "retained_Ksmag_maximum": float(retained_magnitude.max()),
        "population_counts": population_counts.tolist(),
        "selection_support_sum": support.tolist(),
        "expected_count_sum": expected_counts.reshape(6, -1).sum(axis=1).tolist(),
        "positive_selection_voxel_fraction": np.mean(
            selection.reshape(6, -1) > 0.0, axis=1
        ).tolist(),
        "volume_quadrature_subpoints_per_voxel": 8,
    }


def _load_fixed_module(program: Mapping[str, Any]) -> Any:
    path = Path(str(program["bindings"]["fixed_field_kernels"]["path"]))
    return _load_module(path, "_cf4_fixed_field_kernels_for_joint_information_v1")


def _full_mode_assignment(
    fixed: Any, manifest_path: Path, grid: int, box: float
) -> tuple[dict[str, Any], np.ndarray]:
    body, _, _ = fixed.load_bin_manifest(manifest_path)
    plan = fixed.global_merged_mode_plan(body, grid_size=grid, box_size=box)
    signed = np.rint(np.fft.fftfreq(grid) * grid).astype(np.int64)
    frequency = 2.0 * np.pi * signed / box
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    assignment = np.full(kmag.shape, -1, dtype=np.int64)
    native = np.full(kmag.shape, -1, dtype=np.int64)
    for item in body["native_bins"]:
        lower = float(item["lower_h_Mpc"])
        upper = float(item["upper_h_Mpc"])
        inside = (kmag >= lower) & (
            (kmag < upper)
            | (bool(item["terminal_upper_inclusive"]) & (kmag <= upper))
        )
        native[inside] = int(item["index"])
    native_to_merged = np.full(38, -1, dtype=np.int64)
    for item in body["merged_bins"]:
        native_to_merged[np.asarray(item["native_bin_indices"], dtype=int)] = int(
            item["merged_bin_index"]
        )
    valid = native >= 0
    assignment[valid] = native_to_merged[native[valid]]
    return plan, assignment


def _velocity_precision(
    program: Mapping[str, Any], assignment: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(str(program["bindings"]["velocity_metrics"]["path"]))
    scenario = str(program["design"]["velocity_baseline_scenario"])
    with np.load(path, allow_pickle=False) as metrics:
        bin_ids = np.asarray(metrics["bin_ids"], dtype=np.int64)
        delta_ids = np.asarray(metrics["delta_bin_ids"], dtype=np.int64)
        theta_ids = np.asarray(metrics["theta_bin_ids"], dtype=np.int64)
        delta_trace = np.asarray(
            metrics[
                f"scenario_{scenario}_delta_posterior_prior_trace_fraction"
            ],
            dtype=np.float64,
        )
        theta_trace = np.asarray(
            metrics[
                f"scenario_{scenario}_theta_posterior_prior_trace_fraction"
            ],
            dtype=np.float64,
        )
    residual = np.ones(bin_ids.size, dtype=np.float64)
    for column, bin_id in enumerate(bin_ids):
        delta_match = np.flatnonzero(delta_ids == bin_id)
        theta_match = np.flatnonzero(theta_ids == bin_id)
        if delta_match.size == 1 and theta_match.size == 1:
            residual[column] = max(
                float(delta_trace[delta_match[0]]), float(theta_trace[theta_match[0]])
            )
    if np.any(residual <= 0.0) or np.any(residual > 1.1):
        raise PilotError("velocity residual spectrum is invalid")
    residual = np.clip(residual, 1.0e-6, 1.0)
    precision = np.ones(assignment.shape, dtype=np.float64)
    for column, bin_id in enumerate(bin_ids):
        precision[assignment == bin_id] = 1.0 / residual[column]
    return precision, bin_ids, residual, theta_ids


def canonical_probe(
    grid: int, flat: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian probe with unit variance in each canonical complex mode."""

    rng = np.random.default_rng(seed)
    probe_k = np.zeros((grid, grid, grid), dtype=np.complex128)
    coordinates = np.column_stack(np.unravel_index(flat, (grid,) * 3))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(flat.size, 2))
    for index, coordinate in enumerate(coordinates):
        conjugate = (-coordinate) % grid
        self_conjugate = np.array_equal(coordinate, conjugate)
        value = (
            complex(signs[index, 0], 0.0)
            if self_conjugate
            else complex(signs[index, 0], signs[index, 1]) / math.sqrt(2.0)
        )
        probe_k[tuple(coordinate)] = value
        probe_k[tuple(conjugate)] = np.conjugate(value)
    probe = np.fft.ifftn(probe_k, norm="ortho").real
    return probe, probe_k.ravel()[flat]


def _apply_transfer(field: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(
        np.fft.fftn(field, norm="ortho") * transfer, norm="ortho"
    ).real


def joint_trace_spectrum(
    *,
    transfer: np.ndarray,
    velocity_precision: np.ndarray,
    plan: Mapping[str, Any],
    expected_counts: np.ndarray,
    bias: np.ndarray,
    bin_ids: np.ndarray,
    retention: float,
    marginalize_normalizations: bool,
    probe_count: int,
    probe_seed: int,
    cg_rtol: float,
    cg_maxiter: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Estimate the prior-weighted joint posterior trace by Hutchinson probes."""

    from scipy.sparse.linalg import LinearOperator, cg

    grid = transfer.shape[0]
    shape = (grid,) * 3
    size = grid**3
    flat = np.asarray(plan["flat_independent_field_indices"], dtype=np.int64)
    mode_bin = np.asarray(plan["mode_merged_bin_index"], dtype=np.int64)
    transfer_flat = transfer.ravel()[flat]
    denominator = np.asarray(
        [np.sum(transfer_flat[mode_bin == bin_id] ** 2) for bin_id in bin_ids],
        dtype=np.float64,
    )
    if np.any(denominator <= 0.0):
        raise PilotError("an information bin has zero prior variance")
    lambdas = np.asarray(expected_counts, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    density_diagonal = np.sum(lambdas * bias[:, None, None, None] ** 2, axis=0)
    normalization = lambdas.reshape(6, -1).sum(axis=1)

    def matvec(vector: np.ndarray) -> np.ndarray:
        field = np.asarray(vector, dtype=np.float64).reshape(shape)
        velocity = np.fft.ifftn(
            np.fft.fftn(field, norm="ortho") * velocity_precision,
            norm="ortho",
        ).real
        delta = _apply_transfer(field, transfer)
        middle = density_diagonal * delta
        if marginalize_normalizations:
            for population in range(6):
                cross = lambdas[population] * bias[population]
                coefficient = float(np.vdot(cross, delta).real / normalization[population])
                middle -= cross * coefficient
        return (velocity + retention * _apply_transfer(middle, transfer)).ravel()

    mean_density = float(np.mean(density_diagonal))
    preconditioner_spectrum = velocity_precision + retention * mean_density * transfer**2

    def precondition(vector: np.ndarray) -> np.ndarray:
        field = np.asarray(vector, dtype=np.float64).reshape(shape)
        return np.fft.ifftn(
            np.fft.fftn(field, norm="ortho") / preconditioner_spectrum,
            norm="ortho",
        ).real.ravel()

    operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    preconditioner = LinearOperator(
        (size, size), matvec=precondition, dtype=np.float64
    )
    trace_probes = np.empty((probe_count, bin_ids.size), dtype=np.float64)
    relative_residuals = np.empty(probe_count, dtype=np.float64)
    iterations = np.zeros(probe_count, dtype=np.int64)
    for probe_index in range(probe_count):
        rhs, canonical = canonical_probe(grid, flat, probe_seed + probe_index)

        def callback(_: np.ndarray) -> None:
            iterations[probe_index] += 1

        solution, info = cg(
            operator,
            rhs.ravel(),
            M=preconditioner,
            rtol=cg_rtol,
            atol=0.0,
            maxiter=cg_maxiter,
            callback=callback,
        )
        if info != 0:
            raise PilotError(f"CG failed for trace probe {probe_index}: info={info}")
        residual_vector = matvec(solution) - rhs.ravel()
        relative_residuals[probe_index] = np.linalg.norm(residual_vector) / np.linalg.norm(
            rhs
        )
        solution_k = np.fft.fftn(solution.reshape(shape), norm="ortho").ravel()[flat]
        contribution = transfer_flat**2 * np.real(np.conjugate(canonical) * solution_k)
        for column, bin_id in enumerate(bin_ids):
            trace_probes[probe_index, column] = (
                np.sum(contribution[mode_bin == bin_id]) / denominator[column]
            )
    if np.max(relative_residuals) > max(10.0 * cg_rtol, 1.0e-4):
        raise PilotError("joint trace solve residual exceeds the numerical gate")
    trace = np.mean(trace_probes, axis=0)
    trace_standard_error = np.std(trace_probes, axis=0, ddof=1) / math.sqrt(probe_count)
    information = 1.0 - trace
    information_lower = 1.0 - (trace + 1.96 * trace_standard_error)
    correlation = np.sqrt(np.clip(information, 0.0, 1.0))
    metrics = {
        "posterior_prior_trace_fraction": trace.tolist(),
        "trace_probe_standard_error": trace_standard_error.tolist(),
        "recovered_information_fraction": information.tolist(),
        "recovered_information_numerical_95_lower": information_lower.tolist(),
        "expected_response": information.tolist(),
        "expected_correlation_r": correlation.tolist(),
        "expected_residual_power_ratio": trace.tolist(),
        "maximum_CG_relative_residual": float(np.max(relative_residuals)),
        "maximum_CG_iterations": int(np.max(iterations)),
    }
    arrays = {
        "trace_probes": trace_probes,
        "posterior_prior_trace_fraction": trace,
        "trace_probe_standard_error": trace_standard_error,
        "recovered_information_fraction": information,
        "recovered_information_numerical_95_lower": information_lower,
        "expected_correlation_r": correlation,
        "CG_relative_residual": relative_residuals,
        "CG_iterations": iterations,
    }
    return metrics, arrays


def _gate_metrics(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> np.ndarray:
    information = np.asarray(metrics["recovered_information_fraction"])
    lower = np.asarray(metrics["recovered_information_numerical_95_lower"])
    correlation = np.asarray(metrics["expected_correlation_r"])
    residual = np.asarray(metrics["expected_residual_power_ratio"])
    return (
        (information >= float(gates["response_min_inclusive"]))
        & (lower >= float(gates["numerical_information_lower_min_inclusive"]))
        & (correlation >= float(gates["correlation_r_min_inclusive"]))
        & (residual <= float(gates["residual_power_ratio_max_inclusive"]))
    )


def run_pilot(
    program_path: str | Path,
    stage_path: str | Path,
    output_path: str | Path,
    implementation_commit: str,
) -> dict[str, Any]:
    program, program_sha = load_program(program_path)
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise PilotError("implementation commit must be lowercase 40-hex")
    stage = Path(stage_path)
    target = Path(output_path)
    if not stage.is_dir() or not (stage / "selection.npz").is_file():
        raise PilotError("selection staging artifact is absent")
    if target.exists():
        raise PilotError("joint pilot output already exists")
    design = program["design"]
    fixed = _load_fixed_module(program)
    args = fixed.frozen_args(ROOT / "data/cf4_clean.npz")
    transfer, growth_rate = fixed.build_density_transfer(args)
    grid = int(design["grid_N"])
    if transfer.shape != (grid,) * 3:
        raise PilotError("frozen density transfer shape changed")
    manifest_path = Path(str(program["bindings"]["bin_manifest"]["path"]))
    plan, full_assignment = _full_mode_assignment(
        fixed, manifest_path, grid, float(design["box_size_cMpc_h"])
    )
    velocity_precision, bin_ids, velocity_residual, theta_ids = _velocity_precision(
        program, full_assignment
    )
    with np.load(stage / "selection.npz", allow_pickle=False) as selection:
        expected_counts = np.asarray(selection["expected_counts"], dtype=np.float64)
        bias = np.asarray(selection["reference_bias"], dtype=np.float64)
        population_counts = np.asarray(selection["population_counts"], dtype=np.int64)
        selection_summary = {
            "selection_sha256": sha256_file(stage / "selection.npz"),
            "selection_bytes": (stage / "selection.npz").stat().st_size,
            "population_counts": population_counts.tolist(),
            "expected_count_sum": expected_counts.reshape(6, -1).sum(axis=1).tolist(),
            "positive_expected_count_voxel_fraction": np.mean(
                expected_counts.reshape(6, -1) > 0.0, axis=1
            ).tolist(),
        }
    scenarios: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "bin_ids": bin_ids,
        "theta_available_bin_ids": theta_ids,
        "velocity_baseline_posterior_prior_trace_fraction": velocity_residual,
        "velocity_baseline_recovered_information_fraction": 1.0 - velocity_residual,
        "reference_bias_by_population": bias,
        "population_counts": population_counts,
    }
    gates = design["information_gates"]
    scenario_order = list(design["density_scenarios"])
    for scenario_name in scenario_order:
        specification = design["density_scenarios"][scenario_name]
        metrics, scenario_arrays = joint_trace_spectrum(
            transfer=transfer,
            velocity_precision=velocity_precision,
            plan=plan,
            expected_counts=expected_counts,
            bias=bias,
            bin_ids=bin_ids,
            retention=float(specification["density_Fisher_retention"]),
            marginalize_normalizations=bool(
                specification["marginalize_population_normalizations"]
            ),
            probe_count=int(design["trace_probe_count"]),
            probe_seed=int(design["trace_probe_seed"]),
            cg_rtol=float(design["CG_relative_tolerance"]),
            cg_maxiter=int(design["CG_max_iterations"]),
        )
        strict = _gate_metrics(metrics, gates)
        theta_available = np.isin(bin_ids, theta_ids)
        joint_strict = strict & theta_available
        prefix = 0
        for value in joint_strict:
            if not value:
                break
            prefix += 1
        metrics["strict_gate"] = strict.tolist()
        metrics["joint_delta_theta_strict_gate"] = joint_strict.tolist()
        metrics["joint_contiguous_prefix_bin_count"] = prefix
        metrics["lowest_joint_bin_strict_pass"] = bool(joint_strict[0])
        scenarios[scenario_name] = {
            "semantics": specification,
            "metrics": metrics,
        }
        for key, value in scenario_arrays.items():
            arrays[f"scenario_{scenario_name}_{key}"] = value
        arrays[f"scenario_{scenario_name}_strict_gate"] = strict
        arrays[f"scenario_{scenario_name}_joint_delta_theta_strict_gate"] = joint_strict
    ceiling_name = str(design["known_ceiling_scenario"])
    ceiling_pass = bool(scenarios[ceiling_name]["metrics"]["lowest_joint_bin_strict_pass"])
    status = (
        "PASS_KNOWN_CEILING_ROUTE_PROMISING_NO_PARENT_PROMOTION"
        if ceiling_pass
        else "NO_GO_KNOWN_CEILING_INSUFFICIENT_NO_PARENT_PROMOTION"
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "program_sha256": program_sha,
        "implementation_commit": implementation_commit,
        "implementation_source_sha256": sha256_file(__file__),
        "grid_N": grid,
        "box_size_cMpc_h": float(design["box_size_cMpc_h"]),
        "cell_size_cMpc_h": float(design["box_size_cMpc_h"]) / grid,
        "growth_rate_z0": float(growth_rate),
        "selection": selection_summary,
        "velocity_surrogate": {
            "scenario": design["velocity_baseline_scenario"],
            "posterior_prior_trace_fraction": velocity_residual.tolist(),
            "recovered_information_fraction": (1.0 - velocity_residual).tolist(),
            "construction": "isotropic diagonal Fourier precision matching the completed per-bin mean posterior trace; full velocity covariance unavailable",
        },
        "scenarios": scenarios,
        "known_ceiling_lowest_joint_bin_strict_pass": ceiling_pass,
        "technical_validation_pass": True,
        "truth_array_generated_or_deserialized": False,
        "galaxy_positions_consumed_as_field_likelihood_datum": False,
        "observed_population_totals_used_for_shot_noise_normalization": True,
        "field_inference_executed": False,
        "present_density_posterior_created": False,
        "IC_inference_executed": False,
        "observational_resolution_0p3_cMpc_h_established": False,
        "parent_posterior_promotion_allowed": False,
        "exact_bias_RSD_discrepancy_marginalization_executed": False,
        "interpretation": (
            "This covariance-only N32 pilot can reject an inadequate known-selection "
            "reference-bias ceiling or mark the independent density-tracer route as "
            "promising. It cannot promote a parent posterior. Exact bias, RSD, FoG, "
            "selection-calibration, and model-discrepancy marginalization requires "
            "development mocks and a datum-dependent forward likelihood."
        ),
        "next_action_requires_user_approval": True,
    }
    metrics_path = stage / "metrics.npz"
    np.savez_compressed(metrics_path, **arrays)
    result_bytes = canonical_json_bytes(result)
    (stage / "result.json").write_bytes(result_bytes)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "files": {
            name: {
                "bytes": (stage / name).stat().st_size,
                "sha256": sha256_file(stage / name),
            }
            for name in ("selection.npz", "metrics.npz", "result.json")
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (stage / "manifest.json").write_bytes(manifest_bytes)
    complete = {
        "schema": COMPLETE_SCHEMA,
        "status": status,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "metrics_sha256": sha256_file(metrics_path),
        "selection_sha256": sha256_file(stage / "selection.npz"),
    }
    (stage / "COMPLETE").write_bytes(canonical_json_bytes(complete))
    os.rename(stage, target)
    return validate_pilot(target)


def validate_pilot(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise PilotError("joint pilot artifact file set is not exact")
    manifest_raw = (root / "manifest.json").read_bytes()
    result_raw = (root / "result.json").read_bytes()
    manifest = json.loads(manifest_raw)
    result = json.loads(result_raw)
    complete = json.loads((root / "COMPLETE").read_bytes())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PilotError("joint pilot manifest schema changed")
    if result.get("schema") != RESULT_SCHEMA or not result.get(
        "technical_validation_pass", False
    ):
        raise PilotError("joint pilot result is not technically valid")
    for name, binding in manifest["files"].items():
        _verify_binding({"path": root / name, **binding}, f"published {name}")
    if complete.get("schema") != COMPLETE_SCHEMA:
        raise PilotError("joint pilot COMPLETE schema changed")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest():
        raise PilotError("joint pilot manifest binding changed")
    if complete.get("result_sha256") != hashlib.sha256(result_raw).hexdigest():
        raise PilotError("joint pilot result binding changed")
    if result.get("field_inference_executed") or result.get(
        "parent_posterior_promotion_allowed"
    ):
        raise PilotError("joint pilot crossed its inference firewall")
    with np.load(root / "metrics.npz", allow_pickle=False) as metrics:
        if "bin_ids" not in metrics or metrics["bin_ids"].ndim != 1:
            raise PilotError("joint pilot metrics are incomplete")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-selection")
    build.add_argument("--program", required=True)
    build.add_argument("--stage", required=True)
    run = sub.add_parser("run-pilot")
    run.add_argument("--program", required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--implementation-commit", required=True)
    validate = sub.add_parser("validate-pilot")
    validate.add_argument("--directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-selection":
        print(json.dumps(build_selection(args.program, args.stage), sort_keys=True))
    elif args.command == "run-pilot":
        result = run_pilot(
            args.program, args.stage, args.output, args.implementation_commit
        )
        print(json.dumps({"status": result["status"]}, sort_keys=True))
    else:
        result = validate_pilot(args.directory)
        print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
