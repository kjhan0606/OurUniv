#!/usr/bin/env python3
"""Freeze reference-only calibration for the V8 N64-to-N192 mode audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cf4_lg_peak_cr import free_rfft_mask
from cf4_linear_cr import build_forward, prepare_catalog


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def profile_gaussian_nuisance(
    observed: np.ndarray,
    prediction: np.ndarray,
    variance: np.ndarray,
    design: np.ndarray,
    prior_sigma: np.ndarray,
) -> dict[str, Any]:
    """Analytically profile/marginalize the common Gaussian bulk/H0 nuisance.

    The nuisance-marginalized log likelihood differs from ``-deviance/2`` only
    by a field-independent log determinant and normalization.
    """
    observed = np.asarray(observed, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    prior_sigma = np.asarray(prior_sigma, dtype=np.float64)
    if observed.shape != prediction.shape or observed.shape != variance.shape:
        raise ValueError("observed, prediction, and variance must have one shape")
    if design.shape != (observed.size, prior_sigma.size):
        raise ValueError("nuisance design has the wrong shape")
    if np.any(variance <= 0.0) or np.any(prior_sigma <= 0.0):
        raise ValueError("noise variances and nuisance prior sigmas must be positive")

    raw = observed - prediction
    precision = 1.0 / variance
    normal = design.T @ (precision[:, None] * design)
    normal += np.diag(1.0 / prior_sigma**2)
    rhs = design.T @ (precision * raw)
    qhat = np.linalg.solve(normal, rhs)
    residual = raw - design @ qhat
    data_term = float(np.sum(residual**2 * precision))
    prior_term = float(np.sum((qhat / prior_sigma) ** 2))
    deviance = data_term + prior_term
    woodbury = float(
        np.sum(raw**2 * precision) - rhs @ np.linalg.solve(normal, rhs)
    )
    if not math.isclose(deviance, woodbury, rel_tol=2e-11, abs_tol=2e-7):
        raise RuntimeError("profile and Woodbury nuisance deviances disagree")
    return {
        "qhat": qhat,
        "residual": residual,
        "standardized_residual": residual / np.sqrt(variance),
        "data_deviance": data_term,
        "nuisance_prior_deviance": prior_term,
        "marginal_deviance": deviance,
    }


def radial_residual_metrics(
    standardized_residual: np.ndarray,
    cz: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(standardized_residual, dtype=np.float64)
    cz = np.asarray(cz, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    bias, rms, count = [], [], []
    for number, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (cz >= low) & (cz < high)
        if number == len(edges) - 2:
            selected = (cz >= low) & (cz <= high)
        if not np.any(selected):
            raise ValueError(f"empty CF4 radial bin {low:g}--{high:g} km/s")
        values = residual[selected]
        bias.append(float(values.mean()))
        rms.append(float(np.sqrt(np.mean(values**2))))
        count.append(int(values.size))
    return np.asarray(bias), np.asarray(rms), np.asarray(count)


def parse_shell_edges(values: list[float | None]) -> np.ndarray:
    """Parse a JSON-safe final ``null`` as the required positive infinity."""
    if len(values) < 2 or values[-1] is not None:
        raise ValueError("released shell edges must end in JSON null for +infinity")
    parsed = np.asarray([
        np.inf if value is None else float(value) for value in values
    ], dtype=np.float64)
    if parsed[0] != 0.0 or np.any(np.diff(parsed) <= 0.0):
        raise ValueError("released shell edges must increase strictly from zero")
    return parsed


def released_shell_geometry(
    n: int,
    box_size: float,
    frozen_n: int,
    edges: np.ndarray,
) -> dict[str, Any]:
    edges = np.asarray(edges, dtype=np.float64)
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box_size / n)
    k2 = (
        kx[:, None, None] ** 2
        + kx[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    kmag = np.sqrt(k2)
    released = free_rfft_mask(n, frozen_n)
    hermitian_weight = np.full((1, 1, n // 2 + 1), 2.0, dtype=np.float64)
    hermitian_weight[..., 0] = 1.0
    hermitian_weight[..., -1] = 1.0
    masks, weight_sums, mode_counts = [], [], []
    covered = np.zeros_like(released)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = released & (kmag >= low) & (kmag < high)
        if not np.any(mask):
            raise ValueError(f"empty released Fourier shell {low:g}--{high:g}")
        if np.any(covered & mask):
            raise RuntimeError("released Fourier shells overlap")
        covered |= mask
        masks.append(mask)
        weight_sums.append(float(np.broadcast_to(hermitian_weight, mask.shape)[mask].sum()))
        mode_counts.append(int(mask.sum()))
    if not np.array_equal(covered, released):
        raise RuntimeError("released Fourier shells do not exactly partition F192 minus F64")
    return {
        "released_mask": released,
        "weights": hermitian_weight,
        "masks": masks,
        "weight_sums": np.asarray(weight_sums),
        "mode_counts_rfft": np.asarray(mode_counts),
    }


def released_shell_metrics(
    field: np.ndarray,
    reference_mean_fft: np.ndarray,
    parent_fft: np.ndarray,
    geometry: dict[str, Any],
) -> dict[str, np.ndarray]:
    field_fft = np.fft.rfftn(np.asarray(field, dtype=np.float64), norm="ortho")
    weight = np.broadcast_to(geometry["weights"], field_fft.shape)
    rows = {"Eres": [], "Pwhite": [], "delta_E_parent3429": []}
    for mask, denominator in zip(geometry["masks"], geometry["weight_sums"]):
        rows["Eres"].append(float(
            np.sum(weight[mask] * np.abs(field_fft[mask] - reference_mean_fft[mask]) ** 2)
            / denominator
        ))
        rows["Pwhite"].append(float(
            np.sum(weight[mask] * np.abs(field_fft[mask]) ** 2) / denominator
        ))
        rows["delta_E_parent3429"].append(float(
            np.sum(weight[mask] * np.abs(field_fft[mask] - parent_fft[mask]) ** 2)
            / denominator
        ))
    return {key: np.asarray(value) for key, value in rows.items()}


def higher_tail_conformal_p(value: float, calibration: np.ndarray) -> float:
    calibration = np.asarray(calibration, dtype=np.float64)
    return float((1 + np.count_nonzero(calibration >= value)) / (calibration.size + 1))


def summary_coordinates(matrix: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Median, Q90, and Q99 exceedance fraction for correlated coordinates."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(q99):
        raise ValueError("summary matrix and Q99 vector do not align")
    return np.concatenate((
        np.quantile(matrix, 0.5, axis=0, method="linear"),
        np.quantile(matrix, 0.9, axis=0, method="linear"),
        np.mean(matrix > q99[None, :], axis=0, dtype=np.float64),
    ))


def bootstrap_simultaneous_calibration(
    families: dict[str, np.ndarray],
    iterations: int,
    seed: int,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Reference-row bootstrap with correlation-preserving common resamples."""
    if not families:
        raise ValueError("at least one calibration family is required")
    sizes = {np.asarray(value).shape[0] for value in families.values()}
    if len(sizes) != 1:
        raise ValueError("all calibration families must share reference rows")
    nrow = sizes.pop()
    prepared = {}
    total_statistics = 0
    for name, values in families.items():
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError(f"family {name} must be a finite 2-D matrix")
        q99 = np.quantile(matrix, 0.99, axis=0, method="linear")
        baseline = summary_coordinates(matrix, q99)
        prepared[name] = {"matrix": matrix, "q99": q99, "baseline": baseline}
        total_statistics += baseline.size

    bootstrap = np.empty((iterations, total_statistics), dtype=np.float32)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        indices = rng.integers(0, nrow, size=(stop - start, nrow))
        offset = 0
        for item in prepared.values():
            draws = item["matrix"][indices]
            ncoord = item["matrix"].shape[1]
            stats = np.concatenate((
                np.quantile(draws, 0.5, axis=1, method="linear"),
                np.quantile(draws, 0.9, axis=1, method="linear"),
                np.mean(
                    draws > item["q99"][None, None, :], axis=1, dtype=np.float64
                ),
            ), axis=1)
            bootstrap[start:stop, offset:offset + 3 * ncoord] = stats
            offset += 3 * ncoord

    baseline = np.concatenate([item["baseline"] for item in prepared.values()])
    differences = bootstrap.astype(np.float64) - baseline[None, :]
    scale = differences.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        bad = np.flatnonzero((~np.isfinite(scale)) | (scale <= 0.0)).tolist()
        raise RuntimeError(f"degenerate bootstrap studentization coordinates: {bad}")
    maxima = np.max(differences / scale[None, :], axis=1)
    critical = float(np.quantile(maxima, 0.99, method="linear"))

    output_families = {}
    offset = 0
    for name, item in prepared.items():
        ncoord = item["matrix"].shape[1]
        width = 3 * ncoord
        output_families[name] = {
            "n_rows": nrow,
            "n_coordinates": ncoord,
            "coordinate_q99": item["q99"],
            "reference_summary": item["baseline"],
            "bootstrap_studentization_scale": scale[offset:offset + width],
        }
        offset += width
    return {
        "iterations": iterations,
        "rng": "NumPy Generator PCG64DXSM",
        "seed": seed,
        "quantile_method": "linear",
        "simultaneous_one_sided_alpha": 0.01,
        "simultaneous_studentized_max_critical": critical,
        "families": output_families,
    }


def mahalanobis_distance(
    values: np.ndarray, centre: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 2 or centre.shape != (values.shape[1],):
        raise ValueError("Mahalanobis values and centre do not align")
    if covariance.shape != (values.shape[1], values.shape[1]):
        raise ValueError("Mahalanobis covariance has the wrong shape")
    inverse = np.linalg.inv(covariance)
    residual = values - centre[None, :]
    squared = np.einsum("ni,ij,nj->n", residual, inverse, residual)
    if np.any(squared < -1e-10):
        raise RuntimeError("negative Mahalanobis squared distance")
    return np.sqrt(np.maximum(squared, 0.0))


def load_program(path: Path) -> dict[str, Any]:
    program = json.loads(path.read_text())
    if program.get("status") != "frozen_before_reference_only_CF4_calibration":
        raise RuntimeError("reference calibration program is not frozen")
    if program["information_firewall"]["V8_projection_CF4_metrics_opened"]:
        raise RuntimeError("reference-only information firewall is open")
    return program


def validate_source(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected_sha256}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    program = load_program(args.program)
    canonical_out = Path(program["storage"]["canonical_output"])
    if args.out.resolve() != canonical_out.resolve():
        raise RuntimeError("output path differs from the frozen canonical output")
    if args.out.exists():
        raise FileExistsError(
            f"canonical reference calibration already exists and is immutable: {args.out}"
        )

    authorization = ROOT / program["authorization"]["path"]
    validate_source(authorization, program["authorization"]["sha256"], "authorization")
    parent_manifest_path = Path(program["reference"]["manifest"])
    validate_source(
        parent_manifest_path,
        program["reference"]["manifest_sha256"],
        "reference manifest",
    )
    parent_manifest = json.loads(parent_manifest_path.read_text())
    catalog_path = Path(parent_manifest["catalog"])
    validate_source(catalog_path, program["reference"]["catalog_sha256"], "CF4 catalog")

    expected_seeds = list(range(
        int(program["reference"]["seed_range_inclusive"][0]),
        int(program["reference"]["seed_range_inclusive"][1]) + 1,
    ))
    if parent_manifest["configuration"]["sample_seeds"] != expected_seeds:
        raise RuntimeError("reference manifest seed bank differs from frozen program")
    if len(parent_manifest["outputs"]) != len(expected_seeds):
        raise RuntimeError("reference manifest output count differs from seed bank")

    config = argparse.Namespace(**parent_manifest["configuration"])
    data = prepare_catalog(config)
    if np.any(data["holdout"]):
        raise RuntimeError("reference manifest must use the all-data CF4 likelihood")
    if data["raw_idx"].size != int(program["reference"]["expected_CF4_rows"]):
        raise RuntimeError("unexpected CF4 likelihood row count")
    forward, _, _, npdtype = build_forward(data["pos"], data["rhat"], config)
    import jax.numpy as jnp

    radial_edges = np.asarray(program["radial_residuals"]["cz_edges_km_s"], dtype=np.float64)
    shell_edges = parse_shell_edges(
        program["released_modes"]["shell_edges_h_mpc"]
    )
    geometry = released_shell_geometry(
        int(config.N), float(config.box_size),
        int(program["released_modes"]["frozen_mesh_size"]), shell_edges,
    )

    parent_seed = int(program["reference"]["parent_seed"])
    parent_index = expected_seeds.index(parent_seed)
    parent_path = Path(parent_manifest["outputs"][parent_index])
    validate_source(
        parent_path, program["reference"]["parent_field_sha256"], "parent 3429 field"
    )
    with np.load(parent_path, allow_pickle=False) as item:
        reference_mean = item["s_map"].astype(np.float64)
        parent_field = item["s_out"].astype(np.float64)
    reference_mean_fft = np.fft.rfftn(reference_mean, norm="ortho")
    parent_fft = np.fft.rfftn(parent_field, norm="ortho")

    rows = []
    field_hashes = []
    for number, (seed, output) in enumerate(zip(expected_seeds, parent_manifest["outputs"]), 1):
        path = Path(output)
        digest = sha256_file(path)
        with np.load(path, allow_pickle=False) as item:
            actual_seed = int(item["sample_seed"])
            field = item["s_out"].astype(np.float64)
            stored_mean = item["s_map"].astype(np.float64)
            if int(item["N"]) != int(config.N) or not np.isclose(item["L"], config.box_size):
                raise RuntimeError(f"reference geometry mismatch for seed {seed}")
        if actual_seed != seed:
            raise RuntimeError(f"reference file seed mismatch: expected {seed}, got {actual_seed}")
        if not np.array_equal(stored_mean, reference_mean):
            raise RuntimeError(f"reference Wiener mean differs for seed {seed}")
        prediction = np.asarray(forward(jnp.asarray(field, dtype=npdtype)), dtype=np.float64)
        nuisance = profile_gaussian_nuisance(
            data["vobs"], prediction, data["variance"], data["B"], data["q_std"]
        )
        bias, radial_rms, radial_count = radial_residual_metrics(
            nuisance["standardized_residual"], data["cz"], radial_edges
        )
        shells = released_shell_metrics(
            field, reference_mean_fft, parent_fft, geometry
        )
        rows.append({
            "seed": seed,
            "field": str(path.resolve()),
            "field_sha256": digest,
            "marginal_deviance": nuisance["marginal_deviance"],
            "deviance_per_CF4_row": nuisance["marginal_deviance"] / data["raw_idx"].size,
            "data_deviance": nuisance["data_deviance"],
            "nuisance_prior_deviance": nuisance["nuisance_prior_deviance"],
            "qhat": nuisance["qhat"],
            "radial_bias": bias,
            "radial_rms": radial_rms,
            "released_Eres": shells["Eres"],
            "released_Pwhite": shells["Pwhite"],
            "released_delta_E_parent3429": shells["delta_E_parent3429"],
        })
        field_hashes.append({"seed": seed, "path": str(path.resolve()), "sha256": digest})
        if number % 16 == 0 or number == len(expected_seeds):
            print(f"[reference] {number}/{len(expected_seeds)}", flush=True)

    calibration_rows = [row for row in rows if row["seed"] != parent_seed]
    if len(calibration_rows) != 255:
        raise RuntimeError("parent-excluded reference calibration must contain 255 rows")
    deviance = np.asarray([row["marginal_deviance"] for row in calibration_rows])
    qhat = np.asarray([row["qhat"] for row in calibration_rows])
    radial_bias = np.asarray([row["radial_bias"] for row in calibration_rows])
    radial_rms = np.asarray([row["radial_rms"] for row in calibration_rows])
    eres = np.asarray([row["released_Eres"] for row in calibration_rows])
    pwhite = np.asarray([row["released_Pwhite"] for row in calibration_rows])
    delta_e = np.asarray([
        row["released_delta_E_parent3429"] for row in calibration_rows
    ])
    parent_deviance = rows[parent_index]["marginal_deviance"]
    parent_p = higher_tail_conformal_p(parent_deviance, deviance)

    qhat_median = np.median(qhat, axis=0)
    qhat_centre = np.mean(qhat, axis=0)
    qhat_covariance = np.cov(qhat, rowvar=False, ddof=1)
    qhat_mahalanobis = mahalanobis_distance(
        qhat, qhat_centre, qhat_covariance
    )
    bias_centre = np.median(radial_bias, axis=0)
    l4_matrix = np.column_stack((
        np.abs(qhat - qhat_median[None, :]),
        qhat_mahalanobis,
        np.abs(radial_bias - bias_centre[None, :]),
        radial_rms,
    ))
    l5_matrix = np.column_stack((eres, np.abs(pwhite - 1.0), delta_e))
    bootstrap = bootstrap_simultaneous_calibration(
        {"L4_qhat_radial": l4_matrix, "L5_released_modes": l5_matrix},
        iterations=int(program["calibration"]["bootstrap_iterations"]),
        seed=int(program["calibration"]["bootstrap_seed"]),
        chunk_size=int(program["calibration"]["bootstrap_chunk_size"]),
    )

    q99_deviance = float(np.quantile(deviance, 0.99, method="linear"))
    # The reference-row bootstrap count is already one coordinate of the
    # generic calibration; calculate its explicit 99.9% decision bound with
    # the same prescribed RNG stream in a dedicated, reproducible pass.
    iterations = int(program["calibration"]["bootstrap_iterations"])
    count_rng = np.random.Generator(np.random.PCG64DXSM(
        int(program["calibration"]["bootstrap_seed"])
    ))
    exceedance_fractions = np.empty(iterations, dtype=np.float64)
    chunk = int(program["calibration"]["bootstrap_chunk_size"])
    for start in range(0, iterations, chunk):
        stop = min(start + chunk, iterations)
        indices = count_rng.integers(0, deviance.size, size=(stop - start, deviance.size))
        exceedance_fractions[start:stop] = np.mean(
            deviance[indices] > q99_deviance, axis=1, dtype=np.float64
        )

    report = {
        "schema": "ouruniv-cf4-lg-v8-mode-release-reference-calibration-v1",
        "status": (
            "complete_reference_calibration_parent3429_pass"
            if parent_p >= float(program["gates"]["L2_parent_conformal_p_min"])
            else "complete_reference_calibration_parent3429_fail_stop"
        ),
        "program": str(args.program.resolve()),
        "program_sha256": sha256_file(args.program),
        "authorization_sha256": sha256_file(authorization),
        "reference_manifest": str(parent_manifest_path.resolve()),
        "reference_manifest_sha256": sha256_file(parent_manifest_path),
        "catalog": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "reference_field_hashes": field_hashes,
        "reference_seed_count": len(rows),
        "calibration_seed_count": len(calibration_rows),
        "excluded_parent_seed": parent_seed,
        "CF4_likelihood_rows": int(data["raw_idx"].size),
        "nuisance": {
            "parameters": ["bulk_x_km_s", "bulk_y_km_s", "bulk_z_km_s", "delta_H0_km_s_Mpc"],
            "prior_sigma": data["q_std"],
            "method": "analytic Gaussian marginal deviance via Woodbury; identical qhat profile",
        },
        "radial_residuals": {
            "cz_edges_km_s": radial_edges,
            "counts": radial_count,
            "metrics": ["bias", "RMS"],
        },
        "released_modes": {
            "mask": "exact cubical F192 minus F64 rFFT mask",
            "shell_edges_h_mpc": [
                None if np.isposinf(value) else float(value)
                for value in shell_edges
            ],
            "rfft_mode_counts": geometry["mode_counts_rfft"],
            "Hermitian_weight_sums": geometry["weight_sums"],
            "metrics": ["Eres", "Pwhite", "delta_E_parent3429"],
        },
        "L2_parent3429": {
            "marginal_deviance": parent_deviance,
            "one_sided_upper_tail_conformal_p": parent_p,
            "minimum_p": float(program["gates"]["L2_parent_conformal_p_min"]),
            "pass": parent_p >= float(program["gates"]["L2_parent_conformal_p_min"]),
        },
        "L3_reference_thresholds": {
            "deviance_Q95": float(np.quantile(deviance, 0.95, method="linear")),
            "deviance_Q99": q99_deviance,
            "deviance_Q99p5": float(np.quantile(deviance, 0.995, method="linear")),
            "Q99_exceedance_fraction_bootstrap_Q99p9": float(
                np.quantile(exceedance_fractions, 0.999, method="linear")
            ),
            "quantile_method": "linear",
        },
        "simultaneous_bootstrap_calibration": bootstrap,
        "L4_transform": {
            "qhat_reference_median": qhat_median,
            "qhat_reference_mean": qhat_centre,
            "qhat_reference_covariance": qhat_covariance,
            "radial_bias_reference_median": bias_centre,
            "coordinates": "abs(qhat-median), joint qhat Mahalanobis distance, abs(radial_bias-median), radial_RMS",
        },
        "L5_transform": {
            "coordinates": "Eres, abs(Pwhite-1), delta_E_parent3429 for eight shells"
        },
        "rows": rows,
        "decision": {
            "authorize_opening_V8_projection_CF4_metrics": parent_p >= float(
                program["gates"]["L2_parent_conformal_p_min"]
            ),
            "fresh_V9_authorized": False,
            "seed_promotion_authorized": False,
            "RAMSES_authorized": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")
    print(json.dumps({
        "status": report["status"],
        "L2_parent3429": report["L2_parent3429"],
        "authorize_opening_V8_projection_CF4_metrics": report["decision"][
            "authorize_opening_V8_projection_CF4_metrics"
        ],
        "output": str(args.out.resolve()),
    }, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
