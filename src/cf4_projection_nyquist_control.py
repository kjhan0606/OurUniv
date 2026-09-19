#!/usr/bin/env python3
"""Consumed-only white-field control for even-grid Fourier projection.

The legacy N576-to-N192 projector copies one representative of every output
Nyquist equivalence class.  On an even output mesh, however, +Nout/2 and
-Nout/2 are the same discrete frequency.  This module compares that exact
legacy operation with a variance-preserving fold of every such +/- pair.  It
does not read CF4 observations, construct a new constrained field, or select a
V8 proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cf4_lg_peak_cr import free_rfft_mask
from cf4_make_ic import fourier_resample_white_field


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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _replace_nonfinite(value.tolist())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _replace_nonfinite(value: Any) -> Any:
    if isinstance(value, list):
        return [_replace_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_rfft(source_fft: np.ndarray, source_n: int, output_n: int) -> None:
    expected = (source_n, source_n, source_n // 2 + 1)
    if source_fft.shape != expected:
        raise ValueError(f"source_fft shape {source_fft.shape} != {expected}")
    if output_n <= 0 or output_n % 2 or output_n > source_n:
        raise ValueError("output_n must be positive, even, and no larger than source_n")


def legacy_projection_rfft(
    source_fft: np.ndarray,
    source_n: int,
    output_n: int,
) -> np.ndarray:
    """Reproduce the exact legacy coefficient slice without another FFT."""
    _validate_rfft(source_fft, source_n, output_n)
    half = output_n // 2
    index = np.r_[0:half, source_n - half:source_n]
    output = source_fft[
        np.ix_(index, index, np.arange(half + 1))
    ].copy()
    output *= (float(output_n) / source_n) ** 1.5
    return output


def _source_plane(
    source_fft: np.ndarray,
    source_n: int,
    signed_x: np.ndarray,
    signed_y: np.ndarray,
    signed_z: int,
) -> np.ndarray:
    """Read a signed-kz plane from a real-input rFFT."""
    signed_x = np.asarray(signed_x, dtype=np.int64)
    signed_y = np.asarray(signed_y, dtype=np.int64)
    if signed_z >= 0:
        ix = np.mod(signed_x, source_n)
        iy = np.mod(signed_y, source_n)
        return source_fft[ix[:, None], iy[None, :], signed_z]
    ix = np.mod(-signed_x, source_n)
    iy = np.mod(-signed_y, source_n)
    return np.conjugate(source_fft[ix[:, None], iy[None, :], -signed_z])


def _fold_xy_plane(
    source_fft: np.ndarray,
    source_n: int,
    output_n: int,
    signed_z: int,
) -> np.ndarray:
    """Fold the +/- output-Nyquist classes on x and y for one signed kz."""
    half = output_n // 2
    signed = np.r_[np.arange(half), np.arange(-half, 0)]
    plane = _source_plane(source_fft, source_n, signed, signed, signed_z).copy()
    root_two = math.sqrt(2.0)

    x_minus = _source_plane(
        source_fft, source_n, np.array([-half]), signed, signed_z
    )[0]
    x_plus = _source_plane(
        source_fft, source_n, np.array([half]), signed, signed_z
    )[0]
    plane[half, :] = (x_minus + x_plus) / root_two

    y_minus = _source_plane(
        source_fft, source_n, signed, np.array([-half]), signed_z
    )[:, 0]
    y_plus = _source_plane(
        source_fft, source_n, signed, np.array([half]), signed_z
    )[:, 0]
    plane[:, half] = (y_minus + y_plus) / root_two

    corner = _source_plane(
        source_fft,
        source_n,
        np.array([-half, half]),
        np.array([-half, half]),
        signed_z,
    )
    plane[half, half] = np.sum(corner) / 2.0
    return plane


def variance_preserving_projection_rfft(
    source_fft: np.ndarray,
    source_n: int,
    output_n: int,
) -> np.ndarray:
    """Fold every output-Nyquist +/- class with unit-variance normalization."""
    _validate_rfft(source_fft, source_n, output_n)
    half = output_n // 2
    output = np.empty((output_n, output_n, half + 1), dtype=np.complex128)
    for signed_z in range(half):
        output[..., signed_z] = _fold_xy_plane(
            source_fft, source_n, output_n, signed_z
        )
    output[..., half] = (
        _fold_xy_plane(source_fft, source_n, output_n, -half)
        + _fold_xy_plane(source_fft, source_n, output_n, half)
    ) / math.sqrt(2.0)
    output *= (float(output_n) / source_n) ** 1.5
    return output


def spatial_from_output_rfft(output_fft: np.ndarray, output_n: int) -> np.ndarray:
    return np.fft.irfftn(
        output_fft, s=(output_n, output_n, output_n), axes=(0, 1, 2)
    ).astype(np.float32)


def projection_geometry(
    n: int,
    box_size: float,
    frozen_n: int,
    shell_edges: np.ndarray,
) -> dict[str, Any]:
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box_size / n)
    kmag = np.sqrt(
        kx[:, None, None] ** 2
        + kx[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    released = free_rfft_mask(n, frozen_n)
    boundary = np.zeros_like(released)
    half = n // 2
    boundary[half, :, :] = True
    boundary[:, half, :] = True
    boundary[:, :, half] = True
    weights = np.full((1, 1, half + 1), 2.0, dtype=np.float64)
    weights[..., 0] = 1.0
    weights[..., -1] = 1.0
    broadcast_weights = np.broadcast_to(weights, released.shape)

    masks = []
    covered = np.zeros_like(released)
    for low, high in zip(shell_edges[:-1], shell_edges[1:]):
        mask = released & (kmag >= low) & (kmag < high)
        if not np.any(mask):
            raise ValueError(f"empty released shell {low:g}--{high:g}")
        if np.any(mask & covered):
            raise RuntimeError("released shells overlap")
        covered |= mask
        masks.append(mask)
    if not np.array_equal(covered, released):
        raise RuntimeError("released shells do not partition the released mask")
    return {
        "masks": masks,
        "boundary": boundary,
        "weights": broadcast_weights,
        "boundary_weight_fractions": np.asarray([
            np.sum(broadcast_weights[mask & boundary])
            / np.sum(broadcast_weights[mask])
            for mask in masks
        ]),
    }


def shell_power(field: np.ndarray, geometry: dict[str, Any]) -> dict[str, np.ndarray]:
    field_fft = np.fft.rfftn(np.asarray(field, dtype=np.float64), norm="ortho")
    power = np.abs(field_fft) ** 2
    weights = geometry["weights"]
    boundary = geometry["boundary"]
    all_rows, interior_rows, boundary_rows = [], [], []
    for mask in geometry["masks"]:
        interior_mask = mask & ~boundary
        boundary_mask = mask & boundary

        def weighted_mean(part: np.ndarray) -> float:
            denominator = np.sum(weights[part])
            if denominator == 0:
                return math.nan
            return float(np.sum(weights[part] * power[part]) / denominator)

        all_rows.append(weighted_mean(mask))
        interior_rows.append(weighted_mean(interior_mask))
        boundary_rows.append(weighted_mean(boundary_mask))
    return {
        "all": np.asarray(all_rows),
        "interior": np.asarray(interior_rows),
        "boundary": np.asarray(boundary_rows),
    }


def mean_and_standard_error(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a sample-by-coordinate matrix")
    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    standard_error = np.full(values.shape[1], np.nan, dtype=np.float64)
    for coordinate in range(values.shape[1]):
        finite = values[np.isfinite(values[:, coordinate]), coordinate]
        if finite.size:
            mean[coordinate] = np.mean(finite)
        if finite.size > 1:
            standard_error[coordinate] = np.std(finite, ddof=1) / np.sqrt(finite.size)
    return mean, standard_error


def normalized_errors(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = np.asarray(left) - np.asarray(right)
    scale = float(np.sqrt(np.mean(np.abs(np.asarray(right)) ** 2)))
    return {
        "relative_RMS": float(np.sqrt(np.mean(np.abs(difference) ** 2)) / scale),
        "maximum_normalized_error": float(np.max(np.abs(difference)) / scale),
    }


def _within_white_envelope(
    mean: np.ndarray,
    standard_error: np.ndarray,
    absolute_floor: float,
    standard_error_multiple: float,
) -> np.ndarray:
    tolerance = np.maximum(absolute_floor, standard_error_multiple * standard_error)
    return np.abs(mean - 1.0) <= tolerance


def run(program: dict[str, Any]) -> dict[str, Any]:
    mesh = program["mesh"]
    source_n = int(mesh["source_N"])
    output_n = int(mesh["output_N"])
    frozen_n = int(mesh["frozen_N"])
    box_size = float(mesh["box_size_mpc_h"])
    edges = np.asarray([
        math.inf if value is None else float(value)
        for value in program["shells"]["edges_h_mpc"]
    ])
    geometry = projection_geometry(output_n, box_size, frozen_n, edges)
    seeds = [int(seed) for seed in program["sampling"]["seeds"]]
    rows = []
    helper_error = None

    for number, seed in enumerate(seeds, start=1):
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        source = rng.standard_normal((source_n, source_n, source_n))
        source_fft = np.fft.rfftn(source)
        legacy_fft = legacy_projection_rfft(source_fft, source_n, output_n)
        folded_fft = variance_preserving_projection_rfft(
            source_fft, source_n, output_n
        )
        legacy = spatial_from_output_rfft(legacy_fft, output_n)
        folded = spatial_from_output_rfft(folded_fft, output_n)
        if helper_error is None:
            production = fourier_resample_white_field(source, output_n)
            helper_error = normalized_errors(legacy, production)
        rows.append({
            "seed": seed,
            "source_mean": float(np.mean(source)),
            "source_std": float(np.std(source)),
            "legacy": shell_power(legacy, geometry),
            "variance_preserving_fold": shell_power(folded, geometry),
        })
        print(f"[control] {number}/{len(seeds)} seed={seed}", flush=True)
        del source, source_fft, legacy_fft, folded_fft, legacy, folded

    summary: dict[str, Any] = {}
    for variant in ("legacy", "variance_preserving_fold"):
        summary[variant] = {}
        for region in ("all", "interior", "boundary"):
            matrix = np.asarray([row[variant][region] for row in rows])
            mean, standard_error = mean_and_standard_error(matrix)
            summary[variant][region] = {
                "mean": mean,
                "standard_error": standard_error,
            }

    gate = program["gates"]
    boundary_shells = geometry["boundary_weight_fractions"] > 0.0
    legacy_interior = summary["legacy"]["interior"]
    folded_all = summary["variance_preserving_fold"]["all"]
    folded_boundary = summary["variance_preserving_fold"]["boundary"]
    legacy_boundary = summary["legacy"]["boundary"]
    legacy_all_mean = summary["legacy"]["all"]["mean"]
    folded_all_mean = folded_all["mean"]
    v8_mean = np.asarray(program["comparison"]["V8_released_shell_mean_Pwhite"])

    helper_pass = bool(
        helper_error["relative_RMS"] <= gate["helper_relative_RMS_max"]
        and helper_error["maximum_normalized_error"]
        <= gate["helper_maximum_normalized_error_max"]
    )
    interior_pass = bool(np.all(_within_white_envelope(
        legacy_interior["mean"], legacy_interior["standard_error"],
        float(gate["white_absolute_floor"]),
        float(gate["standard_error_multiple"]),
    )))
    folded_all_pass = bool(np.all(_within_white_envelope(
        folded_all["mean"], folded_all["standard_error"],
        float(gate["white_absolute_floor"]),
        float(gate["standard_error_multiple"]),
    )))
    folded_boundary_pass = bool(np.all(_within_white_envelope(
        folded_boundary["mean"][boundary_shells],
        folded_boundary["standard_error"][boundary_shells],
        float(gate["boundary_white_absolute_floor"]),
        float(gate["standard_error_multiple"]),
    )))
    legacy_boundary_deficit = 1.0 - legacy_boundary["mean"][boundary_shells]
    legacy_boundary_fail = bool(np.max(legacy_boundary_deficit) >= float(
        gate["legacy_boundary_deficit_min"]
    ))
    v8_match_error = np.abs(legacy_all_mean[boundary_shells] - v8_mean[boundary_shells])
    v8_match_pass = bool(np.all(
        v8_match_error <= float(gate["legacy_V8_shell_match_absolute_max"])
    ))
    fold_improves = bool(np.all(
        np.abs(folded_all_mean[boundary_shells] - 1.0)
        < np.abs(legacy_all_mean[boundary_shells] - 1.0)
    ))
    mechanism_pass = bool(
        helper_pass and interior_pass and folded_all_pass
        and folded_boundary_pass and legacy_boundary_fail
        and v8_match_pass and fold_improves
    )

    return {
        "schema": "ouruniv-cf4-projection-nyquist-control-result-v1",
        "status": (
            "complete_pass_output_Nyquist_boundary_mechanism_isolated"
            if mechanism_pass else
            "complete_fail_output_Nyquist_boundary_mechanism_not_isolated"
        ),
        "information_firewall": program["information_firewall"],
        "mesh": mesh,
        "sampling": program["sampling"],
        "shells": {
            "edges_h_mpc": program["shells"]["edges_h_mpc"],
            "boundary_weight_fractions": geometry["boundary_weight_fractions"],
            "boundary_affected": boundary_shells,
        },
        "legacy_helper_vs_production": helper_error,
        "summary": summary,
        "comparison": {
            "V8_released_shell_mean_Pwhite": v8_mean,
            "legacy_minus_V8_absolute": np.abs(legacy_all_mean - v8_mean),
            "boundary_shell_legacy_minus_V8_absolute": v8_match_error,
            "legacy_boundary_power_deficit": legacy_boundary_deficit,
            "folded_closer_to_white_on_every_boundary_shell": fold_improves,
        },
        "gates": {
            "legacy_helper_exact": helper_pass,
            "legacy_interior_white": interior_pass,
            "variance_preserving_all_shells_white": folded_all_pass,
            "variance_preserving_boundary_white": folded_boundary_pass,
            "legacy_boundary_deficit_detected": legacy_boundary_fail,
            "legacy_matches_V8_boundary_shells": v8_match_pass,
            "fold_improves_every_boundary_shell": fold_improves,
            "mechanism_isolated": mechanism_pass,
        },
        "decision": {
            "projection_fix_authorized_for_future_architecture": mechanism_pass,
            "retroactive_V8_mutation_authorized": False,
            "V9_or_seed_promotion_authorized": False,
            "RAMSES_authorized": False,
            "next_if_pass": (
                "Freeze a future-only variance-preserving projection contract, then "
                "design proposals over independent CF4 posterior parents."
            ),
            "next_if_fail": (
                "Do not change projection; expand the numerical control before any "
                "new constrained-realization architecture."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    program_path = Path(args.program).resolve()
    output_path = Path(args.out).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with program_path.open() as stream:
        program = json.load(stream)

    implementation = (ROOT / program["implementation"]["path"]).resolve()
    production = (ROOT / program["legacy_projection"]["path"]).resolve()
    if sha256_file(implementation) != program["implementation"]["sha256"]:
        raise RuntimeError("Nyquist-control implementation hash mismatch")
    if sha256_file(production) != program["legacy_projection"]["sha256"]:
        raise RuntimeError("legacy projection hash mismatch")

    result = run(program)
    result["lineage"] = {
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "implementation": str(implementation),
        "implementation_sha256": sha256_file(implementation),
        "legacy_projection": str(production),
        "legacy_projection_sha256": sha256_file(production),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(
            result, stream, indent=2, sort_keys=True,
            default=json_default, allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(output_path)
    print(f"[control] status={result['status']}", flush=True)


if __name__ == "__main__":
    main()
