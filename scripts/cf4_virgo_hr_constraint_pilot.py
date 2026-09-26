#!/usr/bin/env python3
"""Disk-lean Hoffman--Ribak pilot for a Virgo-scale parent constraint.

The input is the accepted z=0 linear density field.  Its isotropised Fourier
power defines the stationary covariance.  For each requested target this
program applies the exact one-constraint Matheron correction at the frozen
Virgo coordinate and reports scientific side effects without writing a field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOX = 384.0
VIRGO = np.array([189.329829033528, 203.676745420703, 191.508856275301])
LG = np.array([192.0, 192.0, 192.0])
COMA = np.array([192.469352982037, 262.121364740826, 202.263080174214])
PRESERVE = np.array([
    [192.0, 192.0, 192.0],
    [208.5, 185.0, 208.5],
    [213.0, 185.0, 189.0],
    [190.0, 190.0, 199.0],
    [223.0, 209.0, 204.0],
])


def shell_power(fk: np.ndarray, kmag: np.ndarray, nbin: int):
    edges = np.linspace(0.0, float(kmag.max()) * (1.0 + 1e-10), nbin + 1)
    index = np.clip(np.digitize(kmag.ravel(), edges) - 1, 0, nbin - 1)
    count = np.bincount(index, minlength=nbin)
    power = np.bincount(index, weights=np.abs(fk).ravel() ** 2,
                        minlength=nbin) / np.maximum(count, 1)
    return edges, index.reshape(kmag.shape), power, count


def sphere_mean(field: np.ndarray, position: np.ndarray, radius: float) -> float:
    n = field.shape[0]
    dx = BOX / n
    axis = (np.arange(n) + 0.5) * dx
    ids = [np.where(np.minimum(abs(axis - position[k]),
                               BOX - abs(axis - position[k])) <= radius)[0]
           for k in range(3)]
    values = []
    for i in ids[0]:
        for j in ids[1]:
            for k in ids[2]:
                delta = np.array([axis[i], axis[j], axis[k]]) - position
                delta -= BOX * np.rint(delta / BOX)
                if delta @ delta < radius * radius:
                    values.append(field[i, j, k])
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--targets", type=float, nargs="+",
                        default=[1.0, 1.5, 1.686, 2.0])
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--joint-preserve", action="store_true",
                        help="preserve the current smoothed density at the LG centre "
                             "and four Local-Voice probe positions")
    args = parser.parse_args()

    source = np.load(args.field, allow_pickle=False)
    field = np.asarray(source["field"], dtype=np.float32)
    n = field.shape[0]
    if field.shape != (n, n, n):
        raise ValueError("field must be a periodic cube")
    dx = BOX / n
    point = np.floor(VIRGO / dx).astype(int) % n
    point_position = (point + 0.5) * dx

    fk = np.fft.rfftn(field, norm="ortho")
    kxy = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    kmag = np.sqrt(kxy[:, None, None] ** 2 +
                   kxy[None, :, None] ** 2 + kz[None, None, :] ** 2)
    window = np.exp(-0.5 * (kmag * args.radius) ** 2)
    edges, shell_index, power, count = shell_power(fk, kmag, args.bins)
    power_mode = power[shell_index]
    power_mode[0, 0, 0] = 0.0

    axes = (0, 1, 2)
    smooth = np.fft.irfftn(
        fk * window, s=field.shape, axes=axes, norm="ortho")
    positions = np.vstack([VIRGO, PRESERVE]) if args.joint_preserve else VIRGO[None, :]
    points = np.floor(positions / dx).astype(int) % n
    predicted_all = smooth[tuple(points.T)].astype(np.float64)
    predicted = float(predicted_all[0])
    norm = np.sqrt(float(field.size))
    covariance_grid = np.fft.irfftn(
        power_mode * window, s=field.shape, axes=axes, norm="ortho") / norm
    constraint_covariance_grid = np.fft.irfftn(
        power_mode * window**2, s=field.shape, axes=axes, norm="ortho") / norm
    variance = float(constraint_covariance_grid[0, 0, 0])
    if not np.isfinite(variance) or variance <= 0:
        raise RuntimeError(f"invalid constraint variance {variance}")
    covariance = np.empty((len(points), len(points)), dtype=np.float64)
    for i in range(len(points)):
        for j in range(len(points)):
            covariance[i, j] = constraint_covariance_grid[
                tuple(np.mod(points[i] - points[j], n))]

    base_shell_power = power.copy()
    trials = []
    for target in args.targets:
        desired = predicted_all.copy()
        desired[0] = target
        weights = np.linalg.solve(covariance, desired - predicted_all)
        correction = np.zeros_like(field, dtype=np.float64)
        for weight, constraint_point in zip(weights, points):
            correction += weight * np.roll(
                covariance_grid, shift=tuple(constraint_point), axis=axes)
        candidate = field + correction.astype(np.float32)
        achieved_all = np.fft.irfftn(
            np.fft.rfftn(candidate, norm="ortho") * window,
            s=field.shape, axes=axes, norm="ortho")[tuple(points.T)]
        achieved = float(achieved_all[0])
        cfk = np.fft.rfftn(candidate, norm="ortho")
        _, _, candidate_power, _ = shell_power(cfk, kmag, args.bins)
        valid = (count >= 32) & (base_shell_power > 0)
        ratio = candidate_power[valid] / base_shell_power[valid]
        trials.append({
            "target": float(target),
            "achieved": achieved,
            "preserved_max_abs_error": float(np.max(
                np.abs(achieved_all[1:] - predicted_all[1:]), initial=0.0)),
            "correction_rms": float(correction.std()),
            "correction_to_field_rms": float(correction.std() / field.std()),
            "field_cross_correlation": float(np.corrcoef(
                field.ravel(), candidate.ravel())[0, 1]),
            "power_ratio_median": float(np.median(ratio)),
            "power_ratio_p05_p95": [float(np.percentile(ratio, 5)),
                                      float(np.percentile(ratio, 95))],
            "lg_mean_delta_R8": sphere_mean(candidate, LG, 8.0),
            "virgo_mean_delta_R8": sphere_mean(candidate, VIRGO, 8.0),
            "coma_mean_delta_R8": sphere_mean(candidate, COMA, 8.0),
        })

    result = {
        "schema": "ouruniv-cf4-virgo-hr-constraint-pilot-v1",
        "status": "diagnostic_only_no_field_written",
        "source": str(args.field.resolve()),
        "box_cMpc_h": BOX,
        "grid": n,
        "cell_cMpc_h": dx,
        "gaussian_radius_cMpc_h": args.radius,
        "virgo_target_cMpc_h": VIRGO.tolist(),
        "constraint_grid_index": point.tolist(),
        "constraint_grid_position_cMpc_h": point_position.tolist(),
        "constraint_position_error_cMpc_h": float(np.linalg.norm(
            point_position - VIRGO)),
        "predicted_before": predicted,
        "constraint_variance": variance,
        "joint_preserve": bool(args.joint_preserve),
        "constraint_points_cMpc_h": positions.tolist(),
        "predicted_before_all": predicted_all.tolist(),
        "base": {
            "field_rms": float(field.std()),
            "lg_mean_delta_R8": sphere_mean(field, LG, 8.0),
            "virgo_mean_delta_R8": sphere_mean(field, VIRGO, 8.0),
            "coma_mean_delta_R8": sphere_mean(field, COMA, 8.0),
        },
        "trials": trials,
        "decision": "No field is promoted by this pilot.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
