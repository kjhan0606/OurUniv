#!/usr/bin/env python
"""Frozen V30 deterministic-backbone and local-condition audit."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file


SCHEMA = "hong2021-v30-deterministic-backbone-local-condition-audit-v1"
PROGRAM_SCHEMA = "hong2021-v30-deterministic-backbone-local-condition-audit-program-v1"
PROGRAM_SHA256 = "c3978322153870964084f437b23367a52dc0607ba125790350f361e836e2a1ac"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
Q = 0.99999


@dataclass
class StreamingPearson:
    """Stable-enough float64 sufficient statistics for a pooled correlation."""

    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0

    def add(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> None:
        first = np.asarray(x, dtype=np.float64)
        second = np.asarray(y, dtype=np.float64)
        if first.shape != second.shape:
            raise ValueError("correlation arrays differ in shape")
        selected = np.isfinite(first) & np.isfinite(second)
        if mask is not None:
            if np.asarray(mask).shape != first.shape:
                raise ValueError("correlation mask differs in shape")
            selected &= np.asarray(mask, dtype=bool)
        first = first[selected]
        second = second[selected]
        self.n += int(first.size)
        self.sx += float(first.sum(dtype=np.float64))
        self.sy += float(second.sum(dtype=np.float64))
        self.sxx += float(np.square(first).sum(dtype=np.float64))
        self.syy += float(np.square(second).sum(dtype=np.float64))
        self.sxy += float((first * second).sum(dtype=np.float64))

    def result(self) -> dict[str, float | int]:
        if self.n < 2:
            return {"n": self.n, "pearson": float("nan")}
        n = float(self.n)
        covariance = self.sxy - self.sx * self.sy / n
        variance_x = self.sxx - self.sx * self.sx / n
        variance_y = self.syy - self.sy * self.sy / n
        denominator = np.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
        return {
            "n": self.n,
            "pearson": float(covariance / denominator) if denominator > 0 else float("nan"),
        }


def _finite_edges(values: Iterable[Any]) -> np.ndarray:
    edges = np.asarray(
        [np.inf if value == "inf" else float(value) for value in values],
        dtype=np.float64,
    )
    if edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("V30 Fourier edges must increase from zero")
    return edges


def fourier_masks(grid: int, voxel_mpc_h: float, edges: np.ndarray) -> list[np.ndarray]:
    if grid <= 0 or voxel_mpc_h <= 0:
        raise ValueError("invalid grid or voxel size")
    frequency = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel_mpc_h)
    radius = np.sqrt(
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    return [
        (radius >= lower) & (radius < upper) & (radius > 0)
        for lower, upper in zip(edges[:-1], edges[1:], strict=True)
    ]


def block_sum(value: np.ndarray, factor: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or len(set(array.shape)) != 1 or array.shape[0] % factor:
        raise ValueError("block_sum requires a divisible cubic field")
    size = array.shape[0] // factor
    return array.reshape(size, factor, size, factor, size, factor).sum(axis=(1, 3, 5))


def block_mean(value: np.ndarray, factor: int) -> np.ndarray:
    return block_sum(value, factor) / float(factor**3)


def tail_diagnostics(truth_y: np.ndarray, generated_y: np.ndarray) -> dict[str, float | bool]:
    truth = 4.5 * np.asarray(truth_y, dtype=np.float64).reshape(-1)
    generated = 4.5 * np.asarray(generated_y, dtype=np.float64).reshape(-1)
    if truth.size != generated.size or not np.isfinite(truth).all() or not np.isfinite(generated).all():
        raise ValueError("tail diagnostic fields differ or are not finite")
    truth_q = float(np.quantile(truth, Q))
    generated_q = float(np.quantile(generated, Q))
    truth_max = float(truth.max())
    generated_max = float(generated.max())
    truth_delta = np.power(10.0, truth) - 1.0
    generated_delta = np.power(10.0, generated) - 1.0
    truth_q4 = float(np.mean(np.square(truth_delta)))
    generated_q4 = float(np.mean(np.square(generated_delta)))
    result: dict[str, float | bool] = {
        "truth_q99_999_log10rho": truth_q,
        "generated_q99_999_log10rho": generated_q,
        "delta_q99_999_dex": generated_q - truth_q,
        "truth_max_log10rho": truth_max,
        "generated_max_log10rho": generated_max,
        "generated_max_above_truth_max_dex": generated_max - truth_max,
        "truth_mean_delta_squared": truth_q4,
        "generated_mean_delta_squared": generated_q4,
        "generated_over_truth_mean_delta_squared": generated_q4 / truth_q4,
    }
    result["Q3_pass"] = bool(
        abs(float(result["delta_q99_999_dex"])) <= 0.1
        and float(result["generated_max_above_truth_max_dex"]) <= 0.3
    )
    result["Q4_pass"] = bool(float(result["generated_over_truth_mean_delta_squared"]) <= 1.5)
    return result


def classify(backbone_improves: bool, coupling_material: bool) -> tuple[str, str]:
    """Return the predeclared V30 branch without inspecting metric magnitudes."""
    if backbone_improves and not coupling_material:
        return (
            "deterministic_backbone_and_local_residual_separation_adequate",
            "audit_target_free_donor_descriptor_instead_of_changing_backbone",
        )
    if not backbone_improves and coupling_material:
        return (
            "deterministic_backbone_underfit_and_local_residual_coupling_unmodeled",
            "replace_backbone_with_multiscale_conditional_density_model_and_condition_stochastic_residual_locally",
        )
    if not backbone_improves:
        return (
            "deterministic_current_density_backbone_underfit",
            "replace_and_validate_deterministic_multiscale_backbone_before_residual_modeling",
        )
    return (
        "global_backbone_improves_but_local_residual_coupling_is_unmodeled",
        "retain_improved_low_k_backbone_and_build_explicitly_local_conditional_residual_model",
    )


def _summary_error(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    residual = truth - prediction
    return {
        "bias_truth_minus_prediction": float(residual.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "pearson": float(np.corrcoef(prediction, truth)[0, 1]),
        "prediction_std": float(prediction.std()),
        "truth_std": float(truth.std()),
    }


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if sha256_file(resolved) != PROGRAM_SHA256:
        raise ValueError("V30 program hash differs")
    program = json.loads(resolved.read_text())
    if program.get("schema") != PROGRAM_SCHEMA or tuple(program["development_domains"]) != DOMAIN_ORDER:
        raise ValueError("V30 program schema or domain order differs")
    parent = program["parent_evidence"]
    parent_path = Path(parent["v29_decision"])
    if sha256_file(parent_path) != parent["v29_decision_sha256"]:
        raise ValueError("V30 parent V29 decision hash differs")
    decision = json.loads(parent_path.read_text())
    if (
        decision.get("decision_digest_sha256") != parent["v29_decision_digest_sha256"]
        or canonical_digest(decision) != parent["v29_decision_digest_sha256"]
        or decision.get("classification", {}).get("class") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
    ):
        raise ValueError("V30 parent V29 decision content differs")
    for domain in DOMAIN_ORDER:
        specification = program["development_domains"][domain]
        for key in ("data", "v4_baseline_cache", "v14_cache"):
            artifact = Path(specification[key])
            if not artifact.is_file() or sha256_file(artifact) != specification[f"{key}_sha256"]:
                raise ValueError(f"V30 {domain} {key} hash differs")
            lower = str(artifact).lower()
            if "astrid" in lower or "refl0100n1504" in lower:
                raise ValueError("V30 firewall path violation")
    return program


def _audit_domain(name: str, specification: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(specification["data"])
    v4_path = Path(specification["v4_baseline_cache"])
    v14_path = Path(specification["v14_cache"])
    edges = _finite_edges(program["frozen_measurements"]["fourier_edges_h_mpc"])
    factors = tuple(int(value) for value in program["frozen_measurements"]["input_pool_factors"])
    shifts = tuple(
        tuple(int(axis) for axis in value)
        for value in program["frozen_measurements"]["coupling_control"]["translations_cells"]
    )
    pooled = {
        factor: {
            key: StreamingPearson()
            for key in ("count_truth", "velocity_truth", "v4_truth", "v14_truth")
        }
        for factor in factors
    }
    occupancy = {factor: [0, 0] for factor in factors}
    observed_error = {"observed": [0.0, 0], "unobserved": [0.0, 0]}
    truth_parts: list[np.ndarray] = []
    v4_parts: list[np.ndarray] = []
    v14_parts: list[np.ndarray] = []
    shifted_parts: dict[tuple[int, int, int], list[np.ndarray]] = {value: [] for value in shifts}
    power: dict[str, np.ndarray] = {
        key: np.zeros(len(edges) - 1, dtype=np.float64)
        for key in ("truth", "v4", "v14", "residual", "v4_cross_truth", "v14_cross_truth", "residual_cross_v14")
    }
    modes = np.zeros(len(edges) - 1, dtype=np.int64)

    with h5py.File(data_path, "r") as data, h5py.File(v4_path, "r") as v4_cache, h5py.File(v14_path, "r") as v14_cache:
        objects = int(specification["objects"])
        expected = (objects, 1, 64, 64, 64)
        if (
            tuple(data["target"].shape) != expected
            or tuple(v4_cache["conditional_mean"].shape) != expected
            or tuple(v14_cache["conditional_mean"].shape) != expected
            or tuple(data["input"].shape) != (objects, 2, 64, 64, 64)
            or tuple(v14_cache["predicted_residual_dc"].shape) != (objects,)
        ):
            raise ValueError(f"V30 {name} source shapes differ")
        if (
            Path(str(v4_cache.attrs.get("source_data", ""))).resolve() != data_path.resolve()
            or Path(str(v14_cache.attrs.get("source_data", ""))).resolve() != data_path.resolve()
        ):
            raise ValueError(f"V30 {name} cache provenance differs")
        voxel = float(data.attrs["voxel_mpc_h"])
        masks = fourier_masks(64, voxel, edges)
        modes[:] = [int(mask.sum()) for mask in masks]
        if modes.sum() != 64**3 - 1:
            raise RuntimeError("V30 Fourier masks are not exhaustive")

        for index in range(objects):
            truth = np.asarray(data["target"][index, 0], dtype=np.float32)
            v4 = np.asarray(v4_cache["conditional_mean"][index, 0], dtype=np.float32)
            v14 = np.asarray(v14_cache["conditional_mean"][index, 0], dtype=np.float32)
            v14 = v14 + np.float32(v14_cache["predicted_residual_dc"][index])
            residual = truth - v14
            count = np.asarray(data["input"][index, 0], dtype=np.float64)
            velocity = np.asarray(data["input"][index, 1], dtype=np.float64)
            truth_parts.append(truth.reshape(-1).copy())
            v4_parts.append(v4.reshape(-1).copy())
            v14_parts.append(v14.reshape(-1).copy())
            for shift in shifts:
                shifted = v14 + np.roll(residual, shift=shift, axis=(0, 1, 2))
                shifted_parts[shift].append(shifted.reshape(-1).astype(np.float32, copy=False))

            for label, selected in (("observed", count > 0), ("unobserved", count <= 0)):
                error = residual[selected].astype(np.float64)
                observed_error[label][0] += float(np.square(error).sum())
                observed_error[label][1] += int(error.size)

            for factor in factors:
                count_sum = block_sum(count, factor)
                truth_pool = block_mean(truth, factor)
                v4_pool = block_mean(v4, factor)
                v14_pool = block_mean(v14, factor)
                velocity_numerator = block_sum(count * velocity, factor)
                occupied = count_sum > 0
                velocity_mean = np.zeros_like(count_sum)
                velocity_mean[occupied] = velocity_numerator[occupied] / count_sum[occupied]
                count_signal = np.log1p(count_sum)
                pooled[factor]["count_truth"].add(count_signal, truth_pool)
                pooled[factor]["velocity_truth"].add(velocity_mean, truth_pool, occupied)
                pooled[factor]["v4_truth"].add(v4_pool, truth_pool)
                pooled[factor]["v14_truth"].add(v14_pool, truth_pool)
                occupancy[factor][0] += int(occupied.sum())
                occupancy[factor][1] += int(occupied.size)

            transforms = {
                "truth": np.fft.fftn(truth.astype(np.float64) - float(truth.mean(dtype=np.float64))),
                "v4": np.fft.fftn(v4.astype(np.float64) - float(v4.mean(dtype=np.float64))),
                "v14": np.fft.fftn(v14.astype(np.float64) - float(v14.mean(dtype=np.float64))),
                "residual": np.fft.fftn(residual.astype(np.float64) - float(residual.mean(dtype=np.float64))),
            }
            for band, mask in enumerate(masks):
                for field in ("truth", "v4", "v14", "residual"):
                    power[field][band] += float(np.square(np.abs(transforms[field][mask])).sum())
                power["v4_cross_truth"][band] += float(
                    np.real(transforms["v4"][mask] * np.conj(transforms["truth"][mask])).sum()
                )
                power["v14_cross_truth"][band] += float(
                    np.real(transforms["v14"][mask] * np.conj(transforms["truth"][mask])).sum()
                )
                power["residual_cross_v14"][band] += float(
                    np.real(transforms["residual"][mask] * np.conj(transforms["v14"][mask])).sum()
                )
            if (index + 1) % 16 == 0 or index + 1 == objects:
                print(f"[v30] {name} {index + 1}/{objects}", flush=True)

    truth_all = np.concatenate(truth_parts).astype(np.float32, copy=False)
    v4_all = np.concatenate(v4_parts).astype(np.float32, copy=False)
    v14_all = np.concatenate(v14_parts).astype(np.float32, copy=False)
    residual_all = truth_all - v14_all
    translated = {
        ",".join(str(axis) for axis in shift): tail_diagnostics(
            truth_all, np.concatenate(shifted_parts[shift])
        )
        for shift in shifts
    }
    translated_median = {
        key: float(np.median([float(row[key]) for row in translated.values()]))
        for key in (
            "delta_q99_999_dex",
            "generated_max_above_truth_max_dex",
            "generated_over_truth_mean_delta_squared",
        )
    }
    translated_median["Q3_pass"] = bool(
        abs(translated_median["delta_q99_999_dex"]) <= 0.1
        and translated_median["generated_max_above_truth_max_dex"] <= 0.3
    )
    translated_median["Q4_pass"] = bool(
        translated_median["generated_over_truth_mean_delta_squared"] <= 1.5
    )
    decile_edges = np.quantile(v14_all.astype(np.float64), np.linspace(0.0, 1.0, 11))
    decile_index = np.searchsorted(decile_edges[1:-1], v14_all, side="right")
    residual_deciles = []
    for decile in range(10):
        values = residual_all[decile_index == decile].astype(np.float64)
        residual_deciles.append(
            {
                "decile": decile,
                "lower_v14_y": float(decile_edges[decile]),
                "upper_v14_y": float(decile_edges[decile + 1]),
                "count": int(values.size),
                "residual_mean_y": float(values.mean()),
                "residual_std_y": float(values.std()),
            }
        )

    def coherence(cross: str, first: str, second: str) -> np.ndarray:
        denominator = np.sqrt(power[first] * power[second])
        return np.divide(power[cross], denominator, out=np.zeros_like(denominator), where=denominator > 0)

    v4_coherence = coherence("v4_cross_truth", "v4", "truth")
    v14_coherence = coherence("v14_cross_truth", "v14", "truth")
    residual_v14_correlation = coherence("residual_cross_v14", "residual", "v14")
    v4_error = _summary_error(v4_all, truth_all)
    v14_error = _summary_error(v14_all, truth_all)
    rmse_ratio = v14_error["rmse"] / v4_error["rmse"]
    low_k_no_harm = bool(np.all(v14_coherence[:2] >= v4_coherence[:2] - 0.02))
    backbone_improves = bool(rmse_ratio <= 0.98 and low_k_no_harm)
    coupling_material = not bool(translated_median["Q3_pass"] and translated_median["Q4_pass"])
    horizon = 0.0
    for band, value in enumerate(v14_coherence):
        if value < 0.5:
            break
        horizon = float(edges[band + 1])
    return {
        "objects": int(specification["objects"]),
        "voxel_mpc_h": voxel,
        "native_input_occupied_fraction": occupancy[1][0] / occupancy[1][1],
        "v4_error": v4_error,
        "v14_error": v14_error,
        "v14_over_v4_rmse": rmse_ratio,
        "low_k_phase_no_harm": low_k_no_harm,
        "backbone_improves": backbone_improves,
        "fourier": {
            "edges_h_mpc": [float(value) if np.isfinite(value) else "inf" for value in edges],
            "mode_count_per_cube": modes.tolist(),
            "v4_power_over_truth": (power["v4"] / power["truth"]).tolist(),
            "v14_power_over_truth": (power["v14"] / power["truth"]).tolist(),
            "v4_truth_coherence": v4_coherence.tolist(),
            "v14_truth_coherence": v14_coherence.tolist(),
            "v14_residual_cross_correlation": residual_v14_correlation.tolist(),
            "contiguous_v14_coherence_ge_0p5_horizon_h_mpc": horizon,
        },
        "pooled_local_condition": {
            str(factor): {
                "occupied_fraction": occupancy[factor][0] / occupancy[factor][1],
                **{key: statistic.result() for key, statistic in pooled[factor].items()},
            }
            for factor in factors
        },
        "v14_residual_rmse_by_native_observation": {
            label: {
                "cells": count,
                "rmse": float(np.sqrt(squared / count)),
            }
            for label, (squared, count) in observed_error.items()
        },
        "v14_residual_by_backbone_decile": residual_deciles,
        "tail_diagnostics": {
            "v4_backbone": tail_diagnostics(truth_all, v4_all),
            "v14_backbone": tail_diagnostics(truth_all, v14_all),
            "own_residual_exact_reconstruction": tail_diagnostics(truth_all, v14_all + residual_all),
            "translated_own_residual": translated,
            "translated_median": translated_median,
        },
        "material_local_backbone_residual_coupling": coupling_material,
    }


def evaluate(program_path: Path, repo: Path) -> dict[str, Any]:
    program = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V30 audit requires a clean committed worktree")
    domains = {
        name: _audit_domain(name, program["development_domains"][name], program)
        for name in DOMAIN_ORDER
    }
    backbone_improves = all(row["backbone_improves"] for row in domains.values())
    coupling_material = any(
        row["material_local_backbone_residual_coupling"] for row in domains.values()
    )
    classification, next_step = classify(backbone_improves, coupling_material)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_development_only_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "domains": domains,
        "backbone_improves_all_domains": backbone_improves,
        "material_local_coupling_any_domain": coupling_material,
        "classification": classification,
        "next": next_step,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite V30 audit: {args.out}")
    report = evaluate(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
