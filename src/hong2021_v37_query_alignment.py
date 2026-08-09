#!/usr/bin/env python
"""Frozen V37 bounded query-aligned conditional-copula transport."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v31_copula import conditional_forward, conditional_inverse, load_model
from hong2021_v34_nonlinear_sufficiency import pooled_fields
from hong2021_v35_spectrum_phase import (
    _backbone,
    _open_split,
    load_program as load_v35_program,
)


PROGRAM_SCHEMA = "hong2021-v37-query-aligned-conditional-copula-development-program-v1"
PROGRAM_SHA256 = "aff92b603e94bb1c3ed52d0303490efe63259ae823912e3ddf91690d24a2499a"
DESCRIPTOR_SCHEMA = "hong2021-v37-train-only-query-alignment-descriptor-v1"
PREFLIGHT_SCHEMA = "hong2021-v37-query-aligned-copula-hard-preflight-v1"
ENSEMBLE_SCHEMA = "hong2021-v37-query-aligned-conditional-copula-ensemble-v1"
ARMS = ("aligned", "shuffled_query_control")
CHANNELS = (
    "log1p_block_count",
    "block_mean_velocity_kms",
    "exact_population_velocity_dispersion_kms",
    "backbone_mean_y",
)
POOL_FACTOR = 4
GRID = 16
MAX_SHIFT = 4
SHIFT_CANDIDATES = tuple(
    sorted(
        product(range(-MAX_SHIFT, MAX_SHIFT + 1), repeat=3),
        key=lambda row: (sum(value * value for value in row), row),
    )
)


def _verified_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "V37 program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_implementation_sampling_or_development_evaluation"
    ):
        raise ValueError("V37 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v36_record"]).resolve(),
        parent["v36_record_sha256"],
        "V37 V36 record",
    )
    audit = record.get("audit", {})
    if (
        audit.get("classification") != parent["required_classification"]
        or audit.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V37 parent conclusion or firewall differs")
    inherited = program["inherited_inputs"]
    v35_path = (repo / inherited["v35_program"]).resolve()
    if sha256_file(v35_path) != inherited["v35_program_sha256"]:
        raise ValueError("V37 V35 program hash differs")
    v35, _ = load_v35_program(v35_path, repo)
    for name in ("v31_program", "v31_record"):
        artifact = (repo / inherited[name]).resolve()
        if sha256_file(artifact) != inherited[f"{name}_sha256"]:
            raise ValueError(f"V37 {name} hash differs")
    if (
        sha256_file(Path(inherited["conditional_copula_artifact"]))
        != inherited["conditional_copula_artifact_sha256"]
        or sha256_file(Path(inherited["conditional_copula_report"]))
        != inherited["conditional_copula_report_sha256"]
    ):
        raise ValueError("V37 V31 copula artifact or report differs")
    report = json.loads(Path(inherited["conditional_copula_report"]).read_text())
    if report.get("artifact_sha256") != inherited["conditional_copula_artifact_sha256"]:
        raise ValueError("V37 V31 copula report binding differs")
    for domain in DOMAIN_ORDER:
        reference = program["development_evaluation"]["V31_reference_metrics"][domain]
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError(f"V37 {domain} V31 metrics hash differs")
    return program, v35


def descriptor_cube(data: h5py.File, cache: h5py.File, index: int) -> np.ndarray:
    fields = pooled_fields(
        np.asarray(data["input"][index, 0], dtype=np.float32),
        np.asarray(data["input"][index, 1], dtype=np.float32),
        np.asarray(data["input"][index, 2], dtype=np.float32),
        _backbone(cache, index),
        POOL_FACTOR,
    )
    result = np.stack([fields[name] for name in CHANNELS]).astype(np.float32)
    if result.shape != (len(CHANNELS), GRID, GRID, GRID) or not np.isfinite(result).all():
        raise ValueError("V37 descriptor shape or values differ")
    return result


def source_balanced_moments(
    domain_moments: Mapping[str, tuple[np.ndarray, np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if tuple(domain_moments) != DOMAIN_ORDER:
        raise ValueError("V37 source moment order differs")
    means = []
    seconds = []
    for domain in DOMAIN_ORDER:
        total, square_total, count = domain_moments[domain]
        total = np.asarray(total, dtype=np.float64)
        square_total = np.asarray(square_total, dtype=np.float64)
        if total.shape != (len(CHANNELS),) or square_total.shape != total.shape or count <= 0:
            raise ValueError("V37 source moment payload differs")
        means.append(total / count)
        seconds.append(square_total / count)
    mean = np.mean(means, axis=0)
    second = np.mean(seconds, axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 1.0e-12))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("V37 descriptor normalization is invalid")
    return mean, std


def fit_descriptor(
    program_path: Path, repo: Path, output: Path
) -> dict[str, Any]:
    _, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V37 descriptor fit requires a clean worktree")
    moments: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    per_source: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        total = np.zeros(len(CHANNELS), dtype=np.float64)
        square_total = np.zeros(len(CHANNELS), dtype=np.float64)
        count = 0
        data, cache = _open_split(row, "train")
        try:
            for index in range(int(row["train_objects"])):
                value = descriptor_cube(data, cache, index).astype(np.float64)
                total += value.sum(axis=(1, 2, 3))
                square_total += np.square(value).sum(axis=(1, 2, 3))
                count += GRID**3
                if (index + 1) % 32 == 0 or index + 1 == int(row["train_objects"]):
                    print(f"[v37-fit] {domain} {index + 1}/{row['train_objects']}", flush=True)
        finally:
            data.close()
            cache.close()
        moments[domain] = (total, square_total, count)
        per_source[domain] = {
            "cells": count,
            "mean": (total / count).tolist(),
            "second_moment": (square_total / count).tolist(),
        }
    mean, std = source_balanced_moments(moments)
    report: dict[str, Any] = {
        "schema": DESCRIPTOR_SCHEMA,
        "status": "complete_train_only_source_balanced_fit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "channels": list(CHANNELS),
        "pool_factor": POOL_FACTOR,
        "grid": GRID,
        "per_source": per_source,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "target_density_read": False,
        "validation_opened": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    if output.exists():
        raise RuntimeError("V37 refuses to overwrite descriptor artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_descriptor(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = _verified_json(path, expected_sha256, "V37 descriptor")
    if (
        payload.get("schema") != DESCRIPTOR_SCHEMA
        or payload.get("status") != "complete_train_only_source_balanced_fit"
        or payload.get("program_sha256") != PROGRAM_SHA256
        or payload.get("channels") != list(CHANNELS)
        or payload.get("target_density_read") is not False
        or payload.get("validation_opened") is not False
    ):
        raise ValueError("V37 descriptor metadata differs")
    mean = np.asarray(payload["mean"], dtype=np.float64)
    std = np.asarray(payload["std"], dtype=np.float64)
    if mean.shape != (len(CHANNELS),) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("V37 descriptor normalization payload differs")
    return payload


def standardize_descriptor(value: np.ndarray, fit: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (len(CHANNELS), GRID, GRID, GRID):
        raise ValueError("V37 descriptor standardization shape differs")
    mean = np.asarray(fit["mean"], dtype=np.float64)[:, None, None, None]
    std = np.asarray(fit["std"], dtype=np.float64)[:, None, None, None]
    result = (array - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("V37 standardized descriptor is nonfinite")
    return result.astype(np.float32)


def best_periodic_shift(
    query: np.ndarray, donor: np.ndarray
) -> tuple[tuple[int, int, int], float, float]:
    query = np.asarray(query, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    expected = (len(CHANNELS), GRID, GRID, GRID)
    if query.shape != expected or donor.shape != expected:
        raise ValueError("V37 alignment descriptor shape differs")
    correlation = np.zeros((GRID, GRID, GRID), dtype=np.float64)
    for channel in range(len(CHANNELS)):
        correlation += np.fft.ifftn(
            np.fft.fftn(query[channel]) * np.conj(np.fft.fftn(donor[channel]))
        ).real
    scores = np.asarray(
        [correlation[tuple(value % GRID for value in shift)] for shift in SHIFT_CANDIDATES]
    )
    shift = SHIFT_CANDIDATES[int(np.argmax(scores))]
    before = float(np.mean(np.square(query - donor), dtype=np.float64))
    shifted = np.roll(donor, shift=shift, axis=(1, 2, 3))
    after = float(np.mean(np.square(query - shifted), dtype=np.float64))
    if after > before + 1.0e-9:
        raise RuntimeError("V37 selected alignment is worse than zero shift")
    return shift, before, after


def _transport(
    oriented_uniform: np.ndarray,
    query_backbone: np.ndarray,
    shift: tuple[int, int, int],
    model: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    native_shift = tuple(POOL_FACTOR * value for value in shift)
    aligned = np.roll(
        oriented_uniform, shift=native_shift, axis=(-3, -2, -1)
    )
    residual = conditional_inverse(aligned, query_backbone, model).astype(np.float64)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
    sample = np.asarray(query_backbone, dtype=np.float64) + residual
    if not np.isfinite(sample).all():
        raise RuntimeError("V37 generated a nonfinite sample")
    return sample.astype(np.float32), maximum_dc


def _selection_arrays(v35: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for domain in DOMAIN_ORDER:
        path = Path(v35["development_domains"][domain]["phase_object_selection"])
        with h5py.File(path, "r") as handle:
            result[domain] = {
                name: np.asarray(handle[name])
                for name in (
                    "source_index",
                    "donor_source",
                    "donor_index",
                    "donor_isometry",
                    "donor_distance",
                    "predicted_residual_dc",
                    "predicted_band_scales",
                )
            }
    return result


def hard_preflight(
    program_path: Path,
    repo: Path,
    descriptor_path: Path,
    descriptor_sha256: str,
    output: Path,
) -> dict[str, Any]:
    _, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V37 preflight requires a clean worktree")
    fit = load_descriptor(descriptor_path, descriptor_sha256)
    if fit.get("code_commit") != commit:
        raise ValueError("V37 descriptor fit commit differs")
    synthetic = np.zeros((len(CHANNELS), GRID, GRID, GRID), dtype=np.float64)
    synthetic[:, 2, 3, 4] = np.arange(1, len(CHANNELS) + 1)
    expected_shift = (2, -1, 3)
    query = np.roll(synthetic, shift=expected_shift, axis=(1, 2, 3))
    recovered, before, after = best_periodic_shift(query, synthetic)
    if recovered != expected_shift or not after < before:
        raise RuntimeError("V37 FFT shift preflight failed")
    selection = _selection_arrays(v35)
    first_domain = DOMAIN_ORDER[0]
    row = v35["development_domains"][first_domain]
    query_index = int(selection[first_domain]["source_index"][0])
    donor_source = DOMAIN_ORDER[int(selection[first_domain]["donor_source"][0, 0])]
    donor_index = int(selection[first_domain]["donor_index"][0, 0])
    isometry = int(selection[first_domain]["donor_isometry"][0, 0])
    query_data, query_cache = _open_split(row, "validation")
    donor_row = v35["development_domains"][donor_source]
    donor_data, donor_cache = _open_split(donor_row, "train")
    model_info = program_path.resolve()
    inherited = json.loads(model_info.read_text())["inherited_inputs"]
    model = load_model(
        Path(inherited["conditional_copula_artifact"]),
        inherited["conditional_copula_artifact_sha256"],
    )
    try:
        q_desc = standardize_descriptor(
            descriptor_cube(query_data, query_cache, query_index), fit
        )
        d_desc = standardize_descriptor(
            descriptor_cube(donor_data, donor_cache, donor_index), fit
        )
        permutation, reflections = CUBE_ISOMETRIES[isometry]
        d_desc = apply_cube_isometry(d_desc, permutation, reflections)
        shift, descriptor_before, descriptor_after = best_periodic_shift(q_desc, d_desc)
        donor_backbone = _backbone(donor_cache, donor_index)[None]
        donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
        uniform = conditional_forward(donor_truth - donor_backbone, donor_backbone, model)
        uniform = apply_cube_isometry(uniform, permutation, reflections)
        query_backbone = _backbone(query_cache, query_index)[None]
        sample, dc = _transport(uniform, query_backbone, shift, model)
    finally:
        query_data.close()
        query_cache.close()
        donor_data.close()
        donor_cache.close()
    if not np.isfinite(sample).all() or dc > 1.0e-7:
        raise RuntimeError("V37 real sample preflight failed")
    report: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "descriptor": str(descriptor_path.resolve()),
        "descriptor_sha256": descriptor_sha256,
        "synthetic_shift": list(recovered),
        "selection_sha256": {
            domain: sha256_file(
                Path(v35["development_domains"][domain]["phase_object_selection"])
            )
            for domain in DOMAIN_ORDER
        },
        "real_sample": {
            "query_domain": first_domain,
            "query_index": query_index,
            "donor_source": donor_source,
            "donor_index": donor_index,
            "isometry": isometry,
            "shift_coarse": list(shift),
            "descriptor_mse_before": descriptor_before,
            "descriptor_mse_after": descriptor_after,
            "maximum_absolute_residual_dc": dc,
        },
        "validation_truth_used_for_shift_selection": False,
        "donor_reselection": False,
        "field_clipping": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    if output.exists():
        raise RuntimeError("V37 refuses to overwrite preflight")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _create_ensemble(handle: h5py.File) -> dict[str, h5py.Dataset]:
    return {
        "sample": handle.create_dataset(
            "sample",
            shape=(16, 16, 1, 64, 64, 64),
            dtype="f4",
            chunks=(1, 1, 1, 64, 64, 64),
            compression="lzf",
        ),
        "conditional_mean": handle.create_dataset(
            "conditional_mean", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"
        ),
        "truth": handle.create_dataset(
            "truth", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"
        ),
        "alignment_shift_coarse": handle.create_dataset(
            "alignment_shift_coarse", shape=(16, 16, 3), dtype="i1"
        ),
        "alignment_descriptor_mse_before": handle.create_dataset(
            "alignment_descriptor_mse_before", shape=(16, 16), dtype="f4"
        ),
        "alignment_descriptor_mse_after": handle.create_dataset(
            "alignment_descriptor_mse_after", shape=(16, 16), dtype="f4"
        ),
        "alignment_reference_query_index": handle.create_dataset(
            "alignment_reference_query_index", shape=(16,), dtype="i4"
        ),
    }


def sample_all(
    program_path: Path,
    repo: Path,
    descriptor_path: Path,
    descriptor_sha256: str,
    preflight_path: Path,
    preflight_sha256: str,
    output_root: Path,
) -> None:
    program, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V37 sampling requires a clean worktree")
    fit = load_descriptor(descriptor_path, descriptor_sha256)
    preflight = _verified_json(preflight_path, preflight_sha256, "V37 preflight")
    if (
        fit.get("code_commit") != commit
        or preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "pass"
        or preflight.get("code_commit") != commit
        or preflight.get("descriptor_sha256") != descriptor_sha256
    ):
        raise ValueError("V37 fit or preflight binding differs")
    if output_root.exists():
        raise RuntimeError("V37 refuses a pre-existing output root")
    inherited = program["inherited_inputs"]
    model = load_model(
        Path(inherited["conditional_copula_artifact"]),
        inherited["conditional_copula_artifact_sha256"],
    )
    selection = _selection_arrays(v35)
    train_handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    train_data = {domain: train_handles[domain][0] for domain in DOMAIN_ORDER}
    train_cache = {domain: train_handles[domain][1] for domain in DOMAIN_ORDER}
    donor_descriptors: dict[tuple[str, int], np.ndarray] = {}
    try:
        for query_domain in DOMAIN_ORDER:
            row = v35["development_domains"][query_domain]
            indices = np.asarray(selection[query_domain]["source_index"], dtype=np.int64)
            query_data, query_cache = _open_split(row, "validation")
            try:
                query_descriptors = [
                    standardize_descriptor(
                        descriptor_cube(query_data, query_cache, int(index)), fit
                    )
                    for index in indices
                ]
                handles: dict[str, h5py.File] = {}
                datasets: dict[str, dict[str, h5py.Dataset]] = {}
                partials: dict[str, Path] = {}
                for arm in ARMS:
                    path = output_root / arm / "development_candidate" / DOMAIN_KEYS[query_domain] / "ensemble16.h5"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(path.suffix + ".partial")
                    partials[arm] = partial
                    handles[arm] = h5py.File(partial, "w")
                    datasets[arm] = _create_ensemble(handles[arm])
                    for name, value in selection[query_domain].items():
                        handles[arm].create_dataset(name, data=value)
                maximum_dc = {arm: 0.0 for arm in ARMS}
                try:
                    for object_index, query_index in enumerate(indices):
                        query_backbone = _backbone(query_cache, int(query_index))[None]
                        alignment_reference = {
                            "aligned": object_index,
                            "shuffled_query_control": (object_index + 1) % len(indices),
                        }
                        for arm in ARMS:
                            datasets[arm]["alignment_reference_query_index"][object_index] = int(
                                indices[alignment_reference[arm]]
                            )
                        for member in range(16):
                            donor_source = DOMAIN_ORDER[
                                int(selection[query_domain]["donor_source"][object_index, member])
                            ]
                            donor_index = int(
                                selection[query_domain]["donor_index"][object_index, member]
                            )
                            isometry = int(
                                selection[query_domain]["donor_isometry"][object_index, member]
                            )
                            key = (donor_source, donor_index)
                            if key not in donor_descriptors:
                                donor_descriptors[key] = standardize_descriptor(
                                    descriptor_cube(
                                        train_data[donor_source], train_cache[donor_source], donor_index
                                    ),
                                    fit,
                                )
                            permutation, reflections = CUBE_ISOMETRIES[isometry]
                            donor_descriptor = apply_cube_isometry(
                                donor_descriptors[key], permutation, reflections
                            )
                            donor_backbone = _backbone(train_cache[donor_source], donor_index)[None]
                            donor_truth = np.asarray(
                                train_data[donor_source]["target"][donor_index], dtype=np.float32
                            )
                            uniform = conditional_forward(
                                donor_truth - donor_backbone, donor_backbone, model
                            )
                            uniform = apply_cube_isometry(uniform, permutation, reflections)
                            for arm in ARMS:
                                shift, before, after = best_periodic_shift(
                                    query_descriptors[alignment_reference[arm]], donor_descriptor
                                )
                                sample, dc = _transport(uniform, query_backbone, shift, model)
                                datasets[arm]["sample"][object_index, member] = sample
                                datasets[arm]["alignment_shift_coarse"][object_index, member] = shift
                                datasets[arm]["alignment_descriptor_mse_before"][object_index, member] = before
                                datasets[arm]["alignment_descriptor_mse_after"][object_index, member] = after
                                maximum_dc[arm] = max(maximum_dc[arm], dc)
                        for arm in ARMS:
                            datasets[arm]["conditional_mean"][object_index] = query_backbone
                            datasets[arm]["truth"][object_index] = np.asarray(
                                query_data["target"][int(query_index)], dtype=np.float32
                            )
                        print(
                            f"[v37-sample] {query_domain} {object_index + 1}/16",
                            flush=True,
                        )
                    for arm in ARMS:
                        handles[arm].attrs.update(
                            {
                                "schema": ENSEMBLE_SCHEMA,
                                "method": "bounded_query_aligned_conditional_copula",
                                "arm": arm,
                                "v37_program_sha256": PROGRAM_SHA256,
                                "descriptor": str(descriptor_path.resolve()),
                                "descriptor_sha256": descriptor_sha256,
                                "preflight": str(preflight_path.resolve()),
                                "preflight_sha256": preflight_sha256,
                                "parent_selection": str(Path(row["phase_object_selection"]).resolve()),
                                "parent_selection_sha256": row["phase_object_selection_sha256"],
                                "conditional_copula_model": inherited["conditional_copula_artifact"],
                                "conditional_copula_model_sha256": inherited[
                                    "conditional_copula_artifact_sha256"
                                ],
                                "pool_factor": POOL_FACTOR,
                                "maximum_shift_coarse_cells": MAX_SHIFT,
                                "diagnostic_k_h_mpc": 1.0,
                                "maximum_absolute_residual_dc": maximum_dc[arm],
                                "ensemble_members": 16,
                                "donor_reselection": False,
                                "validation_truth_used_for_alignment_fit_or_shift_selection": False,
                                "field_clipping": False,
                                "posthoc_Ak_used": False,
                                "worktree_clean_at_sampling": clean,
                                "sampling_code_commit": commit,
                                "Astrid_accessed": False,
                                "historical_EAGLE_accessed": False,
                                "complete": True,
                            }
                        )
                finally:
                    for handle in handles.values():
                        handle.close()
                for arm in ARMS:
                    final = partials[arm].with_suffix("")
                    os.replace(partials[arm], final)
            finally:
                query_data.close()
                query_cache.close()
    finally:
        for handle in train_data.values():
            handle.close()
        for handle in train_cache.values():
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--program", type=Path, required=True)
    fit.add_argument("--repo", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--program", type=Path, required=True)
    preflight.add_argument("--repo", type=Path, required=True)
    preflight.add_argument("--descriptor", type=Path, required=True)
    preflight.add_argument("--descriptor-sha256", required=True)
    preflight.add_argument("--out", type=Path, required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--program", type=Path, required=True)
    sample.add_argument("--repo", type=Path, required=True)
    sample.add_argument("--descriptor", type=Path, required=True)
    sample.add_argument("--descriptor-sha256", required=True)
    sample.add_argument("--preflight", type=Path, required=True)
    sample.add_argument("--preflight-sha256", required=True)
    sample.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit_descriptor(args.program, args.repo, args.out)
    elif args.command == "preflight":
        hard_preflight(
            args.program,
            args.repo,
            args.descriptor,
            args.descriptor_sha256,
            args.out,
        )
    else:
        sample_all(
            args.program,
            args.repo,
            args.descriptor,
            args.descriptor_sha256,
            args.preflight,
            args.preflight_sha256,
            args.out,
        )


if __name__ == "__main__":
    main()
