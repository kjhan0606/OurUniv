#!/usr/bin/env python
"""Frozen V39 train-only bijective local-patch copula transport."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v31_copula import conditional_forward, conditional_inverse, load_model
from hong2021_v34_nonlinear_sufficiency import periodic_oriented_patch, pooled_fields
from hong2021_v35_spectrum_phase import (
    _backbone,
    _open_split,
    load_program as load_v35_program,
)
from hong2021_v37_query_alignment import _selection_arrays, source_balanced_moments


PROGRAM_SCHEMA = "hong2021-v39-train-only-bijective-local-patch-copula-development-program-v1"
PROGRAM_SHA256 = "47f864be2511c4b33e599b3578cf6033e0362373d028d68edbb14b6969108edb"
DESCRIPTOR_SCHEMA = "hong2021-v39-train-only-overlap-patch-descriptor-v1"
PREFLIGHT_SCHEMA = "hong2021-v39-bijective-patch-copula-hard-preflight-v1"
ENSEMBLE_SCHEMA = "hong2021-v39-bijective-local-patch-copula-ensemble-v1"
ARMS = ("aligned_patch", "shuffled_query_control")
FIELDS = (
    "log1p_block_count",
    "block_mean_velocity_kms",
    "exact_population_velocity_dispersion_kms",
    "backbone_mean_y",
)
FACTOR = 8
GRID = 8
BLOCK = 8
BLOCKS = GRID**3
CONTEXT_FEATURES = len(FIELDS) * 27


def _spatial_cost() -> np.ndarray:
    coordinate = np.column_stack(np.unravel_index(np.arange(BLOCKS), (GRID,) * 3))
    difference = np.abs(coordinate[:, None, :] - coordinate[None, :, :])
    difference = np.minimum(difference, GRID - difference)
    return np.square(difference, dtype=np.float64).sum(axis=-1) / (3.0 * (GRID / 2.0) ** 2)


SPATIAL_COST = _spatial_cost()


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "V39 program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_implementation_sampling_or_development_evaluation"
    ):
        raise ValueError("V39 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v38_record"]).resolve(),
        parent["v38_record_sha256"],
        "V39 V38 record",
    )
    decision = record.get("decision", {})
    if (
        decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V39 parent conclusion or firewall differs")
    inherited = program["inherited_inputs"]
    v35_path = (repo / inherited["v35_program"]).resolve()
    if sha256_file(v35_path) != inherited["v35_program_sha256"]:
        raise ValueError("V39 V35 program hash differs")
    v35, _ = load_v35_program(v35_path, repo)
    if sha256_file((repo / inherited["v31_record"]).resolve()) != inherited["v31_record_sha256"]:
        raise ValueError("V39 V31 record hash differs")
    if sha256_file(Path(inherited["conditional_copula_artifact"])) != inherited["conditional_copula_artifact_sha256"]:
        raise ValueError("V39 V31 copula hash differs")
    return program, v35


def central_fields(data: h5py.File, cache: h5py.File, index: int) -> np.ndarray:
    values = pooled_fields(
        np.asarray(data["input"][index, 0], dtype=np.float32),
        np.asarray(data["input"][index, 1], dtype=np.float32),
        np.asarray(data["input"][index, 2], dtype=np.float32),
        _backbone(cache, index),
        FACTOR,
    )
    result = np.stack([values[name] for name in FIELDS]).astype(np.float32)
    if result.shape != (len(FIELDS), GRID, GRID, GRID) or not np.isfinite(result).all():
        raise ValueError("V39 central descriptor differs")
    return result


def descriptor_vectors(
    value: np.ndarray, fit: Mapping[str, Any]
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    mean = np.asarray(fit["mean"], dtype=np.float64)[:, None, None, None]
    std = np.asarray(fit["std"], dtype=np.float64)[:, None, None, None]
    standardized = (array - mean) / std
    patches = np.concatenate(
        [periodic_oriented_patch(field) for field in standardized], axis=-1
    )
    result = patches.reshape(BLOCKS, CONTEXT_FEATURES)
    if result.shape != (BLOCKS, CONTEXT_FEATURES) or not np.isfinite(result).all():
        raise ValueError("V39 overlap descriptor differs")
    return result


def assignment(
    query: np.ndarray, donor: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    query = np.asarray(query, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    if query.shape != (BLOCKS, CONTEXT_FEATURES) or donor.shape != query.shape:
        raise ValueError("V39 assignment descriptor shape differs")
    query_power = np.square(query).mean(axis=1)
    donor_power = np.square(donor).mean(axis=1)
    descriptor_cost = (
        query_power[:, None]
        + donor_power[None, :]
        - 2.0 * (query @ donor.T) / CONTEXT_FEATURES
    )
    descriptor_cost = np.maximum(descriptor_cost, 0.0)
    total = descriptor_cost + SPATIAL_COST
    rows, columns = linear_sum_assignment(total)
    if not np.array_equal(rows, np.arange(BLOCKS)) or not np.array_equal(
        np.sort(columns), np.arange(BLOCKS)
    ):
        raise RuntimeError("V39 assignment is not a complete permutation")
    selected = (np.arange(BLOCKS), columns)
    return columns.astype(np.int16), {
        "mean_descriptor_cost": float(descriptor_cost[selected].mean()),
        "mean_spatial_cost": float(SPATIAL_COST[selected].mean()),
        "mean_total_cost": float(total[selected].mean()),
        "nonidentity_fraction": float(np.mean(columns != np.arange(BLOCKS))),
    }


def cube_to_blocks(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field)
    if value.shape != (64, 64, 64):
        raise ValueError("V39 rank field must be a 64-cube")
    return (
        value.reshape(GRID, BLOCK, GRID, BLOCK, GRID, BLOCK)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(BLOCKS, BLOCK, BLOCK, BLOCK)
    )


def blocks_to_cube(blocks: np.ndarray) -> np.ndarray:
    value = np.asarray(blocks)
    if value.shape != (BLOCKS, BLOCK, BLOCK, BLOCK):
        raise ValueError("V39 block payload differs")
    return (
        value.reshape(GRID, GRID, GRID, BLOCK, BLOCK, BLOCK)
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(64, 64, 64)
    )


def transport_blocks(field: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (BLOCKS,) or not np.array_equal(
        np.sort(permutation), np.arange(BLOCKS)
    ):
        raise ValueError("V39 transport requires a block permutation")
    source = np.asarray(field)
    output = blocks_to_cube(cube_to_blocks(source)[permutation])
    if not np.array_equal(np.sort(output.reshape(-1)), np.sort(source.reshape(-1))):
        raise RuntimeError("V39 transport changed the donor rank multiset")
    return output


def fit_descriptor(
    program_path: Path, repo: Path, output: Path
) -> dict[str, Any]:
    _, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V39 descriptor fit requires a clean worktree")
    moments: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        total = np.zeros(len(FIELDS)); second = np.zeros(len(FIELDS)); count = 0
        data, cache = _open_split(row, "train")
        try:
            for index in range(int(row["train_objects"])):
                value = central_fields(data, cache, index).astype(np.float64)
                total += value.sum(axis=(1, 2, 3))
                second += np.square(value).sum(axis=(1, 2, 3))
                count += BLOCKS
                if (index + 1) % 32 == 0 or index + 1 == int(row["train_objects"]):
                    print(f"[v39-fit] {domain} {index + 1}/{row['train_objects']}", flush=True)
        finally:
            data.close(); cache.close()
        moments[domain] = (total, second, count)
    mean, std = source_balanced_moments(moments)
    report: dict[str, Any] = {
        "schema": DESCRIPTOR_SCHEMA,
        "status": "complete_train_only_source_balanced_fit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "fields": list(FIELDS),
        "factor": FACTOR,
        "grid": GRID,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "target_density_read": False,
        "validation_opened": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    if output.exists():
        raise RuntimeError("V39 refuses to overwrite descriptor artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_descriptor(path: Path, digest: str) -> dict[str, Any]:
    result = _verified_json(path, digest, "V39 descriptor")
    if (
        result.get("schema") != DESCRIPTOR_SCHEMA
        or result.get("status") != "complete_train_only_source_balanced_fit"
        or result.get("program_sha256") != PROGRAM_SHA256
        or result.get("fields") != list(FIELDS)
    ):
        raise ValueError("V39 descriptor metadata differs")
    return result


def make_sample(
    oriented_rank: np.ndarray,
    block_permutation: np.ndarray,
    query_backbone: np.ndarray,
    copula: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    transported = transport_blocks(oriented_rank[0], block_permutation)[None]
    residual = conditional_inverse(transported, query_backbone, copula).astype(np.float64)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
    sample = np.asarray(query_backbone, dtype=np.float64) + residual
    if not np.isfinite(sample).all():
        raise RuntimeError("V39 generated nonfinite density")
    return sample.astype(np.float32), dc


def preflight(
    program_path: Path,
    repo: Path,
    descriptor_path: Path,
    descriptor_sha: str,
    output: Path,
) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V39 preflight requires a clean worktree")
    fit = load_descriptor(descriptor_path, descriptor_sha)
    if fit.get("code_commit") != commit:
        raise ValueError("V39 descriptor commit differs")
    synthetic = np.arange(BLOCKS * CONTEXT_FEATURES, dtype=np.float64).reshape(BLOCKS, CONTEXT_FEATURES)
    synthetic /= synthetic.std()
    synthetic_permutation, _ = assignment(synthetic, synthetic)
    if not np.array_equal(synthetic_permutation, np.arange(BLOCKS)):
        raise RuntimeError("V39 synthetic identity assignment failed")
    selections = _selection_arrays(v35)
    domain = DOMAIN_ORDER[0]
    query_index = int(selections[domain]["source_index"][0])
    donor_source = DOMAIN_ORDER[int(selections[domain]["donor_source"][0, 0])]
    donor_index = int(selections[domain]["donor_index"][0, 0])
    isometry = int(selections[domain]["donor_isometry"][0, 0])
    query_data, query_cache = _open_split(v35["development_domains"][domain], "validation")
    donor_data, donor_cache = _open_split(v35["development_domains"][donor_source], "train")
    copula = load_model(Path(program["inherited_inputs"]["conditional_copula_artifact"]), program["inherited_inputs"]["conditional_copula_artifact_sha256"])
    try:
        query_vector = descriptor_vectors(central_fields(query_data, query_cache, query_index), fit)
        donor_central = central_fields(donor_data, donor_cache, donor_index)
        permutation_axes, reflections = CUBE_ISOMETRIES[isometry]
        donor_central = apply_cube_isometry(donor_central, permutation_axes, reflections)
        donor_vector = descriptor_vectors(donor_central, fit)
        mapping, diagnostics = assignment(query_vector, donor_vector)
        donor_backbone = _backbone(donor_cache, donor_index)[None]
        donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
        rank = conditional_forward(donor_truth - donor_backbone, donor_backbone, copula)
        rank = apply_cube_isometry(rank, permutation_axes, reflections)
        query_backbone = _backbone(query_cache, query_index)[None]
        sample, dc = make_sample(rank, mapping, query_backbone, copula)
    finally:
        query_data.close(); query_cache.close(); donor_data.close(); donor_cache.close()
    if not np.isfinite(sample).all() or dc > 1e-7:
        raise RuntimeError("V39 real sample preflight failed")
    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "descriptor": str(descriptor_path.resolve()),
        "descriptor_sha256": descriptor_sha,
        "real_assignment": diagnostics,
        "real_maximum_residual_dc": dc,
        "conditional_rank_multiset_preserved": True,
        "validation_truth_used_for_assignment": False,
        "donor_translation": False,
        "donor_reselection": False,
        "density_field_clipping": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    if output.exists():
        raise RuntimeError("V39 refuses existing preflight")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _new_ensemble(handle: h5py.File) -> dict[str, h5py.Dataset]:
    return {
        "sample": handle.create_dataset(
            "sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4",
            chunks=(1, 1, 1, 64, 64, 64), compression="lzf",
        ),
        "conditional_mean": handle.create_dataset(
            "conditional_mean", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"
        ),
        "truth": handle.create_dataset(
            "truth", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"
        ),
        "block_permutation": handle.create_dataset(
            "block_permutation", shape=(16, 16, BLOCKS), dtype="i2"
        ),
        "mean_descriptor_cost": handle.create_dataset(
            "mean_descriptor_cost", shape=(16, 16), dtype="f4"
        ),
        "mean_spatial_cost": handle.create_dataset(
            "mean_spatial_cost", shape=(16, 16), dtype="f4"
        ),
        "mean_total_cost": handle.create_dataset(
            "mean_total_cost", shape=(16, 16), dtype="f4"
        ),
        "nonidentity_fraction": handle.create_dataset(
            "nonidentity_fraction", shape=(16, 16), dtype="f4"
        ),
        "alignment_reference_query_index": handle.create_dataset(
            "alignment_reference_query_index", shape=(16,), dtype="i4"
        ),
        "conditional_rank_multiset_sha256": handle.create_dataset(
            "conditional_rank_multiset_sha256", shape=(16, 16, 32), dtype="u1"
        ),
    }


def sample_all(
    program_path: Path,
    repo: Path,
    descriptor_path: Path,
    descriptor_sha: str,
    preflight_path: Path,
    preflight_sha: str,
    output_root: Path,
) -> None:
    program, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V39 sampling requires a clean worktree")
    fit = load_descriptor(descriptor_path, descriptor_sha)
    checked = _verified_json(preflight_path, preflight_sha, "V39 preflight")
    if (
        fit.get("code_commit") != commit
        or checked.get("schema") != PREFLIGHT_SCHEMA
        or checked.get("status") != "pass"
        or checked.get("code_commit") != commit
        or checked.get("descriptor_sha256") != descriptor_sha
    ):
        raise ValueError("V39 fit/preflight binding differs")
    if output_root.exists():
        raise RuntimeError("V39 refuses a pre-existing output root")
    copula = load_model(
        Path(program["inherited_inputs"]["conditional_copula_artifact"]),
        program["inherited_inputs"]["conditional_copula_artifact_sha256"],
    )
    selections = _selection_arrays(v35)
    train = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    donor_descriptor_cache: dict[tuple[str, int, int], np.ndarray] = {}
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            indices = np.asarray(selections[domain]["source_index"], dtype=np.int64)
            query_data, query_cache = _open_split(row, "validation")
            handles: dict[str, h5py.File] = {}
            datasets: dict[str, dict[str, h5py.Dataset]] = {}
            partials: dict[str, Path] = {}
            maximum_dc = {arm: 0.0 for arm in ARMS}
            try:
                query_vectors = [
                    descriptor_vectors(
                        central_fields(query_data, query_cache, int(index)), fit
                    )
                    for index in indices
                ]
                for arm in ARMS:
                    path = output_root / arm / "development_candidate" / DOMAIN_KEYS[domain] / "ensemble16.h5"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(path.suffix + ".partial")
                    partials[arm] = partial
                    handles[arm] = h5py.File(partial, "w")
                    datasets[arm] = _new_ensemble(handles[arm])
                    for name, value in selections[domain].items():
                        handles[arm].create_dataset(name, data=value)
                for object_index, query_index in enumerate(indices):
                    query_backbone = _backbone(query_cache, int(query_index))[None]
                    reference = {
                        "aligned_patch": object_index,
                        "shuffled_query_control": (object_index + 1) % len(indices),
                    }
                    for arm in ARMS:
                        datasets[arm]["alignment_reference_query_index"][object_index] = int(indices[reference[arm]])
                    for member in range(16):
                        donor_source = DOMAIN_ORDER[int(selections[domain]["donor_source"][object_index, member])]
                        donor_index = int(selections[domain]["donor_index"][object_index, member])
                        isometry = int(selections[domain]["donor_isometry"][object_index, member])
                        permutation_axes, reflections = CUBE_ISOMETRIES[isometry]
                        key = (donor_source, donor_index, isometry)
                        donor_data, donor_cache = train[donor_source]
                        if key not in donor_descriptor_cache:
                            donor_central = apply_cube_isometry(
                                central_fields(donor_data, donor_cache, donor_index),
                                permutation_axes,
                                reflections,
                            )
                            donor_descriptor_cache[key] = descriptor_vectors(donor_central, fit)
                        donor_backbone = _backbone(donor_cache, donor_index)[None]
                        donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
                        rank = conditional_forward(donor_truth - donor_backbone, donor_backbone, copula)
                        rank = apply_cube_isometry(rank, permutation_axes, reflections)
                        rank_digest = np.frombuffer(
                            hashlib.sha256(np.sort(rank.reshape(-1)).tobytes()).digest(),
                            dtype=np.uint8,
                        )
                        for arm in ARMS:
                            mapping, diagnostics = assignment(
                                query_vectors[reference[arm]], donor_descriptor_cache[key]
                            )
                            sample, dc = make_sample(rank, mapping, query_backbone, copula)
                            datasets[arm]["sample"][object_index, member] = sample
                            datasets[arm]["block_permutation"][object_index, member] = mapping
                            for name in (
                                "mean_descriptor_cost", "mean_spatial_cost",
                                "mean_total_cost", "nonidentity_fraction",
                            ):
                                datasets[arm][name][object_index, member] = diagnostics[name]
                            datasets[arm]["conditional_rank_multiset_sha256"][object_index, member] = rank_digest
                            maximum_dc[arm] = max(maximum_dc[arm], dc)
                    for arm in ARMS:
                        datasets[arm]["conditional_mean"][object_index] = query_backbone
                        datasets[arm]["truth"][object_index] = np.asarray(
                            query_data["target"][int(query_index)], dtype=np.float32
                        )
                    print(f"[v39-sample] {domain} {object_index + 1}/16", flush=True)
                for arm in ARMS:
                    handles[arm].attrs.update(
                        {
                            "schema": ENSEMBLE_SCHEMA,
                            "method": "train_only_bijective_local_patch_copula",
                            "arm": arm,
                            "v39_program_sha256": PROGRAM_SHA256,
                            "descriptor": str(descriptor_path.resolve()),
                            "descriptor_sha256": descriptor_sha,
                            "preflight": str(preflight_path.resolve()),
                            "preflight_sha256": preflight_sha,
                            "parent_selection": str(Path(row["phase_object_selection"]).resolve()),
                            "parent_selection_sha256": row["phase_object_selection_sha256"],
                            "conditional_copula_model": program["inherited_inputs"]["conditional_copula_artifact"],
                            "conditional_copula_model_sha256": program["inherited_inputs"]["conditional_copula_artifact_sha256"],
                            "block_factor": FACTOR,
                            "block_grid": GRID,
                            "diagnostic_k_h_mpc": 1.0,
                            "maximum_absolute_residual_dc": maximum_dc[arm],
                            "ensemble_members": 16,
                            "conditional_rank_multiset_preserved": True,
                            "additive_query_predictor": False,
                            "donor_translation": False,
                            "donor_reselection": False,
                            "validation_truth_used_for_descriptor_fit_or_assignment": False,
                            "density_field_clipping": False,
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
                query_data.close(); query_cache.close()
            for arm in ARMS:
                os.replace(partials[arm], partials[arm].with_suffix(""))
    finally:
        for data, cache in train.values():
            data.close(); cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--program", type=Path, required=True)
    fit.add_argument("--repo", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--program", type=Path, required=True)
    check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--descriptor", type=Path, required=True)
    check.add_argument("--descriptor-sha256", required=True)
    check.add_argument("--out", type=Path, required=True)
    sample = commands.add_parser("sample")
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
        preflight(args.program, args.repo, args.descriptor, args.descriptor_sha256, args.out)
    else:
        sample_all(
            args.program, args.repo, args.descriptor, args.descriptor_sha256,
            args.preflight, args.preflight_sha256, args.out,
        )


if __name__ == "__main__":
    main()
