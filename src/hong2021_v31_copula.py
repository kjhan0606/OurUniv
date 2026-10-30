#!/usr/bin/env python
"""Train-only physical conditional-copula fit and frozen V31 sampler."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER


REGISTRY_SCHEMA = "hong2021-v31-train-only-physical-conditional-copula-development-program-v1"
REGISTRY_SHA256 = "48fca645621078d512e0e667eac925292cbb22bfa693ffa776fbc2627f8646ff"
MODEL_SCHEMA = "hong2021-v31-train-only-physical-conditional-copula-v1"
ENSEMBLE_SCHEMA = "hong2021-v31-physical-conditional-copula-ensemble-v1"
PREFLIGHT_SCHEMA = "hong2021-v31-physical-conditional-copula-hard-preflight-v1"
BACKBONE_BINS = 32


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    registry = _verified_json(path.resolve(), REGISTRY_SHA256, "V31 registry")
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status")
        != "frozen_before_implementation_fit_sampling_or_development_evaluation"
    ):
        raise ValueError("V31 registry schema or status differs")
    parent = registry["parent_evidence"]
    record_path = (repo / parent["v30_record"]).resolve()
    record = _verified_json(record_path, parent["v30_record_sha256"], "V30 record")
    output = _verified_json(
        Path(parent["v30_output"]), parent["v30_output_sha256"], "V30 output"
    )
    if (
        record.get("classification") != parent["required_classification"]
        or record.get("next") != parent["required_next"]
        or record.get("decision_digest_sha256")
        != parent["v30_decision_digest_sha256"]
        or output.get("decision_digest_sha256")
        != parent["v30_decision_digest_sha256"]
        or canonical_digest(output) != parent["v30_decision_digest_sha256"]
    ):
        raise ValueError("V31 V30 parent conclusion differs")
    for source in DOMAIN_ORDER:
        row = registry["train_only_fit"]["domains"][source]
        for key in ("data", "cache"):
            artifact = Path(row[key])
            if sha256_file(artifact) != row[f"{key}_sha256"]:
                raise ValueError(f"V31 {source} {key} hash differs")
        if int(row["objects"]) <= 0:
            raise ValueError("V31 train object count is invalid")
        selected = registry["frozen_v28_selections"][source]
        if sha256_file(Path(selected["ensemble"])) != selected["sha256"]:
            raise ValueError(f"V31 V28 {source} selection ensemble differs")
    return registry


def quantile_levels() -> np.ndarray:
    central = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    lower = np.power(10.0, np.linspace(-7.0, -1.0, 1024, dtype=np.float64))
    values = np.unique(np.concatenate(([0.0, 1.0], central, lower, 1.0 - lower)))
    if values[0] != 0 or values[-1] != 1 or np.any(np.diff(values) <= 0):
        raise RuntimeError("V31 quantile levels are invalid")
    return values


def equal_source_weighted_quantile(
    values: Mapping[str, np.ndarray], quantiles: np.ndarray
) -> np.ndarray:
    """Empirical quantiles with exactly equal total weight per nonempty source."""
    if tuple(values) != DOMAIN_ORDER:
        raise ValueError("V31 weighted quantile source order differs")
    arrays = [np.asarray(values[source], dtype=np.float64).reshape(-1) for source in DOMAIN_ORDER]
    if any(array.size == 0 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("V31 weighted quantile requires finite nonempty sources")
    combined = np.concatenate(arrays)
    weights = np.concatenate(
        [np.full(array.size, 1.0 / (len(arrays) * array.size)) for array in arrays]
    )
    order = np.argsort(combined, kind="stable")
    sorted_values = combined[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    cumulative[-1] = 1.0
    return np.interp(
        np.asarray(quantiles, dtype=np.float64),
        cumulative,
        sorted_values,
        left=sorted_values[0],
        right=sorted_values[-1],
    )


def strict_monotonic(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Repair empirical ties inside exact finite train-derived endpoints."""
    result = np.asarray(values, dtype=np.float64).copy()
    if not np.isfinite(result).all() or not minimum < maximum:
        raise ValueError("V31 monotonic support is invalid")
    result[0], result[-1] = minimum, maximum
    for index in range(1, len(result) - 1):
        if result[index] <= result[index - 1]:
            result[index] = np.nextafter(result[index - 1], np.inf)
    for index in range(len(result) - 2, 0, -1):
        if result[index] >= result[index + 1]:
            result[index] = np.nextafter(result[index + 1], -np.inf)
    if result[0] != minimum or result[-1] != maximum or np.any(np.diff(result) <= 0):
        raise RuntimeError("V31 empirical table cannot be repaired strictly")
    return result


def lattice_slices(index: int) -> tuple[slice, slice, slice]:
    offsets = (index % 2, (index // 2) % 2, (index // 4) % 2)
    return tuple(slice(offset, None, 2) for offset in offsets)  # type: ignore[return-value]


def _baseline(cache: h5py.File, index: int) -> np.ndarray:
    mean = np.asarray(cache["conditional_mean"][index, 0], dtype=np.float32)
    return mean + np.float32(cache["predicted_residual_dc"][index])


def _fit_samples(registry: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    means: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    for source in DOMAIN_ORDER:
        row = registry["train_only_fit"]["domains"][source]
        mean_parts = []
        residual_parts = []
        with h5py.File(row["data"], "r") as data, h5py.File(row["cache"], "r") as cache:
            objects = int(row["objects"])
            if len(data["target"]) != objects or len(cache["conditional_mean"]) != objects:
                raise ValueError(f"V31 {source} train object count differs")
            for index in range(objects):
                selected = lattice_slices(index)
                mean = _baseline(cache, index)
                truth = np.asarray(data["target"][index, 0], dtype=np.float32)
                mean_parts.append(mean[selected].reshape(-1))
                residual_parts.append((truth - mean)[selected].reshape(-1))
                if (index + 1) % 64 == 0 or index + 1 == objects:
                    print(f"[fit-lattice] {source} {index + 1}/{objects}", flush=True)
        means[source] = np.concatenate(mean_parts).astype(np.float32, copy=False)
        residuals[source] = np.concatenate(residual_parts).astype(np.float32, copy=False)
    return means, residuals


def _exact_tail_anchors(
    registry: dict[str, Any], edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    minima = np.full(BACKBONE_BINS, np.inf, dtype=np.float64)
    maxima = np.full(BACKBONE_BINS, -np.inf, dtype=np.float64)
    counts: dict[str, list[int]] = {}
    for source in DOMAIN_ORDER:
        row = registry["train_only_fit"]["domains"][source]
        source_counts = np.zeros(BACKBONE_BINS, dtype=np.int64)
        with h5py.File(row["data"], "r") as data, h5py.File(row["cache"], "r") as cache:
            for index in range(int(row["objects"])):
                mean = _baseline(cache, index)
                residual = np.asarray(data["target"][index, 0], dtype=np.float32) - mean
                bins = np.searchsorted(edges[1:-1], mean, side="right")
                flattened_bins = bins.reshape(-1)
                flattened_residual = residual.astype(np.float64, copy=False).reshape(-1)
                local_minimum = np.full(BACKBONE_BINS, np.inf, dtype=np.float64)
                local_maximum = np.full(BACKBONE_BINS, -np.inf, dtype=np.float64)
                np.minimum.at(local_minimum, flattened_bins, flattened_residual)
                np.maximum.at(local_maximum, flattened_bins, flattened_residual)
                minima = np.minimum(minima, local_minimum)
                maxima = np.maximum(maxima, local_maximum)
                source_counts += np.bincount(
                    flattened_bins, minlength=BACKBONE_BINS
                ).astype(np.int64)
                if (index + 1) % 64 == 0 or index + 1 == int(row["objects"]):
                    print(f"[fit-anchors] {source} {index + 1}/{row['objects']}", flush=True)
        if np.any(source_counts == 0):
            raise RuntimeError(f"V31 {source} has an empty physical backbone bin")
        counts[source] = source_counts.tolist()
    if not np.isfinite(minima).all() or not np.isfinite(maxima).all() or np.any(minima >= maxima):
        raise RuntimeError("V31 exact physical residual anchors are invalid")
    return minima, maxima, counts


def fit_model(registry_path: Path, repo: Path, artifact: Path, report_path: Path) -> dict[str, Any]:
    registry = load_program(registry_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V31 fit requires a clean committed worktree")
    if any(path.exists() for path in (artifact, report_path)):
        raise RuntimeError("V31 refuses to overwrite fitted artifacts")
    means, residuals = _fit_samples(registry)
    edge_quantiles = np.linspace(0.0, 1.0, BACKBONE_BINS + 1)
    edges = equal_source_weighted_quantile(means, edge_quantiles)
    edges = strict_monotonic(edges, float(min(value.min() for value in means.values())), float(max(value.max() for value in means.values())))
    levels = quantile_levels()
    lattice_bins = {
        source: np.searchsorted(edges[1:-1], means[source], side="right")
        for source in DOMAIN_ORDER
    }
    minima, maxima, full_counts = _exact_tail_anchors(registry, edges)
    table = np.empty((BACKBONE_BINS, len(levels)), dtype=np.float64)
    lattice_counts: dict[str, list[int]] = {
        source: np.bincount(lattice_bins[source], minlength=BACKBONE_BINS).tolist()
        for source in DOMAIN_ORDER
    }
    for bin_index in range(BACKBONE_BINS):
        rows = {
            source: residuals[source][lattice_bins[source] == bin_index]
            for source in DOMAIN_ORDER
        }
        values = equal_source_weighted_quantile(rows, levels)
        table[bin_index] = strict_monotonic(
            values, minima[bin_index], maxima[bin_index]
        )
        print(f"[fit-table] {bin_index + 1}/{BACKBONE_BINS}", flush=True)
    maximum_roundtrip = 0.0
    for row in table:
        encoded = np.interp(row, row, levels)
        decoded = np.interp(encoded, levels, row)
        maximum_roundtrip = max(maximum_roundtrip, float(np.max(np.abs(decoded - row))))
    metadata = {
        "schema": MODEL_SCHEMA,
        "registry_sha256": REGISTRY_SHA256,
        "fit_code_commit": commit,
        "source_weight": "one_third_each",
        "backbone_bins": BACKBONE_BINS,
        "lattice_stride": 2,
        "fit_uses_validation_truth": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    partial = artifact.with_suffix(artifact.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(
            handle,
            backbone_edges=edges,
            quantile_levels=levels,
            residual_quantiles=table,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(partial, artifact)
    report: dict[str, Any] = {
        **metadata,
        "status": "complete_train_only_fit",
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256_file(artifact),
        "backbone_edges": edges.tolist(),
        "quantile_levels": int(len(levels)),
        "residual_table_shape": list(table.shape),
        "lattice_samples": {source: int(len(means[source])) for source in DOMAIN_ORDER},
        "lattice_counts_per_backbone_bin": lattice_counts,
        "full_counts_per_backbone_bin": full_counts,
        "exact_residual_minimum_per_bin": minima.tolist(),
        "exact_residual_maximum_per_bin": maxima.tolist(),
        "maximum_table_roundtrip_absolute_y": maximum_roundtrip,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial_report, report_path)
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_model(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("V31 copula artifact hash differs")
    with np.load(path, allow_pickle=False) as handle:
        model = {name: np.asarray(handle[name]) for name in handle.files}
    metadata = json.loads(str(model.pop("metadata_json").item()))
    if metadata.get("schema") != MODEL_SCHEMA or metadata.get("registry_sha256") != REGISTRY_SHA256:
        raise ValueError("V31 copula model metadata differs")
    edges = np.asarray(model["backbone_edges"], dtype=np.float64)
    levels = np.asarray(model["quantile_levels"], dtype=np.float64)
    table = np.asarray(model["residual_quantiles"], dtype=np.float64)
    if (
        edges.shape != (BACKBONE_BINS + 1,)
        or table.shape != (BACKBONE_BINS, len(levels))
        or np.any(np.diff(edges) <= 0)
        or np.any(np.diff(levels) <= 0)
        or np.any(np.diff(table, axis=1) <= 0)
        or not np.isfinite(table).all()
    ):
        raise ValueError("V31 copula model arrays differ")
    return {**model, "metadata": metadata}


def conditional_forward(
    residual: np.ndarray, backbone: np.ndarray, model: Mapping[str, Any]
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    backbone = np.asarray(backbone, dtype=np.float64)
    if residual.shape != backbone.shape:
        raise ValueError("V31 forward fields differ")
    edges = np.asarray(model["backbone_edges"], dtype=np.float64)
    levels = np.asarray(model["quantile_levels"], dtype=np.float64)
    table = np.asarray(model["residual_quantiles"], dtype=np.float64)
    bins = np.searchsorted(edges[1:-1], backbone, side="right")
    output = np.empty_like(residual)
    for bin_index in range(BACKBONE_BINS):
        selected = bins == bin_index
        output[selected] = np.interp(
            residual[selected], table[bin_index], levels,
            left=levels[0], right=levels[-1],
        )
    return output.astype(np.float32)


def conditional_inverse(
    uniform: np.ndarray, backbone: np.ndarray, model: Mapping[str, Any]
) -> np.ndarray:
    uniform = np.asarray(uniform, dtype=np.float64)
    backbone = np.asarray(backbone, dtype=np.float64)
    if uniform.shape != backbone.shape or np.any((uniform < 0) | (uniform > 1)):
        raise ValueError("V31 inverse fields differ or leave unit support")
    edges = np.asarray(model["backbone_edges"], dtype=np.float64)
    levels = np.asarray(model["quantile_levels"], dtype=np.float64)
    table = np.asarray(model["residual_quantiles"], dtype=np.float64)
    bins = np.searchsorted(edges[1:-1], backbone, side="right")
    output = np.empty_like(uniform)
    for bin_index in range(BACKBONE_BINS):
        selected = bins == bin_index
        output[selected] = np.interp(uniform[selected], levels, table[bin_index])
    return output.astype(np.float32)


def transport_conditional_residual(
    donor_truth: np.ndarray,
    donor_backbone: np.ndarray,
    query_backbone: np.ndarray,
    isometry: int,
    model: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    if not 0 <= isometry < len(CUBE_ISOMETRIES):
        raise ValueError("V31 donor isometry is invalid")
    residual = np.asarray(donor_truth, dtype=np.float32) - np.asarray(
        donor_backbone, dtype=np.float32
    )
    uniform = conditional_forward(residual, donor_backbone, model)
    permutation, reflections = CUBE_ISOMETRIES[isometry]
    oriented = apply_cube_isometry(uniform, permutation, reflections)
    transported = conditional_inverse(oriented, query_backbone, model).astype(np.float64)
    transported -= transported.mean(axis=(-3, -2, -1), keepdims=True)
    maximum_dc = float(np.max(np.abs(transported.mean(axis=(-3, -2, -1)))))
    sample = np.asarray(query_backbone, dtype=np.float64) + transported
    return sample.astype(np.float32), maximum_dc


def sample_all(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry = load_program(args.registry.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V31 sampling requires a clean committed worktree")
    preflight = _verified_json(args.preflight.resolve(), args.preflight_sha256, "V31 preflight")
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "pass"
        or preflight.get("code_commit") != commit
        or preflight.get("registry_sha256") != REGISTRY_SHA256
    ):
        raise ValueError("V31 preflight differs")
    fit_report = _verified_json(args.model_report.resolve(), args.model_report_sha256, "V31 fit report")
    if fit_report.get("artifact_sha256") != args.model_sha256:
        raise ValueError("V31 model/report binding differs")
    model = load_model(args.model.resolve(), args.model_sha256)
    output_root = args.out.resolve()
    if output_root.exists():
        raise RuntimeError("V31 refuses pre-existing candidate output")
    output_root.mkdir(parents=True)
    train_data = {
        source: h5py.File(registry["train_only_fit"]["domains"][source]["data"], "r")
        for source in DOMAIN_ORDER
    }
    train_cache = {
        source: h5py.File(registry["train_only_fit"]["domains"][source]["cache"], "r")
        for source in DOMAIN_ORDER
    }
    try:
        for source in DOMAIN_ORDER:
            parent = registry["frozen_v28_selections"][source]
            domain_root = output_root / DOMAIN_KEYS[source]
            domain_root.mkdir()
            output = domain_root / "ensemble16.h5"
            partial = output.with_suffix(".h5.partial")
            with h5py.File(parent["ensemble"], "r") as old:
                query_cache_path = Path(str(old.attrs["source_cache"]))
                with h5py.File(query_cache_path, "r") as query_cache:
                    query_data_path = Path(str(query_cache.attrs["source_data"]))
                    with h5py.File(query_data_path, "r") as query_data, h5py.File(partial, "w") as new:
                        if (
                            sha256_file(query_cache_path) != str(old.attrs["source_cache_sha256"])
                            or sha256_file(query_data_path) != str(old.attrs["source_data_sha256"])
                        ):
                            raise ValueError(f"V31 {source} validation provenance differs")
                        sample_ds = new.create_dataset(
                            "sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4",
                            chunks=(1, 1, 1, 64, 64, 64), compression="lzf",
                        )
                        mean_ds = new.create_dataset(
                            "conditional_mean", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"
                        )
                        truth_ds = new.create_dataset(
                            "truth", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"
                        )
                        for name in (
                            "source_index", "donor_source", "donor_index", "donor_isometry",
                            "donor_distance", "predicted_residual_dc", "predicted_band_scales",
                        ):
                            old.copy(name, new)
                        maximum_dc = 0.0
                        for object_index, query_index in enumerate(old["source_index"][:]):
                            query_backbone = _baseline(query_cache, int(query_index))[None]
                            for member in range(16):
                                donor_source = DOMAIN_ORDER[int(old["donor_source"][object_index, member])]
                                donor_index = int(old["donor_index"][object_index, member])
                                donor_backbone = _baseline(train_cache[donor_source], donor_index)[None]
                                donor_truth = np.asarray(
                                    train_data[donor_source]["target"][donor_index], dtype=np.float32
                                )
                                sample, dc = transport_conditional_residual(
                                    donor_truth, donor_backbone, query_backbone,
                                    int(old["donor_isometry"][object_index, member]), model,
                                )
                                maximum_dc = max(maximum_dc, dc)
                                if not np.isfinite(sample).all():
                                    raise RuntimeError("V31 produced nonfinite physical density")
                                sample_ds[object_index, member] = sample
                            mean_ds[object_index] = query_backbone
                            # Development truth is copied only after all samples for this query exist.
                            truth_ds[object_index] = np.asarray(
                                query_data["target"][query_index], dtype=np.float32
                            )
                            print(f"[sample] V31 {source} {object_index + 1}/16", flush=True)
                        new.attrs.update(
                            {
                                "schema": ENSEMBLE_SCHEMA,
                                "method": "train_only_physical_residual_conditional_copula",
                                "v31_registry_sha256": REGISTRY_SHA256,
                                "conditional_copula_model": str(args.model.resolve()),
                                "conditional_copula_model_sha256": args.model_sha256,
                                "conditional_copula_report": str(args.model_report.resolve()),
                                "conditional_copula_report_sha256": args.model_report_sha256,
                                "parent_v28_ensemble": str(Path(parent["ensemble"]).resolve()),
                                "parent_v28_ensemble_sha256": parent["sha256"],
                                "source_cache_sha256": str(old.attrs["source_cache_sha256"]),
                                "source_data_sha256": str(old.attrs["source_data_sha256"]),
                                "ensemble_members": 16,
                                "diagnostic_k_h_mpc": 1.0,
                                "maximum_absolute_transported_residual_dc": maximum_dc,
                                "donor_reselection": False,
                                "selection_uses_validation_truth": False,
                                "copula_fit_uses_validation_truth": False,
                                "worktree_clean_at_sampling": clean,
                                "sampling_code_commit": commit,
                                "hard_preflight": str(args.preflight.resolve()),
                                "hard_preflight_sha256": args.preflight_sha256,
                                "Astrid_accessed": False,
                                "historical_EAGLE_accessed": False,
                                "complete": True,
                            }
                        )
            os.replace(partial, output)
    finally:
        for handle in (*train_data.values(), *train_cache.values()):
            handle.close()
    print(f"[out] {output_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--registry", type=Path, required=True)
    fit.add_argument("--repo", type=Path, required=True)
    fit.add_argument("--artifact", type=Path, required=True)
    fit.add_argument("--report", type=Path, required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--registry", type=Path, required=True)
    sample.add_argument("--repo", type=Path, required=True)
    sample.add_argument("--model", type=Path, required=True)
    sample.add_argument("--model-sha256", required=True)
    sample.add_argument("--model-report", type=Path, required=True)
    sample.add_argument("--model-report-sha256", required=True)
    sample.add_argument("--preflight", type=Path, required=True)
    sample.add_argument("--preflight-sha256", required=True)
    sample.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fit":
        fit_model(args.registry, args.repo, args.artifact, args.report)
    else:
        sample_all(args)


if __name__ == "__main__":
    main()
