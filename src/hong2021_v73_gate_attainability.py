#!/usr/bin/env python
"""Train-truth, cluster-aware attainability audit for the frozen V72 gate."""
from __future__ import annotations

import argparse
import heapq
import json
import os
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_residual_evaluate import (
    BANDS,
    SpectralBinner,
    density_statistics,
)
from hong2021_evaluate import OpenBoundaryTwoPoint
from hong2021_v6_gate import ENVIRONMENT_FIELDS
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v72_sqt import scalar_energy_score


PROGRAM_SCHEMA = "hong2021-v73-train-truth-gate-attainability-audit-program-v1"
PROGRAM_STATUS = "frozen_before_any_train_target_payload_read_or_audit_implementation"
PROGRAM_SHA256 = "cf92b53504c6501faf9d6043661f070a3e7458dc57ccd94bff77365760a1cf05"
PROGRAM_FREEZE_COMMIT = "1607dc0064e1ce1a57629196156985403b893214"
RESULT_SCHEMA = "hong2021-v73-train-truth-gate-attainability-audit-result-v1"
SUMMARY_SCHEMA = "hong2021-v73-train-truth-summary-cache-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
TOP_VALUES = 2048
VOXELS_PER_CUBE = 64**3
HISTOGRAM_EDGES = np.linspace(-4.0, 6.0, 401)
SCALE_RANGES = ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0))
V72_ROOT = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v72_sqt_stage_A"
)
V72_DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
V72_CANDIDATE = "conditioning_stratified_spatial_quantile_transport"
V72_CONTROL = "independent_voxel_V63_marginal"
ALL_ENVIRONMENT_FIELDS = (
    "volume_fraction_rho_lt_0.1",
    "volume_fraction_rho_lt_0.5",
    "volume_fraction_rho_gt_10",
    "volume_fraction_rho_gt_100",
    "local_peak_count_rho_gt_10",
    "local_peak_count_rho_gt_100",
    "local_void_count_rho_lt_0.1",
)


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    path = path.resolve()
    if sha256_file(path) != PROGRAM_SHA256:
        raise ValueError("V73 attainability program hash differs")
    program = strict_json(path)
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("scope_limits", {}).get("this_is_not_a_V73_generator")
        is not True
        or program.get("scope_limits", {}).get("no_fresh_partition_consumed")
        is not True
    ):
        raise ValueError("V73 attainability program schema or firewall differs")
    for key in (
        "v72_result_record",
        "v72_program",
        "v70_model_program",
        "v70_train_gate_program",
        "v35_data_registry",
    ):
        if sha256_file(resolve_path(repo, program["parent_evidence"][key])) != program[
            "parent_evidence"
        ][f"{key}_sha256"]:
            raise ValueError(f"V73 local parent differs: {key}")
    for key, value in program["frozen_measurement_sources"].items():
        if key.endswith("_sha256"):
            source_key = key.removesuffix("_sha256")
            if sha256_file(resolve_path(repo, program["frozen_measurement_sources"][source_key])) != value:
                raise ValueError(f"V73 measurement source differs: {source_key}")
    return program


def tng_group(position: np.ndarray) -> np.ndarray:
    value = np.asarray(position, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("TNG positions must have shape [objects,3]")
    return ((value[:, 0] >= 37.5).astype(np.int8) * 2 + (value[:, 2] >= 37.5)).astype(
        np.int16
    )


def _summarize_chunk(
    domain: str,
    data_path: str,
    cache_path: str,
    indices: list[int],
) -> dict[str, np.ndarray]:
    binner = SpectralBinner(64, 0.3125)
    estimator = OpenBoundaryTwoPoint(64, 0.3125, 10.0)
    env_names = ALL_ENVIRONMENT_FIELDS
    result: dict[str, list[Any]] = {
        "index": [],
        "truth_top": [],
        "truth_max": [],
        "truth_delta2": [],
        "truth_power": [],
        "truth_2pcf": [],
        "truth_hist": [],
        "truth_env": [],
        "residual_ms": [],
        "det_top": [],
        "det_max": [],
        "det_delta2": [],
        "det_power": [],
        "det_2pcf": [],
        "det_hist": [],
        "det_env": [],
    }
    with h5py.File(data_path, "r") as data, h5py.File(cache_path, "r") as cache:
        for index in indices:
            target = np.asarray(data["target"][index, 0], dtype=np.float32)
            deterministic = np.asarray(
                cache["conditional_mean"][index, 0], dtype=np.float32
            )
            if (
                target.shape != (64, 64, 64)
                or deterministic.shape != target.shape
                or not np.isfinite(target).all()
                or not np.isfinite(deterministic).all()
            ):
                raise ValueError(f"invalid {domain} train field at {index}")
            log_truth = 4.5 * target
            flat = log_truth.reshape(-1)
            top = np.partition(flat, flat.size - TOP_VALUES)[-TOP_VALUES:]
            top = np.sort(top)[::-1].astype(np.float32)
            density = np.power(10.0, log_truth, dtype=np.float64)
            log_deterministic = 4.5 * deterministic
            deterministic_flat = log_deterministic.reshape(-1)
            deterministic_top = np.partition(
                deterministic_flat, deterministic_flat.size - TOP_VALUES
            )[-TOP_VALUES:]
            deterministic_top = np.sort(deterministic_top)[::-1].astype(np.float32)
            deterministic_density = np.power(
                10.0, log_deterministic, dtype=np.float64
            )
            if not np.isfinite(density).all() or not np.isfinite(
                deterministic_density
            ).all():
                raise ValueError(f"non-finite {domain} physical density at {index}")
            delta = density - 1.0
            deterministic_delta = deterministic_density - 1.0
            residual = target.astype(np.float64) - deterministic.astype(np.float64)
            residual -= residual.mean()
            truth_env = density_statistics(density)
            deterministic_env = density_statistics(deterministic_density)
            result["index"].append(index)
            result["truth_top"].append(top)
            result["truth_max"].append(float(top[0]))
            result["truth_delta2"].append(float(np.square(delta).mean()))
            result["truth_power"].append(binner.power(target[None])[0])
            result["truth_2pcf"].append(estimator(delta))
            result["truth_hist"].append(
                np.histogram(log_truth, HISTOGRAM_EDGES)[0].astype(np.int64)
            )
            result["truth_env"].append([truth_env[name] for name in env_names])
            result["residual_ms"].append(float(np.square(residual).mean()))
            result["det_top"].append(deterministic_top)
            result["det_max"].append(float(deterministic_top[0]))
            result["det_delta2"].append(
                float(np.square(deterministic_delta).mean())
            )
            result["det_power"].append(binner.power(deterministic[None])[0])
            result["det_2pcf"].append(estimator(deterministic_delta))
            result["det_hist"].append(
                np.histogram(log_deterministic, HISTOGRAM_EDGES)[0].astype(np.int64)
            )
            result["det_env"].append(
                [deterministic_env[name] for name in env_names]
            )
    return {key: np.asarray(value) for key, value in result.items()}


def _fit_indices(row: dict[str, Any]) -> np.ndarray:
    held = set(map(int, row["excluded_V70_mechanism_holdout_indices"]))
    indices = np.asarray(
        [index for index in range(int(row["total_objects"])) if index not in held],
        dtype=np.int64,
    )
    if len(indices) != int(row["audit_fit_objects"]):
        raise ValueError("V73 fit-object complement differs")
    return indices


def build_summary(
    program: dict[str, Any], output: Path, workers: int
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    binner = SpectralBinner(64, 0.3125)
    estimator = OpenBoundaryTwoPoint(64, 0.3125, 10.0)
    env_names = ALL_ENVIRONMENT_FIELDS
    arrays["fourier_k"] = binner.k
    arrays["fourier_mode_count"] = binner.count
    arrays["radius_mpc_h"] = estimator.radius_mpc_h
    arrays["environment_names"] = np.asarray(env_names)
    manifest_domains: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        row = program["already_consumed_train_inputs"][domain]
        data_path = Path(row["truth_data"])
        cache_path = Path(row["deterministic_cache"])
        if (
            sha256_file(data_path) != row["truth_data_sha256"]
            or sha256_file(cache_path) != row["deterministic_cache_sha256"]
        ):
            raise ValueError(f"V73 {domain} train input hash differs")
        fit = _fit_indices(row)
        with h5py.File(data_path, "r") as data, h5py.File(cache_path, "r") as cache:
            if (
                tuple(data["target"].shape) != (int(row["total_objects"]), 1, 64, 64, 64)
                or tuple(cache["conditional_mean"].shape) != tuple(data["target"].shape)
            ):
                raise ValueError(f"V73 {domain} aligned shape differs")
            if domain == "TNG100":
                groups_all = tng_group(data["center_position_mpc_h"][:])
            else:
                groups_all = np.asarray(data["realization"][:], dtype=np.int16)
        chunk_size = max(1, int(np.ceil(len(fit) / workers)))
        chunks = [fit[start : start + chunk_size].tolist() for start in range(0, len(fit), chunk_size)]
        rows: list[dict[str, np.ndarray]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _summarize_chunk,
                    domain,
                    str(data_path),
                    str(cache_path),
                    chunk,
                )
                for chunk in chunks
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(
                    f"[v73-summary] {domain} chunk {completed}/{len(futures)}",
                    flush=True,
                )
        combined = {
            key: np.concatenate([current[key] for current in rows], axis=0)
            for key in rows[0]
        }
        order = np.argsort(combined["index"])
        combined = {key: value[order] for key, value in combined.items()}
        if not np.array_equal(combined["index"], fit):
            raise ValueError(f"V73 {domain} summary index order differs")
        prefix = domain.lower()
        arrays[f"{prefix}_index"] = fit
        arrays[f"{prefix}_group"] = groups_all[fit]
        for key, value in combined.items():
            if key != "index":
                arrays[f"{prefix}_{key}"] = value
        unique, count = np.unique(groups_all[fit], return_counts=True)
        if np.min(count) < 2:
            raise ValueError(f"V73 {domain} group has fewer than two fit objects")
        manifest_domains[domain] = {
            "objects": int(len(fit)),
            "original_indices": fit.tolist(),
            "groups": {str(int(key)): int(value) for key, value in zip(unique, count)},
            "truth_data": str(data_path.resolve()),
            "truth_data_sha256": row["truth_data_sha256"],
            "deterministic_cache": str(cache_path.resolve()),
            "deterministic_cache_sha256": row["deterministic_cache_sha256"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, output)
    return {
        "schema": SUMMARY_SCHEMA,
        "program_sha256": PROGRAM_SHA256,
        "summary_cache": str(output.resolve()),
        "summary_cache_sha256": sha256_file(output),
        "domains": manifest_domains,
        "top_values_per_cube": TOP_VALUES,
        "voxels_per_cube": VOXELS_PER_CUBE,
        "environment_names": list(env_names),
        "validation_or_fresh_payload_accessed": False,
    }


def pooled_quantile_from_top(
    top: np.ndarray,
    selected: np.ndarray,
    quantile: float = 0.99999,
    voxels_per_cube: int = VOXELS_PER_CUBE,
) -> float:
    """Exact pooled upper quantile from sufficient per-cube upper order values."""
    values = np.asarray(top)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or not len(selected) or not 0.5 < quantile < 1.0:
        raise ValueError("invalid upper-quantile inputs")
    unique, counts = np.unique(selected, return_counts=True)
    total = int(len(selected) * voxels_per_cube)
    location = (total - 1) * quantile
    lower = int(np.floor(location))
    upper = int(np.ceil(location))
    descending_ranks = [total - 1 - lower, total - 1 - upper]
    heap: list[tuple[float, int, int, int]] = []
    for row, multiplicity in zip(unique.tolist(), counts.tolist()):
        heapq.heappush(heap, (-float(values[row, 0]), row, 0, multiplicity))
    found: dict[int, float] = {}
    consumed = 0
    while heap and len(found) < len(set(descending_ranks)):
        negative, row, position, multiplicity = heapq.heappop(heap)
        for rank in set(descending_ranks):
            if consumed <= rank < consumed + multiplicity:
                found[rank] = -negative
        consumed += multiplicity
        next_position = position + 1
        if next_position < values.shape[1]:
            heapq.heappush(
                heap,
                (-float(values[row, next_position]), row, next_position, multiplicity),
            )
    if any(rank not in found for rank in descending_ranks):
        raise ValueError("stored upper order values are insufficient")
    lower_value = found[descending_ranks[0]]
    upper_value = found[descending_ranks[1]]
    fraction = location - lower
    return float((1.0 - fraction) * lower_value + fraction * upper_value)


def ks_statistic_fast(first: np.ndarray, second: np.ndarray) -> float:
    first = np.sort(np.asarray(first, dtype=np.float64).reshape(-1))
    second = np.sort(np.asarray(second, dtype=np.float64).reshape(-1))
    if not len(first) or not len(second) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("invalid KS sample")
    support = np.sort(np.concatenate((first, second)))
    first_cdf = np.searchsorted(first, support, side="right") / len(first)
    second_cdf = np.searchsorted(second, support, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def ks_scale_means(
    truth: np.ndarray, prediction: np.ndarray, radius: np.ndarray
) -> np.ndarray:
    values = np.asarray(
        [ks_statistic_fast(truth[:, i], prediction[:, i]) for i in range(truth.shape[1])]
    )
    return np.asarray(
        [values[(radius >= low) & (radius < high)].mean() for low, high in SCALE_RANGES],
        dtype=np.float64,
    )


def band_values(
    numerator: np.ndarray,
    denominator: np.ndarray,
    k: np.ndarray,
    count: np.ndarray,
) -> np.ndarray:
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(np.asarray(numerator, dtype=np.float64), np.nan),
        where=np.asarray(denominator) > 0,
    )
    output = []
    for low, high in BANDS[-2:]:
        selected = (k >= low) & (k < high) & np.isfinite(ratio)
        output.append(float(np.average(ratio[selected], weights=count[selected])))
    return np.asarray(output)


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return float(0.5 * np.abs(first / first.sum() - second / second.sum()).sum())


def sample_queries(
    domain: str, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    selected: list[int] = []
    if domain == "TNG100":
        if len(unique) != 4:
            raise ValueError("TNG grouping differs")
        quota = {int(group): 4 for group in unique}
    elif domain == "SIMBA":
        if len(unique) != 8:
            raise ValueError("SIMBA grouping differs")
        quota = {int(group): 2 for group in unique}
    elif domain == "Swift":
        if len(unique) != 20:
            raise ValueError("Swift grouping differs")
        chosen = rng.choice(unique, size=16, replace=False)
        quota = {int(group): 1 for group in chosen}
    else:
        raise ValueError("unknown V73 domain")
    for group, number in quota.items():
        pool = np.flatnonzero(groups == group)
        selected.extend(rng.choice(pool, size=number, replace=False).tolist())
    result = np.asarray(selected, dtype=np.int64)
    if len(result) != 16 or len(np.unique(result)) != 16:
        raise ValueError("V73 query selection differs")
    return result


def sample_same_group_oracle(
    groups: np.ndarray,
    queries: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    excluded = set(map(int, queries))
    rows = []
    for query in queries:
        pool = np.asarray(
            [
                index
                for index in np.flatnonzero(groups == groups[query])
                if int(index) not in excluded
            ],
            dtype=np.int64,
        )
        if not len(pool):
            raise ValueError("V73 same-group donor pool is empty")
        rows.append(rng.choice(pool, size=16, replace=True))
    return np.asarray(rows, dtype=np.int64)


def sample_balanced_cross_oracle(
    groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    unique = np.unique(groups)
    chosen_groups = rng.choice(unique, size=16 * 16, replace=True)
    rows = np.empty(16 * 16, dtype=np.int64)
    for group in unique:
        selected = chosen_groups == group
        pool = np.flatnonzero(groups == group)
        rows[selected] = rng.choice(pool, size=int(selected.sum()), replace=True)
    return rows.reshape(16, 16)


def _domain_summary(cache: Any, domain: str) -> dict[str, np.ndarray]:
    prefix = domain.lower()
    names = (
        "index",
        "group",
        "truth_top",
        "truth_max",
        "truth_delta2",
        "truth_power",
        "truth_2pcf",
        "truth_hist",
        "truth_env",
        "residual_ms",
        "det_top",
        "det_max",
        "det_delta2",
        "det_power",
        "det_2pcf",
        "det_hist",
        "det_env",
    )
    return {name: np.asarray(cache[f"{prefix}_{name}"]) for name in names}


def trial_metrics(
    query: dict[str, np.ndarray],
    queries: np.ndarray,
    donor: dict[str, np.ndarray],
    donors: np.ndarray,
    k: np.ndarray,
    count: np.ndarray,
    radius: np.ndarray,
    energy_donors_B: np.ndarray | None = None,
    absolute_only: bool = False,
) -> dict[str, Any]:
    flat_donors = donors.reshape(-1)
    q_truth = pooled_quantile_from_top(query["truth_top"], queries)
    q_oracle = pooled_quantile_from_top(donor["truth_top"], flat_donors)
    q_delta = q_oracle - q_truth
    q4_ratio = float(
        donor["truth_delta2"][flat_donors].mean()
        / query["truth_delta2"][queries].mean()
    )
    power = band_values(
        donor["truth_power"][flat_donors].mean(axis=0),
        query["truth_power"][queries].mean(axis=0),
        k,
        count,
    )
    rms_ratio = float(
        np.sqrt(
            donor["residual_ms"][flat_donors].mean()
            / query["residual_ms"][queries].mean()
        )
    )
    checks = {
        "q99_999": abs(q_delta) <= 0.1,
        "Q4": (2.0 / 3.0) <= q4_ratio <= 1.5,
        "high_k_power": bool(np.all((power >= 0.9) & (power <= 1.1))),
        "residual_RMS": 0.9 <= rms_ratio <= 1.1,
    }
    absolute = all(checks[key] for key in ("q99_999", "Q4", "high_k_power", "residual_RMS"))
    result: dict[str, Any] = {
        **checks,
        "absolute_core": absolute,
        "q_delta": q_delta,
        "q4_ratio": q4_ratio,
        "power_3_6": float(power[0]),
        "power_6_10": float(power[1]),
        "rms_ratio": rms_ratio,
        "unique_donors": len(np.unique(flat_donors)),
    }
    if absolute_only:
        return result
    truth_hist = query["truth_hist"][queries].sum(axis=0)
    oracle_hist = donor["truth_hist"][flat_donors].sum(axis=0)
    deterministic_hist = query["det_hist"][queries].sum(axis=0)
    oracle_tv = total_variation(oracle_hist, truth_hist)
    deterministic_tv = total_variation(deterministic_hist, truth_hist)
    oracle_ks = ks_scale_means(
        query["truth_2pcf"][queries], donor["truth_2pcf"][flat_donors], radius
    )
    deterministic_ks = ks_scale_means(
        query["truth_2pcf"][queries], query["det_2pcf"][queries], radius
    )
    truth_env = query["truth_env"][queries].mean(axis=0)
    oracle_env = donor["truth_env"][flat_donors].mean(axis=0)
    deterministic_env = query["det_env"][queries].mean(axis=0)
    selected_env = [ALL_ENVIRONMENT_FIELDS.index(name) for name in ENVIRONMENT_FIELDS]
    environment_pass = bool(
        np.all(
            np.abs(oracle_env[selected_env] - truth_env[selected_env])
            < np.abs(deterministic_env[selected_env] - truth_env[selected_env])
        )
    )
    checks.update(
        {
            "density_PDF": oracle_tv < deterministic_tv,
            "two_point": bool(np.all(oracle_ks < deterministic_ks)),
            "environment": environment_pass,
        }
    )
    morphology = all(checks[key] for key in ("density_PDF", "two_point", "environment"))
    result.update(
        {
            "density_PDF": checks["density_PDF"],
            "two_point": checks["two_point"],
            "environment": checks["environment"],
            "morphology_core": morphology,
            "joint": absolute and morphology,
            "oracle_pdf_tv": oracle_tv,
            "deterministic_pdf_tv": deterministic_tv,
        }
    )
    for index, label in enumerate(("0_1", "1_3", "3_10")):
        result[f"oracle_ks_{label}"] = float(oracle_ks[index])
        result[f"deterministic_ks_{label}"] = float(deterministic_ks[index])
    if energy_donors_B is not None:
        truth_max = query["truth_max"][queries]
        energy_A = np.mean(
            [
                scalar_energy_score(donor["truth_max"][donors[index]], truth_max[index])
                for index in range(16)
            ]
        )
        energy_B = np.mean(
            [
                scalar_energy_score(
                    donor["truth_max"][energy_donors_B[index]], truth_max[index]
                )
                for index in range(16)
            ]
        )
        result["energy_A_better_B"] = energy_A < energy_B
        result["energy_delta_A_minus_B"] = float(energy_A - energy_B)
    return result


def rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def run_same_domain(
    summaries: dict[str, dict[str, np.ndarray]],
    k: np.ndarray,
    count: np.ndarray,
    radius: np.ndarray,
    trials: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    generator = np.random.default_rng(seed)
    output = {}
    for domain in DOMAIN_ORDER:
        rows = []
        summary = summaries[domain]
        for trial in range(trials):
            queries = sample_queries(domain, summary["group"], generator)
            donor_A = sample_same_group_oracle(summary["group"], queries, generator)
            donor_B = sample_same_group_oracle(summary["group"], queries, generator)
            rows.append(
                trial_metrics(
                    summary,
                    queries,
                    summary,
                    donor_A,
                    k,
                    count,
                    radius,
                    donor_B,
                )
            )
            if (trial + 1) % 1000 == 0:
                print(f"[v73-bootstrap] same {domain} {trial + 1}/{trials}", flush=True)
        output[domain] = rows_to_arrays(rows)
    return output


def run_cross_domain(
    summaries: dict[str, dict[str, np.ndarray]],
    k: np.ndarray,
    count: np.ndarray,
    radius: np.ndarray,
    trials: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    generator = np.random.default_rng(seed)
    output = {}
    for source in DOMAIN_ORDER:
        for target in DOMAIN_ORDER:
            if source == target:
                continue
            rows = []
            for trial in range(trials):
                queries = sample_queries(target, summaries[target]["group"], generator)
                donors = sample_balanced_cross_oracle(
                    summaries[source]["group"], generator
                )
                rows.append(
                    trial_metrics(
                        summaries[target],
                        queries,
                        summaries[source],
                        donors,
                        k,
                        count,
                        radius,
                        absolute_only=True,
                    )
                )
                if (trial + 1) % 1000 == 0:
                    print(
                        f"[v73-bootstrap] cross {source}->{target} {trial + 1}/{trials}",
                        flush=True,
                    )
            output[f"{source}_to_{target}"] = rows_to_arrays(rows)
    return output


def _absolute_band(power: np.ndarray, k: np.ndarray, count: np.ndarray) -> np.ndarray:
    output = []
    for low, high in BANDS[-2:]:
        selected = (k >= low) & (k < high) & np.isfinite(power)
        output.append(float(np.average(power[selected], weights=count[selected])))
    return np.asarray(output)


def tng_jackknife(
    summary: dict[str, np.ndarray], k: np.ndarray, count: np.ndarray, radius: np.ndarray
) -> dict[str, Any]:
    rows = {}
    groups = np.unique(summary["group"])
    selections = {"full": np.arange(len(summary["group"]))}
    selections.update(
        {f"leave_group_{int(group)}_out": np.flatnonzero(summary["group"] != group) for group in groups}
    )
    for label, selected in selections.items():
        q = pooled_quantile_from_top(summary["truth_top"], selected)
        delta2 = float(summary["truth_delta2"][selected].mean())
        power = _absolute_band(summary["truth_power"][selected].mean(axis=0), k, count)
        two_point = summary["truth_2pcf"][selected].mean(axis=0)
        scale_mean = [
            float(two_point[(radius >= low) & (radius < high)].mean())
            for low, high in SCALE_RANGES
        ]
        maxima = summary["truth_max"][selected]
        rows[label] = {
            "objects": int(len(selected)),
            "q99_999_log10rho": q,
            "mean_delta_squared": delta2,
            "power_3_6": float(power[0]),
            "power_6_10": float(power[1]),
            "two_point_scale_means": scale_mean,
            "mean_cube_maximum_log10rho": float(maxima.mean()),
            "q95_cube_maximum_log10rho": float(np.quantile(maxima, 0.95)),
        }
    leave = [value for key, value in rows.items() if key != "full"]
    q_range = max(row["q99_999_log10rho"] for row in leave) - min(
        row["q99_999_log10rho"] for row in leave
    )
    q4_ratio = max(row["mean_delta_squared"] for row in leave) / min(
        row["mean_delta_squared"] for row in leave
    )
    power_ratios = [
        max(row[key] for row in leave) / min(row[key] for row in leave)
        for key in ("power_3_6", "power_6_10")
    ]
    material = q_range > 0.1 or q4_ratio > 1.5 or max(power_ratios) > 1.1
    return {
        "claim_limit": "within-box spatial heterogeneity, not independent-box cosmic variance",
        "rows": rows,
        "leave_one_block_q99_999_range_dex": q_range,
        "leave_one_block_Q4_maximum_over_minimum": q4_ratio,
        "leave_one_block_power_maximum_over_minimum": {
            "3-6_h_mpc": power_ratios[0],
            "6-10_h_mpc": power_ratios[1],
        },
        "material": material,
    }


def _energy_per_query(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        sample = np.asarray(handle["sample"], dtype=np.float32)
        truth = np.asarray(handle["truth"], dtype=np.float32)
        source = np.asarray(handle["source_index"], dtype=np.int64)
    generated_max = 4.5 * sample.max(axis=(-3, -2, -1))[:, :, 0]
    truth_max = 4.5 * truth.max(axis=(-3, -2, -1))[:, 0]
    score = np.asarray(
        [scalar_energy_score(generated_max[index], truth_max[index]) for index in range(16)]
    )
    return score, truth_max, source


def v72_energy_stability(
    program: dict[str, Any], trials: int, seed: int
) -> dict[str, Any]:
    expected = program["consumed_V72_energy_stability"]["bound_artifact_hashes"]
    generator = np.random.default_rng(seed)
    result = {}
    for domain in DOMAIN_ORDER:
        leaf = V72_DOMAIN_KEYS[domain]
        candidate = V72_ROOT / V72_CANDIDATE / "fresh_candidate" / leaf / "ensemble16.h5"
        control = V72_ROOT / V72_CONTROL / "fresh_candidate" / leaf / "ensemble16.h5"
        if (
            sha256_file(candidate) != expected[f"candidate/{domain}"]
            or sha256_file(control) != expected[f"control/{domain}"]
        ):
            raise ValueError(f"V73 consumed V72 energy artifact differs: {domain}")
        candidate_score, candidate_truth, candidate_source = _energy_per_query(candidate)
        control_score, control_truth, control_source = _energy_per_query(control)
        if not np.array_equal(candidate_truth, control_truth) or not np.array_equal(
            candidate_source, control_source
        ):
            raise ValueError("V73 V72 paired energy provenance differs")
        delta = candidate_score - control_score
        index = generator.integers(0, 16, size=(trials, 16))
        sampled = delta[index].mean(axis=1)
        interval = np.quantile(sampled, [0.025, 0.975])
        result[domain] = {
            "per_query_candidate_minus_control": delta.tolist(),
            "mean_candidate_minus_control": float(delta.mean()),
            "paired_bootstrap_95": interval.tolist(),
            "bootstrap_probability_mean_below_zero": float(np.mean(sampled < 0.0)),
            "underpowered_interval_contains_zero": bool(interval[0] <= 0.0 <= interval[1]),
            "candidate_selected_by_point_estimate": bool(delta.mean() < 0.0),
        }
    return result


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.dtype == np.bool_:
        return {"pass_probability": float(values.mean()), "trials": int(len(values))}
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "quantiles_2p5_50_97p5": np.quantile(values, [0.025, 0.5, 0.975]).tolist(),
    }


def summarize_bootstrap(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: distribution_summary(value) for key, value in rows.items()}


def decide(
    same: dict[str, dict[str, np.ndarray]],
    cross: dict[str, dict[str, np.ndarray]],
    energy: dict[str, Any],
    jackknife: dict[str, Any],
) -> dict[str, Any]:
    domain_absolute = {
        domain: float(same[domain]["absolute_core"].mean()) for domain in DOMAIN_ORDER
    }
    domain_joint = {domain: float(same[domain]["joint"].mean()) for domain in DOMAIN_ORDER}
    all_domain_joint = float(
        np.mean(np.logical_and.reduce([same[domain]["joint"] for domain in DOMAIN_ORDER]))
    )
    gate_redesign = all_domain_joint < 0.8 or any(
        probability < 0.8 for probability in domain_absolute.values()
    )
    cross_absolute = {
        pair: float(values["absolute_core"].mean()) for pair, values in cross.items()
    }
    domain_stress = any(probability < 0.2 for probability in cross_absolute.values())
    energy_underpowered = {
        domain: bool(energy[domain]["underpowered_interval_contains_zero"])
        for domain in DOMAIN_ORDER
    }
    if gate_redesign:
        classification = "sampling_sensitive_V72_gate_requires_null_calibrated_redesign"
        next_step = "stop_new_Hong_candidates_and_seek_explicit_approval_for_gate_redesign"
    elif domain_stress or jackknife["material"]:
        classification = "gate_sampling_is_reliable_but_domain_or_TNG_heterogeneity_is_material"
        next_step = "run_target_free_identifiability_and_independent_box_design_audit"
    else:
        classification = "gate_sampling_is_reliable_and_no_material_population_stress_detected"
        next_step = "a_separately_frozen_candidate_may_be_considered_after_explicit_approval"
    return {
        "maximum_acceptable_false_rejection_probability": 0.2,
        "same_domain_absolute_core_pass_probability": domain_absolute,
        "same_domain_joint_pass_probability": domain_joint,
        "all_three_domain_sampling_sensitive_core_joint_pass_probability": all_domain_joint,
        "gate_redesign_required": gate_redesign,
        "cross_domain_absolute_core_pass_probability": cross_absolute,
        "domain_blind_common_law_materially_incompatible": domain_stress,
        "V72_energy_difference_underpowered": energy_underpowered,
        "TNG_spatial_heterogeneity_material": bool(jackknife["material"]),
        "classification": classification,
        "next": next_step,
    }


def run(program_path: Path, repo: Path, output_root: Path, workers: int) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V73 audit requires clean frozen Lageunha")
    expected_root = Path(program["outputs"]["root"]).resolve()
    if output_root.resolve() != expected_root:
        raise ValueError("V73 audit output root differs")
    result_path = Path(program["outputs"]["audit_result"])
    arrays_path = Path(program["outputs"]["bootstrap_arrays"])
    summary_path = Path(program["outputs"]["summary_cache"])
    summary_record_path = output_root / "train_truth_summary.json"
    if result_path.exists() or arrays_path.exists():
        raise FileExistsError("V73 audit refuses an existing result")
    output_root.mkdir(parents=True, exist_ok=True)
    if summary_path.exists() or summary_record_path.exists():
        if not summary_path.exists() or not summary_record_path.exists():
            raise ValueError("V73 partial summary cache differs")
        summary_record = strict_json(summary_record_path)
        if (
            summary_record.get("schema") != SUMMARY_SCHEMA
            or summary_record.get("program_sha256") != PROGRAM_SHA256
            or sha256_file(summary_path) != summary_record.get("summary_cache_sha256")
        ):
            raise ValueError("V73 existing summary cache provenance differs")
    else:
        summary_record = build_summary(program, summary_path, workers)
        partial_record = summary_record_path.with_suffix(".json.partial")
        partial_record.write_text(json.dumps(summary_record, indent=2) + "\n")
        os.replace(partial_record, summary_record_path)
    with np.load(summary_path, allow_pickle=False) as cache:
        summaries = {domain: _domain_summary(cache, domain) for domain in DOMAIN_ORDER}
        k = np.asarray(cache["fourier_k"], dtype=np.float64)
        count = np.asarray(cache["fourier_mode_count"], dtype=np.int64)
        radius = np.asarray(cache["radius_mpc_h"], dtype=np.float64)
    same_program = program["same_domain_truth_oracle_bootstrap"]
    same = run_same_domain(
        summaries,
        k,
        count,
        radius,
        int(same_program["trials"]),
        int(same_program["seed"]),
    )
    cross_program = program["cross_domain_common_law_stress"]
    cross = run_cross_domain(
        summaries,
        k,
        count,
        radius,
        int(cross_program["trials_per_ordered_source_target_pair"]),
        int(cross_program["seed"]),
    )
    jackknife = tng_jackknife(summaries["TNG100"], k, count, radius)
    energy_program = program["consumed_V72_energy_stability"]
    energy = v72_energy_stability(
        program,
        int(energy_program["bootstrap_trials"]),
        int(energy_program["seed"]),
    )
    arrays: dict[str, np.ndarray] = {}
    for domain, values in same.items():
        for key, value in values.items():
            arrays[f"same__{domain}__{key}"] = value
    for pair, values in cross.items():
        for key, value in values.items():
            arrays[f"cross__{pair}__{key}"] = value
    partial_arrays = arrays_path.with_suffix(arrays_path.suffix + ".partial")
    with partial_arrays.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial_arrays, arrays_path)
    decision = decide(same, cross, energy, jackknife)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_train_truth_gate_attainability_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "summary_cache": str(summary_path.resolve()),
        "summary_cache_sha256": sha256_file(summary_path),
        "summary_record": str(summary_record_path.resolve()),
        "summary_record_sha256": sha256_file(summary_record_path),
        "bootstrap_arrays": str(arrays_path.resolve()),
        "bootstrap_arrays_sha256": sha256_file(arrays_path),
        "same_domain_truth_oracle": {
            domain: summarize_bootstrap(same[domain]) for domain in DOMAIN_ORDER
        },
        "cross_domain_common_law_stress": {
            pair: summarize_bootstrap(values) for pair, values in cross.items()
        },
        "TNG_spatial_heterogeneity_jackknife": jackknife,
        "consumed_V72_energy_stability": energy,
        "decision": decision,
        "claim_limits": {
            "truth_oracle": "coarse group-conditional empirical population oracle, not an exact repeated draw at identical conditioning",
            "TNG": "within-box spatial heterogeneity, not independent-box cosmic variance",
            "cross_domain": "domain-blind stress, not a mathematical impossibility proof for conditional models",
            "excluded_gate_components": ["rank histogram", "voxel coverage", "exact DC", "model-specific energy ordering"],
        },
        "training_or_model_sampling_performed": False,
        "validation_or_fresh_payload_accessed": False,
        "V72_stage_B_accessed": False,
        "Astrid_accessed": False,
        "historical_or_independent_EAGLE_accessed": False,
        "V72_verdict_changed": False,
        "new_candidate_or_gate_change_authorized": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    partial_result = result_path.with_suffix(result_path.suffix + ".partial")
    partial_result.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial_result, result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("V73 workers must be in [1,16]")
    result = run(args.program, args.repo, args.output_root, args.workers)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
