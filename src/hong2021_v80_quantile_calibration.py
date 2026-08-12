#!/usr/bin/env python
"""Fit the frozen V80 monotone y-space calibration on consumed V72 data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = "hong2021-v80-consumed-development-quantile-calibration-training-program-v1"
PROGRAM_STATUS = "frozen_before_V80_calibration_implementation_or_consumed_V72_payload_reread"
PROGRAM_SHA256 = "5c4fec18d492c9871ab67815e72ffd5f135393944f33f0939ec28c999c6a346c"
PROGRAM_FREEZE_COMMIT = "cf368e13269f892385ef730d2d4c741c4bca410d"
REPORT_SCHEMA = "hong2021-v80-consumed-development-quantile-calibration-report-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
MINIMUM_Y = -2.0
MAXIMUM_Y = 2.0
BINS = 65_536
QUERIES = 16
MEMBERS = 16
FIELD_SHAPE = (1, 64, 64, 64)
SAMPLE_SHAPE = (QUERIES, MEMBERS, *FIELD_SHAPE)
TRUTH_SHAPE = (QUERIES, *FIELD_SHAPE)


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    path = path.resolve()
    repo = repo.resolve()
    if sha256_file(path) != PROGRAM_SHA256:
        raise ValueError("V80 calibration program hash differs")
    program = strict_json(path)
    algorithm = program.get("frozen_calibration_algorithm", {})
    authorization = program.get("authorization", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or float(algorithm.get("histogram_minimum_y", np.nan)) != MINIMUM_Y
        or float(algorithm.get("histogram_maximum_y", np.nan)) != MAXIMUM_Y
        or int(algorithm.get("histogram_bins", -1)) != BINS
        or authorization.get(
            "implement_and_run_exactly_one_consumed_development_calibration_fit"
        )
        is not True
        or authorization.get("read_any_V79_selected_input_or_target_during_calibration")
        is not False
        or authorization.get("candidate_design_change_after_training_diagnostics")
        is not False
    ):
        raise ValueError("V80 calibration schema, algorithm, or authorization differs")
    parent = program["parent_evidence"]
    for key in ("V79_program", "V72_result_record"):
        bound = resolve_path(repo, parent[key])
        if sha256_file(bound) != parent[f"{key}_sha256"]:
            raise ValueError(f"V80 calibration parent differs: {key}")
    v72 = strict_json(resolve_path(repo, parent["V72_result_record"]))
    if (
        v72.get("status") != parent["required_V72_status"]
        or v72.get("scientific_conclusion", {}).get("no_posthoc_override_or_V72_retry")
        is not True
        or v72.get("firewall", {}).get("stage_B_accessed") is not False
    ):
        raise ValueError("V80 consumed V72 boundary differs")
    for domain in DOMAIN_ORDER:
        row = program["consumed_training_inputs"][domain]
        for arm in ("candidate", "control"):
            if sha256_file(Path(row[arm]).resolve()) != row[f"{arm}_sha256"]:
                raise ValueError(f"V80 consumed {domain}/{arm} differs")
    return program


def histogram_edges() -> np.ndarray:
    return np.linspace(MINIMUM_Y, MAXIMUM_Y, BINS + 1, dtype=np.float64)


def add_histogram(counts: np.ndarray, values: np.ndarray, edges: np.ndarray) -> None:
    current = np.asarray(values, dtype=np.float64)
    if (
        not np.isfinite(current).all()
        or float(current.min()) < MINIMUM_Y
        or float(current.max()) > MAXIMUM_Y
    ):
        raise ValueError("V80 calibration value is nonfinite or outside frozen range")
    counts += np.histogram(current, bins=edges)[0].astype(np.int64)


def stream_histogram(dataset: h5py.Dataset, edges: np.ndarray) -> np.ndarray:
    counts = np.zeros(BINS, dtype=np.int64)
    if dataset.shape[0] != QUERIES:
        raise ValueError("V80 calibration query count differs")
    for query in range(QUERIES):
        add_histogram(counts, dataset[query], edges)
    if int(counts.sum()) != int(dataset.size):
        raise ValueError("V80 calibration histogram did not conserve voxels")
    return counts


def fit_monotone_map(
    source_counts: np.ndarray, truth_counts: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_counts = np.asarray(source_counts, dtype=np.int64)
    truth_counts = np.asarray(truth_counts, dtype=np.int64)
    if (
        source_counts.shape != (BINS,)
        or truth_counts.shape != (BINS,)
        or np.any(source_counts < 0)
        or np.any(truth_counts < 0)
        or source_counts.sum() <= 0
        or truth_counts.sum() <= 0
    ):
        raise ValueError("V80 calibration histogram differs")
    centers = 0.5 * (edges[:-1] + edges[1:])
    source_used = source_counts > 0
    truth_used = truth_counts > 0
    source_cumulative = np.cumsum(source_counts, dtype=np.float64)
    truth_cumulative = np.cumsum(truth_counts, dtype=np.float64)
    source_probability = (
        source_cumulative[source_used] - 0.5 * source_counts[source_used]
    ) / source_counts.sum()
    truth_probability = (
        truth_cumulative[truth_used] - 0.5 * truth_counts[truth_used]
    ) / truth_counts.sum()
    source_knots = centers[source_used]
    mapped_knots = np.interp(
        source_probability,
        truth_probability,
        centers[truth_used],
    )
    if (
        len(source_knots) < 2
        or not np.isfinite(mapped_knots).all()
        or np.any(np.diff(source_knots) <= 0)
        or np.any(np.diff(mapped_knots) < 0)
    ):
        raise ValueError("V80 calibration map is not finite monotone")
    return source_knots, mapped_knots


def apply_monotone_map(
    values: np.ndarray, source_knots: np.ndarray, mapped_knots: np.ndarray
) -> np.ndarray:
    current = np.asarray(values, dtype=np.float64)
    if not np.isfinite(current).all():
        raise ValueError("V80 map input is nonfinite")
    mapped = np.interp(current, source_knots, mapped_knots)
    below = current < source_knots[0]
    above = current > source_knots[-1]
    mapped[below] = mapped_knots[0] + current[below] - source_knots[0]
    mapped[above] = mapped_knots[-1] + current[above] - source_knots[-1]
    if not np.isfinite(mapped).all():
        raise ValueError("V80 map output is nonfinite")
    return mapped


def map_and_project(
    sample: np.ndarray,
    conditional_mean: np.ndarray,
    source_knots: np.ndarray,
    mapped_knots: np.ndarray,
) -> tuple[np.ndarray, float]:
    current = np.asarray(sample, dtype=np.float64)
    mean = np.asarray(conditional_mean, dtype=np.float64)
    mapped = apply_monotone_map(current, source_knots, mapped_knots)
    residual = mapped - mean
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True, dtype=np.float64)
    output = (mean + residual).astype(np.float32)
    actual = output.astype(np.float64) - mean
    maximum_dc = float(
        np.max(np.abs(actual.mean(axis=(-3, -2, -1), dtype=np.float64)))
    )
    if not np.isfinite(output).all():
        raise ValueError("V80 projected output is nonfinite")
    return output, maximum_dc


def histogram_quantile(counts: np.ndarray, edges: np.ndarray, probability: float) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    if not 0.0 < probability < 1.0 or counts.sum() <= 0:
        raise ValueError("V80 histogram quantile input differs")
    target = probability * (counts.sum() - 1)
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, target + 1, side="left"))
    before = int(cumulative[index - 1]) if index else 0
    fraction = (target - before) / max(1, int(counts[index]) - 1)
    return float(edges[index] + np.clip(fraction, 0.0, 1.0) * (edges[index + 1] - edges[index]))


def metric_row(
    counts: np.ndarray,
    truth_counts: np.ndarray,
    delta_squared_sum: float,
    truth_delta_squared_sum: float,
    voxels: int,
    truth_voxels: int,
    edges: np.ndarray,
) -> dict[str, float]:
    first = counts / counts.sum()
    second = truth_counts / truth_counts.sum()
    q = histogram_quantile(counts, edges, 0.99999)
    truth_q = histogram_quantile(truth_counts, edges, 0.99999)
    return {
        "histogram_total_variation_to_truth": float(0.5 * np.abs(first - second).sum()),
        "q99_999_y": q,
        "q99_999_log10rho": 4.5 * q,
        "q99_999_error_dex": 4.5 * (q - truth_q),
        "mean_delta_squared": float(delta_squared_sum / voxels),
        "mean_delta_squared_ratio_to_truth": float(
            (delta_squared_sum / voxels)
            / (truth_delta_squared_sum / truth_voxels)
        ),
    }


def _delta_squared_sum(values: np.ndarray) -> float:
    log10rho = 4.5 * np.asarray(values, dtype=np.float64)
    delta = np.power(10.0, log10rho) - 1.0
    if not np.isfinite(delta).all():
        raise ValueError("V80 physical density diagnostic is nonfinite")
    return float(np.square(delta).sum(dtype=np.float64))


def fit_domain(candidate_path: Path, control_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    edges = histogram_edges()
    with h5py.File(candidate_path, "r") as candidate, h5py.File(control_path, "r") as control:
        if (
            tuple(candidate["sample"].shape) != SAMPLE_SHAPE
            or tuple(candidate["truth"].shape) != TRUTH_SHAPE
            or tuple(candidate["conditional_mean"].shape) != TRUTH_SHAPE
            or tuple(control["sample"].shape) != SAMPLE_SHAPE
            or not np.array_equal(candidate["truth"][:], control["truth"][:])
            or not np.array_equal(
                candidate["conditional_mean"][:], control["conditional_mean"][:]
            )
            or not np.array_equal(candidate["source_index"][:], control["source_index"][:])
        ):
            raise ValueError("V80 consumed candidate/control contract differs")
        source_counts = stream_histogram(candidate["sample"], edges)
        truth_counts = stream_histogram(candidate["truth"], edges)
        control_counts = stream_histogram(control["sample"], edges)
        source_knots, mapped_knots = fit_monotone_map(source_counts, truth_counts, edges)
        after = {
            "candidate": np.zeros(BINS, dtype=np.int64),
            "control": np.zeros(BINS, dtype=np.int64),
        }
        sums_before = {"candidate": 0.0, "control": 0.0}
        sums_after = {"candidate": 0.0, "control": 0.0}
        truth_sum = 0.0
        maximum_dc = {"candidate": 0.0, "control": 0.0}
        for query in range(QUERIES):
            mean = np.asarray(candidate["conditional_mean"][query], dtype=np.float64)
            truth = np.asarray(candidate["truth"][query], dtype=np.float64)
            truth_sum += _delta_squared_sum(truth)
            for arm, handle in (("candidate", candidate), ("control", control)):
                raw = np.asarray(handle["sample"][query], dtype=np.float64)
                sums_before[arm] += _delta_squared_sum(raw)
                calibrated, dc = map_and_project(raw, mean, source_knots, mapped_knots)
                maximum_dc[arm] = max(maximum_dc[arm], dc)
                add_histogram(after[arm], calibrated, edges)
                sums_after[arm] += _delta_squared_sum(calibrated)
        truth_voxels = int(candidate["truth"].size)
        sample_voxels = int(candidate["sample"].size)
    before_counts = {"candidate": source_counts, "control": control_counts}
    diagnostics = {
        arm: {
            "before": metric_row(
                before_counts[arm], truth_counts, sums_before[arm], truth_sum,
                sample_voxels, truth_voxels, edges,
            ),
            "after": metric_row(
                after[arm], truth_counts, sums_after[arm], truth_sum,
                sample_voxels, truth_voxels, edges,
            ),
            "maximum_absolute_residual_DC_after": maximum_dc[arm],
        }
        for arm in ("candidate", "control")
    }
    diagnostics["map"] = {
        "knots": int(len(source_knots)),
        "source_support_y": [float(source_knots[0]), float(source_knots[-1])],
        "mapped_support_y": [float(mapped_knots[0]), float(mapped_knots[-1])],
        "minimum_mapped_increment": float(np.diff(mapped_knots).min()),
        "monotone": bool(np.all(np.diff(mapped_knots) >= 0)),
        "outside_support_slope": 1.0,
    }
    return {
        "source_knots_y": source_knots,
        "mapped_knots_y": mapped_knots,
    }, diagnostics


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)


def run(program_path: Path, repo: Path, output_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit):
        raise RuntimeError("V80 calibration requires clean frozen ancestry")
    outputs = {key: Path(program["outputs"][key]).resolve() for key in ("root", "calibration", "report")}
    if output_root.resolve() != outputs["root"] or any(path.exists() for path in outputs.values()):
        raise FileExistsError("V80 calibration refuses an existing or differing output")
    arrays: dict[str, np.ndarray] = {
        "histogram_edges_y": histogram_edges(),
        "program_sha256": np.asarray(PROGRAM_SHA256),
    }
    diagnostics = {}
    for domain in DOMAIN_ORDER:
        row = program["consumed_training_inputs"][domain]
        fitted, diagnostics[domain] = fit_domain(Path(row["candidate"]), Path(row["control"]))
        for key, value in fitted.items():
            arrays[f"{domain}__{key}"] = value
        print(f"[v80-calibration] {domain} complete", flush=True)
    output_root.mkdir(parents=True)
    write_npz_atomic(outputs["calibration"], arrays)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_single_consumed_development_quantile_calibration_fit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "training_code_commit": commit,
        "worktree_clean": clean,
        "calibration": str(outputs["calibration"]),
        "calibration_sha256": sha256_file(outputs["calibration"]),
        "diagnostics": diagnostics,
        "diagnostics_have_no_selection_or_refit_role": True,
        "fits_per_domain": 1,
        "second_fit_or_changed_algorithm": False,
        "V79_selected_input_or_target_accessed": False,
        "V72_verdict_changed": False,
        "V72_stage_B_accessed": False,
        "Astrid_or_EAGLE_accessed": False,
        "candidate_sampling_performed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    partial = outputs["report"].with_suffix(".json.partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, outputs["report"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.program, args.repo, args.output_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
