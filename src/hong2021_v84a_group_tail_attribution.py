#!/usr/bin/env python
"""V84A group-leakage, direct-PIT, tail, and SQT attribution audit."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v63_train import _is_ancestor
from hong2021_v83_contract import DOMAIN_ORDER, partition_indices
from hong2021_v83_network import conditional_cdf
from hong2021_v83_train import CHECKPOINT_SCHEMA, seeded_model


PROGRAM_SCHEMA = "hong2021-v84a-group-tail-SQT-attribution-program-v1"
PROGRAM_STATUS = "frozen_before_train_or_consumed_payload_audit"
REPORT_SCHEMA = "hong2021-v84a-group-tail-SQT-attribution-report-v1"
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
CONDITION_NAMES = (
    "log1p_count",
    "mean_radial_velocity",
    "radial_velocity_dispersion",
    "backbone",
    "radius",
    "block_risk",
    "object_amplitude",
)
PIT_BINS = 100
TAIL_PROBABILITIES = (0.01, 0.001, 0.0001)
PROBE_VOXELS = 4096
PROBE_SEED = 840084
TAIL_RATIO_INTERVAL = (0.8, 1.25)
MAXIMUM_CANDIDATE_CONTROL_RANK_TV_DIFFERENCE_FOR_MARGINAL_ATTRIBUTION = 0.005


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def program_freeze_commit(program_path: Path, repo: Path) -> str:
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(program_path.resolve())],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("V84A program freeze commit cannot be resolved")
    return commit


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    program = strict_json(path.resolve())
    authorization = program.get("authorization", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("engineering_only") is not True
        or program.get("statistically_independent") is not False
        or authorization.get("user_approved_V84A_audit") is not True
        or authorization.get("training_or_refit") is not False
        or authorization.get("resampling") is not False
        or authorization.get("independent_validation") is not False
    ):
        raise ValueError("V84A scope boundary differs")
    for label, row in program["implementation_sources"].items():
        if sha256_file(resolve_path(repo, row["path"])) != row["sha256"]:
            raise ValueError(f"V84A implementation differs: {label}")
    for label, row in program["frozen_artifacts"].items():
        path_value = resolve_path(repo, row["path"])
        if sha256_file(path_value) != row["sha256"]:
            raise ValueError(f"V84A frozen artifact differs: {label}")
    if set(program["domains"]) != set(DOMAIN_ORDER):
        raise ValueError("V84A domain set differs")
    for domain in DOMAIN_ORDER:
        row = program["domains"][domain]
        for split in ("train", "validation"):
            objects = int(row[f"{split}_objects"])
            if objects <= 0:
                raise ValueError(f"V84A {domain} {split} object count differs")
        selection = list(map(int, row["consumed_selection"]))
        if (
            len(selection) != 32
            or len(set(selection)) != 32
            or min(selection) < 0
            or max(selection) >= int(row["validation_objects"])
        ):
            raise ValueError(f"V84A {domain} consumed selection differs")
        for key in (
            "train_data",
            "train_cache",
            "validation_data",
            "validation_cache",
            "candidate_ensemble",
            "control_ensemble",
            "candidate_metrics",
            "control_metrics",
        ):
            artifact = Path(row[key]).resolve()
            if sha256_file(artifact) != row[f"{key}_sha256"]:
                raise ValueError(f"V84A {domain} {key} differs")
    return program


def periodic_nearest_distance(
    query: np.ndarray,
    reference: np.ndarray,
    box_mpc_h: float,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (
        query.ndim != 2
        or reference.ndim != 2
        or query.shape[1:] != (3,)
        or reference.shape[1:] != (3,)
    ):
        raise ValueError("V84A position shape differs")
    if len(reference) == 0:
        return np.full(len(query), np.nan)
    delta = np.abs(query[:, None] - reference[None])
    delta = np.minimum(delta, box_mpc_h - delta)
    return np.sqrt(np.square(delta).sum(axis=-1)).min(axis=1)


def _distance_summary(value: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"available": False}
    return {
        "available": True,
        "minimum_mpc_h": float(finite.min()),
        "median_mpc_h": float(np.median(finite)),
        "maximum_mpc_h": float(finite.max()),
        "fraction_below_10_mpc_h": float(np.mean(finite < 10.0)),
    }


def group_leakage(
    domain: str,
    train: h5py.File,
    validation: h5py.File,
    fit_indices: list[int],
    holdout_indices: list[int],
    validation_indices: list[int],
) -> dict[str, Any]:
    fit = np.asarray(fit_indices, dtype=np.int64)
    holdout = np.asarray(holdout_indices, dtype=np.int64)
    consumed = np.asarray(validation_indices, dtype=np.int64)
    box = 75.0 if domain == "TNG100" else 25.0
    train_positions = np.asarray(train["center_position_mpc_h"], dtype=np.float64)
    validation_positions = np.asarray(validation["center_position_mpc_h"], dtype=np.float64)
    if "realization" in train:
        train_groups = np.asarray(train["realization"], dtype=np.int64)
        validation_groups = np.asarray(validation["realization"], dtype=np.int64)
    else:
        train_groups = np.zeros(len(train_positions), dtype=np.int64)
        validation_groups = np.zeros(len(validation_positions), dtype=np.int64)
    fit_groups = set(map(int, train_groups[fit]))
    holdout_groups = set(map(int, train_groups[holdout]))
    consumed_groups = set(map(int, validation_groups[consumed]))

    def within_group_distance(
        positions: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        output = np.full(len(positions), np.nan)
        for index, (position, group) in enumerate(zip(positions, groups, strict=True)):
            selected = fit[train_groups[fit] == group]
            if len(selected):
                output[index] = periodic_nearest_distance(
                    position[None], train_positions[selected], box
                )[0]
        return output

    holdout_distance = within_group_distance(
        train_positions[holdout], train_groups[holdout]
    )
    consumed_distance = within_group_distance(
        validation_positions[consumed], validation_groups[consumed]
    )
    return {
        "group_field": "realization" if "realization" in train else "single_TNG100_realization",
        "simulation_box_mpc_h": box,
        "fit_groups": sorted(fit_groups),
        "holdout_groups": sorted(holdout_groups),
        "consumed_validation_groups": sorted(consumed_groups),
        "fit_holdout_group_intersection": sorted(fit_groups & holdout_groups),
        "fit_consumed_group_intersection": sorted(fit_groups & consumed_groups),
        "holdout_groups_also_in_fit_fraction": (
            len(fit_groups & holdout_groups) / len(holdout_groups)
        ),
        "consumed_groups_also_in_fit_fraction": (
            len(fit_groups & consumed_groups) / len(consumed_groups)
        ),
        "holdout_nearest_fit_center_within_same_group": _distance_summary(holdout_distance),
        "consumed_nearest_fit_center_within_same_group": _distance_summary(consumed_distance),
        "random_object_holdout_is_group_independent": not bool(fit_groups & holdout_groups),
    }


class PopulationAccumulator:
    def __init__(self, probe_positions: np.ndarray, direct_pit: bool) -> None:
        self.probe_positions = probe_positions
        self.direct_pit = direct_pit
        self.condition_rows: list[np.ndarray] = []
        self.target_rows: list[np.ndarray] = []
        self.histogram = np.zeros(PIT_BINS, dtype=np.int64)
        self.voxels = 0
        self.pit_sum = 0.0
        self.coverage = {"50": 0, "80": 0, "95": 0}
        self.tail_counts = {
            f"{probability:g}": {"lower": 0, "upper": 0}
            for probability in TAIL_PROBABILITIES
        }
        self.quartile_sum = np.zeros(4, dtype=np.float64)
        self.quartile_count = np.zeros(4, dtype=np.int64)
        self.target_minimum = np.inf
        self.target_maximum = -np.inf

    def add_probe(self, condition: np.ndarray, target: np.ndarray) -> None:
        self.condition_rows.append(
            condition.reshape(len(CONDITION_NAMES), -1)[:, self.probe_positions].T
        )
        self.target_rows.append(target.reshape(-1)[self.probe_positions])

    def add_pit(self, uniform: np.ndarray, condition: np.ndarray, target: np.ndarray) -> None:
        values = np.asarray(uniform, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("V84A PIT value differs")
        self.histogram += np.histogram(values, bins=PIT_BINS, range=(0.0, 1.0))[0]
        self.voxels += len(values)
        self.pit_sum += float(values.sum())
        self.coverage["50"] += int(np.count_nonzero(np.abs(values - 0.5) <= 0.25))
        self.coverage["80"] += int(np.count_nonzero(np.abs(values - 0.5) <= 0.40))
        self.coverage["95"] += int(np.count_nonzero(np.abs(values - 0.5) <= 0.475))
        for probability in TAIL_PROBABILITIES:
            row = self.tail_counts[f"{probability:g}"]
            row["lower"] += int(np.count_nonzero(values < probability))
            row["upper"] += int(np.count_nonzero(values > 1.0 - probability))
        score = condition[3].reshape(-1)
        for quartile, positions in enumerate(
            np.array_split(np.argsort(score, kind="stable"), 4)
        ):
            self.quartile_sum[quartile] += float(values[positions].sum())
            self.quartile_count[quartile] += len(positions)
        self.target_minimum = min(self.target_minimum, float(target.min()))
        self.target_maximum = max(self.target_maximum, float(target.max()))

    def summary(self) -> dict[str, Any]:
        condition = np.concatenate(self.condition_rows, axis=0).astype(np.float64)
        target = np.concatenate(self.target_rows).astype(np.float64)
        row: dict[str, Any] = {
            "objects": len(self.condition_rows),
            "probe_voxels_per_object": len(self.probe_positions),
            "condition": {
                name: {
                    "mean": float(condition[:, channel].mean()),
                    "standard_deviation": float(condition[:, channel].std()),
                    "q01_q50_q99": np.quantile(
                        condition[:, channel], [0.01, 0.5, 0.99]
                    ).tolist(),
                }
                for channel, name in enumerate(CONDITION_NAMES)
            },
            "standardized_target_probe": {
                "mean": float(target.mean()),
                "standard_deviation": float(target.std()),
                "q0001_q001_q01_q50_q99_q999_q9999": np.quantile(
                    target, [0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]
                ).tolist(),
            },
        }
        if self.direct_pit:
            probability = self.histogram.astype(np.float64) / self.voxels
            row["direct_PIT"] = {
                "voxels": self.voxels,
                "mean": self.pit_sum / self.voxels,
                "histogram_bins": PIT_BINS,
                "histogram_probabilities": probability.tolist(),
                "total_variation_from_uniform": float(
                    0.5 * np.abs(probability - 1.0 / PIT_BINS).sum()
                ),
                "central_coverage": {
                    key: value / self.voxels for key, value in self.coverage.items()
                },
                "backbone_quartile_means": (
                    self.quartile_sum / self.quartile_count
                ).tolist(),
                "tail_exceedance": {
                    key: {
                        "expected_probability_each_side": float(key),
                        "lower_probability": value["lower"] / self.voxels,
                        "upper_probability": value["upper"] / self.voxels,
                        "lower_over_expected": value["lower"] / self.voxels / float(key),
                        "upper_over_expected": value["upper"] / self.voxels / float(key),
                    }
                    for key, value in self.tail_counts.items()
                },
                "standardized_target_minimum": self.target_minimum,
                "standardized_target_maximum": self.target_maximum,
            }
        return row


@torch.inference_mode()
def inspect_population(
    model: torch.nn.Module | None,
    data: h5py.File,
    cache: h5py.File,
    prepared: h5py.File,
    domain: str,
    split: str,
    indices: list[int],
    probe_positions: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    accumulator = PopulationAccumulator(probe_positions, model is not None)
    for position, index in enumerate(indices):
        condition, target, _ = condition_cube(
            data, cache, prepared, domain, split, int(index)
        )
        accumulator.add_probe(condition, target)
        if model is not None:
            condition_tensor = torch.from_numpy(condition[None]).to(device)
            target_tensor = torch.from_numpy(target[None]).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                parameters = model(condition_tensor)
            uniform = conditional_cdf(parameters.float(), target_tensor)
            accumulator.add_pit(uniform.cpu().numpy(), condition, target)
        if (position + 1) % 16 == 0 or position + 1 == len(indices):
            print(
                f"[v84a-population] {domain}/{split} {position + 1}/{len(indices)}",
                flush=True,
            )
    return accumulator.summary()


def _metric_summary(path: Path) -> dict[str, Any]:
    metrics = next(iter(strict_json(path)["candidates"].values()))
    calibration = metrics["residual_calibration"]
    power = metrics["fourier_log_density"]
    two_point = metrics["two_point_cosmic_mean"]["generated_vs_truth_ks"]["by_scale"]
    return {
        "rank_TV": calibration["rank_histogram_total_variation_from_uniform"],
        "coverage68": calibration["voxel_coverage_68"],
        "coverage95": calibration["voxel_coverage_95"],
        "residual_RMS_ratio": calibration["generated_over_truth_rms"],
        "residual_power_ratio": power["generated_residual_power_over_truth_residual"],
        "density_PDF_TV": metrics["density_pdf"][
            "log10_density_total_variation_generated_truth"
        ],
        "two_point_KS_by_scale": {
            key: value["mean"] for key, value in two_point.items()
        },
    }


def ensemble_attribution(program: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        rows: dict[str, Any] = {}
        for arm in ("candidate", "control"):
            ensemble = Path(program["domains"][domain][f"{arm}_ensemble"])
            metrics = Path(program["domains"][domain][f"{arm}_metrics"])
            rows[arm] = {
                "marginal": marginal_diagnostics(ensemble),
                "metrics": _metric_summary(metrics),
            }
        candidate_rank = float(rows["candidate"]["metrics"]["rank_TV"])
        control_rank = float(rows["control"]["metrics"]["rank_TV"])
        rows["candidate_control_attribution"] = {
            "absolute_rank_TV_difference": abs(candidate_rank - control_rank),
            "rank_failure_attributable_to_marginal_if_control_also_miscalibrated": (
                abs(candidate_rank - control_rank)
                <= MAXIMUM_CANDIDATE_CONTROL_RANK_TV_DIFFERENCE_FOR_MARGINAL_ATTRIBUTION
            ),
            "candidate_over_control_mean_delta_squared": (
                rows["candidate"]["marginal"]["generated_mean_delta_squared"]
                / rows["control"]["marginal"]["generated_mean_delta_squared"]
            ),
            "candidate_minus_control_q99_999_log10rho_dex": (
                rows["candidate"]["marginal"]["generated_q99_999_log10rho"]
                - rows["control"]["marginal"]["generated_q99_999_log10rho"]
            ),
        }
        output[domain] = rows
        print(f"[v84a-ensemble] {domain} complete", flush=True)
    return output


def direct_tail_failure(population: dict[str, Any]) -> bool:
    tail = population["direct_PIT"]["tail_exceedance"]
    return any(
        not (
            TAIL_RATIO_INTERVAL[0]
            <= tail[f"{probability:g}"][side + "_over_expected"]
            <= TAIL_RATIO_INTERVAL[1]
        )
        for probability in (0.001, 0.0001)
        for side in ("lower", "upper")
    )


def audit(program_path: Path, repo: Path, output_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program_path = program_path.resolve()
    program = load_program(program_path, repo)
    commit, clean = git_state(repo)
    freeze_commit = program_freeze_commit(program_path, repo)
    if (
        not clean
        or not _is_ancestor(repo, freeze_commit, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or not torch.cuda.is_available()
        or "ada" not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError("V84A requires clean frozen Lageunha Ada")
    if output_path.resolve() != Path(program["output"]).resolve() or output_path.exists():
        raise FileExistsError("V84A output exists or differs")
    checkpoint_path = Path(program["frozen_artifacts"]["v83_checkpoint"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256")
        != program["frozen_artifacts"]["v83_program"]["sha256"]
        or checkpoint.get("validation_payload_accessed") is not False
        or checkpoint.get("independent_gate_locked") is not True
    ):
        raise ValueError("V84A frozen V83 checkpoint differs")
    train_gate = strict_json(Path(program["frozen_artifacts"]["v83_train_gate"]["path"]))
    if train_gate.get("train_holdout_mechanism_pass") is not True:
        raise ValueError("V84A V83 train gate did not pass")
    device = torch.device("cuda")
    model = seeded_model(device)
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()
    cache_path = Path(program["frozen_artifacts"]["conditioning_cache"]["path"])
    cache_sha = program["frozen_artifacts"]["conditioning_cache"]["sha256"]
    prepared = load_cache(cache_path, cache_sha, commit)
    counts = {
        domain: int(program["domains"][domain]["train_objects"])
        for domain in DOMAIN_ORDER
    }
    partition = partition_indices(counts)
    generator = np.random.default_rng(PROBE_SEED)
    probe_positions = np.sort(
        generator.choice(64**3, size=PROBE_VOXELS, replace=False)
    )
    grouping: dict[str, Any] = {}
    populations: dict[str, Any] = {}
    try:
        for domain in DOMAIN_ORDER:
            definition = program["domains"][domain]
            train_data, train_cache = _open_split(definition, "train")
            validation_data, validation_cache = _open_split(definition, "validation")
            try:
                grouping[domain] = group_leakage(
                    domain,
                    train_data,
                    validation_data,
                    partition[domain]["fit"],
                    partition[domain]["holdout"],
                    definition["consumed_selection"],
                )
                populations[domain] = {
                    "fit_probe": inspect_population(
                        None,
                        train_data,
                        train_cache,
                        prepared,
                        domain,
                        "train",
                        partition[domain]["fit"],
                        probe_positions,
                        device,
                    ),
                    "random_object_holdout": inspect_population(
                        model,
                        train_data,
                        train_cache,
                        prepared,
                        domain,
                        "train",
                        partition[domain]["holdout"],
                        probe_positions,
                        device,
                    ),
                    "consumed_validation_selection": inspect_population(
                        model,
                        validation_data,
                        validation_cache,
                        prepared,
                        domain,
                        "validation",
                        list(map(int, definition["consumed_selection"])),
                        probe_positions,
                        device,
                    ),
                }
            finally:
                train_data.close()
                train_cache.close()
                validation_data.close()
                validation_cache.close()
    finally:
        prepared.close()
    ensembles = ensemble_attribution(program)
    group_leakage_detected = any(
        not row["random_object_holdout_is_group_independent"]
        for row in grouping.values()
    )
    consumed_direct_tail_failure = {
        domain: direct_tail_failure(
            populations[domain]["consumed_validation_selection"]
        )
        for domain in DOMAIN_ORDER
    }
    rank_marginal_attribution = all(
        ensembles[domain]["candidate_control_attribution"][
            "rank_failure_attributable_to_marginal_if_control_also_miscalibrated"
        ]
        for domain in DOMAIN_ORDER
    )
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_V84A_group_tail_SQT_attribution",
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "program_freeze_commit": freeze_commit,
        "code_commit": commit,
        "worktree_clean": clean,
        "group_leakage": grouping,
        "populations": populations,
        "ensemble_candidate_control": ensembles,
        "decision": {
            "random_object_holdout_group_leakage_detected": group_leakage_detected,
            "consumed_direct_tail_failure_by_domain": consumed_direct_tail_failure,
            "rank_failure_primary_attribution_is_conditional_marginal_or_generalization": rank_marginal_attribution,
            "V70_V72_spatial_copula_should_be_retained": rank_marginal_attribution,
            "V83_train_holdout_gate_was_optimistically_leaked": group_leakage_detected,
            "next": "design_V84B_group_held_out_spliced_conditional_tail_model" if group_leakage_detected and any(consumed_direct_tail_failure.values()) and rank_marginal_attribution else "audit_unresolved_mechanism_before_any_V84B_training",
        },
        "thresholds": {
            "direct_tail_observed_over_expected_interval": list(TAIL_RATIO_INTERVAL),
            "direct_tail_probabilities_used_for_decision": [0.001, 0.0001],
            "maximum_candidate_control_rank_TV_difference_for_marginal_attribution": MAXIMUM_CANDIDATE_CONTROL_RANK_TV_DIFFERENCE_FOR_MARGINAL_ATTRIBUTION,
        },
        "training_or_refit_performed": False,
        "sampling_or_resampling_performed": False,
        "only_train_and_already_consumed_validation_payload_accessed": True,
        "independent_validation_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.program, args.repo, args.out)


if __name__ == "__main__":
    main()
