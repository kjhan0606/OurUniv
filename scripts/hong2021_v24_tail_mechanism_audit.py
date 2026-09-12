#!/usr/bin/env python
"""Trace V24 high-density failures through the frozen physical/latent maps.

This is a read-only development-domain audit.  It opens only the frozen V24
TNG100, SIMBA, and Swift development ensembles and their already-attested
validation caches.  Astrid and historical EAGLE are intentionally absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from hong2021_v14_multiscale import fourier_band_masks
from hong2021_v18_init import sha256_file
from hong2021_v20_gaussianize import exact_zero_dc_projection
from hong2021_v21_conditional_affine import apply_profile


SCHEMA = "hong2021-v24-high-density-tail-mechanism-audit-v1"
TARGET_DENSITY_SCALE = 4.5
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
GATE_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
V24_DECISION = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v24_e12_base48/development_decision.json"
)
V24_FAILURE = V24_DECISION.parent / "automatic_failure_audit.json"
V21_ARTIFACTS = Path("config/hong2021_v21_derived_artifacts.json")
V20_REGISTRY = Path("config/hong2021_v20_development_program.json")


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_number(value: Any) -> int | float:
    item = value.item() if isinstance(value, np.generic) else value
    return int(item) if isinstance(item, (int, np.integer)) else float(item)


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    levels = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
    result = np.quantile(array, levels, method="linear")
    return {
        name: float(value)
        for name, value in zip(
            ("minimum", "median", "p90", "p95", "p99", "maximum"),
            result,
            strict=True,
        )
    }


def _correlation(first: list[float], second: list[float]) -> float | None:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def recover_v14_standardized(
    sample: np.ndarray,
    conditional_mean_with_location: np.ndarray,
    predicted_scales: np.ndarray,
    *,
    voxel_mpc_h: float,
) -> np.ndarray:
    """Undo the V14 band scales from a stored physical ensemble member.

    The stored conditional mean already includes the predicted scalar residual
    location.  Subtracting it leaves exactly the four scaled non-DC bands.
    """
    sample = np.asarray(sample, dtype=np.float32)
    mean = np.asarray(conditional_mean_with_location, dtype=np.float32)
    scales = np.asarray(predicted_scales, dtype=np.float64)
    if sample.shape != mean.shape or sample.ndim != 3:
        raise ValueError("sample and conditional mean must be same-shape 3-D cubes")
    masks = fourier_band_masks(sample.shape[0], float(voxel_mpc_h))
    if scales.shape != (len(masks),) or np.any(scales <= 0.0):
        raise ValueError("one positive predicted scale is required per Fourier band")
    mode_scale = np.ones(sample.shape, dtype=np.float64)
    for mask, scale in zip(masks, scales, strict=True):
        mode_scale[mask] = scale
    spectrum = np.fft.fftn(sample.astype(np.float64) - mean.astype(np.float64))
    standardized = np.fft.ifftn(spectrum / mode_scale).real
    standardized -= standardized.mean(dtype=np.float64)
    return standardized.astype(np.float32)


def forward_latent_diagnostics(
    standardized: np.ndarray,
    corrected_mean: np.ndarray,
    profile: Mapping[str, Any],
    transform: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Forward-map one V14 residual and expose G21 support occupancy."""
    u = apply_profile(standardized, corrected_mean, profile)
    residual_knots = np.asarray(transform["residual_value_knots"], dtype=np.float64)
    z_knots = np.asarray(transform["z_knots"], dtype=np.float64)
    raw_latent = np.interp(
        u.astype(np.float64), residual_knots, z_knots
    ).astype(np.float32)
    latent_dc = float(raw_latent.mean(dtype=np.float64))
    centered = (raw_latent - latent_dc).astype(np.float32)
    centered, projection = exact_zero_dc_projection(centered)
    low, high = float(residual_knots[0]), float(residual_knots[-1])
    total = int(u.size)
    support = {
        "voxels": total,
        "u_minimum": float(np.min(u)),
        "u_maximum": float(np.max(u)),
        "raw_latent_minimum": float(np.min(raw_latent)),
        "raw_latent_maximum": float(np.max(raw_latent)),
        "centered_latent_minimum": float(np.min(centered)),
        "centered_latent_maximum": float(np.max(centered)),
        "low_at_or_outside_count": int(np.count_nonzero(u <= low)),
        "high_at_or_outside_count": int(np.count_nonzero(u >= high)),
        "high_within_0p01_count": int(np.count_nonzero(u >= high - 0.01)),
        "high_within_0p05_count": int(np.count_nonzero(u >= high - 0.05)),
        "high_within_0p10_count": int(np.count_nonzero(u >= high - 0.10)),
        "projection_final_ortho_dc": float(projection["final_ortho_dc"]),
    }
    for name in (
        "low_at_or_outside_count",
        "high_at_or_outside_count",
        "high_within_0p01_count",
        "high_within_0p05_count",
        "high_within_0p10_count",
    ):
        support[name.replace("_count", "_fraction")] = support[name] / total
    return u, centered, support


def _maximum_cell(
    physical: np.ndarray,
    stored_mean: np.ndarray,
    corrected_mean: np.ndarray,
    standardized: np.ndarray,
    u: np.ndarray,
    centered_latent: np.ndarray,
) -> dict[str, Any]:
    flat_index = int(np.argmax(physical))
    coordinate = tuple(int(value) for value in np.unravel_index(flat_index, physical.shape))
    return {
        "coordinate": list(coordinate),
        "target_y": float(physical[coordinate]),
        "physical_log10rho": float(TARGET_DENSITY_SCALE * physical[coordinate]),
        "conditional_mean_with_location_y": float(stored_mean[coordinate]),
        "corrected_conditional_mean_y": float(corrected_mean[coordinate]),
        "target_y_minus_conditional_mean_y": float(
            physical[coordinate] - stored_mean[coordinate]
        ),
        "recovered_v14_standardized": float(standardized[coordinate]),
        "affine_normalized_u": float(u[coordinate]),
        "forward_centered_latent": float(centered_latent[coordinate]),
    }


def _aggregate_fields(
    rows: list[dict[str, Any]], *, truth_global_maximum: float
) -> dict[str, Any]:
    maxima = [float(row["physical_maximum"]) for row in rows]
    threshold = truth_global_maximum + 0.3
    high = [row for row in rows if float(row["physical_maximum"]) > threshold]
    ordinary = [row for row in rows if float(row["physical_maximum"]) <= threshold]

    def group(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not group_rows:
            return {"fields": 0}
        return {
            "fields": len(group_rows),
            "mean_physical_maximum": float(np.mean([
                row["physical_maximum"] for row in group_rows
            ])),
            "mean_high_at_or_outside_fraction": float(np.mean([
                row["support"]["high_at_or_outside_fraction"] for row in group_rows
            ])),
            "mean_high_within_0p10_fraction": float(np.mean([
                row["support"]["high_within_0p10_fraction"] for row in group_rows
            ])),
            "mean_raw_latent_maximum": float(np.mean([
                row["support"]["raw_latent_maximum"] for row in group_rows
            ])),
            "mean_forward_centered_latent_maximum": float(np.mean([
                row["support"]["centered_latent_maximum"] for row in group_rows
            ])),
        }

    support_high = [
        float(row["support"]["high_at_or_outside_fraction"]) for row in rows
    ]
    support_near = [
        float(row["support"]["high_within_0p10_fraction"]) for row in rows
    ]
    raw_latent_max = [float(row["support"]["raw_latent_maximum"]) for row in rows]
    centered_latent_max = [
        float(row["support"]["centered_latent_maximum"]) for row in rows
    ]
    mean_at_max = [
        float(row["maximum_cell"]["conditional_mean_with_location_y"]) for row in rows
    ]
    result = {
        "fields": len(rows),
        "physical_maximum_quantiles": _quantiles(maxima),
        "truth_global_maximum": truth_global_maximum,
        "fields_above_truth_global_plus_dex": {
            str(delta): int(np.count_nonzero(np.asarray(maxima) > truth_global_maximum + delta))
            for delta in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        },
        "physical_voxels_above_truth_global_maximum": int(sum(
            row["physical_voxels_above_truth_global_maximum"] for row in rows
        )),
        "physical_voxels_above_truth_global_plus_0p3": int(sum(
            row["physical_voxels_above_truth_global_plus_0p3"] for row in rows
        )),
        "support_counts": {
            name: int(sum(row["support"][name] for row in rows))
            for name in (
                "voxels", "low_at_or_outside_count", "high_at_or_outside_count",
                "high_within_0p01_count", "high_within_0p05_count",
                "high_within_0p10_count",
            )
        },
        "correlations_with_physical_maximum": {
            "high_at_or_outside_fraction": _correlation(maxima, support_high),
            "high_within_0p10_fraction": _correlation(maxima, support_near),
            "raw_latent_maximum": _correlation(maxima, raw_latent_max),
            "forward_centered_latent_maximum": _correlation(
                maxima, centered_latent_max
            ),
            "conditional_mean_at_maximum_cell": _correlation(maxima, mean_at_max),
            **{
                f"predicted_band_scale_{index}": _correlation(
                    maxima, [float(row["predicted_band_scales"][index]) for row in rows]
                )
                for index in range(4)
            },
        },
        "truth_plus_0p3_groups": {
            "above": group(high),
            "at_or_below": group(ordinary),
        },
    }
    total = result["support_counts"]["voxels"]
    result["support_fractions"] = {
        name.replace("_count", "_fraction"): count / total
        for name, count in result["support_counts"].items()
        if name.endswith("_count")
    }
    return result


def audit_domain(
    domain: str,
    ensemble_path: Path,
    source_cache_path: Path,
    v21_cache_path: Path,
    data_path: Path,
    profile: Mapping[str, Any],
    transform: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    integrity = {
        "maximum_absolute_stored_mean_minus_source_mean_plus_location": 0.0,
        "maximum_absolute_stored_truth_minus_source_target": 0.0,
        "maximum_absolute_recomputed_truth_latent_minus_v21_cache": 0.0,
        "exact_recomputed_truth_latent_fields": 0,
    }
    with h5py.File(ensemble_path, "r") as ensemble, h5py.File(
        source_cache_path, "r"
    ) as source, h5py.File(v21_cache_path, "r") as v21, h5py.File(
        data_path, "r"
    ) as data:
        if not bool(ensemble.attrs.get("complete", False)):
            raise ValueError(f"incomplete V24 ensemble: {ensemble_path}")
        if int(ensemble.attrs.get("checkpoint_step", -1)) != 30000:
            raise ValueError("tail audit requires the frozen V24 step-30000 ensemble")
        voxel_mpc_h = float(source.attrs["voxel_mpc_h"])
        indices = [int(value) for value in ensemble["source_index"][:]]
        if len(indices) != 16 or len(set(indices)) != 16:
            raise ValueError("V24 development ensemble requires 16 unique source indices")
        truth_global_maximum = float(
            TARGET_DENSITY_SCALE * np.max(ensemble["truth"][:])
        )
        for object_index, source_index in enumerate(indices):
            location = float(ensemble["predicted_residual_dc"][object_index])
            scales = np.asarray(
                ensemble["predicted_band_scales"][object_index], dtype=np.float64
            )
            stored_mean = np.asarray(
                ensemble["conditional_mean"][object_index, 0], dtype=np.float32
            )
            corrected_mean = (stored_mean - np.float32(location)).astype(np.float32)
            source_mean = np.asarray(
                source["conditional_mean"][source_index, 0], dtype=np.float32
            )
            integrity[
                "maximum_absolute_stored_mean_minus_source_mean_plus_location"
            ] = max(
                integrity[
                    "maximum_absolute_stored_mean_minus_source_mean_plus_location"
                ],
                float(np.max(np.abs(stored_mean - (source_mean + location)))),
            )
            truth = np.asarray(ensemble["truth"][object_index, 0], dtype=np.float32)
            source_truth = np.asarray(data["target"][source_index, 0], dtype=np.float32)
            integrity["maximum_absolute_stored_truth_minus_source_target"] = max(
                integrity["maximum_absolute_stored_truth_minus_source_target"],
                float(np.max(np.abs(truth - source_truth))),
            )

            truth_standardized = np.asarray(
                source["standardized_residual"][source_index, 0], dtype=np.float32
            )
            truth_u, truth_latent, truth_support = forward_latent_diagnostics(
                truth_standardized, source_mean, profile, transform
            )
            cached_truth_latent = np.asarray(
                v21["standardized_residual"][source_index, 0], dtype=np.float32
            )
            difference = float(np.max(np.abs(truth_latent - cached_truth_latent)))
            integrity[
                "maximum_absolute_recomputed_truth_latent_minus_v21_cache"
            ] = max(
                integrity[
                    "maximum_absolute_recomputed_truth_latent_minus_v21_cache"
                ],
                difference,
            )
            integrity["exact_recomputed_truth_latent_fields"] += int(
                np.array_equal(truth_latent, cached_truth_latent)
            )
            truth_rows.append({
                "object_index": object_index,
                "source_index": source_index,
                "physical_maximum": float(TARGET_DENSITY_SCALE * np.max(truth)),
                "support": truth_support,
                "maximum_cell": _maximum_cell(
                    truth, source_mean + np.float32(location), source_mean,
                    truth_standardized, truth_u, truth_latent,
                ),
            })

            for member in range(ensemble["sample"].shape[1]):
                physical = np.asarray(
                    ensemble["sample"][object_index, member, 0], dtype=np.float32
                )
                standardized = recover_v14_standardized(
                    physical, stored_mean, scales, voxel_mpc_h=voxel_mpc_h
                )
                u, latent, support = forward_latent_diagnostics(
                    standardized, corrected_mean, profile, transform
                )
                rows.append({
                    "object_index": object_index,
                    "source_index": source_index,
                    "member": member,
                    "physical_maximum": float(
                        TARGET_DENSITY_SCALE * np.max(physical)
                    ),
                    "corresponding_truth_maximum": float(
                        TARGET_DENSITY_SCALE * np.max(truth)
                    ),
                    "maximum_minus_corresponding_truth_maximum": float(
                        TARGET_DENSITY_SCALE * (np.max(physical) - np.max(truth))
                    ),
                    "predicted_residual_dc": location,
                    "predicted_band_scales": scales.tolist(),
                    "physical_voxels_above_truth_global_maximum": int(
                        np.count_nonzero(
                            TARGET_DENSITY_SCALE * physical > truth_global_maximum
                        )
                    ),
                    "physical_voxels_above_truth_global_plus_0p3": int(
                        np.count_nonzero(
                            TARGET_DENSITY_SCALE * physical
                            > truth_global_maximum + 0.3
                        )
                    ),
                    "support": support,
                    "maximum_cell": _maximum_cell(
                        physical, stored_mean, corrected_mean, standardized, u, latent
                    ),
                })

    by_object = []
    for object_index in range(16):
        selected = [row for row in rows if row["object_index"] == object_index]
        truth_row = truth_rows[object_index]
        by_object.append({
            "object_index": object_index,
            "source_index": truth_row["source_index"],
            "truth_maximum": truth_row["physical_maximum"],
            "generated_maximum_quantiles": _quantiles([
                row["physical_maximum"] for row in selected
            ]),
            "members_above_corresponding_truth_plus_0p3": int(sum(
                row["physical_maximum"] > truth_row["physical_maximum"] + 0.3
                for row in selected
            )),
            "members_above_domain_truth_global_plus_0p3": int(sum(
                row["physical_maximum"] > truth_global_maximum + 0.3
                for row in selected
            )),
        })
    return {
        "ensemble": str(ensemble_path.resolve()),
        "ensemble_sha256": sha256_file(ensemble_path),
        "source_indices": [row["source_index"] for row in truth_rows],
        "voxel_mpc_h": voxel_mpc_h,
        "integrity": integrity,
        "truth": {
            "physical_global_maximum": truth_global_maximum,
            "physical_maximum_quantiles": _quantiles([
                row["physical_maximum"] for row in truth_rows
            ]),
            "support_counts": {
                name: int(sum(row["support"][name] for row in truth_rows))
                for name in (
                    "voxels", "low_at_or_outside_count", "high_at_or_outside_count",
                    "high_within_0p01_count", "high_within_0p05_count",
                    "high_within_0p10_count",
                )
            },
            "fields": truth_rows,
        },
        "generated": _aggregate_fields(
            rows, truth_global_maximum=truth_global_maximum
        ),
        "objects": by_object,
        "top_generated_fields": sorted(
            rows, key=lambda row: row["physical_maximum"], reverse=True
        )[:32],
        "all_generated_field_rows": rows,
    }


def classify(domains: Mapping[str, Any]) -> dict[str, Any]:
    high_fields = {
        domain: row["generated"]["fields_above_truth_global_plus_dex"]["0.3"]
        for domain, row in domains.items()
    }
    generated_support = {
        domain: row["generated"]["support_fractions"][
            "high_at_or_outside_fraction"
        ]
        for domain, row in domains.items()
    }
    truth_support = {}
    for domain, row in domains.items():
        counts = row["truth"]["support_counts"]
        truth_support[domain] = counts["high_at_or_outside_count"] / counts["voxels"]
    broad = high_fields["TNG100"] > 1 and high_fields["SIMBA"] > 1
    endpoint_inflation = any(
        generated_support[domain] > max(5.0 * truth_support[domain], 1.0e-6)
        for domain in DOMAIN_ORDER
    )
    if broad and endpoint_inflation:
        mechanism = "multi_field_tail_overdispersion_with_forward_support_saturation"
        next_step = "audit_exact_terminal_sampler_latents_before_selecting_v25_tail_control"
    elif broad:
        mechanism = "multi_field_tail_overdispersion_without_dominant_forward_support_saturation"
        next_step = "audit_terminal_sampler_trajectory_and_design_sampler_aligned_v25"
    else:
        mechanism = "isolated_extreme_field_failure"
        next_step = "inspect_ranked_field_provenance_before_v25"
    return {
        "mechanism": mechanism,
        "fields_above_domain_truth_global_plus_0p3": high_fields,
        "generated_high_support_fraction": generated_support,
        "truth_high_support_fraction": truth_support,
        "broad_multi_field_failure": broad,
        "forward_support_saturation_inflated": endpoint_inflation,
        "next": next_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decision", type=Path, default=V24_DECISION)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.out.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise RuntimeError(f"refusing to overwrite V24 tail audit: {output}")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("V24 tail audit requires a clean committed worktree")

    decision_path = args.decision.resolve()
    decision = json.loads(decision_path.read_text())
    failure = json.loads(V24_FAILURE.read_text())
    if decision.get("development_pass") is not False or int(
        decision["candidates"][-1]["step"]
    ) != 30000:
        raise ValueError("V24 tail audit requires the failed final frozen decision")
    if failure.get("Astrid_accessed") is not False or failure.get(
        "historical_EAGLE_accessed"
    ) is not False:
        raise ValueError("V24 failure audit violated the data firewall")

    artifacts_path = (repo / V21_ARTIFACTS).resolve()
    registry_path = (repo / V20_REGISTRY).resolve()
    artifacts = json.loads(artifacts_path.read_text())
    v20 = json.loads(registry_path.read_text())["e8_gaussianized_marginal_retrain"]
    profile_path = Path(artifacts["profile"]["path"])
    transform_path = Path(artifacts["gaussianization"]["path"])
    if sha256_file(profile_path) != artifacts["profile"]["sha256"]:
        raise ValueError("V21 profile hash mismatch")
    if sha256_file(transform_path) != artifacts["gaussianization"]["sha256"]:
        raise ValueError("V21 transform hash mismatch")
    profile = json.loads(profile_path.read_text())
    transform = json.loads(transform_path.read_text())

    domains = {}
    final = decision["candidates"][-1]["domains"]
    for domain in DOMAIN_ORDER:
        data_spec = v20["data"][domain]
        source_spec = data_spec["source_validation_cache"]
        data_validation = data_spec["validation_data"]
        v21_spec = artifacts["caches"][f"{domain}_validation"]
        for specification in (source_spec, data_validation, v21_spec):
            if sha256_file(Path(specification["path"])) != specification["sha256"]:
                raise ValueError(f"frozen {domain} audit input hash mismatch")
        ensemble_path = Path(final[GATE_KEYS[domain]]["ensemble"])
        domains[domain] = audit_domain(
            domain,
            ensemble_path,
            Path(source_spec["path"]),
            Path(v21_spec["path"]),
            Path(data_validation["path"]),
            profile,
            transform,
        )

    report = {
        "schema": SCHEMA,
        "code_commit": _git(repo, "rev-parse", "HEAD"),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "v24_decision": str(decision_path),
            "v24_decision_sha256": sha256_file(decision_path),
            "v24_failure_audit": str(V24_FAILURE),
            "v24_failure_audit_sha256": sha256_file(V24_FAILURE),
            "v21_artifacts": str(artifacts_path),
            "v21_artifacts_sha256": sha256_file(artifacts_path),
            "v20_registry": str(registry_path),
            "v20_registry_sha256": sha256_file(registry_path),
            "profile": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "transform": str(transform_path),
            "transform_sha256": sha256_file(transform_path),
        },
        "support": {
            "target_density_mapping": "log10rho=4.5*y",
            "z": [float(transform["z_knots"][0]), float(transform["z_knots"][-1])],
            "u": [
                float(transform["residual_value_knots"][0]),
                float(transform["residual_value_knots"][-1]),
            ],
        },
        "domains": domains,
        "classification": classify(domains),
        "interpretation_limits": {
            "physical_ensemble_is_forward_mapped_back_to_the_canonical_latent": True,
            "raw_preinverse_sampler_terminal_latent_is_not_stored": True,
            "exact_terminal_latent_requires_draw_paired_sampler_replay": True,
        },
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["audit_digest_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps({
        "out": str(output),
        "classification": report["classification"],
        "audit_digest_sha256": report["audit_digest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
