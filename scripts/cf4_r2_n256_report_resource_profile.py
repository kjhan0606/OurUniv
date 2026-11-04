#!/usr/bin/env python3
"""Build synthetic native-shape inputs and validate an R2 report profile.

The fixture values are deterministic sufficient-statistic moments, not CF4
data or posterior draws. Run ``build`` and ``finalize`` in separate processes
around the existing four-chain report CLI to keep their RSS lifetimes separate.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_field_reporting_regions import training_count_cell_region
from cf4_r2_retained_field_mcse import RetainedFieldMCSEDiagnostics


STATISTICS_NAME = "field_mcse_sufficient_statistics_unassessed.npz"
PROFILE_SCHEMA = "cf4-r2-n256-report-resource-profile-v1"
_MAP_MEMBERS = (
    "density_contrast_mean", "density_contrast_posterior_sd",
    "velocity_conditional_mean", "velocity_conditional_posterior_sd",
    "velocity_variance_defined", "velocity_occupancy_fraction",
    "velocity_occupancy_posterior_sd")


def _training_keys(path):
    with np.load(path, allow_pickle=False) as split:
        return np.asarray(split["train_keys"], dtype=np.int64)


def _spatial_pattern(size):
    coordinate = (np.arange(size, dtype=np.float64) - size / 2) / size
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    return (np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y)
        + np.sin(2 * np.pi * z)) / 3.


def _populate_synthetic_sufficient_statistics(diagnostics, spatial, *,
        chain_index, draws):
    batch_sizes = diagnostics.batch_sizes
    quantum = min(batch_sizes)
    if any(size % quantum for size in batch_sizes):
        raise ValueError("profile batch sizes must be integer multiples of the smaller scale")
    if draws % quantum or (draws // quantum) % 2:
        raise ValueError("profile needs an even number of complete smallest-scale batches")

    delta_count = max(1, quantum // 16)
    base_count = np.rint((0.82 + 0.08 * spatial) * quantum).astype(np.int64)
    base_count = np.clip(base_count, 1 + delta_count, quantum - delta_count)
    density_base = 1. + 0.03 * spatial + 0.01 * (chain_index - 1.5)
    density_delta = 0.02 + 0.005 * spatial
    density_within_variance = 0.04 + 0.005 * (spatial + 1.)
    velocity_base = np.stack([
        0.10 * (axis + 1) * spatial + 0.015 * (chain_index - 1.5)
        for axis in range(3)])
    velocity_delta = np.asarray([0.02, 0.03, 0.04])[:, None, None, None]
    velocity_within_variance = np.stack([
        0.80 + 0.05 * (axis + 1) + 0.05 * (spatial + 1.)
        for axis in range(3)])

    diagnostics.accepted_transitions = draws // 2
    diagnostics.rejected_transitions = draws // 2
    for batch_size, accumulator in diagnostics.accumulator.accumulators.items():
        batches = draws // batch_size
        if batches < 2 or batches % 2:
            raise ValueError("both profile scales need an even batch count")
        accumulator.states = draws
        accumulator.pending_states = 0
        accumulator.batches = batches
        for name in accumulator._PENDING_ARRAYS + accumulator._SUMMARY_ARRAYS:
            getattr(accumulator, name).fill(0.)

        for index in range(batches):
            sign = 1. if index % 2 == 0 else -1.
            occupancy_count = base_count + int(sign * delta_count)
            occupancy = occupancy_count.astype(np.float64) / quantum
            density_mean_state = density_base + sign * density_delta
            density_first = occupancy * density_mean_state
            density_second = occupancy * (
                density_mean_state * density_mean_state
                + density_within_variance)
            velocity_mean_state = velocity_base + sign * velocity_delta
            velocity_second_state = (velocity_mean_state * velocity_mean_state
                + velocity_within_variance)
            velocity_first = occupancy[None] * velocity_mean_state
            velocity_second = occupancy[None] * velocity_second_state

            accumulator.sum_rho += density_first
            accumulator.sum_rho2 += density_second
            accumulator.sum_rho_sq += density_first * density_first
            accumulator.sum_rho2_sq += density_second * density_second
            accumulator.sum_rho_rho2 += density_first * density_second
            accumulator.sum_occupied += occupancy
            accumulator.sum_occupied_sq += occupancy * occupancy
            accumulator.sum_v += velocity_first
            accumulator.sum_v2 += velocity_second
            accumulator.sum_v_sq += velocity_first * velocity_first
            accumulator.sum_v2_sq += velocity_second * velocity_second
            accumulator.sum_v_v2 += velocity_first * velocity_second
            accumulator.sum_occupied_v += occupancy[None] * velocity_first
            accumulator.sum_occupied_v2 += occupancy[None] * velocity_second


def build_fixtures(job_root, training_split, *, grid_size=256, draws=512,
        batch_sizes=(32, 64), minimum_batches=8, count_grid_size=128,
        population_count=6):
    job_root = Path(job_root).resolve(strict=True)
    fixture_root = job_root / "fixtures"
    fixture_root.mkdir(exist_ok=False)
    commit = os.environ.get("CF4_EXPECTED_COMMIT", "")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("CF4_EXPECTED_COMMIT must be a full lowercase commit hash")
    if grid_size % count_grid_size:
        raise ValueError("native grid must be an integer refinement of the count grid")
    if draws % min(batch_sizes) or any(draws % size for size in batch_sizes):
        raise ValueError("draw count must end on complete batches at both scales")
    if any(draws // size < minimum_batches for size in batch_sizes):
        raise ValueError("draw count is below the declared minimum batch count")

    keys = _training_keys(training_split)
    spatial = _spatial_pattern(grid_size)
    map_name = f"field_mcse_unassessed_batch_{min(batch_sizes)}.npz"
    unused_map_name = "unused_second_scale_map_resource_profile_only.txt"
    reports = []
    for chain_index in range(4):
        chain_dir = fixture_root / f"chain_{chain_index}"
        chain_dir.mkdir()
        diagnostics = RetainedFieldMCSEDiagnostics(
            (grid_size,) * 3, train_keys=keys, batch_sizes=batch_sizes,
            minimum_batches=minimum_batches,
            count_grid_size=count_grid_size, population_count=population_count)
        _populate_synthetic_sufficient_statistics(diagnostics, spatial,
            chain_index=chain_index, draws=draws)

        statistics_path = chain_dir / STATISTICS_NAME
        np.savez(statistics_path, **diagnostics.checkpoint_arrays())

        smallest = min(batch_sizes)
        maps = diagnostics.finalize_scale(smallest)
        if maps["status"] != "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE":
            raise RuntimeError("synthetic profile fixture did not finalize its first scale")
        selected = {name: maps[name] for name in _MAP_MEMBERS}
        np.savez(chain_dir / map_name, **selected)
        (chain_dir / unused_map_name).write_text(
            "Synthetic profile placeholder. This second-scale map is not read.\n",
            encoding="utf-8")
        diagnostics_summary = diagnostics.summary()
        diagnostics_summary.update(
            sufficient_statistics_file=STATISTICS_NAME,
            sufficient_statistics_raw_array_bytes_estimate=
                diagnostics.estimated_accumulator_bytes,
            sufficient_statistics_file_status=
                "UNASSESSED_CHAIN_STATISTICS_FOR_POSTPROCESSING",
            scales={str(size): dict(
                status="FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE",
                map_file=map_name if size == smallest else unused_map_name,
                regional_summary={}) for size in batch_sizes})
        del selected, maps, diagnostics
        gc.collect()

        streams = {name: dict(stream_id=stream_id,
            spawn_key=[chain_index, stream_id])
            for name, stream_id in (("prior", 0), ("audit", 1),
                ("transition", 2))}
        report = dict(status="N256_CHAIN_FINISHED_REQUIRES_DIAGNOSTICS",
            R2_complete=False, heldout_scored=False,
            initialization_policy="independent_prior", long_chain=True,
            synthetic_fixture=True, source_commit=commit,
            target="SYNTHETIC_RESOURCE_PROFILE_ONLY_NOT_POSTERIOR",
            N=grid_size, box_cMpc_h=384.,
            observation_target_revision="synthetic-resource-profile-only",
            source_conditioning_rule="deterministic non-data moments",
            field_mcse_contract=dict(batch_sizes=list(batch_sizes),
                minimum_batches=minimum_batches),
            chain_id=f"synthetic-resource-profile-chain-{chain_index}",
            chain_rng_design=dict(chain_index=chain_index,
                master_seed=20261105, chain_count=4,
                bit_generator="PCG64", streams=streams,
                synthetic_fixture=True),
            field_mcse=diagnostics_summary)
        report_path = chain_dir / "result.json"
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, allow_nan=False)
            handle.write("\n")
        reports.append(str(report_path))

    fixture_files = [dict(path=str(path.relative_to(fixture_root)),
        size_bytes=path.stat().st_size)
        for path in sorted(fixture_root.rglob("*")) if path.is_file()]
    manifest = dict(schema=PROFILE_SCHEMA, synthetic_fixture=True,
        job_root=str(job_root), fixture_dir="fixtures",
        fixture_files=fixture_files, grid_size=grid_size, draws=draws,
        batch_sizes=list(batch_sizes), minimum_batches=minimum_batches,
        chain_count=4, source_commit=commit,
        report_inputs=[str(path) for path in reports])
    with (fixture_root / "SYNTHETIC_PROFILE_MANIFEST.json").open(
            "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return reports


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _assert_finite_json(value):
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_json(child)
    elif isinstance(value, float):
        _require(math.isfinite(value), "report contains a nonfinite JSON number")


def validate_synthetic_report(report, *, grid_size, draws, batch_sizes,
        training_keys, count_grid_size=128, population_count=6):
    _require(report.get("status") ==
        "FOUR_CHAIN_FIELD_MCSE_DIAGNOSTIC_NOT_CONVERGENCE_EVIDENCE",
        "unexpected four-chain report status")
    _require(report.get("R2_complete") is False
        and report.get("heldout_scored") is False
        and report.get("posterior_promoted") is False,
        "profile report must fail closed on promotion/held-out status")
    _require(report.get("retained_states_per_chain") == [draws] * 4,
        "four synthetic retained-state counts differ")
    _require(report.get("pooled_retained_states") == 4 * draws
        and report.get("pooled_chain_count") == 4,
        "pooled synthetic chain counts differ")
    _require(report.get("batch_sizes") == list(batch_sizes),
        "pooled batch scales differ from the profile contract")
    _require(len(report.get("statistics_files", [])) == 4
        and len(report.get("map_files", [])) == 4,
        "the report did not read all four fixture chains")
    _require(all(record.get("zip_member_crc_verified") is True
        for record in report["statistics_files"]),
        "a synthetic statistics archive failed CRC verification")
    _require(all(record.get("selected_members_crc_verified") is True
        for record in report["map_files"]),
        "a selected synthetic map member failed CRC verification")
    expected_raw_bytes = (37 * grid_size**3 * np.dtype(np.float64).itemsize
        * len(batch_sizes))
    _require(all(record.get("raw_array_bytes_estimate") == expected_raw_bytes
        for record in report["statistics_files"]),
        "N256 checkpoint raw-array estimates differ from the profile schema")

    expected_region = training_count_cell_region(training_keys,
        count_grid_size=count_grid_size, field_grid_size=grid_size,
        population_count=population_count)
    expected_training_cells = int(expected_region.sum())
    for batch_size in batch_sizes:
        scale = report["pooled_field_mcse"][str(batch_size)]
        expected_batches = 4 * draws // batch_size
        _require(scale.get("status") ==
            "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE"
            and scale.get("retained_states") == 4 * draws
            and scale.get("complete_batches") == expected_batches,
            f"pooled scale {batch_size} did not finalize the expected counts")
        for metric, regional in scale["regional_summary"].items():
            summaries = regional if isinstance(regional, list) else [regional]
            for component in summaries:
                full = component["full_box"]
                training = component["training_count_cells"]
                _require(full.get("region_cells") == grid_size**3
                    and full.get("defined_cells") == grid_size**3
                    and full.get("undefined_cells") == 0,
                    f"{metric} full-box map is unexpectedly undefined")
                _require(training.get("region_cells") == expected_training_cells
                    and training.get("defined_cells") == expected_training_cells
                    and training.get("undefined_cells") == 0,
                    f"{metric} training-region denominator differs")
                _require(full.get("p50") is not None and full.get("p95") is not None
                    and 0. <= full["p50"] <= full["p95"],
                    f"{metric} percentiles are not finite/ordered")
    _assert_finite_json(report)
    return expected_raw_bytes


def finalize_profile(job_root, raw_report_path, training_split, *,
        grid_size=256, draws=512, batch_sizes=(32, 64),
        generation_seconds=None, report_seconds=None,
        requested_mem_gib=28, expected_peak_gib=21,
        count_grid_size=128, population_count=6):
    job_root = Path(job_root).resolve(strict=True)
    manifest_path = job_root / "fixtures" / "SYNTHETIC_PROFILE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_dir = manifest.get("fixture_dir")
    _require(fixture_dir == "fixtures",
        "manifest does not name the expected job-scoped fixture directory")
    fixture_root = (job_root / fixture_dir).resolve(strict=True)
    _require(fixture_root.parent == job_root and fixture_root.name == fixture_dir,
        "refusing to inspect or remove a fixture directory outside this job")
    _require(not manifest_path.is_symlink(),
        "refusing cleanup when the fixture manifest is a symlink")
    _require(manifest.get("schema") == PROFILE_SCHEMA
        and manifest.get("synthetic_fixture") is True
        and manifest.get("job_root") == str(job_root),
        "synthetic fixture manifest does not match this Slurm job")
    expected_files = {item["path"]: item["size_bytes"]
        for item in manifest.get("fixture_files", [])}
    actual_files = {str(path.relative_to(fixture_root)): path.stat().st_size
        for path in fixture_root.rglob("*") if path.is_file()
        and path != manifest_path}
    _require(expected_files == actual_files,
        "fixture contents differ from the job manifest; refusing cleanup")
    raw_report_path = Path(raw_report_path).resolve(strict=True)
    _require(raw_report_path.parent == job_root,
        "raw report must remain inside the job-specific output directory")
    report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    keys = _training_keys(training_split)
    expected_raw_bytes = validate_synthetic_report(report,
        grid_size=grid_size, draws=draws, batch_sizes=batch_sizes,
        training_keys=keys, count_grid_size=count_grid_size,
        population_count=population_count)

    fixture_files = [dict(path=f"{fixture_dir}/{item['path']}",
        size_bytes=item["size_bytes"])
        for item in manifest["fixture_files"]]
    fixture_files.append(dict(
        path=f"{fixture_dir}/SYNTHETIC_PROFILE_MANIFEST.json",
        size_bytes=manifest_path.stat().st_size))
    fixture_bytes = sum(item["size_bytes"] for item in fixture_files)
    selected_map_bytes_per_chain = grid_size**3 * (
        10 * np.dtype(np.float64).itemsize + 3 * np.dtype(np.bool_).itemsize)

    payload = dict(schema=PROFILE_SCHEMA,
        status="SYNTHETIC_N256_REPORT_RESOURCE_PROFILE_ONLY",
        synthetic_fixture=True, R2_complete=False, heldout_scored=False,
        posterior_promoted=False, job_id=os.environ.get("SLURM_JOB_ID"),
        source_commit=manifest["source_commit"],
        grid_size=grid_size, box_cMpc_h=384., draws_per_chain=draws,
        chains=4, batch_sizes=list(batch_sizes),
        minimum_batches=manifest["minimum_batches"],
        expected_raw_array_bytes_per_chain=expected_raw_bytes,
        expected_two_chain_raw_array_floor_bytes=2 * expected_raw_bytes,
        expected_total_fixture_bytes=fixture_bytes,
        expected_selected_map_member_bytes_per_chain=selected_map_bytes_per_chain,
        requested_memory_gib=requested_mem_gib,
        planning_peak_memory_gib=expected_peak_gib,
        measured_peak_gate_gib=requested_mem_gib / 1.2,
        measured_peak_headroom_status="PENDING_POST_JOB_SLURM_ACCOUNTING",
        fixture_generation_write_seconds=generation_seconds,
        report_step_elapsed_seconds=report_seconds,
        report_read_cache_caveat=("report follows fixture writes immediately; "
            "filesystem page-cache state is uncontrolled and may be warm"),
        integrity=dict(report_status=report["status"],
            pooled_retained_states=report["pooled_retained_states"],
            pooled_chain_count=report["pooled_chain_count"],
            pooled_scales={key: dict(status=value["status"],
                retained_states=value["retained_states"],
                complete_batches=value["complete_batches"])
                for key, value in report["pooled_field_mcse"].items()},
            statistics_crc_verified=True, selected_map_crc_verified=True),
        fixture_files=fixture_files, fixture_cleanup_bytes=fixture_bytes,
        fixture_cleanup_status="PENDING_VALIDATED_PROFILE_CLEANUP",
        four_chain_report=report,
        interpretation=("synthetic sufficient statistics only; no CF4 target, "
            "posterior, convergence, MCSE calibration, or held-out score"))
    profile_path = job_root / "resource_profile.json"
    with profile_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")

    shutil.rmtree(fixture_root)
    payload["fixture_cleanup_status"] = "REMOVED_AFTER_VALIDATED_PROFILE"
    temp_path = job_root / ".resource_profile.json.tmp"
    with temp_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temp_path.replace(profile_path)
    raw_report_path.unlink()
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build")
    build.add_argument("--job-root", type=Path, required=True)
    build.add_argument("--training-split", type=Path, required=True)
    build.add_argument("--grid-size", type=int, default=256)
    build.add_argument("--draws", type=int, default=512)
    build.add_argument("--batch-sizes", type=int, nargs=2, default=(32, 64))
    build.add_argument("--minimum-batches", type=int, default=8)
    build.add_argument("--count-grid-size", type=int, default=128)
    build.add_argument("--population-count", type=int, default=6)
    finish = sub.add_parser("finalize")
    finish.add_argument("--job-root", type=Path, required=True)
    finish.add_argument("--raw-report", type=Path, required=True)
    finish.add_argument("--training-split", type=Path, required=True)
    finish.add_argument("--grid-size", type=int, default=256)
    finish.add_argument("--draws", type=int, default=512)
    finish.add_argument("--batch-sizes", type=int, nargs=2, default=(32, 64))
    finish.add_argument("--generation-seconds", type=float)
    finish.add_argument("--report-seconds", type=float)
    finish.add_argument("--requested-mem-gib", type=float, default=28)
    finish.add_argument("--expected-peak-gib", type=float, default=21)
    finish.add_argument("--count-grid-size", type=int, default=128)
    finish.add_argument("--population-count", type=int, default=6)
    args = parser.parse_args(argv)
    if args.mode == "build":
        reports = build_fixtures(args.job_root, args.training_split,
            grid_size=args.grid_size, draws=args.draws,
            batch_sizes=tuple(args.batch_sizes),
            minimum_batches=args.minimum_batches,
            count_grid_size=args.count_grid_size,
            population_count=args.population_count)
        print(json.dumps(dict(synthetic_fixture=True, reports=reports),
            allow_nan=False))
    else:
        result = finalize_profile(args.job_root, args.raw_report,
            args.training_split, grid_size=args.grid_size, draws=args.draws,
            batch_sizes=tuple(args.batch_sizes),
            generation_seconds=args.generation_seconds,
            report_seconds=args.report_seconds,
            requested_mem_gib=args.requested_mem_gib,
            expected_peak_gib=args.expected_peak_gib,
            count_grid_size=args.count_grid_size,
            population_count=args.population_count)
        print(json.dumps(dict(status=result["status"],
            fixture_cleanup_bytes=result["fixture_cleanup_bytes"],
            fixture_cleanup_status=result["fixture_cleanup_status"]),
            allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
