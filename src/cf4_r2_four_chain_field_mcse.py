"""Post-process four independent R2 chain field-MCSE artifacts.

This report pools fixed-batch sufficient statistics only after computing and
recording chain-separated field disagreement. It is a diagnostic, not a
convergence verdict, posterior promotion, or held-out scorer.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_r2_cross_chain_field_diagnostics import compare_chain_field_maps
from cf4_r2_field_reporting_regions import training_count_cell_region
from cf4_r2_retained_field_mcse import RetainedFieldMCSEDiagnostics


_REPORT_INVARIANTS = (
    "source_commit", "target", "N", "box_cMpc_h",
    "observation_target_revision", "source_conditioning_rule",
    "field_mcse_contract")


def _read_reports(report_paths):
    paths = [Path(path).resolve(strict=True) for path in report_paths]
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ValueError("exactly four distinct chain report paths are required")
    reports = []
    records = []
    for path in paths:
        raw = path.read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError("each chain report must be a JSON object")
        reports.append(report)
        records.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            chain_id=report.get("chain_id")))
    return paths, reports, records


def _validate_reports(reports):
    for report in reports:
        if (report.get("status") != "N256_CHAIN_FINISHED_REQUIRES_DIAGNOSTICS"
                or report.get("R2_complete") is not False
                or report.get("heldout_scored") is not False
                or report.get("initialization_policy") != "independent_prior"
                or report.get("long_chain") is not True):
            raise ValueError("inputs must be completed, pre-heldout independent-prior chains")
    for key in _REPORT_INVARIANTS:
        if (key not in reports[0]
                or any(report.get(key) != reports[0].get(key)
                    for report in reports[1:])):
            raise ValueError(f"four-chain field-MCSE contract mismatch: {key}")

    designs = [report.get("chain_rng_design", {}) for report in reports]
    indices = [design.get("chain_index") for design in designs]
    seeds = [design.get("master_seed") for design in designs]
    chain_ids = [report.get("chain_id") for report in reports]
    if (sorted(indices) != [0, 1, 2, 3]
            or any(design.get("chain_count") != 4 for design in designs)
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in seeds)
            or len(set(seeds)) != 1
            or any(not isinstance(chain_id, str) or not chain_id
                for chain_id in chain_ids)
            or len(set(chain_ids)) != 4):
        raise ValueError("four unique chain-indexed streams from one master seed are required")
    expected_streams = {"prior": 0, "audit": 1, "transition": 2}
    for index, design in zip(indices, designs):
        streams = design.get("streams", {})
        if (design.get("bit_generator") != "PCG64"
                or set(streams) != set(expected_streams)
                or any(streams[name].get("stream_id") != stream_id
                    or streams[name].get("spawn_key") != [index, stream_id]
                    for name, stream_id in expected_streams.items())):
            raise ValueError("independent prior/audit/transition stream map differs")

    metadata = []
    for report in reports:
        field = report.get("field_mcse")
        if (not isinstance(field, dict)
                or field.get("sufficient_statistics_file_status")
                    != "UNASSESSED_CHAIN_STATISTICS_FOR_POSTPROCESSING"):
            raise ValueError("chain report lacks normal-completion field-MCSE statistics")
        statistics_name = field.get("sufficient_statistics_file")
        if (not isinstance(statistics_name, str)
                or Path(statistics_name).name != statistics_name):
            raise ValueError("field-MCSE statistics path must be a local basename")
        scales = field.get("scales")
        if not isinstance(scales, dict) or not scales:
            raise ValueError("chain report lacks finalized field-MCSE scale records")
        scale_records = {}
        for raw_size, record in scales.items():
            if not isinstance(record, dict):
                raise ValueError("invalid field-MCSE scale record")
            map_name = record.get("map_file")
            if (record.get("status") != "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE"
                    or not isinstance(map_name, str)
                    or Path(map_name).name != map_name):
                raise ValueError("each chain needs complete field-MCSE map output")
            scale_records[str(raw_size)] = record
        metadata.append(dict(statistics_name=statistics_name,
            field_summary=field, scales=scale_records))

    reference_scales = set(metadata[0]["scales"])
    if any(set(item["scales"]) != reference_scales for item in metadata[1:]):
        raise ValueError("chain field-MCSE batch scales differ")
    return metadata, indices, seeds[0]


def _load_statistics(path, train_keys, expected_summary):
    with np.load(path, allow_pickle=False) as state:
        diagnostics = RetainedFieldMCSEDiagnostics.from_checkpoint(
            state, train_keys=train_keys)
    actual = diagnostics.summary()
    for key in ("retained_states", "accepted_transitions", "rejected_transitions",
            "batch_sizes", "minimum_batches", "estimated_accumulator_bytes",
            "training_key_sha256", "reporting_region_sha256"):
        expected = expected_summary.get(key)
        if key == "batch_sizes" and expected is not None:
            expected = tuple(expected)
        if actual[key] != expected:
            raise ValueError(f"field-MCSE checkpoint/report mismatch: {key}")
    if (expected_summary.get("pooled_chain_count", 1) != 1
            or expected_summary.get("sufficient_statistics_raw_array_bytes_estimate")
                != actual["estimated_accumulator_bytes"]):
        raise ValueError("field-MCSE checkpoint/report pooling or size estimate mismatch")
    return diagnostics


def _load_chain_maps(paths, shape, batch_size):
    density_mean = np.empty((4,) + shape, dtype=np.float64)
    density_sd = np.empty_like(density_mean)
    velocity_mean = np.empty((4, 3) + shape, dtype=np.float64)
    velocity_sd = np.empty_like(velocity_mean)
    velocity_valid = np.empty((4, 3) + shape, dtype=bool)
    occupancy_mean = np.empty((4,) + shape, dtype=np.float64)
    occupancy_sd = np.empty_like(occupancy_mean)
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as maps:
            density_mean[index] = maps["density_contrast_mean"]
            density_sd[index] = maps["density_contrast_posterior_sd"]
            velocity_mean[index] = maps["velocity_conditional_mean"]
            velocity_sd[index] = maps["velocity_conditional_posterior_sd"]
            velocity_valid[index] = maps["velocity_variance_defined"]
            occupancy_mean[index] = maps["velocity_occupancy_fraction"]
            occupancy_sd[index] = maps["velocity_occupancy_posterior_sd"]
    return dict(density_mean=density_mean, density_sd=density_sd,
        velocity_mean=velocity_mean, velocity_sd=velocity_sd,
        velocity_valid=velocity_valid, occupancy_mean=occupancy_mean,
        occupancy_sd=occupancy_sd)


def _cross_chain_summary(means, sds, draw_counts, regions, valid=None):
    result = compare_chain_field_maps(means, sds, draw_counts,
        valid=valid, regions=regions)
    summary = dict(status=result["status"], draw_counts=result["draw_counts"].tolist(),
        regional_summary=result["regional_summary"])
    del result
    return summary


def evaluate_four_chain_field_mcse(report_paths, *, train_keys):
    """Create a non-promoting pooled-MCSE and cross-chain field report.

    ``train_keys`` must be read from the frozen training-only split. The
    function never opens held-out outcomes, full canonical states, likelihood
    inputs, or native truth identities. Per-chain map disagreement is
    computed before the in-place sufficient-statistics merge.
    """
    paths, reports, input_records = _read_reports(report_paths)
    metadata, chain_indices, master_seed = _validate_reports(reports)
    order = sorted(range(4), key=lambda index: chain_indices[index])
    paths = [paths[index] for index in order]
    reports = [reports[index] for index in order]
    input_records = [input_records[index] for index in order]
    metadata = [metadata[index] for index in order]
    chain_indices = [chain_indices[index] for index in order]
    keys = np.asarray(train_keys)
    if keys.ndim != 1 or (keys.size and not np.issubdtype(keys.dtype, np.integer)):
        raise ValueError("training keys must be a one-dimensional integer array")
    keys = np.asarray(keys, dtype=np.int64)

    first_batch_size = min(int(size) for size in metadata[0]["scales"])
    try:
        stat_paths = [(path.parent / item["statistics_name"]).resolve(strict=True)
            for path, item in zip(paths, metadata)]
        map_paths = [(path.parent / item["scales"][str(first_batch_size)]["map_file"])
            .resolve(strict=True) for path, item in zip(paths, metadata)]
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "a declared field-MCSE statistics or map file is missing") from error
    if len(set(stat_paths)) != 4 or len(set(map_paths)) != 4:
        raise ValueError("chain diagnostics must come from four distinct output files")
    if any(not path.is_file() for path in stat_paths + map_paths):
        raise FileNotFoundError("a declared field-MCSE statistics or map file is missing")

    first = _load_statistics(stat_paths[0], keys, metadata[0]["field_summary"])
    shape = first.shape
    if tuple(report.get("N") for report in reports) != (shape[0],) * 4:
        raise ValueError("field-MCSE grid shape differs from chain reports")
    scales = first.batch_sizes
    for item in metadata:
        field = item["field_summary"]
        if (tuple(field.get("batch_sizes", ())) != scales
                or set(item["scales"]) != {str(size) for size in scales}):
            raise ValueError("field-MCSE batch-scale report differs from checkpoint")

    region = training_count_cell_region(keys, count_grid_size=first.count_grid_size,
        field_grid_size=shape[0], population_count=first.population_count)
    regions = {"full_box": np.ones(shape, dtype=bool),
        RetainedFieldMCSEDiagnostics.REGION_NAME: region}
    chain_counts = np.asarray([item["field_summary"]["retained_states"]
        for item in metadata], dtype=np.int64)

    # Preserve chain-specific disagreement before the destructive pooled merge.
    maps = _load_chain_maps(map_paths, shape, first_batch_size)
    cross_chain = dict(
        density=_cross_chain_summary(maps["density_mean"], maps["density_sd"],
            chain_counts, regions),
        velocity_components=[_cross_chain_summary(
            maps["velocity_mean"][:, axis], maps["velocity_sd"][:, axis],
            chain_counts, regions, valid=maps["velocity_valid"][:, axis])
            for axis in range(3)],
        occupancy=_cross_chain_summary(maps["occupancy_mean"],
            maps["occupancy_sd"], chain_counts, regions))
    del maps
    gc.collect()

    # Load/merge one source chain at a time: only two full accumulators are
    # live during each merge. N256's theoretical raw-array floor is 18.5 GiB.
    pooled = first
    for index in range(1, 4):
        source = _load_statistics(stat_paths[index], keys,
            metadata[index]["field_summary"])
        pooled.merge_independent_chains_inplace(source)
        del source
        gc.collect()

    pooled_scales = {}
    for batch_size in scales:
        result = pooled.finalize_scale(batch_size)
        pooled_scales[str(batch_size)] = dict(
            status=result["status"], retained_states=result["retained_states"],
            complete_batches=result.get("complete_batches"),
            batch_size=batch_size,
            regional_summary=result.get("regional_summary"),
            reporting_region_cell_denominators=
                result.get("reporting_region_cell_denominators"))
        del result
        gc.collect()

    return dict(status="FOUR_CHAIN_FIELD_MCSE_DIAGNOSTIC_NOT_CONVERGENCE_EVIDENCE",
        R2_complete=False, heldout_scored=False, posterior_promoted=False,
        scalar_convergence="NOT_EVALUATED_BY_THIS_FIELD_REPORT",
        chain_indices=chain_indices, master_seed=master_seed,
        chain_ids=[record["chain_id"] for record in input_records],
        input_reports=input_records,
        statistics_files=[dict(path=str(path), size_bytes=path.stat().st_size,
            raw_array_bytes_estimate=item["field_summary"]["estimated_accumulator_bytes"],
            zip_member_crc_verified=True)
            for path, item in zip(stat_paths, metadata)],
        map_files=[dict(path=str(path), size_bytes=path.stat().st_size,
            selected_members_crc_verified=True)
            for path in map_paths],
        retained_states_per_chain=chain_counts.tolist(),
        pooled_retained_states=pooled.retained_states,
        pooled_accepted_transitions=pooled.accepted_transitions,
        pooled_rejected_transitions=pooled.rejected_transitions,
        batch_sizes=list(scales), minimum_batches=pooled.minimum_batches,
        raw_array_bytes_per_chain=first.estimated_accumulator_bytes,
        two_accumulator_raw_array_floor_bytes=2*first.estimated_accumulator_bytes,
        pooled_chain_count=pooled.accumulator.chain_count,
        cross_chain_field_disagreement=cross_chain,
        pooled_field_mcse=pooled_scales,
        observation_scope="conditional v6 only; no CF4 mark-availability/group-selection calibration",
        LG_limit="MW/M31 roles remain ambiguous and M33 unresolved; their observables must constrain the same NEW evolved LG field at <=0.3 cMpc/h; native truth identities calibration/evaluation only",
        interpretation="diagnostic summaries only; not stationarity, posterior, held-out, or field-promotion evidence")
