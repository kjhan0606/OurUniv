"""One fixed stationary synthetic stress case for the R2 field-MCSE gate.

This is a case-specific negative-control study. It is not a CF4 posterior,
sampler convergence test, general gate calibration, or map promotion.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

from cf4_r2_cross_chain_field_diagnostics import compare_chain_field_maps
from cf4_r2_field_batch_mcse import MultiScaleFieldMCSEAccumulator
from cf4_r2_mcse_spatial_calibration import (
    correlated_gaussian_sequence,
    exact_reference_maps,
)


DEFAULT_SEED = 20261105
DEFAULT_ORACLE_SEED = 20261106
DEFAULT_ORACLE_HISTORIES = 250_000
DEFAULT_ORACLE_CHUNK_SIZE = 8_192
DEFAULT_REPLICATES = 256
DEFAULT_GRID_SIZE = 16
DEFAULT_DRAWS = 512
DEFAULT_CHAINS = 4
DEFAULT_BATCH_SIZES = (32, 64)
DEFAULT_MINIMUM_BATCHES = 8
DEFAULT_SPATIAL_CORRELATION_LENGTH = 2.0
DEFAULT_THRESHOLD = 0.10
DEFAULT_OCCUPANCY = 0.20
DEFAULT_OCCUPANCY_PERSISTENCE = 0.20
DEFAULT_VELOCITY_PERSISTENCE = (0.20, 0.90, 0.20)
DEFAULT_DENSITY_PERSISTENCE = 0.20
DEFAULT_LOGNORMAL_SIGMA = 0.02
ORACLE_RELATIVE_MCSE_LIMIT = 0.005
ALPHA_ORACLE = 0.025
ALPHA_BINOMIAL = 0.025


def reporting_regions(size: int) -> dict[str, np.ndarray]:
    """Return the fixed full-box and central-radius-0.2 reporting regions."""
    if not isinstance(size, int) or size < 4:
        raise ValueError("grid size must be an integer >=4")
    coordinate = (np.arange(size, dtype=np.float64) - size // 2) / size
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    radius2 = x * x + y * y + z * z
    regions = {
        "full_box": np.ones((size, size, size), dtype=bool),
        "training_region": radius2 <= 0.20**2,
    }
    if not regions["training_region"].any():
        raise ValueError("central reporting sphere is empty")
    return regions


def _occupancy_step(rng: np.random.Generator, previous: np.ndarray,
        probability: float, persistence: float) -> np.ndarray:
    """Advance independent Bernoulli cells with stationary lag-one rho."""
    previous = np.asarray(previous, dtype=bool)
    if (not np.isfinite(probability) or not 0. < probability < 1.
            or not np.isfinite(persistence) or abs(persistence) >= 1.):
        raise ValueError("occupancy probability/persistence are invalid")
    threshold = np.where(previous,
        probability + persistence * (1. - probability),
        probability * (1. - persistence))
    return rng.random(previous.shape) < threshold


def stationary_occupancy_path(rng: np.random.Generator, *, draws: int,
        size: int, probability: float = DEFAULT_OCCUPANCY,
        persistence: float = DEFAULT_OCCUPANCY_PERSISTENCE) -> np.ndarray:
    """Generate a spatial product of stationary two-state Markov paths."""
    if not isinstance(draws, int) or draws < 2:
        raise ValueError("at least two occupancy states are required")
    path = np.empty((draws, size, size, size), dtype=bool)
    path[0] = rng.random((size, size, size)) < probability
    for index in range(1, draws):
        path[index] = _occupancy_step(rng, path[index - 1],
            probability, persistence)
    return path


def stationary_gaussian_path(rng: np.random.Generator, *, draws: int,
        size: int, persistence: float,
        correlation_length: float = DEFAULT_SPATIAL_CORRELATION_LENGTH) -> np.ndarray:
    """Generate a jointly stationary periodic Gaussian AR(1) field path.

    The initial map and all innovations are independent draws from the same
    filtered spatial Gaussian law. A single scalar temporal coefficient is
    applied uniformly over the grid, preserving the joint spatial covariance.
    """
    if (not isinstance(draws, int) or draws < 2
            or not isinstance(size, int) or size < 4
            or not np.isfinite(persistence) or abs(persistence) >= 1.
            or not np.isfinite(correlation_length)
            or correlation_length < 0.):
        raise ValueError("invalid stationary Gaussian field settings")
    innovations = correlated_gaussian_sequence(rng, draws, size,
        correlation_length)
    path = np.empty_like(innovations)
    path[0] = innovations[0]
    innovation_scale = math.sqrt(1. - persistence * persistence)
    for index in range(1, draws):
        path[index] = (persistence * path[index - 1]
            + innovation_scale * innovations[index])
    return path


def conditional_velocity_mean_variance(occupancy, velocity_persistence: float) -> float:
    """Compute Q/M^2 for one set of chain occupancy histories in O(T).

    The returned value is the exact conditional Gaussian variance of the
    pooled occupied-state velocity mean, for unit latent velocity variance.
    """
    occupied = np.asarray(occupancy, dtype=bool)
    if (occupied.ndim != 2 or occupied.shape[0] < 1 or occupied.shape[1] < 1
            or not np.isfinite(velocity_persistence)
            or abs(velocity_persistence) >= 1.):
        raise ValueError("occupancy must be (chains, draws) with valid persistence")
    count = int(occupied.sum())
    if count == 0:
        raise ValueError("pooled occupied-state velocity mean is undefined when M=0")
    running = np.zeros(occupied.shape[0], dtype=np.float64)
    quadratic = np.zeros_like(running)
    for index in range(occupied.shape[1]):
        current = occupied[:, index].astype(np.float64)
        previous_weighted = velocity_persistence * running
        quadratic += current * (1. + 2. * previous_weighted)
        running = previous_weighted + current
    return float(quadratic.sum() / (count * count))


def _zero_occupancy_probability(*, draws: int, chains: int,
        probability: float, persistence: float) -> float:
    """Exact P(M=0) for the pooled histories of one field cell."""
    if (draws < 1 or chains < 1 or not 0. < probability < 1.
            or abs(persistence) >= 1.):
        raise ValueError("invalid dimensions or occupancy law")
    p00 = 1. - probability * (1. - persistence)
    per_chain = (1. - probability) * p00 ** (draws - 1)
    return float(per_chain ** chains)


def estimate_velocity_mean_oracle(*, histories=DEFAULT_ORACLE_HISTORIES,
        chunk_size=DEFAULT_ORACLE_CHUNK_SIZE, chains=DEFAULT_CHAINS,
        draws=DEFAULT_DRAWS, probability=DEFAULT_OCCUPANCY,
        occupancy_persistence=DEFAULT_OCCUPANCY_PERSISTENCE,
        velocity_persistence=DEFAULT_VELOCITY_PERSISTENCE,
        seed=DEFAULT_ORACLE_SEED, alpha=ALPHA_ORACLE) -> dict:
    """Estimate finite-sample velocity-mean references with bounded memory.

    Occupancy histories are independent across replicas and cells. All velocity
    components share those oracle histories, but use their own fixed temporal
    covariance. The oracle RNG is separate from the gate-replicate RNG.
    """
    velocity_persistence = tuple(float(value) for value in velocity_persistence)
    if (not isinstance(histories, int) or histories < 2
            or not isinstance(chunk_size, int) or chunk_size < 1
            or not isinstance(chains, int) or chains < 1
            or not isinstance(draws, int) or draws < 2
            or len(velocity_persistence) != 3
            or any(not np.isfinite(value) or abs(value) >= 1.
                for value in velocity_persistence)
            or not 0. < probability < 1.
            or not np.isfinite(occupancy_persistence)
            or abs(occupancy_persistence) >= 1.
            or not 0. < alpha < 0.5):
        raise ValueError("invalid finite-sample oracle settings")

    rng = np.random.default_rng(seed)
    coefficients = np.asarray(velocity_persistence, dtype=np.float64)
    processed = 0
    count = 0
    mean = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)
    zero_count = 0
    while processed < histories:
        batch = min(chunk_size, histories - processed)
        previous = rng.random((batch, chains)) < probability
        total_occupied = previous.sum(axis=1, dtype=np.int64)
        running = np.broadcast_to(previous, (3, batch, chains)).astype(
            np.float64, copy=True)
        quadratic = running.copy()  # diagonal covariance contribution
        for _ in range(1, draws):
            threshold = np.where(previous,
                probability + occupancy_persistence * (1. - probability),
                probability * (1. - occupancy_persistence))
            current = rng.random((batch, chains)) < threshold
            current_float = current.astype(np.float64)
            previous_weighted = coefficients[:, None, None] * running
            quadratic += current_float[None] * (1. + 2. * previous_weighted)
            running = previous_weighted + current_float[None]
            total_occupied += current.sum(axis=1, dtype=np.int64)
            previous = current

        nonzero = total_occupied > 0
        zero_count += int(np.count_nonzero(~nonzero))
        processed += batch
        if np.any(nonzero):
            values = quadratic[:, nonzero].sum(axis=2)
            denominator = total_occupied[nonzero].astype(np.float64) ** 2
            values /= denominator[None]
            block_n = values.shape[1]
            block_mean = values.mean(axis=1)
            block_m2 = np.sum((values - block_mean[:, None]) ** 2, axis=1)
            new_count = count + block_n
            delta = block_mean - mean
            m2 += block_m2 + delta * delta * count * block_n / new_count
            mean += delta * block_n / new_count
            count = new_count

    if count < 2:
        raise FloatingPointError("oracle had fewer than two nonzero histories")
    sample_variance = m2 / (count - 1.)
    se_variance = np.sqrt(sample_variance / count)
    ratio = np.sqrt(mean)
    se_ratio = se_variance / (2. * ratio)
    relative_mcse = se_variance / (2. * mean)
    critical = float(student_t.ppf(1. - alpha, count - 1))
    lower = np.sqrt(np.maximum(mean - critical * se_variance, 0.))
    upper = np.sqrt(np.maximum(mean + critical * se_variance, 0.))
    zero_probability = _zero_occupancy_probability(draws=draws,
        chains=chains, probability=probability,
        persistence=occupancy_persistence)
    labels = ("x", "y", "z")
    components = {}
    for index, label in enumerate(labels):
        components[label] = dict(
            velocity_persistence=velocity_persistence[index],
            variance_ratio_mean=float(mean[index]),
            variance_ratio_mcse=float(se_variance[index]),
            finite_sample_mean_error_ratio=float(ratio[index]),
            ratio_delta_method_mcse=float(se_ratio[index]),
            ratio_relative_mcse=float(relative_mcse[index]),
            one_sided_lower_confidence=float(lower[index]),
            one_sided_upper_confidence=float(upper[index]),
            confidence_level=float(1. - alpha),
            confidence_method="Student-t delta-method interval for sqrt(E[Q/M^2])",
            monte_carlo_histories=int(count),
            zero_occupancy_history_count=int(zero_count),
            zero_occupancy_probability=float(zero_probability),
            conditioned_on_nonzero_occupancy=True)
    status = "READY" if zero_count == 0 else "INCONCLUSIVE_ZERO_OCCUPANCY_HISTORY"
    return dict(status=status, seed=int(seed), histories_requested=int(histories),
        histories_processed=int(processed), histories_used=int(count),
        chunk_size=int(chunk_size), chains=int(chains),
        draws_per_chain=int(draws), occupancy_probability=float(probability),
        occupancy_persistence=float(occupancy_persistence),
        oracle_relative_mcse_limit=ORACLE_RELATIVE_MCSE_LIMIT,
        required_component="y", required_lower_bound=float(DEFAULT_THRESHOLD),
        lower_bound_alpha=float(alpha), components=components)


def _truth_reference_maps(*, size, draws, chains, lognormal_sigma,
        oracle_components):
    shape = (size, size, size)
    design = dict(
        probability=np.full(shape, DEFAULT_OCCUPANCY, dtype=np.float64),
        phi_occupancy=np.full(shape, DEFAULT_OCCUPANCY_PERSISTENCE,
            dtype=np.float64),
        phi_density=np.full(shape, DEFAULT_DENSITY_PERSISTENCE,
            dtype=np.float64),
        phi_velocity=np.stack([
            np.full(shape, value, dtype=np.float64)
            for value in DEFAULT_VELOCITY_PERSISTENCE]))
    references = exact_reference_maps(design, draws=draws, chains=chains,
        lognormal_sigma=lognormal_sigma)
    for axis, label in enumerate(("x", "y", "z")):
        references[f"velocity_mean_{label}"] = np.full(shape,
            oracle_components[label]["finite_sample_mean_error_ratio"],
            dtype=np.float64)
    regions = reporting_regions(size)
    regional = {
        key: {name: float(np.percentile(values[mask], 95))
            for name, mask in regions.items()}
        for key, values in references.items()}
    return references, regional, regions


def _oracle_is_ready(oracle: dict) -> tuple[bool, list[str]]:
    reasons = []
    if oracle.get("status") != "READY":
        reasons.append("oracle reports an invalid occupancy history")
    for label, component in oracle.get("components", {}).items():
        if component["ratio_relative_mcse"] > ORACLE_RELATIVE_MCSE_LIMIT:
            reasons.append(f"velocity-{label} oracle relative MCSE exceeds limit")
    y = oracle.get("components", {}).get("y", {})
    if y.get("one_sided_lower_confidence", -math.inf) <= DEFAULT_THRESHOLD:
        reasons.append("velocity-y truth lower bound does not exceed 0.10")
    return not reasons, reasons


def _field_components(report: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    shape = np.asarray(report["density_mean_mcse_over_sd"]).shape
    all_defined = np.ones(shape, dtype=bool)
    velocity_defined = np.asarray(report["velocity_variance_defined"], dtype=bool)
    result = {
        "density_mean": (np.asarray(report["density_mean_mcse_over_sd"]), all_defined),
        "density_sd": (np.asarray(report["density_sd_mcse_over_sd"]), all_defined),
        "occupancy": (np.asarray(report["velocity_occupancy_mcse_over_sd"]),
            all_defined),
    }
    for axis, label in enumerate(("x", "y", "z")):
        result[f"velocity_mean_{label}"] = (
            np.asarray(report["velocity_mean_mcse_over_sd"])[axis],
            velocity_defined[axis])
        result[f"velocity_sd_{label}"] = (
            np.asarray(report["velocity_sd_mcse_over_sd"])[axis],
            velocity_defined[axis])
    return result


def _regional_gate_summary(values, defined, region, threshold) -> dict:
    values = np.asarray(values, dtype=np.float64)
    defined = np.asarray(defined, dtype=bool)
    region = np.asarray(region, dtype=bool)
    if values.shape != region.shape or defined.shape != region.shape:
        raise ValueError("gate maps and reporting region must align")
    selected_defined = defined[region]
    selected = values[region]
    finite = np.isfinite(selected)
    p95 = (float(np.percentile(selected, 95))
        if selected.size and finite.all() and selected_defined.all() else None)
    passed = bool(selected_defined.all() and finite.all()
        and p95 is not None and p95 <= threshold)
    return dict(passed=passed, region_cells=int(region.sum()),
        defined_cells=int(selected_defined.sum()),
        undefined_cells=int((~selected_defined).sum()),
        finite_cells=int(finite.sum()), nonfinite_cells=int((~finite).sum()),
        p95=p95, threshold=float(threshold))


def _cross_chain_report(chain_reports, regions, threshold) -> dict:
    """Evaluate chain disagreement once, independent of MCSE batch scale."""
    shape = np.asarray(chain_reports[0]["density_contrast_mean"]).shape
    all_valid = np.ones((len(chain_reports),) + shape, dtype=bool)
    density = compare_chain_field_maps(
        np.stack([row["density_contrast_mean"] for row in chain_reports]),
        np.stack([row["density_contrast_posterior_sd"] for row in chain_reports]),
        np.full(len(chain_reports), int(chain_reports[0]["retained_states"])),
        valid=all_valid, regions=regions)
    occupancy = compare_chain_field_maps(
        np.stack([row["velocity_occupancy_fraction"] for row in chain_reports]),
        np.stack([row["velocity_occupancy_posterior_sd"] for row in chain_reports]),
        np.full(len(chain_reports), int(chain_reports[0]["retained_states"])),
        valid=all_valid, regions=regions)
    velocity_means = np.stack([
        row["velocity_conditional_mean"] for row in chain_reports])
    velocity_sds = np.stack([
        row["velocity_conditional_posterior_sd"] for row in chain_reports])
    velocity_valid = np.stack([
        row["velocity_variance_defined"] for row in chain_reports])
    draw_counts = np.full(len(chain_reports),
        int(chain_reports[0]["retained_states"]))
    # Region masks describe one scalar 3-D field. Velocity maps have an extra
    # leading component axis, so compare each component against the same
    # scalar masks rather than passing a 3-D mask to a 4-D map comparison.
    velocity = {}
    for axis, label in enumerate(("x", "y", "z")):
        velocity[label] = compare_chain_field_maps(
            velocity_means[:, axis], velocity_sds[:, axis], draw_counts,
            valid=velocity_valid[:, axis], regions=regions)

    outputs = {}
    for name, report, component in (
            ("density", density, None), ("occupancy", occupancy, None)):
        outputs[name] = {}
        for region_name, region in regions.items():
            defined = report["defined_mask"]
            mean = _regional_gate_summary(
                report["mean_map_disagreement_over_within_sd"], defined,
                region, threshold)
            sd = _regional_gate_summary(
                report["posterior_sd_map_disagreement_over_mean_sd"], defined,
                region, threshold)
            outputs[name][region_name] = dict(mean_map=mean,
                posterior_sd_map=sd,
                passed=bool(mean["passed"] and sd["passed"]))
    for axis, label in enumerate(("x", "y", "z")):
        name = f"velocity_{label}"
        report = velocity[label]
        outputs[name] = {}
        for region_name, region in regions.items():
            defined = report["defined_mask"]
            mean = _regional_gate_summary(
                report["mean_map_disagreement_over_within_sd"], defined,
                region, threshold)
            sd = _regional_gate_summary(
                report["posterior_sd_map_disagreement_over_mean_sd"],
                defined, region, threshold)
            outputs[name][region_name] = dict(mean_map=mean,
                posterior_sd_map=sd,
                passed=bool(mean["passed"] and sd["passed"]))
    passed = all(region_result["passed"]
        for quantity in outputs.values()
        for region_result in quantity.values())
    return dict(passed=bool(passed), quantities=outputs,
        diagnostic_status="CROSS_CHAIN_DISAGREEMENT_NOT_LATENT_TRUTH")


def _one_sided_binomial_upper(successes: int, trials: int, alpha: float) -> float:
    if (not isinstance(successes, int) or not isinstance(trials, int)
            or trials < 1 or successes < 0 or successes > trials
            or not 0. < alpha < 1.):
        raise ValueError("invalid exact binomial confidence-bound inputs")
    if successes == trials:
        return 1.0
    return float(beta_distribution.ppf(1. - alpha,
        successes + 1, trials - successes))


def _run_chain(rngs, *, size, draws, batch_sizes, minimum_batches,
        correlation_length, lognormal_sigma):
    occupancy_rng, density_rng, vx_rng, vy_rng, vz_rng = rngs
    occupancy = stationary_occupancy_path(occupancy_rng, draws=draws,
        size=size)
    density = stationary_gaussian_path(density_rng, draws=draws,
        persistence=DEFAULT_DENSITY_PERSISTENCE, size=size,
        correlation_length=correlation_length)
    velocity_paths = [
        stationary_gaussian_path(rng, draws=draws, persistence=persistence,
            size=size, correlation_length=correlation_length)
        for rng, persistence in zip((vx_rng, vy_rng, vz_rng),
            DEFAULT_VELOCITY_PERSISTENCE)]
    accumulator = MultiScaleFieldMCSEAccumulator((size, size, size),
        batch_sizes=batch_sizes, minimum_batches=minimum_batches,
        density_reference=1.0)
    shift = 0.5 * lognormal_sigma**2
    for index in range(draws):
        occupied = occupancy[index]
        rho = occupied * np.exp(lognormal_sigma * density[index] - shift)
        velocities = np.stack([path[index] for path in velocity_paths]) * 100.
        accumulator.update(dict(rho=rho, mean_velocity_km_s=velocities,
            velocity_valid=occupied))
    # The posterior mean and SD maps are identical across batch scales. Save
    # one copy before pooled sufficient statistics are merged.
    chain_map = accumulator.finalize_scale(max(batch_sizes))
    if chain_map.get("status") != "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE":
        raise FloatingPointError("chain diagnostic map is unavailable")
    return accumulator, chain_map


def _one_replicate(replicate_index, *, seed, size, draws, chains,
        batch_sizes, minimum_batches, correlation_length, lognormal_sigma,
        regions, threshold):
    replicate_sequence = np.random.SeedSequence([int(seed), int(replicate_index)])
    chain_sequences = replicate_sequence.spawn(chains)
    accumulators = []
    chain_reports = []
    for sequence in chain_sequences:
        streams = [np.random.default_rng(child)
            for child in sequence.spawn(5)]
        accumulator, chain_map = _run_chain(streams, size=size, draws=draws,
            batch_sizes=batch_sizes, minimum_batches=minimum_batches,
            correlation_length=correlation_length,
            lognormal_sigma=lognormal_sigma)
        accumulators.append(accumulator)
        chain_reports.append(chain_map)

    # The cross-chain diagnostic is evaluated once for these retained states,
    # not once per batch scale.
    cross_chain = _cross_chain_report(chain_reports, regions, threshold)
    pooled = accumulators[0]
    pooled.merge_independent_chains_inplace(accumulators[1:])

    mcse = {}
    for batch_size in batch_sizes:
        report = pooled.finalize_scale(batch_size, regions=regions)
        if report.get("status") != "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE":
            raise FloatingPointError("pooled field-MCSE report is unavailable")
        components = _field_components(report)
        for metric, (values, defined) in components.items():
            for region_name, region in regions.items():
                result = _regional_gate_summary(values, defined, region,
                    threshold)
                mcse[f"batch{batch_size}/{region_name}/{metric}"] = result

    mcse_passed = all(row["passed"] for row in mcse.values())
    return dict(replicate_index=int(replicate_index),
        mcse_components=mcse, mcse_only_pass=bool(mcse_passed),
        cross_chain=cross_chain, cross_chain_pass=cross_chain["passed"],
        g_field_pass=bool(mcse_passed and cross_chain["passed"]))


def _stress_status(*, risk_passed: bool, g_field_passes: int) -> str:
    """Separate the risk-bound result from whether the gate ever passed."""
    if g_field_passes < 0:
        raise ValueError("G_field pass count must be nonnegative")
    all_rejected = g_field_passes == 0
    if risk_passed:
        return ("STRESS_CASE_RISK_BOUND_MET_BUT_GATE_REJECTED_ALL_REPLICATES"
            if all_rejected else "STRESS_CASE_RISK_BOUND_MET")
    return ("STRESS_CASE_RISK_BOUND_NOT_MET_BUT_GATE_REJECTED_ALL_REPLICATES"
        if all_rejected else "STRESS_CASE_RISK_BOUND_NOT_MET")


def run_gate_replicates(*, reference, seed=DEFAULT_SEED,
        replicates=DEFAULT_REPLICATES, size=DEFAULT_GRID_SIZE,
        draws=DEFAULT_DRAWS, chains=DEFAULT_CHAINS,
        batch_sizes=DEFAULT_BATCH_SIZES,
        minimum_batches=DEFAULT_MINIMUM_BATCHES,
        correlation_length=DEFAULT_SPATIAL_CORRELATION_LENGTH,
        lognormal_sigma=DEFAULT_LOGNORMAL_SIGMA,
        threshold=DEFAULT_THRESHOLD,
        on_replicate: Callable[[dict], None] | None = None) -> dict:
    """Run the frozen alternative and preserve each paired gate outcome."""
    ready, reasons = _oracle_is_ready(reference)
    if not ready:
        return dict(status="INCONCLUSIVE_ORACLE", reasons=reasons,
            replicate_count=0)
    batch_sizes = tuple(batch_sizes)
    if (replicates < 1 or chains < 2 or draws < 2
            or tuple(sorted(batch_sizes)) != tuple(batch_sizes)
            or len(batch_sizes) != 2
            or any(draws % size_ or draws // size_ < minimum_batches
                for size_ in batch_sizes)):
        raise ValueError("invalid fixed field-MCSE stress dimensions")
    regions = reporting_regions(size)
    gate_counts = {"g_field_passes": 0, "mcse_only_passes": 0,
        "cross_chain_passes": 0}
    component_failures = {}
    false_acceptances = 0
    for index in range(replicates):
        row = _one_replicate(index, seed=seed, size=size, draws=draws,
            chains=chains, batch_sizes=batch_sizes,
            minimum_batches=minimum_batches,
            correlation_length=correlation_length,
            lognormal_sigma=lognormal_sigma, regions=regions,
            threshold=threshold)
        row["false_acceptance"] = row["g_field_pass"]
        gate_counts["g_field_passes"] += int(row["g_field_pass"])
        gate_counts["mcse_only_passes"] += int(row["mcse_only_pass"])
        gate_counts["cross_chain_passes"] += int(row["cross_chain_pass"])
        false_acceptances += int(row["false_acceptance"])
        for name, result in row["mcse_components"].items():
            component_failures[name] = component_failures.get(name, 0) + int(
                not result["passed"])
        for quantity, regions_result in row["cross_chain"]["quantities"].items():
            for region_name, result in regions_result.items():
                for subgate in ("mean_map", "posterior_sd_map"):
                    name = f"cross_chain/{region_name}/{quantity}/{subgate}"
                    component_failures[name] = component_failures.get(name, 0) + int(
                        not result[subgate]["passed"])
        if on_replicate is not None:
            on_replicate(row)

    upper95 = _one_sided_binomial_upper(false_acceptances, replicates, 0.05)
    upper975 = _one_sided_binomial_upper(false_acceptances, replicates,
        ALPHA_BINOMIAL)
    risk_passed = upper95 <= 0.05
    status = _stress_status(risk_passed=risk_passed,
        g_field_passes=gate_counts["g_field_passes"])
    return dict(status=status, stress_case_only=True,
        posterior_promoted=False, heldout_read=False,
        false_acceptance_definition=(
            "complete G_field pass under the prequalified fixed alternative "
            "whose true velocity-y mean-error ratio lower bound exceeds 0.10"),
        alternative_truth_reference=reference["components"]["y"],
        replicate_count=int(replicates),
        false_acceptances=int(false_acceptances),
        false_acceptance_fraction=float(false_acceptances / replicates),
        one_sided_95_exact_clopper_pearson_upper=upper95,
        one_sided_97_5_exact_clopper_pearson_upper=upper975,
        one_sided_95_risk_criterion=float(upper95) <= 0.05,
        familywise_95_nominal_risk_criterion=(upper975 <= 0.05
            and reference["components"]["y"]["one_sided_lower_confidence"]
                > threshold),
        risk_alpha=0.05, familywise_oracle_alpha=ALPHA_ORACLE,
        familywise_binomial_alpha=ALPHA_BINOMIAL,
        gate_counts=gate_counts, component_failure_counts=component_failures,
        interpretation=(
            "Bounds false acceptance for this one synthetic stationary "
            "alternative only. It does not calibrate acceptance of genuinely "
            "precise fields or any production posterior."))


def make_oracle_references(*, histories=DEFAULT_ORACLE_HISTORIES,
        chunk_size=DEFAULT_ORACLE_CHUNK_SIZE, seed=DEFAULT_ORACLE_SEED,
        size=DEFAULT_GRID_SIZE, draws=DEFAULT_DRAWS, chains=DEFAULT_CHAINS,
        lognormal_sigma=DEFAULT_LOGNORMAL_SIGMA) -> dict:
    oracle = estimate_velocity_mean_oracle(histories=histories,
        chunk_size=chunk_size, chains=chains, draws=draws, seed=seed)
    reference_maps, regional, regions = _truth_reference_maps(size=size,
        draws=draws, chains=chains, lognormal_sigma=lognormal_sigma,
        oracle_components=oracle["components"])
    ready, reasons = _oracle_is_ready(oracle)
    # References for variance maps are asymptotic influence approximations;
    # only the exact finite-sample velocity-y mean oracle defines the stress.
    kinds = {key: ("finite-sample occupancy oracle" if key.startswith("velocity_mean_")
        else "finite-sample analytic" if key in ("density_mean", "occupancy")
        else "asymptotic influence-function descriptive reference")
        for key in reference_maps}
    return dict(status="READY" if ready else "INCONCLUSIVE_ORACLE",
        oracle=oracle, reasons=reasons,
        truth_reference_kind=kinds,
        truth_regional_p95=regional,
        reporting_region_cell_counts={name: int(mask.sum())
            for name, mask in regions.items()},
        velocity_mean_true_p95_equal_across_regions=True,
        threshold=DEFAULT_THRESHOLD,
        gate_replicates_may_start=bool(ready),
        posterior_promoted=False, heldout_read=False)


def run_fixed_stress(*, seed=DEFAULT_SEED,
        replicates=DEFAULT_REPLICATES, size=DEFAULT_GRID_SIZE,
        draws=DEFAULT_DRAWS, chains=DEFAULT_CHAINS,
        batch_sizes=DEFAULT_BATCH_SIZES,
        minimum_batches=DEFAULT_MINIMUM_BATCHES,
        correlation_length=DEFAULT_SPATIAL_CORRELATION_LENGTH,
        lognormal_sigma=DEFAULT_LOGNORMAL_SIGMA,
        threshold=DEFAULT_THRESHOLD,
        reference: dict,
        on_replicate: Callable[[dict], None] | None = None) -> dict:
    """Run one fixed, prequalified case and return non-promoting metadata."""
    if threshold != DEFAULT_THRESHOLD:
        raise ValueError("the field precision threshold is frozen at 0.10")
    summary = run_gate_replicates(reference=reference["oracle"], seed=seed,
        replicates=replicates, size=size, draws=draws, chains=chains,
        batch_sizes=batch_sizes, minimum_batches=minimum_batches,
        correlation_length=correlation_length,
        lognormal_sigma=lognormal_sigma, threshold=threshold,
        on_replicate=on_replicate)
    return dict(schema="cf4-r2-joint-field-mcse-stress-v1",
        status=summary["status"], seed=int(seed), grid_size=int(size),
        draws_per_chain=int(draws), chains=int(chains),
        replicates=int(replicates), batch_sizes=list(batch_sizes),
        minimum_batches=int(minimum_batches),
        spatial_correlation_length_cells=float(correlation_length),
        density_persistence=DEFAULT_DENSITY_PERSISTENCE,
        velocity_persistence=list(DEFAULT_VELOCITY_PERSISTENCE),
        occupancy_probability=DEFAULT_OCCUPANCY,
        occupancy_persistence=DEFAULT_OCCUPANCY_PERSISTENCE,
        lognormal_sigma=float(lognormal_sigma), threshold=float(threshold),
        reporting_region_cell_counts=reference["reporting_region_cell_counts"],
        truth_reference_kind=reference["truth_reference_kind"],
        truth_regional_p95=reference["truth_regional_p95"],
        oracle=reference["oracle"], reasons=reference["reasons"],
        summary=summary, posterior_promoted=False, heldout_read=False,
        q_goal=("This qualifies one precision-gate stress alternative for the "
            "384 cMpc/h N256 conditional CF4+galaxy z=0 field only; it does "
            "not produce that posterior. MW/M31 remain role-ambiguous and M33 "
            "unresolved; their observables must constrain those same roles on "
            "the same NEW evolved LG field at <=0.3 cMpc/h, with native truth "
            "IDs reserved for calibration/evaluation."),
        q_lean=("One fixed stationary stress case reuses the field accumulator; "
            "no target call, held-out access, parameter sweep, GPU run, or "
            "gravity simulation."))
