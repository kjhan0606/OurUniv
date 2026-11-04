"""Spatially correlated synthetic calibration for R2 field MCSE summaries.

The synthetic chains are stationary by construction.  Their per-cell temporal
correlations are known, while spatial correlation changes the effective number
of cells contributing to a regional percentile.  This is estimator
calibration only: it is not an R2 target evaluation, chain-convergence test,
or field promotion.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from cf4_r2_field_batch_mcse import MultiScaleFieldMCSEAccumulator


def _spectral_filter(size: int, correlation_length: float) -> np.ndarray:
    frequencies = np.fft.fftfreq(size)
    kx, ky, kz = np.meshgrid(frequencies, frequencies, frequencies,
        indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    filt = np.exp(-2. * np.pi**2 * correlation_length**2 * k2)
    norm = np.sqrt(np.mean(filt * filt))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("spatial filter has invalid normalization")
    return filt / norm


def correlated_gaussian_sequence(rng: np.random.Generator, count: int,
        size: int, correlation_length: float) -> np.ndarray:
    """Return unit-marginal Gaussian maps with periodic spatial correlation."""
    if (not isinstance(count, int) or count < 1
            or not isinstance(size, int) or size < 4
            or not np.isfinite(correlation_length) or correlation_length < 0):
        raise ValueError("positive sequence/grid and nonnegative correlation length required")
    white = rng.standard_normal((count, size, size, size))
    if correlation_length == 0:
        return white
    filt = _spectral_filter(size, float(correlation_length))
    modes = np.fft.fftn(white, axes=(-3, -2, -1))
    del white
    modes *= filt[None, ...]
    return np.fft.ifftn(modes, axes=(-3, -2, -1)).real


def _ar1_field_path(rng, draws, phi, size, correlation_length):
    innovations = correlated_gaussian_sequence(rng, draws, size,
        correlation_length)
    path = np.empty_like(innovations)
    path[0] = innovations[0]
    innovation_scale = np.sqrt(1. - phi * phi)
    for index in range(1, draws):
        path[index] = phi * path[index - 1] + innovation_scale * innovations[index]
    del innovations
    return path


def _occupancy_path(rng, draws, probability, phi, size, correlation_length):
    uniforms = ndtr(correlated_gaussian_sequence(rng, draws, size,
        correlation_length))
    path = np.empty(uniforms.shape, dtype=bool)
    path[0] = uniforms[0] < probability
    for index in range(1, draws):
        previous = path[index - 1]
        threshold = np.where(previous,
            probability + phi * (1. - probability),
            probability * (1. - phi))
        path[index] = uniforms[index] < threshold
    del uniforms
    return path


def synthetic_design(size: int):
    """Build fixed spatially heterogeneous persistence and reporting regions."""
    coordinate = (np.arange(size, dtype=np.float64) - size // 2) / size
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    radius2 = x * x + y * y + z * z
    central = np.exp(-radius2 / 0.035)
    probability = 0.2 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * x)
        * np.cos(2 * np.pi * y))
    phi_density = 0.55 + 0.40 * central
    phi_occupancy = 0.65 + 0.30 * central
    phi_velocity = np.stack((
        0.48 + 0.42 * central,
        0.54 + 0.40 * central,
        0.60 + 0.36 * central))
    regions = {
        "full_box": np.ones((size, size, size), dtype=bool),
        "training_region": radius2 <= 0.20**2,
    }
    if not regions["training_region"].any():
        raise ValueError("synthetic training region is empty at this grid size")
    return dict(probability=probability, phi_density=phi_density,
        phi_occupancy=phi_occupancy, phi_velocity=phi_velocity,
        regions=regions)


def _finite_mean_ratio(gamma0, gamma_for_lag, draws, chains):
    variance_sum = draws * gamma0
    for lag in range(1, draws):
        variance_sum += 2. * (draws - lag) * gamma_for_lag(lag)
    variance_mean = variance_sum / (draws * draws * chains)
    ratio2 = np.divide(variance_mean, gamma0,
        out=np.full_like(variance_mean, np.nan), where=gamma0 > 0)
    return np.sqrt(np.maximum(ratio2, 0.))


def exact_reference_maps(design, *, draws, chains, lognormal_sigma=0.02):
    """Known finite/asymptotic MCSE-to-SD references for the synthetic law."""
    p = design["probability"]
    phi_o = design["phi_occupancy"]
    phi_rho = design["phi_density"]
    velocity_phi = design["phi_velocity"]
    eps2 = float(lognormal_sigma) ** 2

    def pair_occupancy(lag):
        return p * p + p * (1. - p) * phi_o**lag

    density_variance = p * np.exp(eps2) - p * p

    def density_covariance(lag):
        return pair_occupancy(lag) * np.exp(eps2 * phi_rho**lag) - p * p

    density_mean = _finite_mean_ratio(density_variance,
        density_covariance, draws, chains)

    # Delta-method long-run variance for the sample SD.  R_t is unit-mean
    # lognormal and independent of the Bernoulli occupancy chain O_t.
    sd_lrv = np.zeros_like(p)
    for lag in range(draws):
        q = phi_rho**lag
        p11 = pair_occupancy(lag)
        centered_fourth = (
            p11 * np.exp(eps2 * (2. + 4. * q))
            - 4. * p * p11 * np.exp(eps2 * (1. + 2. * q))
            + 2. * p**3 * np.exp(eps2)
            + 4. * p * p * p11 * np.exp(eps2 * q)
            - 3. * p**4)
        covariance = centered_fourth - density_variance**2
        sd_lrv += covariance if lag == 0 else 2. * covariance
    density_sd = np.sqrt(np.maximum(sd_lrv, 0.) / (chains * draws))
    density_sd /= 2. * density_variance

    occ_var = p * (1. - p)
    occupancy = _finite_mean_ratio(occ_var,
        lambda lag: occ_var * phi_o**lag, draws, chains)

    velocity_mean = []
    velocity_sd = []
    for phi_v in velocity_phi:
        mean_lrv = np.zeros_like(p)
        sd_lrv = np.zeros_like(p)
        for lag in range(draws):
            p11 = pair_occupancy(lag)
            mean_cov = p11 * phi_v**lag / (p * p)
            sd_cov = 2. * p11 * phi_v**(2 * lag) / (p * p)
            mean_lrv += mean_cov if lag == 0 else 2. * mean_cov
            sd_lrv += sd_cov if lag == 0 else 2. * sd_cov
        velocity_mean.append(np.sqrt(np.maximum(mean_lrv, 0.)
            / (chains * draws)))
        # The conditional velocity SD is one; d sqrt(var) / d var = 1/2.
        velocity_sd.append(0.5 * np.sqrt(np.maximum(sd_lrv, 0.)
            / (chains * draws)))
    result = dict(density_mean=density_mean, density_sd=density_sd,
        occupancy=occupancy)
    for axis, label in enumerate(("x", "y", "z")):
        result[f"velocity_mean_{label}"] = velocity_mean[axis]
        result[f"velocity_sd_{label}"] = velocity_sd[axis]
    return result


def _make_chain_accumulator(rng, design, *, draws, size, correlation_length,
        batch_sizes, minimum_batches, lognormal_sigma):
    probability = design["probability"]
    occupancy = _occupancy_path(rng, draws, probability,
        design["phi_occupancy"], size, correlation_length)
    latent = np.empty((4, draws, size, size, size), dtype=np.float64)
    latent[0] = _ar1_field_path(rng, draws, design["phi_density"], size,
        correlation_length)
    for axis in range(3):
        latent[axis + 1] = _ar1_field_path(rng, draws,
            design["phi_velocity"][axis], size, correlation_length)

    accumulator = MultiScaleFieldMCSEAccumulator((size, size, size),
        batch_sizes=batch_sizes, minimum_batches=minimum_batches,
        density_reference=1.)
    shift = 0.5 * lognormal_sigma**2
    for index in range(draws):
        occupied = occupancy[index]
        rho = occupied * np.exp(lognormal_sigma * latent[0, index] - shift)
        accumulator.update(dict(rho=rho,
            mean_velocity_km_s=100. * latent[1:, index],
            velocity_valid=occupied))
    del latent, occupancy
    return accumulator


def _regional_p95(values, regions):
    result = {}
    for name, region in regions.items():
        selected = np.asarray(values)[region]
        finite = selected[np.isfinite(selected)]
        result[name] = dict(cell_count=int(region.sum()),
            finite_cell_count=int(finite.size),
            p95=float(np.percentile(finite, 95)) if finite.size else None)
    return result


def _estimate_maps(report):
    result = dict(density_mean=report["density_mean_mcse_over_sd"],
        density_sd=report["density_sd_mcse_over_sd"],
        occupancy=report["velocity_occupancy_mcse_over_sd"])
    for axis, label in enumerate(("x", "y", "z")):
        result[f"velocity_mean_{label}"] = report[
            "velocity_mean_mcse_over_sd"][axis]
        result[f"velocity_sd_{label}"] = report[
            "velocity_sd_mcse_over_sd"][axis]
    return result


def _summarize_estimates(samples, truth, threshold):
    def wilson95(successes, trials):
        if trials == 0:
            return None
        z = 1.959963984540054
        p_hat = successes / trials
        denominator = 1. + z * z / trials
        center = (p_hat + z * z / (2. * trials)) / denominator
        half = z * np.sqrt(p_hat * (1. - p_hat) / trials
            + z * z / (4. * trials * trials)) / denominator
        return [float(max(0., center - half)),
            float(min(1., center + half))]

    result = {}
    for region, exact in truth.items():
        values = np.asarray(samples[region], dtype=np.float64)
        false_passes = int(np.sum(values <= threshold))
        false_fails = int(np.sum(values > threshold))
        false_pass = (false_passes / values.size
            if exact > threshold else None)
        false_fail = (false_fails / values.size
            if exact <= threshold else None)
        result[region] = dict(true_p95=float(exact),
            replicate_count=int(values.size),
            estimated_p95_median=float(np.median(values)),
            estimated_p95_q05=float(np.quantile(values, 0.05)),
            estimated_p95_q95=float(np.quantile(values, 0.95)),
            estimated_p95_q99=float(np.quantile(values, 0.99)),
            underestimation_fraction=float(np.mean(values < exact)),
            false_pass_fraction=false_pass,
            false_pass_wilson95=(wilson95(false_passes, values.size)
                if exact > threshold else None),
            false_fail_fraction=false_fail,
            false_fail_wilson95=(wilson95(false_fails, values.size)
                if exact <= threshold else None),
            threshold=float(threshold))
    return result


def run_spatial_calibration(*, size=16, draws=512, chains=4, replicates=256,
        seed=20261104, batch_sizes=(32, 64), minimum_batches=8,
        correlation_lengths=(0., 1., 2.), threshold=0.10,
        lognormal_sigma=0.02):
    """Calibrate region-p95 field MCSE using known stationary synthetic maps."""
    if (not isinstance(size, int) or size < 4
            or not isinstance(draws, int) or draws < 2
            or not isinstance(chains, int) or chains < 2
            or not isinstance(replicates, int) or replicates < 1
            or not np.isfinite(threshold) or threshold <= 0
            or not np.isfinite(lognormal_sigma) or lognormal_sigma <= 0):
        raise ValueError("invalid synthetic calibration dimensions or scales")
    batch_sizes = tuple(batch_sizes)
    if (len(batch_sizes) != 2 or len(set(batch_sizes)) != 2
            or any(not isinstance(value, int) or value < 2 for value in batch_sizes)
            or any(draws % value or draws // value < minimum_batches
                for value in batch_sizes)):
        raise ValueError("two batch scales must divide draws and meet minimum batches")
    if not correlation_lengths or any(not np.isfinite(value) or value < 0
            for value in correlation_lengths):
        raise ValueError("at least one nonnegative spatial correlation length is required")

    base = synthetic_design(size)
    reference = exact_reference_maps(base, draws=draws, chains=chains,
        lognormal_sigma=lognormal_sigma)
    true_p95 = {key: {region: float(np.percentile(values[mask], 95))
        for region, mask in base["regions"].items()}
        for key, values in reference.items()}
    rng = np.random.default_rng(seed)
    output_scenarios = []

    for scenario_index, correlation_length in enumerate(correlation_lengths):
        collected = {key: {str(size): {region: []
            for region in base["regions"]} for size in batch_sizes}
            for key in reference}
        for replicate in range(replicates):
            chain_accumulators = []
            for chain_index in range(chains):
                chain_seed = int(rng.integers(0, np.iinfo(np.uint64).max,
                    dtype=np.uint64))
                chain_rng = np.random.default_rng(chain_seed)
                chain_accumulators.append(_make_chain_accumulator(chain_rng,
                    base, draws=draws, size=size,
                    correlation_length=float(correlation_length),
                    batch_sizes=batch_sizes, minimum_batches=minimum_batches,
                    lognormal_sigma=lognormal_sigma))
            pooled = chain_accumulators[0]
            for source in chain_accumulators[1:]:
                pooled.merge_independent_chains_inplace(source)
            for batch_size in batch_sizes:
                maps = _estimate_maps(pooled.finalize_scale(batch_size))
                for metric, values in maps.items():
                    for region, summary in _regional_p95(values,
                            base["regions"]).items():
                        collected[metric][str(batch_size)][region].append(
                            summary["p95"])
        metrics = {}
        for metric, scales in collected.items():
            metrics[metric] = {}
            for batch_size, regions in scales.items():
                metrics[metric][batch_size] = _summarize_estimates(regions,
                    true_p95[metric], threshold)
        output_scenarios.append(dict(correlation_length_cells=float(
            correlation_length), grid_size=size, replicates=replicates,
            regional_cell_counts={name: int(mask.sum())
                for name, mask in base["regions"].items()},
            true_regional_p95=true_p95,
            mcse_p95_estimate_distribution=metrics))

    return dict(schema="cf4-r2-spatial-field-mcse-calibration-v1",
        status="SYNTHETIC_STATIONARY_ESTIMATOR_CALIBRATION_ONLY",
        posterior_promoted=False, heldout_read=False,
        interpretation=("Known stationary synthetic fields only; no CF4 target, "
            "R2 convergence, held-out prediction, or posterior qualification."),
        draws_per_chain=draws, chains=chains, replicates=replicates,
        seed=seed, grid_size=size, batch_sizes=list(batch_sizes),
        minimum_batches=minimum_batches, threshold=threshold,
        lognormal_sigma=lognormal_sigma,
        temporal_persistence=dict(
            density_min=float(base["phi_density"].min()),
            density_max=float(base["phi_density"].max()),
            occupancy_min=float(base["phi_occupancy"].min()),
            occupancy_max=float(base["phi_occupancy"].max()),
            velocity_min=float(base["phi_velocity"].min()),
            velocity_max=float(base["phi_velocity"].max())),
        scenarios=output_scenarios)
