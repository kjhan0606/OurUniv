"""Explicit cross-chain spatial disagreement for posterior field summaries.

These maps are diagnostics, not convergence certificates. A cell is defined
only when every supplied chain has a finite mean and posterior SD; missing or
invalid chains are counted instead of silently omitted from regional metrics.
"""
from __future__ import annotations

import numpy as np


def _ratio(numerator, denominator):
    result = np.full_like(denominator, np.nan, dtype=np.float64)
    finite = np.isfinite(denominator)
    positive = finite & (denominator > 0)
    result[positive] = numerator[positive] / denominator[positive]
    zero = finite & (denominator == 0)
    result[zero & (numerator == 0)] = 0.
    result[zero & (numerator > 0)] = np.inf
    return result


def _summaries(ratio, defined, denominator, regions):
    output = {}
    for name, region in regions.items():
        mask = np.asarray(region)
        if mask.dtype != np.bool_ or mask.shape != defined.shape:
            raise ValueError(f"region {name!r} must be an aligned boolean map")
        selected_defined = defined & mask
        selected = ratio[mask]
        selected_defined = selected_defined[mask]
        selected_denominator = denominator[mask]
        values = selected[selected_defined]
        finite = values[np.isfinite(values)]
        output[name] = dict(
            region_cells=int(mask.sum()),
            defined_cells=int(selected_defined.sum()),
            undefined_cells=int(mask.sum() - selected_defined.sum()),
            finite_ratio_cells=int(finite.size),
            nonfinite_ratio_cells=int(selected_defined.sum() - finite.size),
            zero_denominator_cells=int(np.sum(
                selected_defined & (selected_denominator == 0))),
            p50=float(np.percentile(finite, 50)) if finite.size else None,
            p95=float(np.percentile(finite, 95)) if finite.size else None,
            p99=float(np.percentile(finite, 99)) if finite.size else None,
            max_finite=float(np.max(finite)) if finite.size else None)
    return output


def compare_chain_field_maps(chain_means, chain_posterior_sds, draw_counts,
                             *, valid=None, regions=None):
    """Compare chain maps without trimming chains or dropping undefined cells.

    Parameters
    ----------
    chain_means, chain_posterior_sds:
        Arrays shaped ``(nchain, ...)`` with matching field shapes. Each SD is
        the within-chain posterior sample SD, not its Monte Carlo error.
    draw_counts:
        Number of retained states per chain. Unequal counts are supported and
        are used only for the descriptive pooled-map weighting; chain-to-chain
        spread is computed across chains, not weighted to hide a short chain.
    valid:
        Optional boolean map shaped like the chain arrays. If false for any
        chain, that cell is undefined for cross-chain comparison. A true flag
        cannot mask nonfinite means/SDs or a negative SD.
    regions:
        Named boolean masks over the field shape. Defaults to the full box.
    """
    means = np.asarray(chain_means, dtype=np.float64)
    sds = np.asarray(chain_posterior_sds, dtype=np.float64)
    counts = np.asarray(draw_counts)
    if means.ndim < 2 or means.shape != sds.shape or means.shape[0] < 2:
        raise ValueError("matching maps from at least two chains required")
    if (counts.shape != (means.shape[0],)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 2)):
        raise ValueError("each chain needs an integer count of at least two states")
    finite = np.isfinite(means) & np.isfinite(sds) & (sds >= 0)
    if valid is None:
        valid_mask = finite
    else:
        valid_mask = np.asarray(valid)
        if valid_mask.dtype != np.bool_ or valid_mask.shape != means.shape:
            raise ValueError("validity mask must be boolean and match chain maps")
        if np.any(valid_mask & ~finite):
            raise ValueError("valid chain cells must have finite means and nonnegative SDs")
        valid_mask = valid_mask & finite
    defined = np.all(valid_mask, axis=0)

    shape = means.shape[1:]
    pooled_mean = np.full(shape, np.nan, dtype=np.float64)
    between_mean_sd = np.full(shape, np.nan, dtype=np.float64)
    within_sd = np.full(shape, np.nan, dtype=np.float64)
    pooled_sd = np.full(shape, np.nan, dtype=np.float64)
    mean_sd = np.full(shape, np.nan, dtype=np.float64)
    between_sd_sd = np.full(shape, np.nan, dtype=np.float64)
    mean_ratio = np.full(shape, np.nan, dtype=np.float64)
    sd_ratio = np.full(shape, np.nan, dtype=np.float64)

    counts = counts.astype(np.int64, copy=False)
    weights = counts.astype(np.float64) / counts.sum()
    if np.any(defined):
        cell_means = means[:, defined]
        cell_sds = sds[:, defined]
        mean = np.tensordot(weights, cell_means, axes=(0, 0))
        mean_between_var = np.var(cell_means, axis=0, ddof=1)
        mixture_between_var = np.tensordot(
            weights, (cell_means - mean[None]) ** 2, axes=(0, 0))
        within_var = np.tensordot(weights, cell_sds ** 2, axes=(0, 0))
        sd_bar = np.tensordot(weights, cell_sds, axes=(0, 0))
        sd_between_var = np.var(cell_sds, axis=0, ddof=1)
        pooled_mean[defined] = mean
        between_mean_sd[defined] = np.sqrt(mean_between_var)
        within_sd[defined] = np.sqrt(within_var)
        pooled_sd[defined] = np.sqrt(within_var + mixture_between_var)
        mean_sd[defined] = sd_bar
        between_sd_sd[defined] = np.sqrt(sd_between_var)
        mean_ratio[defined] = _ratio(
            between_mean_sd[defined], within_sd[defined])
        sd_ratio[defined] = _ratio(
            between_sd_sd[defined], mean_sd[defined])

    if regions is None:
        regions = {"full_box": np.ones(shape, dtype=bool)}
    elif "full_box" not in regions:
        regions = {"full_box": np.ones(shape, dtype=bool), **regions}
    return dict(
        status="CROSS_CHAIN_FIELD_DISAGREEMENT_DIAGNOSTIC_NOT_CONVERGENCE_EVIDENCE",
        n_chains=int(means.shape[0]), draw_counts=counts.copy(),
        total_retained_states=int(counts.sum()), defined_mask=defined,
        valid_chain_count=np.sum(valid_mask, axis=0).astype(np.int16),
        pooled_mean=pooled_mean,
        between_chain_mean_sd=between_mean_sd,
        within_chain_posterior_sd=within_sd,
        descriptive_pooled_sd_not_promotable=pooled_sd,
        mean_map_disagreement_over_within_sd=mean_ratio,
        mean_chain_posterior_sd=mean_sd,
        between_chain_sd_map_sd=between_sd_sd,
        posterior_sd_map_disagreement_over_mean_sd=sd_ratio,
        regional_summary=dict(
            mean_map_disagreement_over_within_sd=_summaries(
                mean_ratio, defined, within_sd, regions),
            posterior_sd_map_disagreement_over_mean_sd=_summaries(
                sd_ratio, defined, mean_sd, regions)))
