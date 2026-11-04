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


class StreamingChainFieldMapComparison:
    """Summarize cross-chain map disagreement while loading one chain at a time.

    Unlike :func:`compare_chain_field_maps`, this accumulator does not retain
    the chain axis or return full-grid maps. It is intended for large saved
    fields where the caller can stream map arrays from their source archive.
    The returned regional summaries use the same definitions as the full-map
    comparison.
    """

    def __init__(self, map_shape, draw_counts, *, regions=None):
        self.map_shape = tuple(map_shape)
        counts = np.asarray(draw_counts)
        if (len(self.map_shape) not in (3, 4)
                or min(self.map_shape) < 1
                or counts.ndim != 1 or counts.size < 2
                or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 2)):
            raise ValueError("valid map shape and at least two chain draw counts required")
        self.draw_counts = counts.astype(np.int64, copy=True)
        self.weights = self.draw_counts.astype(np.float64) / self.draw_counts.sum()
        if regions is None:
            regions = {"full_box": np.ones(self.map_shape[-3:], dtype=bool)}
        elif "full_box" not in regions:
            regions = {"full_box": np.ones(self.map_shape[-3:], dtype=bool), **regions}
        self.regions = {}
        for name, raw_mask in regions.items():
            mask = np.asarray(raw_mask)
            if mask.dtype != np.bool_ or mask.ndim != 3:
                raise ValueError(f"region {name!r} must be a three-dimensional boolean map")
            if (self.map_shape[-3:] != mask.shape
                    or self.map_shape[:-3] not in ((), (self.map_shape[0],))):
                raise ValueError(f"region {name!r} does not match field map shape")
            self.regions[name] = mask
        self._leading_components = (self.map_shape[0]
            if len(self.map_shape) == 4 else None)
        self.valid_chain_count = np.zeros(self.map_shape, dtype=np.uint16)
        self.mean_average = np.zeros(self.map_shape, dtype=np.float64)
        self.mean_m2 = np.zeros(self.map_shape, dtype=np.float64)
        self.sd_average = np.zeros(self.map_shape, dtype=np.float64)
        self.sd_m2 = np.zeros(self.map_shape, dtype=np.float64)
        self.within_sd_variance = np.zeros(self.map_shape, dtype=np.float64)
        self.weighted_sd_average = np.zeros(self.map_shape, dtype=np.float64)
        self._updates = 0

    @staticmethod
    def _update_welford(average, m2, values, valid, count_ratio, count_new):
        delta = np.subtract(values, average)
        np.copyto(delta, 0., where=~valid)
        squared = delta * delta
        squared *= count_ratio
        m2 += squared
        np.divide(delta, count_new, out=delta, where=valid)
        average += delta

    def update(self, means, posterior_sds, *, valid=None):
        if self._updates >= self.draw_counts.size:
            raise ValueError("more chain maps supplied than declared")
        means = np.asarray(means, dtype=np.float64)
        sds = np.asarray(posterior_sds, dtype=np.float64)
        if means.shape != self.map_shape or sds.shape != self.map_shape:
            raise ValueError("streamed chain maps must match the frozen map shape")
        finite = np.isfinite(means) & np.isfinite(sds) & (sds >= 0.)
        if valid is None:
            good = finite
        else:
            valid = np.asarray(valid)
            if valid.dtype != np.bool_ or valid.shape != self.map_shape:
                raise ValueError("streamed validity map must be boolean and aligned")
            if np.any(valid & ~finite):
                raise ValueError("valid chain cells must have finite means and nonnegative SDs")
            good = valid & finite

        old_count = self.valid_chain_count
        new_count = old_count.copy()
        np.add(new_count, good, out=new_count, casting="unsafe")
        count_ratio = np.zeros(self.map_shape, dtype=np.float64)
        np.divide(old_count, new_count, out=count_ratio, where=good)
        self._update_welford(self.mean_average, self.mean_m2,
            means, good, count_ratio, new_count)
        self._update_welford(self.sd_average, self.sd_m2,
            sds, good, count_ratio, new_count)

        temporary = np.empty(self.map_shape, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            np.multiply(sds, sds, out=temporary)
        if not np.isfinite(temporary[good]).all():
            raise ValueError("chain posterior-SD map squares overflow float64")
        np.copyto(temporary, 0., where=~good)
        temporary *= self.weights[self._updates]
        self.within_sd_variance += temporary
        np.multiply(sds, self.weights[self._updates], out=temporary)
        np.copyto(temporary, 0., where=~good)
        self.weighted_sd_average += temporary
        self.valid_chain_count = new_count
        self._updates += 1

    @staticmethod
    def _regional_ratio_summary(numerator_sd, denominator_sd, defined,
            region_cell_count):
        zero = defined & (denominator_sd == 0.)
        positive = defined & (denominator_sd > 0.)
        ratio = np.full(denominator_sd.shape, np.nan, dtype=np.float64)
        np.divide(numerator_sd, denominator_sd, out=ratio, where=positive)
        ratio[zero & (numerator_sd == 0.)] = 0.
        ratio[zero & (numerator_sd > 0.)] = np.inf
        finite = ratio[defined & np.isfinite(ratio)]
        return dict(region_cells=int(region_cell_count),
            defined_cells=int(defined.sum()),
            undefined_cells=int(region_cell_count - defined.sum()),
            finite_ratio_cells=int(finite.size),
            nonfinite_ratio_cells=int(defined.sum() - finite.size),
            zero_denominator_cells=int(zero.sum()),
            p50=float(np.percentile(finite, 50)) if finite.size else None,
            p95=float(np.percentile(finite, 95)) if finite.size else None,
            p99=float(np.percentile(finite, 99)) if finite.size else None,
            max_finite=float(np.max(finite)) if finite.size else None)

    def _summarize_region(self, region):
        components = (range(self._leading_components)
            if self._leading_components is not None else (None,))
        summaries = []
        for component in components:
            index = (component, region) if component is not None else region
            counts = self.valid_chain_count[index]
            defined = counts == self.draw_counts.size
            mean_between = np.sqrt(self.mean_m2[index] / (self.draw_counts.size - 1))
            within_sd = np.sqrt(self.within_sd_variance[index])
            sd_between = np.sqrt(self.sd_m2[index] / (self.draw_counts.size - 1))
            mean_sd = self.weighted_sd_average[index]
            summaries.append(dict(
                mean_map_disagreement_over_within_sd=
                    self._regional_ratio_summary(mean_between, within_sd,
                        defined, int(region.sum())),
                posterior_sd_map_disagreement_over_mean_sd=
                    self._regional_ratio_summary(sd_between, mean_sd,
                        defined, int(region.sum()))))
        return summaries if self._leading_components is not None else summaries[0]

    def finalize(self):
        if self._updates != self.draw_counts.size:
            raise ValueError("stream ended before all declared chain maps were read")
        by_region = {name: self._summarize_region(region)
            for name, region in self.regions.items()}
        if self._leading_components is None:
            mean_summary = {name: value[
                "mean_map_disagreement_over_within_sd"]
                for name, value in by_region.items()}
            sd_summary = {name: value[
                "posterior_sd_map_disagreement_over_mean_sd"]
                for name, value in by_region.items()}
        else:
            mean_summary = [{name: value[axis][
                "mean_map_disagreement_over_within_sd"]
                for name, value in by_region.items()}
                for axis in range(self._leading_components)]
            sd_summary = [{name: value[axis][
                "posterior_sd_map_disagreement_over_mean_sd"]
                for name, value in by_region.items()}
                for axis in range(self._leading_components)]
        return dict(
            status="CROSS_CHAIN_FIELD_DISAGREEMENT_DIAGNOSTIC_NOT_CONVERGENCE_EVIDENCE",
            n_chains=int(self.draw_counts.size), draw_counts=self.draw_counts.copy(),
            total_retained_states=int(self.draw_counts.sum()),
            regional_summary=dict(
                mean_map_disagreement_over_within_sd=mean_summary,
                posterior_sd_map_disagreement_over_mean_sd=sd_summary))


def compare_chain_field_maps_streaming(chain_maps, draw_counts, *, regions=None):
    """Consume ``(mean, posterior_sd, valid)`` maps one chain at a time."""
    iterator = iter(chain_maps)
    try:
        first_mean, first_sd, first_valid = next(iterator)
    except StopIteration as error:
        raise ValueError("at least two streamed chain maps are required") from error
    accumulator = StreamingChainFieldMapComparison(
        np.asarray(first_mean).shape, draw_counts, regions=regions)
    accumulator.update(first_mean, first_sd, valid=first_valid)
    del first_mean, first_sd, first_valid
    for mean, posterior_sd, valid in iterator:
        accumulator.update(mean, posterior_sd, valid=valid)
        del mean, posterior_sd, valid
    return accumulator.finalize()
