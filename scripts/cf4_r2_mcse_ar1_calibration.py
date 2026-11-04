#!/usr/bin/env python3
"""Synthetic calibration of fixed-batch field-mean MCSE at the R2 draw cap.

This tests only the batch-means estimator under stationary Gaussian AR(1)
chains. It does not evaluate the CF4 target, stationarity of an R2 run, or a
spatial field. The regional p95 proxy is the estimator distribution over
independent scalar cells with the same AR(1) coefficient.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def exact_pooled_mean_mcse_ratio(draws: int, chains: int, phi: float) -> float:
    """Exact stationary AR(1) SD(mean)/marginal-SD for equal chains."""
    lag = np.arange(1, draws, dtype=np.float64)
    variance_single_mean = (draws + 2. * np.sum(
        (draws - lag) * np.power(phi, lag))) / draws ** 2
    return float(np.sqrt(variance_single_mean / chains))


def simulate_ar1(replicates: int, chains: int, draws: int, phi: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Draw independent stationary AR(1) chains, shape (rep, chain, draw)."""
    values = np.empty((replicates, chains, draws), dtype=np.float64)
    values[:, :, 0] = rng.standard_normal((replicates, chains))
    innovations = rng.standard_normal((replicates, chains, draws - 1))
    scale = np.sqrt(1. - phi * phi)
    for index in range(1, draws):
        values[:, :, index] = (phi * values[:, :, index - 1]
            + scale * innovations[:, :, index - 1])
    return values


def run(*, draws=512, chains=4, replicates=5000, seed=20261104,
        phis=(0., .5, .7, .9, .95, .98), batch_sizes=(32, 64),
        threshold=.10):
    if draws < 2 or chains < 2 or replicates < 100:
        raise ValueError("draws/chains must be >=2 and replicates >=100")
    if (len(batch_sizes) != 2 or len(set(batch_sizes)) != 2
            or any(size < 2 or draws % size for size in batch_sizes)):
        raise ValueError("two distinct batch sizes must divide the retained draws")
    if any(draws // size < 8 for size in batch_sizes):
        raise ValueError("each chain needs at least eight batches at both scales")
    if any(not 0. <= phi < 1. for phi in phis):
        raise ValueError("stationary AR(1) coefficients must be in [0,1)")

    rng = np.random.default_rng(seed)
    rows = []
    for phi in phis:
        values = simulate_ar1(replicates, chains, draws, phi, rng)
        pooled_sd = values.reshape(replicates, -1).std(axis=1, ddof=1)
        estimates = {}
        for size in batch_sizes:
            batches_per_chain = draws // size
            batch_means = values.reshape(
                replicates, chains, batches_per_chain, size).mean(axis=3)
            batch_means = batch_means.reshape(replicates, -1)
            mcse = batch_means.std(axis=1, ddof=1) / np.sqrt(
                chains * batches_per_chain)
            estimates[str(size)] = mcse / pooled_sd

        both_pass = np.ones(replicates, dtype=bool)
        scale_summary = {}
        for size in batch_sizes:
            ratio = estimates[str(size)]
            both_pass &= ratio <= threshold
            scale_summary[str(size)] = dict(
                batches_per_chain=draws // size,
                median=float(np.median(ratio)),
                p95=float(np.quantile(ratio, .95)),
                p99=float(np.quantile(ratio, .99)))
        rows.append(dict(phi=float(phi),
            exact_pooled_mean_mcse_over_sd=exact_pooled_mean_mcse_ratio(
                draws, chains, phi),
            scale_estimates=scale_summary,
            both_scales_point_estimate_below_threshold_fraction=float(
                both_pass.mean())))
    return dict(schema="cf4-r2-mcse-ar1-calibration-v1",
        interpretation="synthetic estimator calibration only; not CF4 inference or convergence",
        draws_per_chain=draws, chains=chains, replicates=replicates,
        seed=seed, batch_sizes=list(batch_sizes),
        minimum_batches_per_chain=min(draws // size for size in batch_sizes),
        threshold=threshold, results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--phis", type=float, nargs="+",
        default=(0., .5, .7, .9, .95, .98))
    parser.add_argument("--batch-sizes", type=int, nargs=2, default=(32, 64))
    parser.add_argument("--threshold", type=float, default=.10)
    args = parser.parse_args()
    print(json.dumps(run(draws=args.draws, chains=args.chains,
        replicates=args.replicates, seed=args.seed, phis=args.phis,
        batch_sizes=tuple(args.batch_sizes), threshold=args.threshold),
        sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
