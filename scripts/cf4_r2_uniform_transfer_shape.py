#!/usr/bin/env python3
"""Field-free diagnostic of the CF4 apparent/absolute-K transfer shape.

This reads only the saved R2 training-bin table and source cosmology tables.
It is not a fit to the active field, a calibration, or posterior inference.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_marked_tracer_jax import (
    intrinsic_lf_reference_weights, source_mark_transfer,
)

jax.config.update('jax_enable_x64', True)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
PROFILE = BASE/'r2_n256_frozen_field_tracer_profile_20261002_v3/result.json'
SOURCE = BASE/'r2_marked_source_geometry_v1/geometry.npz'
SHELL_EDGES = np.arange(36.0, 96.0 + 12.0, 12.0)
QUADRATURE_ORDER = 24
GRID = np.linspace(-3.0, 3.0, 61)


def main():
    profile = json.loads(PROFILE.read_text())
    if profile.get('status') != 'FROZEN_FIELD_TRACER_PROFILE_BOUNDED_NOT_CONVERGED':
        raise ValueError('expected the saved, bounded, nonconverged training profile')
    tracer = np.asarray(profile['initial']['tracer_coordinates'], dtype=np.float64)
    if tracer.shape != (9,) or not np.isfinite(tracer).all():
        raise ValueError('saved endpoint-A initial tracer coordinates are invalid')

    observed = np.zeros((len(SHELL_EDGES) - 1, 3), dtype=np.float64)
    for row in profile['initial']['radial_by_population']:
        lo = float(row['radius_lower_cMpc_h'])
        if SHELL_EDGES[0] <= lo < SHELL_EDGES[-1] and row['population'] < 3:
            ibin = int(np.flatnonzero(SHELL_EDGES == lo)[0])
            observed[ibin, row['population']] = row['observed_count']
    bright_total = observed.sum(axis=1)
    if np.any(bright_total <= 0.0):
        raise ValueError('one or more selected training shells have no bright objects')
    observed_share = observed[:, 0] / bright_total

    with np.load(SOURCE, allow_pickle=False) as data:
        radial = data['radial_table'].copy()
        modulus = data['modulus_table'].copy()
        redshift = data['redshift_table'].copy()
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    radii = np.concatenate([
        (lo + hi) / 2.0 + (hi - lo) / 2.0 * nodes
        for lo, hi in zip(SHELL_EDGES[:-1], SHELL_EDGES[1:])
    ])
    radial_weights = np.concatenate([
        weights * (hi - lo) / 2.0
        * ((lo + hi) / 2.0 + (hi - lo) / 2.0 * nodes) ** 2
        for lo, hi in zip(SHELL_EDGES[:-1], SHELL_EDGES[1:])
    ]).reshape(len(SHELL_EDGES) - 1, QUADRATURE_ORDER)
    true_modulus = jnp.asarray(np.interp(radii, radial, modulus))
    true_redshift = jnp.asarray(np.interp(radii, radial, redshift))
    radial_weights_j = jnp.asarray(radial_weights)

    @jax.jit
    def one_share(alpha_white, mstar_white):
        mstar = -23.28 + 0.2 * mstar_white
        alpha = -1.0 + 0.06 * jnp.exp(0.5 * alpha_white)
        transfer = source_mark_transfer(
            true_modulus, true_modulus, true_redshift, true_redshift,
            mstar=mstar, alpha=alpha)
        lf = intrinsic_lf_reference_weights(
            mstar=mstar, alpha=alpha, reference_interval=(-25.0, -21.0))
        population_rate = jnp.sum(transfer * lf[None, :, None], axis=1)
        shell_rate = jnp.sum(
            population_rate.reshape(6, len(SHELL_EDGES) - 1, QUADRATURE_ORDER)
            * radial_weights_j[None, :, :], axis=-1)
        return shell_rate[0] / jnp.sum(shell_rate[:3], axis=0)

    parameter_pairs = np.asarray(
        [(a, m) for a in GRID for m in GRID], dtype=np.float64)
    predicted = np.asarray(jax.jit(jax.vmap(one_share))(
        jnp.asarray(parameter_pairs[:, 0]), jnp.asarray(parameter_pairs[:, 1])))
    if predicted.shape != (len(parameter_pairs), len(observed_share)):
        raise AssertionError('unexpected transfer-grid output shape')
    if not np.isfinite(predicted).all() or np.any((predicted <= 0.0) | (predicted >= 1.0)):
        raise FloatingPointError('non-finite or invalid predicted bright-population share')

    def binomial_deviance(shares):
        shares = np.clip(shares, 1e-12, 1.0 - 1e-12)
        return 2.0 * float(np.sum(bright_total * (
            observed_share * np.log(observed_share / shares)
            + (1.0 - observed_share) * np.log((1.0 - observed_share) / (1.0 - shares))
        )))

    deviances = np.asarray([binomial_deviance(row) for row in predicted])
    prior_penalty = parameter_pairs[:, 0] ** 2 + parameter_pairs[:, 1] ** 2
    best_raw = int(np.argmin(deviances))
    best_regularized = int(np.argmin(deviances + prior_penalty))
    baseline = np.asarray(one_share(jnp.asarray(tracer[7]), jnp.asarray(tracer[8])))

    report = dict(
        classification='FIELD_FREE_TRAINING_TRANSFER_SHAPE_DIAGNOSTIC_NOT_CALIBRATION',
        data_source=str(PROFILE), source_geometry=str(SOURCE),
        observed_data='training counts only; populations 0-2; radial shells 36-96 cMpc/h',
        field='uniform density; zero coherent/stochastic velocity; no angular mask',
        transfer='existing source_mark_transfer with true=observed modulus/redshift and intrinsic LF reference weights',
        shell_edges_cMpc_h=SHELL_EDGES.tolist(),
        quadrature=dict(rule='Gauss-Legendre in radius, weighted by r^2', order=QUADRATURE_ORDER),
        grid=dict(alpha_white=[float(GRID[0]), float(GRID[-1])],
                  mstar_white=[float(GRID[0]), float(GRID[-1])],
                  step=float(GRID[1] - GRID[0]), points_per_axis=len(GRID)),
        saved_initial_tracer_coordinates=tracer.tolist(),
        saved_initial_mstar_white=float(tracer[8]),
        saved_initial_alpha_white=float(tracer[7]),
        observed_bright_population0_share=observed_share.tolist(),
        uniform_transfer_share_at_saved_initial=baseline.tolist(),
        best_grid_binomial_deviance=dict(
            value=float(deviances[best_raw]),
            alpha_white=float(parameter_pairs[best_raw, 0]),
            mstar_white=float(parameter_pairs[best_raw, 1]),
            predicted_share=predicted[best_raw].tolist()),
        best_grid_deviance_plus_standard_gaussian_prior=dict(
            value=float(deviances[best_regularized] + prior_penalty[best_regularized]),
            binomial_deviance=float(deviances[best_regularized]),
            alpha_white=float(parameter_pairs[best_regularized, 0]),
            mstar_white=float(parameter_pairs[best_regularized, 1]),
            prior_nll=float(0.5 * prior_penalty[best_regularized]),
            predicted_share=predicted[best_regularized].tolist()),
        native_truth_ids_read=False, heldout_outcome_values_read=False,
        field_fitted=False, production_law_changed=False,
        interpretation='This tests only whether the two-parameter LF/mark transfer shape can represent these five training bright-share summaries under a uniform field. Luminosity-bias coordinates are unidentifiable here because normalized rho^beta equals one at rho=1. A good fit is not actual-field fit, calibration, posterior evidence, or LG identification.',
        Q_GOAL='diagnose one R2 population-split component upstream of same-new-field MW/M31/M33 inference',
        Q_LEAN='one uniform-field transfer calculation and a 61x61 two-coordinate grid; no PMWD, adjoint, chain, mock, heldout outcome or law edit',
        MW_M31='role-ambiguous; future observables must constrain both roles on the same NEW field',
        M33='unresolved; future observables must constrain it on the same NEW field at <=0.3 cMpc/h')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
