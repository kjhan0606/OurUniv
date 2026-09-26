"""Conditional distance marks with shared group distance and global zero point.

The caller owns the joint redshift law and the selected-group distance prior.
This module conditions on that redshift vector; it never exports its marginal
as another independent redshift likelihood. Covariance inputs need physical
calibration before production. Catalogue identities label observations only.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from cf4_r2_fp_distance import fp_log_likelihood_ratio


def selected_group_logweights(distance, quadrature_weight, density_ratio,
                             tracer_bias, log_group_inclusion):
    """Selected-group radial measure from the SAME state's matter density.

    w = dd * d^2 * rho^b * P(group included | d, F, covariates).
    This is a caller-specified power-law tracer model, not calibrated bias.
    Inclusion means group/sample inclusion, NOT the FP fn correction already
    present in source distance summaries. Use w in BOTH conditional integrals.
    A distance-independent amplitude cancels; spatial shape generally does not.
    Zero density/inclusion remains zero support, without a numerical floor.
    Requires positive distance, weights, bias; nonnegative finite density;
    inclusion log probability <=0. Invalid inputs return NaN, not a fit repair.
    """
    positive = density_ratio > 0
    log_rho = jnp.log(jnp.where(positive, density_ratio, 1.))
    log_intensity = jnp.where(positive, tracer_bias*log_rho, -jnp.inf)
    valid = ((distance > 0) & jnp.isfinite(distance)
             & (quadrature_weight > 0) & jnp.isfinite(quadrature_weight)
             & (density_ratio >= 0) & jnp.isfinite(density_ratio)
             & (tracer_bias > 0) & jnp.isfinite(tracer_bias)
             & (log_group_inclusion <= 0))
    return jnp.where(valid, jnp.log(quadrature_weight)+2*jnp.log(distance)
                     +log_intensity+log_group_inclusion, jnp.nan)


def redshift_sufficient(observed_cz, covariance_v, redshift_ids):
    """Compress a supplied correlated Gaussian with common model mean.

    covariance_v is in peculiar-(km/s)^2. The model's observed-cz covariance
    is (1+zcos)^2 covariance_v. Duplicate datum IDs are forbidden. A catalogue
    group mean and member measurements may be distinct but correlated rows;
    the supplied covariance must describe that correlation, not ignore it.
    """
    y, c = np.asarray(observed_cz, float), np.asarray(covariance_v, float)
    if y.ndim != 1 or not y.size or c.shape != (y.size, y.size):
        raise ValueError('invalid redshift vector/covariance shape')
    if len(redshift_ids) != y.size or len(set(redshift_ids)) != y.size:
        raise ValueError('duplicate or mismatched redshift datum IDs')
    if not np.isfinite(y).all() or not np.isfinite(c).all() or not np.allclose(c, c.T):
        raise ValueError('nonfinite or asymmetric covariance')
    chol = np.linalg.cholesky(c)
    one_white = np.linalg.solve(chol, np.ones_like(y))
    y_white = np.linalg.solve(chol, y)
    precision = one_white @ one_white
    centre = (one_white @ y_white) / precision
    residual = y_white - centre * one_white
    return np.array([precision, centre, residual @ residual,
                     2*np.log(np.diag(chol)).sum(), y.size])


def joint_redshift_logkernel(predicted_cz, zcos, sufficient):
    """Full joint-redshift kernel at each distance, including its Jacobian."""
    p, centre, q, logdet, n = [sufficient[..., k, None] for k in range(5)]
    scale = 1 + zcos
    return -.5*((q+p*(centre-predicted_cz)**2)/scale**2 + logdet
                + 2*n*jnp.log(scale) + n*jnp.log(2*jnp.pi))


def conditional_group_scores(distance, log_distance_weight, redshift_logkernel,
                             row_group, dz_row, mean, std, alpha, zero_nodes,
                             extra_log_marks=None):
    """Return log conditional mark factor per zero-point node and group.

    distance/weights/kernel are (groups, quadrature). weights include dd and
    the explicit selected-group prior, not another FP measurement correction.
    Product of member distance likelihoods is INSIDE the group integral.
    The fixed reference eta=0 only cancels source-data constants.
    Optional extra_log_marks (groups, quadrature) adds OTHER observations
    conditional on this SAME distance and caller's shared nuisance values.
    Do not supply a separately marginalized factor or duplicated FP/BGc data.
    """
    groups = distance.shape[0]
    base = log_distance_weight + redshift_logkernel
    denominator = logsumexp(base, axis=-1)
    eta = jnp.log10(dz_row[:, None] / distance[row_group])
    def at_zero(b):
        member = fp_log_likelihood_ratio(eta+b, 0., mean[:, None],
                                        std[:, None], alpha[:, None])
        marks = jax.ops.segment_sum(member, row_group, num_segments=groups)
        if extra_log_marks is not None:
            marks = marks+extra_log_marks
        return logsumexp(base + marks, axis=-1) - denominator
    return jax.lax.map(at_zero, zero_nodes)


def shared_zero_logfactor(scores, log_zero_weights, group_mask):
    """Integrate ONE global zero point after multiplying all selected groups."""
    logw = log_zero_weights - logsumexp(log_zero_weights)
    return logsumexp(logw + jnp.sum(jnp.where(group_mask[None, :], scores, 0.), axis=1))


def _supported_logsumexp(values, axis):
    """Preserve zero support and a zero tangent when all terms are -inf."""
    maximum = jnp.max(values, axis=axis, keepdims=True)
    maximum = jnp.where(jnp.isneginf(maximum), 0., maximum)
    total = jnp.sum(jnp.exp(values-maximum), axis=axis, keepdims=True)
    result = maximum + jnp.log(jnp.where(total == 0, 1., total))
    return jnp.squeeze(jnp.where(total == 0, -jnp.inf, result), axis=axis)


def _segment_logsumexp(values, row_group, groups):
    maximum = jax.ops.segment_max(values, row_group, num_segments=groups)
    maximum = jnp.where(jnp.isneginf(maximum), 0., maximum)
    total = jax.ops.segment_sum(jnp.exp(values-maximum[row_group]), row_group,
                                num_segments=groups)
    result = maximum + jnp.log(jnp.where(total == 0, 1., total))
    return jnp.where(total == 0, -jnp.inf, result)


def latent_central_logmark(log_satellite, log_central, row_group,
                           log_none_weight, log_central_weight):
    """Marginalize no observed central OR exactly one observed central.

    Marks are (rows, nodes). Weights are (groups,) and (rows,), respectively;
    they are normalized jointly WITHIN each group, not independently per row.
    Caller must supply nonnegative weights (log zero = -inf), at least one
    positive hypothesis per group, and valid consecutive group indices.
    Weights describe the SELECTED observed subset conditional on the supplied
    redshifts/other covariates, not an unselected central fraction. A common
    source-data reference must be used for BOTH role likelihood ratios.

    The single-host approximation is explicit: an observed FoF group need
    not obey it (interlopers/merged hosts are NOT implemented). No truth IDs
    or role labels are used. This is a model interface, not calibrated roles.
    """
    groups = log_none_weight.shape[0]
    supported_sat = ~jnp.isneginf(log_satellite)
    safe_sat = jnp.where(supported_sat, log_satellite, 0.)
    summed = jax.ops.segment_sum(safe_sat, row_group, num_segments=groups)
    missing = jax.ops.segment_sum((~supported_sat).astype(jnp.int32), row_group,
                                  num_segments=groups)
    no_central = jnp.where(missing == 0, summed, -jnp.inf) + log_none_weight[:, None]
    # A central branch can rescue ONE zero satellite factor. Never subtract
    # -inf from -inf to form the leave-one-out product.
    others_valid = missing[row_group] - (~supported_sat) == 0
    central = jnp.where(others_valid, summed[row_group]-safe_sat, -jnp.inf)
    central = central + log_central + log_central_weight[:, None]
    central_sum = _segment_logsumexp(central, row_group, groups)
    norm = _supported_logsumexp(jnp.stack((log_none_weight,
        _segment_logsumexp(log_central_weight, row_group, groups))), axis=0)
    return _supported_logsumexp(jnp.stack((no_central, central_sum)), axis=0) - norm[:, None]


def conditional_latent_group_scores(distance, log_distance_weight, redshift_logkernel,
                                    row_group, dz_row, mean, std, alpha, zero_nodes,
                                    log_none_weight, log_central_weight,
                                    central_offset, satellite_offset,
                                    group_offset_nodes, log_group_offset_weights,
                                    extra_log_marks=None):
    """Conditional group marks including latent roles and ONE shared offset.

    eta_pred + global_zero + group_offset + role_offset is evaluated against
    the supplied source measurement PDF. Offset/zero units are dex in eta.
    group_offset_nodes/weights specify a caller-provided common prior rule;
    offsets are independent ACROSS groups, shared WITHIN each group. Integrate
    the product of member factors, not each member separately. Global zero is
    still integrated only by shared_zero_logfactor after group multiplication.

    The SAME selected radial measure and redshift kernel enter numerator and
    denominator. This implementation assumes their kernels do not depend on
    the mark offset/role. A role-dependent FoG/selection law requires a joint
    numerator AND denominator, not just replacement of mark mixture weights.
    Weights/offsets are explicit nuisance inputs, NOT measurements inferred
    from mock truth. An extra group scatter may overlap source-fit uncertainty;
    fitting/calibration must resolve that, not automatically add variances.
    extra_log_marks is for non-FP observations of the SAME group distance,
    conditional on their nuisance values. The FP-specific group/role offsets
    must NOT also shift those independent distance indicators. Shared source
    calibration covariance still requires caller modelling, not a product
    of independently marginalized calibrations.
    """
    base = log_distance_weight + redshift_logkernel
    denominator = _supported_logsumexp(base, axis=-1)
    eta = jnp.log10(dz_row[:, None] / distance[row_group])
    loguw = log_group_offset_weights-logsumexp(log_group_offset_weights)

    @jax.checkpoint
    def at_zero(b):
        @jax.checkpoint
        def at_offset(u):
            common = eta+b+u
            sat = fp_log_likelihood_ratio(common+satellite_offset, 0., mean[:, None],
                                          std[:, None], alpha[:, None])
            cen = fp_log_likelihood_ratio(common+central_offset, 0., mean[:, None],
                                          std[:, None], alpha[:, None])
            marks = latent_central_logmark(sat, cen, row_group,
                                           log_none_weight, log_central_weight)
            if extra_log_marks is not None:
                marks = marks+extra_log_marks
            return _supported_logsumexp(base+marks, axis=-1)
        integrals = jax.lax.map(at_offset, group_offset_nodes)
        return _supported_logsumexp(integrals+loguw[:, None], axis=0)-denominator
    return jax.lax.map(at_zero, zero_nodes)


def nonfp_modulus_logmarks(predicted_modulus, row_group, observed_modulus,
                           error, method_index, method_offset, reference_modulus=35.):
    """Conditional Gaussian non-FP marks at the SAME caller-predicted distance.

    predicted_modulus is (groups, distance nodes), already in luminosity
    distance modulus with the caller's redshift/Doppler convention. Do not
    silently equate comoving Mpc/h with luminosity Mpc. method_offset is a
    SHARED nuisance vector, not one independent calibration per galaxy.
    This conditional independent-error approximation is NOT a calibrated
    source-fit covariance, and is not yet adopted for production inference.
    The fixed data reference is common to all models/offsets.
    """
    prediction = predicted_modulus[row_group]+method_offset[method_index, None]
    member = -.5*((observed_modulus[:, None]-prediction)**2
                   -(observed_modulus[:, None]-reference_modulus)**2)/error[:, None]**2
    return jax.ops.segment_sum(member, row_group, num_segments=predicted_modulus.shape[0])
