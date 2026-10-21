"""Declared coarsened-count and conditional-CF4-mark partial target.

The count and mark arrays are different observations; 2M++ member redshifts
are conditioned covariates inside the group kernel. Association and group
selection are fixed observed-design approximations, not calibrated factors.
"""
import jax.numpy as jnp
from cf4_r2_continuous_tracer import predict_continuous_intensity
from cf4_r2_rate_marginal import gamma_poisson_log_marginal_sparse
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_hierarchical_marks import hierarchical_group_scores


def count_and_mark_parts(rho_node,velocity_node,white_ic,white_hyper,white_group,
                         geometry,calibration_sd,exposure,keys,counts,rate_mean,
                         rate_shape,published_bias,fog,redshift,*,box=384.,hubble=74.6,h=.746):
    """Return count, conditional mark, and Gaussian-prior log factors.

    Fixed six-population RSD/bias parameters are only development inputs.
    Rate amplitudes are integrated, not fitted to the observed count totals.
    The count likelihood includes every eligible point once via its voxel.
    """
    density,velocity = native_mass_momentum_to_count_cells(rho_node,velocity_node,box)
    unit = predict_continuous_intensity(density,velocity,exposure,
        jnp.ones(6),published_bias,jnp.zeros(6),box=box,
        observer=jnp.full(3,box/2),hubble=hubble,little_h=h,
        origin_fraction=.5,sigma_fog=fog,sigma_redshift=redshift)
    count = gamma_poisson_log_marginal_sparse(unit,keys,counts,rate_mean,rate_shape)
    marks = hierarchical_group_scores(rho_node,velocity_node,white_hyper,
                                       white_group,geometry,calibration_sd,box=box)
    mark = jnp.where(geometry['group_holdout'],0.,marks).sum()
    prior = -.5*(jnp.vdot(white_ic,white_ic)+jnp.vdot(white_hyper,white_hyper)
                +jnp.vdot(white_group,white_group))
    return jnp.stack((count,mark,prior)),unit
