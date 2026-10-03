"""Partial source-mark likelihood on the same live PM density/velocity state.

Native particle_grid/PMWD scatter output is NODE centred, origin i*dx, NOT
(i+.5)*dx. Geometry carries no cached state. Selection and velocity covariance
remain explicit development assumptions; this is not the full count/mark law.
"""
import jax.numpy as jnp
from cf4_z0_physical_field import read_centred
from cf4_r2_fp_group_marginal import selected_group_logweights, joint_redshift_logkernel
from cf4_r2_joint_calibration import group_scores


def bind_state_geometry(rho, velocity, geometry, *, box=384., origin_fraction=0.,
                        selected_bias=1.):
    """Recompute BOTH density measure and redshift kernel from this state.

    origin_fraction=.5 exists solely for the recorded historical-error
    comparison; production of new native PM readouts must use0. No remap,
    observer motion, density floor or velocity rescaling is applied.
    """
    d,dirs,zcos = (geometry[k] for k in ('distance','directions','zcos'))
    pos = box/2 + dirs[:,None,:]*d[:,:,None]
    density = read_centred(rho,pos,box,origin_fraction)
    radial = sum(read_centred(velocity[k],pos,box,origin_fraction)*dirs[:,k,None]
                 for k in range(3))
    logw = selected_group_logweights(d,geometry['quadrature_weight'],density,
                                    selected_bias,jnp.zeros_like(d))
    redshift = joint_redshift_logkernel(299792.458*zcos+(1+zcos)*radial,
                                      zcos,geometry['redshift_sufficient'])
    return dict(geometry,log_distance_weight=logw,redshift_logkernel=redshift)


def live_group_scores(rho, velocity, parameters, geometry, *, box=384.):
    return group_scores(parameters,bind_state_geometry(rho,velocity,geometry,box=box))


def joint_white_logdensity(white_ic, white_nuisance, rho, velocity, geometry, prior_sd,
                          *, box=384.):
    scores = live_group_scores(rho,velocity,white_nuisance*prior_sd,geometry,box=box)
    train = jnp.where(geometry['group_holdout'],0.,scores).sum()
    return train-.5*(jnp.vdot(white_ic,white_ic)+jnp.vdot(white_nuisance,white_nuisance))
