"""Finite-volume z=0 parameterization; not a fine-field prior or reconstruction."""
import numpy as np


def cell_axis(n,dx,lower_face):
    return lower_face+(np.arange(n)+.5)*dx


def child_origin(parent_origin,ratio):
    if int(ratio)!=ratio or ratio<1:
        raise ValueError("positive integer ratio required")
    return ratio*(parent_origin-.5)+.5


def _blocks(field,ratio,xp):
    if field.ndim!=3 or len(set(field.shape))!=1 or field.shape[0]%ratio:
        raise ValueError("cubic grid divisible by ratio required")
    n=field.shape[0]//ratio
    return xp.reshape(field,(n,ratio,n,ratio,n,ratio))


def restrict(rho,velocity,ratio,*,xp=np):
    """Density volume average; velocity from conserved momentum, not mean(v)."""
    if velocity.shape!=(3,)+rho.shape:
        raise ValueError("component-first mean velocity required")
    mass=xp.mean(_blocks(rho,ratio,xp),axis=(1,3,5))
    momentum=xp.stack([xp.mean(_blocks(rho*v,ratio,xp),axis=(1,3,5)) for v in velocity])
    return mass,momentum/mass[None]


def refine(rho_parent,velocity_parent,log_detail,velocity_detail,ratio,*,xp=np):
    """Child density via within-parent softmax, velocity via momentum projection.

    All fine detail coordinates are model parameters, NOT measurements. Caller
    must specify their physical prior and actual observational likelihood.
    Supports NumPy and jax.numpy without a separate implementation.
    """
    n=rho_parent.shape[0]
    if rho_parent.shape!=(n,)*3 or velocity_parent.shape!=(3,n,n,n):
        raise ValueError("invalid parent field shapes")
    if log_detail.shape!=(n*ratio,)*3 or velocity_detail.shape!=(3,)+log_detail.shape:
        raise ValueError("invalid child detail shapes")
    blocks=_blocks(log_detail,ratio,xp)
    exp=xp.exp(blocks-xp.max(blocks,axis=(1,3,5),keepdims=True))
    weights=exp/xp.mean(exp,axis=(1,3,5),keepdims=True)
    rho=(weights*rho_parent[:,None,:,None,:,None]).reshape(log_detail.shape)
    velocity=[]
    for a in range(3):
        detail=_blocks(velocity_detail[a],ratio,xp)
        weighted_mean=xp.mean(weights*detail,axis=(1,3,5),keepdims=True)
        velocity.append((velocity_parent[a,:,None,:,None,:,None]+detail-weighted_mean).reshape(log_detail.shape))
    return rho,xp.stack(velocity)
