"""Streaming Eulerian posterior moments, including repeated rejected states.

Call once per retained Markov-chain state, NOT once per accepted proposal.
Velocity moments condition on nonempty cells; occupancy is reported separately.
Physical within-cell dispersion is not posterior uncertainty or MC error.
"""
import numpy as np


class PresentMomentAccumulator:
    def __init__(self,shape):
        shape=tuple(shape)
        if len(shape)!=3 or len(set(shape))!=1 or min(shape)<1:
            raise ValueError('native cubic field shape required')
        self.shape=shape;self.n=0
        self.rho_mean=np.zeros(shape,dtype=np.float64)
        self.rho_m2=np.zeros(shape,dtype=np.float64)
        self.occupied=np.zeros(shape,dtype=np.int32)
        self.velocity_mean=np.zeros((3,)+shape,dtype=np.float64)
        self.velocity_m2=np.zeros((3,)+shape,dtype=np.float64)
        self.physical_variance_mean=np.zeros((3,)+shape,dtype=np.float64)

    def update(self,field):
        rho=np.asarray(field['rho']);v=np.asarray(field['mean_velocity_km_s'])
        physical=np.asarray(field['physical_velocity_variance_km2_s2'])
        valid=np.asarray(field['velocity_valid'],dtype=bool)
        if rho.shape!=self.shape or valid.shape!=self.shape or v.shape!=(3,)+self.shape or physical.shape!=v.shape:
            raise ValueError('aligned native field moments required')
        if not all(np.isfinite(x).all() for x in (rho,v,physical)) or np.any(rho<0) or np.any(physical<0):
            raise ValueError('finite nonnegative density and physical variance required')
        if not np.array_equal(valid,rho>0):raise ValueError('velocity validity must match nonempty native cells')
        # Validate before any state mutation.
        self.n+=1
        delta=rho-self.rho_mean
        self.rho_mean+=delta/self.n
        self.rho_m2+=delta*(rho-self.rho_mean)
        self.occupied+=valid
        inverse=np.divide(1.,self.occupied,out=np.zeros(self.shape),where=self.occupied>0)*valid
        for axis in range(3):
            delta=v[axis]-self.velocity_mean[axis]
            self.velocity_mean[axis]+=delta*inverse
            self.velocity_m2[axis]+=delta*(v[axis]-self.velocity_mean[axis])*valid
            self.physical_variance_mean[axis]+=(physical[axis]-self.physical_variance_mean[axis])*inverse

    def arrays(self):
        """Sample variance, not MC variance; undefined cells remain NaN."""
        if self.n<1:raise ValueError('no retained states')
        rho_var=self.rho_m2/(self.n-1) if self.n>1 else np.full(self.shape,np.nan)
        vvar=np.divide(self.velocity_m2,self.occupied[None]-1,
            out=np.full_like(self.velocity_m2,np.nan),where=self.occupied[None]>1)
        return dict(retained_states=np.array(self.n),rho_mean=self.rho_mean.copy(),
            rho_posterior_variance=rho_var,occupied_states=self.occupied.copy(),
            occupancy_fraction=self.occupied/self.n,
            conditional_mean_velocity_km_s=np.where(self.occupied[None]>0,self.velocity_mean,np.nan),
            conditional_velocity_posterior_variance_km2_s2=vvar,
            conditional_mean_physical_velocity_variance_km2_s2=np.where(
                self.occupied[None]>0,self.physical_variance_mean,np.nan),
            native_mesh_origin_fraction=np.array(0.),posterior_MC_error_estimated=np.array(False))

    def checkpoint_arrays(self):
        """Sufficient accumulator state for continuing, without raw field draws."""
        return dict(n=np.array(self.n),rho_mean=self.rho_mean,rho_m2=self.rho_m2,
            occupied=self.occupied,velocity_mean=self.velocity_mean,velocity_m2=self.velocity_m2,
            physical_variance_mean=self.physical_variance_mean)

    @classmethod
    def from_checkpoint(cls,state):
        result=cls(np.asarray(state['rho_mean']).shape)
        n=np.asarray(state['n'])
        if n.shape or not np.issubdtype(n.dtype,np.integer) or int(n)<0:
            raise ValueError('nonnegative scalar integer count required')
        result.n=int(n)
        for name in ('rho_mean','rho_m2','occupied','velocity_mean','velocity_m2','physical_variance_mean'):
            value=np.asarray(state[name]);old=getattr(result,name)
            if value.shape!=old.shape or not np.isfinite(value).all():raise ValueError('invalid moment checkpoint')
            if name=='occupied' and (not np.issubdtype(value.dtype,np.integer) or np.any(value<0) or np.any(value>result.n)):
                raise ValueError('invalid occupancy count')
            setattr(result,name,value.astype(old.dtype,copy=True))
        return result
