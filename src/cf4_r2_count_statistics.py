"""Exact Poisson readout: occupied-key intensities AND full exposed integral.

Empty exposed cells are NOT discarded. Their contribution survives in the
last scalar. This linear projection commutes with source/volume/LOS sums.
"""
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln


def count_statistic_layout(grid_size,keys,exposure=None):
    """Static observed-data layout; no field-dependent selection or clipping."""
    keys=np.asarray(keys)
    if grid_size<3 or keys.ndim!=1 or not np.issubdtype(keys.dtype,np.integer):
        raise ValueError('integer keys on a TSC grid required')
    size=6*grid_size**3
    if np.any(keys<0) or np.any(keys>=size) or len(np.unique(keys))!=len(keys):
        raise ValueError('unique in-range observed keys required')
    mask=np.ones(size,dtype=bool) if exposure is None else np.asarray(exposure,dtype=bool).reshape(-1)
    if mask.size not in (grid_size**3,size):raise ValueError('spatial or population-specific exposure required')
    mask=np.tile(mask,size//mask.size)
    if not np.all(mask[keys]):raise ValueError('observed keys must lie in training exposure')
    lookup=np.full(size,-1,dtype=np.int32);lookup[keys]=np.arange(len(keys),dtype=np.int32)
    return dict(lookup=jnp.asarray(lookup),exposure=jnp.asarray(mask))


def tsc_count_statistics(positions,masses,population,grid_size,box_size_cMpc_h,
                         layout,statistic_size):
    """Same periodic cell-centred27-point TSC as the full-grid deposit.

    The last output is the exposed total, all others are occupied-key values.
    Lookup misses scatter OUTSIDE the output, not into that final scalar.
    """
    if not 0<=population<6 or grid_size<3 or statistic_size<1:
        raise ValueError('valid population/TSC/output geometry required')
    cell=(positions%box_size_cMpc_h)/(box_size_cMpc_h/grid_size)-.5
    nearest=jnp.floor(cell+.5).astype(jnp.int32);offset=cell-nearest
    def weights(x):return .5*(.5-x)**2,.75-x*x,.5*(.5+x)**2
    wx,wy,wz=(weights(offset[:,axis]) for axis in range(3))
    result=jnp.zeros(statistic_size,dtype=masses.dtype)
    integral=jnp.array(0.,dtype=masses.dtype)
    for ix,dx in enumerate((-1,0,1)):
        for iy,dy in enumerate((-1,0,1)):
            for iz,dz in enumerate((-1,0,1)):
                x=(nearest[:,0]+dx)%grid_size;y=(nearest[:,1]+dy)%grid_size;z=(nearest[:,2]+dz)%grid_size
                flat=((population*grid_size+x)*grid_size+y)*grid_size+z
                contribution=masses*wx[ix]*wy[iy]*wz[iz]
                row=layout['lookup'][flat]
                destination=jnp.where(row>=0,row,statistic_size)
                result=result.at[destination].add(contribution,mode='drop')
                integral+=jnp.sum(contribution*layout['exposure'][flat])
    return result.at[-1].add(integral)


def poisson_from_statistics(statistics,counts):
    if statistics.shape!=(len(counts)+1,):raise ValueError('occupied intensities plus one integral required')
    return jnp.sum(counts*jnp.log(statistics[:-1])-gammaln(counts+1.))-statistics[-1]
