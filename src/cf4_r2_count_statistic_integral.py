"""The existing source/volume/LOS law projected to exact Poisson readouts.

Numerical shell nodes and LF/RSD operators are shared with the full-grid
reference. This output projection retains empty exposed-cell contributions.
The separate entrypoint avoids changing the pinned running full-grid pilot.
"""
from itertools import product
import numbers
import jax
import jax.numpy as jnp
from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax
from cf4_r2_marked_tracer_jax import source_mark_transfer
from cf4_r2_shell_cdf_count import _shell_intervals,_probability_nodes
from cf4_r2_count_statistics import tsc_count_statistics
from cf4_r2_raw_volume_target import volume_rule


def shell_count_statistics(positions,velocities,intrinsic,angular,layout,statistic_size,*,
        observer,box_size_cMpc_h,hubble_km_s_Mpc,little_h,radius_table_cMpc_h,
        modulus_table_h,redshift_table,grid_size,sigma_los_km_s,
        radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,mstar=-23.28,alpha=-.94,
        order=4,segments=8):
    if not 0<radial_min_cMpc_h<radial_max_cMpc_h<box_size_cMpc_h/2 or min(order,segments)<1:
        raise ValueError('valid disjoint observer shell and integration rule required')
    sigma=little_h*sigma_los_km_s/hubble_km_s_Mpc
    if isinstance(sigma,numbers.Real) and not 0<8*sigma<box_size_cMpc_h/2:
        raise ValueError('27-image law requires0<8sigma_radius<box/2; caller checks traced widths')
    shifted,_,direction=observer_centred_spherical_rsd_jax(positions,velocities,observer,
        box_size_cMpc_h,hubble_km_s_Mpc,little_h=little_h,scale_factor=1.)
    def radius(x):
        relative=(x-observer+box_size_cMpc_h/2)%box_size_cMpc_h-box_size_cMpc_h/2
        r2=jnp.sum(relative*relative,axis=1)
        return jnp.where(r2>0,jnp.sqrt(jnp.where(r2>0,r2,1.)),0.)
    mu=radius(shifted);rt=radius(positions)
    mt=jnp.interp(rt,radius_table_cMpc_h,modulus_table_h)
    zt=jnp.interp(rt,radius_table_cMpc_h,redshift_table)
    @jax.checkpoint
    def add_node(total,qw):
        q,w=qw;observed=(observer+q[:,None]*direction)%box_size_cMpc_h
        ro=radius(observed)
        transfer=source_mark_transfer(mt,jnp.interp(ro,radius_table_cMpc_h,modulus_table_h),
            zt,jnp.interp(ro,radius_table_cMpc_h,redshift_table),mstar=mstar,alpha=alpha)
        for p in range(6):
            mass=w*angular[p//3]*jnp.sum(transfer[p]*intrinsic,axis=0)
            total+=tsc_count_statistics(observed,mass,p,grid_size,box_size_cMpc_h,layout,statistic_size)
        return total,None
    @jax.checkpoint
    def image(total,center):
        def integrate(total):
            lo,hi,active=_shell_intervals(mu,direction,sigma,center,radial_min_cMpc_h,radial_max_cMpc_h,8.)
            width=jnp.where(active,hi-lo,0.)
            @jax.checkpoint
            def segment(total,index):
                branch=index//segments;fraction=(index%segments).astype(mu.dtype)/segments
                start=lo[branch]+fraction*width[branch];end=start+width[branch]/segments
                def quadrature(total):
                    q,w=_probability_nodes(mu,sigma,start[None],end[None],active[branch][None],order)
                    return jax.lax.scan(add_node,total,(q[0],w[0]))[0]
                return jax.lax.cond(jnp.any(active[branch]),quadrature,lambda x:x,total),None
            return jax.lax.cond(jnp.any(active),
                lambda x:jax.lax.scan(segment,x,jnp.arange(2*segments))[0],lambda x:x,total)
        possible=jnp.all(center==0)|(8*sigma>=box_size_cMpc_h/2-radial_max_cMpc_h)
        return jax.lax.cond(possible,integrate,lambda x:x,total),None
    images=jnp.asarray(list(product((-1,0,1),repeat=3)))*box_size_cMpc_h
    dtype=jnp.result_type(positions,velocities,intrinsic,angular,observer,sigma)
    return jax.lax.scan(image,jnp.zeros(statistic_size,dtype=dtype),images)[0]


def volume_count_statistics(positions,velocities,intrinsic,angular,layout,statistic_size,*,
                            source_spacing,volume_order=4,source_chunk_size=131072,**geometry):
    """Stream disjoint source chunks and volume nodes; no per-chunk bias fit."""
    count=len(positions)
    if count<1 or source_chunk_size<1 or intrinsic.shape!=(5,count) or angular.shape!=(2,count):
        raise ValueError('nonempty aligned source geometry required')
    offsets,weights=map(jnp.asarray,volume_rule(source_spacing,volume_order))
    nc=(count+source_chunk_size-1)//source_chunk_size
    ids=jnp.arange(nc*source_chunk_size).reshape(nc,source_chunk_size)
    active=ids<count;ids=jnp.minimum(ids,count-1)
    @jax.checkpoint
    def chunk(total,item):
        index,mask=item
        pos,vel,mass,sky=positions[index],velocities[index],intrinsic[:,index]*mask[None],angular[:,index]
        @jax.checkpoint
        def node(total,node_weight):
            offset,weight=node_weight
            return total+shell_count_statistics((pos+offset)%geometry['box_size_cMpc_h'],vel,mass*weight,sky,
                layout,statistic_size,**geometry),None
        return jax.lax.scan(node,total,(offsets,weights))[0],None
    dtype=jnp.result_type(positions,velocities,intrinsic,angular,offsets)
    return jax.lax.scan(chunk,jnp.zeros(statistic_size,dtype=dtype),(ids,active))[0]
