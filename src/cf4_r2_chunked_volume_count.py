"""Bound source-workspace memory without changing the count-volume integral."""
import jax
import jax.numpy as jnp
from cf4_r2_shell_cdf_count import predict_source_volume_intensity


def predict_chunked_volume_intensity(positions,velocities,intrinsic,angular,*,
                                     source_chunk_size,source_spacing,volume_order=4,**geometry):
    """Sum disjoint source chunks; caller normalized bias over the WHOLE box.

    No per-chunk renormalization, source omission, extra smoothing or target
    resolution change. Padding duplicates valid geometry with exactly zero
    mass. This is source streaming, not independent subvolume likelihoods.
    """
    count=len(positions)
    if count<1 or source_chunk_size<1:raise ValueError('positive source count/chunk size required')
    if velocities.shape!=positions.shape or intrinsic.shape!=(5,count) or angular.shape!=(2,count):
        raise ValueError('aligned global source geometry required')
    nc=(count+source_chunk_size-1)//source_chunk_size
    indices=jnp.arange(nc*source_chunk_size).reshape(nc,source_chunk_size)
    valid=indices<count;indices=jnp.minimum(indices,count-1)
    @jax.checkpoint
    def add(total,item):
        ids,mask=item
        piece=predict_source_volume_intensity(positions[ids],velocities[ids],
            intrinsic[:,ids]*mask[None],angular[:,ids],source_spacing=source_spacing,
            volume_order=volume_order,**geometry)
        return total+piece,None
    n=geometry['grid_size'];shape=() if geometry.get('target_population') is not None else (6,n,n,n)
    dtype=jnp.result_type(positions,velocities,intrinsic,angular,jnp.asarray(0.,dtype=jnp.float64))
    return jax.lax.scan(add,jnp.zeros(shape,dtype=dtype),(indices,valid))[0]
