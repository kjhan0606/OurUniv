"""Count-intensity forward map for a retained present-field state.

This is the same N128-observation-grid count operator used by
``ResolutionObservationTarget``. It exposes the intensity (rather than its
training-data likelihood) so a post-fit evaluator can score a disjoint frozen
holdout with the unchanged count law. It does not make a state a posterior
draw or authorize reading the holdout before fitting/evaluation is ready.
"""

import numpy as np
import jax.numpy as jnp

from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_raw_volume_target import tracer_geometry, tracer_masses


def count_intensity_from_field(density, mean_velocity_km_s, tracer_white,
                               source, geometry, *, volume_order,
                               source_chunk_size=2097152):
    """Predict the complete six-population 2M++ intensity for one field.

    ``density`` is native cell-centered mass density and velocity is in the
    native ``(3,n,n,n)`` convention. ``source`` is the frozen source geometry
    at that same resolution; ``geometry`` retains the original observed-grid
    and survey-selection contract. The source-cell volume factor and
    GL2/GL4/8 settings intentionally match the active conditional-v6 target.
    """
    density = jnp.asarray(density)
    velocity = jnp.asarray(mean_velocity_km_s)
    tracer = jnp.asarray(tracer_white)
    if density.ndim != 3 or len(set(density.shape)) != 1 or density.shape[0] < 1:
        raise ValueError('density must be a nonempty cubic native grid')
    n = int(density.shape[0])
    if velocity.shape != (3, n, n, n):
        raise ValueError('mean velocity must align with the native cubic field')
    if tracer.shape != (9,):
        raise ValueError('the frozen count operator requires nine tracer coordinates')
    if volume_order not in (1, 2, 4) or source_chunk_size < 1:
        raise ValueError('unsupported source-volume order or chunk size')
    if 'box_size_cMpc_h' not in geometry or 'grid_size' not in geometry:
        raise ValueError('frozen count geometry is incomplete')
    if geometry.get('target_population') is not None:
        raise ValueError('held-out prediction requires the complete six-population intensity')
    box = float(geometry['box_size_cMpc_h'])
    if not np.isfinite(box) or box <= 0.:
        raise ValueError('box size must be finite and positive')
    grid_size = int(geometry['grid_size'])
    if grid_size < 1:
        raise ValueError('observed count grid size must be positive')
    positions = jnp.asarray(source['positions'])
    angular = jnp.asarray(source['angular'])
    if (positions.shape != (n ** 3, 3)
            or angular.shape != (2, n ** 3)):
        raise ValueError('source geometry must align with every native source cell')

    mass_density, cell_velocity = native_mass_momentum_to_count_cells(
        density, velocity, box)
    cell_velocity = jnp.moveaxis(cell_velocity, 0, -1).reshape(-1, 3)
    spacing = box / n
    intrinsic = tracer_masses(mass_density, tracer) * (spacing / 3.) ** 3
    intensity = predict_chunked_volume_intensity(
        positions, cell_velocity, intrinsic, angular,
        source_chunk_size=source_chunk_size,
        source_spacing=spacing,
        volume_order=volume_order,
        **tracer_geometry(tracer, geometry),
        order=4,
        segments=8,
    )
    if tuple(intensity.shape) != (6, grid_size, grid_size, grid_size):
        raise ValueError('forward operator did not return the complete six-population grid')
    return intensity
