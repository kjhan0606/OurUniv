"""Check whether the fixed observed-ray/no-wrap bound covers all training links."""
import json
import os
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_linked_fp_sparse_train import (
    FP, load_train_singletons, select_training_single_mark_links,
)
from cf4_r2_native_to_count_cells import native_moments_to_count_cells

BASE=Path('/gpfs/kjhan/CF4/z0_density')
MIXTURE=BASE/'r2_raw_source_mixture_v1/raw_source_mixture.npz'
STATE=BASE/'r2_raw_joint_pilot_v1/accepted_present_state.npz'
SPLIT=BASE/'r2_sky_closed_split_v6/split.npz'


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    started=time.monotonic()
    options,point,_=load_train_singletons(SPLIT)
    with np.load(FP,allow_pickle=False) as f:
        options=select_training_single_mark_links(
            options,f['membership_state'].astype(str),include_grouped=True)
        chosen=[item for p in range(6) for item in options
                if point['population'][item[2]]==p]
        pgc=np.asarray([f['PGC'][item[3]] for item in chosen])
        direction=np.asarray([f['directions'][item[3]] for item in chosen],dtype=np.float64)
    with np.load(MIXTURE,allow_pickle=False) as f:
        np.testing.assert_array_equal(pgc,f['PGC'])
        radius=np.asarray(f['observed_radius'],dtype=np.float64)
    with np.load(SPLIT,allow_pickle=False) as f:
        keys=f['train_keys']
    training=np.isin(point['population'].astype(np.int64)*128**3+point['flat_cell'],keys)
    if len(chosen)!=1414 or int(training.sum())!=47121:
        raise ValueError('frozen training cohort ownership changed')
    if direction.shape!=(len(chosen),3) or not np.isfinite(direction).all():
        raise ValueError('training FP direction registration changed')
    direction_norm_error=float(np.max(np.abs(np.linalg.norm(direction,axis=1)-1.)))
    if direction_norm_error>2e-12:raise ValueError('FP source direction is not unit-normalized')

    with np.load(STATE,allow_pickle=False) as f:
        rho,velocity,variance=map(jnp.asarray,(f['rho'],f['mean_velocity_km_s'],
            f['physical_velocity_variance_km2_s2']))
    rho,velocity,variance=jax.jit(native_moments_to_count_cells,static_argnums=3)(
        rho,velocity,variance,384.)
    velocity=jnp.moveaxis(velocity,0,-1).reshape(-1,3)
    variance=jnp.moveaxis(variance,0,-1).reshape(-1,3)
    vmax=float(jnp.max(jnp.linalg.norm(velocity,axis=1)))
    variance_max=float(jnp.max(variance))
    sigma_bound=float(jnp.sqrt(30.**2+.5**2*variance_max))
    displacement_bound=.01*(vmax+8*sigma_bound)
    margin=192.-radius-displacement_bound
    safe=margin>0.
    result=dict(status=('ALL_TRAINING_LINKS_WITHIN_NO_WRAP_BOUND' if np.all(safe)
                        else 'NO_WRAP_BOUND_NOT_VALID_FOR_ALL_TRAINING_LINKS'),
        job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        training_points=int(training.sum()),training_FP_links=len(chosen),heldout_scored=False,
        PM_evolutions=0,observed_ray_factors_scored=0,
        bound=dict(box_half_cMpc_h=192.,conversion_h_over_H0=.01,
            maximum_count_grid_speed_km_s=vmax,maximum_diagonal_variance_km2_s2=variance_max,
            mixture_sigma_upper_bound_km_s=sigma_bound,eight_sigma_plus_coherent_shift_cMpc_h=displacement_bound),
        margin_cMpc_h=dict(minimum=float(np.min(margin)),p05=float(np.quantile(margin,.05)),
            median=float(np.median(margin)),p95=float(np.quantile(margin,.95)),maximum=float(np.max(margin))),
        positive_margin_rows=int(safe.sum()),nonpositive_margin_rows=int((~safe).sum()),
        nonpositive_examples=[dict(PGC=int(pgc[i]),observed_radius_cMpc_h=float(radius[i]),
            margin_cMpc_h=float(margin[i])) for i in np.flatnonzero(~safe)[:20]],
        FP_direction_norm_max_error=direction_norm_error,
        interpretation='A positive margin proves the stated fixed-ray 8-sigma support cannot reach a periodic face for that row under this saved field; a nonpositive margin means the bound is inconclusive, not that the row has zero physical support.',
        R2_complete=False,posterior=False)
    result['seconds']=time.monotonic()-started
    result['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
