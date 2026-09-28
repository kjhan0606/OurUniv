"""Localize a saved-state FP restart discrepancy; no PM, fit or heldout."""
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from cf4_r2_linked_fp_sparse_train import load_train_singletons, FP, SOURCE
from cf4_r2_linked_singleton_target import partial_v6_count_singleton_parts
from cf4_r2_linked_singleton_jax import linked_singleton_logfactors_for_population
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_marked_tracer_jax import intrinsic_biased_source_masses, intrinsic_lf_bin_fractions
from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax

BASE = Path('/gpfs/kjhan/CF4/z0_density')
N, BOX, WIDTH = 128, 384., 8192


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
                  source_commit=os.environ['CF4_EXPECTED_COMMIT'], device=str(jax.devices()),
                  PM_evolutions=0, optimizer_steps=0, heldout_scored=False, R2_complete=False)
    def save():
        report['seconds'] = time.monotonic()-started
        (out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    paths = [BASE/'r2_v6_fixed_field_nuisance_v1/final_state.npz',
             BASE/'r2_v6_joint_secant_map_v2/initial_state.npz']
    states = []
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            states.append({k:f[k].copy() for k in
                ('white_ic','rho','velocity_km_s','tracer','white_fp_zero')})
    report['state_max_abs_differences'] = {
        k:float(np.max(np.abs(states[0][k]-states[1][k]))) for k in states[0]}
    options, point, fp = load_train_singletons(BASE/'r2_sky_closed_split_v6/split.npz')
    with np.load(FP, allow_pickle=False) as f:
        membership = f['membership_state'].astype(str)
    options = [o for o in options if membership[o[3]] in {
        'source_ungrouped_catalogue_present','source_ungrouped_catalogue_absent'}]
    assert len(options) == 429
    with np.load(SOURCE, allow_pickle=False) as f:
        source = {k:jnp.asarray(f[k]) for k in f.files}
    selected = {p:[o for o in options if point['population'][o[2]] == p] for p in range(6)}
    metadata, centers = {}, {}
    for p, batch in selected.items():
        voxels = np.array([np.unravel_index(int(point['flat_cell'][o[2]]),(N,)*3)
                           for o in batch], dtype=np.int32).reshape(-1,3)
        centers[p] = (voxels+.5)*(BOX/N)
        fi = np.array([o[4] for o in batch],dtype=np.int32)
        metadata[p] = dict(voxel_ijk=jnp.asarray(voxels),
            observed_radius_cMpc_h=jnp.array([point['radius_cMpc_h'][o[2]] for o in batch]),
            **{k:jnp.asarray(fp[k][fi]) for k in ('dz_row','eta_mean','eta_std','eta_alpha')})

    @jax.jit
    def shifted(rho, vel):
        _, v = native_mass_momentum_to_count_cells(rho, vel, BOX)
        return observer_centred_spherical_rsd_jax(source['positions'],
            jnp.moveaxis(v,0,-1).reshape(-1,3),jnp.full(3,192.),BOX,74.6,
            little_h=.746,scale_factor=1.)[0]

    @jax.jit
    def mark(rho, vel, tracer, zero, links):
        return partial_v6_count_singleton_parts(rho,vel,jnp.zeros(0),tracer,source,links,
            jnp.zeros(0,dtype=jnp.int32),jnp.zeros(0),jnp.zeros(6*N**3,dtype=bool),
            white_fp_zero=zero,count_integration='shell_cdf')[0][1]

    def per_row(rho, vel, tracer, zero, links):
        density, v = native_mass_momentum_to_count_cells(rho,vel,BOX)
        intrinsic = intrinsic_biased_source_masses(density,
            jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tracer[0],
            jnp.exp(.5*tracer[1:6]),mstar=-23.28+.2*tracer[8],
            alpha=-1+.06*jnp.exp(.5*tracer[7]),reference_interval=(-25.,-21.))
        geometry = dict(observer=jnp.full(3,192.),box_size_cMpc_h=BOX,hubble_km_s_Mpc=74.6,
            little_h=.746,radius_table_cMpc_h=source['radial_table'],
            modulus_table_h=source['modulus_table'],redshift_table=source['redshift_table'],
            grid_size=N,radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,
            mstar=-23.28+.2*tracer[8],alpha=-1+.06*jnp.exp(.5*tracer[7]))
        return jnp.concatenate([linked_singleton_logfactors_for_population(
            source['positions'],jnp.moveaxis(v,0,-1).reshape(-1,3),intrinsic,source['angular'],
            **links[p],population=p,sigma_los_km_s=100*jnp.exp(.5*tracer[6]),
            radial_geometry=geometry,fp_zero_dex=.004*zero)[0]
            for p in range(6) if len(selected[p])])
    rows_compiled = jax.jit(per_row)
    report['states'] = []
    for index, state in enumerate(states):
        args = [jnp.asarray(state[k]) for k in ('rho','velocity_km_s','tracer','white_fp_zero')]
        radius = 8*.01*100*np.exp(.5*state['tracer'][6])+np.sqrt(3)*1.5*BOX/N
        tree = cKDTree(np.asarray(shifted(*args[:2])) % BOX,boxsize=BOX)
        links = {}
        for p, batch in selected.items():
            neighborhoods = tree.query_ball_point(centers[p],radius,workers=1)
            ids = np.zeros((len(batch),WIDTH),dtype=np.int32)
            active = np.zeros_like(ids,dtype=bool)
            for i, neighbors in enumerate(neighborhoods):
                assert 0 < len(neighbors) <= WIDTH
                ids[i,:] = neighbors[0]; ids[i,:len(neighbors)] = neighbors
                active[i,:len(neighbors)] = True
            links[p] = dict(metadata[p],candidate_source_ids=jnp.asarray(ids),
                candidate_mask=jnp.asarray(active),association_logprob=jnp.zeros((len(batch),5,WIDTH)))
        value = float(mark(*args,links))
        rows = np.asarray(rows_compiled(*args,links))
        np.savez(out/f'rows_{index}.npz',factors=rows,
                 labels=np.array([o[0] for p in range(6) for o in selected[p]]))
        report['states'].append(dict(path=str(paths[index]),FP=value,per_row_sum=float(rows.sum())))
        save(); print(json.dumps(report['states'][-1]),flush=True)
    report['status'] = 'SAVED_STATE_FP_CHECK_COMPLETE_NOT_POSTERIOR'
    save()


if __name__ == '__main__':
    main()
