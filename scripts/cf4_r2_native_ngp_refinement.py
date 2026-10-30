"""Fixed-training-endpoint NGP support refinement, not a model selection fit."""
import json
import os
from pathlib import Path
import resource
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from cf4_r2_native_mock import distance_tables, place_native_halves, observe
from cf4_r2_raw_volume_target import tracer_masses, tracer_geometry
from cf4_r2_shell_cdf_count import predict_source_volume_intensity

jax.config.update('jax_enable_x64', True)
BASE=Path('/gpfs/kjhan/CF4/z0_density')
SOURCE=BASE/'r2_native_rsd_mock_fit_20261002_v2'
OUT=Path(os.environ.get('CF4_R2_OUT_DIR',str(BASE/'r2_native_ngp_refinement_20261002_v1')))


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    endpoint=json.loads((SOURCE/'result.json').read_text())
    if not endpoint['stationarity_verified']:
        raise ValueError('stationary training endpoint required')
    OUT.mkdir(exist_ok=False)
    start=time.monotonic()
    with h5py.File(endpoint['source_matter'],'r') as f:
        moments=f['coarse'][:];h=float(f.attrs['h']);omega=float(f.attrs['Omega_m'])
    with np.load(SOURCE/'mock_observations.npz',allow_pickle=False) as f:
        keys=f['keys'];counts=f['counts'];train=f['train_exposure'];test=f['test_exposure']
    axis=(np.arange(50)+.5)*1.5
    positions=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    pos=jnp.asarray(place_native_halves(positions))
    vel=jnp.asarray(np.moveaxis(moments[1:4]/moments[0],0,-1).reshape(-1,3))
    rho=jnp.asarray(moments[0]/moments[0].mean())
    q=jnp.asarray(endpoint['final_coordinates'])
    intrinsic=tracer_masses(rho,q)/8.
    angular=jnp.ones((2,len(positions)))
    radius,z,modulus=distance_tables(omega)
    geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
        hubble_km_s_Mpc=100*h,little_h=h,grid_size=128,
        radius_table_cMpc_h=jnp.asarray(radius),redshift_table=jnp.asarray(z),
        modulus_table_h=jnp.asarray(modulus),radial_min_cMpc_h=5.,radial_max_cMpc_h=180.)
    report=dict(job_id=os.environ['SLURM_JOB_ID'],endpoint=str(SOURCE/'result.json'),
        classification='FIXED_NATIVE_NGP_QUADRATURE_NOT_LAW_ADOPTION',controls={},
        no_fit=True,no_actual_CF4_outcomes=True,no_gravity=True,
        limits=endpoint['limits'],MW_M31=endpoint['MW_M31'],M33=endpoint['M33'])
    baseline=None
    controls=() if os.environ.get('CF4_R2_NATIVE_ROWS_ONLY')=='1' else ((2,8),(2,64),(4,64))
    for volume,segments in controls:
        tic=time.monotonic()
        function=jax.jit(lambda:predict_source_volume_intensity(pos,vel,intrinsic,
            angular,source_spacing=1.5,volume_order=volume,order=4,
            segments=segments,deposition='ngp',**tracer_geometry(q,geometry)))
        lam=function();jax.block_until_ready(lam);host=np.asarray(lam).reshape(6,-1)
        if not np.isfinite(host).all() or np.any(host<0):
            raise ValueError('invalid refined intensity')
        flat=host.reshape(-1);at=flat[keys]
        if baseline is None:baseline=at.copy()
        control=dict(seconds=time.monotonic()-tic,
            baseline_zero_key_intensities=at[baseline<=0].tolist(),
            baseline_zero_keys=keys[baseline<=0].tolist())
        for name,mask in (('train',train),('test',test)):
            selected=mask[keys%128**3];expected=float(host[:,mask].sum())
            zero=selected&(at<=0)
            control[name]=dict(expected_count=expected,observed_count=int(counts[selected].sum()),
                zero_keys=keys[zero].tolist(),zero_observed_galaxies=int(counts[zero].sum()),
                logscore=None if zero.any() else float(np.sum(counts[selected]*
                    np.log(at[selected])-gammaln(counts[selected]+1))-expected))
        report['controls'][f'GL{volume}_LOS{segments}']=control
        report['elapsed_seconds']=time.monotonic()-start
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Exact native rows behind the failed occupied training keys. Local
    # cell-mean residual is evidence about the Gaussian closure, not proof
    # that no OTHER source location can populate an observed voxel.
    with np.load(endpoint['source_galaxies'],allow_pickle=False) as f:
        galaxies={k:f[k] for k in f.files}
    mapped=place_native_halves(galaxies['position'])
    mock=observe(mapped,galaxies['velocity'],galaxies['K_h_proxy'],radius,z,modulus)
    cell=np.floor((galaxies['position']%75.)/1.5).astype(int)
    local_velocity=np.moveaxis(moments[1:4]/moments[0],0,-1)[tuple(cell.T)]
    local_los=np.sum(local_velocity*mock['direction'],axis=1)
    residual=mock['coherent_vlos']-local_los
    sigma=endpoint['physical_sigma_los_km_s']
    voxel=np.floor(mock['positions']/3.).astype(int)%128
    row_keys=mock['population']*128**3+np.ravel_multi_index(tuple(voxel.T),(128,128,128))
    failed_keys=[10098880,11246902] # pre-existing occupied zeros in410555
    rows=np.flatnonzero(mock['selected']&np.isin(row_keys,failed_keys))
    report['native_rows']=[dict(key=int(row_keys[i]),native_id=int(galaxies['native_id'][i]),
        source_position=mapped[i].tolist(),observed_position=mock['positions'][i].tolist(),
        source_voxel=cell[i].tolist(),galaxy_vlos_km_s=float(mock['coherent_vlos'][i]),
        matter_cell_vlos_km_s=float(local_los[i]),residual_km_s=float(residual[i]),
        residual_over_sigma=float(residual[i]/sigma),
        K_h=float(galaxies['K_h_proxy'][i]),stellar_particles=int(galaxies['star_count'][i])) for i in rows]
    selected_residual=np.abs(residual[mock['selected']])/sigma
    report['selected_velocity_residual']=dict(sigma_km_s=sigma,
        absolute_standardized_percentiles=np.percentile(selected_residual,[50,90,95,99,100]).tolist(),
        number_beyond_eight_sigma=int(np.sum(selected_residual>8)),
        interpretation='local-cell residual; source-volume alternate support remains separate')
    report['elapsed_seconds']=time.monotonic()-start
    report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
