"""Full native marked count prediction with voxel-boundary LOS integration."""
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

from cf4_r2_native_mock import distance_tables,place_native_halves,exposure_and_radial_bins,aggregate
from cf4_r2_raw_volume_target import tracer_masses,tracer_geometry
from cf4_r2_shell_cdf_count import predict_source_volume_intensity

jax.config.update('jax_enable_x64',True)
BASE=Path('/gpfs/kjhan/CF4/z0_density')
OUT=BASE/'r2_native_grid_closure_20261002_v1'


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    endpoint=json.loads((BASE/'r2_native_rsd_mock_fit_20261002_v2/result.json').read_text())
    closure=json.loads((BASE/'r2_native_velocity_closure_20261002_v2/result.json').read_text())['results']['cell_mean']['conditional_mixture']
    scalar=json.loads((BASE/'r2_native_voxel_closure_20261002_v2/result.json').read_text())
    with h5py.File(endpoint['source_matter'],'r') as f:
        m=f['coarse'][:];h=float(f.attrs['h']);omega=float(f.attrs['Omega_m'])
    with np.load(BASE/'r2_native_rsd_mock_fit_20261002_v2/mock_observations.npz',allow_pickle=False) as f:
        keys=f['keys'];counts=f['counts']
    train,test,rbin=exposure_and_radial_bins()
    mean=np.moveaxis(m[1:4]/m[0],0,-1).reshape(-1,3)
    second=np.moveaxis(m[4:7]/m[0],0,-1).reshape(-1,3)
    variance=second-mean**2
    if np.any(variance < -1e-8*np.maximum(1.,np.abs(second))):raise ValueError('invalid second moments')
    variance=np.maximum(variance,0.)
    upper=np.sqrt(closure['sigma_core']**2+closure['matter_dispersion_scale']**2*variance.max())
    if not 0<8*.01*upper<192:raise ValueError('conditional support domain exceeded')
    axis=(np.arange(50)+.5)*1.5
    pos=jnp.asarray(place_native_halves(np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)))
    vel=jnp.asarray(mean);var=jnp.asarray(variance);angular=jnp.ones((2,len(mean)))
    q=jnp.asarray(endpoint['final_coordinates']);intrinsic=tracer_masses(jnp.asarray(m[0]/m[0].mean()),q)/8.
    radial,z,modulus=distance_tables(omega)
    geometry=tracer_geometry(q,dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
        hubble_km_s_Mpc=100*h,little_h=h,grid_size=128,
        radius_table_cMpc_h=jnp.asarray(radial),redshift_table=jnp.asarray(z),
        modulus_table_h=jnp.asarray(modulus),radial_min_cMpc_h=5.,radial_max_cMpc_h=180.))
    report=dict(job_id=os.environ['SLURM_JOB_ID'],classification='FULL_NATIVE_VOXEL_CDF_FIXED_CLOSURE_DEVELOPMENT',
        no_fit=True,no_CF4_outcomes=True,no_prior_injection=True,closure=closure,
        conservative_max_sigma_km_s=float(upper),prediction={},scalar_checks={},
        limits=endpoint['limits']+['Only source GL2 and mark4 in this first full-grid readout; convergence unproven.'],
        MW_M31=endpoint['MW_M31'],M33=endpoint['M33'])
    def prediction(base,scale,conditional):
        geom=dict(geometry,sigma_los_km_s=base)
        if conditional:geom.update(source_velocity_variances_km2_s2=var,dispersion_scale=scale)
        return predict_source_volume_intensity(pos,vel,intrinsic,angular,source_spacing=1.5,
            volume_order=2,order=4,segments=1,deposition='voxel_cdf_grid',**geom)
    core_fn=jax.jit(lambda width:prediction(width,1.,False))
    broad_fn=jax.jit(lambda scale:prediction(closure['sigma_core'],scale,True))
    tic=time.monotonic();core=core_fn(closure['sigma_core']);jax.block_until_ready(core)
    report['core_seconds']=time.monotonic()-tic
    tic=time.monotonic();broad=broad_fn(closure['matter_dispersion_scale']);jax.block_until_ready(broad)
    report['broad_seconds']=time.monotonic()-tic
    host=np.asarray((1-closure['fraction_broad'])*core+closure['fraction_broad']*broad)
    if not np.isfinite(host).all() or np.any(host<0):raise ValueError('invalid full-grid prediction')
    flat=host.reshape(-1)
    for key,record in scalar['keys'].items():
        expected=record['rules']['GL2_mark4']['conditional_mixture'];value=float(flat[int(key)])
        error=abs(value-expected)/max(1e-20,abs(value),abs(expected))
        report['scalar_checks'][key]=dict(scalar=expected,full_grid=value,relative_error=error)
        if error>1e-7:raise ValueError('full-grid/scalar boundary integral mismatch')
    for split,mask in (('train',train),('test',test)):
        keep=mask[keys%128**3];at=flat[keys[keep]];nn=counts[keep]
        table=np.stack([np.bincount(rbin[mask],weights=row[mask],minlength=16) for row in host.reshape(6,-1)])
        observed=aggregate(keys,counts,mask,rbin);zeros=at<=0;expected=float(table.sum())
        report['prediction'][split]=dict(expected_count=expected,observed_count=int(nn.sum()),
            occupied_zero_keys=keys[keep][zeros].tolist(),occupied_zero_galaxies=int(nn[zeros].sum()),
            count_logscore=None if zeros.any() else float(np.sum(nn*np.log(at)-gammaln(nn+1))-expected),
            aggregate_L1=float(np.abs(table-observed).sum()),predicted_table=table.tolist(),observed_table=observed.tolist())
    np.savez_compressed(OUT/'predicted_counts.npz',conditional=host)
    report['elapsed_seconds']=time.monotonic()-start
    report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
