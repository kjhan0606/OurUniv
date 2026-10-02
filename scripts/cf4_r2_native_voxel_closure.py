"""Two occupied-voxel marked integrals with exact ray/voxel boundaries."""
import json
import os
from pathlib import Path
import time
import resource

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_native_mock import distance_tables, place_native_halves
from cf4_r2_raw_volume_target import tracer_masses, tracer_geometry
from cf4_r2_shell_cdf_count import predict_source_volume_intensity

jax.config.update('jax_enable_x64',True)
BASE=Path('/gpfs/kjhan/CF4/z0_density')
OUT=BASE/'r2_native_voxel_closure_20261002_v1'


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    endpoint=json.loads((BASE/'r2_native_rsd_mock_fit_20261002_v2/result.json').read_text())
    fit=json.loads((BASE/'r2_native_velocity_closure_20261002_v1/result.json').read_text())
    mixture=fit['results']['cell_mean']['preselection']['mixture']
    if not mixture['success']:raise ValueError('native training mixture fit failed')
    with h5py.File(endpoint['source_matter'],'r') as f:
        m=f['coarse'][:];h=float(f.attrs['h']);omega=float(f.attrs['Omega_m'])
    rho=jnp.asarray(m[0]/m[0].mean())
    vel=jnp.asarray(np.moveaxis(m[1:4]/m[0],0,-1).reshape(-1,3))
    axis=(np.arange(50)+.5)*1.5
    native=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    pos=jnp.asarray(place_native_halves(native));angular=jnp.ones((2,len(native)))
    q=jnp.asarray(endpoint['final_coordinates']);intrinsic=tracer_masses(rho,q)/8.
    radial,z,modulus=distance_tables(omega)
    geometry=tracer_geometry(q,dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
        hubble_km_s_Mpc=100*h,little_h=h,grid_size=128,
        radius_table_cMpc_h=jnp.asarray(radial),redshift_table=jnp.asarray(z),
        modulus_table_h=jnp.asarray(modulus),radial_min_cMpc_h=5.,radial_max_cMpc_h=180.))
    report=dict(job_id=os.environ['SLURM_JOB_ID'],classification='TWO_KEY_MARKED_VOXEL_DEVELOPMENT_PROBE',
        mixture=mixture,keys={},no_fit=True,no_prior_injection=True,no_actual_CF4_outcomes=True,
        MW_M31=endpoint['MW_M31'],M33=endpoint['M33'],limits=endpoint['limits'],
        geometry='exact LOS voxel interval; finite source-volume and varying mark quadrature',
        mixture_support='each Gaussian component clipped at8sigma,27 periodic images; unconditional weights')
    for key in (10098880,11246902):
        population,spatial=divmod(key,128**3)
        x,rem=divmod(spatial,128**2);y,zcell=divmod(rem,128)
        record=dict(population=population,voxel=[x,y,zcell],rules={})
        for volume,order in ((2,4),(4,8)):
            def evaluate(log_sigma):
                geom=dict(geometry,sigma_los_km_s=jnp.exp(log_sigma))
                return predict_source_volume_intensity(pos,vel,intrinsic,angular,
                    source_spacing=1.5,volume_order=volume,order=order,segments=1,
                    deposition='voxel_cdf',target_population=population,
                    target_voxel=(x,y,zcell),**geom)
            fn=jax.jit(evaluate);tic=time.monotonic()
            widths=[endpoint['physical_sigma_los_km_s'],mixture['sigma_core'],mixture['sigma_broad']]
            values=[]
            for width in widths:
                if not 0<8*.01*width<192:raise ValueError('component periodic support violated')
                result=fn(jnp.log(width));jax.block_until_ready(result);values.append(float(result))
            if not np.isfinite(values).all() or min(values)<0:raise ValueError('invalid count integral')
            f=mixture['fraction_broad']
            record['rules'][f'GL{volume}_mark{order}']=dict(seconds=time.monotonic()-tic,
                single_gaussian=values[0],core=values[1],broad=values[2],
                mixture=(1-f)*values[1]+f*values[2])
            if volume==2:
                log_sigma=np.log(widths[2]);eps=1e-4
                derivative=jax.jit(jax.grad(evaluate))(log_sigma)
                fd=(fn(log_sigma+eps)-fn(log_sigma-eps))/(2*eps)
                derivative,fd=float(derivative),float(fd)
                error=abs(derivative-fd)/max(1e-10,abs(derivative),abs(fd))
                record['broad_width_derivative']=dict(reverse=derivative,finite_difference=fd,relative_error=error)
                if not np.isfinite(error) or error>2e-3:raise ValueError('voxel-width derivative failed')
        report['keys'][str(key)]=record
        report['elapsed_seconds']=time.monotonic()-start
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
