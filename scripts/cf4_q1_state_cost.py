"""Bounded Q1 state-dependent NumPy cost benchmark."""
import json, os, time, resource
import numpy as np
from cf4_q1_cell_integrated_convolution import predict_selected_intensity_cell_integrated

def main():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm required')
    rng=np.random.default_rng(20260919); out='/gpfs/kjhan/CF4/q1_state_cost/job_'+os.environ['SLURM_JOB_ID']; os.makedirs(out,exist_ok=False)
    m=8; positions=rng.uniform(0,12.,(m,3)); velocities=rng.normal(0,100.,(m,3)); masses=np.abs(rng.normal(1.,.2,(6,m)))
    kw=dict(observer=np.array([6.,6.,6.]),box_size_cMpc_h=12.,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.,sigma_fog_km_s=np.full(6,100.),sigma_redshift_km_s=np.full(6,30.))
    rows=[]
    for n in (128,256):
        start=time.perf_counter(); field=predict_selected_intensity_cell_integrated(positions,velocities,masses,np.full((6,n,n,n),.8),**kw); elapsed=time.perf_counter()-start
        rows.append(dict(grid=n,sources=m,populations=6,seconds=elapsed,peak_host_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,total_mass=float(field.sum()),finite=bool(np.isfinite(field).all())))
        print(json.dumps(rows[-1]),flush=True)
    json.dump(dict(status='COMPLETE_DEVELOPMENT_BENCHMARK',source_commit=os.environ['EXPECTED_COMMIT'],rows=rows,limitations='8-source representative timing; not a full R2 catalog and no JAX state gradient.'),open(out+'/result.json','w'),indent=2)
if __name__=='__main__': main()
