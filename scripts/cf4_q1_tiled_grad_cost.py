import json, os, resource, time
import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_state_dependent_jax import predict_selected_intensity_state_jax_tiled

jax.config.update('jax_enable_x64', True)
if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm required')
rng=np.random.default_rng(20260920); box=384.; n=256; m=32
order=int(os.environ.get('Q1_TILED_ORDER','256')); tile=int(os.environ.get('Q1_TILE_DEPTH','32'))
pos=jnp.asarray(rng.uniform(0.,box,(m,3))); vel=jnp.asarray(rng.normal(0.,100.,(m,3)))
masses=jnp.asarray(np.abs(rng.normal(1.,.2,(6,m)))); exposure=jnp.full((6,n,n,n),.8,dtype=jnp.float64)
kw=dict(observer=jnp.full(3,box/2.),box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.,sigma_fog_km_s=jnp.full(6,100.),sigma_redshift_km_s=jnp.full(6,30.),quadrature_order=order)
def objective(p,v): return jnp.sum(predict_selected_intensity_state_jax_tiled(p,v,masses,exposure,tile_depth=tile,**kw)**2)
start=time.perf_counter(); value,(gp,gv)=jax.value_and_grad(objective,argnums=(0,1))(pos,vel); jax.block_until_ready(value); jax.block_until_ready(gp); jax.block_until_ready(gv); elapsed=time.perf_counter()-start
row=dict(grid=n,sources=m,order=order,tile_depth=tile,seconds=elapsed,peak_host_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,grad_pos_norm=float(jnp.linalg.norm(gp)),finite=bool(np.isfinite(np.asarray(gp)).all()))
print(json.dumps(row),flush=True)
out='/gpfs/kjhan/CF4/q1_tiled_grad_cost/job_'+os.environ['SLURM_JOB_ID']; os.makedirs(out,exist_ok=False)
open(out+'/OWNER.txt','w').write('CF4 tiled bounded benchmark owner=kjhan\n'); json.dump(dict(status='TILED_GRAD_COST',source_commit=os.environ['EXPECTED_COMMIT'],row=row),open(out+'/result.json','w'),indent=2)
