"""One bounded algebra/cost comparison; no dynamics, inference or heldout."""
import json
import os
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_2mpp_joint_likelihood_jax import tsc_deposit_jax


def deposit_populations(positions,masses,n,box):
    """Same TSC weights, with a contiguous population axis in each scatter."""
    cell=(positions%box)/(box/n)-.5
    nearest=jnp.floor(cell+.5).astype(jnp.int32)
    offset=cell-nearest
    weights=[(.5*(.5-offset[:,a])**2,.75-offset[:,a]**2,
              .5*(.5+offset[:,a])**2) for a in range(3)]
    result=jnp.zeros((n,n,n,masses.shape[0]),dtype=masses.dtype)
    for ix,dx in enumerate((-1,0,1)):
        for iy,dy in enumerate((-1,0,1)):
            for iz,dz in enumerate((-1,0,1)):
                weight=weights[0][ix]*weights[1][iy]*weights[2][iz]
                result=result.at[(nearest[:,0]+dx)%n,(nearest[:,1]+dy)%n,
                                 (nearest[:,2]+dz)%n,:].add(masses.T*weight[:,None])
    return jnp.moveaxis(result,-1,0)


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(exist_ok=False)
    report=dict(job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
                classification='DEPOSITION_EQUIVALENCE_AND_COST_ONLY',PM_evolutions=0,
                heldout_scored=False,R2_complete=False,cases=[])
    # Deliberately non-lattice positions and unequal population weights.
    # This is a numerical fixture, not a synthetic survey/calibration mock.
    for n,number in ((8,127),(128,128**3)):
        rng=np.random.default_rng(2026092807)
        x=jnp.asarray(rng.uniform(-.1*n,1.1*n,(number,3)))
        mass=jnp.asarray(rng.uniform(.1,2.,(6,number)))
        def old(pos,m):
            return jnp.stack([tsc_deposit_jax(pos,m[p],n,float(n)) for p in range(6)])
        def new(pos,m):
            return deposit_populations(pos,m,n,float(n))
        cotangent=jnp.asarray(rng.standard_normal((6,n,n,n)))
        results=[]
        row=dict(n=n,sources=number)
        for name,fn in (('separate',old),('batched',new)):
            forward=jax.jit(fn)
            derivative=jax.jit(jax.grad(lambda pos,m:jnp.vdot(fn(pos,m),cotangent),argnums=(0,1)))
            y=forward(x,mass); y.block_until_ready()
            g=derivative(x,mass); jax.block_until_ready(g)
            timing={}
            for label,call in (('forward',lambda:forward(x,mass)),
                               ('derivative',lambda:derivative(x,mass))):
                seconds=[]
                for _ in range(3):
                    start=time.monotonic(); jax.block_until_ready(call())
                    seconds.append(time.monotonic()-start)
                timing[label+'_median_seconds']=float(np.median(seconds))
            row[name]=timing
            results.append((y,g))
        row['maximum_value_difference']=float(jnp.max(jnp.abs(results[0][0]-results[1][0])))
        row['maximum_gradient_differences']=[float(jnp.max(jnp.abs(a-b)))
            for a,b in zip(results[0][1],results[1][1])]
        np.testing.assert_allclose(np.asarray(results[0][0]),np.asarray(results[1][0]),rtol=1e-12,atol=1e-11)
        for a,b in zip(results[0][1],results[1][1]):
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-12,atol=1e-10)
        report['cases'].append(row)
        print(json.dumps(row),flush=True)
        del results,x,mass,cotangent,y,g
        jax.clear_caches()
    report.update(status='COMPLETE_NOT_PRODUCTION_IMPLEMENTATION',
                  host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
