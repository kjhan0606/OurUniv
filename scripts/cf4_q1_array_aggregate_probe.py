import json, os
from pathlib import Path
import numpy as np
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator_shared_geometry

task=int(os.environ['SLURM_ARRAY_TASK_ID']); n=64; total=64; chunks=4; m=total//chunks; box=6.; obs=np.array([3.,3.,3.]); disp=.746*np.hypot(24.,11.)/74.6
rng=np.random.default_rng(20261001); pos=rng.uniform(.05,box-.05,size=(total,3)); rel=(pos-obs+box/2)%box-box/2; los=rel/np.linalg.norm(rel,axis=1)[:,None]; scale=np.full(total,disp); masses=rng.uniform(.1,1.2,size=(6,total)); start=task*m; stop=start+m
field=evaluate_chunked_q1_operator_shared_geometry(pos[start:stop],los[start:stop],scale[start:stop],masses[:,start:stop],n,box,chunk_size=8)
out=Path('/gpfs/kjhan/CF4/q1_array_aggregate_probe'); out.mkdir(parents=True,exist_ok=True); np.savez(out/f'part_{task}.npz',field=field,start=start,stop=stop)
print(json.dumps({'task':task,'start':start,'stop':stop,'mass':float(field.sum())}),flush=True)
