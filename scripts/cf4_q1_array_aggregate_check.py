import json
from pathlib import Path
import numpy as np
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator_shared_geometry

n=64; total=64; m=16; box=6.; obs=np.array([3.,3.,3.]); disp=.746*np.hypot(24.,11.)/74.6; rng=np.random.default_rng(20261001)
pos=rng.uniform(.05,box-.05,size=(total,3)); rel=(pos-obs+box/2)%box-box/2; los=rel/np.linalg.norm(rel,axis=1)[:,None]; scale=np.full(total,disp); masses=rng.uniform(.1,1.2,size=(6,total))
root=Path('/gpfs/kjhan/CF4/q1_array_aggregate_probe'); parts=[np.load(root/f'part_{i}.npz')['field'] for i in range(4)]; summed=np.sum(parts,axis=0); mono=evaluate_chunked_q1_operator_shared_geometry(pos,los,scale,masses,n,box,chunk_size=8)
print(json.dumps({'status':'ARRAY_AGGREGATE_GATE','relative_l1':float(np.sum(np.abs(summed-mono))/np.sum(np.abs(mono))),'max_abs':float(np.max(np.abs(summed-mono))),'mass_error':float(np.max(np.abs(summed.sum((1,2,3))-masses.sum(axis=1))))}),flush=True)
