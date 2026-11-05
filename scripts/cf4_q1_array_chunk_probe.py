import json, os, time, resource
import numpy as np
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator_shared_geometry

task=int(os.environ['SLURM_ARRAY_TASK_ID']); n=128; m=16; box=6.; obs=np.array([3.,3.,3.]); disp=.746*np.hypot(24.,11.)/74.6
rng=np.random.default_rng(20260920+task); pos=rng.uniform(.05,box-.05,size=(m,3)); rel=(pos-obs+box/2)%box-box/2; los=rel/np.linalg.norm(rel,axis=1)[:,None]; scale=np.full(m,disp); masses=rng.uniform(.1,1.2,size=(6,m))
t=time.perf_counter(); out=evaluate_chunked_q1_operator_shared_geometry(pos,los,scale,masses,n,box,chunk_size=4); dt=time.perf_counter()-t
print(json.dumps({'task':task,'n':n,'sources':m,'seconds':dt,'rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,'mass':float(out.sum())},sort_keys=True),flush=True)
