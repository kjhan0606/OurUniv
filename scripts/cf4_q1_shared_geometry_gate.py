import time
import numpy as np
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator, evaluate_chunked_q1_operator_shared_geometry

rng=np.random.default_rng(20260920); box=6.; obs=np.array([3.,3.,3.]); disp=.746*np.hypot(24.,11.)/74.6
for n,m in ((256,16),):
    pos=rng.uniform(.05,box-.05,size=(m,3)); rel=(pos-obs+box/2)%box-box/2; los=rel/np.linalg.norm(rel,axis=1)[:,None]; scale=np.full(m,disp); masses=rng.uniform(.1,1.2,size=(6,m))
    t=time.perf_counter(); baseline=evaluate_chunked_q1_operator(pos,los,scale,masses,n,box,chunk_size=4); t0=time.perf_counter()-t
    t=time.perf_counter(); shared=evaluate_chunked_q1_operator_shared_geometry(pos,los,scale,masses,n,box,chunk_size=4); t1=time.perf_counter()-t
    err=float(np.sum(np.abs(shared-baseline))/np.sum(np.abs(baseline)))
    print({'n':n,'sources':m,'baseline_seconds':t0,'shared_seconds':t1,'speedup':t0/t1,'relative_l1':err,'mass_error':float(np.max(np.abs(shared.sum((1,2,3))-masses.sum(axis=1))))},flush=True)
