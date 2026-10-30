import time
import numpy as np
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator

rng=np.random.default_rng(20260920); box=6.; obs=np.array([3.,3.,3.]); disp=.746*np.hypot(24.,11.)/74.6
for n,m in ((256,8),):
    pos=rng.uniform(.05,box-.05,size=(m,3)); rel=(pos-obs+box/2)%box-box/2; los=rel/np.linalg.norm(rel,axis=1)[:,None]; scale=np.full(m,disp)
    masses=rng.uniform(.1,1.2,size=(6,m)); t=time.perf_counter()
    oracle=np.zeros((6,n,n,n))
    for p in range(6): oracle[p]=cell_integrated_tsc_deposit(pos,masses[p],los,scale,n,box)
    pred=evaluate_chunked_q1_operator(pos,los,scale,masses,n,box,chunk_size=4)
    print({'n':n,'sources':m,'relative_l1':float(np.sum(np.abs(pred-oracle))/np.sum(np.abs(oracle))),'mass_error':float(np.max(np.abs(pred.sum((1,2,3))-masses.sum(axis=1)))),'seconds':time.perf_counter()-t},flush=True)
