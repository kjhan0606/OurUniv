import time
import resource
import numpy as np
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit

rng=np.random.default_rng(20260920); box=6.; obs=np.array([3.,3.,3.]); disp=.746*np.hypot(24.,11.)/74.6
for n in (64,128):
  for m in (16,64):
    pos=rng.uniform(0.05,box-0.05,size=(m,3)); rel=(pos-obs+box/2)%box-box/2; rh=rel/np.linalg.norm(rel,axis=1)[:,None]
    t=time.perf_counter(); total=0.
    for s in range(m):
      k=cell_integrated_tsc_deposit(pos[s:s+1],np.array([1.]),rh[s:s+1],np.array([disp]),n,box)
      _=np.fft.fftn(k); total += float(k.sum())
    dt=time.perf_counter()-t; rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    print({'n':n,'sources':m,'seconds':dt,'rss_MiB':rss,'mass':total},flush=True)
