import time
import numpy as np
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit

rng=np.random.default_rng(20260920); box=6.; obs=np.array([3.,3.,3.]); disp=.746*np.hypot(24.,11.)/74.6
for n in (64,128):
  for m in (16,64):
    pos=rng.uniform(0.05,box-0.05,size=(m,3)); rel=(pos-obs+box/2)%box-box/2; rh=rel/np.linalg.norm(rel,axis=1)[:,None]; t=time.perf_counter()
    oracle=cell_integrated_tsc_deposit(pos,np.ones(m),rh,np.full(m,disp),n,box)
    pred=np.zeros_like(oracle)
    for s in range(m):
      cell=np.floor(pos[s]/(box/n)).astype(int)%n
      k=cell_integrated_tsc_deposit(pos[s:s+1],np.array([1.]),rh[s:s+1],np.array([disp]),n,box)
      kernel=np.fft.fftn(np.roll(k,-cell,axis=(0,1,2)))
      delta=np.zeros((n,n,n)); delta[tuple(cell)]=1.
      pred += np.real(np.fft.ifftn(np.fft.fftn(delta)*kernel))
    err=float(np.sum(np.abs(pred-oracle))/np.sum(np.abs(oracle))); mass=float(pred.sum()); dt=time.perf_counter()-t
    print({'n':n,'sources':m,'relative_l1':err,'mass':mass,'seconds':dt},flush=True)
