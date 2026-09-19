import time
import resource
import numpy as np
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit

box=6.; obs=np.array([3.,3.,3.]); pos=np.array([[1.3125,2.0625,4.3125],[3.5625,4.0625,1.1875]])
vel=np.array([[70.,-15.,10.],[-55.,25.,-12.]])
# Fixed post-RSD positions/directions from the sealed diagnostic.
shifted=np.array([[1.5941129032258066,2.2189516129032256,4.093467741935484],[3.583160611065235,4.101525598678778,1.1209269199009084]])
rel=(shifted-obs+box/2)%box-box/2; rh=rel/np.linalg.norm(rel,axis=1)[:,None]
disp=.746*np.hypot(24.,11.)/74.6
for n in (16,32,64,128,256):
    t=time.perf_counter(); total=0.; kernels=[]
    for s in range(2):
        k=cell_integrated_tsc_deposit(shifted[s:s+1],np.array([1.]),rh[s:s+1],np.array([disp]),n,box)
        kernels.append(np.fft.fftn(k)); total += float(k.sum())
    dt=time.perf_counter()-t; rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    print({'n':n,'kernel_bytes':int(2*n**3*16),'seconds':dt,'rss_MiB':rss,'mass':total},flush=True)
