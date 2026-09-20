import json, numpy as np
f=np.load('/gpfs/kjhan/CF4/q1_low_highk_blend_dev_v1.npz'); x=f['field']; n=x.shape[0]; box=384.; k0=2*np.pi/box; q=np.meshgrid(*([np.fft.fftfreq(n)*n*k0]*3),indexing='ij'); km=np.sqrt(sum(z*z for z in q)); p=np.abs(np.fft.fftn(x-x.mean()))**2/n**6; cut=.30
def mean(lo,hi):
    z=p[(km>=lo)&(km<hi)]; return float(z.mean()) if z.size else 0.
low=mean(.25,.30); high=mean(.30,.35); below=mean(.20,.25); above=mean(.35,.40)
print(json.dumps({'status':'LOW_HIGHK_CONTINUITY','kcut':cut,'P_below':below,'P_low_edge':low,'P_high_edge':high,'P_above':above,'edge_ratio_high_low':high/low if low>0 else None,'finite':bool(np.all(np.isfinite(x))),'mean':float(x.mean())}),flush=True)
