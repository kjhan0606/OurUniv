import json, numpy as np
f=np.load('/gpfs/kjhan/CF4/q1_z0_selection_sensitivity_v1/posterior_sensitivity.npz'); box=384.; n=256; k0=2*np.pi/box; kk=np.meshgrid(*([np.fft.fftfreq(n)*n*k0]*3),indexing='ij'); km=np.sqrt(sum(x*x for x in kk)); bins=np.arange(0,33)*k0
def spectrum(x):
    y=x-x.mean(); p=np.abs(np.fft.fftn(y))**2/n**6; out=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        q=p[(km>=lo)&(km<hi)]; out.append(float(q.mean()) if q.size else 0.)
    return np.asarray(out)
a=spectrum(f['no_selection_mean']); b=spectrum(f['v1_mean']); c=spectrum(f['v2_mean']); valid=(a>0)&(b>0)&(c>0)
print(json.dumps({'status':'Z0_POSTERIOR_POWER_COMPARE','k_bins_used':int(valid.sum()),'v1_no_selection_power_ratio_median':float(np.median(b[valid]/a[valid])),'v2_no_selection_power_ratio_median':float(np.median(c[valid]/a[valid])),'v2_v1_power_ratio_max_abs_log':float(np.max(np.abs(np.log(c[valid]/b[valid])))),'mean_field_std_no_selection':float(f['no_selection_std'].mean()),'mean_field_std_v1':float(f['v1_std'].mean()),'mean_field_std_v2':float(f['v2_std'].mean())}),flush=True)
