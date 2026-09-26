import json, numpy as np
f=np.load('/gpfs/kjhan/CF4/q1_z0_selection_sensitivity_v1/posterior_sensitivity.npz')
a=f['no_selection_mean']; b=f['v1_mean']; c=f['v2_mean']
def rel(x,y): return float(np.sum(np.abs(x-y))/np.sum(np.abs(x)))
print(json.dumps({'status':'Z0_SELECTION_COMPARE','v1_vs_no_selection_l1':rel(a,b),'v2_vs_no_selection_l1':rel(a,c),'v2_vs_v1_l1':rel(b,c),'v2_v1_max_abs':float(np.max(np.abs(b-c))),'v1_v2_total_ratio':float(c.sum()/b.sum())}),flush=True)
