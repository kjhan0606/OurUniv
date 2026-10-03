import json
import numpy as np
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit

n=16; box=6.; h=box/n; center=np.array([box/2]*3); direction=np.array([[1.,0.,0.]]); scale=np.array([.2])
base=cell_integrated_tsc_deposit(center[None,:],np.array([1.]),direction,scale,n,box)
rows=[]
for shift in (np.array([1,0,0]),np.array([0,2,0]),np.array([0,0,-3])):
    moved=(center+shift*h)%box
    target=cell_integrated_tsc_deposit(moved[None,:],np.array([1.]),direction,scale,n,box)
    rolled=np.roll(base,shift.astype(int),axis=(0,1,2))
    error=float(np.max(np.abs(target-rolled)))
    rows.append({'shift':shift.tolist(),'max_abs':error,'target_mass':float(target.sum())})
print(json.dumps({'status':'IMPULSE_TRANSLATION_CHECK','rows':rows},indent=2),flush=True)
assert max(row['max_abs'] for row in rows) < 1e-12
