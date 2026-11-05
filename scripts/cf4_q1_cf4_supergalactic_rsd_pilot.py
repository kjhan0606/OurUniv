import csv, json
from pathlib import Path
import numpy as np
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator_shared_geometry

rows=[]
with Path('data/cf4_galaxies.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        try:
            vals=[float(row[k]) for k in ('DM','SGL','SGB','Vcmb')]
            if np.all(np.isfinite(vals)): rows.append(vals)
        except (TypeError,ValueError): continue
        if len(rows)==64: break
if len(rows)!=64: raise RuntimeError('fewer than 64 valid supergalactic rows')
box=1000.; h=.746; H=74.6; rows=np.asarray(rows); dist=h*10.0**((rows[:,0]-25.)/5.)
sgl=np.deg2rad(rows[:,1]); sgb=np.deg2rad(rows[:,2]); unit=np.column_stack((np.cos(sgb)*np.cos(sgl),np.cos(sgb)*np.sin(sgl),np.sin(sgb)))
obs=np.full(3,box/2.); positions=(obs+dist[:,None]*unit)%box
vr=rows[:,3]; coherent=h*vr/H; shifted=(positions+coherent[:,None]*unit)%box
rel=(shifted-obs+box/2)%box-box/2; los=rel/np.linalg.norm(rel,axis=1)[:,None]; sigma=np.full(64,h*np.hypot(24.,11.)/H); masses=np.random.default_rng(20261003).uniform(.5,1.5,size=(6,64))
field=evaluate_chunked_q1_operator_shared_geometry(shifted,los,sigma,masses,64,box,chunk_size=8)
out=Path('/gpfs/kjhan/CF4/q1_cf4_supergalactic_rsd_pilot_v1'); out.mkdir(parents=True,exist_ok=True); np.savez(out/'field.npz',field=field,positions=shifted,dist_cMpc_h=dist,radial_velocity_km_s=vr)
print(json.dumps({'status':'CF4_SUPERGALACTIC_RSD_PILOT','rows':64,'grid':64,'finite':bool(np.all(np.isfinite(field))),'nonnegative':bool(np.all(field>=0)),'mass':float(field.sum()),'output':str(out/'field.npz')}),flush=True)
