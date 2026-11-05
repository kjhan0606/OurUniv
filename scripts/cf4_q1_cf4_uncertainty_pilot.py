import csv, json
from pathlib import Path
import numpy as np
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator_shared_geometry

rows=[]
with Path('data/cf4_galaxies.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        try:
            dm=float(row['DM']); sgl=float(row['SGL']); sgb=float(row['SGB']); vc=float(row['Vcmb'])
            edms=[float(row[k]) for k in ('e_DM','e_DMsnIa','e_DMtf','e_DMfp','e_DMsbf','e_DMsnII','e_DMtrgb','e_DMceph','e_DMmas') if row.get(k,'') not in ('',None)]
            edm=edms[0] if edms else np.nan
            if np.all(np.isfinite([dm,sgl,sgb,vc,edm])) and edm>=0: rows.append((dm,sgl,sgb,vc,edm))
        except (TypeError,ValueError): continue
        if len(rows)==64: break
if len(rows)!=64: raise RuntimeError('fewer than 64 valid CF4 rows with distance errors')
box=1000.; h=.746; H=74.6; rows=np.asarray(rows); dist=h*10.0**((rows[:,0]-25.)/5.); sgl=np.deg2rad(rows[:,1]); sgb=np.deg2rad(rows[:,2]); unit=np.column_stack((np.cos(sgb)*np.cos(sgl),np.cos(sgb)*np.sin(sgl),np.sin(sgb))); obs=np.full(3,box/2.); pos=(obs+dist[:,None]*unit)%box
shifted=(pos+(h*rows[:,3]/H)[:,None]*unit)%box; rel=(shifted-obs+box/2)%box-box/2; los=rel/np.linalg.norm(rel,axis=1)[:,None]; sigma_r=np.log(10.)/5.*dist*rows[:,4]; sigma_v=h*np.hypot(24.,11.)/H; sigma=np.hypot(sigma_r,sigma_v); masses=np.ones((6,64))
field=evaluate_chunked_q1_operator_shared_geometry(shifted,los,sigma,masses,64,box,chunk_size=8)
out=Path('/gpfs/kjhan/CF4/q1_cf4_uncertainty_pilot_v1'); out.mkdir(parents=True,exist_ok=True); np.savez(out/'field.npz',field=field,positions=shifted,sigma_cMpc_h=sigma,dm_error=rows[:,4])
print(json.dumps({'status':'CF4_UNCERTAINTY_PILOT','rows':64,'finite':bool(np.all(np.isfinite(field))),'mass':float(field.sum()),'sigma_min':float(sigma.min()),'sigma_max':float(sigma.max()),'tangential_velocity':'unobserved_zero_assumption','selection_weights':'not_available_in_catalog_no_correction_applied','output':str(out/'field.npz')}),flush=True)
