import csv, json
from pathlib import Path
import numpy as np
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator_shared_geometry

path=Path('data/cf4_galaxies.csv'); rows=[]
with path.open(newline='') as f:
    for row in csv.DictReader(f):
        try:
            dm=float(row['DM']); ra=np.deg2rad(float(row['RAJ2000'])); dec=np.deg2rad(float(row['DEJ2000']))
            if not (np.isfinite(dm) and np.isfinite(ra) and np.isfinite(dec)): continue
            rows.append((dm,ra,dec))
        except (TypeError,ValueError): continue
        if len(rows)==64: break
if len(rows)!=64: raise RuntimeError('fewer than 64 valid CF4 rows')
box=1000.; h=.746; dist=h*10.0**((np.array([r[0] for r in rows])-25.)/5.)
ra=np.array([r[1] for r in rows]); dec=np.array([r[2] for r in rows]); unit=np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
observer=np.full(3,box/2.); positions=(observer+dist[:,None]*unit)%box; scale=np.full(64,h*np.hypot(24.,11.)/74.6); masses=np.ones((6,64)); t=np.random.default_rng(20261002).uniform(.5,1.5,size=(6,64))
field=evaluate_chunked_q1_operator_shared_geometry(positions,unit,scale,masses*t,64,box,chunk_size=8)
out=Path('/gpfs/kjhan/CF4/q1_cf4_catalog_pilot_v1'); out.mkdir(parents=True,exist_ok=True); np.savez(out/'field.npz',field=field,positions=positions,dist_cMpc_h=dist)
print(json.dumps({'status':'CF4_CATALOG_PILOT','rows':64,'grid':64,'box_cMpc_h':box,'finite':bool(np.all(np.isfinite(field))),'nonnegative':bool(np.all(field>=0)),'mass':float(field.sum()),'output':str(out/'field.npz')}),flush=True)
