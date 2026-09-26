import csv, json, h5py
from pathlib import Path
import numpy as np

rows=[]
with Path('data/cf4_galaxies.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        try:
            vals=[float(row[k]) for k in ('DM','SGL','SGB')]
            if np.all(np.isfinite(vals)): rows.append(vals)
        except (TypeError,ValueError): continue
rows=np.asarray(rows); h=.746; box=384.; dist=h*10**((rows[:,0]-25)/5); sgl=np.deg2rad(rows[:,1]); sgb=np.deg2rad(rows[:,2]); u=np.column_stack((np.cos(sgb)*np.cos(sgl),np.cos(sgb)*np.sin(sgl),np.sin(sgb))); x=dist[:,None]*u; inside=np.all(np.abs(x)<box/2,axis=1); idx=np.floor((x[inside]+box/2)/1.5).astype(int).clip(0,255)
with h5py.File('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/selection_1p5_v2/selection.h5','r') as f: sel=np.asarray(f['selection_shells'][:,:, :, :, :])
exposure=sel.sum(axis=1); values=exposure[:,idx[:,0],idx[:,1],idx[:,2]].T if idx.size else np.empty((0,6)); per_pop=(np.mean(values>0,axis=0).tolist() if idx.size else [0.]*6)
print(json.dumps({'status':'SELECTION_MAP_COVERAGE','rows':int(rows.shape[0]),'map_status':'SUPPORT_REPAIRED_NOT_SELECTION_CALIBRATION','map_dx_cMpc_h':1.5,'map_box_cMpc_h':box,'inside_fraction':float(np.mean(inside)),'inside_rows':int(inside.sum()),'positive_exposure_fraction_per_population':per_pop,'exposure_min':float(values.min()) if idx.size else 0.,'exposure_max':float(values.max()) if idx.size else 0.},sort_keys=True),flush=True)
