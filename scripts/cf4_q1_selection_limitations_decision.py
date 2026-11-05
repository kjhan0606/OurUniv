import csv, json, h5py
from pathlib import Path
import numpy as np

rows=[]
with Path('data/cf4_galaxies.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        try:
            v=[float(row[k]) for k in ('DM','SGL','SGB')]
            if np.all(np.isfinite(v)): rows.append(v)
        except (TypeError,ValueError): continue
r=np.asarray(rows); h=.746; box=384.; d=h*10**((r[:,0]-25)/5); a=np.deg2rad(r[:,1]); b=np.deg2rad(r[:,2]); u=np.column_stack((np.cos(b)*np.cos(a),np.cos(b)*np.sin(a),np.sin(b))); x=d[:,None]*u; inside=np.all(np.abs(x)<box/2,axis=1)&(d>=5)&(d<=180)
with h5py.File('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/selection_1p5_v1/selection.h5','r') as f: s=np.asarray(f['selection_shells']).sum(axis=1)
dx=1.5; cell=np.floor((x[inside]+box/2)/dx).astype(int).clip(0,255); e=s[:,cell[:,0],cell[:,1],cell[:,2]].T; per=np.mean(e>0,axis=0)
decision={'status':'SELECTION_LIMITATIONS_DECISION','provenance':'PASS_OFFICIAL_ARES','map_boundary':'CONDITIONAL_ONLY','map_inside_fraction':float(np.mean(inside)),'map_outside_fraction':float(1-np.mean(inside)),'positive_exposure_fraction_per_population':per.tolist(),'survival_bias_calibration':'UNAVAILABLE_IN_ARES_MAP_AND_CF4_CSV','normalization':'EFFECTIVE_VOLUME_ONLY_NOT_FULL_SELECTION_LIKELIHOOD','development_use':'GO_WITH_EXPLICIT_MASK_AND_SELECTION_NUISANCE','production_use':'NO_GO_UNTIL_EXTERNAL_SURVIVAL_BIAS_OR_JOINT_CALIBRATION','required_action':'restrict baseline posterior to map-supported volume and carry v1/v2/no-selection sensitivity'}
print(json.dumps(decision,sort_keys=True),flush=True)
