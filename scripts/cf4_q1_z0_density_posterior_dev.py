import csv, json, h5py
from pathlib import Path
import numpy as np
from cf4_2mpp_joint_likelihood_local import tsc_deposit

catalog=[]
with Path('data/cf4_galaxies.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        try:
            vals=[float(row[k]) for k in ('DM','e_DM','SGL','SGB','Vcmb')]
            if np.all(np.isfinite(vals)) and vals[1]>=0: catalog.append(vals)
        except (TypeError,ValueError): continue
cat=np.asarray(catalog); h=.746; H=74.6; box=384.; n=256; dx=box/n; obs=np.full(3,box/2.); edges=np.array([5.,30.,60.,90.,120.,150.,180.])
with h5py.File('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/selection_1p5_v2/selection.h5','r') as f: selection=np.asarray(f['selection_shells'])
draws=[]; rng=np.random.default_rng(20261005)
for draw in range(8):
    dm=cat[:,0]+rng.normal(size=cat.shape[0])*cat[:,1]; dist=h*10**((dm-25)/5); sgl=np.deg2rad(cat[:,2]); sgb=np.deg2rad(cat[:,3]); u=np.column_stack((np.cos(sgb)*np.cos(sgl),np.cos(sgb)*np.sin(sgl),np.sin(sgb))); pos=(obs+dist[:,None]*u)%box; pos=(pos+(h*cat[:,4]/H)[:,None]*u)%box; rel=(pos-obs+box/2)%box-box/2; inside=np.all(np.abs(rel)<box/2,axis=1)&(np.linalg.norm(rel,axis=1)>=5)&(np.linalg.norm(rel,axis=1)<=180); pos=pos[inside]; rel=rel[inside]; radius=np.linalg.norm(rel,axis=1); shell=np.clip(np.searchsorted(edges,radius,side='right')-1,0,5); cell=np.floor(pos/dx).astype(int)%n; weights=np.zeros((6,pos.shape[0])); valid=np.zeros(pos.shape[0],bool)
    for p in range(6): weights[p]=selection[p,shell,cell[:,0],cell[:,1],cell[:,2]]; valid|=weights[p]>0
    pos=pos[valid]; weights=weights[:,valid]; fields=np.stack([tsc_deposit(pos,weights[p],n,box) for p in range(6)]); draws.append(fields.sum(axis=0));
posterior=np.stack(draws); out=Path('/gpfs/kjhan/CF4/q1_z0_density_posterior_dev_v1'); out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/'posterior.npz',mean=posterior.mean(0),std=posterior.std(0),draws=posterior)
print(json.dumps({'status':'Z0_DENSITY_POSTERIOR_DEVELOPMENT','catalog_rows':int(cat.shape[0]),'draws':8,'grid':n,'box_cMpc_h':box,'mean_total':float(posterior.mean(0).sum()),'relative_std_total':float(posterior.sum((1,2,3)).std()/posterior.sum((1,2,3)).mean()),'selection_map_status':'SUPPORT_REPAIRED_NOT_SELECTION_CALIBRATION','output':str(out/'posterior.npz')}),flush=True)
