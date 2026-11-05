import csv, json, h5py
from pathlib import Path
import numpy as np
from cf4_2mpp_joint_likelihood_local import tsc_deposit

rows=[]
with Path('data/cf4_galaxies.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        try:
            vals=[float(row[k]) for k in ('DM','e_DM','SGL','SGB','Vcmb')]
            if np.all(np.isfinite(vals)) and vals[1]>=0: rows.append(vals)
        except (TypeError,ValueError): continue
cat=np.asarray(rows); h=.746; H=74.6; box=384.; n=256; dx=box/n; obs=np.full(3,box/2.); edges=np.array([5.,30.,60.,90.,120.,150.,180.]); rng=np.random.default_rng(20261006)
maps={}
for name in ('v1','v2'):
    path=f'/gpfs/kjhan/CF4/z0_density/bundle_c_v1/selection_1p5_{name}/selection.h5'
    with h5py.File(path,'r') as f: maps[name]=np.asarray(f['selection_shells']).sum(axis=1)
variants={k:[] for k in ('no_selection','v1','v2')}
for draw in range(4):
    dm=cat[:,0]+rng.normal(size=cat.shape[0])*cat[:,1]; dist=h*10**((dm-25)/5); sgl=np.deg2rad(cat[:,2]); sgb=np.deg2rad(cat[:,3]); u=np.column_stack((np.cos(sgb)*np.cos(sgl),np.cos(sgb)*np.sin(sgl),np.sin(sgb))); pos=(obs+dist[:,None]*u)%box; pos=(pos+(h*cat[:,4]/H)[:,None]*u)%box; rel=(pos-obs+box/2)%box-box/2; radius=np.linalg.norm(rel,axis=1); inside=(radius>=5)&(radius<=180)&np.all(np.abs(rel)<box/2,axis=1); pos=pos[inside]; radius=radius[inside]; cell=np.floor(pos/dx).astype(int)%n; shell=np.clip(np.searchsorted(edges,radius,side='right')-1,0,5)
    weights={'no_selection':np.ones(pos.shape[0])}
    for name in ('v1','v2'): weights[name]=maps[name][shell*0+0,cell[:,0],cell[:,1],cell[:,2]] if False else maps[name][:,cell[:,0],cell[:,1],cell[:,2]].sum(axis=0)
    for name,w in weights.items(): variants[name].append(tsc_deposit(pos,w,n,box))
out=Path('/gpfs/kjhan/CF4/q1_z0_selection_sensitivity_v1'); out.mkdir(parents=True,exist_ok=True); arrays={}
for name,draws in variants.items(): arrays[f'{name}_mean']=np.mean(draws,axis=0); arrays[f'{name}_std']=np.std(draws,axis=0); arrays[f'{name}_totals']=np.asarray([x.sum() for x in draws])
np.savez_compressed(out/'posterior_sensitivity.npz',**arrays)
summary={'status':'Z0_SELECTION_SENSITIVITY_DEVELOPMENT','catalog_rows':int(cat.shape[0]),'draws':4,'grid':n,'box_cMpc_h':box,'variants':{k:{'mean_total':float(arrays[k+'_mean'].sum()),'relative_draw_std':float(arrays[k+'_totals'].std()/arrays[k+'_totals'].mean())} for k in variants},'selection_status':'v1/v2 support maps are not calibrated','output':str(out/'posterior_sensitivity.npz')}; print(json.dumps(summary,sort_keys=True),flush=True)
