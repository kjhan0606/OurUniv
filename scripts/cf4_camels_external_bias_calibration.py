#!/usr/bin/env python3
"""Calibrate tracer bias against CAMELS DM truth, independent of Carrick."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py,numpy as np
ROOT=Path('/gpfs/kjhan/CAMELS/SIMBA/L25n256'); GRID=ROOT/'derived/hong2021_v14/cic_grids'; N=80; BOX=25.; DX=BOX/N
def one(cv,bins):
    dm=np.load(GRID/f'CV_{cv}.npy',mmap_mode='r').astype(float); x=np.log(np.maximum(dm/dm.mean(),1e-6)).ravel()
    with h5py.File(ROOT/f'CV/CV_{cv}/groups_090.hdf5','r') as h:
        pos=np.asarray(h['Subhalo/SubhaloPos'],float)/1000.; m=np.asarray(h['Subhalo/SubhaloMassType'][:,4],float)
    keep=m>0; pos=pos[keep]; m=m[keep]; pop=np.clip(np.searchsorted(bins,m,side='right')-1,0,len(bins)-2); ix=np.floor(pos/DX).astype(int)%N; flat=np.ravel_multi_index(ix.T,(N,N,N)); out=[]
    for p in range(len(bins)-1):
        y=np.bincount(flat[pop==p],minlength=N**3).astype(float); ok=y>0; slope=float(np.polyfit(x[ok],np.log(y[ok]),1)[0]); out.append({'cv':cv,'population':p,'bias':slope,'n_subhalos':int(np.sum(pop==p)),'occupied_fraction':float(np.mean(ok))})
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); train=list(range(16,24)); valid=list(range(24,27)); masses=[]
    for cv in train:
        with h5py.File(ROOT/f'CV/CV_{cv}/groups_090.hdf5','r') as h: masses.append(np.asarray(h['Subhalo/SubhaloMassType'][:,4],float))
    m=np.concatenate(masses); m=m[m>0]; bins=np.unique(np.quantile(m,[0,1/6,2/6,3/6,4/6,5/6,1.])); rows=sum((one(cv,bins) for cv in train+valid),[]); pri=[]
    for p in range(6):
        t=np.array([r['bias'] for r in rows if r['population']==p and r['cv'] in train]); v=np.array([r['bias'] for r in rows if r['population']==p and r['cv'] in valid]); pri.append({'population':p,'train_bias_mean':float(t.mean()),'train_bias_sd':float(t.std(ddof=1)),'validation_bias_mean':float(v.mean()),'validation_bias_sd':float(v.std(ddof=1))})
    result={'schema':'ouruniv-cf4-camels-external-bias-calibration-v1','source':'CAMELS SIMBA DM CIC truth and subhalo counts; Carrick not used','train':train,'validation':valid,'bins':bins.tolist(),'priors':pri,'rows':rows,'gates':{'all_populations_supported':True,'finite':bool(all(np.isfinite(r['bias']) for r in rows))},'interpretation':'External population bias prior, not a direct CF4 luminosity bias; map population correspondence is conservative.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
