#!/usr/bin/env python3
"""External CAMELS/SIMBA validation of tracer bias and physical RSD impact."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np

ROOT=Path('/gpfs/kjhan/CAMELS/SIMBA/L25n256')
GRID=ROOT/'derived/hong2021_v14/cic_grids'
BOX=25.0; N=80; DX=BOX/N

def load_cv(cv, bins):
    cat=ROOT/f'CV/CV_{cv}/groups_090.hdf5'; grid=np.load(GRID/f'CV_{cv}.npy',mmap_mode='r').astype(np.float64)
    delta=np.log(np.maximum(grid/grid.mean(),1e-6)).ravel()
    with h5py.File(cat,'r') as h:
        pos=np.asarray(h['Subhalo/SubhaloPos'],dtype=float)/1000.0
        vel=np.asarray(h['Subhalo/SubhaloVel'],dtype=float)
        mass=np.asarray(h['Subhalo/SubhaloMassType'][:,4],dtype=float)
    positive=mass>0; pos=pos[positive]; vel=vel[positive]; mass=mass[positive]
    pop=np.clip(np.searchsorted(bins,mass,side='right')-1,0,len(bins)-2)
    counts=[]
    for rsd in (False,True):
        p=pos.copy()
        if rsd: p[:,2]=(p[:,2]+vel[:,2]/100.0)%BOX
        ix=np.floor(p/DX).astype(int)%N; flat=np.ravel_multi_index(ix.T,(N,N,N))
        arr=np.zeros((len(bins)-1,N**3),dtype=np.float64)
        for k in range(arr.shape[0]): arr[k]=np.bincount(flat[pop==k],minlength=N**3)
        counts.append(arr)
    return delta,counts[0],counts[1],int(pos.shape[0])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    train=list(range(16,24)); valid=list(range(24,27));
    # Fixed bins in the simulation's stellar-mass code units; selected from train only.
    masses=[]
    for cv in train:
        with h5py.File(ROOT/f'CV/CV_{cv}/groups_090.hdf5','r') as h: masses.append(np.asarray(h['Subhalo/SubhaloMassType'][:,4],float))
    m=np.concatenate(masses); m=m[m>0]; bins=np.unique(np.quantile(m,[0,.2,.4,.6,.8,1.0])).astype(float)
    train_rows=[]; val_rows=[]
    for cv in train+valid:
        d,real,rsd,n=load_cv(cv,bins)
        target=train_rows if cv in train else val_rows
        for p in range(real.shape[0]):
            # Fit external bias on train realization cells with nonzero counts.
            target.append((cv,p,d,real[p],rsd[p],n))
    fit=[]
    for p in range(real.shape[0]):
        d=np.concatenate([r[2] for r in train_rows if r[1]==p]); y=np.concatenate([r[3] for r in train_rows if r[1]==p]);
        ok=y>0; slope=float(np.polyfit(d[ok],np.log(y[ok]),1)[0]); fit.append(slope)
    val=[]
    for cv,p,d,real,rsd,n in val_rows:
        pred=np.exp(np.median([np.log(np.maximum(real.mean(),1e-6))])+fit[p]*d)
        def metrics(y):
            ok=y>0; corr=float(np.corrcoef(np.log1p(y),d)[0,1]); over=float(np.var(y-pred)/max(np.mean(pred),1e-9)); return {'log_density_corr':corr,'count_overdispersion':over,'nonzero_fraction':float(np.mean(ok))}
        val.append({'cv':cv,'population':p,'external_bias_train':fit[p],'real_space':metrics(real),'redshift_space':metrics(rsd),'rsd_delta_corr':metrics(rsd)['log_density_corr']-metrics(real)['log_density_corr'],'subhalo_count':n})
    result={'schema':'ouruniv-cf4-camels-simba-external-rsd-validation-v1','source':'CAMELS SIMBA L25n256 CV groups_090 + CIC DM grids','train_realizations':train,'validation_realizations':valid,'stellar_mass_bins':bins.tolist(),'grid':{'N':N,'box_Mpc_h':BOX,'cell_Mpc_h':DX,'rsd_shift':'z position += v_z/100 Mpc/h at z=0'},'population_results':val,'gates':{'external_velocity_present':True,'train_validation_disjoint':True,'all_validation_finite':bool(all(np.isfinite(v['redshift_space']['log_density_corr']) for v in val))},'production_IC_allowed':False,'interpretation':'Physical external RSD validation; CAMELS tracer population is not CF4-calibrated and does not by itself authorize production.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
