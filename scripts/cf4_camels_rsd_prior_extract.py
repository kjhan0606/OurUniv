#!/usr/bin/env python3
"""Extract external velocity-conditioned FoG priors from CAMELS/SIMBA."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py,numpy as np
ROOT=Path('/gpfs/kjhan/CAMELS/SIMBA/L25n256')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    train=list(range(16,24)); valid=list(range(24,27)); masses=[]
    for cv in train:
        with h5py.File(ROOT/f'CV/CV_{cv}/groups_090.hdf5','r') as h: masses.append(np.asarray(h['Subhalo/SubhaloMassType'][:,4],float))
    m=np.concatenate(masses); m=m[m>0]; bins=np.unique(np.quantile(m,[0,1/6,2/6,3/6,4/6,5/6,1.0])).astype(float)
    rows=[]
    for cv in train+valid:
        with h5py.File(ROOT/f'CV/CV_{cv}/groups_090.hdf5','r') as h:
            mass=np.asarray(h['Subhalo/SubhaloMassType'][:,4],float); vz=np.asarray(h['Subhalo/SubhaloVel'][:,2],float)
        keep=mass>0; mass=mass[keep]; vz=vz[keep]; pop=np.clip(np.searchsorted(bins,mass,side='right')-1,0,len(bins)-2)
        for p in range(len(bins)-1):
            v=vz[pop==p]; med=float(np.median(v)); sig=float(np.std(v)); mad=float(1.4826*np.median(np.abs(v-med)))
            rows.append({'cv':cv,'population':p,'n':int(v.size),'velocity_mean_km_s':med,'velocity_std_km_s':sig,'velocity_mad_km_s':mad,'fog_sigma_cMpc_h_std':sig/100.0,'fog_sigma_cMpc_h_mad':mad/100.0,'split':'train' if cv in train else 'validation'})
    arr=np.array([[r['population'],r['fog_sigma_cMpc_h_std'],r['fog_sigma_cMpc_h_mad']] for r in rows if r['split']=='train'])
    priors=[]
    for p in range(len(bins)-1):
        z=arr[arr[:,0]==p,1:]; priors.append({'population':p,'sigma_cMpc_h_mean':float(np.mean(z[:,0])),'sigma_cMpc_h_sd_across_CV':float(np.std(z[:,0],ddof=1)),'mad_sigma_cMpc_h_mean':float(np.mean(z[:,1]))})
    result={'schema':'ouruniv-cf4-camels-rsd-prior-v1','source':'CAMELS SIMBA CV16-26 group velocities','bins':bins.tolist(),'train_realizations':train,'validation_realizations':valid,'train_priors':priors,'row_statistics':rows,'interpretation':'Velocity-conditioned external prior; not a CF4 bias calibration and not an NB-k prior. Apply as a radial FoG kernel prior in the CF4 likelihood.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
