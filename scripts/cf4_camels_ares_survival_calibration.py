#!/usr/bin/env python3
"""External survival calibration: CAMELS subhalos thinned by official ARES maps."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py,numpy as np, healpy as hp
ROOT=Path('/gpfs/kjhan/CAMELS/SIMBA/L25n256'); MAP=Path('/gpfs/kjhan/CF4/software/ares-6cf608ed/examples/completeness_12_5.fits.gz')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260920); a=ap.parse_args(); rng=np.random.default_rng(a.seed)
    comp=np.asarray(hp.read_map(str(MAP),field=0,nest=False,dtype=np.float64)); nside=hp.npix2nside(comp.size); rows=[]
    for cv in (24,25,26):
        with h5py.File(ROOT/f'CV/CV_{cv}/groups_090.hdf5','r') as h:
            pos=np.asarray(h['Subhalo/SubhaloPos'],float)/1000.; mass=np.asarray(h['Subhalo/SubhaloMassType'][:,4],float)
        keep=mass>0; pos=pos[keep]; mass=mass[keep]; d=pos-12.5; r=np.sqrt(np.sum(d*d,axis=1)); keep=r>0; d=d[keep]; mass=mass[keep]; r=r[keep]
        theta=np.arccos(np.clip(d[:,2]/r,-1,1)); phi=np.mod(np.arctan2(d[:,1],d[:,0]),2*np.pi); p=hp.ang2pix(nside,theta,phi,nest=False); c=comp[p]; survived=rng.random(c.size)<c; pop=np.clip(np.searchsorted(np.quantile(mass,[0,1/6,2/6,3/6,4/6,5/6,1.0]),mass,side='right')-1,0,5)
        for j in range(6):
            z=pop==j; rows.append({'cv':cv,'population':j,'n_total':int(z.sum()),'n_survived':int(np.sum(survived[z])),'mean_ares_completeness':float(np.mean(c[z])),'realized_survival':float(np.mean(survived[z])),'binomial_se':float(np.sqrt(max(np.mean(survived[z])*(1-np.mean(survived[z])),0)/max(z.sum(),1)))})
    result={'schema':'ouruniv-cf4-camels-ares-survival-calibration-v1','source':'CAMELS SIMBA CV24-26 subhalos + official ARES completeness_12_5','seed':a.seed,'nside':nside,'rows':rows,'gates':{'all_populations_nonempty':all(r['n_total']>0 for r in rows),'finite':bool(all(np.isfinite(r['realized_survival']) for r in rows))},'interpretation':'External angular-survival calibration. Radial luminosity selection and CF4 luminosity matching remain separate nuisance terms; this result must not be treated as a direct CF4 truth.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
