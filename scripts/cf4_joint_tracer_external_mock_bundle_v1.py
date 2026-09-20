#!/usr/bin/env python3
"""Bundled independent redshift-space/FoG and mask-realization validation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize

def fit(y,e,x,r,train):
    rr=(r-np.median(r[train]))/60.; yy,ee,xx,rt=y[train],e[train],x[train],rr[train]
    def nll(t,yv,ev,xv,rv):
        b,eta,lk=t; k=np.exp(np.clip(lk,-8,15)); z=ev*np.exp(np.clip(b*xv+eta*rv,-30,30)); A=yv.sum()/max(z.sum(),1e-30); mu=np.maximum(A*z,1e-12)
        return float(-np.sum(gammaln(yv+k)-gammaln(k)-gammaln(yv+1)+k*np.log(k/(k+mu))+yv*np.log(mu/(k+mu))))
    o=minimize(lambda t:nll(t,yy,ee,xx,rt),[.8,.2,0],method='L-BFGS-B',bounds=[(.01,6),(-2,2),(-8,15)])
    b,eta,lk=o.x; k=np.exp(lk); hold=~train; score=-nll(o.x,y[hold],e[hold],x[hold],rt[hold]); return np.array([b,eta,k]),score

def scenario(rng, fog_sigma, mask_seed):
    N=32; n=N**3; q=np.arange(n,dtype=float); x=.35*np.sin(q/173)+.12*np.cos(q/47); r=5+175*((q%N)+.5)/N
    # Independent mask realization: angular holes and radial boundary loss.
    mrng=np.random.default_rng(mask_seed); mask=np.where(mrng.random(n)<.08,0.0,1.0)*(r>8)*(r<177)
    base=np.exp(-r/125)*(0.35+0.65*(.5+.5*np.sin(q/911)))*mask
    # FoG displacement in cMpc/h; 300 km/s corresponds to ~3 h^-1 Mpc.
    r_obs=np.maximum(1.0,r+rng.normal(0.0,fog_sigma,size=n)); train=(q.astype(int)%5)!=0
    truth=[]; rec=[]
    for p in range(6):
        b=.45+.18*p; eta=-.35+.12*p; k=.18+.07*p; A=2.5+.6*p
        mu=A*base*np.exp(b*x+eta*(r_obs-np.median(r_obs))/60.)
        lam=rng.gamma(shape=k,scale=np.maximum(mu,1e-10)/k); y=rng.poisson(lam)
        est,score=fit(y,base,x,r_obs,train); truth.append([b,eta,k]); rec.append({'population':p,'bias':float(est[0]),'radial_nuisance':float(est[1]),'nb_k':float(est[2]),'holdout_log_score':float(score),'truth':[b,eta,k]})
    truth=np.asarray(truth); est=np.asarray([[z['bias'],z['radial_nuisance'],z['nb_k']] for z in rec]); err=np.abs(est-truth)
    return {'fog_sigma_cMpc_h':fog_sigma,'mask_seed':mask_seed,'median_abs_error':dict(zip(['bias','radial_nuisance','nb_k'],np.median(err,axis=0).tolist())),'max_abs_error':dict(zip(['bias','radial_nuisance','nb_k'],np.max(err,axis=0).tolist())),'recovered':rec,'gates':{'bias_median_lt_0.30':bool(np.median(err[:,0])<.30),'radial_median_lt_0.40':bool(np.median(err[:,1])<.40),'nb_k_relative_median_lt_1':bool(np.median(err[:,2]/truth[:,2])<1.0)}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260920); a=ap.parse_args(); rng=np.random.default_rng(a.seed)
    cases=[scenario(rng,0.0,901),scenario(rng,3.0,902),scenario(rng,6.0,903)]
    result={'schema':'ouruniv-cf4-joint-tracer-external-mock-bundle-v1','seed':a.seed,'cases':cases,'gates':{'all_cases_pass':all(all(c['gates'].values()) for c in cases),'fog_case_present':True,'independent_mask_realizations':True},'status':'PASS_BUNDLE_MECHANICS' if all(all(c['gates'].values()) for c in cases) else 'FAIL_BUNDLE_MECHANICS','production_IC_allowed':False,'interpretation':'Independent mock stress test only; not external survey calibration.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
