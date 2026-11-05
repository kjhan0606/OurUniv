#!/usr/bin/env python3
"""Independent seeded recovery test for the NB joint-tracer estimator."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize


def fit(y, e, x, r, train):
    rr=(r-np.median(r[train]))/60.0; yy,ee,xx,rt=y[train],e[train],x[train],rr[train]
    def nll(t,yv,ev,xv,rv):
        b,eta,lk=t; k=np.exp(np.clip(lk,-8,15)); z=ev*np.exp(np.clip(b*xv+eta*rv,-30,30)); A=yv.sum()/max(z.sum(),1e-30); mu=np.maximum(A*z,1e-12)
        return float(-np.sum(gammaln(yv+k)-gammaln(k)-gammaln(yv+1)+k*np.log(k/(k+mu))+yv*np.log(mu/(k+mu))))
    o=minimize(lambda t:nll(t,yy,ee,xx,rt),[0.8,0.3,0.0],method='L-BFGS-B',bounds=[(.01,6),(-2,2),(-8,15)])
    return np.asarray(o.x), np.exp(o.x[2])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260920); a=ap.parse_args()
    rng=np.random.default_rng(a.seed); N=32; n=N**3
    # Independent covariate/exposure geometry; no CF4 or Carrick counts are reused.
    q=np.arange(n,dtype=float); x=0.35*np.sin(q/173.0)+0.12*np.cos(q/47.0); r=5+175*((q%N)+.5)/N
    e=np.exp(-r/125.0)*(0.35+0.65*(0.5+0.5*np.sin(q/911.0))); train=(q.astype(int)%5)!=0
    truth=[]; recovered=[]
    for p in range(6):
        b=0.45+0.18*p; eta=-0.35+0.12*p; k=0.18+0.07*p; A=2.5+0.6*p
        mu=A*e*np.exp(b*x+eta*(r-np.median(r))/60.0)
        # Gamma-Poisson construction gives exact NB counts with mean mu and shape k.
        lam=rng.gamma(shape=k,scale=mu/k); y=rng.poisson(lam)
        est, khat=fit(y,e,x,r,train); truth.append({'population':p,'bias':b,'radial_nuisance':eta,'nb_k':k}); recovered.append({'population':p,'bias':float(est[0]),'radial_nuisance':float(est[1]),'nb_k':float(khat),'abs_error':{'bias':abs(float(est[0])-b),'radial_nuisance':abs(float(est[1])-eta),'nb_k':abs(float(khat)-k)}})
    errs=np.array([[v['abs_error']['bias'],v['abs_error']['radial_nuisance'],v['abs_error']['nb_k']] for v in recovered])
    result={'schema':'ouruniv-cf4-joint-tracer-mock-validation-v1','seed':a.seed,'independent_from_observed_counts':True,'grid_N':N,'truth':truth,'recovered':recovered,'median_abs_error':dict(zip(['bias','radial_nuisance','nb_k'],np.median(errs,axis=0).tolist())),'gates':{'all_bias_error_lt_0.25':bool(np.all(errs[:,0]<.25)),'all_radial_error_lt_0.35':bool(np.all(errs[:,1]<.35)),'all_nb_k_relative_error_lt_1':bool(np.all(errs[:,2]/np.array([v['nb_k'] for v in truth])<1.0))}}
    result['status']='PASS_ESTIMATOR_RECOVERY' if all(result['gates'].values()) else 'FAIL_ESTIMATOR_RECOVERY'; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))

if __name__=='__main__': main()
