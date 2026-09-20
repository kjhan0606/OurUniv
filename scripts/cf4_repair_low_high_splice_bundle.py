#!/usr/bin/env python3
"""Repair low/high-k splice by a bounded low-k amplitude fit and audit it."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
SRC=Path('/gpfs/kjhan/CF4/kf_design/highk_lcdm_binding_bundle_v1/field_lcdm_bound.npz')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    z=np.load(SRC,allow_pickle=False); field=np.asarray(z['field'],float); n=field.shape[0]; box=384.; dx=box/n; fk=np.fft.fftn(field); q=2*np.pi*np.fft.fftfreq(n,d=dx); KX,KY,KZ=np.meshgrid(q,q,q,indexing='ij'); km=np.sqrt(KX*KX+KY*KY+KZ*KZ)
    p=np.abs(fk)**2/field.size**2; low=(km>=.25)&(km<.30)&(p>0); high=(km>=.50)&(p>0); ratio=float(np.median(p[low])/max(np.median(p[high]),1e-30)); alpha=float(np.clip(1/np.sqrt(ratio),.25,1.0)); taper=np.ones_like(km); taper[km<.25]=alpha; mid=(km>=.25)&(km<.35); taper[mid]=alpha+(1-alpha)*(km[mid]-.25)/.10; taper[km>=.35]=1.0; repaired=np.fft.ifftn(fk*taper).real; pr=np.abs(np.fft.fftn(repaired))**2/repaired.size**2; repaired_ratio=float(np.median(pr[low])/max(np.median(pr[high]),1e-30)); out=a.output; out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,field=repaired,low_k_scale=alpha,source=str(SRC),continuity_ratio=repaired_ratio)
    result={'schema':'ouruniv-cf4-low-high-splice-repair-bundle-v1','source':str(SRC),'low_high_ratio_before':ratio,'low_k_scale':alpha,'low_high_ratio_after':repaired_ratio,'checks':{'source_finite':bool(np.isfinite(field).all()),'repaired_finite':bool(np.isfinite(repaired).all()),'continuity_in_range':bool(.5<=repaired_ratio<=2.0),'low_k_scale_bounded':bool(.25<=alpha<=1.0)},'decision':'IC_PREFLIGHT_GO' if (.5<=repaired_ratio<=2.0 and np.isfinite(repaired).all()) else 'BLOCK_IC_PRODUCTION','ic_preflight':{'field':str(out),'production_evolution_started':False},'warning':'Low-k amplitude was rescaled; posterior-consistency audit is still required.'}
    (out.parent/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
