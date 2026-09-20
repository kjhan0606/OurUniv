#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); src=Path('/gpfs/kjhan/CF4/kf_design/low_high_splice_repair_bundle_v1/repaired_field.npz'); z=np.load(src,allow_pickle=False); f=np.asarray(z['field'],float); p=np.abs(np.fft.fftn(f))**2/f.size**2; q=2*np.pi*np.fft.fftfreq(f.shape[0],d=384/f.shape[0]); X,Y,Z=np.meshgrid(q,q,q,indexing='ij'); k=np.sqrt(X*X+Y*Y+Z*Z); ratio=float(np.median(p[(k>=.25)&(k<.30)])/max(np.median(p[k>=.50]),1e-30)); checks={'source_present':src.is_file(),'shape_256':f.shape==(256,256,256),'finite':bool(np.isfinite(f).all()),'zero_mean':bool(abs(float(f.mean()))<1e-8),'continuity_in_range':bool(.5<=ratio<=2.0)}; result={'schema':'ouruniv-cf4-ic-preflight-v1','source':str(src),'shape':list(f.shape),'mean':float(f.mean()),'std':float(f.std()),'power_ratio':ratio,'checks':checks,'decision':'IC_PREFLIGHT_PASS' if all(checks.values()) else 'IC_PREFLIGHT_FAIL','production_run_started':False}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
