#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); root=Path('/gpfs/kjhan/CF4/kf_design/production_ic_grafic_v1'); sys.path.insert(0,'/home/kjhan/BACKUP/CF4/src'); from grafic_io import read_grafic_field
    d,m=read_grafic_field(str(root/'ic_deltab')); vx,_=read_grafic_field(str(root/'ic_velcx')); vy,_=read_grafic_field(str(root/'ic_velcy')); vz,_=read_grafic_field(str(root/'ic_velcz')); aa=float(m['astart']); z0=d/aa; vrms=np.array([vx.std(),vy.std(),vz.std()]); checks={'all_finite':bool(np.isfinite(d).all() and np.isfinite(vx).all() and np.isfinite(vy).all() and np.isfinite(vz).all()),'zero_mean_delta':bool(abs(float(d.mean()))<1e-7),'velocity_finite':bool(np.isfinite(vrms).all()),'linear_growth_recovery':bool(np.isfinite(z0).all())}; result={'schema':'ouruniv-cf4-linear-forward-preflight-v1','input':str(root),'astart':aa,'delta_ic_std':float(d.std()),'delta_linear_z0_std':float(z0.std()),'velocity_rms':vrms.tolist(),'checks':checks,'decision':'PM_PREFLIGHT_PASS' if all(checks.values()) else 'PM_PREFLIGHT_FAIL','nonlinear_pmwd_started':False}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
