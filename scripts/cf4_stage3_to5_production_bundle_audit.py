#!/usr/bin/env python3
"""Bundle audit: calibrated tracer -> z=0 posterior -> low/high-k -> IC gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    v8=Path('/gpfs/kjhan/CF4/kf_design/joint_tracer_calibration_v8/result.json'); sel=Path('/gpfs/kjhan/CF4/q1_z0_selection_sensitivity_v1/posterior_sensitivity.npz'); blend=Path('/gpfs/kjhan/CF4/q1_low_highk_blend_dev_v1.npz')
    r=json.loads(v8.read_text()); checks={'v8_artifact_exists':v8.is_file(),'v8_all_holdout_positive':all(x['log_score_improvement']>0 for x in r['population_results'])}
    if sel.is_file():
        z=np.load(sel,allow_pickle=False); checks.update({'selection_arrays_finite':bool(all(np.isfinite(z[k]).all() for k in z.files)),'selection_sensitivity_present':len(z.files)>0}); sel_keys=z.files
    else: checks.update({'selection_arrays_finite':False,'selection_sensitivity_present':False}); sel_keys=[]
    if blend.is_file():
        b=np.load(blend,allow_pickle=False); checks.update({'low_high_arrays_finite':bool(all(np.isfinite(b[k]).all() for k in b.files)),'low_high_artifact_present':len(b.files)>0}); blend_keys=b.files
    else: checks.update({'low_high_arrays_finite':False,'low_high_artifact_present':False}); blend_keys=[]
    # The existing blend is explicitly development-only until absolute high-k normalization is bound.
    checks['high_k_absolute_normalization_bound']=False
    checks['ic_production_gate']=False
    result={'schema':'ouruniv-cf4-stage3-to5-production-bundle-audit-v1','inputs':{'v8':str(v8),'selection_keys':list(sel_keys),'blend_keys':list(blend_keys)},'checks':checks,'decision':'BLOCK_IC_PRODUCTION' if not all(checks.values()) else 'IC_PREFLIGHT_GO','blocking_reasons':['high-k absolute normalization remains development/unbound','low/high-k blend is not a calibrated LCDM posterior'] if not checks['high_k_absolute_normalization_bound'] else [],'next_required':'Bind calibrated LCDM high-k amplitude and rerun continuity/power audit before IC generation.','stages':{'stage3_z0_posterior':'development_audited','stage4_low_highk':'development_not_production','stage5_IC':'blocked'}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
