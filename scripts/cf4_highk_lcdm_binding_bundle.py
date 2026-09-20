#!/usr/bin/env python3
"""Bind LCDM high-k normalization and audit low/high-k continuity."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
TRANSFER=Path('/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_grafic_p3429_s5108_n576/transfer_p3429_s5108_n576.npz')
BLEND=Path('/gpfs/kjhan/CF4/q1_low_highk_blend_dev_v1.npz')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    t=np.load(TRANSFER,allow_pickle=False); b=np.load(BLEND,allow_pickle=False); keys=list(t.files); field=np.asarray(b['field'],float); finite=bool(np.isfinite(field).all())
    k=None
    for name in ('k','kphys','kf','k_table','k_grid'):
        if name in t and np.asarray(t[name]).ndim==1: k=np.asarray(t[name],float); break
    transfer_name=next((n for n in ('transfer','T','delta_transfer','sqrtP') if n in t),None)
    # Development binding: primordial As/ns plus the pinned transfer table define the target shape.
    As=1.63e-9; ns=.96; kp=.05
    shape_ok=k is not None and transfer_name is not None and np.isfinite(k).all() and np.isfinite(t[transfer_name]).all()
    target=None
    if shape_ok:
        tr=np.asarray(t[transfer_name],float); target=As*(np.maximum(k,1e-8)/kp)**(ns-1.0)*tr**2
    fk=np.fft.fftn(field); n=field.shape[0]; box=384.; kk=2*np.pi*np.fft.fftfreq(n,d=box/n); KX,KY,KZ=np.meshgrid(kk,kk,kk,indexing='ij'); km=np.sqrt(KX*KX+KY*KY+KZ*KZ); power=np.abs(fk)**2/field.size**2
    band=(km>=.25)&(km<=.5)&(power>0); continuity_ratio=float(np.median(power[band])/max(np.median(power[km>=.5]),1e-30)) if np.any(band) else float('nan')
    bound_path=Path('/gpfs/kjhan/CF4/kf_design/highk_lcdm_binding_bundle_v1/field_lcdm_bound.npz'); bound_written=False; bound_ratio=None
    if shape_ok:
        tk=np.asarray(k,float); tv=np.asarray(target,float); order=np.argsort(tk); tgt=np.interp(km.ravel(),tk[order],tv[order],left=tv[order][0],right=tv[order][-1]).reshape(km.shape); band2=(km>=.35)&(km<=1.5)&(power>0)&(tgt>0); bound_ratio=float(np.median(tgt[band2]/power[band2])); fkb=fk.copy(); fkb[km>=.35]*=np.sqrt(bound_ratio); bound_field=np.fft.ifftn(fkb).real; bound_path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(bound_path,field=bound_field,As=As,ns=ns,pivot_h_Mpc=kp,normalization_factor=bound_ratio); bound_written=True
    bound_continuity=None
    if bound_written:
        pb=np.abs(np.fft.fftn(bound_field))**2/bound_field.size**2; bound_continuity=float(np.median(pb[band])/max(np.median(pb[km>=.5]),1e-30))
    checks={'transfer_table_present':TRANSFER.is_file(),'blend_present':BLEND.is_file(),'field_finite':finite,'lcdm_shape_bound':bool(shape_ok),'lcdm_bound_field_written':bound_written,'low_high_continuity_finite':bool(np.isfinite(continuity_ratio)),'bound_field_continuity_finite':bool(bound_continuity is not None and np.isfinite(bound_continuity))}
    result={'schema':'ouruniv-cf4-highk-lcdm-binding-bundle-v1','transfer_keys':keys,'transfer_field':transfer_name,'lcdm_binding':{'As':As,'ns':ns,'pivot_h_Mpc':kp,'shape_bound':bool(shape_ok),'bound_field':str(bound_path) if bound_written else None,'normalization_factor':bound_ratio},'power_audit':{'continuity_ratio_original':continuity_ratio,'continuity_ratio_bound_field':bound_continuity,'low_k_cut':float(b['low_k_cut']) if 'low_k_cut' in b else None,'taper_width':float(b['taper_width']) if 'taper_width' in b else None},'checks':checks,'decision':'IC_PREFLIGHT_GO' if all(checks.values()) else 'BLOCK_IC_PRODUCTION','next':'Generate IC preflight only after bound field is independently re-audited.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
