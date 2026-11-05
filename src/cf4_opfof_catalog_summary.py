#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
ANCHOR=np.array([189.25386007706237,191.60229489929446,192.32376166515513])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    z=np.load(args.catalog); mass=np.asarray(z['mass'],float); pos=np.asarray(z['pos'],float); n=np.asarray(z['n'],int)
    d=pos-ANCHOR[None,:]; d-=384*np.round(d/384); r=np.linalg.norm(d,axis=1)
    order=np.argsort(-mass)
    result={'schema':'ouruniv-cf4-opfof-catalog-summary-v1','catalog':str(args.catalog),
            'halo_count':int(len(mass)),'max_mass_msun_h':float(mass.max()) if len(mass) else 0.0,
            'mass_ge_5e11':int(np.count_nonzero(mass>=5e11)),'mass_ge_1e11':int(np.count_nonzero(mass>=1e11)),
            'local_r8_count':int(np.count_nonzero(r<=8.0)),
            'top20':[{'mass_msun_h':float(mass[i]),'n':int(n[i]),'pos_cMpc_h':pos[i].tolist(),'r_anchor_cMpc_h':float(r[i])} for i in order[:20]],
            'interpretation':'OPFoF catalog uses the finest-particle extraction in the current gate; compare its mass completeness with multi-mass HOP before using it for MW/M31 claims.'}
    args.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:result[k] for k in ('halo_count','max_mass_msun_h','mass_ge_5e11','local_r8_count')},sort_keys=True))
if __name__=='__main__': main()
