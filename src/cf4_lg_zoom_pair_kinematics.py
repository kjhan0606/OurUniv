#!/usr/bin/env python3
"""Compute kinematic diagnostics for screened LG HOP pairs."""
import argparse, json
from pathlib import Path
import numpy as np

ANCHOR=np.array([189.25386007706237,191.60229489929446,192.32376166515513])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--box',type=float,default=384.0); args=ap.parse_args()
    d=json.loads(args.input.read_text()); rows=[]; h0=74.6
    for x in d['candidates']:
        pi=np.array(x['v_i_km_s']); pj=np.array(x['v_j_km_s'])
        # Positions are not retained in the screening JSON, so kinematics are
        # reported as velocity-space diagnostics only; pair separation is the
        # already verified periodic position separation from the upstream scan.
        dv=pj-pi; speed=float(np.linalg.norm(dv))
        dr=np.asarray(x['pair_displacement_j_minus_i_cMpc_h'],float); sep=float(np.linalg.norm(dr)); rhat=dr/sep
        vtot=dv+h0*dr; vr=float(np.dot(vtot,rhat)); vtan=float(np.sqrt(max(0.0,np.dot(vtot,vtot)-vr*vr)))
        rows.append({**x,'relative_speed_km_s':speed,'relative_velocity_km_s':dv.tolist(),
                     'hubble_corrected_velocity_km_s':vtot.tolist(),'radial_velocity_km_s':vr,
                     'tangential_velocity_km_s':vtan,'radial_approaching':vr<0.0})
    result={**d,'kinematics':rows,'interpretation':d['interpretation']+' Hubble-corrected radial and tangential pair velocities are diagnostic only; no target promotion is performed.'}
    args.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'candidate_count':len(rows),'out':str(args.out)},sort_keys=True))
if __name__=='__main__': main()
