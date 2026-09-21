#!/usr/bin/env python3
"""Compute kinematic diagnostics for screened LG HOP pairs."""
import argparse, json
from pathlib import Path
import numpy as np

ANCHOR=np.array([189.25386007706237,191.60229489929446,192.32376166515513])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--box',type=float,default=384.0); args=ap.parse_args()
    d=json.loads(args.input.read_text()); rows=[]
    for x in d['candidates']:
        pi=np.array(x['v_i_km_s']); pj=np.array(x['v_j_km_s'])
        # Positions are not retained in the screening JSON, so kinematics are
        # reported as velocity-space diagnostics only; pair separation is the
        # already verified periodic position separation from the upstream scan.
        dv=pj-pi; speed=float(np.linalg.norm(dv))
        # Radial sign is not identifiable without the pair displacement vector;
        # retain this explicitly rather than inventing an orientation.
        rows.append({**x,'relative_speed_km_s':speed,'relative_velocity_km_s':dv.tolist(),
                     'radial_approach_status':'UNAVAILABLE_POSITION_VECTOR'})
    result={**d,'kinematics':rows,'interpretation':d['interpretation']+' Relative speed is reported; radial approach requires retaining pair displacement vectors in a follow-up diagnostic.'}
    args.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'candidate_count':len(rows),'out':str(args.out)},sort_keys=True))
if __name__=='__main__': main()
