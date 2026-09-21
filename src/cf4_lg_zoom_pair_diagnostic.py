#!/usr/bin/env python3
"""Read-only LG-anchor HOP pair diagnostic for a completed zoom snapshot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from cf4_zoom_z0_gate import catalog_from_hop_tags, particle_files, read_info, scan_mass_species

MSUN_G = 1.98847e33
ANCHOR = np.array([189.25386007706237, 191.60229489929446, 192.32376166515513])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--hop-tags', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--box', type=float, default=384.0)
    args = ap.parse_args()
    info = read_info(args.output)
    files = particle_files(args.output)
    mass_unit = info['unit_d'] * info['unit_l']**3 / MSUN_G * (info['H0']/100.0)
    velocity_unit = info['unit_l'] / info['unit_t'] / 1e5
    ntotal, species = scan_mass_species(files, mass_unit)
    fine_mass = species[0]['mass_code']
    cat = catalog_from_hop_tags(args.output, args.hop_tags, args.box, mass_unit, velocity_unit, fine_mass)
    pos = np.asarray(cat['pos'], float); mass = np.asarray(cat['mass'], float)
    vel = np.asarray(cat['vel'], float); contam = np.asarray(cat['contamination_fof'], float)
    dx = pos - ANCHOR[None, :]; dx -= args.box*np.round(dx/args.box)
    r = np.linalg.norm(dx, axis=1)
    local = np.flatnonzero(r <= 8.0)
    rows=[]
    for ii in range(len(local)):
        i=int(local[ii])
        for jj in range(ii+1,len(local)):
            j=int(local[jj]); sep=float(np.linalg.norm(dx[i]-dx[j]-args.box*np.round((dx[i]-dx[j])/args.box)))
            if not (0.3 <= sep <= 1.2): continue
            if not (5e11 <= mass[i] <= 4e12 and 5e11 <= mass[j] <= 4e12): continue
            dr=dx[j]-dx[i]; dr-=args.box*np.round(dr/args.box)
            rows.append({'i':i,'j':j,'group_i':int(cat['group_id'][i]),'group_j':int(cat['group_id'][j]),
                         'sep_cMpc_h':sep,'mass_i_msun_h':float(mass[i]),'mass_j_msun_h':float(mass[j]),
                         'r_i_cMpc_h':float(r[i]),'r_j_cMpc_h':float(r[j]),
                         'pos_i_cMpc_h':pos[i].tolist(),'pos_j_cMpc_h':pos[j].tolist(),
                         'pair_displacement_j_minus_i_cMpc_h':dr.tolist(),
                         'contam_i':float(contam[i]),'contam_j':float(contam[j]),
                         'v_i_km_s':vel[i].tolist(),'v_j_km_s':vel[j].tolist()})
    rows.sort(key=lambda x: (x['contam_i']+x['contam_j'], abs(x['sep_cMpc_h']-0.78)))
    result={'schema':'ouruniv-cf4-lg-zoom-hop-pair-diagnostic-v1','stage':'8/8',
            'snapshot':str(args.output),'aexp':float(info['aexp']),'particle_count':int(ntotal),
            'anchor_cMpc_h':ANCHOR.tolist(),'local_group_count_r8':int(len(local)),
            'candidate_count':len(rows),'candidates':rows[:100],
            'interpretation':'Candidate screening only; no MW/M31/M33 promotion or final reproduction claim.'}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'candidate_count':len(rows),'local_group_count_r8':int(len(local)),'out':str(args.out)},sort_keys=True))
if __name__=='__main__': main()
