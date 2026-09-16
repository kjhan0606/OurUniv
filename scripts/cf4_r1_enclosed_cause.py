"""Fixed-center enclosed moments to separate HOP boundary from field dynamics."""
import json, os
from pathlib import Path
import numpy as np
from cf4_r1_particle_entry import write
from cf4_zoom_z0_gate import _record, _skip_header

BASE = Path('/gpfs/kjhan/CF4/r1_hop/job_361459'); BOX = 12.0

def load_hop(label):
    with (BASE/label/'grp.tag').open('rb') as f:
        n, groups = map(int, _record(f, '<i4')); tag = _record(f, '<i4')
    with (BASE/label/'particles00001').open('rb') as f:
        _, nfile = _skip_header(f)
        x = np.stack([_record(f, '<f8') for _ in range(3)], axis=1)*BOX
        v = np.stack([_record(f, '<f8') for _ in range(3)], axis=1); m = _record(f, '<f8')
    assert n == nfile == len(tag) == len(x)
    return tag, x, v, m

def centers_from(tag, x, m, minimum=1000, count=32):
    ids, counts = np.unique(tag[tag >= 0], return_counts=True); ids = ids[counts >= minimum]; rows=[]
    for g in ids:
        keep=tag==g; a=2*np.pi*x[keep]/BOX
        c=np.mod(np.arctan2(np.sin(a).mean(0),np.cos(a).mean(0)),2*np.pi)*BOX/(2*np.pi)
        rows.append((float(np.sum(m[keep])),int(g),c))
    rows.sort(reverse=True,key=lambda z:z[0]); return rows[:count]

def enclosed(x,v,m,centers,radii):
    out=[]
    for center in centers:
        d=(x-center+BOX/2)%BOX-BOX/2; r2=np.sum(d*d,axis=1); row=[]
        for radius in radii:
            keep=r2 <= radius*radius; w=m[keep]; vv=v[keep]
            if not len(w): row.append(dict(count=0,mass=0.,mean_velocity=[0.]*3,sigma=[0.]*3)); continue
            mu=np.average(vv,axis=0,weights=w); sig=np.sqrt(np.average((vv-mu)**2,axis=0,weights=w))
            row.append(dict(count=int(len(w)),mass=float(w.sum()),mean_velocity=mu.tolist(),sigma=sig.tolist()))
        out.append(row)
    return out

def main():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm required')
    root=Path('/home/kjhan/BACKUP/CF4'); out=Path('/gpfs/kjhan/CF4/r1_enclosed_cause')/('job_'+os.environ['SLURM_JOB_ID']); out.mkdir(parents=True,exist_ok=False)
    radii=[.1875,.3,.5,.75,1.,1.5]; loaded={k:load_hop(k) for k in ['amr9','cic','tsc','amr8']}
    # loaded=(tags, positions, velocities, masses); pass masses explicitly.
    top=centers_from(loaded['amr9'][0], loaded['amr9'][1], loaded['amr9'][3]); centers=[z[2] for z in top]
    report=dict(status='RUNNING',source_commit=os.environ['EXPECTED_COMMIT'],sample='AMR9 top-32 HOP masses >=1000 particles',radii_cMpc_h=radii,centers_cMpc_h=np.asarray(centers).tolist(),moments={},contrasts={},limits='Fixed AMR9 centers remove solver-specific group boundaries but are not MW/M31/M33 identities, M200c or bound M33.')
    try:
        for label,(tag,x,v,m) in loaded.items(): report['moments'][label]=enclosed(x,v,m,centers,radii); write(out/'result.json',report); print(label+' enclosed complete',flush=True)
        for other in ['cic','tsc','amr8']:
            rows=[]
            for i in range(len(centers)):
                for j,rad in enumerate(radii):
                    a,b=report['moments']['amr9'][i][j],report['moments'][other][i][j]
                    rows.append(dict(index=i,radius_cMpc_h=rad,mass_relative=a['mass']/b['mass']-1,mean_velocity_difference=float(np.linalg.norm(np.asarray(a['mean_velocity'])-b['mean_velocity'])),sigma_relative=float(np.max(np.abs(np.asarray(a['sigma'])/np.maximum(b['sigma'],1e-30)-1))),count_amr9=a['count'],count_other=b['count']))
            report['contrasts'][other]=rows
        report['status']='COMPLETE_DRIVER_JUDGMENT_REQUIRED'
    except Exception as exc: report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}'); raise
    finally: write(out/'result.json',report)
if __name__=='__main__': main()
