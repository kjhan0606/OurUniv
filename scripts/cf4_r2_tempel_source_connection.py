"""Acquire source group/member data and connect observation-law inputs once."""
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cf4_r2_coarsened_observation import within_cell_logdensity, CountSupportError

BASE = Path('/gpfs/kjhan/CF4/z0_density')
EXTERNAL = Path('/gpfs/kjhan/CF4/external/tempel2017_cds')
OUT = BASE/'r2_tempel_source_connection_v1'
C = 299792.458


def acquire(table,columns):
    path = EXTERNAL/f'{table}_selected_columns.tsv'
    url = 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?'+urlencode({
        '-source':f'J/A+A/602/A100/{table}','-out':','.join(columns),'-out.max':'unlimited'})
    if not path.exists():
        # A partial network product is never promoted to the reusable filename.
        partial = path.with_suffix('.partial')
        with urlopen(url,timeout=180) as response, partial.open('wb') as target:
            total = 0
            while chunk := response.read(1024*1024):
                total += len(chunk)
                if total > 100_000_000:
                    raise RuntimeError('unexpected source size above100MB bound')
                target.write(chunk)
        partial.rename(path)
    return path,url


def rows(path,columns):
    with path.open() as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith('#'):
                continue
            values = [v.strip() for v in raw.rstrip('\n').split('\t')]
            if not values[0].isdigit():
                continue
            if len(values) != len(columns):
                raise ValueError('source column mismatch')
            yield dict(zip(columns,values,strict=True))


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    if OUT.exists():
        raise FileExistsError(OUT)
    EXTERNAL.mkdir(parents=True,exist_ok=True)
    colg = ['GroupID','Ngal','zcmb','sig.v']
    colm = ['GalID','objID','GroupID','Ngal','zcmb','e_zobs']
    pg,ug = acquire('table2',colg)
    pm,um = acquire('table1',colm)
    groups = {int(r['GroupID']):r for r in rows(pg,colg)}
    if len(groups) != 88662:
        raise ValueError(f'incomplete group source: {len(groups)}')
    with np.load(BASE/'r2_sdss_fp_source_link_v1/source_link.npz') as f:
        linked = {k:f[k].copy() for k in f.files}
    source_path = Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat')
    if hashlib.md5(source_path.read_bytes()).hexdigest() != 'b5b6e31caf7ea469c2ac2cb775fa8d14':
        raise ValueError('SDSS source changed')
    wanted = set(map(int,linked['PGC']))
    with source_path.open() as f:
        columns = f.readline().lstrip('#').split()
        fp = {int(r['PGC']):r for r in (dict(zip(columns,l.split(),strict=True)) for l in f if l.strip()) if int(r['PGC']) in wanted}
    obj_to_pgc = {r['objid']:p for p,r in fp.items()}
    if len(obj_to_pgc) != len(fp):
        raise ValueError('nonunique SDSS photometric IDs')
    ids = sorted({int(r['IDgroupT17']) for r in fp.values()}-{0})
    ids_set = set(ids)
    selected_members = defaultdict(list)
    exact = {}
    duplicate_matches = []
    total = 0
    for r in rows(pm,colm):
        total += 1
        g = int(r['GroupID'])
        if g in groups and g in ids_set:
            selected_members[g].append(r)
        p = obj_to_pgc.get(r['objID'])
        if p is not None:
            if p in exact:
                duplicate_matches.append(p)
            exact[p] = r
    if total != 584449 or duplicate_matches:
        raise ValueError(f'incomplete/nonunique galaxy source {total}, {duplicate_matches[:5]}')
    mismatches = [p for p,r in exact.items() if int(r['GroupID']) != int(fp[p]['IDgroupT17'])]
    richness_mismatches = [p for p,r in exact.items() if int(r['Ngal']) != int(fp[p]['NgroupT17'])]
    packed = []
    group_readout = []
    for g in ids:
        if g not in groups:
            continue
        members = selected_members[g]
        if len(members) != int(groups[g]['Ngal']):
            raise ValueError(f'incomplete group membership {g}')
        z = np.array([float(r['zcmb']) for r in members])
        mean = z.mean()
        sample_sigma = np.std(C*z,ddof=1)/(1+mean)
        published_sigma = float(groups[g]['sig.v'])
        group_readout.append((g,len(members),mean,sample_sigma,published_sigma))
        for r in members:
            packed.append((g,int(r['GalID']),int(r['objID']),float(r['zcmb']),float(r['e_zobs'])))
    readout = np.asarray(group_readout)
    source_offsets = [C*(float(fp[p]['zcmb_group'])-float(groups[int(fp[p]['IDgroupT17'])]['zcmb']))
                      for p in exact if int(fp[p]['IDgroupT17']) in groups]
    # For the existing cell-modulated intensity, the cell amplitude cancels
    # from the conditional. Original point selection must remain positive.
    with np.load(BASE/'r2_point_mark_manifest_v1/points.npz') as f:
        keys = f['population'].astype(np.int64)*128**3+f['flat_cell']
        selection = f['point_selection'].copy()
        exposure = f['integrated_cell_exposure'].copy()
        recnos = f['recno'].copy()
    support_zero = recnos[(selection<=0)|(exposure<=0)]
    try:
        conditional = within_cell_logdensity(keys,selection/27.,exposure)
    except CountSupportError:
        conditional = None
    result = dict(classification='SOURCE_MEMBERSHIP_AND_WITHIN_CELL_CONNECTION_NOT_POSTERIOR',
        job_id=os.environ['SLURM_JOB_ID'],source_group_rows=len(groups),source_member_rows=total,
        FP_rows=len(fp),exact_photometric_ID_matches=len(exact),unmatched_PGC=sorted(wanted-set(exact)),
        group_ID_mismatch_PGC=mismatches,richness_mismatch_PGC=richness_mismatches,
        source_multimember_groups=len(group_readout),source_members_in_those_groups=len(packed),
        FP_singleton_rows=sum(int(r['IDgroupT17'])==0 for r in fp.values()),
        published_sigma_km_s_quantiles=np.quantile(readout[:,4],[0,.1,.5,.9,1]).tolist(),
        groups_with_zero_published_sigma=int(np.sum(readout[:,4]<=0)),
        recomputed_to_published_sigma_median=float(np.median(readout[readout[:,4]>0,3]/readout[readout[:,4]>0,4])),
        source_group_cz_offset_abs_quantiles_km_s=np.quantile(np.abs(source_offsets),[.5,.9,1]).tolist(),
        conditional_position_logdensity=conditional,unsupported_point_recnos=support_zero.tolist(),
        all_count_points_preserved=len(keys),source_dispersion_is_independent_calibration=False,
        group_covariance_calibrated=False,selected_group_distance_prior_calibrated=False,
        complete_joint_likelihood=False,R2_posterior=False,new_gravity_runs=0,
        source_manifest=[dict(path=str(p),url=u,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,u in ((pg,ug),(pm,um))])
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'source_group_members.npz',group_id=readout[:,0].astype(np.int64),
        richness=readout[:,1].astype(np.int32),mean_zcmb=readout[:,2],sample_sigma_km_s=readout[:,3],
        published_sigma_km_s=readout[:,4],member_group=np.array([p[0] for p in packed],dtype=np.int64),
        member_gal_id=np.array([p[1] for p in packed],dtype=np.int64),
        member_obj_id=np.array([p[2] for p in packed],dtype=np.int64),
        member_zcmb=np.array([p[3] for p in packed]),member_zerr=np.array([p[4] for p in packed]))
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
