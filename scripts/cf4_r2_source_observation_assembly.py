"""Assemble source-labelled FP inputs without inventing missing memberships."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_source_membership import membership_state
from cf4_r2_tempel_source_connection import rows

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_source_observation_assembly_v1'
C = 299792.458


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads((BASE/'r2_tempel_source_connection_v1/result.json').read_text())
    for binding in previous['source_manifest']:
        if hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest() != binding['sha256']:
            raise ValueError('Tempel source changed')
    source = Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat')
    if hashlib.md5(source.read_bytes()).hexdigest() != 'b5b6e31caf7ea469c2ac2cb775fa8d14':
        raise ValueError('SDSS source changed')
    with source.open() as f:
        cols = f.readline().lstrip('#').split()
        fp = {int(r['PGC']): r for r in
              (dict(zip(cols, line.split(), strict=True)) for line in f if line.strip())}
    with np.load(BASE/'r2_sdss_fp_source_link_v1/source_link.npz') as f:
        data = {k: f[k].copy() for k in f.files}
    obj_to_pgc = {r['objid']: p for p, r in fp.items()}
    if len(obj_to_pgc) != len(fp):
        raise ValueError('duplicate full-source photometric IDs')
    exact = {}
    table = Path(previous['source_manifest'][1]['path'])
    for r in rows(table, ['GalID','objID','GroupID','Ngal','zcmb','e_zobs']):
        p = obj_to_pgc.get(r['objID'])
        if p is not None:
            if p in exact:
                raise ValueError('duplicate source association')
            exact[p] = r
    states = {}
    for p, r in fp.items():
        t = exact.get(p)
        states[p] = membership_state(int(r['IDgroupT17']), int(r['NgroupT17']),
            None if t is None else int(t['GroupID']), None if t is None else int(t['Ngal']))
    selected = [int(p) for p in data['PGC']]
    labels = np.asarray([states[p] for p in selected])
    # Preserve the previous physical-field/eta connection EXACTLY; append
    # source semantics rather than replacing the source's group redshift.
    data['membership_state'] = labels
    data['source_richness'] = np.array([int(fp[p]['NgroupT17']) for p in selected])
    data['source_obj_id'] = np.array([int(fp[p]['objid']) for p in selected], dtype=np.int64)
    data['source_individual_zcmb'] = np.array([float(fp[p]['zcmb']) for p in selected])
    data['source_individual_zhelio'] = np.array([float(fp[p]['zhelio']) for p in selected])
    data['source_individual_zhelio_error'] = np.array([float(fp[p]['zhelioerr']) for p in selected])
    data['tempel_member_zcmb'] = np.array([float(exact[p]['zcmb']) if p in exact else np.nan for p in selected])
    # One comparison of known additive-vs-multiplicative conventions, not a
    # fitted correction. Source rounding and changed spectra also contribute.
    zh = data['source_individual_zhelio']
    zc = data['source_individual_zcmb']
    q = (zc-zh)/(1+zh)
    additive = zh+q
    good = np.isfinite(data['tempel_member_zcmb'])
    delta = C*(zc[good]-data['tempel_member_zcmb'][good])
    residual = C*(additive[good]-data['tempel_member_zcmb'][good])
    quant = lambda a: np.quantile(np.abs(a), [.5,.9,1]).tolist() if len(a) else []
    absent = labels == 'source_ungrouped_catalogue_absent'
    singleton_delta = C*(data['zgroup'][absent]-zc[absent])
    # Retain source evidence about the one zero map point; no rescan/smoothing
    # or arbitrary selection modification. It has no FP mark in this assembly.
    with np.load(BASE/'r2_point_mark_manifest_v1/points.npz') as f:
        zero_recs = set(map(int, f['recno'][f['point_selection'] <= 0]))
    zero_edges = []
    with (ROOT/'data/cf4_2mpp_crossmatch_v1.csv').open() as f:
        for r in csv.DictReader(f):
            if r['twompp_recno'] and int(r['twompp_recno']) in zero_recs:
                zero_edges.append({k:r[k] for k in ('PGC','twompp_recno','match_class')})
    unresolved = [p for p in selected if states[p].startswith('unresolved')]
    result = dict(classification='SOURCE_SEMANTIC_OBSERVATION_ASSEMBLY_NOT_POSTERIOR',
        job_id=os.environ['SLURM_JOB_ID'], full_source_rows=len(fp),
        full_source_states=dict(Counter(states.values())), selected_rows=len(selected),
        selected_states=dict(Counter(labels.tolist())), unresolved_selected_PGC=unresolved,
        absent_selected_PGC=[p for p in selected if p not in exact],
        absent_group_minus_individual_cz_abs_quantiles=quant(singleton_delta),
        matched_individual_original_cz_abs_quantiles=quant(delta),
        matched_individual_additive_convention_residual_abs_quantiles=quant(residual),
        redshift_comparison='fixed convention diagnostic only; no data correction adopted',
        zero_selection_recnos=sorted(zero_recs), zero_selection_crossmatch_edges=zero_edges,
        zero_selection_treatment='unchanged; finite coarsened count support is not full point support',
        preserved_eta_and_field_connection=True, covariance_calibrated=False,
        complete_joint_likelihood=False, R2_posterior=False, new_gravity_runs=0,
        source_paper='https://arxiv.org/html/2201.03112',
        source_manifest=previous['source_manifest'])
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'observations.npz', **data)
    (OUT/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
