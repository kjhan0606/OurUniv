"""Full N32 plug-in LOS quadrature diagnostic; no catalog/count datum reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cf4_datum_bearing_z0_phasec_pilot as phasec
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit


def rule(order):
    nodes, weights = roots_legendre(order)
    nodes = 8 * nodes
    weights = weights * np.exp(-0.5 * nodes**2)
    return nodes, weights / weights.sum()


def deposit(mass, positions, los, sigma, nodes, weights, n, box):
    out = np.zeros((n, n, n))
    for node, weight in zip(nodes, weights, strict=True):
        out += weight * phasec.tsc_deposit_numpy(mass, (positions + node*sigma*los) % box, n, box)
    if not np.isfinite(out).all() or (out < 0).any():
        raise ValueError('invalid integrated intensity')
    np.testing.assert_allclose(out.sum(), mass.sum(), atol=1e-10, rtol=1e-12)
    return out


def poisson_kl(reference, candidate):
    if np.any((reference > 0) & (candidate <= 0)):
        raise ValueError('candidate has zero intensity on reference support')
    mask = reference > 0
    r, c = reference[mask], candidate[mask]
    t = (c-r)/r
    # Stable when reference and candidate nearly coincide.
    return float(np.sum(r*(t-np.log1p(t))) + np.sum(candidate[~mask]))


def metrics(reference, candidate, box):
    l1 = np.abs(reference-candidate).sum()/reference.sum()
    n = reference.shape[0]
    freq = 2*np.pi*np.fft.fftfreq(n, d=box/n)
    k = np.sqrt(sum(a*a for a in np.meshgrid(freq, freq, freq, indexing='ij')))
    edges = np.linspace(0, k.max()*(1+1e-12), 10)
    rf = np.fft.fftn(reference-reference.mean(), norm='ortho')
    cf = np.fft.fftn(candidate-candidate.mean(), norm='ortho')
    shells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (k >= lo) & (k < hi) & (k > 0)
        if not use.any():
            continue
        pr = float(np.mean(np.abs(rf[use])**2))
        pc = float(np.mean(np.abs(cf[use])**2))
        residual = float(np.mean(np.abs(cf[use]-rf[use])**2))
        shells.append({'k_h_Mpc': float(k[use].mean()), 'mode_count': int(use.sum()),
                       'power_ratio': pc/pr if pr > 0 else None,
                       'residual_to_poisson_shot_power': residual/float(reference.mean())})
    return {'relative_L1': float(l1), 'poisson_KL_nats': poisson_kl(reference, candidate), 'shells': shells}


def run(index):
    start = time.perf_counter()
    plan_path = ROOT/'config/cf4_z2_operator_necessity_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    program_path = ROOT/plan['inputs']['base_program']
    base = json.loads(program_path.read_text())
    task = plan['inputs']['tasks'][index]
    source = Path(plan['inputs']['posterior_root'])/task/'posterior_summary.npz'
    with np.load(source, allow_pickle=False) as data:
        delta = data['posterior_density_mean'].astype(float)
        vel = np.moveaxis(data['posterior_velocity_mean'].astype(float), 0, -1)
    n = delta.shape[0]
    box = float(base['grid']['box_size_cMpc_h'])
    assert delta.shape == (32,)*3 and vel.shape == (32,32,32,3) and box == 384.
    with np.load(base['input_bindings']['Phase_A_datum']['path'], allow_pickle=False) as data:
        exposure = data['raw_selection_exposure'].astype(float)
    assert exposure.shape == (6,n,n,n) and np.isfinite(exposure).all() and (exposure >= 0).all()
    exposure = np.clip(exposure, 0, 1)*base['heldout']['training_fraction']
    nbar, bias = phasec._published_prior_arrays(base)
    axis = (np.arange(n)+.5)*box/n
    centres = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1,3)
    los = centres-box/2
    los /= np.linalg.norm(los, axis=1)[:,None]
    coherent = np.sum(vel.reshape(-1,3)*los, axis=1)/100
    positions = (centres+coherent[:,None]*los) % box
    eta = (delta-delta.mean()).reshape(-1)
    model = base['inference_model']
    old_nodes = np.asarray(model['Gaussian_radial_quadrature_offsets_sigma'])
    old_weights = np.asarray(model['Gaussian_radial_quadrature_weights'])
    nodes128, weights128 = rule(128)
    nodes256, weights256 = rule(256)
    probes = np.linspace(0,n**3-1,8,dtype=int)
    q1_digest = hashlib.sha256((ROOT/'src/cf4_q1_cell_integrated_convolution.py').read_bytes()).hexdigest()
    assert q1_digest == '74ae1bb12171a2baac76c8052d592b4dc5098043bf7c11bca6ffb9eea852d6b2'
    # Compare existing NumPy deposition with the actual JAX primitive once.
    import jax.numpy as jnp
    test_mass = np.linspace(.5, 1.5,8)
    old_pos = (positions[probes]+old_nodes[0]*los[probes]) % box
    check = phasec._jax_tsc_deposit_one(jnp.asarray(test_mass), jnp.asarray(old_pos), n, box/n)
    np.testing.assert_allclose(np.asarray(check), phasec.tsc_deposit_numpy(test_mass, old_pos,n,box), atol=1e-12,rtol=1e-12)
    cases = []
    for offset in plan['method']['fog_log_sigma_offsets']:
        rows, refs, olds, lows = [], [], [], []
        for p in range(6):
            fog = model['FoG_prior_median_km_s'][p]*np.exp(offset*model['FoG_log_sigma'])
            sigma = np.hypot(fog, model['fixed_redshift_error_km_s'][p])/100
            mass = np.exp(np.log(nbar[p])+bias[p]*eta)
            # Sourcewise comparison separates cancellation from true reference accuracy.
            probe_errors = []
            for i in probes:
                oracle = cell_integrated_tsc_deposit(positions[i:i+1],[1.],los[i:i+1],[sigma],n,box)
                candidate = deposit(np.array([1.]),positions[i:i+1],los[i:i+1],sigma,nodes256,weights256,n,box)
                probe_errors.append(float(np.abs(oracle-candidate).sum()))
            old = deposit(mass,positions,los,sigma,old_nodes,old_weights,n,box)*exposure[p]
            low = deposit(mass,positions,los,sigma,nodes128,weights128,n,box)*exposure[p]
            ref = deposit(mass,positions,los,sigma,nodes256,weights256,n,box)*exposure[p]
            convergence = metrics(ref,low,box)
            comparison = metrics(ref,old,box)
            valid = max(probe_errors) <= 1e-5 and convergence['relative_L1'] <= 1e-5
            rows.append({'population':p,'sigma_cMpc_h':float(sigma),'max_Q1_probe_L1':max(probe_errors),
                         'reference_convergence':convergence,'comparison':comparison,'reference_valid':bool(valid)})
            refs.append(ref); olds.append(old); lows.append(low)
            print(json.dumps({'task':index,'fog_offset':offset,'population':p,'KL':comparison['poisson_KL_nats'],'reference_valid':bool(valid)}),flush=True)
        kl = sum(r['comparison']['poisson_KL_nats'] for r in rows)
        self_kl = sum(r['reference_convergence']['poisson_KL_nats'] for r in rows)
        valid = all(r['reference_valid'] for r in rows) and self_kl <= .01 and (kl < 1 or self_kl <= .01*kl)
        cases.append({'fog_log_sigma_offset':offset,'populations':rows,'joint_KL_nats':kl,
                      'reference_self_KL_nats':self_kl,'reference_valid':bool(valid),
                      'total_field':metrics(np.sum(refs,axis=0),np.sum(olds,axis=0),box),
                      'decision':'INCONCLUSIVE_REFERENCE' if not valid else ('MATERIAL_AT_TESTED_STATE' if kl>=1 else 'BELOW_ONE_NAT_AT_TESTED_STATE_ONLY')})
    result = {'task':task,'job_id':os.environ['SLURM_JOB_ID'],'implementation_commit':os.environ['EXPECTED_COMMIT'],
              'source_fields':str(source),'program':str(program_path),'Q1_sha256':q1_digest,
              'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'cases':cases,'elapsed_seconds':time.perf_counter()-start,
              'peak_host_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'scope':'Two nuisance plug-in states per mock, no full posterior or observational inference',
              'observational_frontier_claim':False,'actual_count_datum_read':False}
    path = Path(plan['output_root'])/f'task_{index}_{os.environ["SLURM_JOB_ID"]}.json'
    with path.open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False)
    print('RESULT '+str(path),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=int,choices=(0,1),required=True)
    run(parser.parse_args().task)
