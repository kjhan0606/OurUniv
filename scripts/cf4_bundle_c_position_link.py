"""Frozen observation-space link and same-field derivative; Slurm, no fit or map."""
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
import unittest

import h5py
import numpy as np
import torch

from cf4_population_locations import PopulationRoleModel, native_features, DX, OBSERVER
from cf4_position_link import features_torch, observation_kernel, log_likelihood, conservative_permutation
from cf4_lg_observation_contract import approximate_covariance, basis, solar_reference
from cf4_bundle_c_continuous import read_periodic_patch
from cf4_continuous_matter import restrict
from cf4_conditional_split_flow import configure_precision

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
SOURCE = ROOT/'population_locations_v1'
CFG = json.loads(Path('config/cf4_position_link_v1.json').read_text())
OUT = ROOT/CFG['output_name']
START = time.monotonic()
CREATED = False
RESULT = dict(status='NOT_STARTED', density_posterior=False, field_selection=False,
    training_updates=0, field_prior=False, physical_halo_catalogue=False)


def dump(name, value):
    with (OUT/name).open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def check():
    if time.monotonic()-START > CFG['application_seconds']:
        raise TimeoutError('fixed25-minute application limit')
    memory = dict(host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
    if any(value > CFG[name] for name, value in memory.items()):
        raise MemoryError(f'predeclared envelope exceeded: {memory}')
    return memory


def event(status, **extra):
    check()
    value = dict(status=status, job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-START, **extra)
    dump('status.json', value)
    print(json.dumps(value), flush=True)


def make_inputs(contract):
    frozen = json.loads((SOURCE/'population.json').read_text())['rollout_observers_frozen_before_fit']
    source_cases = {c['mw']: c for c in json.loads((SOURCE/'cases.json').read_text())['test']}
    cases = [source_cases[sid] for sid in frozen]
    wanted = np.array(sorted({sid for c in cases for sid in
        [c['mw']]+[p['m31'] for p in c['pairs']]+[t for p in c['pairs'] for t in p['thirds']]}), dtype=np.int64)
    positions, offset = {}, 0
    with h5py.File(ROOT/'tng_operator_v2/native_catalog.h5', 'r') as source:
        if source.attrs['status'] != 'COMPLETE_NATIVE_FIELD_COPY_NO_SELECTION':
            raise ValueError('incorrect catalogue source')
        for index in range(448):
            part = source[f'chunks/{index}']
            count = int(part['Header'].attrs['Nsubgroups_ThisFile'])
            ids = wanted[(wanted >= offset) & (wanted < offset+count)]
            if len(ids):
                rows = part['Subhalo/SubhaloPos'][ids-offset].astype(float)/1000
                positions.update({int(sid): p for sid, p in zip(ids, rows)})
            offset += count
    if len(positions) != len(wanted):
        raise ValueError('missing native position rows')
    observer = np.asarray(OBSERVER)*DX
    covariance = approximate_covariance(contract)[np.ix_([0, 4], [0, 4])]
    observations = []
    for case in cases:
        origin = np.asarray(case['origin'])*DX
        point = lambda sid: (positions[sid]-origin) % 75.
        for pair in case['pairs']:
            for third in pair['thirds']:
                distance_vector = np.array([point(pair['m31']), point(third)])-observer
                distance = np.linalg.norm(distance_vector, axis=1)
                observations.append(dict(mw=case['mw'], native_ids=[case['mw'], pair['m31'], third],
                    point_w=point(case['mw']).tolist(), directions=(distance_vector/distance[:, None]).tolist(),
                    modulus=(5*np.log10(distance*1000/CFG['h'])+10).tolist(),
                    weight=1/len(case['pairs'])/len(pair['thirds'])))
    actual = dict(point_w=(observer-solar_reference(contract)[0]*CFG['h']/1000).tolist(),
        directions=[basis(contract['measurements'][g]['ra_deg'], contract['measurements'][g]['dec_deg'])[0].tolist() for g in ('M31', 'M33')],
        modulus=[contract['measurements'][g]['value'][0] for g in ('M31', 'M33')])
    dump('observations.json', dict(mocks=observations, actual=actual, covariance=covariance.tolist(),
        mock='Native halo centers as measurements with adopted distance error; NOT stellar mock calibration',
        actual_caveat='Fixed solar reference, zero extra stellar/halo offset, approximate shared distance covariance',
        field_caveat='The8 fields are interface diagnostics, NOT field selection or posterior weights'))
    event('INPUTS_READY', observers=len(cases), native_triples=len(observations),
        weighted_triples_per_observer={c['mw']: sum(o['weight'] for o in observations if o['mw'] == c['mw']) for c in cases})
    make = lambda o, tolerance: observation_kernel(o['point_w'], observer, o['directions'], o['modulus'], covariance,
        h=CFG['h'], tolerance=tolerance)
    kernels = [make(o, CFG['integration_tolerance']) for o in observations]
    actual_kernel = make(actual, CFG['integration_tolerance'])
    tighter = make(actual, CFG['integration_tighter_tolerance'])
    return cases, observations, kernels, actual_kernel, tighter


def kernel_report(kernel):
    return {key: value for key, value in kernel.items() if key not in ('pairs', 'hosts', 'observer') } | dict(
        cell_pairs=len(kernel['pairs']), M31_cells=len({a for a, _, _ in kernel['pairs']}),
        shared_cell_pairs=sum(a == t for a, t, _ in kernel['pairs']))


def detached_score(model, features, kernel):
    value, factors = log_likelihood(model, features, kernel)
    values = dict(log_L=float(value.detach()), **{name: float(v.detach()) for name, v in factors.items()})
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('nonfinite score; zero support must not be a finite score')
    # qA*qT <=1 gives a conservative absolute-to-relative integration bound.
    error = kernel['numerical_probability_error']+kernel['clipped_tail_probability_bound']
    values['integration_relative_error_bound'] = error*math.exp(-values['log_weighted_cell_probability'])
    return values


def sensitivity(model, native, kernel):
    tensor = torch.from_numpy(native).cuda().requires_grad_(True)
    value, _ = log_likelihood(model, features_torch(tensor), kernel)
    gradient, = torch.autograd.grad(value, tensor)
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError('nonfinite native sensitivity')
    scale_sensitivity = (gradient*tensor).sum(0)[4:-4, 4:-4, 4:-4].detach().cpu().numpy()
    baseline = tensor.detach()
    rows = []
    parent = restrict(native[:, 4:-4, 4:-4, 4:-4], 8)
    for axis in range(3):
        check()
        permuted = conservative_permutation(baseline, axis)
        changed_parent = restrict(permuted[:, 4:-4, 4:-4, 4:-4].cpu().numpy(), 8)
        conservation = float(np.max(abs(changed_parent-parent)/np.maximum(abs(parent), 1.)))
        if conservation > 1e-9:
            raise ValueError('parent extensive moments not conserved')
        direction = permuted-baseline
        middle = (baseline+CFG['gradient_mixture_interior']*direction).requires_grad_(True)
        loss, _ = log_likelihood(model, features_torch(middle), kernel)
        derivative, = torch.autograd.grad(loss, middle)
        analytic = float((derivative*direction).sum())
        finite = []
        for step in CFG['gradient_steps']:
            with torch.no_grad():
                plus, _ = log_likelihood(model, features_torch(middle+step*direction), kernel)
                minus, _ = log_likelihood(model, features_torch(middle-step*direction), kernel)
            finite.append(float((plus-minus)/(2*step)))
        passed = all(abs(analytic-fd) <= CFG['gradient_absolute_tolerance']+CFG['gradient_relative_tolerance']*abs(analytic) for fd in finite)
        rows.append(dict(axis=axis, epsilon=CFG['gradient_mixture_interior'], log_L=float(loss),
            derivative=analytic, finite_differences=finite, conservation_relative_max=conservation, passed=passed))
        del permuted, direction, middle, loss, derivative
    np.savez_compressed(OUT/'sensitivity_projection.npz', log_mass=np.log1p(native[0, 4:-4, 4:-4, 4:-4].sum(2)),
        log_L_scaling_sensitivity=scale_sensitivity.sum(2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].imshow(np.log1p(native[0, 4:-4, 4:-4, 4:-4].sum(2)).T, origin='lower', extent=(0, 12, 0, 12), cmap='Greys')
    axes[0].set_title('Native TNG mass — NOT reconstructed LG')
    bound = max(float(abs(scale_sensitivity.sum(2)).max()), 1e-15)
    artist = axes[1].imshow(scale_sensitivity.sum(2).T, origin='lower', extent=(0, 12, 0, 12), cmap='coolwarm', vmin=-bound, vmax=bound)
    axes[1].set_title('Actual position log-likelihood sensitivity')
    fig.colorbar(artist, ax=axes[1])
    for ax in axes:
        ax.set(xlabel='native x [cMpc/h]', ylabel='native y [cMpc/h]')
    fig.savefig(OUT/'position_sensitivity.png', dpi=130)
    plt.close(fig)
    report = dict(native_log_L=float(value), directions=rows, passed=all(r['passed'] for r in rows),
        map='Derivative w.r.t. cellwise common scaling of M/P/Q; NOT a posterior or optimized field',
        cold_convention='zero derivative at cold/empty feature branches; direction checks at interior eps=.05',
        maximum_absolute_native_scaling_sensitivity=float(abs(scale_sensitivity).max()))
    dump('sensitivity.json', report)
    return report


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('source-pinned Slurm required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    torch.set_num_threads(2)
    configure_precision(True)
    RESULT.update(job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'])
    dump('request.json', dict(**CFG, **{k: RESULT[k] for k in ('job_id', 'source_commit')}, gpu=torch.cuda.get_device_name()))
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_position_link']))
    dump('tests.json', dict(tests=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors)))
    if not tests.wasSuccessful():
        raise ValueError('focused link tests failed')
    source_result = json.loads((SOURCE/'result.json').read_text())
    if source_result['status'] != 'PASS_POPULATION_LOCATION_FEASIBILITY_ONLY' or source_result['source_commit'] != CFG['source_location_commit']:
        raise ValueError('incorrect frozen location source')
    checkpoint = torch.load(SOURCE/'checkpoint_final.pt', map_location='cpu', weights_only=False)
    model = PopulationRoleModel().double().cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.alpha = source_result['alpha']
    model.requires_grad_(False)
    contract = json.loads(Path('config/cf4_lg_observation_contract_v1.json').read_text())
    cases, observations, kernels, actual, tighter = make_inputs(contract)
    dump('kernels.json', dict(mocks=[kernel_report(k) for k in kernels], actual=kernel_report(actual), tighter=kernel_report(tighter)))
    mock_rows, actual_rows, agreement = [], [], []
    with h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as source:
        if source.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE' or not np.isclose(source.attrs['h'], CFG['h']):
            raise ValueError('incorrect field source/cosmology')
        for index, case in enumerate(cases):
            check()
            native = read_periodic_patch(source['fine'], (np.asarray(case['origin'])-4)%400, 72)
            with torch.no_grad():
                field = features_torch(torch.from_numpy(native).cuda())
                reference = native_features(native)[:, 4:-4, 4:-4, 4:-4]
                relative_error = float(np.max(abs(field.cpu().numpy()-reference)/(1+abs(reference))))
                if relative_error > 1e-4:
                    raise ValueError(f'native feature mismatch {relative_error}')
                agreement.append(relative_error)
                for obs, kernel in zip(observations, kernels):
                    check()
                    mock_rows.append(dict(field_mw=case['mw'], observed_mw=obs['mw'], weight=obs['weight'],
                        **detached_score(model, field, kernel)))
                ordinary = detached_score(model, field, actual)
                tight = detached_score(model, field, tighter)
                actual_rows.append(dict(field_mw=case['mw'], ordinary=ordinary, tighter=tight,
                    absolute_log_L_difference=abs(ordinary['log_L']-tight['log_L'])))
            if index == 0:
                RESULT['sensitivity'] = sensitivity(model, native, actual)
            dump('mock_scores.json', mock_rows)
            dump('actual_scores.json', actual_rows)
            event('FIELD_SCORED', field_mw=case['mw'], completed=index+1, total=len(cases))
            del field, native, reference
    cross = []
    for observed in cases:
        for candidate in cases:
            selected = [r for r in mock_rows if r['field_mw'] == candidate['mw'] and r['observed_mw'] == observed['mw']]
            cross.append(dict(observed_mw=observed['mw'], field_mw=candidate['mw'],
                **{key: sum(r['weight']*r[key] for r in selected) for key in ('log_L', 'MW', 'M31_given_MW', 'M33_given_MW_M31_data')}))
    dump('cross_scores.json', cross)
    integration_ok = all(r['absolute_log_L_difference'] < 1e-5 and r['tighter']['integration_relative_error_bound'] < 1e-4 for r in actual_rows)
    integration_ok &= all(r['integration_relative_error_bound'] < 1e-4 for r in mock_rows)
    RESULT.update(status=('PASS_POSITION_LIKELIHOOD_COMPONENT_ONLY' if integration_ok and RESULT['sensitivity']['passed'] else
        'INCONCLUSIVE_POSITION_LINK_NUMERICS'), numerical_integration_pass=integration_ok,
        native_feature_scaled_errors=agreement, mock_triples=len(observations), mock_cross_scores=len(mock_rows),
        actual_field_scores=actual_rows, application_seconds=time.monotonic()-START, memory=check(),
        next='Close link diagnostic; next substantive design addresses joint fine-field/mass/COM and selection, no location tuning')
    dump('result.json', RESULT)
    event(RESULT['status'], integration_pass=integration_ok, derivative_pass=RESULT['sensitivity']['passed'])


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        if CREATED:
            RESULT.update(status='INCONCLUSIVE_TECHNICAL_FAILURE', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            dump('result.json', RESULT)
        raise
