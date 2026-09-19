"""Frozen stable-field one-step vs saved rollout; no fitting or target changes."""
import json
import os
from pathlib import Path
import resource
import time
import unittest

import h5py
import numpy as np
import torch

from cf4_bundle_c_stable_field import CFG, ROOT, cube, OBSERVER_FULL, validate_evaluation_checkpoint
from cf4_stable_field import StableField, conditioning, sample
from cf4_spatial_diffusion import unpack
from cf4_continuous_matter import restrict
from cf4_split_moments import state
from cf4_bundle_c_flow_pilot import metrics, compare
from cf4_conditional_split_flow import configure_precision

OUT = ROOT/'stable_field_scale_link_v1'
START = time.monotonic()


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def check():
    if time.monotonic()-START > 540:
        raise TimeoutError('nine-minute application cap')


def score(field, truth, root, dx, factor):
    measured, native = metrics(field, dx, root, factor), metrics(truth, dx, root, factor)
    result = compare(measured, native)
    if max(measured['coarse_error']) > 1e-8:
        raise ValueError('field does not preserve native1.5 parent')
    variance = state(root)[1]
    budget = np.sum(root[0]*variance, axis=(1, 2, 3))/root[0].sum()
    for label, value in (('generated', measured), ('native', native)):
        bulk = np.asarray(value['bulk_residual_rms_km_s'])**2
        internal = np.asarray(value['physical_sigma_rms_km_s'])**2
        np.testing.assert_allclose(bulk+internal, budget, rtol=1e-9, atol=1e-6)
        result[label+'_bulk_budget_fraction'] = (bulk/budget).tolist()
    result['log_density_boundary'] = {}
    for label, value in (('generated', field), ('native', truth)):
        log = np.log1p(value[0]/value[0].mean())
        result['log_density_boundary'][label] = {}
        for period in sorted({2, factor}):
            ratios = []
            faces = np.arange(1, log.shape[-1])
            for axis in range(3):
                plane = np.moveaxis(np.diff(log, axis=axis)**2, axis, 0).mean(axis=(1, 2))
                ratios.append(float(plane[faces % period == 0].mean()/plane[faces % period != 0].mean()))
            result['log_density_boundary'][label][str(period)] = ratios
    return result


@torch.no_grad()
def run():
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('source-pinned Slurm required')
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    configure_precision(True)
    with torch.enable_grad():
        tested = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_stable_field']))
    if not tested.wasSuccessful():
        raise ValueError('focused tests failed')
    fit = ROOT/'stable_field_v1'
    saved = torch.load(fit/'checkpoint_24000.pt', map_location='cpu', weights_only=False)
    validate_evaluation_checkpoint(saved, json.loads((fit/'request.json').read_text()), json.loads((fit/'result.json').read_text()))
    model = StableField().cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['ema'], strict=True)
    location, spread = saved['location'], saved['spread']
    del saved
    cases = {c['mw']: c for c in json.loads((ROOT/'population_locations_v1/cases.json').read_text())['test']}
    previous = json.loads((ROOT/'stable_field_eval_v2/draws.json').read_text())
    first = [row for row in previous if row['draw'] == 0]
    if len(first) != 8 or not all(row['valid'] for row in first):
        raise ValueError('all eight stored first draws required')
    rows = []
    with h5py.File(ROOT/'stable_field_eval_v2/first_draw_fields.h5', 'r') as source, h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as native:
        for index, case in enumerate(first):
            truth80 = cube(native['fine'], cases[case['mw']])
            generated80 = source[f'mw_{case["mw"]}/moments80'][...]
            root = restrict(truth80, 8)[:, 1:-1, 1:-1, 1:-1]
            for level in (2, 1, 0):
                check()
                factor, pad, dx = 2**(3-level), 8//2**level, .1875*2**level
                truth = restrict(truth80, 2**level) if level else truth80
                rollout = restrict(generated80, 2**level) if level else generated80
                sl = (slice(None),)+(slice(pad, -pad),)*3
                base = dict(mw=case['mw'], level=level, dx=dx)
                rows.append(dict(**base, mode='stored_rollout', score=score(rollout[sl], truth[sl], root, dx, factor)))
                if level < 2:
                    torch.manual_seed(913021+100*index+level)
                    parent = restrict(truth, 2)
                    ctx = torch.from_numpy(conditioning(parent, dx, OBSERVER_FULL))[None].cuda()
                    try:
                        z, codes, trace = sample(model, ctx, dx, location, spread, CFG['diffusion_steps'], check)
                        teacher, decode_report = unpack(z, codes, parent)
                    except ValueError as error:
                        rows.append(dict(**base, mode='native_parent_step', valid=False, error=str(error)))
                        continue
                    rows.append(dict(**base, mode='native_parent_step', seed=913021+100*index+level,
                        valid=True, score=score(teacher[sl], truth[sl], root, dx, factor), trace=trace, decoder=decode_report))
            dump('scores.json', rows)
            print(json.dumps(dict(status='SCORED', observers=index+1, total=8)), flush=True)
    dump('result.json', dict(status='COMPLETE_FROZEN_SCALE_COMPARISON_NOT_POSTERIOR',
        tests=tested.testsRun, cases=8, rows=len(rows), new_one_step_draws=16, optimizer_steps=0,
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'],
        seconds=time.monotonic()-START, host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30))


if __name__ == '__main__':
    run()
