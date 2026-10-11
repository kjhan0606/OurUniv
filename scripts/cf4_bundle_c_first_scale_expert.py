"""First-scale specialization; finer-scale networks remain frozen."""
import json
import os
import time
import unittest

import h5py
import numpy as np
import torch
from torch import nn

import cf4_bundle_c_field_link_repair as repair
import cf4_bundle_c_stable_field as base
from cf4_stable_field import StableField
from cf4_continuous_matter import restrict
from cf4_bundle_c_scale_link import score
from cf4_conditional_split_flow import configure_precision


class FirstScale(nn.Module):
    def __init__(self, fixed, expert):
        super().__init__()
        self.fixed, self.expert = fixed, expert

    def forward(self, z, codes, coarse, t, scale):
        if not bool((scale == scale[0]).all()):
            raise ValueError('homogeneous-scale batch required')
        model = self.expert if float(scale[0]) == 2 else self.fixed
        return model(z, codes, coarse, t, scale)


def run():
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('source-pinned Slurm required')
    repair.CFG = dict(repair.CFG, output_name='stable_field_first_expert_v1',
        fixed_level=2, learning_seconds_per_branch=1200, application_seconds=1680)
    repair.OUT = base.ROOT/repair.CFG['output_name']
    repair.OUT.mkdir(exist_ok=False)
    repair.CREATED = True
    base.OUT, base.check = repair.OUT, repair.check
    torch.set_num_threads(2)
    configure_precision(True)
    repair.dump('request.json', dict(config=repair.CFG, source_commit=os.environ['EXPECTED_COMMIT'],
        job_id=os.environ['SLURM_JOB_ID'], observed_posterior=False,
        baseline='stable_field_eval_v2', finer_networks_frozen=True))
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_stable_field']))
    if not tests.wasSuccessful():
        raise ValueError('existing regressions failed')
    fit = base.ROOT/'stable_field_v1'
    saved = torch.load(fit/'checkpoint_24000.pt', map_location='cpu', weights_only=False)
    base.validate_evaluation_checkpoint(saved, json.loads((fit/'request.json').read_text()),
        json.loads((fit/'result.json').read_text()))
    initial, location, spread = saved['ema'], saved['location'], saved['spread']
    del saved
    fixed = StableField().cuda().eval().requires_grad_(False)
    fixed.load_state_dict(initial, strict=True)
    cases, test, slab = base.cases_and_slab()
    expert, steps, seconds = repair.train('control', initial, cases, slab, location, spread)
    del slab
    model = FirstScale(fixed, expert).eval().requires_grad_(False)
    # Routing and frozen-state controls, not an extra validation framework.
    with torch.no_grad():
        z = torch.zeros(1, 49, 2, 2, 2, device='cuda')
        c = torch.zeros_like(z, dtype=torch.long)
        ctx = torch.zeros(1, 10, 2, 2, 2, device='cuda')
        t = torch.tensor([50], device='cuda')
        for level in (0, 1, 2):
            scale = torch.tensor([float(level)], device='cuda')
            expected = (expert if level == 2 else fixed)(z, c, ctx, t, scale)
            for a, b in zip(model(z, c, ctx, t, scale), expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key, value in fixed.state_dict().items():
            torch.testing.assert_close(value.cpu(), initial[key], rtol=0, atol=0)
    base.denoising(model, test, location, spread)
    draws, ensembles = base.evaluate(model, test, location, spread)
    repair.endpoints(test)
    rows = []
    with h5py.File(base.OUT/'first_draw_fields.h5', 'r') as generated, h5py.File(base.ROOT/'total_matter_v1/matter_moments.h5', 'r') as native:
        for case in test:
            key = f'mw_{case["mw"]}/moments80'
            if key not in generated:
                rows.append(dict(mw=case['mw'], valid=False))
                continue
            truth = restrict(base.cube(native['fine'], case), 4)[:, 2:-2, 2:-2, 2:-2]
            field = restrict(generated[key][...], 4)[:, 2:-2, 2:-2, 2:-2]
            rows.append(dict(mw=case['mw'], valid=True, score=score(field, truth, restrict(truth, 2), .75, 2)))
    repair.dump('first_scale.json', rows)
    repair.dump('result.json', dict(status=('COMPLETE_DRIVER_JUDGMENT_REQUIRED' if steps == 6000 else 'INCONCLUSIVE_LEARNING_BUDGET'),
        steps=steps, learning_seconds=seconds, tests=tests.testsRun, exact_routing_checks=3,
        fixed_network_unchanged=True, categorical_unchanged=True,
        valid_draws=sum(r['valid'] for r in draws),
        individual_pass=sum(r.get('comparison', {}).get('development_pass', False) for r in draws),
        ensemble_pass=sum(r['development_pass'] for r in ensembles),
        observed_posterior=False, memory=repair.check(), seconds=time.monotonic()-repair.START,
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT']))


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        if repair.CREATED:
            repair.dump('failure.json', dict(error=str(error)))
        raise
