"""Matched finite-capacity correction, one allocation, no sample rescaling."""
import copy
import json
import os
from pathlib import Path
import resource
import time
import traceback
import unittest

import h5py
import numpy as np
import torch

import cf4_bundle_c_stable_field as base
from cf4_stable_field import StableField, noisy_record
from cf4_field_link_repair import LinkedContinuous, budget_weights
from cf4_spatial_diffusion import augment
from cf4_role_locations import transform_positions
from cf4_continuous_matter import restrict
from cf4_conditional_split_flow import configure_precision
from cf4_bundle_c_scale_link import score as physical_score

CFG = json.loads(Path('config/cf4_field_link_repair_v1.json').read_text())
OUT = base.ROOT/CFG['output_name']
START = time.monotonic()
CREATED = False


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def check():
    if time.monotonic()-START > CFG['application_seconds']:
        raise TimeoutError('110-minute total application budget')
    usage = dict(host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
    if any(usage[key] > CFG[key] for key in usage):
        raise MemoryError(f'fixed memory envelope exceeded: {usage}')
    return usage


def train(branch, initial, cases, slab, location, spread):
    model = StableField()
    model.load_state_dict(initial, strict=True)
    if branch == 'linked_budget':
        model.continuous = LinkedContinuous(model.continuous)
    model = model.cuda().train()
    model.categorical.eval().requires_grad_(False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.continuous.parameters(), lr=CFG['learning_rate'],
        weight_decay=CFG['weight_decay'], foreach=False)
    rng = np.random.default_rng(CFG['seed'])
    order = rng.permutation(len(cases))
    history, completed = [], 0
    started = time.monotonic()
    def save():
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
            branch=branch, step=completed, location=location, spread=spread, config=CFG,
            initial_checkpoint=str(base.ROOT/'stable_field_v1/checkpoint_24000.pt'),
            source_commit=os.environ['EXPECTED_COMMIT'], numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), base.OUT/f'checkpoint_{completed:05d}.pt')
        base.dump('history.json', history)
    try:
        for step in range(1, CFG['steps_per_branch']+1):
            check()
            if time.monotonic()-started > CFG['learning_seconds_per_branch']:
                break
            offset = (step-1) % len(cases)
            if offset == 0 and step > 1:
                order = rng.permutation(len(cases))
            case = cases[int(order[offset])]
            symmetry = int(rng.integers(48))
            level = CFG.get('fixed_level', (step-1) % 3)
            field = augment(base.cube(slab, case), symmetry)
            observer = transform_positions(torch.tensor([44.]*3), symmetry, 80).numpy()*base.DX
            clean, codes, coarse = base.inputs(field, level, observer, location, spread)
            weights = None
            if branch == 'linked_budget':
                weights = torch.from_numpy(budget_weights(restrict(field, 2**(level+1))))[None].cuda()
            del field
            # Identical examples, times, noise and masks across BOTH branches.
            torch.manual_seed(CFG['seed']+step)
            t = torch.randint(1, base.CFG['diffusion_steps']+1, (1,), device='cuda')
            noisy, masked, target, _ = noisy_record(clean, codes, t, base.CFG['diffusion_steps'])
            optimizer.zero_grad(set_to_none=True)
            predicted = model.continuous(noisy, masked, coarse, t, torch.tensor([float(level)], device='cuda'))
            errors = (predicted-target).square()
            loss = errors.mean() if weights is None else (errors*weights).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite repair objective')
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.continuous.parameters(), CFG['gradient_clip'], error_if_nonfinite=True))
            optimizer.step()
            with torch.no_grad():
                for dest, src in zip(ema.continuous.parameters(), model.continuous.parameters()):
                    dest.lerp_(src, 1-CFG['ema_decay'])
            completed = step
            history.append(dict(step=step, mw=case['mw'], level=level, symmetry=symmetry, t=int(t[0]),
                objective=float(loss.detach()), unweighted_v_MSE=float(errors.detach().mean()), gradient_norm=norm))
            if step == 1 or step % 200 == 0:
                base.dump('history.json', history)
                base.event('REPAIR_TRAINING', branch=branch, step=step, target=CFG['steps_per_branch'],
                    seconds_learning=time.monotonic()-started, last=history[-1], memory=check())
            if step % CFG['checkpoint_every'] == 0:
                save()
            del loss, errors, predicted, clean, codes, coarse, noisy, masked, target, weights
    finally:
        save()
    for key, value in model.categorical.state_dict().items():
        torch.testing.assert_close(value.cpu(), initial['categorical.'+key], rtol=0, atol=0)
    return ema, completed, time.monotonic()-started


def endpoints(cases):
    rows = []
    with h5py.File(base.OUT/'first_draw_fields.h5', 'r') as saved, h5py.File(base.ROOT/'total_matter_v1/matter_moments.h5', 'r') as source:
        for case in cases:
            key = f'mw_{case["mw"]}/moments80'
            if key not in saved:
                rows.append(dict(mw=case['mw'], valid=False))
                continue
            native = base.cube(source['fine'], case)[:, 8:-8, 8:-8, 8:-8]
            generated = saved[key][:, 8:-8, 8:-8, 8:-8]
            rows.append(dict(mw=case['mw'], valid=True,
                score=physical_score(generated, native, restrict(native, 8), base.DX, 8)))
    base.dump('physical_endpoint.json', rows)
    return rows


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('source-pinned Slurm required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    torch.set_num_threads(2)
    configure_precision(True)
    torch.manual_seed(CFG['seed'])
    base.OUT, base.check = OUT, check
    dump('request.json', dict(config=CFG, source_commit=os.environ['EXPECTED_COMMIT'], job_id=os.environ['SLURM_JOB_ID'],
        gpu=torch.cuda.get_device_name(), original_criteria_unchanged=True, observed_LG_posterior=False))
    tested = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_cf4_stable_field', 'test_cf4_field_link_repair']))
    dump('tests.json', dict(tests=tested.testsRun, failures=len(tested.failures), errors=len(tested.errors)))
    if not tested.wasSuccessful():
        raise ValueError('focused tests failed')
    fit = base.ROOT/'stable_field_v1'
    saved = torch.load(fit/'checkpoint_24000.pt', map_location='cpu', weights_only=False)
    base.validate_evaluation_checkpoint(saved, json.loads((fit/'request.json').read_text()), json.loads((fit/'result.json').read_text()))
    initial, location, spread = saved['ema'], saved['location'], saved['spread']
    del saved
    cases, test, slab = base.cases_and_slab()
    reports = []
    for branch in CFG['branches']:
        base.OUT = OUT/branch
        base.OUT.mkdir(exist_ok=False)
        # This is a controlled fresh-optimizer comparison, not exact Adam resume.
        model, steps, seconds = train(branch, initial, cases, slab, location, spread)
        base.denoising(model, test, location, spread)
        draws, ensembles = base.evaluate(model, test, location, spread)
        physical = endpoints(test)
        valid = sum(row['valid'] for row in draws)
        passed = sum(row['development_pass'] for row in ensembles)
        summary = dict(branch=branch, additional_steps=steps, learning_seconds=seconds,
            training_complete=steps == CFG['steps_per_branch'], valid_draws=valid,
            individual_morphology_pass=sum(row.get('comparison', {}).get('development_pass', False) for row in draws),
            ensemble_morphology_pass=passed, full_original_feasibility=(steps == CFG['steps_per_branch'] and valid == 16 and passed == 8),
            categorical_unchanged=True, observed_LG_posterior=False, physical_endpoint_cases=len(physical))
        base.dump('result.json', summary)
        reports.append(summary)
        status = ('INCONCLUSIVE_LEARNING_BUDGET' if not summary['training_complete'] else
            'PARTIAL_COMPARISON' if len(reports) < 2 else 'COMPLETE_MATCHED_COMPARISON_DRIVER_JUDGMENT_REQUIRED')
        dump('result.json', dict(status=status,
            branches=reports, source_commit=os.environ['EXPECTED_COMMIT'], job_id=os.environ['SLURM_JOB_ID'],
            memory=check(), seconds=time.monotonic()-START))
        del model
        if not summary['training_complete']:
            break  # Do not spend another fit after an unmatched budget-short arm.


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        if CREATED:
            dump('failure.json', dict(status='INCONCLUSIVE_TECHNICAL_OR_RESOURCE_FAILURE',
                error=str(error), traceback=traceback.format_exc(), seconds=time.monotonic()-START))
        raise
