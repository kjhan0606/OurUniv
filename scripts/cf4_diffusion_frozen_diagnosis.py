"""Fixed saved-state diagnostics. Never instantiate an optimizer or take a step."""
import json
import os
from pathlib import Path
import resource
import time
import traceback

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

from cf4_bundle_c_continuous import read_periodic_patch
from cf4_bundle_c_spatial_diffusion import native_record
from cf4_continuous_matter import restrict
from cf4_conditional_split_flow import configure_precision
from cf4_split_moments import NODES, decode
from cf4_spatial_diffusion import SpatialDenoiser, pack, context, canonical, sample, gaussian_schedule

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUT = ROOT / 'diffusion_frozen_diagnosis_v1'
REPORT = dict(status='INCOMPLETE', optimizer_updates=0)
START = time.monotonic()
CREATED = False


def save():
    with (OUT / 'result.json').open('w') as f:
        json.dump(REPORT, f, indent=2, allow_nan=False)


def check():
    if time.monotonic()-START > 1100:
        raise TimeoutError('bounded diagnostic deadline')


def rms(x):
    return float(x.detach().float().square().mean().sqrt())


def inputs(native, level, location, spread):
    return native_record(native, level, location, spread)


def fixed_noise(data, level, t):
    clean, codes, coarse = data
    torch.manual_seed(782100+level*100+t)
    _, abar = gaussian_schedule(100, 'cuda')
    a = abar[t]
    noise = torch.randn_like(clean)
    noisy = a.sqrt()*clean+(1-a).sqrt()*noise
    masked = torch.rand_like(clean) < t/100
    return noisy, torch.where(masked, 4, codes), noise, a, masked


def denoising(model, native, location, spread, name):
    rows = []
    model.eval()
    with torch.no_grad():
        for level in [2, 0]:
            data = inputs(native, level, location, spread)
            for t in [1, 25, 50, 75, 100]:
                check()
                noisy, codes, noise, a, masked = fixed_noise(data, level, t)
                pred, logits = model(noisy, codes, data[2], torch.tensor([t], device='cuda'), torch.tensor([float(level)], device='cuda'))
                # Explicit fixed-confidence majority baseline, not a learned generator.
                majority = data[1].flatten(2).mode(2).values[:, :, None, None, None]
                baseline_nll = torch.where(data[1] == majority, -np.log(.99), -np.log(.01/3))
                model_ce = F.cross_entropy(logits.flatten(0, 1), data[1].flatten(0, 1), reduction='none').reshape_as(noise)
                reference = noisy/torch.sqrt(1-a)
                denominator = rms(pred)*rms(noise)
                rows.append(dict(model=name, dx=.1875*2**level, t=t, alpha_bar=float(a),
                    clean_rms=rms(data[0]), clean_absmax=float(data[0].abs().max()),
                    coarse_rms=rms(data[2]), noisy_rms=rms(noisy), prediction_rms=rms(pred),
                    noise_rms=rms(noise), model_MSE=rms(pred-noise)**2,
                    zero_MSE=rms(noise)**2, elementary_MSE=rms(reference-noise)**2,
                    categorical_loss=float((model_ce*masked*(100/t)).mean()),
                    majority_baseline_loss=float((baseline_nll*masked*(100/t)).mean()),
                    majority_baseline_confidence=.99,
                    majority_accuracy=float((data[1] == majority).float().mean()),
                    pred_noise_cosine=float((pred*noise).mean())/denominator if denominator else None))
            del data
    return rows


def weight_report(raw, ema, optimizer_state):
    keys = ['stem.weight', 'down.0.0.c1.weight', 'up.2.1.c2.weight', 'norm.weight', 'norm.bias', 'head.weight', 'head.bias']
    report = {}
    for key in keys:
        for group, sl in ([('noise', slice(0, 49)), ('category', slice(49, None))] if key.startswith('head.') else [('all', slice(None))]):
            r, e = raw[key][sl], ema[key][sl]
            report[key+'/'+group] = dict(raw_rms=rms(r), ema_rms=rms(e), raw_ema_delta_rms=rms(r-e))
    steps = [float(v['step']) for v in optimizer_state.values() if 'step' in v]
    return dict(parameters=report, optimizer_state_count=len(steps), minimum_step=min(steps), maximum_step=max(steps),
                initialization_comparison='UNAVAILABLE: no saved step-0 checkpoint or initialization hash; seed replay is not verified and is not evidence of weight updates')


def gradients(model, data):
    noisy, codes, noise, _, masked = fixed_noise(data, 2, 50)
    named = dict(model.named_parameters())
    reports, grads = {}, {}
    for mode, train in [('plain', False), ('checkpointed', True)]:
        model.train(train)
        for objective in ['continuous', 'categorical']:
            model.zero_grad(set_to_none=True)
            prediction, logits = model(noisy, codes, data[2], torch.tensor([50], device='cuda'), torch.tensor([2.], device='cuda'))
            loss = (prediction-noise).square().mean() if objective == 'continuous' else (
                F.cross_entropy(logits.flatten(0, 1), data[1].flatten(0, 1), reduction='none').reshape_as(noise)*masked*2).mean()
            loss.backward()
            selected = {k: p.grad.detach().cpu().clone() for k, p in named.items() if p.grad is not None}
            grads[(mode, objective)] = selected
            reports[mode+'/'+objective] = dict(loss=float(loss.detach()),
                global_gradient_norm=float(torch.sqrt(sum(g.double().square().sum() for g in selected.values()))),
                layer_gradient_rms={k: rms(selected[k]) for k in ['stem.weight', 'down.0.0.c1.weight', 'norm.weight', 'head.weight']},
                head_noise_gradient_rms=rms(selected['head.weight'][:49]),
                head_noise_gradient_exact_zero=not bool(torch.count_nonzero(selected['head.weight'][:49])),
                missing_gradient_parameters=[k for k in named if k not in selected],
                head_category_gradient_rms=rms(selected['head.weight'][49:]))
    for objective in ['continuous', 'categorical']:
        a, b = grads[('plain', objective)], grads[('checkpointed', objective)]
        reports[objective+'_checkpoint_max_absolute_difference'] = max(float((a[k]-b[k]).abs().max()) for k in a)
        denominator = sum(a[k].double().square().sum() for k in a)
        relative = float(torch.sqrt(sum((a[k].double()-b[k].double()).square().sum() for k in a)/denominator)) if denominator > 0 else None
        reports[objective+'_checkpoint_relative_L2_difference'] = relative
        reports[objective+'_checkpoint_agreement_rtol_1e_4_atol_1e_7'] = all(torch.allclose(a[k], b[k], rtol=1e-4, atol=1e-7) for k in a)
    model.zero_grad(set_to_none=True)
    model.eval()
    return reports


def trace(model, parent, location, spread):
    records = []
    def hook(module, args, output):
        t = int(args[3][0])
        if t in [100, 75, 50, 25, 1]:
            records.append(dict(t=t, input_rms=rms(args[0]), input_absmax=float(args[0].abs().max()),
                                eps_rms=rms(output[0]), eps_absmax=float(output[0].abs().max())))
    handle = model.register_forward_hook(hook)
    torch.manual_seed(99803)
    try:
        z, codes = sample(model, torch.from_numpy(context(parent, .75))[None].cuda(), .75, location, spread, 100, check)
    finally:
        handle.remove()
    details = dict(reverse=records, final_coordinate_absmax_per_channel=np.max(abs(z), axis=(1, 2, 3)).tolist(), nodes=[])
    tree = {(0, 8): parent}
    for i, (lo, hi) in enumerate(NODES):
        p = tree[(lo, hi)]
        raw = codes[7*i:7*i+7]
        legal = canonical(raw, p)
        nodez = z[7*i:7*i+7]
        fractions = [0, 4, 5, 6]  # Velocity coordinates use tanh, not sigmoid fractions.
        saturated = ((expit(nodez[fractions]) == 0) | (expit(-nodez[fractions]) == 0)) & (legal[fractions] == 2)
        item = dict(node=i, legal_continuous_counts=(legal == 2).sum(axis=(1, 2, 3)).tolist(),
                    corrected_counts=(legal != raw).sum(axis=(1, 2, 3)).tolist(),
                    fraction_channels=fractions, active_sigmoid_zero_counts=saturated.sum(axis=(1, 2, 3)).tolist())
        details['nodes'].append(item)
        try:
            middle = (lo+hi)//2
            tree[(lo, middle)], tree[(middle, hi)] = decode(nodez, legal, p)
        except ValueError as exc:
            item['failure'] = str(exc)
            break
    return details


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Slurm required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    REPORT.update(job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'])
    torch.set_num_threads(2)
    configure_precision(True)
    saved = torch.load(ROOT / 'spatial_diffusion_v1/checkpoint_30000.pt', map_location='cpu', weights_only=False)
    if saved['step'] != 30000 or saved['source_commit'] != '341b419':
        raise ValueError('unexpected saved-state provenance')
    torch.manual_seed(saved['config']['training_seed'])
    model = SpatialDenoiser(tuple(saved['config']['widths']))
    REPORT['weights'] = weight_report(saved['model'], saved['ema'], saved['optimizer']['state'])
    location, spread = saved['location'], saved['spread']
    recorded = json.loads((ROOT / 'spatial_diffusion_v1/representation.json').read_text())
    assert np.array_equal(location, np.asarray(recorded['location']))
    assert np.array_equal(spread, np.asarray(recorded['spread']))
    REPORT['normalization'] = dict(training_record_equals_checkpoint=True,
        training_input_path='original native_record, unchanged',
        sampler_constants='same checkpoint arrays passed to unchanged sample; no separate normalization',
        location=location.tolist(), spread=spread.tolist())
    REPORT['preceding_run_verdict'] = '338402: 8/8 retained draws are generation/support failures, not measured morphology failures'
    save()
    del saved['optimizer']
    model.cuda()
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        native = read_periodic_patch(source['fine'], np.array([13, 8, 10])*8, 128)
        parent = restrict(read_periodic_patch(source['fine'], np.array([33, 0, 0])*8, 128), 8)
    REPORT['denoising'] = []
    for name in ['model', 'ema']:
        model.load_state_dict(saved[name])
        REPORT['denoising'] += denoising(model, native, location, spread, name)
        save()
    model.load_state_dict(saved['model'])
    REPORT['gradients'] = gradients(model, inputs(native, 2, location, spread))
    save()
    model.load_state_dict(saved['ema'])
    REPORT['failed_seed_trace'] = trace(model.eval(), parent, location, spread)
    REPORT.update(status='FROZEN_DIAGNOSIS_COMPLETE_NOT_MODEL_ADOPTION', seconds=time.monotonic()-START,
                  host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
    save()
    print(json.dumps({k: REPORT[k] for k in ['status', 'seconds', 'optimizer_updates']}), flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if CREATED:
            REPORT.update(status='DIAGNOSIS_INCOMPLETE', error=str(exc), traceback=traceback.format_exc())
            save()
        raise
