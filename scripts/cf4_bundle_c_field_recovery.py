"""Single approved control/energy-score experiment with a terminal decision."""
import copy
import itertools
import json
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch

import cf4_bundle_c_flow_pilot as pilot
from cf4_bundle_c_flow_decision import likelihood, score
from cf4_bundle_c_continuous import read_periodic_patch
from cf4_conditional_split_flow import ConditionalSplitFlow, configure_precision
from cf4_continuous_matter import restrict
from cf4_flow_energy import (sample_trace, trace_backward, feature_groups,
                             normalize_groups, energy_coefficients)
from cf4_split_moments import encode_tree
from cf4_conditional_split_flow import condition

ROOT = pilot.ROOT
SOURCE = ROOT / 'conditional_flow_v2_full_context'
OUT = ROOT / 'field_recovery_v1'
CFG = json.loads(Path('config/cf4_field_recovery_v1.json').read_text())
START = float(os.environ.get('CF4_JOB_START', time.time()))
CREATED = False
RESULT = dict(status='NOT_STARTED', branches={}, end_state=None,
              limits=CFG['limits'], actual_high_resolution_LG_posterior=False)


class TrialStop(RuntimeError):
    pass


def dump(name, value):
    with (OUT / name).open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def event(status, **values):
    row = dict(status=status, elapsed_seconds=time.time()-START, **values)
    dump('status.json', row)
    print(json.dumps(row), flush=True)


def check(learning=True):
    cap = CFG['learning_deadline_seconds'] if learning else CFG['total_deadline_seconds']
    if time.time()-START > cap:
        raise TrialStop('INCOMPLETE_TIME_CAP')
    if torch.cuda.max_memory_allocated() > CFG['gpu_peak_limit_GiB']*2**30:
        raise TrialStop('STOP_MEMORY_ESTIMATE_EXCEEDED')


def vector(model):
    result = torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                        for p in model.parameters()])
    if not torch.isfinite(result).all():
        raise TrialStop('STOP_NONFINITE_GRADIENT')
    return result


def make_model(checkpoint):
    state = checkpoint['model']
    model = ConditionalSplitFlow(state['location'].flatten().numpy(), state['spread'].flatten().numpy()).cuda()
    model.load_state_dict(state)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])
    # load_state_dict can retain CPU scalar-step tensor references: isolate forks.
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    if any(group['lr'] != CFG['learning_rate'] for group in optimizer.param_groups):
        raise ValueError('checkpoint optimizer rate differs from approved rate')
    return model, optimizer


def levels_at(source, origin):
    fine = read_periodic_patch(source['fine'], np.array(origin)*8, 128)
    return {level: restrict(fine, 2**level) if level else fine for level in range(7)}


def mode_data(levels, mode):
    # modes0/1/2: .75/.375/.1875 teacher-parent; mode3: full rollout.
    target_level = (2, 1, 0, 0)[mode]
    parent_level = (3, 2, 1, 3)[mode]
    return levels[parent_level], levels[target_level], 2**(parent_level-target_level)


def prepare_features(fields):
    raw, normalization = {}, {}
    for mode in range(4):
        rows = []
        for levels in fields:
            parent, target, ratio = mode_data(levels, mode)
            rows.append(feature_groups(target, CFG['mode_scales'][mode][-1], parent, ratio))
        scales = []
        for group in range(3):
            scales.append(float(np.mean([np.linalg.norm(a[group]-b[group])
                                        for a, b in itertools.combinations(rows, 2)])))
        if not np.isfinite(scales).all():
            raise ValueError('nonfinite training-only feature scale')
        raw[mode] = [normalize_groups(row, scales) for row in rows]
        normalization[mode] = scales
    dump('features.json', dict(scales=normalization, unavailable_zero_groups={k: [i for i, s in enumerate(v) if s == 0]
                                                                           for k, v in normalization.items()},
        rule='Mean Euclidean distance across ALL unordered pairs of12 original training cubes, separately by mode/group.',
        group_names=['eight fixed log1p-density blocks', 'eight natural log-power bands', 'three bulk and three sigma RMS km/s'],
        block_slices={str(n): [[c-4, c+4] for c in (n//4, 3*n//4)] for n in (32, 64, 128)},
        context='Original native coarse conditions; reused one-box development, no actual LG input.'))
    return raw, normalization


def structural_gradient(model, levels, mode, target, scales, seed, multiplier=1., baseline=0.):
    torch.manual_seed(seed)
    parent, _, ratio = mode_data(levels, mode)
    features, traces = [], []
    for _ in range(2):
        field, trace = sample_trace(model, parent, CFG['mode_scales'][mode], check)
        features.append(normalize_groups(feature_groups(field, CFG['mode_scales'][mode][-1], parent, ratio), scales))
        traces.append(trace)
        del field
    energy, coefficients = energy_coefficients(*features, target, baseline)
    scores = [trace_backward(model, trace, multiplier*c, check) for trace, c in zip(traces, coefficients)]
    return dict(energy=energy, coefficients=coefficients.tolist(), logq=scores,
                next_baseline_target=float(coefficients.sum()+baseline))


def screen(model, data, fields, targets, scales):
    model.zero_grad(set_to_none=True)
    start = time.monotonic()
    for level in range(6):
        value = -model.log_prob(*pilot.tensors(data[level][0], full_context=True)).mean()/6
        value.backward()
    torch.cuda.synchronize()
    nll_seconds = (time.monotonic()-start)/6
    g_native = vector(model).clone()
    halves, rows, seconds = [], [], []
    for half in range(2):
        aggregate = torch.zeros_like(g_native)
        for mode, weight in enumerate(CFG['screen_mode_weights']):
            check()
            model.zero_grad(set_to_none=True)
            begin = time.monotonic()
            info = structural_gradient(model, fields[0], mode, targets[mode][0], scales[mode],
                                       CFG['screen_seed']+4*half+mode)
            grad = vector(model)
            norm = float(torch.linalg.vector_norm(grad))
            if norm == 0:
                raise TrialStop('STOP_ZERO_STRUCTURAL_GRADIENT')
            aggregate += weight*grad
            torch.cuda.synchronize()
            elapsed = time.monotonic()-begin
            seconds.append(elapsed)
            rows.append(dict(half=half, mode=mode, gradient_norm=norm, seconds=elapsed, **info))
            dump('screen_progress.json', rows)
            event('SCREEN', half=half, mode=mode, seconds=elapsed)
        halves.append(aggregate)
    g_es = (halves[0]+halves[1])/2
    native_norm = float(torch.linalg.vector_norm(g_native))
    es_norm = float(torch.linalg.vector_norm(g_es))
    agreement = float(torch.dot(halves[0], halves[1]))
    if not (native_norm > 0 and es_norm > 0 and np.isfinite([native_norm, es_norm]).all()):
        raise TrialStop('STOP_INVALID_WEIGHT_GRADIENTS')
    coefficient = native_norm/es_norm
    # Time both branches plus structural mixture; worst observed time per mode,
    # 20% headroom, and a further native-step allowance for Adam/clip overhead.
    mode_seconds = np.maximum(seconds[:4], seconds[4:])
    forecast = CFG['steps_per_branch']*(3*nll_seconds + np.dot(CFG['screen_mode_weights'], mode_seconds))
    remaining = CFG['learning_deadline_seconds']-(time.time()-START)
    report = dict(rows=rows, native_gradient_norm=native_norm, structural_gradient_norm=es_norm,
        split_half_inner_product=agreement, coefficient=coefficient, nll_seconds_per_record=nll_seconds,
        projected_learning_seconds=float(forecast), with_20pct_headroom=float(1.2*forecast),
        remaining_learning_seconds=remaining, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        status='NO_GROSS_PATHOLOGY_DETECTED_NOT_SNR_CERTIFICATION' if agreement > 0 and 1.2*forecast < remaining
               else 'STOP_ESTIMATOR_SCREEN_NO_NEW_BATCH',
        screen_context='Both halves same first training cube, independent pairs, all4 modes; no heldout data.')
    dump('screen.json', report)
    RESULT['screen'] = report
    if not np.isfinite(coefficient) or agreement <= 0 or 1.2*forecast >= remaining:
        raise TrialStop(report['status'])
    model.zero_grad(set_to_none=True)
    return coefficient


@torch.no_grad()
def fixed_nll(model, fields, retained):
    return [dict(split=split, case=i, dx=.1875*2**level, **likelihood(model, levels[level], .1875*2**level))
            for split, cases in [('training', fields[:2]), ('heldout', retained)]
            for i, levels in enumerate(cases) for level in range(6)]


def train_branch(name, checkpoint, data, fields, targets, scales, coefficient, schedule):
    model, optimizer = make_model(checkpoint)
    output = OUT / name
    output.mkdir(exist_ok=False)
    baseline = [0.]*4
    structural_index = 0
    history = []
    RESULT['branches'][name] = dict(steps=0, evaluation=None)
    for step, (level, index, field_index) in enumerate(schedule, 1):
        check()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -model.log_prob(*pilot.tensors(data[level][index], full_context=True)).mean()
        if not torch.isfinite(loss):
            raise TrialStop('STOP_NONFINITE_NATIVE_NLL')
        loss.backward()
        info = None
        if name == 'repair':
            mode = 3 if step % 8 == 0 else structural_index % 3
            structural_index += int(mode != 3)
            info = structural_gradient(model, fields[field_index], mode, targets[mode][field_index], scales[mode],
                                       CFG['structural_seed']+step, coefficient, baseline[mode])
            baseline[mode] = CFG['past_baseline_decay']*baseline[mode] + (1-CFG['past_baseline_decay'])*info['next_baseline_target']
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'])
        if not torch.isfinite(grad):
            raise TrialStop('STOP_NONFINITE_COMBINED_GRADIENT')
        optimizer.step()
        RESULT['branches'][name]['steps'] = step
        if step == 1 or step % 100 == 0:
            history.append(dict(step=step, nll=float(loss.detach()), unclipped_gradient_norm=float(grad), structural=info))
            event('TRAINING', branch=name, **history[-1])
            with (output / 'history.json').open('w') as f:
                json.dump(history, f, indent=2, allow_nan=False)
        if step == 500:
            with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
                retained = [levels_at(source, o) for o in checkpoint['request']['heldout_origins_1p5_cells']]
            dump(f'{name}_midpoint_nll.json', fixed_nll(model, fields, retained))
            del retained
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=checkpoint['step']+step,
                    new_steps=step, source_commit=os.environ['EXPECTED_COMMIT'], config=CFG,
                    parent_checkpoint=str(SOURCE / 'checkpoint.pt'), structural_coefficient=coefficient if name == 'repair' else 0),
               output / 'checkpoint.pt')
    del optimizer, model
    torch.cuda.empty_cache()


def high_band_table(rows):
    output = []
    for split, mode, dx in itertools.product(['training', 'heldout'], ['teacher', 'rollout'], [.75, .375, .1875]):
        selected = [r for r in rows if r['split'] == split and r['mode'] == mode and r['dx'] == dx and r['kind'] == 'fine']
        output.append(dict(split=split, mode=mode, dx=dx, comparisons=len(selected),
            high_band_mean=float(np.mean([r['ratios']['power'][4:] for r in selected])) if selected else None,
            high_band_log_rms=float(np.mean([r['high_band_log_rms'] for r in selected])) if selected else None))
    return output


@torch.no_grad()
def trained_inverse(model, native):
    # Reuse the original trained-flow inverse/Jacobian check, one fixed record.
    tree = encode_tree(native)
    z, mask, parent = tree[0]
    context, valid = condition(parent, parent, .1875, 0)
    data = pilot.tensors((z, mask, context, valid), full_context=True)
    active = data[1] == 2
    original = ((data[0]-model.location)/model.spread)*active
    x, jac = original.clone(), torch.zeros_like(original[:, 0])
    for layer in reversed(model.layers):
        x, det = layer(x, data[2], data[1], inverse=True)
        jac += det
    for layer in model.layers:
        x, det = layer(x, data[2], data[1])
        jac += det
    relative = float((abs(x-original)/torch.maximum(abs(original), torch.ones_like(original))).max())
    determinant = float(abs(jac).max())
    report = dict(inverse_relative_error=relative, logdet_error=determinant,
                  passed=bool(relative < 1e-4 and determinant < 1e-3))
    return report


@torch.no_grad()
def evaluate_branch(name, checkpoint, fields, request):
    model, optimizer = make_model(checkpoint)
    del optimizer
    saved = torch.load(OUT / name / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    del saved
    model.eval()
    inverse = trained_inverse(model, fields[0][0])
    dump(f'{name}_inverse.json', inverse)
    if not inverse['passed']:
        raise TrialStop('STOP_TRAINED_INVERSE_FAILED')
    # Original16 draws and thresholds, unchanged implementation/seeds.
    pilot.OUT = OUT / name
    comparisons = pilot.evaluate(model, request['config'])
    dump(f'{name}_development.json', comparisons)
    diagnostic, slices = [], {}
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        retained = [levels_at(source, o) for o in request['heldout_origins_1p5_cells']]
    dump(f'{name}_end_nll.json', fixed_nll(model, fields, retained))
    for split, cases in [('training', fields[:2]), ('heldout', retained)]:
        for case, levels in enumerate(cases):
            for draw in range(2):
                seed = 99803+100*case+draw+(10000 if split == 'training' else 0)
                torch.manual_seed(seed)
                base = levels[3]
                rolled = base
                for stage, dx in enumerate([.75, .375, .1875], 1):
                    check(False)
                    before = torch.cuda.get_rng_state()
                    rolled = pilot.refine(model, rolled, dx)
                    after = torch.cuda.get_rng_state()
                    torch.cuda.set_rng_state(before)
                    teacher = pilot.refine(model, levels[4-stage], dx)
                    torch.cuda.set_rng_state(after)
                    if stage == 1 and not np.array_equal(rolled, teacher):
                        raise ValueError('same-parent paired RNG control failed')
                    for mode, value in [('rollout', rolled), ('teacher', teacher)]:
                        diagnostic.append(dict(split=split, case=case, draw=draw, mode=mode,
                            kind='fine', dx=dx, **score(value, levels[3-stage], dx, base, 2**stage)))
                        if stage == 3 and draw == 0:
                            slices[f'{split}_{case}/{mode}'] = value[0, :, :, 60:68].mean(2)/levels[0][0].mean()
                    del teacher
                slices[f'{split}_{case}/native'] = levels[0][0, :, :, 60:68].mean(2)/levels[0][0].mean()
            event('EVALUATING', branch=name, split=split, case=case)
    with h5py.File(OUT / name / 'diagnostic_slabs.h5', 'x') as output:
        for key, value in slices.items():
            output.create_dataset(key, data=value)
        output.attrs['status'] = 'NOT_OBSERVED_LG'
    table = high_band_table(diagnostic)
    dump(f'{name}_paired_generation.json', diagnostic)
    report = dict(all16_development_pass=all(r['development_pass'] for r in comparisons),
                  passed_draws=sum(r['development_pass'] for r in comparisons), comparisons=len(comparisons), high_band=table)
    RESULT['branches'][name]['evaluation'] = report
    dump('result.json', RESULT)
    del model
    torch.cuda.empty_cache()
    return report


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    with h5py.File(OUT / 'control/diagnostic_slabs.h5', 'r') as control, h5py.File(OUT / 'repair/diagnostic_slabs.h5', 'r') as repair:
        for case in range(2):
            for col, field in enumerate([control[f'heldout_{case}/native'][:], control[f'heldout_{case}/rollout'][:], repair[f'heldout_{case}/rollout'][:]]):
                axes[case, col].imshow(np.log10(1+field), origin='lower', extent=(0, 24, 0, 24), vmin=0, vmax=2.5, cmap='magma')
    for ax, title in zip(axes[0], ['Native TNG', 'NLL control', 'NLL + energy score']):
        ax.set_title(title)
    fig.suptitle('Fixed draw0, .1875 grid; NOT observed LG; same projection/color scale')
    fig.savefig(OUT / 'density_comparison.png', dpi=140)
    plt.close(fig)


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU required; no CPU/login fallback')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    torch.set_num_threads(2)
    precision = configure_precision(True)
    checkpoint = torch.load(SOURCE / 'checkpoint.pt', map_location='cpu', weights_only=False)
    request = json.loads((SOURCE / 'request.json').read_text())
    checkpoint['request'] = request
    if checkpoint['step'] != 6000 or not checkpoint['config']['full_context_training'] or not checkpoint['config']['strict_fp32']:
        raise ValueError('not the approved v2 starting checkpoint')
    RESULT.update(job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'],
                  precision=precision, parent_checkpoint_commit=checkpoint['source_commit'], config=CFG)
    historical = json.loads((ROOT / 'flow_decision_v1/result.json').read_text())
    RESULT['starting_high_band'] = high_band_table([r for r in historical['paired_generation'] if r['model'] == 'v2'])
    RESULT['starting_native_likelihood'] = [r for r in historical['native_likelihood'] if r['model'] == 'v2']
    RESULT['starting_development_passed_draws'] = sum(r['development_pass'] for r in json.loads((SOURCE / 'result.json').read_text())['comparisons'])
    dump('result.json', RESULT)
    pilot.OUT = OUT / 'preparation'
    pilot.OUT.mkdir(exist_ok=False)
    data, location, spread = pilot.prepare(request['config'])
    prepared = json.loads((pilot.OUT / 'request.json').read_text())
    for key in ('train_origins_1p5_cells', 'heldout_origins_1p5_cells'):
        if prepared[key] != request[key]:
            raise ValueError('native split changed')
    if not (np.array_equal(location.astype(np.float32), checkpoint['model']['location'].numpy().flatten())
            and np.array_equal(spread.astype(np.float32), checkpoint['model']['spread'].numpy().flatten())):
        raise ValueError('coordinate normalization differs from frozen v2')
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        fields = [levels_at(source, o) for o in request['train_origins_1p5_cells']]
    targets, scales = prepare_features(fields)
    model, optimizer = make_model(checkpoint)
    del optimizer
    event('SCREEN_START')
    coefficient = screen(model, data, fields, targets, scales)
    del model
    rng = np.random.default_rng(CFG['paired_native_seed'])
    schedule = [((step-1) % 6, int(rng.integers(len(data[(step-1) % 6]))), int(rng.integers(len(fields))))
                for step in range(1, CFG['steps_per_branch']+1)]
    dump('frozen_run.json', dict(config=CFG, coefficient=coefficient, paired_schedule=schedule,
        rng_scope='Fresh identical native record order, not restoration of unsaved v2 RNG state.',
        parent=str(SOURCE / 'checkpoint.pt'), approved_plan='BUNDLE_C_FIELD_RECOVERY_PLAN.md'))
    for name in ['control', 'repair']:
        train_branch(name, checkpoint, data, fields, targets, scales, coefficient, schedule)
    del data
    reports = {name: evaluate_branch(name, checkpoint, fields, request) for name in ['control', 'repair']}
    control, repair = reports['control'], reports['repair']
    contrasts = []
    for mode in ('teacher', 'rollout'):
        values = [next(r['high_band_log_rms'] for r in reports[name]['high_band']
                       if r['split'] == 'heldout' and r['mode'] == mode and r['dx'] == .1875)
                  for name in ('control', 'repair')]
        contrasts.append(dict(mode=mode, control=values[0], repair=values[1], repair_improves=values[1] < values[0]))
    if not control['all16_development_pass'] and not repair['all16_development_pass']:
        verdict = 'CLOSE_THIS_REPAIR_LINE_BOTH_FAIL'
    elif repair['all16_development_pass'] and all(r['repair_improves'] for r in contrasts):
        verdict = 'REPAIR_DEVELOPMENT_CANDIDATE_NOT_POSTERIOR'
    elif control['all16_development_pass']:
        verdict = 'CONTROL_DEVELOPMENT_CANDIDATE_NO_ES_SUPERIORITY'
    else:
        verdict = 'NO_PROMOTION_AMBIGUOUS_COMPARISON'
    plots()
    RESULT.update(status=verdict, end_state=reports, heldout_contrasts=contrasts)


if __name__ == '__main__':
    try:
        run()
    except TrialStop as exc:
        RESULT.update(status=str(exc), terminal_trial=True, end_state=RESULT.get('end_state'))
    except Exception:
        if CREATED:
            RESULT.update(status='FAILED_NO_AUTOMATIC_RETRY', traceback=traceback.format_exc())
        raise
    finally:
        if CREATED:
            RESULT.update(elapsed_seconds=time.time()-START, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                          gpu=torch.cuda.get_device_name())
            dump('result.json', RESULT)
            event(RESULT['status'], branches={k: v['steps'] for k, v in RESULT['branches'].items()})
