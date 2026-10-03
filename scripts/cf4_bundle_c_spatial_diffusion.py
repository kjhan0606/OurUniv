"""One Slurm diffusion fit and fixed fine-field evaluation; no downstream launch."""
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

from cf4_bundle_c_continuous import read_periodic_patch
from cf4_bundle_c_flow_pilot import metrics, compare
from cf4_bundle_c_spatial_model import overlaps
from cf4_continuous_matter import restrict
from cf4_conditional_split_flow import configure_precision
from cf4_split_moments import encode_tree, roundtrip, state
from cf4_spatial_diffusion import (BRANCH_VERSION, SpatialDenoiser, augment, pack,
    unpack, canonical, context, diffusion_loss, sample)

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
CFG = json.loads(Path('config/cf4_spatial_diffusion_v1.json').read_text())
SIZING = json.loads(Path('config/cf4_spatial_diffusion_sizing_v1.json').read_text())
OUT = ROOT / CFG['output_name']
START = float(os.environ.get('CF4_JOB_START', time.time()))
CREATED = False
RESULT = dict(status='NOT_STARTED', completed_updates=0, observed_LG_posterior=False)


def dump(name, value):
    with (OUT / name).open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def event(status, **fields):
    row = dict(status=status, elapsed_seconds=time.time()-START, **fields)
    dump('status.json', row)
    print(json.dumps(row), flush=True)


def deadline():
    if time.time()-START > CFG['total_cap_seconds']:
        raise TimeoutError('total allocation application deadline; evaluation incomplete')


def memory():
    record = dict(host_maxrss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  gpu_max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  gpu_max_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
    if record['host_maxrss_GiB'] > CFG['host_peak_limit_GiB'] or record['gpu_max_reserved_GiB'] > CFG['gpu_peak_limit_GiB']:
        dump('memory_exceeded.json', record)
        raise RuntimeError('measured memory exceeds frozen sizing allowance; no automatic model change')
    return record


def crop_origin(rng):
    origin = np.array([rng.integers(17), rng.integers(50), rng.integers(50)])
    if any(overlaps(origin*1.5, np.asarray(h)*1.5) for h in CFG['heldout_origins_1p5_cells']):
        raise ValueError('training/heldout spatial overlap; no reject-until-pass selection')
    return origin


def environment_summary(field):
    coarse = restrict(field, 8)
    _, variance = state(coarse)
    return np.r_[np.log(coarse[0].mean()/(1e11*1.5**3)),
                 np.sqrt((coarse[0]*variance).sum(axis=(1, 2, 3))/coarse[0].sum())].tolist()


def native_record(field, level, location=None, spread=None):
    fine = restrict(field, 2**level) if level else field
    z, codes, root = pack(fine)
    if location is None:
        return z, codes, root
    # Unused coordinates have deterministic zero targets in the padded joint
    # representation; they are not extra physical degrees of freedom.
    clean = np.where(codes == 2, (z-location[:, None, None, None])/spread[:, None, None, None], 0)
    tensors = [torch.from_numpy(np.ascontiguousarray(a))[None].to('cuda') for a in
               (clean.astype(np.float32), codes.astype(np.int64), context(root, .1875*2**level))]
    return tensors


def prepare():
    rng = np.random.default_rng(CFG['normalization_seed'])
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        if source.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE':
            raise ValueError('complete native total-matter source required')
        # One read of the approved training slab (2.14GiB); no heldout voxels.
        slab = source['fine'][:, :256, :, :]
    sums, squares, count = np.zeros(49), np.zeros(49), np.zeros(49)
    reports, environments = [], []
    for case in range(CFG['normalization_cubes']):
        deadline()
        origin = crop_origin(rng)
        field = augment(read_periodic_patch(slab, origin*8, 128), int(rng.integers(48)))
        environments.append(environment_summary(field))
        for level in range(3):
            fine = restrict(field, 2**level) if level else field
            physical = roundtrip(fine, report=True)
            for _, codes, parent in encode_tree(fine):
                if not np.array_equal(codes, canonical(codes, parent)):
                    raise ValueError('canonical branch table changes native codes')
            z, codes, root = pack(fine)
            restored, corrections = unpack(z, codes, root)
            if np.sum(corrections['corrected_per_node_channel']) != 0:
                raise ValueError('native decoded-parent canonicalization is not identity')
            scale = np.maximum(np.max(abs(fine), axis=(1, 2, 3)), 1)
            error = np.max(abs(restored-fine), axis=(1, 2, 3))/scale
            if max(error) > 1e-9:
                raise ValueError('native joint canonical roundtrip failed')
            selected = rng.choice(z[0].size, min(4096, z[0].size), replace=False)
            values, active = z.reshape(49, -1)[:, selected], codes.reshape(49, -1)[:, selected] == 2
            sums += (values*active).sum(1)
            squares += (values*values*active).sum(1)
            count += active.sum(1)
            reports.append(dict(case=case, level=level, origin=origin.tolist(), physical=physical,
                                identity_error=error.tolist()))
            del z, codes, restored, root
        event('PREPARING_NATIVE', cube=case+1)
    if np.any(count == 0):
        raise ValueError('training normalization missing coordinate support')
    location = sums/count
    spread = np.sqrt(np.maximum(squares/count-location*location, 0))
    if np.any(spread <= 0):
        raise ValueError('constant coordinate normalization; no invented spread')
    dump('representation.json', dict(status='NATIVE_CANONICAL_IDENTITY_PASS_BEFORE_OPTIMIZER',
         branch_version=BRANCH_VERSION, records=reports, location=location.tolist(), spread=spread.tolist(),
         continuous_count=count.tolist(), normalization_environments=environments))
    return slab, location, spread


def checkpoint_model(model, ema, optimizer, step, location, spread, rng):
    # No storage probes/custom filesystem primitives. Each recovery file is new.
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, location=location, spread=spread, config=CFG,
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), source_commit=os.environ['EXPECTED_COMMIT']),
               OUT / f'checkpoint_{step:05d}.pt')


def learn(slab, location, spread):
    torch.manual_seed(CFG['training_seed'])
    rng = np.random.default_rng(CFG['training_seed'])
    model = SpatialDenoiser(tuple(CFG['widths'])).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != SIZING['parameters']:
        raise ValueError(f'model parameter count {parameters} differs from submitted sizing')
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'], foreach=False)
    history, environments = [], []
    training_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, CFG['steps']+1):
        if time.time()-START >= CFG['learning_cap_seconds']:
            break
        deadline()
        tick = time.time()
        origin = crop_origin(rng)
        field = augment(read_periodic_patch(slab, origin*8, 128), int(rng.integers(48)))
        #32 operational updates really include worst-case64^3 fwd/bwd/Adam/EMA.
        #After that, native scales are balanced; no second optimizer/trial.
        level = 0 if step <= CFG['operational_updates'] else (step-CFG['operational_updates']-1) % 3
        if step <= 32 or step % 100 == 0:
            environments.append(dict(step=step, origin=origin.tolist(), summary=environment_summary(field)))
        inputs = native_record(field, level, location, spread)
        del field
        optimizer.zero_grad(set_to_none=True)
        loss, losses = diffusion_loss(model, *inputs, torch.tensor([float(level)], device='cuda'), CFG['diffusion_steps'])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite diffusion loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 1-CFG['ema_decay'])
        torch.cuda.synchronize()
        history.append(dict(step=step, level=level, loss=float(loss.detach()), gradient_norm=float(norm),
                            seconds=time.time()-tick, **losses))
        RESULT['completed_updates'] = step
        del loss, inputs
        if step == CFG['operational_updates']:
            measured = memory()
            dump('operational.json', dict(status='WITHIN_SIZING_CONTINUE_SAME_FIT', updates=step,
                 parameters=parameters, measured=measured, submitted_sizing=SIZING,
                 worst_scale_update_seconds=[r['seconds'] for r in history],
                 note='Operational feasibility only, not evidence of learned morphology.'))
        if step == 1 or step % 100 == 0 or step == CFG['operational_updates']:
            event('TRAINING', step=step, target=CFG['steps'], losses=history[-1], memory=memory())
            dump('history.json', history)
        if step % CFG['checkpoint_every'] == 0:
            checkpoint_model(model, ema, optimizer, step, location, spread, rng)
    completed = RESULT['completed_updates']
    if completed % CFG['checkpoint_every'] != 0 or completed == 0:
        checkpoint_model(model, ema, optimizer, completed, location, spread, rng)
    dump('history.json', history)
    dump('training_environments.json', environments)
    RESULT.update(status='EVALUATING', training_seconds=time.time()-training_start, training_complete=completed == CFG['steps'],
                  model_parameters=parameters)
    dump('result.json', RESULT)
    del optimizer, model
    torch.cuda.empty_cache()
    return ema


def refine(model, parent, dx, location, spread):
    coarse = torch.from_numpy(context(parent, dx))[None].cuda()
    z, codes = sample(model, coarse, dx, location, spread, CFG['diffusion_steps'], deadline)
    return unpack(z, codes, parent)


def score(value, native, dx, base, ratio):
    measured, reference = metrics(value, dx, base, ratio), metrics(native, dx, base, ratio)
    if max(measured['coarse_error']) > 1e-8:
        raise ValueError('rollout/teacher seven-moment conservation failed')
    result = compare(measured, reference)
    result['reference'] = reference
    if not all(np.isfinite(np.asarray(r)).all() for r in result['ratios'].values()):
        raise ValueError('nonfinite fixed morphology metric')
    return result


def evaluate(model, location, spread):
    old = json.loads((ROOT / 'conditional_flow_v2_full_context/request.json').read_text())
    cases = [('training', i, o) for i, o in enumerate(old['train_origins_1p5_cells'][:2])]
    cases += [('heldout', i, o) for i, o in enumerate(CFG['heldout_origins_1p5_cells'])]
    rows, final, env = [], [], []
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source, \
         h5py.File(OUT / 'mock_fields.h5', 'x') as fields, h5py.File(OUT / 'diagnostic_slabs.h5', 'x') as slabs:
        fields.attrs['status'] = 'INCOMPLETE_NOT_OBSERVED_LG'
        for split, case, origin in cases:
            native = read_periodic_patch(source['fine'], np.asarray(origin)*8, 128)
            levels = [native]+[restrict(native, 2**i) for i in range(1, 4)]
            base = levels[3]
            env.append(dict(split=split, case=case, summary=environment_summary(native)))
            if split == 'heldout':
                fields.create_dataset(f'case_{case}/native', data=native, compression='gzip', compression_opts=1)
            slabs.create_dataset(f'{split}_{case}/native', data=native[0, :, :, 60:68].mean(2)/native[0].mean())
            for draw in range(CFG['draws_per_case'] if split == 'heldout' else 2):
                torch.manual_seed(CFG['sample_seed']+100*case+draw+(10000 if split == 'training' else 0))
                rolled, rollout_error = base, None
                for stage, dx in enumerate([.75, .375, .1875], 1):
                    deadline()
                    before = torch.cuda.get_rng_state()
                    try:
                        if rollout_error is not None:
                            raise ValueError(f'unavailable after upstream failure: {rollout_error}')
                        rolled, fixes = refine(model, rolled, dx, location, spread)
                        current = dict(split=split, case=case, draw=draw, mode='rollout', dx=dx,
                                       canonicalization=fixes, **score(rolled, levels[3-stage], dx, base, 2**stage))
                    except (ValueError, FloatingPointError) as exc:
                        rollout_error = str(exc)
                        current = dict(split=split, case=case, draw=draw, mode='rollout', dx=dx,
                                       development_pass=False, invalid=True, reason=rollout_error)
                    after = torch.cuda.get_rng_state()
                    rows.append(current)
                    if stage == 3 and split == 'heldout':
                        final.append(current)
                        if rollout_error is None:
                            fields.create_dataset(f'case_{case}/fine_{draw}', data=rolled, compression='gzip', compression_opts=1)
                    if stage == 3 and draw == 0 and rollout_error is None:
                        slabs.create_dataset(f'{split}_{case}/rollout', data=rolled[0, :, :, 60:68].mean(2)/native[0].mean())
                    #Paired true-parent diagnostics use the original2 draws/case.
                    if draw < 2:
                        if stage == 1:
                            teacher_row = dict(current, mode='teacher')
                        else:
                            torch.cuda.set_rng_state(before)
                            try:
                                teacher, fixes = refine(model, levels[4-stage], dx, location, spread)
                                teacher_row = dict(split=split, case=case, draw=draw, mode='teacher', dx=dx,
                                      canonicalization=fixes, **score(teacher, levels[3-stage], dx, base, 2**stage))
                                if stage == 3 and draw == 0:
                                    slabs.create_dataset(f'{split}_{case}/teacher', data=teacher[0, :, :, 60:68].mean(2)/native[0].mean())
                                del teacher
                            except (ValueError, FloatingPointError) as exc:
                                teacher_row = dict(split=split, case=case, draw=draw, mode='teacher', dx=dx,
                                                   development_pass=False, invalid=True, reason=str(exc))
                            finally:
                                torch.cuda.set_rng_state(after)
                        rows.append(teacher_row)
                    dump('paired_generation.json', rows)
                event('EVALUATING', split=split, case=case, draw=draw, final_draw_pass=current['development_pass'])
            del native, levels, base, rolled
        fields.attrs['status'] = 'FIXED_DRAW_EVALUATION_COMPLETE_NOT_OBSERVED_LG'
    dump('development.json', final)
    dump('evaluation_environments.json', env)
    return final, rows


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with h5py.File(OUT / 'diagnostic_slabs.h5', 'r') as new, \
         h5py.File(ROOT / 'field_recovery_evaluation_v1/control/diagnostic_slabs.h5', 'r') as control, \
         h5py.File(ROOT / 'field_recovery_evaluation_v1/repair/diagnostic_slabs.h5', 'r') as repair:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
        for case in range(2):
            keys = [f'heldout_{case}/native']+[f'heldout_{case}/rollout']*3
            for col, (source, key) in enumerate(zip([new, control, repair, new], keys)):
                if key in source:
                    axes[case, col].imshow(np.log10(1+source[key][:]), origin='lower', extent=(0, 24, 0, 24), vmin=0, vmax=2.5, cmap='magma')
                else:
                    axes[case, col].text(.1, .5, 'INVALID / UNAVAILABLE DRAW')
        for ax, title in zip(axes[0], ['Native TNG', 'NLL control', 'ES repair', 'Spatial diffusion']):
            ax.set_title(title)
        fig.suptitle('Fixed draw0, .1875 cMpc/h; no observed LG / no native fine-phase recovery claim')
        fig.savefig(OUT / 'density_comparison.png', dpi=140)
        plt.close(fig)
    history = json.loads((OUT / 'history.json').read_text())
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for key in ['continuous', 'categorical']:
        blocks = [history[i:i+100] for i in range(0, len(history), 100)]
        ax.plot([b[-1]['step'] for b in blocks], [np.mean([r[key] for r in b]) for b in blocks], label=key)
    ax.set(xlabel='Optimizer update', ylabel='100-update averaged denoising loss')
    ax.legend()
    fig.savefig(OUT / 'learning_curve.png', dpi=140)
    plt.close(fig)
    collections = {name: [r for r in json.loads((ROOT / 'field_recovery_evaluation_v1' / f'{name}_development.json').read_text())
                         if r['kind'] == 'fine'] for name in ['control', 'repair']}
    collections['diffusion'] = json.loads((OUT / 'development.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    keys = ['power', 'hot_connected_fraction', 'bulk_residual_rms_km_s', 'physical_sigma_rms_km_s']
    summary = {}
    for name, records in collections.items():
        available = [r for r in records if 'ratios' in r]
        summary[name] = dict(fixed_draws=len(records), invalid_or_unavailable=len(records)-len(available),
                             passed_draws=sum(r['development_pass'] for r in records))
        if not available:
            continue
        for ax, key in zip(axes.flat, keys):
            values = np.array([r['ratios'][key] for r in available])
            summary[name][key] = dict(mean=values.mean(0).tolist(), individual=values.tolist())
            ax.plot(np.atleast_1d(values.mean(0)), marker='.', label=f'{name} ({len(available)}/{len(records)} valid)')
    for ax, key in zip(axes.flat, keys):
        ax.axhline(1, color='black', linestyle=':')
        ax.set_title(key+' / native')
        ax.legend(fontsize=7)
    fig.suptitle('Fixed fine draws; invalid draws count as failures, not replacements')
    fig.savefig(OUT / 'physical_comparison.png', dpi=140)
    plt.close(fig)
    dump('comparison.json', summary)


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('Slurm allocation and submitted source required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    RESULT.update(status='TESTING', job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'])
    dump('request.json', dict(config=CFG, sizing=SIZING, **RESULT))
    torch.set_num_threads(4)
    precision = configure_precision(True)
    dump('precision.json', precision)
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_split_flow', 'test_cf4_spatial_diffusion'])
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    dump('tests.json', dict(run=tested.testsRun, failures=len(tested.failures), errors=len(tested.errors), passed=tested.wasSuccessful()))
    if not tested.wasSuccessful():
        raise RuntimeError('focused regression failure before training')
    if not torch.cuda.is_available():
        raise RuntimeError('allocated CUDA GPU required, no CPU fallback')
    event('PREPARING', device=torch.cuda.get_device_name(0))
    slab, location, spread = prepare()
    model = learn(slab, location, spread)
    del slab
    event('EVALUATING', completed_updates=RESULT['completed_updates'])
    final, rows = evaluate(model, location, spread)
    plots()
    passed = sum(row['development_pass'] for row in final)
    status = ('INCONCLUSIVE_BUDGET' if not RESULT['training_complete'] else
              'DEVELOPMENT_FINE_PRIOR_CANDIDATE' if passed == len(final) == 8 else 'NO_GO_DIFFUSION_MORPHOLOGY')
    RESULT.update(status=status, passed_draws=passed, total_fine_draws=len(final),
         invalid_final_draws=sum(r.get('invalid', False) for r in final), memory=memory(),
         elapsed_seconds=time.time()-START,
         limitations='Only8 fine draws comparable to old fine subset; old environment8 not trained/tested here. No LG conditioning or IC. No automatic follow-up.')
    dump('result.json', RESULT)
    event(status, completed_updates=RESULT['completed_updates'], passed_draws=passed, total_fine_draws=len(final))


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if CREATED:
            RESULT.update(status='INCOMPLETE_TIME_CAP' if isinstance(exc, TimeoutError) else 'EXECUTION_FAILED',
                          error=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.time()-START)
            dump('result.json', RESULT)
            event(RESULT['status'], error=str(exc), completed_updates=RESULT['completed_updates'])
        raise
