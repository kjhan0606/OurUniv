"""One bounded multiscale conditional flow fit and native mock field evaluation."""
import json
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch
from scipy.ndimage import label

from cf4_continuous_matter import restrict, check_realizable
from cf4_split_moments import NODES, state, encode_tree, decode, merge_octets, roundtrip
from cf4_conditional_split_flow import condition, ConditionalSplitFlow
from cf4_bundle_c_continuous import read_periodic_patch
from cf4_spatial_copula import expand

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUT = ROOT / os.environ.get('CF4_FLOW_OUTPUT_NAME', 'conditional_flow_v1')
CONFIG = Path(os.environ.get('CF4_FLOW_CONFIG', 'config/cf4_bundle_c_flow_pilot_v1.json'))
CREATED_OUTPUT = False


def dump(name, value):
    with (OUT / name).open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def metrics(field, dx, parent, ratio):
    mass = field[0]
    n = mass.shape[-1]
    delta = mass/mass.mean()-1
    ft = np.fft.rfftn(delta, norm='ortho')
    k = np.fft.fftfreq(n, d=dx)*2*np.pi
    kz = np.fft.rfftfreq(n, d=dx)*2*np.pi
    wave = np.sqrt(k[:, None, None]**2+k[None, :, None]**2+kz[None, None, :]**2)
    edges = np.geomspace(2*np.pi/(n*dx), np.pi/dx, 9)
    mult = np.ones_like(wave)*2
    mult[:, :, [0, -1]] = 1
    power = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        select = (wave >= lo) & (wave < hi)
        power.append(float(np.sum(abs(ft[select])**2*mult[select])/mult[select].sum()))
    hot = mass > np.quantile(mass, .99)
    labels, _ = label(hot)
    sizes = np.bincount(labels.ravel())[1:]
    v, variance = state(field)
    pv, _ = state(parent)
    velocity_rms = np.sqrt(np.sum(mass*(v-expand(pv, ratio))**2, axis=(1, 2, 3))/mass.sum())
    sigma_rms = np.sqrt(np.sum(mass*variance, axis=(1, 2, 3))/mass.sum())
    gradients = []
    for axis in range(3):
        d = np.moveaxis(np.diff(delta, axis=axis), axis, 0)**2
        boundary = (np.arange(1, n) % ratio) == 0
        gradients.append(float(d[boundary].mean()/d[~boundary].mean()))
    error = np.max(abs(restrict(field, ratio)-parent), axis=(1, 2, 3))/np.maximum(np.max(abs(parent), axis=(1, 2, 3)), 1)
    return dict(power=power, k_edges=edges.tolist(),
        top_mass_fraction=float(mass[hot].sum()/mass.sum()),
        hot_connected_fraction=float(sizes.max(initial=0)/max(hot.sum(), 1)),
        bulk_residual_rms_km_s=velocity_rms.tolist(), physical_sigma_rms_km_s=sigma_rms.tolist(),
        parent_boundary_gradient_ratio=gradients, coarse_error=error.tolist())


def compare(measured, reference):
    ratios = {k: (np.asarray(measured[k])/reference[k]).tolist() for k in
              ('power', 'top_mass_fraction', 'hot_connected_fraction', 'bulk_residual_rms_km_s',
               'physical_sigma_rms_km_s', 'parent_boundary_gradient_ratio')}
    tested = np.r_[ratios['power'][4:], ratios['top_mass_fraction'], ratios['hot_connected_fraction'],
                   ratios['bulk_residual_rms_km_s'], ratios['physical_sigma_rms_km_s'], ratios['parent_boundary_gradient_ratio']]
    return dict(ratios=ratios, development_pass=bool(np.all((tested >= .5) & (tested <= 2))), measured=measured)


def prepare(cfg):
    rng = np.random.default_rng(cfg['data_seed'])
    origins = [([int(rng.integers(0, 17)), int(rng.integers(0, 50)), int(rng.integers(0, 50))]) for _ in range(cfg['training_cubes'])]
    held = cfg['heldout_origins_1p5_cells']
    # Training x faces<=48; test x faces49.5..73.5, source is periodic75.
    # Check exact cube overlap, never filesystem/storage behavior.
    from cf4_bundle_c_spatial_model import overlaps
    if any(overlaps(np.array(t)*1.5, np.array(h)*1.5) for t in origins for h in held):
        raise ValueError('training and evaluation source voxels overlap')
    if overlaps(np.array(held[0])*1.5, np.array(held[1])*1.5):
        raise ValueError('evaluation source cubes overlap')
    dump('request.json', dict(config=cfg, train_origins_1p5_cells=origins,
         heldout_origins_1p5_cells=held, source_commit=os.environ['EXPECTED_COMMIT'],
         split_status='DISJOINT_WITHIN_THIS_FIT_HISTORICALLY_USED_SINGLE_BOX_DEVELOPMENT_ONLY',
         limit='Earlier finite-bank/model work used this simulation; no fresh-independent validation claim.'))
    data = [[] for _ in range(6)]
    report = []
    sums, squares, count = np.zeros(7), np.zeros(7), np.zeros(7)
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        for case, origin in enumerate(origins+held):
            field = read_periodic_patch(source['fine'], np.array(origin)*8, 128)
            errors = []
            for level in range(6):
                errors.append(roundtrip(field, report=True))
                if case < len(origins):
                    records = encode_tree(field)
                    root = records[0][2]
                    for node, (z, mask, parent) in enumerate(records):
                        context, valid = condition(parent, root, .1875*2**level, node)
                        choose = rng.choice(z[0].size, min(4096, z[0].size), replace=False)
                        selected = z.reshape(7, -1)[:, choose]
                        active = mask.reshape(7, -1)[:, choose] == 2
                        sums += (selected*active).sum(1)
                        squares += (selected*selected*active).sum(1)
                        count += active.sum(1)
                        data[level].append((z.astype(np.float32), mask, context, valid))
                field = restrict(field, 2)
            report.append(dict(case=case, origin=origin, training=case < len(origins), moment_roundtrip_errors=errors))
            print(f'Native six-scale roundtrip and data preparation {case+1}/{len(origins)+len(held)}', flush=True)
    if np.any(count == 0):
        raise ValueError('missing continuous coordinate support in training')
    location = sums/count
    spread = np.sqrt(np.maximum(squares/count-location**2, 0))
    if np.any(spread <= 0):
        raise ValueError('constant continuous training coordinate; no invented spread')
    dump('representation.json', dict(status='NATIVE_SEVEN_MOMENT_ROUNDTRIP_PASS', cases=report,
        coordinate_location=location.tolist(), coordinate_spread=spread.tolist(),
        note='Binary tree instead of interior-only octet sphere; explicit atoms. Scales and units do not rescale physical density power.'))
    return data, location, spread


def tensors(record, rng=None, full_context=False):
    z, mask, context, valid = record
    n = z.shape[-1]
    side = min(n, 24)
    starts = [0]*3 if rng is None else rng.integers(0, n-side+1, 3)
    # Consume the same RNG draws as v1, preserving record/step selection, but
    # do not substitute artificial crop faces for the evaluated field faces.
    if full_context:
        starts, side = [0]*3, n
    slices = (slice(None),)+tuple(slice(int(s), int(s)+side) for s in starts)
    return tuple(torch.from_numpy(np.ascontiguousarray(x[slices]))[None].to('cuda')
                 for x in (z, mask.astype(np.int64), context, valid))


@torch.no_grad()
def refine(model, root, child_dx, context_intervention=False):
    tree = {(0, 8): root}
    for node, (lo, hi) in enumerate(NODES):
        parent = tree[(lo, hi)]
        ctx, valid = condition(parent, root, child_dx, node)
        if context_intervention:
            # Wrong root-neighbour context only; actual parent/constraints fixed.
            # This intervention is NOT a draw from the fitted physical target.
            ctx[7:14] = np.flip(ctx[7:14], axis=1).copy()
        z, mask = model.sample(torch.from_numpy(ctx)[None].cuda(), torch.from_numpy(valid)[None].cuda())
        middle = (lo+hi)//2
        tree[(lo, middle)], tree[(middle, hi)] = decode(z[0].cpu().numpy().astype(np.float64),
            mask[0].cpu().numpy(), parent)
    result = merge_octets(np.array([tree[(i, i+1)] for i in range(8)]))
    check_realizable(result)
    error = np.max(abs(restrict(result, 2)-root))/max(np.max(abs(root)), 1)
    if error > 1e-9:
        raise ValueError('generated split conservation failed')
    return result


def evaluate(model, cfg):
    records = []
    model.eval()
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source, h5py.File(OUT / 'mock_fields.h5', 'x') as output:
        output.attrs.update(status='INCOMPLETE', limits='Native coarse oracle conditions; not CF4/LG field, not joint halo model, not independent-box validation.')
        for case, origin in enumerate(cfg['heldout_origins_1p5_cells']):
            native = read_periodic_patch(source['fine'], np.array(origin)*8, 128)
            group = output.create_group(f'case_{case}')
            group.create_dataset('native', data=native, compression='gzip', compression_opts=1)
            for kind, ratio, child_scales in [('environment', 8, [6., 3., 1.5]), ('fine', 8, [.75, .375, .1875])]:
                reference = restrict(native, 8) if kind == 'environment' else native
                parent = restrict(reference, ratio)
                reference_stats = metrics(reference, child_scales[-1], parent, ratio)
                for draw in range(cfg['draws_per_case']):
                    seed = cfg['sample_seed']+case*100+draw+(1000 if kind == 'environment' else 0)
                    torch.manual_seed(seed)
                    generated = parent
                    for dx in child_scales:
                        generated = refine(model, generated, dx)
                    stats = metrics(generated, child_scales[-1], parent, ratio)
                    if max(stats['coarse_error']) > 1e-8:
                        raise ValueError('multiscale seven-moment conservation failed')
                    result = compare(stats, reference_stats)
                    result.update(case=case, kind=kind, draw=draw, reference=reference_stats)
                    group.create_dataset(f'{kind}_{draw}', data=generated, compression='gzip', compression_opts=1)
                    if draw == 0:
                        torch.manual_seed(seed)
                        altered = parent
                        for dx in child_scales:
                            altered = refine(model, altered, dx, context_intervention=True)
                        result['root_context_permutation_mass_L1_fraction'] = float(abs(altered[0]-generated[0]).sum()/generated[0].sum())
                        result['context_intervention_limit'] = 'Same RNG, wrong root context, real conservation retained; sensitivity only, not proof of correct information use.'
                        del altered
                    records.append(result)
                    print(json.dumps(dict(case=case, kind=kind, draw=draw, development_pass=result['development_pass'], power_ratios=result['ratios']['power'])), flush=True)
                    del generated
        output.attrs['status'] = 'DEVELOPMENT_MOCK_FIELDS_COMPLETE_NOT_CF4_LG_POSTERIOR'
    return records


def run():
    global CREATED_OUTPUT
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Slurm required')
    OUT.mkdir(exist_ok=False)
    CREATED_OUTPUT = True
    started = time.monotonic()
    cfg = json.loads(CONFIG.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('allocated GPU unavailable; never train on login/CPU fallback')
    torch.ones(1, device='cuda').sum().item()
    torch.set_num_threads(2)
    torch.manual_seed(cfg['training_seed'])
    rng = np.random.default_rng(cfg['training_seed'])
    data, location, spread = prepare(cfg)
    model = ConditionalSplitFlow(location, spread).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    training_started = time.monotonic()
    dump('status.json', dict(status='TRAINING', job_id=os.environ['SLURM_JOB_ID'],
        gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
        preparation_seconds=training_started-started))
    history = []
    with (OUT / 'training.jsonl').open('x', buffering=1) as log:
        for step in range(1, cfg['steps']+1):
            level = (step-1) % 6  # Equal scale exposure, not fine-voxel dominance.
            record = data[level][int(rng.integers(len(data[level])))]
            z, mask, context, valid = tensors(record, rng, full_context=cfg.get('full_context_training', False))
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prob(z, mask, context, valid).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite normalized conditional likelihood')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            if not torch.isfinite(grad):
                raise ValueError('nonfinite training gradient')
            optimizer.step()
            if step == 1 or step % 100 == 0:
                event = dict(step=step, child_dx=.1875*2**level, nll_per_parent=float(loss.detach()),
                    elapsed_seconds=time.monotonic()-started, gpu_peak_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(event)+'\n')
                history.append(event)
                print(json.dumps(event), flush=True)
                if event['gpu_peak_bytes'] > cfg['gpu_peak_limit_GiB']*2**30:
                    raise RuntimeError('GPU allocation exceeded pilot memory budget')
            if time.monotonic()-started > cfg['training_deadline_seconds']:
                break
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, config=cfg,
        coordinate_space='mixed binary split coordinates, not physical-field Lebesgue density',
        source_commit=os.environ['EXPECTED_COMMIT']), OUT / 'checkpoint.pt')
    del data, optimizer, z, mask, context, valid, loss
    torch.cuda.empty_cache()
    dump('status.json', dict(status='EVALUATING', steps=step, training_seconds=time.monotonic()-training_started))
    comparisons = evaluate(model, cfg)
    passed = all(r['development_pass'] for r in comparisons)
    report = dict(status='DEVELOPMENT_DIAGNOSTIC_PASS_NOT_ACTUAL_POSTERIOR' if passed else 'NO_GO_CONDITIONAL_FLOW_DEVELOPMENT',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'], steps=step,
        full_step_budget_completed=step == cfg['steps'], gpu=torch.cuda.get_device_name(),
        gpu_peak_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started,
        comparisons=comparisons, history=history,
        limits=['All conditioning is native coarse truth, not actual CF4/count/LG observations.',
                'Single historically used TNG box; new fit train/test cubes disjoint but not independent confirmation.',
                'One six-block masked affine flow with explicit boundary laws and per-direction variances. No model/seed search.',
                'Global384-domain normalized evaluation, q_S member readout and actual1.5 posterior remain absent.',
                'Four draws per case do not calibrate posterior uncertainty or prove conditional phase accuracy.',
                'A gate failure stops adoption; no automatic longer training or amplitude repair.'],
        full_context_training=cfg.get('full_context_training', False))
    dump('result.json', report)
    dump('status.json', {k: v for k, v in report.items() if k not in ('comparisons', 'history')})
    print(report['status'], flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if CREATED_OUTPUT:
            dump('failure.json', dict(status='FAILED_NO_AUTOMATIC_RETRY', error=repr(exc), traceback=traceback.format_exc()))
        raise
