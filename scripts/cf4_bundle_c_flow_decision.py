"""Frozen v1/v2 fit, scale and morphology analysis; never train or tune."""
import json
import os
import time
import traceback

import h5py
import numpy as np
import torch

from cf4_bundle_c_flow_pilot import ROOT, refine, metrics, compare
from cf4_bundle_c_flow_checkpoint_audit import device_tuple
from cf4_bundle_c_continuous import read_periodic_patch
from cf4_continuous_matter import restrict
from cf4_conditional_split_flow import ConditionalSplitFlow, condition, configure_precision
from cf4_split_moments import encode_tree

OUT = ROOT / 'flow_decision_v1'
SOURCES = {'v1': ROOT / 'conditional_flow_v1',
           'v2': ROOT / 'conditional_flow_v2_full_context'}
CREATED = False


def dump(name, value):
    with (OUT / name).open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def score(field, native, dx, base, ratio):
    measured = metrics(field, dx, base, ratio)
    reference = metrics(native, dx, base, ratio)
    result = compare(measured, reference)
    if max(measured['coarse_error']) > 1e-8:
        raise ValueError('diagnostic generation failed seven-moment conservation')
    power = np.asarray(result['ratios']['power'][4:])
    if not np.all(np.isfinite(power) & (power > 0)):
        raise ValueError('nonfinite or nonpositive power ratio')
    result['high_band_log_rms'] = float(np.sqrt(np.mean(np.log(power)**2)))
    return result


@torch.no_grad()
def likelihood(model, native, dx):
    tree = encode_tree(native)
    nodes = []
    for node, (z, mask, parent) in enumerate(tree):
        context, valid = condition(parent, tree[0][2], dx, node)
        data = device_tuple((z, mask, context, valid))
        logp = model.log_prob(*data)
        _, mask_logp = model.masks(data[2], data[3], data[1])
        if not torch.isfinite(logp).all():
            raise ValueError('nonfinite fixed-input likelihood')
        nodes.append(dict(node=node, nll=float(-logp.mean()),
                          mask_nll=float(-mask_logp.mean()),
                          continuous_nll=float(-(logp-mask_logp).mean())))
    # Each node has the same parent spatial grid; sum node NLL means is
    # conditional split-coordinate NLL per root octet, not physical density.
    return dict(nodes=nodes, nll_per_octet=sum(n['nll'] for n in nodes),
                mask_nll_per_octet=sum(n['mask_nll'] for n in nodes))


def plots(slabs, nll, draws):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = list(slabs)
    titles = ['Native TNG', 'v1 rollout', 'v2 rollout', 'v1 true-parent step', 'v2 true-parent step']
    fig, axes = plt.subplots(len(cases), 5, figsize=(15, 3*len(cases)), layout='constrained')
    for row, key in enumerate(cases):
        for col, name in enumerate(['native', 'v1_rollout', 'v2_rollout', 'v1_teacher', 'v2_teacher']):
            ax = axes[row, col]
            ax.imshow(np.log10(1+slabs[key][name]), origin='lower', extent=(0, 24, 0, 24),
                      vmin=0, vmax=2.5, cmap='magma')
            if row == 0:
                ax.set_title(titles[col])
            if col == 0:
                ax.set_ylabel(key)
    fig.suptitle('Both models strict FP32, fixed draw0; 0.1875 grid / 1.5-deep slab; NOT observed LG')
    fig.savefig(OUT / 'density_comparison.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for row, split in enumerate(['training', 'heldout']):
        for model in SOURCES:
            for case in range(2):
                selected = [r for r in nll if r['model'] == model and r['split'] == split and r['case'] == case]
                selected.sort(key=lambda r: r['dx'])
                axes[row, 0].plot([r['dx'] for r in selected], [r['nll_per_octet'] for r in selected],
                                  marker='.', label=f'{model} case{case}')
            for mode in ['rollout', 'teacher']:
                means = []
                for dx in [.75, .375, .1875]:
                    selected = [r['high_band_log_rms'] for r in draws if r['model'] == model
                                and r['split'] == split and r['mode'] == mode and r['dx'] == dx]
                    means.append(float(np.mean(selected)))
                axes[row, 1].plot([.75, .375, .1875], means, marker='.', label=f'{model} {mode}')
        axes[row, 0].set_title(f'{split}: fixed native conditional NLL')
        axes[row, 1].set_title(f'{split}: high-band log-ratio RMS (lower better)')
        for ax in axes[row]:
            ax.set_xscale('log', base=2)
            ax.set_xlabel('child dx [cMpc/h]')
            ax.legend(fontsize=8)
    fig.savefig(OUT / 'likelihood_and_scale_loss.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout='constrained')
    for case in range(2):
        axes[case, 0].imshow(np.log10(1+slabs[f'heldout_{case}']['native']),
                             origin='lower', vmin=0, vmax=2.5, cmap='magma', extent=(0, 24, 0, 24))
        for col, (name, source) in enumerate(SOURCES.items(), 1):
            with h5py.File(source / 'mock_fields.h5', 'r') as f:
                native_mean = f[f'case_{case}/native'][0].mean()
                projection = f[f'case_{case}/fine_0'][0, :, :, 60:68].mean(2)/native_mean
            axes[case, col].imshow(np.log10(1+projection), origin='lower', vmin=0, vmax=2.5,
                                   cmap='magma', extent=(0, 24, 0, 24))
        axes[case, 0].set_ylabel(f'Original heldout {case}')
    for ax, title in zip(axes[0], ['Native TNG', 'Saved v1 TF32 draw0', 'Saved v2 strict draw0']):
        ax.set_title(title)
    fig.suptitle('Original saved pilot outputs, same projection/color scale; NOT observed LG')
    fig.savefig(OUT / 'saved_before_after.png', dpi=140)
    plt.close(fig)


@torch.no_grad()
def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    start = time.monotonic()
    torch.set_num_threads(2)
    precision = configure_precision(True)
    requests = {k: json.loads((v / 'request.json').read_text()) for k, v in SOURCES.items()}
    for key in ['train_origins_1p5_cells', 'heldout_origins_1p5_cells']:
        if requests['v1'][key] != requests['v2'][key]:
            raise ValueError('v1/v2 native split mismatch')
    models, provenance = {}, {}
    for name, source in SOURCES.items():
        checkpoint = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=False)
        state = checkpoint['model']
        model = ConditionalSplitFlow(state['location'].flatten().numpy(), state['spread'].flatten().numpy())
        model.load_state_dict(state)
        models[name] = model.cuda().eval()
        provenance[name] = dict(path=str(source), steps=checkpoint['step'], source_commit=checkpoint['source_commit'])
        del checkpoint
    historical = {k: json.loads((v / 'result.json').read_text())['comparisons'] for k, v in SOURCES.items()}
    request = dict(job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'],
        models=provenance, precision=precision, draws_per_case=2, cases='First two original train and both original heldout',
        seeds='99803+100*case+draw; training adds10000; environment adds1000',
        limits=['No optimizer, no fitting, no new simulation or original raw source pass.',
                'Historically reused one-box development; two train cases may overlap.',
                'Native train/heldout NLL differs with physical field; gap alone is not proof of overfitting.',
                'Only final checkpoints exist: no fixed-validation convergence curve or optimization-vs-capacity proof.',
                'v1/v2 differ in context, scored-cell exposure and training precision, not one isolated cause.',
                'Two draws per case are paired diagnostics, not an uncertainty calibration or revised acceptance gate.',
                'All coarse conditions native truth; no observed LG, q_S, global law, or actual CF4 fine posterior.'])
    dump('request.json', request)
    nll, draws, slabs = [], [], {}
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        for split, key in [('training', 'train_origins_1p5_cells'), ('heldout', 'heldout_origins_1p5_cells')]:
            for case, origin in enumerate(requests['v1'][key][:2]):
                native = read_periodic_patch(source['fine'], np.array(origin)*8, 128)
                levels = {i: restrict(native, 2**i) if i else native for i in range(7)}
                case_key = f'{split}_{case}'
                slabs[case_key] = {'native': native[0, :, :, 60:68].mean(2)/native[0].mean()}
                for name, model in models.items():
                    for level in range(6):
                        nll.append(dict(model=name, split=split, case=case, dx=.1875*2**level,
                                        **likelihood(model, levels[level], .1875*2**level)))
                    for draw in range(2):
                        seed = 99803+100*case+draw+(10000 if split == 'training' else 0)
                        for kind, scales, base_level in [('fine', [.75, .375, .1875], 3),
                                                         ('environment', [6., 3., 1.5], 6)]:
                            torch.manual_seed(seed+(1000 if kind == 'environment' else 0))
                            base = levels[base_level]
                            generated = base
                            for step, dx in enumerate(scales, 1):
                                before = torch.cuda.get_rng_state()
                                generated = refine(model, generated, dx)
                                after = torch.cuda.get_rng_state()
                                # Existing eight power bands are populated at N>=16.
                                # Environment N4/N8 stages are generated, not scored.
                                if kind == 'environment' and step != 3:
                                    continue
                                truth = levels[base_level-step]
                                torch.cuda.set_rng_state(before)
                                teacher = refine(model, levels[base_level-step+1], dx)
                                torch.cuda.set_rng_state(after)
                                if step == 1 and not np.array_equal(generated, teacher):
                                    raise ValueError('same-parent paired RNG control failed')
                                for mode, field in [('rollout', generated), ('teacher', teacher)]:
                                    result = dict(model=name, split=split, case=case, draw=draw,
                                        kind=kind, mode=mode, dx=dx, stage=step,
                                        **score(field, truth, dx, base, 2**step))
                                    draws.append(result)
                                    if kind == 'fine' and step == 3 and draw == 0:
                                        slabs[case_key][f'{name}_{mode}'] = field[0, :, :, 60:68].mean(2)/native[0].mean()
                                del teacher
                            del generated
                    print(json.dumps(dict(completed=case_key, model=name, elapsed_seconds=time.monotonic()-start)), flush=True)
                    dump('status.json', dict(status='ANALYSING', completed=case_key, model=name,
                         elapsed_seconds=time.monotonic()-start))
                del native, levels
    plots(slabs, nll, draws)
    with h5py.File(OUT / 'density_slabs.h5', 'x') as f:
        for key, fields in slabs.items():
            for name, values in fields.items():
                f.create_dataset(f'{key}/{name}', data=values, compression='gzip')
        f.attrs['limits'] = 'Fixed draw0 projections, no observed LG or saved full diagnostic fields.'
    result = dict(**request, status='ANALYSIS_COMPLETE_REQUIRES_DRIVER_DECISION',
                  historical_comparisons=historical, native_likelihood=nll, paired_generation=draws,
                  elapsed_seconds=time.monotonic()-start, gpu=torch.cuda.get_device_name(),
                  gpu_peak_bytes=torch.cuda.max_memory_allocated())
    dump('result.json', result)
    dump('status.json', {k: result[k] for k in ['status', 'job_id', 'elapsed_seconds', 'gpu', 'gpu_peak_bytes']})
    print(result['status'], flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        if CREATED:
            dump('failure.json', dict(status='FAILED_NO_AUTOMATIC_RETRY', traceback=traceback.format_exc()))
        raise
