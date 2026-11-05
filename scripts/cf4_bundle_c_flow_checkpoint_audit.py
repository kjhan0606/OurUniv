"""Frozen checkpoint: likelihood/inverse and native crop-context diagnosis.

No optimizer or new fit. Paired same-cell log likelihood at full, cropped and
halo-padded contexts isolates padding dependence without changing native data.
"""
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from cf4_split_moments import encode_tree
from cf4_conditional_split_flow import ConditionalSplitFlow, condition, configure_precision
from cf4_bundle_c_continuous import read_periodic_patch
from cf4_bundle_c_flow_pilot import ROOT, refine, metrics
from cf4_continuous_matter import restrict

SOURCE = ROOT / 'conditional_flow_v1'
OUT = ROOT / os.environ.get('CF4_FLOW_AUDIT_OUTPUT_NAME', 'flow_checkpoint_audit_v1')


def device_tuple(record):
    z, mask, context, valid = record
    return tuple(torch.from_numpy(np.ascontiguousarray(x))[None].cuda() for x in
                 (z.astype(np.float32), mask.astype(np.int64), context, valid))


def extract(record, lo, hi):
    return tuple(x[:, lo:hi, lo:hi, lo:hi] for x in record)


@torch.no_grad()
def run():
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU required')
    OUT.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    precision = configure_precision(os.environ.get('CF4_FLOW_STRICT_FP32') == '1')
    print(json.dumps(dict(precision=precision)), flush=True)
    # This is our own checkpoint from336268, not an untrusted external pickle.
    checkpoint = torch.load(SOURCE / 'checkpoint.pt', map_location='cpu', weights_only=False)
    state = checkpoint['model']
    model = ConditionalSplitFlow(state['location'].flatten().numpy(), state['spread'].flatten().numpy()).cuda()
    model.load_state_dict(state)
    model.eval()
    request = json.loads((SOURCE / 'request.json').read_text())
    origins = [('training', request['train_origins_1p5_cells'][0]),
               ('heldout', request['heldout_origins_1p5_cells'][0])]
    reports, images = [], []
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        for split, origin in origins:
            native = read_periodic_patch(source['fine'], np.array(origin)*8, 128)
            records = encode_tree(native)
            root = records[0][2]
            for node in (0, 3):
                z, mask, parent = records[node]
                context, valid = condition(parent, root, .1875, node)
                record = (z, mask, context, valid)
                full = device_tuple(record)
                full_lp = model.log_prob(*full)[0].cpu().numpy()
                # Native target parent indices20:44 in all three axes.
                crop_lp = model.log_prob(*device_tuple(extract(record, 20, 44)))[0].cpu().numpy()
                # Six two-convolution couplings have at most radius12;
                # 12 native context cells recover the same interior score.
                padded_lp = model.log_prob(*device_tuple(extract(record, 8, 56)))[0, 12:36, 12:36, 12:36].cpu().numpy()
                target_lp = full_lp[20:44, 20:44, 20:44]
                active = full[1] == 2
                original = ((full[0]-model.location)/model.spread)*active
                x, jac = original.clone(), torch.zeros_like(original[:, 0])
                for layer in reversed(model.layers):
                    x, det = layer(x, full[2], full[1], inverse=True)
                    jac += det
                latent = x.clone()
                for layer in model.layers:
                    x, det = layer(x, full[2], full[1])
                    jac += det
                inverse_error = float((abs(x-original)/torch.maximum(abs(original), torch.ones_like(original))).max())
                jac_error = float(abs(jac).max())
                latent_stats = []
                for k in range(7):
                    values = latent[:, k][active[:, k]]
                    latent_stats.append(dict(mean=float(values.mean()), std=float(values.std(unbiased=False)), count=values.numel()))
                wrong = full[2].clone()
                wrong[:, 7:14] = torch.flip(wrong[:, 7:14], dims=(2,))
                wrong_lp = model.log_prob(full[0], full[1], wrong, full[3])
                result = dict(split=split, origin=origin, node=node,
                    native_NLL_per_parent=float(-full_lp.mean()),
                    permuted_root_NLL_per_parent=float(-wrong_lp.mean()),
                    cropped_same_cells_mean_abs_logp_change=float(abs(crop_lp-target_lp).mean()),
                    cropped_same_cells_signed_logp_change=float((crop_lp-target_lp).mean()),
                    padded_same_cells_max_abs_logp_change=float(abs(padded_lp-target_lp).max()),
                    flow_inverse_relative_error=inverse_error, forward_inverse_logdet_error=jac_error,
                    inverse_latent_statistics=latent_stats)
                reports.append(result)
                print(json.dumps(result), flush=True)
                del full, x, latent, original, wrong, wrong_lp
            # One diagnostic truth-parent finest refinement, not a new family,
            # not a replacement posterior ensemble and not chosen by outcome.
            torch.manual_seed(77631)
            one_step = refine(model, restrict(native, 2), .1875)
            truth_stats = metrics(native, .1875, restrict(native, 2), 2)
            one_stats = metrics(one_step, .1875, restrict(native, 2), 2)
            reports.append(dict(split=split, diagnostic='one_step_native_parent',
                power_ratio=(np.array(one_stats['power'])/truth_stats['power']).tolist(),
                coarse_error=one_stats['coarse_error']))
            if split == 'heldout':
                with h5py.File(SOURCE / 'mock_fields.h5', 'r') as saved:
                    rolled = saved['case_0/fine_0'][0, :, :, 60:68].mean(axis=2)
                images = [native[0, :, :, 60:68].mean(axis=2), one_step[0, :, :, 60:68].mean(axis=2), rolled]
                images = np.array(images)/native[0].mean()
            del records, native, one_step
    paired = [r for r in reports if 'node' in r]
    justified = all(r['flow_inverse_relative_error'] < 1e-4
                    and r['forward_inverse_logdet_error'] < 1e-3
                    and r['padded_same_cells_max_abs_logp_change'] < 1e-3
                    and r['cropped_same_cells_mean_abs_logp_change'] > 1e-3 for r in paired)
    report = dict(status='CONTEXT_MISMATCH_CONFIRMED_CORRECTION_JUSTIFIED' if justified else 'STOP_REVIEW_NO_AUTOMATIC_REFIT',
        source_checkpoint_commit=checkpoint['source_commit'], source_steps=checkpoint['step'],
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'], comparisons=reports,
        elapsed_seconds=time.monotonic()-start, precision=precision,
        limits=['No optimizer/fit in this audit. Existing checkpoint and used source data only.',
                'Crop/full differences test deterministic likelihood-context mismatch, not sole cause of morphology failure.',
                'Train/held NLL and latent moments do not distinguish insufficient optimization from model capacity by themselves.',
                'Single teacher-parent draw is a diagnostic, not a calibrated conditional ensemble.'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, values, title in zip(axes, images, ['Native TNG', 'One-step truth-parent', 'Saved three-step draw0']):
        ax.imshow(np.log10(1+np.maximum(values, 0)), origin='lower', vmin=0, vmax=2.5, cmap='magma', extent=(0, 24, 0, 24))
        ax.set_title(title)
    fig.suptitle('Frozen model diagnosis, 0.1875 grid; NOT observed LG')
    fig.savefig(OUT / 'comparison.png', dpi=140)
    plt.close(fig)
    with (OUT / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(report['status'], flush=True)
    if not justified:
        raise SystemExit(2)


if __name__ == '__main__':
    run()
