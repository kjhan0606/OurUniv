"""Same checkpoint/input precision comparison; no fitting or gate relaxation."""
import json
import os
import time

import h5py
import numpy as np
import torch

from cf4_bundle_c_flow_checkpoint_audit import SOURCE, device_tuple
from cf4_bundle_c_flow_pilot import ROOT
from cf4_bundle_c_continuous import read_periodic_patch
from cf4_conditional_split_flow import ConditionalSplitFlow, condition
from cf4_split_moments import encode_tree


@torch.no_grad()
def roundtrip(model, record):
    z, mask, context, _ = record
    original = ((z-model.location)/model.spread)*(mask == 2)
    x = original.clone()
    jac = torch.zeros_like(x[:, 0])
    local_errors = []
    for layer in reversed(model.layers):
        y, det = layer(x, context, mask, inverse=True)
        recovered, back_det = layer(y, context, mask)
        local_errors.append(dict(
            inverse_relative_error=float((abs(recovered-x)/abs(x).clamp_min(1)).max()),
            logdet_error=float(abs(det+back_det).max())))
        x = y
        jac += det
    for layer in model.layers:
        x, det = layer(x, context, mask)
        jac += det
    return dict(flow_inverse_relative_error=float((abs(x-original)/abs(original).clamp_min(1)).max()),
                forward_inverse_logdet_error=float(abs(jac).max()),
                single_layer_errors_inverse_order=local_errors,
                finite=bool(torch.isfinite(x).all() and torch.isfinite(jac).all()))


def backend():
    return dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic)


@torch.no_grad()
def run():
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU required')
    out = ROOT / 'flow_precision_diagnosis_v1'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    initial = backend()
    checkpoint = torch.load(SOURCE / 'checkpoint.pt', map_location='cpu', weights_only=False)
    state = checkpoint['model']
    request = json.loads((SOURCE / 'request.json').read_text())
    reports = []
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        for split, origin in [('training', request['train_origins_1p5_cells'][0]),
                              ('heldout', request['heldout_origins_1p5_cells'][0])]:
            native = read_periodic_patch(source['fine'], np.array(origin)*8, 128)
            records = encode_tree(native)
            for node in (0, 3):
                z, mask, parent = records[node]
                context, valid = condition(parent, records[0][2], .1875, node)
                # Cast ONCE: FP64 controls use identical FP32-quantized inputs
                # and checkpoint weights, not a different native field.
                full = device_tuple((z, mask, context, valid))
                for mode in ('default_fp32', 'strict_fp32', 'strict_fp64'):
                    torch.backends.cudnn.allow_tf32 = initial['cudnn_allow_tf32'] if mode == 'default_fp32' else False
                    torch.backends.cuda.matmul.allow_tf32 = initial['matmul_allow_tf32'] if mode == 'default_fp32' else False
                    model = ConditionalSplitFlow(state['location'].flatten().numpy(), state['spread'].flatten().numpy())
                    model.load_state_dict(state)
                    dtype = torch.float64 if mode == 'strict_fp64' else torch.float32
                    model = model.to(device='cuda', dtype=dtype).eval()
                    data = tuple(x.to(dtype=dtype) if x.is_floating_point() else x for x in full)
                    report = dict(split=split, origin=origin, node=node, mode=mode,
                                  backend=backend(), **roundtrip(model, data))
                    report['original_numerical_gate_pass'] = bool(report['finite']
                        and report['flow_inverse_relative_error'] < 1e-4
                        and report['forward_inverse_logdet_error'] < 1e-3)
                    reports.append(report)
                    print(json.dumps(report), flush=True)
                    del model, data
                del full
            del native, records
    passed = {mode: all(r['original_numerical_gate_pass'] for r in reports if r['mode'] == mode)
              for mode in ('default_fp32', 'strict_fp32', 'strict_fp64')}
    if not passed['default_fp32'] and passed['strict_fp32'] and passed['strict_fp64']:
        status = 'REDUCED_PRECISION_BACKEND_CAUSE_SUPPORTED'
    elif passed['strict_fp64'] and not passed['strict_fp32']:
        status = 'FP32_NUMERICAL_SENSITIVITY_REQUIRES_REVIEW'
    else:
        status = 'REVIEW_REQUIRED_NO_AUTOMATIC_FIT'
    result = dict(status=status, mode_all_four_pass=passed, comparisons=reports,
                  initial_backend=initial, torch_version=torch.__version__,
                  cudnn_version=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                  source_checkpoint_commit=checkpoint['source_commit'], source_steps=checkpoint['step'],
                  source_commit=os.environ['EXPECTED_COMMIT'], job_id=os.environ['SLURM_JOB_ID'],
                  elapsed_seconds=time.monotonic()-start,
                  gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                  limits=['Frozen weights and same four native inputs, no optimizer or generated ensemble.',
                          'Numerical cause separation is not an explanation or cure of morphology failure.',
                          'No relaxed tolerances; no automatic training from this diagnostic.'])
    with (out / 'result.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(status, flush=True)


if __name__ == '__main__':
    run()
