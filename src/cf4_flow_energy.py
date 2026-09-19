"""Detached-trace energy-score gradients for the existing mixed split law.

No differentiable decoder, straight-through masks, amplitude correction or halo
readout. All ancestor/mask/continuous scores are included at fixed trace values.
"""
from itertools import product

import numpy as np
import torch

from cf4_split_moments import NODES, decode, merge_octets
from cf4_conditional_split_flow import condition
from cf4_continuous_matter import restrict, check_realizable


def device_record(z, mask, parent, root, dx, node, device):
    context, valid = condition(parent, root, dx, node)
    arrays = (z.astype(np.float32), mask.astype(np.int64), context, valid)
    return tuple(torch.from_numpy(np.ascontiguousarray(x))[None].to(device)
                 for x in arrays)


@torch.no_grad()
def sample_trace(model, root, scales, check=lambda: None):
    """Return field and immutable CPU traces; parameter-free decoder is FP64."""
    device = next(model.parameters()).device
    records = []
    for dx in scales:
        tree = {(0, 8): root}
        for node, (lo, hi) in enumerate(NODES):
            check()
            parent = tree[lo, hi]
            context, valid = condition(parent, root, dx, node)
            z, mask = model.sample(torch.from_numpy(context)[None].to(device),
                                  torch.from_numpy(valid)[None].to(device))
            z = z[0].cpu().numpy().copy()
            mask = mask[0].cpu().numpy().astype(np.uint8)
            records.append((z, mask, parent, root, dx, node))
            middle = (lo+hi)//2
            tree[lo, middle], tree[middle, hi] = decode(z.astype(np.float64), mask, parent)
        field = merge_octets(np.array([tree[i, i+1] for i in range(8)]))
        check_realizable(field)
        error = np.max(abs(restrict(field, 2)-root), axis=(1, 2, 3))
        error /= np.maximum(np.max(abs(root), axis=(1, 2, 3)), 1)
        if error.max() > 1e-8:
            raise ValueError('generated trace violates seven-moment conservation')
        root = field
    return root, records


def trace_backward(model, trace, coefficient, check=lambda: None):
    """Accumulate coefficient*grad(logq), SUM not mean, one graph at a time."""
    if not np.isfinite(coefficient):
        raise ValueError('nonfinite detached score coefficient')
    device = next(model.parameters()).device
    total = 0.
    for record in trace:
        check()
        logp = model.log_prob(*device_record(*record, device))
        if not torch.isfinite(logp).all():
            raise ValueError('nonfinite generated-trace likelihood')
        value = logp.sum()
        (value*float(coefficient)).backward()
        total += float(value.detach())
    return total


def energy_coefficients(first, second, target, baseline=0.):
    """ES value and its TWO detached log-score multipliers (past baseline only)."""
    a1 = np.linalg.norm(first-target)
    a2 = np.linalg.norm(second-target)
    b = np.linalg.norm(first-second)
    return float((a1+a2-b)/2), np.array([a1-b-baseline, a2-b-baseline])/2


def feature_groups(field, dx, parent, ratio):
    # Reuse existing physical/spectral definitions, NOT a new power estimator.
    from cf4_bundle_c_flow_pilot import metrics
    n = field.shape[-1]
    density = np.log1p(field[0]/field[0].mean())
    blocks = [density[tuple(slice(c-4, c+4) for c in center)].ravel()
              for center in product((n//4, 3*n//4), repeat=3)]
    measured = metrics(field, dx, parent, ratio)
    power = np.asarray(measured['power'])
    if not np.all(np.isfinite(power) & (power > 0)):
        raise ValueError('log-power feature has nonpositive/nonfinite native support')
    return [np.concatenate(blocks), np.log(power),
            np.r_[measured['bulk_residual_rms_km_s'], measured['physical_sigma_rms_km_s']]]


def normalize_groups(groups, scales):
    included = [x/s for x, s in zip(groups, scales) if s > 0]
    if not included:
        raise ValueError('all training feature groups unavailable')
    result = np.concatenate(included)/np.sqrt(len(included))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite normalized structural features')
    return result
