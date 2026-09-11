"""Deterministic field-only member mass hypotheses, not a halo posterior."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cf4_split_moments import state
from cf4_spatial_diffusion import SYMMETRIES

ROLES = ('MW', 'M31', 'M33', 'remainder')


def features(moments, observer_center_cells=(68., 68., 68.), dx=.1875):
    """No labels/centers from the halo catalogue accepted by this interface."""
    n = moments.shape[-1]
    velocity, variance = state(moments)
    out = np.empty((10, n, n, n), np.float32)
    out[0] = np.log1p(moments[0]/(1e11*dx**3))
    out[1:4] = np.arcsinh(velocity/300)
    out[4:7] = np.log1p(np.sqrt(variance)/100)
    for a in range(3):
        shape = [1, 1, 1]
        shape[a] = n
        out[7+a] = ((np.arange(n)+.5-observer_center_cells[a])/(n/2)).reshape(shape)
    return out


def transform(value, symmetry, vector_features=False):
    """Transform CXYZ; signed permutations keep velocity/observer vectors aligned."""
    axes, signs = SYMMETRIES[symmetry]
    out = value.permute(0, *(a+1 for a in axes))
    flips = [a+1 for a, s in enumerate(signs) if s < 0]
    if flips:
        out = torch.flip(out, flips)
    if vector_features:
        out = out[[0]+[a+1 for a in axes]+[a+4 for a in axes]+[a+7 for a in axes]].clone()
        sign = out.new_tensor(signs)[:, None, None, None]
        out[1:4] *= sign
        out[7:10] *= sign
    return out.contiguous()


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv3d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                         nn.Conv3d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())


class MemberMassNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([Block(10, 16), Block(32, 32), Block(64, 64)])
        self.down = nn.ModuleList([nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.Conv3d(32, 64, 3, stride=2, padding=1)])
        self.up = nn.ModuleList([nn.Conv3d(64, 32, 3, padding=1), nn.Conv3d(32, 16, 3, padding=1)])
        self.dec = nn.ModuleList([Block(64, 32), Block(32, 16)])
        self.head = nn.Conv3d(16, 4, 1)

    def block(self, layer, x):
        return checkpoint(layer, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else layer(x)

    @torch.no_grad()
    def initialize_mass_fractions(self, fractions):
        """Constant TRAINING integrated fractions, not a spatial template."""
        pi = torch.as_tensor(fractions, device=self.head.bias.device, dtype=self.head.bias.dtype)
        if pi.shape != (4,) or not bool(torch.isfinite(pi).all() and (pi > 0).all()):
            raise ValueError('four positive training mass fractions required')
        pi = pi/pi.sum()
        self.head.weight.zero_()
        self.head.bias.copy_(pi.log())

    def logits(self, field):
        a = self.block(self.enc[0], field)
        b = self.block(self.enc[1], self.down[0](a))
        c = self.block(self.enc[2], self.down[1](b))
        x = self.block(self.dec[0], torch.cat([self.up[0](F.interpolate(c, size=b.shape[-3:], mode='nearest')), b], 1))
        x = self.block(self.dec[1], torch.cat([self.up[1](F.interpolate(x, size=a.shape[-3:], mode='nearest')), a], 1))
        return self.head(x)

    def forward(self, field):
        return self.logits(field).softmax(1)


def mass_shape_loss(logits, total_mass, true_mass):
    """Mean log integrated-mass-ratio squared + spatial KL, N4XYZ tensors.

    True masses are a prevalidated nonnegative partition of total_mass. Empty
    total cells are excluded exactly; safe log arguments below do not floor or
    modify physical masses. Sparse target zeros have zero KL contribution.
    """
    axes = (-3, -2, -1)
    truth_total = true_mass.sum(axes, keepdim=True)
    if not bool(torch.isfinite(truth_total).all() and (truth_total > 0).all()):
        raise ValueError('INCONCLUSIVE_TARGET_UNAVAILABLE')
    positive_total = total_mass > 0
    positive_truth = true_mass > 0
    if bool((positive_truth & ~positive_total).any()):
        raise ValueError('target mass in empty total cell')
    log_total = torch.where(positive_total, total_mass, 1).log()
    log_mass = (logits.log_softmax(1)+log_total).masked_fill(~positive_total, -torch.inf)
    log_predicted_total = torch.logsumexp(log_mass, dim=axes, keepdim=True)
    mass_term = (log_predicted_total-truth_total.log()).square()
    q = true_mass/truth_total
    log_q = torch.where(positive_truth, true_mass, 1).log()-truth_total.log()
    log_predicted_shape = torch.where(positive_truth, log_mass-log_predicted_total, 0)
    shape_term = (q*(log_q-log_predicted_shape)).sum(axes, keepdim=True)
    return (mass_term+shape_term).mean(), mass_term.flatten(1), shape_term.flatten(1)


def map_loss(predicted_mass, true_mass):
    totals = true_mass.sum(dim=(-3, -2, -1))
    if not bool(torch.isfinite(totals).all() and (totals > 0).all()):
        raise ValueError('INCONCLUSIVE_TARGET_UNAVAILABLE')
    errors = (predicted_mass-true_mass).abs().sum(dim=(-3, -2, -1))/totals
    return errors.mean(), errors


def metrics(prediction, truth, total, dx=.1875):
    """Small fixed summaries; native units Msun, not M200c or stellar centers."""
    p = prediction.astype(np.float64)
    t = truth.astype(np.float64)
    pt, tt = p.sum((1, 2, 3)), t.sum((1, 2, 3))
    if np.any(tt <= 0) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('invalid mass map or unavailable target')
    centroids = []
    for r in range(4):
        if pt[r] == 0:
            centroids.append(None)
            continue
        delta = []
        for a in range(3):
            sums = tuple(k for k in range(3) if k != a)
            axis = (np.arange(p.shape[1+a])+.5)*dx
            delta.append(float(np.dot(p[r].sum(sums)/pt[r]-t[r].sum(sums)/tt[r], axis)))
        centroids.append(float(np.linalg.norm(delta)))
    return dict(map_L1=(abs(p-t).sum((1, 2, 3))/tt).tolist(),
        mass_relative_error=abs(pt/tt-1).tolist(), predicted_mass_Msun=pt.tolist(),
        true_bound_or_remainder_mass_Msun=tt.tolist(),
        overlap=(np.minimum(p, t).sum((1, 2, 3))/tt).tolist(), centroid_error_cMpc_h=centroids,
        empty_predicted_roles=(pt == 0).tolist(),
        conservation_relative_max=float(abs(p.sum(0)-total).max()/max(float(np.max(total)), 1)),
        zero_mass_predictor_map_L1=[1., 1., 1., 1.])
