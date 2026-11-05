"""Normalized autoregressive center-cell law, NOT resolved halos or a field posterior."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cf4_split_moments import state
from cf4_spatial_diffusion import SYMMETRIES

ROLES = ('MW', 'M31', 'M33')
WIDTHS = (.75, 1.5, 1.)


def native_center_cells(positions_ckpc_h, lower_cMpc_h, dx=.1875, box=75., n=128):
    """Native SubhaloPos labels, not density peaks; retain periodic patch faces."""
    positions = np.asarray(positions_ckpc_h, dtype=np.float64)
    lower = np.asarray(lower_cMpc_h, dtype=np.float64)
    if positions.shape != (3, 3) or lower.shape != (3,) or not np.isfinite(positions).all() or not np.isfinite(lower).all():
        raise ValueError('three finite native centers and one patch origin required')
    local = (positions/1000-lower) % box
    cells = np.floor(local/dx).astype(np.int64)
    if np.any(cells < 0) or np.any(cells >= n):
        raise ValueError('native center outside patch; no nearest-peak substitution')
    return cells


def native_state(moments):
    """M in1e12 Msun, mean V in km/s, all three variances in(km/s)^2."""
    velocity, variance = state(moments)
    raw = np.concatenate([moments[:1]/1e12, velocity, variance]).astype(np.float32)
    if not np.isfinite(raw).all() or np.any(raw[0] < 0) or np.any(raw[4:] < 0):
        raise ValueError('invalid native state')
    return torch.from_numpy(raw)


def transform_state(raw, symmetry):
    axes, signs = SYMMETRIES[symmetry]
    value = raw.permute(0, *(a+1 for a in axes))
    flips = [a+1 for a, s in enumerate(signs) if s < 0]
    if flips:
        value = value.flip(flips)
    value = value[[0]+[a+1 for a in axes]+[a+4 for a in axes]].clone()
    value[1:4] *= value.new_tensor(signs)[:, None, None, None]
    return value.contiguous()


def transform_positions(positions, symmetry, n, *, cell_indices=False):
    """Physical cell coordinates reflect as n-x; integer labels as n-1-i."""
    axes, signs = SYMMETRIES[symmetry]
    value = positions[..., list(axes)].clone()
    sign = value.new_tensor(signs)
    return torch.where(sign < 0, n-int(cell_indices)-value, value)


def location_features(raw, observer, parents, role, dx=.1875):
    """18 scalar fields; only already conditioned/sampled parent cells allowed."""
    n = raw.shape[-1]
    if raw.shape != (7, n, n, n) or n % 4 or role not in range(3):
        raise ValueError('seven-channel cube divisible by4 and role0..2 required')
    parents = torch.as_tensor(parents, device=raw.device).reshape(-1, 3)
    observer = torch.as_tensor(observer, device=raw.device, dtype=raw.dtype)
    if len(parents) != role or observer.shape != (3,) or not bool(torch.isfinite(observer).all()):
        raise ValueError('condition on exactly the preceding roles and one observer')
    if not bool(torch.isfinite(parents).all()) or bool(((parents < 0) | (parents >= n) | (parents != parents.round())).any()):
        raise ValueError('parent cell indices outside cube or nonintegral')
    axis = torch.arange(n, device=raw.device, dtype=raw.dtype)+.5
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'))
    velocity, variance = raw[1:4], raw[4:7]
    base = [torch.log1p(raw[0]*(10/dx**3)), torch.asinh(velocity.square().sum(0).sqrt()/300),
            torch.log1p(variance.mean(0).sqrt()/100)]
    anchors = [observer]+[p.to(raw.dtype)+.5 for p in parents]
    geometry = None
    for k in range(3):
        if k < len(anchors):
            dr = (grid-anchors[k][:, None, None, None])*dx
            r2 = dr.square().sum(0)
            denominator = (r2+dx*dx).sqrt()  # smooth direction at an anchor, no mass floor
            base.extend([torch.log1p(r2.sqrt()/dx),
                         torch.asinh((velocity*dr).sum(0)/denominator/300),
                         torch.log1p(((variance*dr.square()).sum(0)/(r2+dx*dx)).sqrt()/100),
                         torch.ones_like(r2)])
            if k == role:
                geometry = -.5*r2/WIDTHS[role]**2
        else:
            base.extend([torch.zeros_like(raw[0]) for _ in range(4)])
    base.extend([torch.full_like(raw[0], float(i == role)) for i in range(3)])
    return torch.stack(base), geometry


def normalized(logits):
    return logits-logits.flatten().logsumexp(0)


def reference_log_probs(raw, geometry, role, parents, same_cell_probability):
    """Normalized geometry, density-weighted and training same-cell references."""
    mean = raw[0].mean()
    if not bool(torch.isfinite(mean) and mean > 0):
        raise ValueError('positive mean mass required for density reference')
    plain = normalized(geometry)
    density = normalized(geometry+torch.log1p(raw[0]/mean))
    shared = density
    if role == 2:
        pi = float(same_cell_probability)
        if not 0 < pi < 1:
            raise ValueError('proper interior mixture probability required')
        shared = density+math.log1p(-pi)
        index = tuple(int(x) for x in parents[1])
        shared = shared.clone()
        shared[index] = torch.logaddexp(shared[index], shared.new_tensor(math.log(pi)))
    return plain, density, shared


class CubicConv(nn.Module):
    """Scalar3^3 kernel with four signed-permutation offset orbits."""
    def __init__(self, a, b):
        super().__init__()
        axis = torch.arange(-1, 2)
        orbit = sum(x*x for x in torch.meshgrid(axis, axis, axis, indexing='ij'))
        self.register_buffer('orbit', orbit)
        self.weight = nn.Parameter(torch.empty(b, a, 4))
        self.bias = nn.Parameter(torch.zeros(b))
        nn.init.normal_(self.weight, std=math.sqrt(2/(27*a)))

    def forward(self, value):
        return F.conv3d(value, self.weight[:, :, self.orbit].contiguous(), self.bias, padding=1)


class ScalarBlock(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(CubicConv(a, b), nn.GroupNorm(8, b), nn.SiLU(),
                         CubicConv(b, b), nn.GroupNorm(8, b), nn.SiLU())


class RoleLocationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([ScalarBlock(18, 16), ScalarBlock(32, 32), ScalarBlock(64, 64)])
        self.down = nn.ModuleList([nn.Conv3d(16, 32, 1), nn.Conv3d(32, 64, 1)])
        self.up = nn.ModuleList([CubicConv(64, 32), CubicConv(32, 16)])
        self.dec = nn.ModuleList([ScalarBlock(64, 32), ScalarBlock(32, 16)])
        self.head = nn.Conv3d(16, 1, 1)  # nonzero initialization permits first-update backbone gradients

    def block(self, layer, value):
        return checkpoint(layer, value, use_reentrant=False) if self.training and torch.is_grad_enabled() else layer(value)

    def forward(self, features):
        a = self.block(self.enc[0], features)
        b = self.block(self.enc[1], self.down[0](F.avg_pool3d(a, 2)))
        c = self.block(self.enc[2], self.down[1](F.avg_pool3d(b, 2)))
        x = self.block(self.dec[0], torch.cat([self.up[0](F.interpolate(c, scale_factor=2, mode='nearest')), b], 1))
        x = self.block(self.dec[1], torch.cat([self.up[1](F.interpolate(x, scale_factor=2, mode='nearest')), a], 1))
        return self.head(x)

    def log_prob(self, raw, observer, parents, role, dx=.1875):
        features, geometry = location_features(raw, observer, parents, role, dx)
        return normalized(self(features[None])[0, 0]+geometry)


@torch.no_grad()
def sample_triples(model, raw, observer, count, generator, check=lambda: None):
    """No truth/candidate catalogue accepted. All later parents are earlier draws."""
    if model.training or count <= 0:
        raise ValueError('positive count and evaluation mode required')
    n = raw.shape[-1]
    triples, log_scores = [], []
    mw_logp = model.log_prob(raw, observer, [], 0)
    for _ in range(count):
        parents, score = [], 0.
        for role in range(3):
            check()
            logp = mw_logp if role == 0 else model.log_prob(raw, observer, parents, role)
            if not bool(torch.isfinite(logp).all()):
                raise ValueError('nonfinite conditional probabilities')
            key = int(torch.multinomial(logp.flatten().exp(), 1, generator=generator))
            parents.append([key//(n*n), (key//n) % n, key % n])
            score += float(logp.flatten()[key])
        triples.append(parents)
        log_scores.append(score)
    return np.asarray(triples, dtype=np.int64), log_scores
