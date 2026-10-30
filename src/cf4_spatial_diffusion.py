"""Joint spatial Gaussian/absorbing-state diffusion, not a field likelihood.

49 split coordinates and49 codes are denoised together. Physical decoding is
an explicit many-to-one conservative map; invalid floating-point draws fail.
Absorbing schedule follows Austin et al.2107.03006, Appendix A.3: survival
1-t/T, reverse MASK reveal probability1/t, weighted masked cross entropy.
"""
import math
from itertools import permutations, product

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cf4_split_moments import NODES, state, encode_tree, decode, merge_octets
from cf4_continuous_matter import check_realizable, restrict

SYMMETRIES = list(product(list(permutations(range(3))), list(product((-1, 1), repeat=3))))
BRANCH_VERSION = 'cf4_mixed_joint_canonical_v1'


def augment(value, symmetry):
    axes, signs = SYMMETRIES[int(symmetry)]
    out = value.transpose((0,)+tuple(a+1 for a in axes))
    out = out[(slice(None),)+tuple(slice(None, None, s) for s in signs)]
    out = out[[0]+[a+1 for a in axes]+[a+4 for a in axes]].copy()
    out[1:4] *= np.asarray(signs).reshape(3, 1, 1, 1)
    return out


def pack(fine):
    records = encode_tree(fine)
    return (np.concatenate([r[0] for r in records]),
            np.concatenate([r[1] for r in records]), records[0][2])


def canonical(raw, parent):
    """Complete versioned branch table; codes0..3 only, no remaining MASK4.

    Mass: empty parent->0; otherwise raw0->2, raw1/2/3 retained.
    Relative velocity: mass!=2 or parent cold->0; otherwise raw0->2.
    Internal allocation: relative!=2->0; otherwise raw0->2.
    Unused coordinates are ignored by the physical decoder, not fitted floors.
    """
    if raw.shape != parent.shape or not np.isin(raw, (0, 1, 2, 3)).all():
        raise ValueError('invalid raw category shape/range or unresolved MASK')
    out = raw.astype(np.uint8, copy=True)
    out[0] = np.where(parent[0] > 0, np.where(raw[0] == 0, 2, raw[0]), 0)
    variance = state(parent)[1]
    for a in range(3):
        enabled = (out[0] == 2) & (variance[a] > 0)
        out[a+1] = np.where(enabled, np.where(raw[a+1] == 0, 2, raw[a+1]), 0)
        out[a+4] = np.where(out[a+1] == 2, np.where(raw[a+4] == 0, 2, raw[a+4]), 0)
    return out


def unpack(z, raw, root):
    z = np.asarray(z, np.float64)
    if z.shape != (49,)+root.shape[1:] or raw.shape != z.shape:
        raise ValueError('joint split shape mismatch')
    tree, counts = {(0, 8): root}, []
    for i, (lo, hi) in enumerate(NODES):
        parent = tree[(lo, hi)]
        codes = raw[7*i:7*i+7]
        legal = canonical(codes, parent)
        counts.append((legal != codes).sum(axis=(1, 2, 3)).tolist())
        middle = (lo+hi)//2
        tree[(lo, middle)], tree[(middle, hi)] = decode(z[7*i:7*i+7], legal, parent)
    fine = merge_octets(np.array([tree[(i, i+1)] for i in range(8)]))
    check_realizable(fine)
    scales = np.maximum(np.max(abs(root), axis=(1, 2, 3)), 1)
    error = np.max(abs(restrict(fine, 2)-root), axis=(1, 2, 3))/scales
    if max(error) > 1e-8:
        raise ValueError('joint diffusion seven-moment conservation failed')
    return fine, dict(version=BRANCH_VERSION, corrected_per_node_channel=counts,
                      sites_per_node_channel=int(root[0].size), coarse_error=error.tolist())


def context(root, dx):
    velocity, variance = state(root)
    return np.concatenate([np.log1p(root[:1]/(1e11*(2*dx)**3)),
                           np.arcsinh(velocity/300), np.log1p(np.sqrt(variance)/100)]).astype(np.float32)


class Residual(nn.Module):
    def __init__(self, inputs, outputs, embedding=192):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, inputs), nn.GroupNorm(8, outputs)
        self.c1, self.c2 = nn.Conv3d(inputs, outputs, 3, padding=1), nn.Conv3d(outputs, outputs, 3, padding=1)
        self.emb = nn.Linear(embedding, outputs)
        self.skip = nn.Identity() if inputs == outputs else nn.Conv3d(inputs, outputs, 1)

    def forward(self, x, embedding):
        y = self.c1(F.silu(self.n1(x)))
        y = y+self.emb(F.silu(embedding))[:, :, None, None, None]
        return self.skip(x)+self.c2(F.silu(self.n2(y)))


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv, self.out = nn.Conv3d(channels, 3*channels, 1), nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, 4, c//4, -1).unbind(1)
        y = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x+self.out(y.transpose(-1, -2).reshape(b, c, *spatial))


class SpatialDenoiser(nn.Module):
    def __init__(self, widths=(48, 96, 192, 384)):
        super().__init__()
        self.stem = nn.Conv3d(49+49*5+7, widths[0], 1)
        self.embedding = nn.Sequential(nn.Linear(65, 192), nn.SiLU(), nn.Linear(192, 192))
        self.down = nn.ModuleList([nn.ModuleList([Residual(w, w), Residual(w, w)]) for w in widths])
        self.stride = nn.ModuleList([nn.Conv3d(a, b, 3, stride=2, padding=1) for a, b in zip(widths[:-1], widths[1:])])
        self.attention = Attention(widths[-1])
        self.upconv = nn.ModuleList([nn.Conv3d(a, b, 3, padding=1) for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.up = nn.ModuleList([nn.ModuleList([Residual(2*w, w), Residual(w, w)]) for w in widths[-2::-1]])
        self.norm = nn.GroupNorm(8, widths[0])
        self.head = nn.Conv3d(widths[0], 49*5, 1)

    def block(self, layer, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def forward(self, z, code, coarse, timestep, scale):
        frequency = torch.exp(torch.arange(32, device=z.device)*(-math.log(10000)/31))
        angles = timestep.float()[:, None]*frequency[None]
        embedding = self.embedding(torch.cat([angles.sin(), angles.cos(), scale[:, None]], 1))
        categories = F.one_hot(code.long(), 5).movedim(-1, 2).flatten(1, 2).float()
        x = self.stem(torch.cat([z, categories, coarse], 1))
        skips = []
        for i, layers in enumerate(self.down):
            for layer in layers:
                x = self.block(layer, x, embedding)
            skips.append(x)
            if i < 3:
                x = self.stride[i](x)
        x = self.block(self.attention, x)
        for i, layers in enumerate(self.up):
            x = self.upconv[i](F.interpolate(x, size=skips[-2-i].shape[-3:], mode='nearest'))
            x = torch.cat([x, skips[-2-i]], 1)
            for layer in layers:
                x = self.block(layer, x, embedding)
        output = self.head(F.silu(self.norm(x)))
        return output[:, :49], output[:, 49:].reshape(z.shape[0], 49, 4, *z.shape[-3:])


def gaussian_schedule(steps, device='cpu'):
    # Cosine schedule, finite beta ceiling for the epsilon reverse mean.
    grid = torch.arange(steps+1, dtype=torch.float64, device=device)/steps
    abar = torch.cos((grid+.008)/1.008*math.pi/2)**2
    abar = abar/abar[0]
    beta = (1-abar[1:]/abar[:-1]).clamp(max=.999)
    return beta.float(), torch.cat([torch.ones(1, device=device), torch.cumprod(1-beta, 0)]).float()


def categorical_reverse_probs(logits, current, t):
    """Linear survival => q(reveal at t-1 | MASK at t)=1/t (including t=1)."""
    probabilities = logits.softmax(2)/t
    remain = torch.full_like(probabilities[:, :, :1], 1-1/t)
    masked = torch.cat([probabilities, remain], 2)
    fixed = F.one_hot(current.long(), 5).movedim(-1, 2).float()
    return torch.where((current == 4).unsqueeze(2), masked, fixed)


def diffusion_loss(model, clean, codes, coarse, scale, steps=100):
    t = torch.randint(1, steps+1, (clean.shape[0],), device=clean.device)
    _, abar = gaussian_schedule(steps, clean.device)
    a = abar[t].reshape(-1, 1, 1, 1, 1)
    noise = torch.randn_like(clean)
    noisy = a.sqrt()*clean+(1-a).sqrt()*noise
    masked = torch.rand_like(clean) < (t.float()/steps).reshape(-1, 1, 1, 1, 1)
    noisy_code = torch.where(masked, 4, codes)
    predicted, logits = model(noisy, noisy_code, coarse, t, scale)
    continuous = (predicted-noise).square().mean()
    ce = F.cross_entropy(logits.flatten(0, 1), codes.flatten(0, 1), reduction='none').reshape_as(clean)
    # Uniform-time unbiased absorbing variational loss per category/site.
    discrete = (ce*masked*(steps/t.float()).reshape(-1, 1, 1, 1, 1)).mean()
    return continuous+discrete, dict(continuous=float(continuous.detach()), categorical=float(discrete.detach()), t=int(t[0]))


@torch.no_grad()
def sample(model, coarse, dx, location, spread, steps=100, check=lambda: None):
    shape = (coarse.shape[0], 49)+coarse.shape[-3:]
    z = torch.randn(shape, device=coarse.device)
    codes = torch.full(shape, 4, dtype=torch.long, device=coarse.device)
    beta, abar = gaussian_schedule(steps, coarse.device)
    scale = torch.full((shape[0],), math.log2(dx/.1875), device=coarse.device)
    for t in range(steps, 0, -1):
        check()
        eps, logits = model(z, codes, coarse, torch.full((shape[0],), t, device=coarse.device), scale)
        if not torch.isfinite(eps).all() or not torch.isfinite(logits).all():
            raise ValueError('nonfinite reverse-chain prediction')
        mean = (z-beta[t-1]/torch.sqrt(1-abar[t])*eps)/torch.sqrt(1-beta[t-1])
        variance = beta[t-1]*(1-abar[t-1])/(1-abar[t])
        z = mean+(torch.sqrt(variance)*torch.randn_like(z) if t > 1 else 0)
        # Only masked sites can change; direct reveal avoids materializing a
        # 5-category multinomial for every already fixed spatial site.
        probabilities = logits.softmax(2)
        u = torch.rand_like(z).unsqueeze(2)
        draw = (u > probabilities.cumsum(2)).sum(2).clamp_max(3)
        reveal = (codes == 4) & (torch.rand_like(z) < 1/t)
        codes = torch.where(reveal, draw, codes)
    if not torch.isfinite(z).all() or (codes == 4).any():
        raise ValueError('nonfinite diffusion coordinates or unrevealed category')
    physical_z = z[0].cpu().numpy().astype(np.float64)*spread[:, None, None, None]+location[:, None, None, None]
    return physical_z, codes[0].cpu().numpy().astype(np.uint8)
