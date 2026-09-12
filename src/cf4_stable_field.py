"""Observer-conditional joint moment diffusion with v (NOT physical velocity)."""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from cf4_spatial_diffusion import gaussian_schedule, context


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, width), nn.GroupNorm(8, width)
        self.c1 = nn.Conv3d(width, width, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv3d(width, width, 3, padding=dilation, dilation=dilation)
        self.embedding = nn.Linear(128, width)

    def forward(self, value, embedding):
        hidden = self.c1(functional.silu(self.n1(value)))
        hidden = hidden+self.embedding(functional.silu(embedding))[:, :, None, None, None]
        return value+self.c2(functional.silu(self.n2(hidden)))


class Branch(nn.Module):
    def __init__(self, continuous):
        super().__init__()
        self.stem = nn.Conv3d(304, 64, 1)
        self.embedding = nn.Sequential(nn.Linear(65, 128), nn.SiLU(), nn.Linear(128, 128))
        self.blocks = nn.ModuleList([Block(64, d) for d in (1, 2, 4)])
        self.norm = nn.GroupNorm(8, 64)
        self.head = nn.Conv3d(64, 49 if continuous else 196, 1)
        self.gain = nn.Linear(128, 49) if continuous else None
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if self.gain is not None:
            nn.init.zeros_(self.gain.weight)
            nn.init.zeros_(self.gain.bias)

    def forward(self, z, categories, conditioning, timestep, scale):
        frequency = torch.exp(torch.arange(32, device=z.device)*(-math.log(10000)/31))
        angle = timestep.float()[:, None]*frequency[None]
        embed = self.embedding(torch.cat([angle.sin(), angle.cos(), scale[:, None]], 1))
        code = functional.one_hot(categories.long(), 5).movedim(-1, 2).flatten(1, 2).to(z.dtype)
        hidden = self.stem(torch.cat([z, code, conditioning], 1))
        for block in self.blocks:
            hidden = block(hidden, embed)
        output = self.head(functional.silu(self.norm(hidden)))
        if self.gain is not None:
            return output+self.gain(embed)[:, :, None, None, None]*z
        return output.reshape(z.shape[0], 49, 4, *z.shape[-3:])


class StableField(nn.Module):
    def __init__(self):
        super().__init__()
        self.continuous = Branch(True)
        self.categorical = Branch(False)

    def forward(self, *args):
        return self.continuous(*args), self.categorical(*args)


def conditioning(parent, child_dx, observer_cMpc_h):
    n = parent.shape[-1]
    coordinates = (np.stack(np.meshgrid(*[np.arange(n)+.5]*3, indexing='ij'))*2*child_dx
        -np.asarray(observer_cMpc_h)[:, None, None, None])/12
    return np.concatenate([context(parent, child_dx), coordinates.astype(np.float32)])


def noisy_record(clean, codes, t, steps=100, noise=None, uniform=None):
    _, abar = gaussian_schedule(steps, clean.device)
    a = abar[t].reshape(-1, 1, 1, 1, 1)
    noise = torch.randn_like(clean) if noise is None else noise
    uniform = torch.rand_like(clean) if uniform is None else uniform
    masked = uniform < (t.float()/steps).reshape(-1, 1, 1, 1, 1)
    noisy = a.sqrt()*clean+(1-a).sqrt()*noise
    target = a.sqrt()*noise-(1-a).sqrt()*clean
    return noisy, torch.where(masked, 4, codes), target, masked


def loss(model, clean, codes, coarse, scale, steps=100):
    t = torch.randint(1, steps+1, (clean.shape[0],), device=clean.device)
    noisy, masked_codes, target, mask = noisy_record(clean, codes, t, steps)
    predicted, logits = model(noisy, masked_codes, coarse, t, scale)
    continuous = (predicted-target).square().mean()
    ce = functional.cross_entropy(logits.flatten(0, 1), codes.flatten(0, 1), reduction='none').reshape_as(clean)
    categorical = (ce*mask*(steps/t.float()).reshape(-1, 1, 1, 1, 1)).mean()
    return continuous, categorical, dict(v_loss=float(continuous.detach()),
        category_loss=float(categorical.detach()), zero_v_loss=float(target.square().mean()), t=int(t[0]))


def reverse_mean(z, v, beta, abar_t, abar_previous):
    """Algebraically equivalent to DDPM epsilon mean, no high-noise cancellation."""
    return torch.sqrt(1-beta)*z-beta*torch.sqrt(abar_previous)/torch.sqrt(1-abar_t)*v


@torch.no_grad()
def sample(model, coarse, child_dx, location, spread, steps=100, check=lambda: None):
    shape = (coarse.shape[0], 49)+coarse.shape[-3:]
    z = torch.randn(shape, device=coarse.device)
    codes = torch.full(shape, 4, dtype=torch.long, device=coarse.device)
    beta, abar = gaussian_schedule(steps, coarse.device)
    scale = torch.full((shape[0],), math.log2(child_dx/.1875), device=coarse.device)
    history = []
    for t in range(steps, 0, -1):
        check()
        v, logits = model(z, codes, coarse, torch.full((shape[0],), t, device=coarse.device), scale)
        if not bool(torch.isfinite(v).all()) or not bool(torch.isfinite(logits).all()):
            raise ValueError('nonfinite v or categorical prediction')
        mean = reverse_mean(z, v, beta[t-1], abar[t], abar[t-1])
        variance = beta[t-1]*(1-abar[t-1])/(1-abar[t])
        z = mean+(variance.sqrt()*torch.randn_like(z) if t > 1 else 0)
        probabilities = logits.softmax(2)
        draw = (torch.rand_like(z).unsqueeze(2) > probabilities.cumsum(2)).sum(2).clamp_max(3)
        reveal = (codes == 4) & (torch.rand_like(z) < 1/t)
        codes = torch.where(reveal, draw, codes)
        if t in (100, 50, 10, 1):
            history.append(dict(t=t, latent_RMS=float(z.square().mean().sqrt()), v_RMS=float(v.square().mean().sqrt())))
    if not bool(torch.isfinite(z).all()) or bool((codes == 4).any()):
        raise ValueError('nonfinite latent or unrevealed category')
    native = z[0].cpu().numpy().astype(np.float64)*spread[:, None, None, None]+location[:, None, None, None]
    return native, codes[0].cpu().numpy().astype(np.uint8), history
