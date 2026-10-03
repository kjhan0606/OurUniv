"""Fine-lattice denoising path and parent-only physical-budget loss weights.

The orthogonal lift embeds split coordinates, NOT physical matter fields.
The existing conservative physical decoder and stochastic sampler are unchanged.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cf4_split_moments import NODES, state


class OctetLift(nn.Module):
    def __init__(self):
        super().__init__()
        matrix = torch.zeros(8, 7, dtype=torch.float64)
        for node, (lo, hi) in enumerate(NODES):
            middle = (lo+hi)//2
            matrix[lo:middle, node] = 1/math.sqrt(hi-lo)
            matrix[middle:hi, node] = -1/math.sqrt(hi-lo)
        self.register_buffer('matrix', matrix)

    def forward(self, z):
        batch, _, nx, ny, nz = z.shape
        leaves = torch.einsum('ln,bnqxyz->bqlxyz', self.matrix.to(z.dtype), z.reshape(batch, 7, 7, nx, ny, nz))
        return leaves.reshape(batch, 7, 2, 2, 2, nx, ny, nz).permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch, 7, 2*nx, 2*ny, 2*nz)

    def project(self, fine):
        batch, _, fx, fy, fz = fine.shape
        nx, ny, nz = fx//2, fy//2, fz//2
        leaves = fine.reshape(batch, 7, nx, 2, ny, 2, nz, 2).permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(batch, 7, 8, nx, ny, nz)
        return torch.einsum('ln,bqlxyz->bnqxyz', self.matrix.to(fine.dtype), leaves).reshape(batch, 49, nx, ny, nz)


class FinePath(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = OctetLift()
        self.stem = nn.Conv3d(17, 16, 3, padding=1)
        self.embedding = nn.Sequential(nn.Linear(65, 32), nn.SiLU(), nn.Linear(32, 16))
        self.blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv3d(16, 16, 3, padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv3d(16, 16, 3, padding=1)) for _ in range(2)])
        self.norm = nn.GroupNorm(4, 16)
        self.head = nn.Conv3d(16, 7, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z, coarse, timestep, scale):
        frequency = torch.exp(torch.arange(32, device=z.device)*(-math.log(10000)/31))
        angle = timestep.to(z.dtype)[:, None]*frequency[None]
        embedding = self.embedding(torch.cat([angle.sin(), angle.cos(), scale[:, None]], 1))
        hidden = self.stem(torch.cat([self.lift(z), F.interpolate(coarse, scale_factor=2, mode='nearest')], 1))
        hidden = hidden+embedding[:, :, None, None, None]
        for block in self.blocks:
            hidden = hidden+block(hidden)
        return self.lift.project(self.head(F.silu(self.norm(hidden))))


class LinkedContinuous(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.fine = FinePath()

    def forward(self, z, codes, coarse, timestep, scale):
        return self.base(z, codes, coarse, timestep, scale)+self.fine(z, coarse, timestep, scale)


def budget_weights(parent):
    """Strictly positive weights depend ONLY on the conditioned parent field.

    At fixed (noisy coordinates, codes, parent, observer, t), weighted squared
    error has the same ideal conditional-mean minimizer as unweighted MSE.
    This changes finite-capacity emphasis, not physical output amplitudes.
    """
    mass = parent[0]
    if np.any(mass < 0) or not np.isfinite(parent).all() or mass.mean() <= 0:
        raise ValueError('nonnegative finite positive-total parent required')
    variance = state(parent)[1]
    budgets = np.concatenate([mass[None], mass[None]*variance, mass[None]*variance])
    means = budgets.mean(axis=(1, 2, 3), keepdims=True)
    weights = 1+np.log1p(budgets/np.where(means > 0, means, 1.))
    weights /= weights.mean(axis=(1, 2, 3), keepdims=True)
    return np.tile(weights, (7, 1, 1, 1)).astype(np.float32)
