"""Normalized mixed atom/continuous binary-moment conditional, in split coordinates.

Not a physical56D cell-density likelihood, halo model or observed posterior.
Mask probabilities precede a masked affine flow; inactive dimensions never
contribute a Gaussian density or Jacobian. Spatial coupling is within a patch.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from cf4_split_moments import state


def condition(parent, root, child_dx, node):
    fields = []
    # Fixed units, not training/heldout normalization fitted to observations.
    for x, cells in ((parent, (8, 4, 4, 2, 2, 2, 2)[node]), (root, 8)):
        v, variance = state(x)
        fields.extend([np.log1p(x[0]/(1e11*child_dx**3*cells))[None],
                       np.arcsinh(v/300), np.log1p(np.sqrt(variance)/100)])
    shape = parent.shape[1:]
    node_info = np.zeros((7,)+shape)
    node_info[node] = 1
    fields.extend([node_info, np.full((1,)+shape, np.log2(child_dx/.1875))])
    valid = np.concatenate([(parent[0] > 0)[None], state(parent)[1] > 0])
    return np.concatenate(fields).astype(np.float32), valid


def network(inputs, outputs, zero=False):
    net = nn.Sequential(nn.Conv3d(inputs, 32, 3, padding=1), nn.SiLU(),
                        nn.Conv3d(32, 64, 3, padding=1), nn.SiLU(), nn.Conv3d(64, outputs, 1))
    if zero:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


def onehot(mask):
    return F.one_hot(mask.long(), 4).permute(0, 1, 5, 2, 3, 4).flatten(1, 2).float()


def category(logits):
    shape = logits.shape
    return torch.multinomial(logits.movedim(1, -1).reshape(-1, 3).softmax(-1), 1).reshape(shape[0], *shape[2:])+1


def categorical_logp(logits, state):
    return logits.log_softmax(1).gather(1, (state-1).clamp_min(0).long()[:, None])[:, 0]*(state > 0)


class Coupling(nn.Module):
    def __init__(self, parity):
        super().__init__()
        self.register_buffer('fixed', ((torch.arange(7)+parity) % 2 == 0).reshape(1, 7, 1, 1, 1))
        self.net = network(22+28+7, 14, zero=True)

    def forward(self, x, context, mask, inverse=False):
        active = mask == 2
        fixed = self.fixed & active
        changed = ~self.fixed & active
        shift, raw_scale = self.net(torch.cat([context, onehot(mask), x*fixed], 1)).chunk(2, 1)
        scale = 2*torch.tanh(raw_scale)*changed
        shift = shift*changed
        y = (x-shift)*torch.exp(-scale) if inverse else x*torch.exp(scale)+shift
        return y*active, (-scale if inverse else scale).sum(1)


class ConditionalSplitFlow(nn.Module):
    def __init__(self, location, spread):
        super().__init__()
        self.register_buffer('location', torch.as_tensor(location, dtype=torch.float32).reshape(1, 7, 1, 1, 1))
        self.register_buffer('spread', torch.as_tensor(spread, dtype=torch.float32).reshape(1, 7, 1, 1, 1))
        self.mass_mask = network(22, 3)
        self.velocity_mask = network(22+4, 9)
        self.variance_mask = network(22+16, 9)
        self.layers = nn.ModuleList([Coupling(i) for i in range(6)])

    def masks(self, context, valid, truth=None):
        shape = (context.shape[0], 7)+context.shape[2:]
        mask = torch.zeros(shape, dtype=torch.long, device=context.device)
        logp = torch.zeros_like(context[:, 0])
        logits = self.mass_mask(context)
        mask[:, 0] = (category(logits) if truth is None else truth[:, 0])*valid[:, 0]
        logp += categorical_logp(logits, mask[:, 0])
        wc = onehot(mask[:, :1])
        rlogits = self.velocity_mask(torch.cat([context, wc], 1)).chunk(3, 1)
        for axis in range(3):
            enabled = (mask[:, 0] == 2) & valid[:, 1+axis]
            s = category(rlogits[axis]) if truth is None else truth[:, 1+axis]
            mask[:, 1+axis] = s*enabled
            logp += categorical_logp(rlogits[axis], mask[:, 1+axis])
        alogits = self.variance_mask(torch.cat([context, onehot(mask[:, :4])], 1)).chunk(3, 1)
        for axis in range(3):
            enabled = mask[:, 1+axis] == 2
            s = category(alogits[axis]) if truth is None else truth[:, 4+axis]
            mask[:, 4+axis] = s*enabled
            logp += categorical_logp(alogits[axis], mask[:, 4+axis])
        if truth is not None and not torch.equal(mask, truth):
            raise ValueError('native boundary masks violate conditional support')
        return mask, logp

    def log_prob(self, z, mask, context, valid):
        _, logp = self.masks(context, valid, mask)
        active = mask == 2
        x = ((z-self.location)/self.spread)*active
        logp = logp-(torch.log(self.spread)*active).sum(1)
        for layer in reversed(self.layers):
            x, jac = layer(x, context, mask, inverse=True)
            logp = logp+jac
        return logp-.5*((x*x+np.log(2*np.pi))*active).sum(1)

    @torch.no_grad()
    def sample(self, context, valid):
        mask, _ = self.masks(context, valid)
        x = torch.randn(mask.shape, device=context.device)*(mask == 2)
        for layer in self.layers:
            x, _ = layer(x, context, mask)
        return (x*self.spread+self.location)*(mask == 2), mask
