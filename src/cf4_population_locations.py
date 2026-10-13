"""Small population-weighted center law. Not a halo catalogue or a field prior."""
from collections import defaultdict
import math

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize_scalar
import torch
from torch import nn

from cf4_role_locations import normalized, WIDTHS

DX, BOX_CELLS, CORE, HALO = .1875, 400, 64, 4
OBSERVER = (36., 36., 36.)
SLABS = (('train', 0, 184), ('calibration', 184, 272), ('test', 272, 400))


def native_features(dataset, check=lambda: None):
    """Stream native moments into five scalar fields with <=4-cell support.

    Inputs retain M,Pxyz,Qdiag. Output M,mean3(M),mean9(M),sigma_trace,
    |V-mean3(P)/mean3(M)|; these are features, not a replacement physical state.
    Periodic separable filters are fixed and NOT trained spatial convolutions.
    """
    mass = np.asarray(dataset[0], dtype=np.float64)
    if mass.ndim != 3 or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError('invalid native mass')
    out = np.empty((5,)+mass.shape, dtype=np.float32)
    positive = mass > 0
    out[0] = mass
    mean3 = uniform_filter(mass, size=3, mode='wrap')
    # Roundoff can make a formally zero average slightly negative.
    # Physical source is untouched; nonnegative fixed-filter feature expected.
    mean3 = np.maximum(mean3, 0)
    out[1] = mean3
    out[2] = np.maximum(uniform_filter(mass, size=9, mode='wrap'), 0)
    variance_sum, rough_square = np.zeros_like(mass), np.zeros_like(mass)
    for axis in range(3):
        check()
        velocity = np.array(dataset[1+axis], dtype=np.float64, copy=True)
        local = uniform_filter(velocity, size=3, mode='wrap')
        np.divide(velocity, mass, out=velocity, where=positive)
        velocity[~positive] = 0
        np.divide(local, mean3, out=local, where=mean3 > 0)
        local[mean3 <= 0] = 0
        local -= velocity
        np.square(local, out=local)
        local[~positive] = 0  # no bulk-boost-dependent empty-cell velocity feature
        rough_square += local
        del local
        second = np.array(dataset[4+axis], dtype=np.float64, copy=True)
        np.divide(second, mass, out=second, where=positive)
        second[~positive] = 0
        v2 = velocity*velocity
        tolerance = 64*np.finfo(np.float64).eps*np.maximum(np.maximum(abs(second), v2), 1)
        second -= v2
        np.maximum(second, 0, out=second)
        second[second <= tolerance] = 0  # same cold-roundoff convention as native state()
        variance_sum += second
        del second, v2, tolerance, velocity
    out[3] = np.sqrt(variance_sum/3)
    out[4] = np.sqrt(rough_square)
    if not np.isfinite(out).all() or np.any(out < 0):
        raise ValueError('invalid derived scalar fields')
    return out


def core_origin(position):
    return (np.floor(np.asarray(position)/1.5).astype(np.int64)*8-32) % BOX_CELLS


def cells(position, origin):
    answer = (np.floor(np.asarray(position)/DX).astype(np.int64)-origin) % BOX_CELLS
    if np.any(answer < 0) or np.any(answer >= CORE):
        raise ValueError('eligible native center outside core; no label pruning')
    return answer


def footprint_overlap(origin, previous_lower_cMpc_h):
    center = np.asarray(origin)+CORE/2
    previous = np.asarray(previous_lower_cMpc_h)/DX+128/2
    distance = abs((center-previous+BOX_CELLS/2) % BOX_CELLS-BOX_CELLS/2)
    return bool(np.all(distance < (CORE+2*HALO+128)/2))


def population_cases(identities, branches, positions, previous_lowers):
    """All archived satellite alternatives, with fixed voxel-disjoint slabs."""
    rows = sorted(set(map(tuple, np.asarray(identities)[np.asarray(branches) == 0].tolist())))
    by_mw = defaultdict(lambda: defaultdict(set))
    for mw, m31, m33 in rows:
        if len({mw, m31, m33}) != 3:
            raise ValueError('duplicate native identities in a triple')
        by_mw[mw][m31].add(m33)
    cases = {name: [] for name, *_ in SLABS}
    dropped = dict(slab_boundary=0, former_development_footprint=0)
    for mw, companions in sorted(by_mw.items()):
        origin = core_origin(positions[mw])
        name = next((name for name, lo, hi in SLABS
            if origin[0]-HALO >= lo and origin[0]+CORE+HALO <= hi), None)
        if name is None:
            dropped['slab_boundary'] += 1
            continue
        if name == 'test' and any(footprint_overlap(origin, old) for old in previous_lowers):
            dropped['former_development_footprint'] += 1
            continue
        pairs = [dict(m31=m31, center=cells(positions[m31], origin).tolist(),
            thirds=sorted(thirds), centers=cells([positions[t] for t in sorted(thirds)], origin).tolist())
            for m31, thirds in sorted(companions.items())]
        cases[name].append(dict(mw=mw, origin=origin.tolist(),
            center=cells(positions[mw], origin).tolist(), pairs=pairs))
    split_ids = {name: {sid for c in members for sid in
        [c['mw']]+[p['m31'] for p in c['pairs']]+[t for p in c['pairs'] for t in p['thirds']]}
        for name, members in cases.items()}
    for a, b in (('train', 'calibration'), ('train', 'test'), ('calibration', 'test')):
        if split_ids[a] & split_ids[b]:
            raise ValueError('native identity shared across spatial splits')
    report = dict(archived_satellite_rows=len(rows), archived_satellite_observers=len(by_mw),
        observer_counts={name: len(members) for name, members in cases.items()}, dropped=dropped,
        pair_counts={name: sum(len(c['pairs']) for c in members) for name, members in cases.items()},
        triple_counts={name: sum(len(p['thirds']) for c in members for p in c['pairs']) for name, members in cases.items()},
        cross_split_shared_ids=False, independent_universes=False)
    return cases, report


def tuple_targets(case):
    """Target weights sum to one PER observer, including mapped-cell collisions."""
    locations, weights = [], []
    for pair in case['pairs']:
        for center in pair['centers']:
            locations.append([case['center'], pair['center'], center])
            weights.append(1/len(case['pairs'])/len(pair['centers']))
    return np.asarray(locations, np.int64), np.asarray(weights, np.float64)


def training_shared_probability(cases):
    probability = 0.
    for case in cases:
        targets, weights = tuple_targets(case)
        probability += float(weights[np.all(targets[:, 1] == targets[:, 2], axis=1)].sum())/len(cases)
    return (len(cases)*probability+1)/(len(cases)+2)


def mixture(log0, log1, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('mixture coefficient outside unit interval')
    if alpha == 0:
        return log0
    if alpha == 1:
        return log1
    return torch.logaddexp(log0+math.log1p(-alpha), log1+math.log(alpha))


def calibrate_alpha(log0, log1, weights):
    """One convex scalar calibration, using only predeclared calibration rows."""
    log0, log1, weights = map(lambda v: np.asarray(v, dtype=np.float64), (log0, log1, weights))
    if log0.shape != log1.shape or log0.shape != weights.shape or not np.isfinite([log0, log1, weights]).all() or np.any(weights < 0):
        raise ValueError('invalid calibration records')
    if not np.isclose(weights.sum(), 1, atol=1e-9):
        raise ValueError('calibration weights not normalized')
    def objective(alpha):
        value = log0 if alpha == 0 else log1 if alpha == 1 else np.logaddexp(log0+math.log1p(-alpha), log1+math.log(alpha))
        return -float(weights @ value)
    fitted = minimize_scalar(objective, bounds=(0, 1), method='bounded', options={'xatol': 1e-8})
    if not fitted.success:
        raise ValueError('one-dimensional calibration did not converge')
    candidates = [(a, objective(a)) for a in (0., 1., float(fitted.x))]
    best = min(value for _, value in candidates)
    # Prefer an exact endpoint on numerical ties, not a spurious tiny learned gain.
    alpha = next(a for a, value in candidates if value <= best+1e-10)
    return dict(alpha=alpha, mean_role_NLL=objective(alpha), density_NLL=objective(0),
                raw_NLL=objective(1), endpoint_numerical_tie_NLL=1e-10)


class PopulationRoleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.ParameterList([nn.Parameter(torch.zeros(6+r)) for r in range(3)])
        self.alpha = 1.
        self.register_buffer('grid', None, persistent=False)

    def design(self, raw, observer, parents, role):
        n = raw.shape[-1]
        if raw.shape != (5, n, n, n) or role not in range(3):
            raise ValueError('five scalar fields and role0..2 required')
        parents = torch.as_tensor(parents, device=raw.device).reshape(-1, 3)
        observer = torch.as_tensor(observer, dtype=raw.dtype, device=raw.device)
        if len(parents) != role or observer.shape != (3,) or not bool(torch.isfinite(observer).all()):
            raise ValueError('exact preceding-role conditioning required')
        if not bool(torch.isfinite(parents).all()) or bool(((parents < 0) | (parents >= n) | (parents != parents.round())).any()):
            raise ValueError('invalid parent centers')
        if self.grid is None or self.grid.shape[-1] != n or self.grid.device != raw.device:
            axis = torch.arange(n, dtype=raw.dtype, device=raw.device)+.5
            self.grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'))
        mean = raw[0].mean()
        if not bool(torch.isfinite(mean) and mean > 0):
            raise ValueError('positive mean native mass required')
        density = torch.log1p(raw[:3]/mean)
        features = [*torch.tanh(density/5), *torch.tanh(raw[3:]/300)]
        anchors = [observer]+[p.to(raw.dtype)+.5 for p in parents]
        for anchor in anchors:
            radius2 = ((self.grid-anchor[:, None, None, None])*DX).square().sum(0)
            features.append(torch.tanh(radius2.sqrt()/3))
        geometry = -.5*radius2/WIDTHS[role]**2
        return torch.stack(features), geometry+density[0], geometry

    def distributions(self, raw, observer, parents, role, pi=.5, alpha=None):
        features, base, geom = self.design(raw, observer, parents, role)
        log0 = normalized(base)
        log1 = normalized(base+torch.einsum('c,cxyz->xyz', self.theta[role], features))
        shared = log0
        if role == 2:
            if not 0 < pi < 1:
                raise ValueError('same-cell reference smoothing must be interior')
            shared = log0+math.log1p(-pi)
            shared = shared.clone()
            key = tuple(int(x) for x in parents[1])
            shared[key] = torch.logaddexp(shared[key], shared.new_tensor(math.log(pi)))
        return dict(raw=log1, calibrated=mixture(log0, log1, self.alpha if alpha is None else alpha),
                    density=log0, geometry=normalized(geom), same_cell=shared)

    def log_prob(self, raw, observer, parents, role):
        return self.distributions(raw, observer, parents, role)['calibrated']

    def raw_log_prob(self, raw, observer, parents, role):
        features, base, _ = self.design(raw, observer, parents, role)
        return normalized(base+torch.einsum('c,cxyz->xyz', self.theta[role], features))


def selected(logp, locations):
    indices = torch.as_tensor(locations, dtype=torch.long, device=logp.device).reshape(-1, 3)
    return logp[tuple(indices.T)]


def field_patch(field, origin, n=CORE):
    axes = [(torch.arange(n, device=field.device)+int(o)) % field.shape[-1] for o in origin]
    return field[:, axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]]
