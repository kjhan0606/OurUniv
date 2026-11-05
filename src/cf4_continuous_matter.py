"""Continuous *kinematic* matter/marked-component transport, not an LCDM prior.

Native subhalo member moments form disjoint components. Everything else is
the remainder (including other halos, NOT just diffuse gas). A translation
uses exact overlaps of piecewise-constant source cells; no discrete catalogue
selection. Bound mass and COM velocity change jointly with the moment field.
Native membership/profile topology is held fixed. Binding, M200c, dynamics,
profile scatter and cosmological probability are NOT inferred by this map.
"""
from itertools import product
import numpy as np


def boost_scale(value, factor, boost):
    """Transform extensive mass, momentum, diagonal raw second moments."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('positive finite component mass factor required')
    boost = np.asarray(boost, float)
    if boost.shape != (3,) or not np.isfinite(boost).all():
        raise ValueError('finite three-component bulk velocity required')
    b = boost.reshape((3,) + (1,) * (value.ndim - 1))
    out = value.copy() * factor
    out[1:4] = factor * (value[1:4] + value[0] * b)
    out[4:7] = factor * (value[4:7] + 2 * b * value[1:4] + b*b * value[0])
    return out


def transport(cell, moments, shift_cells, n):
    """Continuous cell-volume overlap remap; reject loss through patch faces."""
    shift = np.asarray(shift_cells, float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError('finite three-component displacement required')
    integer = np.floor(shift).astype(int)
    fraction = shift - integer
    keys, values = [], []
    for corner in product((0, 1), repeat=3):
        corner = np.asarray(corner)
        weight = np.prod(np.where(corner, fraction, 1 - fraction))
        if weight == 0:
            continue
        target = cell + integer + corner
        if np.any(target < 0) or np.any(target >= n):
            raise ValueError('transport exits patch; enlarge source, do not wrap LG')
        keys.append(np.ravel_multi_index(target.T, (n,) * 3))
        values.append(moments * weight)
    key, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    value = np.concatenate(values, axis=1)
    out = np.array([np.bincount(inverse, weights=row, minlength=len(key)) for row in value])
    return np.array(np.unravel_index(key, (n,) * 3)).T, out


def restrict(value, ratio):
    n = value.shape[-1]
    if n % ratio:
        raise ValueError('integer restriction required')
    c = n // ratio
    return value.reshape(7, c, ratio, c, ratio, c, ratio).sum(axis=(2, 4, 6))


def check_realizable(value, rtol=1e-7):
    """Check positive mass and nonnegative component velocity variances."""
    flat = value.reshape(7, -1)
    if not np.isfinite(flat).all() or np.any(flat[0] < 0):
        raise ValueError('nonfinite or negative mass moments')
    valid = flat[0] > 0
    if np.any(flat[1:, ~valid] != 0):
        raise ValueError('empty cells must carry zero extensive moments')
    mean = flat[1:4, valid] / flat[0, valid]
    second = flat[4:7, valid] / flat[0, valid]
    relative_deficit = np.maximum(mean**2 - second, 0) / np.maximum(np.maximum(abs(second), mean**2), 1)
    maximum = float(np.max(relative_deficit, initial=0))
    if maximum > rtol:
        raise ValueError(f'nonrealizable velocity moments: relative deficit {maximum}')
    return maximum


class ContinuousMatter:
    """Joint 7-mark/component state with positive remainder and fixed patch M,P.

    theta[:, :3]: shift cMpc/h; theta[:, 3]: log bound-mass factor;
    theta[:, 4:]: COM-velocity boost km/s. No probability law is implied.
    Only explicitly modeled component marks are returned. Native host M200c
    and unrelated catalogue objects MUST NOT be carried over to a new state.
    """
    def __init__(self, total, components, *, dx):
        self.n = total.shape[-1]
        if total.shape != (7, self.n, self.n, self.n) or dx <= 0:
            raise ValueError('7 cubic moments and positive cell size required')
        check_realizable(total)
        self.dx = dx
        self.components = components
        self.total = total.sum(axis=(1, 2, 3))
        self.remainder = total.copy()
        touched = []
        for comp in components:
            cell, value = comp['cell'], comp['moments']
            check_realizable(value)
            key = np.ravel_multi_index(cell.T, (self.n,) * 3)
            if len(np.unique(key)) != len(key):
                raise ValueError('coalesce each component before construction')
            self.remainder.reshape(7, -1)[:, key] -= value
            touched.append(key)
        # Subtraction of a separately summed native particle subset can leave
        # roundoff in cells containing ONLY that subset. Zero only when ALL
        # seven residuals are within 1e-10 of original extensive scales.
        keys = np.unique(np.concatenate(touched))
        original = total.reshape(7, -1)[:, keys]
        residual = self.remainder.reshape(7, -1)[:, keys]
        scales = np.maximum(np.abs(original), 1.)
        empty = np.all(np.abs(residual) <= 1e-10 * scales, axis=0)
        self.roundoff_removed = residual[:, empty].sum(axis=1)
        self.remainder.reshape(7, -1)[:, keys[empty]] = 0
        self.remainder_variance_deficit = check_realizable(self.remainder)
        self.remaining_total = self.remainder.sum(axis=(1, 2, 3))
        if self.remaining_total[0] <= 0:
            raise ValueError('a positive unassigned matter reservoir is required')

    def state(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (len(self.components), 7) or not np.isfinite(theta).all():
            raise ValueError('finite component-by-7 parameters required')
        moved, marks = [], []
        for comp, row in zip(self.components, theta):
            value = boost_scale(comp['moments'], np.exp(row[3]), row[4:])
            cell, value = transport(comp['cell'], value, row[:3]/self.dx, self.n)
            moved.append((cell, value))
            sums = value.sum(axis=1)
            marks.append(dict(source_subhalo_id=comp['subhalo_id'],
                position_cMpc_h=np.asarray(comp['position_cMpc_h']) + row[:3],
                member_mass_Msun=float(sums[0]), peculiar_velocity_km_s=sums[1:4]/sums[0],
                resolved_halos=False, host_M200c_Msun=None,
                operator='transported_native_membership_not_new_SUBFIND_or_dynamical_halo'))
        halo_total = sum(value.sum(axis=1) for _, value in moved)
        remaining_mass = self.total[0] - halo_total[0]
        alpha = remaining_mass / self.remaining_total[0]
        if alpha <= 0:
            raise ValueError('components exhaust patch mass; no negative compensation')
        boost = (self.total[1:4] - halo_total[1:4])/remaining_mass - self.remaining_total[1:4]/self.remaining_total[0]
        return moved, alpha, boost, marks

    def evaluate(self, theta, *, ratio=1):
        moved, alpha, boost, marks = self.state(theta)
        base = self.remainder if ratio == 1 else restrict(self.remainder, ratio)
        out = boost_scale(base, alpha, boost)
        n = self.n // ratio
        for cell, value in moved:
            key = np.ravel_multi_index((cell // ratio).T, (n,) * 3)
            for k in range(7):
                np.add.at(out.reshape(7, -1)[k], key, value[k])
        return out, marks, dict(remainder_mass_factor=float(alpha), remainder_boost_km_s=boost)

    def selected(self, theta, keys):
        """Evaluate only specified fine cells for a bounded fitting experiment."""
        keys = np.asarray(keys)
        if not np.array_equal(keys, np.unique(keys)):
            raise ValueError('sorted distinct fine-cell keys required')
        moved, alpha, boost, _ = self.state(theta)
        out = boost_scale(self.remainder.reshape(7, -1)[:, keys], alpha, boost)
        for cell, value in moved:
            key = np.ravel_multi_index(cell.T, (self.n,) * 3)
            dest = np.searchsorted(keys, key)
            keep = dest < len(keys)
            keep[keep] &= keys[dest[keep]] == key[keep]
            out[:, dest[keep]] += value[:, keep]
        return out
