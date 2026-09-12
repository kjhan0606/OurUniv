"""Conservative binary-tree coordinates for seven native extensive moments.

Seven binary splits parameterize an octet. Each split has a mass fraction,
three relative bulk velocities, and three internal-variance allocations.
State codes:0 inactive,1 lower atom,2 continuous interior,3 upper atom.
Only floating-point cancellation at64 eps of raw second moments is cold.
No density floor; scalar physical sigma is never the stored representation.
"""
import numpy as np
from scipy.special import expit
from cf4_continuous_matter import check_realizable

NODES = [(0, 8), (0, 4), (4, 8), (0, 2), (2, 4), (4, 6), (6, 8)]


def state(x):
    v = np.divide(x[1:4], x[0], out=np.zeros_like(x[1:4]), where=x[0] > 0)
    q = np.divide(x[4:], x[0], out=np.zeros_like(x[4:]), where=x[0] > 0)
    variance = np.maximum(q-v*v, 0)
    roundoff = 64*np.finfo(np.float64).eps*np.maximum(np.maximum(abs(q), v*v), 1)
    variance[variance <= roundoff] = 0
    return v, variance


def octets(x):
    n = x.shape[-1]//2
    return x.reshape(7, n, 2, n, 2, n, 2).transpose(2, 4, 6, 0, 1, 3, 5).reshape(8, 7, n, n, n)


def merge_octets(x):
    n = x.shape[-1]
    return x.reshape(2, 2, 2, 7, n, n, n).transpose(3, 4, 0, 5, 1, 6, 2).reshape(7, 2*n, 2*n, 2*n)


def fraction_encode(left, right, enabled):
    s = np.zeros(left.shape, dtype=np.uint8)
    s[enabled & (left == 0)] = 1
    s[enabled & (right == 0)] = 3
    active = enabled & (left > 0) & (right > 0)
    s[active] = 2
    z = np.zeros_like(left)
    z[active] = np.log(left[active])-np.log(right[active])
    return z, s


def encode(left, right):
    parent = left+right
    z = np.zeros_like(parent)
    mask = np.zeros(parent.shape, dtype=np.uint8)
    z[0], mask[0] = fraction_encode(left[0], right[0], parent[0] > 0)
    w = np.divide(left[0], parent[0], out=np.zeros_like(parent[0]), where=parent[0] > 0)
    wr = np.divide(right[0], parent[0], out=np.zeros_like(parent[0]), where=parent[0] > 0)
    lv, lvar = state(left)
    rv, rvar = state(right)
    _, pv = state(parent)
    for axis in range(3):
        enabled = (mask[0] == 2) & (pv[axis] > 0)
        internal = w*lvar[axis]+wr*rvar[axis]
        dv = lv[axis]-rv[axis]
        between = w*wr*dv**2
        # Stable equivalent of atanh(r), using the separately calculated
        # internal energy instead of subtracting near-equal total energies.
        interior = enabled & (internal > 0)
        z[1+axis, interior] = np.arcsinh(dv[interior]*np.sqrt(w[interior]*wr[interior]/internal[interior]))
        mask[1+axis, interior] = 2
        boundary = enabled & (internal == 0) & (between > 0)
        mask[1+axis, boundary & (dv < 0)] = 1
        mask[1+axis, boundary & (dv >= 0)] = 3
        if np.any(enabled & ~interior & ~boundary):
            raise ValueError('inconsistent cold/between variance branch')
        z[4+axis], mask[4+axis] = fraction_encode(w*lvar[axis], wr*rvar[axis], interior)
    return z, mask, parent


def fractions(z, s):
    left, right = expit(z), expit(-z)
    left = np.where(s == 1, 0, np.where(s == 3, 1, left))
    right = np.where(s == 1, 1, np.where(s == 3, 0, right))
    if np.any((s == 2) & ((left == 0) | (right == 0))):
        raise ValueError('continuous fraction exceeds float64 support; no floor')
    return left, right


def decode(z, mask, parent):
    if not np.isfinite(z).all():
        raise ValueError('nonfinite split coordinates')
    w, wr = fractions(z[0], mask[0])
    w = np.where(parent[0] > 0, w, 0)
    wr = np.where(parent[0] > 0, wr, 0)
    left, right = np.zeros_like(parent), np.zeros_like(parent)
    left[0], right[0] = parent[0]*w, parent[0]*wr
    mean, variance = state(parent)
    both = (w > 0) & (wr > 0)
    for axis in range(3):
        r = np.tanh(z[1+axis])
        exponential = np.exp(-2*abs(z[1+axis]))
        remain = 4*exponential/(1+exponential)**2
        s = mask[1+axis]
        r = np.where(s == 1, -1, np.where(s == 3, 1, np.where(s == 0, 0, r)))
        remain = np.where((s == 1) | (s == 3), 0, np.where(s == 0, 1, remain))
        a, ar = fractions(z[4+axis], mask[4+axis])
        base = variance[axis]
        ul = np.sqrt(np.divide(base*wr, w, out=np.zeros_like(w), where=both))*r
        ur = -np.sqrt(np.divide(base*w, wr, out=np.zeros_like(w), where=both))*r
        vl, vr = mean[axis]+ul, mean[axis]+ur
        sl = np.divide(base*remain*a, w, out=np.zeros_like(w), where=both)
        sr = np.divide(base*remain*ar, wr, out=np.zeros_like(w), where=both)
        # A single populated child inherits the entire parent state exactly.
        sl = np.where((w > 0) & ~both, base, sl)
        sr = np.where((wr > 0) & ~both, base, sr)
        left[1+axis], right[1+axis] = left[0]*vl, right[0]*vr
        left[4+axis], right[4+axis] = left[0]*(vl*vl+sl), right[0]*(vr*vr+sr)
    return left, right


def encode_tree(fine):
    children = octets(fine)
    records = []
    for lo, hi in NODES:
        middle = (lo+hi)//2
        records.append(encode(children[lo:middle].sum(axis=0), children[middle:hi].sum(axis=0)))
    return records


def decode_tree(records, root):
    tree = {(0, 8): root}
    for (lo, hi), (z, mask) in zip(NODES, records):
        middle = (lo+hi)//2
        tree[(lo, middle)], tree[(middle, hi)] = decode(z, mask, tree[(lo, hi)])
    return merge_octets(np.array([tree[(i, i+1)] for i in range(8)]))


def roundtrip(fine, report=False):
    check_realizable(fine)
    records = encode_tree(fine)
    restored = decode_tree([(z, s) for z, s, p in records], records[0][2])
    check_realizable(restored)
    scales = np.maximum(np.max(abs(fine), axis=(1, 2, 3)), 1)
    error = np.max(abs(restored-fine), axis=(1, 2, 3))/scales
    if np.max(error) > 1e-9:
        raise ValueError(f'native7 moment roundtrip failed: {error}')
    v, variance = state(fine)
    rv, rvariance = state(restored)
    total = fine[0].sum()
    weights = fine[0]/total if total > 0 else np.zeros_like(fine[0])
    velocity_error = np.sqrt(np.sum(weights*(rv-v)**2, axis=(1, 2, 3)))
    sigma_error = np.sqrt(np.sum(weights*(np.sqrt(rvariance)-np.sqrt(variance))**2, axis=(1, 2, 3)))
    if max(velocity_error.max(), sigma_error.max()) > 1e-4:
        raise ValueError(f'native velocity/dispersion roundtrip failed: {velocity_error}, {sigma_error}')
    if report:
        return dict(moment_relative_error=error.tolist(),
                    velocity_RMSE_km_s=velocity_error.tolist(), physical_sigma_RMSE_km_s=sigma_error.tolist())
    return error
