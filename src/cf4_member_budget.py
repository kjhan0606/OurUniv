"""Necessary union moment budgets, not a sufficient member allocation or likelihood."""
from itertools import product
import numpy as np
from cf4_continuous_matter import boost_scale, check_realizable

ROLES = ('MW', 'M31', 'M33')
PASS = 'NOT_EXCLUDED_BY_NECESSARY_TEST'
FAIL = 'EXCLUDED_BY_NECESSARY_TEST'
UNAVAILABLE = 'UNAVAILABLE'
TOL = 1e-10


def support_keys(position, radius, *, dx, n):
    """One cell-center convention for mock construction AND field evaluation."""
    p = np.asarray(position, float)
    if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite([radius, dx]).all() or min(radius, dx) <= 0:
        raise ValueError('finite three-position and positive radius/dx required')
    if np.any(p-radius < 0) or np.any(p+radius > n*dx):
        return None  # No clipped supports or periodic wrapping of the local patch.
    lo = np.ceil((p-radius)/dx-.5).astype(int)
    hi = np.floor((p+radius)/dx-.5).astype(int)
    cells = np.array(list(product(*(range(a, b+1) for a, b in zip(lo, hi)))), dtype=int).reshape(-1, 3)
    keep = np.linalg.norm((cells+.5)*dx-p, axis=1) <= radius
    cells = cells[keep]
    return np.sort(np.ravel_multi_index(cells.T, (n,)*3))


def check_union_budget(total, masses, velocities, *, reference_velocity):
    """PSD of remaining mass/momentum plus unconstrained member internal Q."""
    total, m, v = np.asarray(total, float), np.asarray(masses, float), np.asarray(velocities, float)
    ref = np.asarray(reference_velocity, float)
    if (total.shape != (7,) or m.ndim != 1 or v.shape != (len(m), 3) or ref.shape != (3,)
            or not all(np.isfinite(a).all() for a in (total, m, v, ref)) or np.any(m < 0)):
        raise ValueError('invalid finite union budget inputs')
    centered = boost_scale(total, 1., -ref)
    dv = v-ref
    demand = np.r_[m.sum(), (m[:, None]*dv).sum(0), (m[:, None]*dv**2).sum(0)]
    residual = centered-demand
    mref = max(float(total[0]), float(m.sum()), 1.)
    diagonal = np.array([1/np.sqrt(mref), 1/(300*np.sqrt(mref))])
    matrices = np.zeros((3, 2, 2))
    matrices[:, 0, 0] = residual[0]
    matrices[:, 0, 1] = matrices[:, 1, 0] = residual[1:4]
    matrices[:, 1, 1] = residual[4:7]
    normalized = matrices*diagonal[None, :, None]*diagonal[None, None, :]
    eigen = np.linalg.eigvalsh(normalized)[:, 0]
    mass_ok = bool(residual[0]/mref >= -TOL)
    return dict(mass_status=PASS if mass_ok else FAIL,
        moment_status=PASS if mass_ok and np.all(eigen >= -TOL) else FAIL,
        total_BOX_moments=total.tolist(), total_centered_moments=centered.tolist(),
        demanded_centered_bulk_moments=demand.tolist(), residual_lower_envelope=residual.tolist(),
        mass_reference_Msun=mref, normalized_mass_slack=float(residual[0]/mref),
        normalized_min_eigenvalues=eigen.tolist(), failed_axes=np.flatnonzero(eigen < -TOL).tolist())


def evaluate(cell_keys, field_moments, members, *, dx, n):
    """Members contain supplied positions/radii/masses/COMs, never native masks/IDs."""
    keys = np.asarray(cell_keys, int)
    values = np.asarray(field_moments, float)
    if set(members) != set(ROLES) or values.shape != (7, len(keys)) or not np.array_equal(keys, np.unique(keys)):
        raise ValueError('three named records and unique sorted seven-moment cells required')
    check_realizable(values)
    supports = {}
    for role in ROLES:
        record = members[role]
        supports[role] = support_keys(record['position_cMpc_h'], record['radius_cMpc_h'], dx=dx, n=n)
    ref = np.asarray(members['MW']['velocity_km_s'], float)
    records = []
    for mask in range(1, 8):
        roles = [role for i, role in enumerate(ROLES) if mask & (1 << i)]
        if (not np.isfinite(ref).all() or any(supports[r] is None or len(supports[r]) == 0
                or not np.isfinite(members[r]['mass_Msun']) or members[r]['mass_Msun'] <= 0
                or not np.isfinite(members[r]['velocity_km_s']).all() for r in roles)):
            records.append(dict(roles=roles, mask=mask, mass_status=UNAVAILABLE, moment_status=UNAVAILABLE))
            continue
        union = np.unique(np.concatenate([supports[r] for r in roles]))
        index = np.searchsorted(keys, union)
        if np.any(index >= len(keys)) or not np.array_equal(keys[index], union):
            raise ValueError('required support cells missing from supplied field; not scientific rejection')
        total = values[:, index].sum(1)  # UNION, not the sum of overlapping apertures.
        result = check_union_budget(total, [members[r]['mass_Msun'] for r in roles],
            [members[r]['velocity_km_s'] for r in roles], reference_velocity=ref)
        records.append(dict(roles=roles, mask=mask, union_cell_count=len(union),
            sum_individual_cell_counts=sum(len(supports[r]) for r in roles), **result))
    summary = {}
    for label, selected in (('hosts', records[:3]), ('all_three', records)):
        for mode in ('mass', 'moment'):
            states = [r[mode+'_status'] for r in selected]
            summary[label+'_'+mode] = UNAVAILABLE if UNAVAILABLE in states else FAIL if FAIL in states else PASS
    return dict(summary=summary, unions=records, reference_velocity_km_s=ref.tolist(),
                limits='Necessary only; residual Q includes member internal dispersion, not a unique remainder')
