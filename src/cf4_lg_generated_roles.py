"""Identity-free LG role hypotheses from a generated NewGalFinder catalogue.

This is diagnostic support and an observation interface, not a calibrated
selection function, posterior likelihood, or proof of galaxy identities.
"""

from itertools import permutations

import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord

from cf4_lg_observation_contract import predict_candidate


def _supported_components(children, parent, host_indices):
    parent = np.asarray(parent, dtype=np.int64)
    if len(parent) != len(children):
        raise ValueError("one generated host index per bound component required")
    masses = np.asarray(children["totm"], dtype=float)
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("positive finite bound-component masses required")
    for host in host_indices:
        if not np.any(parent == host):
            raise ValueError(f"host {host} has no bound component")
    return np.flatnonzero(np.isin(parent, host_indices)), parent


def enumerate_role_hypotheses(children, parent, host_indices,
                              *, max_hypotheses=100000):
    """Enumerate all ordered bound-component roles in declared support.

    `host_indices` must be a predeclared, generated-state spatial/resolution
    support, never native truth IDs or a target-ranked shortlist. Components
    may share one FoF host; no mass ranking assigns MW, M31 or M33. M33 may
    be another bound component or unresolved. The size cap FAILS rather than
    truncating support or silently changing the implied role prior.
    """
    hosts = tuple(int(i) for i in host_indices)
    if len(hosts) != len(set(hosts)) or any(i < 0 for i in hosts):
        raise ValueError("unique nonnegative generated host indices required")
    components, parent = _supported_components(children, parent, hosts)
    n = len(components)
    if n * (n - 1) ** 2 > max_hypotheses:
        raise ValueError("declared role support exceeds size cap; no truncation")
    hypotheses = []
    for mw, m31 in permutations(components, 2):
        mw, m31 = int(mw), int(m31)
        base = {"MW_host": int(parent[mw]), "M31_host": int(parent[m31]),
                "MW_component": mw, "M31_component": m31,
                "MW_M31_share_FoF_host": bool(parent[mw] == parent[m31])}
        hypotheses.append({**base, "M33_branch": "unresolved", "M33_component": None})
        for m33 in components:
            m33 = int(m33)
            if m33 in (mw, m31):
                continue
            host = int(parent[m33])
            branch = ("shared_FoF_host" if host in (parent[mw], parent[m31])
                      else "distinct_FoF_host")
            hypotheses.append({**base, "M33_branch": branch,
                "M33_parent_host": host, "M33_component": m33})
    return hypotheses


def predict_role_hypothesis(children, hypothesis, contract, *, h, box_mpc_h,
                            solar_position_kpc, solar_velocity_km_s):
    """Map one generated role hypothesis to observed ICRS quantities.

    Periodic separations are used only inside a common box. The unresolved
    branch deliberately returns no eight-observable likelihood prediction.
    """
    if hypothesis["M33_component"] is None:
        return {"status": "UNRESOLVED_M33", "hypothesis": hypothesis}
    if not np.isfinite([h, box_mpc_h]).all() or min(h, box_mpc_h) <= 0:
        raise ValueError("positive finite h and box size required")
    rotation = SkyCoord(CartesianRepresentation(np.eye(3)),
                        frame="supergalactic").icrs.cartesian.xyz.value
    mw_index = hypothesis["MW_component"]
    mw_pos = np.array([children[name][mw_index] for name in ("x", "y", "z")])
    mw_vel = np.array([children[name][mw_index] for name in ("vx", "vy", "vz")])
    catalog = {"frame": contract["frame"], "resolved_halos": True,
               "MW": {"position_kpc": np.zeros(3),
                      "peculiar_velocity_km_s": np.zeros(3)}}
    for name in ("M31", "M33"):
        index = hypothesis[f"{name}_component"]
        pos = np.array([children[field][index] for field in ("x", "y", "z")])
        vel = np.array([children[field][index] for field in ("vx", "vy", "vz")])
        dr = (pos - mw_pos + box_mpc_h / 2) % box_mpc_h - box_mpc_h / 2
        catalog[name] = {"position_kpc": rotation @ dr * (1000 / h),
                         "peculiar_velocity_km_s": rotation @ (vel - mw_vel)}
    prediction = predict_candidate(catalog, contract, h=h,
        solar_position_kpc=solar_position_kpc,
        solar_velocity_km_s=solar_velocity_km_s)
    return {"status": "GENERATED_ROLE_DIAGNOSTIC_ONLY", "hypothesis": hypothesis,
            **prediction}
