#!/usr/bin/env python
"""Post-processing for OPFoF catalogs: spherical-overdensity masses and unbinding.

OPFoF returns pure friends-of-friends groups (percolation): no unbinding, no SO
masses. These are the two most useful additions and they need no change to the C
code -- they operate on the halo centres plus the particle set.

  so_mass(centers, pos_all, m_p, L)  -> R_Delta, M_Delta (Delta=200c, 500c, ...)
  unbind(pos, vel, m_p)              -> boolean mask of gravitationally bound members

SO masses fix the known FoF mass bias (percolation over-links); unbinding removes
particles a bridge falsely linked in. Use them on the fof_opfof / fof.fof catalog.
"""
import numpy as np

RHO_CRIT = 2.775e11    # rho_crit,0 in (Msun/h)/(Mpc/h)^3
G_KMS = 4.30091e-9     # G in Mpc (km/s)^2 / Msun  (for unbinding potentials)


def so_mass(centers, pos_all, m_p, L, deltas=(200.0, 500.0), ref="crit",
            Om=0.31, R_max=5.0, recenter=None, verbose=False):
    """Spherical-overdensity radius/mass around each halo centre.

    For every centre, grow a sphere until the mean enclosed density drops below
    Delta * rho_ref, where rho_ref = rho_crit,0 (ref='crit') or Om*rho_crit,0
    (ref='mean'). Equal-mass particles: M(<r) = N(<r) * m_p.

    centers  : (Nh,3) halo centres [h^-1Mpc], box frame [0,L)
    pos_all  : (Np,3) all particle positions [h^-1Mpc]  (SO spheres reach beyond
               FoF membership, so the full particle set is required)
    m_p      : particle mass [Msun/h]
    L        : periodic box size [h^-1Mpc]
    deltas   : overdensity thresholds (e.g. 200c, 500c)
    ref      : 'crit' -> Delta*rho_crit ; 'mean' -> Delta*Om*rho_crit
    R_max    : maximum search radius [h^-1Mpc]; raise for very massive clusters
    recenter : None | 'densest' -> shift each centre onto its densest neighbour
               (robust for FoF COMs pulled off-peak by bridges)

    Returns dict: R[delta], M[delta] arrays (Nh,), plus 'centers' actually used.
    NaN where the profile never reaches the threshold within R_max.
    """
    from scipy.spatial import cKDTree
    centers = np.mod(np.asarray(centers, np.float64), L)
    pos_all = np.ascontiguousarray(pos_all, np.float64)
    tree = cKDTree(np.mod(pos_all, L), boxsize=L)
    Nh = centers.shape[0]
    rho = {d: d * (Om * RHO_CRIT if ref == "mean" else RHO_CRIT) for d in deltas}
    R = {d: np.full(Nh, np.nan) for d in deltas}
    M = {d: np.full(Nh, np.nan) for d in deltas}

    # all neighbours within R_max for every centre at once
    nb = tree.query_ball_point(centers, R_max, workers=-1)
    used = centers.copy()
    for h in range(Nh):
        idx = np.asarray(nb[h], dtype=np.int64)
        if idx.size == 0:
            continue
        c = centers[h]
        if recenter == "densest" and idx.size:
            # densest = the neighbour with the most companions within 0.1 R_max
            sub = tree.query_ball_point(pos_all[idx], 0.1 * R_max, workers=-1)
            c = pos_all[idx[np.argmax([len(s) for s in sub])]]
            used[h] = c
            idx = np.asarray(tree.query_ball_point(c, R_max), np.int64)
        dr = pos_all[idx] - c
        dr -= L * np.round(dr / L)                       # minimum image
        r = np.sort(np.sqrt((dr ** 2).sum(1)))
        r = r[r > 0]
        if r.size == 0:
            continue
        n_cum = np.arange(1, r.size + 1)
        dens = n_cum * m_p / (4.0 / 3.0 * np.pi * r ** 3)
        for d in deltas:
            below = dens < rho[d]
            if below.any():
                k = np.argmax(below)                     # first radius below threshold
                if k > 0:
                    R[d][h] = r[k - 1]
                    M[d][h] = n_cum[k - 1] * m_p
        if verbose and h < 3:
            print(f"[so] halo {h}: N(<{R_max})={idx.size} "
                  + " ".join(f"M{int(d)}={M[d][h]:.2e}" for d in deltas))
    out = {"centers": used}
    for d in deltas:
        out[f"R{int(d)}{ref[0]}"] = R[d]
        out[f"M{int(d)}{ref[0]}"] = M[d]
    return out


def unbind(pos, vel, m_p, L=None, soft=None, max_iter=10, verbose=False):
    """Iterative gravitational unbinding of a halo's member particles.

    Remove particles with kinetic energy above the escape energy in the group's
    potential, recompute the bulk velocity and potential, repeat until stable.
    Direct O(n^2) potential -- intended for FoF groups up to a few 1e4 members.

    pos  : (n,3) member positions [h^-1Mpc]        vel : (n,3) [km/s]
    m_p  : particle mass [Msun/h]
    L    : box size for minimum-image (None -> assume compact group, no wrap)
    soft : Plummer softening [h^-1Mpc] (default = 0.05 * median pair sep)

    Returns boolean mask (n,) of bound particles.
    """
    pos = np.asarray(pos, np.float64); vel = np.asarray(vel, np.float64)
    n = pos.shape[0]
    bound = np.ones(n, bool)
    if n < 2:
        return bound
    if soft is None:
        c0 = pos.mean(0)
        soft = 0.05 * np.median(np.sqrt(((pos - c0) ** 2).sum(1)) + 1e-6)
    for it in range(max_iter):
        p = pos[bound]; v = vel[bound]
        m = p.shape[0]
        if m < 2:
            break
        # pairwise potential (direct sum), minimum image if periodic
        d = p[:, None, :] - p[None, :, :]
        if L is not None:
            d -= L * np.round(d / L)
        rij = np.sqrt((d ** 2).sum(-1) + soft ** 2)
        np.fill_diagonal(rij, np.inf)
        phi = -G_KMS * m_p * (1.0 / rij).sum(1)          # (km/s)^2, potential per particle
        vbulk = v.mean(0)
        ke = 0.5 * ((v - vbulk) ** 2).sum(1)             # (km/s)^2
        bnd_sub = ke + phi < 0.0                         # bound: E = KE + PE < 0
        new = bound.copy()
        new[np.flatnonzero(bound)] = bnd_sub
        if verbose:
            print(f"[unbind] iter {it}: {bnd_sub.sum()}/{m} bound")
        if bnd_sub.all() or new.sum() == bound.sum():
            bound = new
            break
        bound = new
    return bound


def add_so_masses(cat, pos_all, m_p, L, deltas=(200.0, 500.0), ref="crit",
                  Om=0.31, R_max=5.0, recenter="densest", verbose=False):
    """Attach SO masses to an OPFoF/fof catalog dict in place; returns it."""
    so = so_mass(cat["pos"], pos_all, m_p, L, deltas=deltas, ref=ref, Om=Om,
                 R_max=R_max, recenter=recenter, verbose=verbose)
    for k, v in so.items():
        if k != "centers":
            cat[k] = v
    cat["so_center"] = so["centers"]
    return cat


if __name__ == "__main__":
    # self-test: a single NFW-like blob should give a sensible M200c
    rng = np.random.default_rng(0)
    L = 50.0; m_p = 1e10
    n = 5000
    r = rng.gamma(2.0, 0.3, n)                 # concentrated blob
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    blob = 25.0 + d * r[:, None]
    bg = rng.uniform(0, L, (20000, 3))
    pos = np.mod(np.concatenate([blob, bg]), L)
    out = so_mass(np.array([[25.0, 25, 25]]), pos, m_p, L, deltas=(200, 500),
                  recenter="densest", verbose=True)
    print("R200c=%.3f M200c=%.3e  R500c=%.3f M500c=%.3e"
          % (out["R200c"][0], out["M200c"][0], out["R500c"][0], out["M500c"][0]))
    assert out["M200c"][0] > out["M500c"][0] > 0, "M200c should exceed M500c"
    print("[selftest] PASS")
