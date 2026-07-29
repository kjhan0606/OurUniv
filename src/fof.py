#!/usr/bin/env python
"""Friends-of-Friends (FoF) halo finder -- self-contained, numpy/scipy only.

Standard percolation FoF: two particles are "friends" if their separation is
below the linking length b_ll; a halo is a connected group of friends
(friend-of-a-friend). Implemented with a periodic KD-tree pair search
(scipy.spatial.cKDTree, which supports `boxsize`) + a union-find via
scipy.sparse.csgraph.connected_components -- O(Np + Npairs), no external deps.

Designed to run on the z=0 particle snapshot from the pmwd PM forward:
    ptcl.pos()  (Np,3) in [0, L)      -> positions
    ptcl.vel    (Np,3)                -> velocities (code units)
but works on ANY particle set (pass pos/vel/box/mean_sep).

Linking length convention: b_ll = b * mean_sep, mean_sep = L / Np^(1/3).
The canonical b = 0.2 selects ~200*rho_crit overdensity halos.

Returns a halo catalog (dict of arrays), sorted by descending mass:
    n        (Nh,)      member count
    mass     (Nh,)      n * m_particle   [Msun/h] if m_particle given, else = n
    pos      (Nh,3)     periodic center of mass       [box units]
    vel      (Nh,3)     mean member velocity          [same units as input vel]
    sigv     (Nh,)      1D velocity dispersion (rms over 3 axes / sqrt3)
    r_rms    (Nh,)      rms member distance from COM (periodic)  [box units]
    head     (Nh,)      offset into `member` for this halo's indices
    member   (sum n,)   concatenated Lagrangian particle indices (int64)
Use halo h's members:  member[head[h]:head[h]+n[h]].

CLI self-test:  python fof.py --selftest
"""
import numpy as np


def _periodic_com(pos, L):
    """Center of mass of points in a periodic box, per-axis, via circular mean.

    pos: (m,3) in [0,L).  Returns (3,) in [0,L).
    Maps x -> angle 2*pi*x/L, averages on the circle, maps back -- robust to
    halos straddling a box face.
    """
    ang = pos * (2 * np.pi / L)
    cbar = np.cos(ang).mean(axis=0)
    sbar = np.sin(ang).mean(axis=0)
    theta = np.arctan2(sbar, cbar)
    theta[theta < 0] += 2 * np.pi
    return theta * (L / (2 * np.pi))


def _min_image(dx, L):
    """Wrap displacement(s) into [-L/2, L/2)."""
    return dx - L * np.round(dx / L)


def fof(pos, vel=None, L=None, b=0.2, mean_sep=None, linking_length=None,
        n_min=20, m_particle=None, periodic=True, verbose=False):
    """Run FoF on a particle set (periodic box, or a non-periodic sub-region).

    Parameters
    ----------
    pos : (Np,3) float   positions in [0, L) (periodic) or arbitrary (non-periodic)
    vel : (Np,3) float   velocities (optional; halo vel/sigv computed if given)
    L   : float          periodic box size (required if periodic)
    b   : float          linking length in units of mean_sep (default 0.2)
    mean_sep : float     mean inter-particle separation; default L/Np**(1/3)
    linking_length : float  absolute linking length; overrides b*mean_sep
    n_min : int          minimum members for a kept halo (default 20)
    m_particle : float   mass per particle (mass = n*m_particle; else mass = n)
    periodic : bool      periodic box (default); False for an extracted sub-region
                         (open boundary, plain COM). For a 0.5 h^-1Mpc IC the full
                         box is too big for the pair search, so we FoF the central
                         sub-cube non-periodically with mean_sep set to the global
                         particle spacing.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pos = np.ascontiguousarray(pos, dtype=np.float64)
    Np = pos.shape[0]
    if periodic:
        if L is None:
            raise ValueError("L (box size) is required for periodic FoF")
        # cKDTree(boxsize=) requires 0 <= x < L strictly
        pos = np.mod(pos, L)
        pos[pos >= L] -= L  # guard exact-L after mod round-off

    if mean_sep is None:
        if L is None:
            raise ValueError("mean_sep or L required")
        mean_sep = L / Np ** (1.0 / 3.0)
    if linking_length is None:
        linking_length = b * mean_sep
    if verbose:
        print(f"[fof] Np={Np}  periodic={periodic}  mean_sep={mean_sep:.4f}  "
              f"b={b}  ll={linking_length:.4f}  n_min={n_min}")

    tree = cKDTree(pos, boxsize=L if periodic else None)
    pairs = tree.query_pairs(r=linking_length, output_type="ndarray")  # (M,2)
    if verbose:
        print(f"[fof] {len(pairs)} friend pairs")

    # union-find via connected components of the friendship graph
    if len(pairs):
        data = np.ones(len(pairs), dtype=np.int8)
        g = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(Np, Np))
        n_comp, labels = connected_components(g, directed=False)
    else:
        labels = np.arange(Np)

    # group members by label
    order = np.argsort(labels, kind="stable")
    lab_sorted = labels[order]
    boundaries = np.flatnonzero(np.diff(lab_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [Np]])
    sizes = ends - starts

    keep = np.flatnonzero(sizes >= n_min)
    # sort kept halos by descending size
    keep = keep[np.argsort(-sizes[keep], kind="stable")]

    n_arr, com, vbar, sigv, rrms, members, head = [], [], [], [], [], [], []
    off = 0
    for h in keep:
        idx = order[starts[h]:ends[h]]      # member particle indices
        m = idx.size
        p = pos[idx]
        c = _periodic_com(p, L) if periodic else p.mean(axis=0)
        dr = _min_image(p - c, L) if periodic else (p - c)
        rrms.append(np.sqrt((dr ** 2).sum(axis=1).mean()))
        n_arr.append(m)
        com.append(c)
        members.append(idx)
        head.append(off)
        off += m
        if vel is not None:
            v = vel[idx]
            vbar.append(v.mean(axis=0))
            sigv.append(np.sqrt(((v - v.mean(axis=0)) ** 2).sum(axis=1).mean() / 3.0))
        else:
            vbar.append(np.zeros(3))
            sigv.append(0.0)

    n_arr = np.array(n_arr, dtype=np.int64)
    mass = n_arr.astype(np.float64) * (m_particle if m_particle else 1.0)
    cat = dict(
        n=n_arr,
        mass=mass,
        pos=np.array(com).reshape(-1, 3),
        vel=np.array(vbar).reshape(-1, 3),
        sigv=np.array(sigv),
        r_rms=np.array(rrms),
        head=np.array(head, dtype=np.int64),
        member=(np.concatenate(members).astype(np.int64) if members
                else np.zeros(0, np.int64)),
        L=float(L), b=float(b), linking_length=float(linking_length),
        m_particle=float(m_particle) if m_particle else 1.0,
    )
    if verbose:
        print(f"[fof] {len(n_arr)} halos with >= {n_min} members "
              f"(largest {n_arr.max() if len(n_arr) else 0})")
    return cat


# --------------------------------------------------------------------------- #
def _selftest():
    """Plant Gaussian blobs in a uniform background; FoF must recover them."""
    rng = np.random.default_rng(0)
    L = 100.0
    # background
    n_bg = 20000
    bg = rng.uniform(0, L, size=(n_bg, 3))
    # 5 tight clusters (each 400 particles, sigma 1.0 Mpc)
    centers = rng.uniform(10, 90, size=(5, 3))
    blobs = []
    for c in centers:
        blobs.append(c + rng.normal(0, 1.0, size=(400, 3)))
    pos = np.concatenate([bg] + blobs, axis=0)
    vel = rng.normal(0, 100.0, size=pos.shape)
    cat = fof(pos, vel, L=L, b=0.2, n_min=20, verbose=True)
    print(f"[selftest] planted 5 clusters; FoF found {len(cat['n'])} halos "
          f"(sizes: {sorted(cat['n'], reverse=True)[:8]})")
    # match recovered halos to planted centers
    ok = 0
    for c in centers:
        d = np.sqrt((_min_image(cat["pos"] - c, L) ** 2).sum(axis=1))
        j = d.argmin()
        if d[j] < 3.0 and cat["n"][j] >= 200:
            ok += 1
    print(f"[selftest] recovered {ok}/5 planted clusters within 3 Mpc & >=200 members")
    # periodic COM test: a blob straddling the x=0 face
    strad = np.concatenate([rng.normal(0.5, 0.6, (300, 1)) % L,
                            rng.uniform(40, 60, (300, 2))], axis=1)
    strad[:150, 0] = (strad[:150, 0] - 1.0) % L   # push half across the face
    cat2 = fof(strad, L=L, b=0.2, n_min=50)
    if len(cat2["n"]):
        cx = cat2["pos"][0, 0]
        wrapped_ok = (cx < 3.0) or (cx > L - 3.0)
        print(f"[selftest] straddling-blob COM x={cx:.2f} "
              f"(expect near 0 or {L:.0f}) -> {'PASS' if wrapped_ok else 'FAIL'}")
    assert ok >= 4, "FoF failed to recover planted clusters"
    print("[selftest] PASS")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print(__doc__)
