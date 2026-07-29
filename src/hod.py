#!/usr/bin/env python
"""Halo Occupation Distribution (HOD) galaxy population -- numpy/scipy only.

Populates a halo catalog (e.g. from fof.py) with mock galaxies using the
Zheng et al. (2007) 5-parameter HOD:

    <N_cen>(M) = 1/2 [ 1 + erf( (log10 M - log10 Mmin) / sigma_logM ) ]
    <N_sat>(M) = <N_cen>(M) * ( (M - M0) / M1 )^alpha ,   M > M0  else 0

Per halo we draw:
    central   : Bernoulli(<N_cen>)                 at halo COM, halo velocity
    satellites: Poisson(<N_sat>)  (only if a central is present)

Satellite phase-space -- two modes:
  * mode="particles" (default when member particles are supplied): each satellite
    IS a randomly chosen member particle of the host halo -> position and
    velocity are drawn from the simulation's own phase-space (fully
    self-consistent, includes real substructure and velocity bias).
  * mode="nfw": analytic -- radius from an NFW profile with a c(M) relation,
    isotropic direction, velocity = halo bulk + Gaussian(sigma_v). Uses the
    halo's measured sigv if available, else the virial estimate. Needs only the
    halo catalog (mass, pos, vel[, sigv]).

Peculiar velocity: every galaxy carries a 3D velocity (same code/physical units
as the input halo/particle velocities). The line-of-sight peculiar velocity for
a given observer is  v_pec = v_gal . r_hat  -- computed downstream.

Returns a galaxy catalog (dict of arrays):
    pos    (Ng,3)   position in [0,L)          [box units]
    vel    (Ng,3)   velocity                   [input vel units]
    host   (Ng,)    index into the halo catalog
    is_cen (Ng,)    bool, True for centrals
    mass   (Ng,)    host halo mass             [same units as halo mass]

CLI self-test:  python hod.py --selftest
"""
import numpy as np
from scipy.special import erf

RHO_CRIT = 2.775e11   # Msun/h per (Mpc/h)^3
G_KMS = 4.30091e-9    # (km/s)^2 Mpc / Msun  (h cancels in G*M[Msun/h]/R[Mpc/h])

# a reasonable luminous-galaxy default (Zheng+07 Mr<-21-ish), log10 masses in Msun/h
DEFAULT_HOD = dict(logMmin=12.6, sigma_logM=0.4, logM0=12.0, logM1=13.7, alpha=1.0)


def n_cen(mass, logMmin, sigma_logM):
    return 0.5 * (1.0 + erf((np.log10(mass) - logMmin) / sigma_logM))


def n_sat(mass, logMmin, sigma_logM, logM0, logM1, alpha):
    M0 = 10.0 ** logM0
    M1 = 10.0 ** logM1
    x = np.clip((mass - M0) / M1, 0.0, None)
    return n_cen(mass, logMmin, sigma_logM) * x ** alpha


def _concentration(mass):
    """Dutton & Maccio (2014) c200-M relation at z=0 (mass in Msun/h)."""
    return 10.0 ** (0.905 - 0.101 * (np.log10(mass) - 12.0))


def _r_vir(mass, Om=0.31, delta=200.0):
    """Virial radius [Mpc/h] for Delta*rho_mean overdensity."""
    rho_m = Om * RHO_CRIT
    return (3.0 * mass / (4.0 * np.pi * delta * rho_m)) ** (1.0 / 3.0)


def _nfw_radii(rng, m_sat, rvir, conc):
    """Sample satellite radii (per satellite) from an NFW profile, r<rvir."""
    # inverse-CDF of NFW enclosed mass m(x)=ln(1+cx)-cx/(1+cx), x=r/rvir in [0,1]
    xs = np.linspace(0.0, 1.0, 512)
    out = np.empty(0)
    # do it per-halo since conc varies; vectorize over satellites within a halo
    r = np.empty(m_sat.shape[0] if hasattr(m_sat, "shape") else 0)
    return r  # unused; kept for API symmetry


def populate(halos, pos_all=None, vel_all=None, params=None, mode=None,
             Om=0.31, seed=0, vel_bias=1.0, verbose=False):
    """Populate a halo catalog with HOD galaxies.

    halos   : dict from fof.fof (needs mass, pos, vel; sigv/head/member/n for
              particle mode)
    pos_all : (Np,3) all particle positions (needed for mode="particles")
    vel_all : (Np,3) all particle velocities (needed for mode="particles")
    params  : dict of HOD params (defaults to DEFAULT_HOD)
    mode    : "particles" | "nfw"  (auto: particles if pos_all/member given)
    vel_bias: multiply satellite velocity dispersion (velocity bias, default 1)
    """
    p = dict(DEFAULT_HOD)
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)

    mass = halos["mass"]
    hpos = halos["pos"]
    hvel = halos["vel"]
    L = float(halos["L"])
    Nh = mass.shape[0]

    have_members = (pos_all is not None and "member" in halos and "head" in halos)
    if mode is None:
        mode = "particles" if have_members else "nfw"
    if mode == "particles" and not have_members:
        raise ValueError('mode="particles" needs pos_all/vel_all + halo member index')

    # expected occupations
    nc = n_cen(mass, p["logMmin"], p["sigma_logM"])
    ns = n_sat(mass, p["logMmin"], p["sigma_logM"], p["logM0"], p["logM1"], p["alpha"])

    has_cen = rng.random(Nh) < nc
    n_sat_draw = rng.poisson(ns) * has_cen   # satellites only if central present

    g_pos, g_vel, g_host, g_cen, g_mass = [], [], [], [], []

    # centrals
    ci = np.flatnonzero(has_cen)
    if ci.size:
        g_pos.append(hpos[ci])
        g_vel.append(hvel[ci])
        g_host.append(ci)
        g_cen.append(np.ones(ci.size, bool))
        g_mass.append(mass[ci])

    # satellites
    if mode == "nfw":
        conc = _concentration(mass)
        rvir = _r_vir(mass, Om=Om)
        # per-halo virial 1d dispersion (fallback to catalog sigv if present & >0)
        sig_vir = np.sqrt(G_KMS * mass / (2.0 * rvir))
        sigv = halos.get("sigv", None)
        for h in np.flatnonzero(n_sat_draw):
            k = int(n_sat_draw[h])
            # radius via NFW inverse-CDF
            c = conc[h]
            xs = np.linspace(1e-4, 1.0, 1024)
            mx = np.log(1 + c * xs) - c * xs / (1 + c * xs)
            u = rng.random(k) * mx[-1]
            x = np.interp(u, mx, xs)
            r = x * rvir[h]
            # isotropic directions
            mu = rng.uniform(-1, 1, k)
            phi = rng.uniform(0, 2 * np.pi, k)
            st = np.sqrt(1 - mu ** 2)
            dirs = np.stack([st * np.cos(phi), st * np.sin(phi), mu], axis=1)
            sp = (hpos[h] + r[:, None] * dirs) % L
            s = sigv[h] if (sigv is not None and sigv[h] > 0) else sig_vir[h]
            sv = hvel[h] + rng.normal(0, s * vel_bias, size=(k, 3))
            g_pos.append(sp)
            g_vel.append(sv)
            g_host.append(np.full(k, h))
            g_cen.append(np.zeros(k, bool))
            g_mass.append(np.full(k, mass[h]))
    else:  # particles
        head = halos["head"]
        nmem = halos["n"]
        member = halos["member"]
        for h in np.flatnonzero(n_sat_draw):
            k = int(n_sat_draw[h])
            m = int(nmem[h])
            idx_local = rng.integers(0, m, size=k)     # sample members (w/ repl.)
            pidx = member[head[h] + idx_local]
            sp = pos_all[pidx] % L
            sv = vel_all[pidx]
            if vel_bias != 1.0:
                sv = hvel[h] + vel_bias * (sv - hvel[h])
            g_pos.append(sp)
            g_vel.append(sv)
            g_host.append(np.full(k, h))
            g_cen.append(np.zeros(k, bool))
            g_mass.append(np.full(k, mass[h]))

    if g_pos:
        cat = dict(
            pos=np.concatenate(g_pos).reshape(-1, 3),
            vel=np.concatenate(g_vel).reshape(-1, 3),
            host=np.concatenate(g_host).astype(np.int64),
            is_cen=np.concatenate(g_cen),
            mass=np.concatenate(g_mass),
            L=L, params=p, mode=mode,
        )
    else:
        cat = dict(pos=np.zeros((0, 3)), vel=np.zeros((0, 3)),
                   host=np.zeros(0, np.int64), is_cen=np.zeros(0, bool),
                   mass=np.zeros(0), L=L, params=p, mode=mode)
    if verbose:
        ng = cat["pos"].shape[0]
        ncen = int(cat["is_cen"].sum())
        print(f"[hod] mode={mode}  {ng} galaxies ({ncen} cen + {ng-ncen} sat) "
              f"from {Nh} halos  f_sat={1-ncen/max(ng,1):.3f}")
    return cat


def line_of_sight_vpec(gal, observer):
    """v_pec,los = v_gal . r_hat  for an observer position (3,) in box units."""
    r = gal["pos"] - np.asarray(observer)
    rn = np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
    rhat = r / rn
    return np.sum(gal["vel"] * rhat, axis=1)


# --------------------------------------------------------------------------- #
def _selftest():
    """Populate a synthetic halo mass function; check <N>(M) matches Zheng07."""
    rng = np.random.default_rng(1)
    L = 200.0
    # synthetic halo catalog: masses ~ 10^[12,15], positions random
    Nh = 40000
    logM = 12.0 + 3.0 * rng.power(0.3, Nh)   # weighted toward low mass
    mass = 10.0 ** logM
    halos = dict(mass=mass, pos=rng.uniform(0, L, (Nh, 3)),
                 vel=rng.normal(0, 300, (Nh, 3)), sigv=np.full(Nh, 200.0),
                 L=L, n=np.full(Nh, 100, np.int64))
    gal = populate(halos, params=DEFAULT_HOD, mode="nfw", seed=3, verbose=True)

    # measured <N_gal>(M) in mass bins vs analytic
    p = DEFAULT_HOD
    bins = np.logspace(12, 15, 16)
    bc = np.sqrt(bins[:-1] * bins[1:])
    counts = np.zeros(len(bc))
    nhalo = np.zeros(len(bc))
    hb = np.digitize(mass, bins) - 1
    for h in range(Nh):
        if 0 <= hb[h] < len(bc):
            nhalo[hb[h]] += 1
    gb = np.digitize(gal["mass"], bins) - 1
    for g in range(gal["pos"].shape[0]):
        if 0 <= gb[g] < len(bc):
            counts[gb[g]] += 1
    meas = counts / np.maximum(nhalo, 1)
    ana = n_cen(bc, p["logMmin"], p["sigma_logM"]) + \
        n_sat(bc, p["logMmin"], p["sigma_logM"], p["logM0"], p["logM1"], p["alpha"])
    print(f"{'logM':>6} {'N_meas':>8} {'N_theory':>9} {'Nhalo':>7}")
    okbins = 0
    for i in range(len(bc)):
        if nhalo[i] > 200:
            rel = abs(meas[i] - ana[i]) / max(ana[i], 1e-3)
            okbins += rel < 0.15
            flag = "" if rel < 0.15 else "  <-- off"
            print(f"{np.log10(bc[i]):6.2f} {meas[i]:8.3f} {ana[i]:9.3f} {int(nhalo[i]):7d}{flag}")
    print(f"[selftest] {okbins} well-populated bins match analytic <N>(M) within 15%")

    # particle-mode smoke: satellites drawn from member particles
    Np = 5000
    pos_all = rng.uniform(0, L, (Np, 3))
    vel_all = rng.normal(0, 500, (Np, 3))
    hsmall = dict(mass=np.array([1e14, 1e13]), pos=rng.uniform(0, L, (2, 3)),
                  vel=np.zeros((2, 3)), L=L, n=np.array([100, 100]),
                  head=np.array([0, 100]),
                  member=rng.integers(0, Np, 200).astype(np.int64))
    g2 = populate(hsmall, pos_all=pos_all, vel_all=vel_all, mode="particles",
                  seed=5, verbose=True)
    # line-of-sight vpec sanity
    vlos = line_of_sight_vpec(g2, observer=[L / 2] * 3)
    print(f"[selftest] particle-mode v_los rms = {np.std(vlos):.0f} km/s")
    assert okbins >= 5, "HOD occupation does not match analytic model"
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
