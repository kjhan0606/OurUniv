#!/usr/bin/env python
"""Condition 3 re-check: is the Local Void underdense (correct position + size) in the
VELOCITY-ONLY reconstruction? (No delta-constraints -- an independent test.)

Fable flagged that our earlier void check used the anti-Virgo direction (SGL 282.9, SGB +2.3),
~88 deg from the real Tully+2019 Local Void (dominantly +SGZ: SGL~78, SGB~+74, ~23 Mpc).
Here we forward the velocity-only HR field and measure 1+delta at the CORRECT void position,
plus a radial profile to gauge the void's SIZE (LG-constraints memo: ~30-45 Mpc), and Virgo
(anti-void) as a positive control.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_pipeline import make_forward, RHO_CRIT
from cf4_make_ic import embed_ic

VOID_SG = (78.0, 74.0, 23.0)     # Tully+2019 Local Void (SGL, SGB, dist[Mpc]) -- +SGZ dominated
VIRGO_SG = (102.9, -2.3, 16.5)   # positive control (anti-void)


def sgdir(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="recon/cf4_map_cf4_real192_hr.npz")
    ap.add_argument("--key", default="s_out", help="s_out=velocity-only HR realization")
    ap.add_argument("--Nfine", type=int, default=576)
    ap.add_argument("--out", default="recon/cf4_void_verify.png")
    args = ap.parse_args()
    import jax.numpy as jnp
    import time

    z = np.load(args.recon)
    s = z[args.key].astype(np.float64); Nc = int(z["N"]); spc = float(z["spacing"])
    L = Nc * spc; c = L / 2.0; H = float(z["hh"]) if "hh" in z else 0.746
    sp = L / args.Nfine
    print(f"[void] {args.recon} key={args.key} L={L:.0f} h={H}", flush=True)

    s_fine = embed_ic(s, args.Nfine, 1)
    t0 = time.time()
    conf, cosmo, fwd = make_forward(args.Nfine, sp, jnp.float32, return_dens=False)
    ptcl = fwd(jnp.asarray(s_fine.reshape(args.Nfine, args.Nfine, args.Nfine)))
    pos = np.asarray(ptcl.pos()).astype(np.float64)
    print(f"[void] forward {pos.shape[0]} ptcl in {time.time()-t0:.0f}s", flush=True)
    nbar = pos.shape[0] / L**3

    def opd(center, R):
        return np.sum(np.linalg.norm(pos - center, axis=1) < R) / (nbar * 4/3*np.pi*R**3)

    void = c + sgdir(*VOID_SG[:2]) * (VOID_SG[2]*H)
    virgo = c + sgdir(*VIRGO_SG[:2]) * (VIRGO_SG[2]*H)
    print(f"[void] Local Void @ SGL{VOID_SG[0]} SGB{VOID_SG[1]} d={VOID_SG[2]*H:.1f} h^-1Mpc "
          f"(box {void[0]:.0f},{void[1]:.0f},{void[2]:.0f})", flush=True)

    # spheres of increasing R at the void (underdense expected); Virgo as positive control
    print("[void] 1+delta in spheres (underdense<1 at void, overdense>1 at Virgo):", flush=True)
    for R in (5, 8, 12, 16):
        print(f"   R={R:>2} h^-1Mpc:  void={opd(void, R):.2f}   Virgo={opd(virgo, R):.2f}", flush=True)

    # radial profile from the void centre outward -> gauge the void SIZE
    Rs = np.arange(2, 42, 2.0)
    prof = np.array([opd(void, R) for R in Rs])
    # cumulative profile: find radius where it first exceeds 1 (void edge)
    edge = Rs[np.argmax(prof > 1.0)] if np.any(prof > 1.0) else Rs[-1]
    print(f"[void] radial 1+delta(<R): " +
          " ".join(f"{R:.0f}:{p:.2f}" for R, p in zip(Rs[:12], prof[:12])), flush=True)
    print(f"[void] void radius (1+delta first >1) ~ {edge:.0f} h^-1Mpc "
          f"(LG-constraints target ~30-45 Mpc / h={H} -> ~22-34 h^-1Mpc)", flush=True)

    # direction test: is the void the LOCAL MINIMUM, or did we just hit a random low spot?
    # sample 12 random directions at 23 Mpc and rank the void among them.
    rng = np.random.default_rng(0)
    dirs = rng.standard_normal((200, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    dvals = np.array([opd(c + d*(VOID_SG[2]*H), 8.0) for d in dirs])
    void8 = opd(void, 8.0)
    pctile = 100.0 * np.mean(dvals > void8)
    print(f"[void] void 1+delta(<8)={void8:.2f} is at the {100-pctile:.0f}th percentile "
          f"of 200 random directions at {VOID_SG[2]*H:.0f} h^-1Mpc "
          f"(low percentile = genuinely underdense, not chance)", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(Rs, prof, "o-", color="steelblue", label="Local Void (correct SGZ+ position)")
    virprof = np.array([opd(virgo, R) for R in Rs])
    ax.plot(Rs, virprof, "s-", color="firebrick", label="Virgo (positive control)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axhline(np.median(dvals), color="gray", lw=0.8, ls=":", label="median random dir")
    ax.set_xlabel("sphere radius R [$h^{-1}$Mpc]"); ax.set_ylabel(r"$1+\delta(<R)$")
    ax.set_yscale("log"); ax.legend()
    ax.set_title(f"Condition 3: Local Void in the VELOCITY-ONLY field\n"
                 f"void 1+d(<8)={void8:.2f} ({100-pctile:.0f}th pctile of random dirs)")
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[void] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
