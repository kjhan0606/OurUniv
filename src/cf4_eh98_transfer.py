#!/usr/bin/env python
"""Replace the empirical tail extrapolation of the measured transfer with a physically
motivated Eisenstein & Hu (1998) no-wiggle CDM transfer, VALIDATED against the measured
T_eff over the overlap band (the measured T is ground truth where r(k)~1).

If the EH98 shape matches the measured T_eff (constant ratio over the band), we adopt
A*T_EH98(k) for k above the reliable measured range -> accurate dwarf-scale power.
If it does NOT match (unit/param error), we keep the empirical tail and report.
"""
import os
import sys
import argparse
import numpy as np


def T_eh98_nowiggle(k_hmpc, Om, Ob, h, Tcmb=2.7255):
    """EH98 zero-baryon 'no-wiggle' CDM transfer. k in h/Mpc. Returns T(k) (T(0)=1)."""
    theta = Tcmb / 2.7
    omh2 = Om * h * h
    obh2 = Ob * h * h
    fb = obh2 / omh2
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2 ** 0.75)          # Mpc
    ag = 1.0 - 0.328 * np.log(431.0 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb ** 2
    k_mpc = k_hmpc * h                                                           # Mpc^-1
    Gamma = Om * h * (ag + (1.0 - ag) / (1.0 + (0.43 * k_mpc * s) ** 4))         # shape [h/Mpc]
    q = k_hmpc * theta ** 2 / np.maximum(Gamma, 1e-30)
    L = np.log(2.0 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L / (L + C * q * q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/gpfs/kjhan/Hydro/CF4_LG/tier1/transfer_cr6_e19.npz")
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--Ob", type=float, default=0.0403, help="Omega_b (Ob h^2=0.0224 @ h=0.746)")
    ap.add_argument("--h", type=float, default=0.746)
    ap.add_argument("--ns", type=float, default=0.9649, help="primordial tilt; T_eff ~ k^(ns/2) T(k)")
    ap.add_argument("--out", default="/gpfs/kjhan/Hydro/CF4_LG/tier1/transfer_cr6_e19_eh98.npz")
    args = ap.parse_args()

    z = np.load(args.npz)
    k = z["kphys"]; T = z["T"]; r = z["r"]; kNyq = float(z["kNyq"])
    kmax = k[(T > 0)].max()
    # reliable band: strong signal, phase-aligned, away from k=0 and the sparse corner
    band = (k > 0.3) & (k < 0.9 * kNyq) & (T > 0) & (r > 0.98)
    # measured T_eff = sqrt(P(k)) ~ A * k^(ns/2) * T_EH(k)   (primordial tilt included)
    kpow = np.where(k > 0, k, 1.0) ** (args.ns / 2.0)
    Teh = kpow * T_eh98_nowiggle(k, args.Om, args.Ob, args.h)
    ratio = T[band] / Teh[band]
    A = np.median(ratio)
    scatter = np.std(ratio / A)
    print(f"[eh98] band k=[{k[band].min():.2f},{k[band].max():.2f}] h/Mpc  npts={band.sum()}")
    print(f"[eh98] amplitude A={A:.4e}  shape scatter std(T_meas/(A*T_EH))={scatter*100:.2f}%")
    ok = scatter < 0.06
    print(f"[eh98] shape match: {'GOOD' if ok else 'POOR'} (<6% => adopt EH98 tail)")
    # print comparison at a few k, incl. extrapolation region
    def Tmodel(kk):
        return A * kk ** (args.ns / 2.0) * T_eh98_nowiggle(np.array([kk]), args.Om, args.Ob, args.h)[0]
    for kk in (2.0, 5.0, 8.0, 15.0, 30.0, 60.0):
        tfit = 10 ** np.polyval(z["tail_coeff"], np.log10(kk))
        print(f"[eh98]   k={kk:5.1f}  A*k^(ns/2)*T_EH={Tmodel(kk):.3e}"
              f"   tailfit={tfit:.3e}   measured={np.interp(kk,k,T) if kk<=kmax else float('nan'):.3e}")

    if ok:
        # combined transfer: measured where reliable (k<=k_switch), A*k^(ns/2)*T_EH98 above.
        # Also refit tail_coeff to the EH98 model over [kNyq,100] so cf4_zoom_ic2.Transfer.eval
        # (which uses tail_coeff above kNyq) reproduces the physical extrapolation.
        k_switch = min(0.9 * kNyq, kmax)
        Tcomb = np.where(k <= k_switch, T, A * Teh)
        kext = np.logspace(np.log10(kNyq), np.log10(100.0), 60)
        Text = A * kext ** (args.ns / 2.0) * T_eh98_nowiggle(kext, args.Om, args.Ob, args.h)
        tail_coeff_eh = np.polyfit(np.log10(kext), np.log10(Text), 2)
        np.savez(args.out, kphys=k, T=Tcomb, r=r, kf=z["kf"], kNyq=kNyq, N=z["N"],
                 box_hmpc=z["box_hmpc"], eh98_A=A, eh98_Om=args.Om, eh98_Ob=args.Ob,
                 eh98_h=args.h, eh98_ns=args.ns, k_switch=k_switch, tail_coeff=tail_coeff_eh)
        print(f"[eh98] wrote combined (measured<{k_switch:.2f}, EH98 above) -> {args.out}")
        print("[eh98] NOTE: cf4_zoom_ic2.py Transfer.eval uses tail_coeff above kNyq; to use EH98,")
        print("[eh98]       point --transfer at this npz AND set its T beyond k_switch (done here).")
    else:
        print("[eh98] NOT writing; keep empirical tail. Check Ob/units.")


if __name__ == "__main__":
    main()
