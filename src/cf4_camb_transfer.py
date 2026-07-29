#!/usr/bin/env python
"""Build the zoom transfer from CAMB (real Boltzmann) instead of EH98, at CF4 cosmology.

T_eff(k) = sqrt(P_lin(k)) up to an overall amplitude; CAMB gives P_lin(k) with the correct
transfer (BAO, baryons). We calibrate the amplitude to the measured T_eff over the overlap
band (ground truth where r(k)~1) and adopt A*sqrt(P_camb) above the reliable measured range.
Validated the same way as EH98 (shape scatter over the band).

Uses the installed CAMB (pycamb). Params match the parent IC / CF4: Om=0.31, h=0.746,
Ob h^2=0.0224, ns=0.9649, As=1.63e-9.
"""
import os
import sys
import argparse
import numpy as np


def camb_sqrtP(kphys, Om, Ob, h, ns, As):
    import camb
    ombh2 = Ob * h * h; omch2 = (Om - Ob) * h * h
    pars = camb.set_params(H0=100.0 * h, ombh2=ombh2, omch2=omch2, ns=ns, As=As, omk=0)
    pars.set_matter_power(redshifts=[0.0], kmax=150.0)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    kh, z, pk = res.get_matter_power_spectrum(minkh=1e-3, maxkh=100.0, npoints=600)
    # interpolate sqrt(P) in log-log onto kphys
    lk = np.log10(kh); lp = np.log10(np.sqrt(pk[0]))
    out = np.zeros_like(kphys)
    m = kphys > 0
    out[m] = 10.0 ** np.interp(np.log10(kphys[m]), lk, lp, left=lp[0], right=lp[-1])
    return out, kh, np.sqrt(pk[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/gpfs/kjhan/Hydro/CF4_LG/tier1/transfer_cr6_e19.npz")
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--Ob", type=float, default=0.0403)
    ap.add_argument("--h", type=float, default=0.746)
    ap.add_argument("--ns", type=float, default=0.9649)
    ap.add_argument("--As", type=float, default=1.63e-9)
    ap.add_argument("--out", default="/gpfs/kjhan/Hydro/CF4_LG/tier1/transfer_cr6_e19_camb.npz")
    args = ap.parse_args()

    z = np.load(args.npz)
    k = z["kphys"]; T = z["T"]; r = z["r"]; kNyq = float(z["kNyq"])
    kmax = k[(T > 0)].max()
    Tc, kh_c, sp_c = camb_sqrtP(k, args.Om, args.Ob, args.h, args.ns, args.As)

    band = (k > 0.3) & (k < 0.9 * kNyq) & (T > 0) & (r > 0.98)
    ratio = T[band] / Tc[band]
    A = np.median(ratio)
    scatter = np.std(ratio / A)
    print(f"[camb] band k=[{k[band].min():.2f},{k[band].max():.2f}] npts={band.sum()}")
    print(f"[camb] amplitude A={A:.4e}  shape scatter std(T_meas/(A*sqrtP_camb))={scatter*100:.2f}%")
    ok = scatter < 0.06
    print(f"[camb] shape match: {'GOOD' if ok else 'POOR'}")
    for kk in (2.0, 5.0, 8.0, 15.0, 30.0, 60.0):
        Tcamb = A * 10.0 ** np.interp(np.log10(kk), np.log10(kh_c), np.log10(sp_c))
        tfit = 10 ** np.polyval(z["tail_coeff"], np.log10(kk))
        meas = np.interp(kk, k, T) if kk <= kmax else float("nan")
        print(f"[camb]   k={kk:5.1f}  A*sqrtP_camb={Tcamb:.3e}   tailfit={tfit:.3e}   measured={meas:.3e}")

    if ok:
        k_switch = min(0.9 * kNyq, kmax)
        Tcomb = np.where(k <= k_switch, T, A * Tc)
        kext = np.logspace(np.log10(kNyq), np.log10(min(100.0, kh_c.max())), 80)
        Text = A * 10.0 ** np.interp(np.log10(kext), np.log10(kh_c), np.log10(sp_c))
        tail_coeff_camb = np.polyfit(np.log10(kext), np.log10(Text), 3)
        np.savez(args.out, kphys=k, T=Tcomb, r=r, kf=z["kf"], kNyq=kNyq, N=z["N"],
                 box_hmpc=z["box_hmpc"], camb_A=A, camb_Om=args.Om, camb_Ob=args.Ob,
                 camb_h=args.h, camb_ns=args.ns, camb_As=args.As, k_switch=k_switch,
                 tail_coeff=tail_coeff_camb)
        print(f"[camb] wrote combined (measured<{k_switch:.2f}, CAMB above) -> {args.out}")
    else:
        print("[camb] NOT writing; check params.")


if __name__ == "__main__":
    main()
