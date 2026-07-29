#!/usr/bin/env python
"""Compare clean vs realistic (sparse+noisy) recovery, flat-CNN vs U-Net.

Clean preds live in src/ (data_train); realistic preds in recon/real/
(data_train_real, same seeds => same delta_m target, so directly comparable).
Per-sample r(k) with error bands; table + 2-panel figure (full / velocity-only).
"""
import os
import numpy as np
from cf4_verify import rk_persample

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BOX = 64 * 2.0

# (label, dir, filename template)
SOURCES = {
    ("flat-CNN", "clean"): (HERE, "cf4_cnn_pred_%s.npz"),
    ("U-Net", "clean"): (HERE, "cf4_unet_pred_%s.npz"),
    ("flat-CNN", "realistic"): (os.path.join(ROOT, "recon", "real"), "cf4_cnn_pred_%s.npz"),
    ("U-Net", "realistic"): (os.path.join(ROOT, "recon", "real"), "cf4_unet_pred_%s.npz"),
}
ABL = ["full", "nogal"]
ABL_LABEL = {"full": "galaxies + velocity", "nogal": "peculiar-velocity ONLY"}


def main():
    res = {}
    for (model, cond), (d, tmpl) in SOURCES.items():
        for ab in ABL:
            f = os.path.join(d, tmpl % ab)
            if not os.path.exists(f):
                continue
            z = np.load(f)
            kc, R = rk_persample(z["true"], z["pred"], BOX)
            res[(model, cond, ab)] = dict(kc=kc, R=R, r_vox=float(z["r_vox"]),
                                          best_val=float(z["best_val"]))
    lo = mid = None
    print(f"{'model':>9} {'cond':>10} {'ablation':>6} {'voxel_r':>8} {'val':>7} "
          f"{'r(>40Mpc)':>10} {'r(~30Mpc)':>10}")
    for cond in ("clean", "realistic"):
        for ab in ABL:
            for model in ("flat-CNN", "U-Net"):
                k = (model, cond, ab)
                if k not in res:
                    continue
                r = res[k]; kc = r["kc"]
                if lo is None:
                    lo = kc < 0.15; mid = (kc > 0.1) & (kc < 0.3)
                print(f"{model:>9} {cond:>10} {ab:>6} {r['r_vox']:8.3f} {r['best_val']:7.3f} "
                      f"{np.nanmean(r['R'][:, lo]):10.3f} {np.nanmean(r['R'][:, mid]):10.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    sty = {("flat-CNN", "clean"): ("C7", ":"), ("U-Net", "clean"): ("C9", ":"),
           ("flat-CNN", "realistic"): ("C3", "--"), ("U-Net", "realistic"): ("C0", "-")}
    for j, ab in enumerate(ABL):
        ax = axes[j]
        for (model, cond) in [("flat-CNN", "clean"), ("U-Net", "clean"),
                              ("flat-CNN", "realistic"), ("U-Net", "realistic")]:
            k = (model, cond, ab)
            if k not in res:
                continue
            r = res[k]; mu = np.nanmean(r["R"], 0); sd = np.nanstd(r["R"], 0)
            c, ls = sty[(model, cond)]
            al = 1.0 if cond == "realistic" else 0.5
            lw = 2.2 if cond == "realistic" else 1.3
            ax.plot(r["kc"], mu, ls, color=c, lw=lw, alpha=al,
                    label=f"{model} / {cond} (r={r['r_vox']:.2f})")
            if cond == "realistic":
                ax.fill_between(r["kc"], mu - sd, mu + sd, color=c, alpha=0.12)
        ax.axhline(0, ls=":", c="gray"); ax.axhline(1, ls=":", c="gray")
        ax.set_xscale("log"); ax.set_xlabel("k [h/Mpc]"); ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"{ab}: {ABL_LABEL[ab]}"); ax.legend(loc="lower left", fontsize=8)
        if j == 0:
            ax.set_ylabel("cross-corr r(k)")
    fig.suptitle("Clean vs realistic (sparse+noisy) recovery — flat-CNN vs U-Net",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(ROOT, "recon", "cf4_real_compare.png")
    fig.savefig(out, dpi=120)
    print(f"\n[fig] saved {out}")


if __name__ == "__main__":
    main()
