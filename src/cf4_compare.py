#!/usr/bin/env python
"""Master comparison: flat-CNN vs U-Net vs FNO, per observable ablation.

Loads saved test predictions from all three architectures x three ablations,
recomputes per-sample r(k) (error bands over 32 test), and produces:
  - a table (voxel r, best val MSE, r(k) in large-scale & intermediate bands)
  - a 3-panel figure (one per ablation) comparing the architectures.
CPU-only.
"""
import os
import numpy as np
from cf4_verify import rk_persample

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BOX = 64 * 2.0

MODELS = {"flat-CNN": "cf4_cnn_pred_%s.npz",
          "U-Net": "cf4_unet_pred_%s.npz",
          "FNO": "cf4_fno_pred_%s.npz"}
ABL = ["full", "novel", "nogal"]
ABL_LABEL = {"full": "galaxies + velocity", "novel": "density only",
             "nogal": "peculiar-velocity ONLY"}


def load(model, ab):
    f = os.path.join(HERE, MODELS[model] % ab)
    if not os.path.exists(f):
        return None
    return np.load(f)


def main():
    results = {}
    for model in MODELS:
        for ab in ABL:
            z = load(model, ab)
            if z is None:
                continue
            kc, R = rk_persample(z["true"], z["pred"], BOX)
            results[(model, ab)] = dict(kc=kc, R=R, r_vox=float(z["r_vox"]),
                                        best_val=float(z["best_val"]))

    lo = None
    print(f"{'model':>10} {'ablation':>10} {'voxel_r':>8} {'val_mse':>8} "
          f"{'r(>40Mpc)':>10} {'r(~30Mpc)':>10}")
    for model in MODELS:
        for ab in ABL:
            if (model, ab) not in results:
                continue
            r = results[(model, ab)]
            kc = r["kc"]
            if lo is None:
                lo = kc < 0.15; mid = (kc > 0.1) & (kc < 0.3)
            rl = np.nanmean(r["R"][:, lo]); rm = np.nanmean(r["R"][:, mid])
            print(f"{model:>10} {ab:>10} {r['r_vox']:8.3f} {r['best_val']:8.4f} "
                  f"{rl:10.3f} {rm:10.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"flat-CNN": ("C7", "--"), "U-Net": ("C0", "-"), "FNO": ("C1", "-.")}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for j, ab in enumerate(ABL):
        ax = axes[j]
        for model in MODELS:
            if (model, ab) not in results:
                continue
            r = results[(model, ab)]
            mu = np.nanmean(r["R"], 0); sd = np.nanstd(r["R"], 0)
            c, ls = styles[model]
            ax.plot(r["kc"], mu, ls, color=c, lw=2,
                    label=f"{model} (r={r['r_vox']:.2f})")
            ax.fill_between(r["kc"], mu - sd, mu + sd, color=c, alpha=0.12)
        ax.axhline(0, ls=":", c="gray"); ax.axhline(1, ls=":", c="gray")
        ax.set_xscale("log"); ax.set_xlabel("k [h/Mpc]"); ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"{ab}: {ABL_LABEL[ab]}"); ax.legend(loc="lower left", fontsize=9)
        if j == 0:
            ax.set_ylabel("cross-corr r(k)")
    fig.suptitle("Density recovery vs architecture (32 test, ±1σ)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(ROOT, "recon", "cf4_arch_compare.png")
    fig.savefig(out, dpi=120)
    print(f"\n[fig] saved {out}")


if __name__ == "__main__":
    main()
