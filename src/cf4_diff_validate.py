#!/usr/bin/env python
"""Validate the CF4 IC-diffusion posterior: recovered s vs truth.

Reads a sample-mode npz (post_mean, post_sample, rk, z per object) and the matching
test data (true s), and renders:
  row 1  true s | posterior-mean s | one posterior sample   (central slab)
  row 2  cross-correlation r(k) [post-mean vs true]  | calibration z-score PDF
         | metrics text

The posterior MEAN is the constrained (Wiener-like) estimate -> smooth, high large-scale
fidelity. A posterior SAMPLE restores full white-noise power (what lagRAMSES needs).
Calibration std(z)~1 in the low-k (data-constrained) band = well-calibrated posterior.
"""
import os, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="cf4_diff_*.npz from sample mode")
    ap.add_argument("--data", required=True, help="data dir with test/ shards")
    ap.add_argument("--obj", type=int, default=0, help="object index within pred")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P = np.load(args.pred)
    rk = P["rk"]; z = P["z"]; pm = P["post_mean"]; ps = P["post_sample"]
    objs = P["obj"]; ablation = str(P["ablation"])
    N = pm.shape[1]
    # true s for the same object index
    tf = sorted(glob.glob(os.path.join(args.data, "test", "shard_*.npz")))
    Strue = np.concatenate([np.load(f)["s"].astype(np.float32).reshape(-1, N, N, N)
                            for f in tf])
    oi = args.obj
    o_global = int(objs[oi])
    s_true = Strue[o_global]; s_mean = pm[oi]; s_samp = ps[oi]

    zc = N // 2
    sl = lambda a: a[:, :, max(zc - 1, 0):zc + 2].mean(2).T
    vlim = np.percentile(np.abs(s_true), 99)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9.6))
    for a, f, t in zip(ax[0], (s_true, s_mean, s_samp),
                       ("true initial s", "posterior MEAN s (constrained)",
                        "posterior SAMPLE s (full power)")):
        im = a.imshow(sl(f), origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        a.set_title(t); a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046)

    # r(k): post-mean vs true, mean over all objects + this object
    kk = (np.arange(rk.shape[1]) + 0.5) / N          # cell^-1
    ax[1, 0].plot(kk, np.nanmean(rk, 0), "C0-", lw=2, label="mean over objs")
    ax[1, 0].plot(kk, rk[oi], "C1--", lw=1.5, label=f"obj {o_global}")
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].set_xlabel("k [cell$^{-1}$]"); ax[1, 0].set_ylabel("r(k)  recon vs true")
    ax[1, 0].set_title("initial-field cross-correlation")
    ax[1, 0].set_ylim(-0.05, 1.02); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    # calibration z PDF (low-k band)
    zlo = z[:, :6].ravel()
    ax[1, 1].hist(zlo, bins=30, density=True, alpha=0.7, color="C2",
                  label=f"low-k std(z)={zlo.std():.2f}")
    xg = np.linspace(-4, 4, 100)
    ax[1, 1].plot(xg, np.exp(-xg**2/2)/np.sqrt(2*np.pi), "k--", label="N(0,1)")
    ax[1, 1].set_xlabel("z = (s_true - <s>)/std"); ax[1, 1].set_title("posterior calibration (low-k)")
    ax[1, 1].legend()

    lo = float(np.nanmean(rk[:, :4])); mid = float(np.nanmean(rk[:, 4:12]))
    txt = (f"CF4 IC diffusion — {ablation}\n\n"
           f"N = {N}   objects = {len(objs)}\n\n"
           f"r(k) low  (k<{4.5/N:.3f}) = {lo:.3f}\n"
           f"r(k) mid              = {mid:.3f}\n\n"
           f"calibration std(z):\n"
           f"  low-k = {z[:, :6].std():.2f}  (1=ideal)\n"
           f"  all   = {z.std():.2f}\n\n"
           f"post-MEAN = constrained estimate\n"
           f"post-SAMPLE -> GRAFIC/lagRAMSES")
    ax[1, 2].axis("off")
    ax[1, 2].text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=11)

    fig.suptitle(f"CF4 Stage-B: initial field s recovered from galaxy observable ({ablation})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or args.pred.replace(".npz", ".png")
    fig.savefig(out, dpi=110)
    print(f"[validate] r(k) low={lo:.3f} mid={mid:.3f} | low-k std(z)={z[:, :6].std():.2f}")
    print(f"[validate] saved {out}")


if __name__ == "__main__":
    main()
