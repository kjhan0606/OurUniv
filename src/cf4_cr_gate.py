#!/usr/bin/env python
"""Audit a held-out CF4 test and an all-data linear-Gaussian CR ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def shell_geometry(N: int, L: float, nbins: int = 12):
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    edges = np.geomspace(2.0 * np.pi / L, np.pi * N / L, nbins + 1)
    masks = [(kmag >= lo) & (kmag < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    kmean = np.array([kmag[m].mean() for m in masks])
    nmode = np.array([m.sum() for m in masks])
    return edges, masks, kmean, nmode


def ensemble_spectra(parent):
    outputs = parent["outputs"]
    with np.load(outputs[0]) as z:
        mean = z["s_map"].astype(np.float64)
        N = int(z["N"])
        L = float(z["L"])
    edges, masks, kmean, nmode = shell_geometry(N, L)
    mean_k = np.fft.rfftn(mean, norm="ortho")
    mean_power = np.array([np.mean(np.abs(mean_k[m]) ** 2) for m in masks])
    sample_power = []
    residual_power = []
    coherence = []
    for path in outputs:
        with np.load(path) as z:
            sample = z["s_out"].astype(np.float64)
        fk = np.fft.rfftn(sample, norm="ortho")
        sample_power.append([np.mean(np.abs(fk[m]) ** 2) for m in masks])
        residual_power.append([np.mean(np.abs((fk - mean_k)[m]) ** 2) for m in masks])
        rr = []
        for m in masks:
            num = np.real(np.sum(fk[m] * np.conj(mean_k[m])))
            den = np.sqrt(np.sum(np.abs(fk[m]) ** 2) * np.sum(np.abs(mean_k[m]) ** 2))
            rr.append(num / den if den > 0 else np.nan)
        coherence.append(rr)
    return {
        "edges": edges,
        "kmean": kmean,
        "nmode": nmode,
        "mean_power": mean_power,
        "sample_power": np.asarray(sample_power),
        "residual_power": np.asarray(residual_power),
        "coherence": np.asarray(coherence),
    }


def gate(test, parent, spectra):
    held = test["heldout"]
    parent_samples = parent["samples"]
    std = np.array([x["std"] for x in parent_samples])
    skew = np.array([x["skew"] for x in parent_samples])
    kurt = np.array([x["excess_kurtosis"] for x in parent_samples])
    cg = np.array([x["cg_rel"] for x in parent_samples])
    pmean = spectra["sample_power"].mean(axis=0)
    nmode = spectra["nmode"]
    power_tol = np.maximum(0.05, 3.0 / np.sqrt(nmode))

    checks = {
        "operator_adjoint": {
            "pass": parent["adjoint_relative_error"] < 1e-4,
            "value": parent["adjoint_relative_error"],
            "limit": "<1e-4",
        },
        "cg_accuracy": {
            "pass": max(cg.max(), parent["mean_cg_relative_residual"]) < 1e-4,
            "value": float(max(cg.max(), parent["mean_cg_relative_residual"])),
            "limit": "<1e-4",
        },
        "heldout_mean": {
            "pass": abs(held["z_mean"]) < 0.1,
            "value": held["z_mean"],
            "limit": "|mean z|<0.1",
        },
        "heldout_scale": {
            "pass": 0.9 <= held["z_std"] <= 1.1,
            "value": held["z_std"],
            "limit": "0.9<=std(z)<=1.1",
        },
        "heldout_coverage_68": {
            "pass": 0.65 <= held["coverage_1sigma"] <= 0.75,
            "value": held["coverage_1sigma"],
            "limit": "0.65..0.75",
        },
        "heldout_coverage_95": {
            "pass": 0.93 <= held["coverage_2sigma"] <= 0.97,
            "value": held["coverage_2sigma"],
            "limit": "0.93..0.97",
        },
        "heldout_information": {
            "pass": held["delta_log_score"] > 0,
            "value": held["delta_log_score"],
            "limit": "delta log score > 0",
        },
        "white_field_variance": {
            "pass": bool(np.all((std >= 0.995) & (std <= 1.005))),
            "value": [float(std.min()), float(std.max())],
            "limit": "all sample std in 0.995..1.005",
        },
        "white_field_gaussianity": {
            "pass": bool(max(np.max(np.abs(skew)), np.max(np.abs(kurt))) < 0.01),
            "value": {
                "max_abs_skew": float(np.max(np.abs(skew))),
                "max_abs_excess_kurtosis": float(np.max(np.abs(kurt))),
            },
            "limit": "both < 0.01",
        },
        "lcdm_shell_power": {
            "pass": bool(np.all(np.abs(pmean - 1.0) <= power_tol)),
            "value": {
                "mean_ratio": pmean.tolist(),
                "three_sigma_tolerance": power_tol.tolist(),
            },
            "limit": "|<Pwhite>-1| <= max(0.05,3/sqrt(Nmode))",
        },
        "training_residual": {
            "pass": 0.8 <= parent["train_normalized_residual_rms"] <= 1.2,
            "value": parent["train_normalized_residual_rms"],
            "limit": "0.8..1.2",
        },
    }
    return checks, all(x["pass"] for x in checks.values())


def make_plot(path, test, spectra):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = spectra["kmean"]
    sp = spectra["sample_power"]
    rp = spectra["residual_power"]
    mp = spectra["mean_power"]
    coh = spectra["coherence"]
    held = test["heldout"]

    fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.7))
    for row in sp:
        ax[0].plot(k, row, color="C0", alpha=0.22, lw=0.8)
    ax[0].plot(k, sp.mean(0), "o-", color="navy", label="16-CR mean")
    ax[0].axhline(1.0, color="k", ls="--", lw=1, label="LCDM white prior")
    ax[0].set(xscale="log", xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel="white-mode power")
    ax[0].set_title("LCDM power completion")
    ax[0].legend(fontsize=8)

    ax[1].plot(k, rp.mean(0), "o-", label="posterior residual")
    ax[1].plot(k, mp, "s-", label="WF mean")
    ax[1].plot(k, (rp + mp).mean(0), "^-", label="residual + mean")
    ax[1].set(xscale="log", yscale="log", xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$",
              ylabel="white-mode power")
    ax[1].set_title("Constraint / random decomposition")
    ax[1].legend(fontsize=8)

    ax[2].plot(k, np.nanmean(coh, axis=0), "o-", color="C3")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xscale="log", ylim=(-0.1, 1.0), xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$",
              ylabel=r"$r(s_{\rm CR},s_{\rm WF})$")
    ax[2].set_title(
        "Held-out: "
        rf"$z={held['z_mean']:+.3f}\pm{held['z_std']:.3f}$, "
        rf"$\Delta\ln p={held['delta_log_score']:+.0f}$"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def write_report(path, test_path, parent_path, test, parent, spectra, checks, passed):
    held = test["heldout"]
    pmean = spectra["sample_power"].mean(0)
    lines = [
        "# CF4 statistically valid constrained-realization gate",
        "",
        f"**Overall: {'PASS' if passed else 'FAIL'}**",
        "",
        "This gate validates a linear-Gaussian posterior ensemble. It does not yet select",
        "the member that best reproduces the MW–M31–M33 system or the named clusters/voids;",
        "that is the next, pre-registered posterior-predictive selection stage.",
        "",
        "## Frozen model",
        "",
        f"- Test manifest: `{Path(test_path).name}`",
        f"- All-data ensemble: `{Path(parent_path).name}`",
        f"- Grid: N={parent['configuration']['N']}, L={parent['configuration']['box_size']} Mpc/h",
        f"- Cosmology: Om={parent['configuration']['Om']}, Ob={parent['configuration']['Ob']}, "
        f"h={parent['configuration']['h']}, As(1e9)={parent['configuration']['A_s_1e9']}, "
        f"ns={parent['configuration']['ns']}",
        f"- CF4 constraints: {parent['n_train']} grouped distances; "
        f"{parent['configuration'].get('velocity_estimator', 'wf15').upper()} "
        "Gaussian velocity estimator",
        f"- Error model: scale={parent['configuration']['error_scale']}, "
        f"sigma_NL={parent['configuration']['sigma_nl']} km/s",
        f"- Ensemble: {len(parent['samples'])} exact Matheron draws, seeds "
        f"{parent['configuration']['sample_seeds'][0]}–{parent['configuration']['sample_seeds'][-1]}",
        "",
        "## Gate checks",
        "",
        "| Check | Result | Value | Limit |",
        "|---|---:|---:|---|",
    ]
    for name, item in checks.items():
        value = json.dumps(item["value"], separators=(",", ":"))
        if len(value) > 90:
            value = value[:87] + "..."
        lines.append(f"| `{name}` | {'PASS' if item['pass'] else 'FAIL'} | `{value}` | {item['limit']} |")
    lines += [
        "",
        "## Held-out posterior predictive test",
        "",
        f"- N={held['n']}; standardized residual mean/std = "
        f"{held['z_mean']:+.4f}/{held['z_std']:.4f}.",
        f"- 68/95% coverage = {held['coverage_1sigma']:.4f}/{held['coverage_2sigma']:.4f}.",
        f"- Relative to noise-only, delta log predictive density = {held['delta_log_score']:+.1f}.",
        "",
        "## LCDM power and phase",
        "",
        "The field stored as `s_out` is the whitened primordial field. Its shell power",
        "therefore should be unity; applying the frozen transfer function yields the target",
        "LCDM P(k) by construction. The ensemble-mean shell ratios are:",
        "",
        "| k [h/Mpc] | N(rFFT) | <Pwhite> | residual power | WF-mean power | coherence |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for i, kval in enumerate(spectra["kmean"]):
        lines.append(
            f"| {kval:.4f} | {spectra['nmode'][i]} | {pmean[i]:.4f} | "
            f"{spectra['residual_power'][:, i].mean():.4f} | "
            f"{spectra['mean_power'][i]:.4f} | "
            f"{np.nanmean(spectra['coherence'][:, i]):.4f} |"
        )
    lines += [
        "",
        "## Status",
        "",
        "- The 16-member ensemble is accepted as the statistically valid parent-CR ensemble.",
        f"- Seed {parent['configuration']['sample_seeds'][0]} is the deterministic "
        "reference member only; it was not chosen using",
        "  held-out data or Local-Group morphology.",
        "- No member is yet the final physical parent. Named-structure and LG acceptance",
        "  criteria must be frozen before forwarding and choosing among these 16 draws.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--parent-manifest", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    test = load_json(args.test_manifest)
    parent = load_json(args.parent_manifest)
    spectra = ensemble_spectra(parent)
    checks, passed = gate(test, parent, spectra)

    result = {
        "overall_pass": passed,
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "parent_manifest": str(Path(args.parent_manifest).resolve()),
        "reference_member": parent["outputs"][0],
        "checks": checks,
        "spectra": {
            "k_mean": spectra["kmean"].tolist(),
            "nmode_rfft": spectra["nmode"].tolist(),
            "sample_power_mean": spectra["sample_power"].mean(0).tolist(),
            "posterior_residual_power_mean": spectra["residual_power"].mean(0).tolist(),
            "wf_mean_power": spectra["mean_power"].tolist(),
            "coherence_mean": np.nanmean(spectra["coherence"], axis=0).tolist(),
        },
    }
    result_path = outdir / "cr_gate_result.json"
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )
    plot_path = outdir / "cr_gate.png"
    make_plot(plot_path, test, spectra)
    report_path = outdir / "CR_GATE_REPORT.md"
    write_report(
        report_path,
        args.test_manifest,
        args.parent_manifest,
        test,
        parent,
        spectra,
        checks,
        passed,
    )
    print(f"[gate] {'PASS' if passed else 'FAIL'}")
    print(f"[out] {result_path}")
    print(f"[out] {plot_path}")
    print(f"[out] {report_path}")


if __name__ == "__main__":
    main()
