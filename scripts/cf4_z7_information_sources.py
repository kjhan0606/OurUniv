"""Same-data likelihood ablations and bounded source/scale comparison."""
import copy
import json
from pathlib import Path

import numpy as np

from cf4_z6_native_physics import load_mock as load_native

ROOT = Path(__file__).resolve().parents[1]


def load_mock(task, plan):
    case = plan["data"]["cases"][task]
    native = json.loads((ROOT / plan["data"]["Z6_plan"]).read_text())
    native = copy.deepcopy(native)
    index = case["Z6_case"]
    native["data"]["cases"][index].update(use_counts=case["use_counts"], use_radial=case["use_radial"])
    model, design, rho, velocity, meta, counts, holdcounts, radial, candidate = load_native(index, native)
    source = Path(plan["data"]["Z6_root"]) / f"task_{index}" / "mock.npz"
    with np.load(source, allow_pickle=False) as saved:
        for key, value in (("counts_train", counts), ("counts_holdout", holdcounts), ("radial_mock", radial),
                           ("truth_density", rho-1), ("truth_velocity", velocity)):
            if not np.array_equal(saved[key], value):
                raise ValueError(f"Z7 changed the saved Z6 datum: {key}")
        counts, holdcounts, radial = (saved[k].copy() for k in ("counts_train", "counts_holdout", "radial_mock"))
    meta.update(channel=case["channel"], Z6_case=index, exact_saved_Z6_data=True)
    return model, design, rho, velocity, meta, counts, holdcounts, radial, candidate


def spectral_summary(directory, prior_amplitude):
    """Process one retained chain at a time; no whole-project or filesystem scan."""
    n = prior_amplitude.shape[0]
    total = 0
    mode_sum = np.zeros((n,n,n), dtype=complex)
    mode_square = np.zeros((n,n,n))
    for chain in range(4):
        with np.load(directory / f"chain_{chain}.npz", allow_pickle=False) as data:
            white = data["retained_vectors"][:, :n**3].copy().reshape(-1,n,n,n)
        for offset in range(0, len(white), 16):
            modes = np.fft.fftn(white[offset:offset+16], axes=(1,2,3), norm="ortho") * prior_amplitude
            mode_sum += modes.sum(axis=0)
            mode_square += (abs(modes)**2).sum(axis=0)
            total += len(modes)
    variance = (mode_square - abs(mode_sum)**2 / total) / (total - 1)
    with np.load(directory / "posterior_fields.npz", allow_pickle=False) as data:
        truth = np.fft.fftn(data["truth_density"], norm="ortho")
        fitted = np.fft.fftn(data["density_mean"], norm="ortho")
    frequency = 2*np.pi*np.fft.fftfreq(n,d=384/n)
    kmag = np.sqrt(sum(x*x for x in np.meshgrid(frequency,frequency,frequency,indexing="ij")))
    edges = np.geomspace(2*np.pi/384*.99,kmag.max()*(1+1e-8),11)
    rows=[]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask=(kmag>=lo)&(kmag<hi)&(prior_amplitude>0)
        if not mask.any(): continue
        denominator=np.sqrt(np.sum(abs(truth[mask])**2)*np.sum(abs(fitted[mask])**2))
        rows.append({"k_h_Mpc":float(kmag[mask].mean()),"grid_modes":int(mask.sum()),
                     "physical_density_shell_correlation":float(np.sum((truth[mask]*fitted[mask].conj()).real)/denominator) if denominator>0 else None,
                     "log_density_variance_reduction":float(1-variance[mask].sum()/(prior_amplitude[mask]**2).sum())})
    return rows


def compare(plan):
    root=Path(plan["output_root"])
    baseline=Path(plan["data"]["Z6_root"])
    with np.load(baseline / "covariance.npz",allow_pickle=False) as data:
        amplitude=data["log_density_amplitude"]
    cases=[(0,"joint",baseline/"task_1"),(5,"joint",baseline/"task_2")]
    cases += [(c["truth_index"],c["channel"],root/f"task_{c['task']}") for c in plan["data"]["cases"]]
    rows=[]
    for truth_index, channel, directory in cases:
        path=directory/"result.json"
        if not path.exists():
            rows.append(dict(truth_index=truth_index,channel=channel,status="MISSING_OR_FAILED")); continue
        result=json.loads(path.read_text())
        row=dict(truth_index=truth_index,channel=channel,status=result["status"],
                 density=result["density"],heldout=result["heldout_pointwise_log_predictive_gain"],
                 convergence=result["convergence"])
        if result["status"] == "MECHANICS_PASS_DEVELOPMENT_ONLY":
            row["spectral"]=spectral_summary(directory,amplitude)
        rows.append(row)
    report=dict(bundle=plan["bundle"],cases=rows,next_bundle_started=False,
                interpretation="Whole-box development source comparisons, not a certified local information frontier. Monte Carlo variance estimates use correlated thinned draws. No power rescaling or post-hoc cutoff selection.")
    (root/"source_comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for i,truth_index in enumerate((0,5)):
        for row in rows:
            if row["truth_index"]!=truth_index or "spectral" not in row: continue
            k=[x["k_h_Mpc"] for x in row["spectral"]]
            axes[i,0].plot(k,[x["physical_density_shell_correlation"] for x in row["spectral"]],label=row["channel"])
            axes[i,1].plot(k,[x["log_density_variance_reduction"] for x in row["spectral"]],label=row["channel"])
        for j in (0,1):
            axes[i,j].set_xscale("log"); axes[i,j].set_xlabel("k [h/cMpc]"); axes[i,j].legend()
        axes[i,0].set_title(f"Truth {truth_index}: physical-density correlation")
        axes[i,1].set_title(f"Truth {truth_index}: log-density variance reduction")
    fig.savefig(root/"source_comparison.png",dpi=130); plt.close(fig)
