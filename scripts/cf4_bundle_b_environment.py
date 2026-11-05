"""Fixed-aperture actual-map diagnosis, not a halo or all-structures pass test."""
import csv
from itertools import product
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_actual_data_preview import read_plan, load_data
from cf4_z0_physical_field import centres


def landmarks():
    old = json.loads((ROOT / "config/p1_targets_v2_observer.json").read_text())
    def xyz(item):
        lon, lat = np.deg2rad([item["sgl_deg"], item["sgb_deg"]])
        r = item.get("distance_mpc_h", item.get("distance_mpc", 0) * .746)
        return r * np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    rows = [(name, xyz(item), 1) for group in ("clusters", "secondary_cluster_anchors")
            for name, item in old[group].items()]
    rows += [(name, np.array(pos), -1) for name, pos in old["local_void"]["probes"].items()]
    rows += [("Bootes", xyz(old["bootes_void"]), -1)]
    return rows


def aperture(center, radius, support, n=32, box=384.):
    """4^3 Gauss volume quadrature over native piecewise-constant cells.

    Survey coverage is evaluated at physical subpoints in count-cell coordinates;
    outside the survey/box is not wrapped or treated as observed zero counts.
    """
    dx = box/n
    grid = centres(n, box, .25).reshape(-1, 3)
    use = np.linalg.norm(grid-center, axis=1) <= radius+np.sqrt(3)*dx/2
    ids = np.flatnonzero(use)
    weights = np.zeros(n**3)
    supported = radial = 0.
    nodes, gauss = np.polynomial.legendre.leggauss(4)
    for a, b, c in product(range(4), repeat=3):
        pos = grid[ids] + dx/2*nodes[[a,b,c]]
        w = gauss[a]*gauss[b]*gauss[c]/8
        inside = (np.linalg.norm(pos-center, axis=1) <= radius)
        weights[ids] += w*inside
        rr = np.linalg.norm(pos-box/2, axis=1)
        covered = inside & (rr >= 5) & (rr <= 180) & np.all((pos >= 0) & (pos < box), axis=1)
        count_idx = np.clip(np.floor(pos/dx).astype(int), 0, n-1)
        observed = support[tuple(count_idx.T)]
        radial += w*covered.sum()
        supported += w*np.sum(covered*observed)
    total = weights.sum()
    if total <= 0:
        raise ValueError("empty aperture")
    return weights/total, dict(support_fraction=float(supported/total),
        radial_box_coverage=float(radial/total),
        quadrature_volume_ratio=float(total*dx**3/(4*np.pi*radius**3/3)))


def main():
    config = json.loads((ROOT / "config/cf4_bundle_b_v1.json").read_text())
    out = Path(config["output_root"])/"environment"
    out.mkdir(parents=True, exist_ok=False)
    plan = read_plan(ROOT/config["actual_plan"])
    model = load_data(0, plan)[0]
    with np.load(Path(plan["input_root"])/"observations.npz", allow_pickle=False) as f:
        support = np.any(f["exposure"] > 0, axis=0)
    source = Path(plan["output_root"])/"task_0"
    marks, rows, windows = landmarks(), [], []
    for name, offset, sign in marks:
        for radius in config["map"]["radii"] + ([31] if name == "Bootes" else []):
            weights, coverage = aperture(offset+192, radius, support)
            windows.append(weights)
            rows.append(dict(name=name, radius_cMpc_h=radius, expected_sign=sign, **coverage))
    windows = np.stack(windows)
    field_fn = jax.jit(lambda x: model.fields(x)[1]-1)
    samples, extrema = [], []
    grid = centres(32, 384, .25).reshape(-1,3)-192
    selections = [np.flatnonzero(np.linalg.norm(grid-pos,axis=1)<=24) for _,pos,_ in marks]
    for chain in range(4):
        with np.load(source/f"chain_{chain}.npz", allow_pickle=False) as f:
            retained = f["retained_vectors"]
            if retained.shape[0] != 512:
                raise ValueError("unexpected retained-chain length")
            draws = retained[8:512:16].copy()
        del retained
        for vector in draws:
            delta = np.asarray(field_fn(jnp.asarray(vector))).ravel()
            samples.append(windows @ delta)
            extrema.append([np.linalg.norm(grid[idx[np.argmax(sign*delta[idx])]]-pos)
                for (_,pos,sign),idx in zip(marks,selections)])
        print(f"chain {chain}: {len(samples)}/128 fields", flush=True)
    samples, extrema = np.array(samples), np.array(extrema)
    for j,row in enumerate(rows):
        lo,med,hi = np.quantile(samples[:,j], [.025,.5,.975])
        sign = 1 if lo>0 else -1 if hi<0 else 0
        category = ("insufficient_support" if row["support_fraction"]<config["map"]["support_threshold"]
            else "uncertain" if sign==0 else "supported_model_sign" if sign==row["expected_sign"] else "opposite_model_sign")
        k = [m[0] for m in marks].index(row["name"])
        row.update(mean=float(samples[:,j].mean()), q025=float(lo), median=float(med), q975=float(hi),
            P_positive=float(np.mean(samples[:,j]>0)), P_negative=float(np.mean(samples[:,j]<0)),
            native_extremum_offset_median_cMpc_h=float(np.median(extrema[:,k])), classification=category)
    np.savez_compressed(out/"apertures.npz", draws=samples, covariance=np.cov(samples,rowvar=False),
                        native_extremum_offsets=extrema)
    with (out/"structures.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(status="MODEL_CONDITIONAL_ENVIRONMENT_DIAGNOSTIC", rows=rows,
        fields=128, retained_indices=list(range(8,512,16)), density_origin_fraction=.25,
        count_origin_fraction=.5, support="any nonzero tracer exposure plus 5..180 cMpc/h radial cut",
        uncertainties="Correlated aperture means from individual draws; not observationally calibrated.",
        positions="Historical p1_targets_v2_observer.json anchors; distances fixed, not marginalized; no imported screening gates.",
        limits="12 cMpc/h grid; no halo/void-boundary accuracy. Optional Carrick overlay omitted.")
    (out/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(source/"posterior_fields.npz", allow_pickle=False) as f:
        mean = f["density_mean"]
    fig, axes = plt.subplots(1,2,figsize=(15,7),constrained_layout=True)
    axis = (np.arange(32)+.25)*12-192
    for ax,(a,b,c) in zip(axes, [(0,1,2),(1,2,0)]):
        slab = np.take(mean, np.flatnonzero(abs(axis)<36), axis=c).mean(axis=c)
        im = ax.imshow(slab.T,origin="lower",extent=[axis[0]-6,axis[-1]+6]*2,
                       cmap="RdBu_r",vmin=-.6,vmax=.6)
        for name,pos,sign in marks:
            ax.scatter(pos[a],pos[b],marker="^" if sign>0 else "o",facecolors="none",edgecolors="k",s=35)
            ax.annotate(name,(pos[a],pos[b]),fontsize=7)
        ax.set(xlabel=f"SG{'XYZ'[a]} [cMpc/h]",ylabel=f"SG{'XYZ'[b]} [cMpc/h]",
               title=f"Mean delta, |SG{'XYZ'[c]}| <36; all anchors projected")
        ax.set_xlim(-185,185); ax.set_ylim(-185,185)
    fig.colorbar(im,ax=axes,label="Model-conditional mean delta (not halo map)")
    fig.savefig(out/"environment.png",dpi=160); plt.close(fig)
    print(json.dumps({"status":report["status"],"output":str(out)}),flush=True)


if __name__ == "__main__":
    main()
