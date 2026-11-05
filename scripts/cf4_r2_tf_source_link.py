"""Disjoint corrected-TF source bridge and one same-state numerical control."""

import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_r2_tf_group_marks import tf_group_logratios

BASE = Path("/gpfs/kjhan/CF4/z0_density")
OUT = BASE / "r2_tf_source_link_v1"
NATIVE = BASE / "bundle_c_v1/native_data_v2/CF4_native.npz"
FP = BASE / "r2_sdss_fp_source_link_v1/source_link.npz"
STATE = BASE / "r2_pm128_unconditional_v1/state.npz"
GROUPS = ROOT / "data/cf4_groups.csv"
GALAXIES = ROOT / "data/cf4_galaxies.csv"
HASHES = {
    GROUPS: "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334",
    GALAXIES: "28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709",
}
OTHER_METHODS = ("o_DMcal", "o_DMsnIa", "o_DMfp", "o_DMsbfo", "o_DMsbfi")


def number(value):
    return float(value) if value else float("nan")


def source_rows():
    for path, expected in HASHES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned CF4 source changed: {path}")
    with GROUPS.open(newline="") as handle:
        groups = list(csv.DictReader(handle))
    with GALAXIES.open(newline="") as handle:
        galaxies = list(csv.DictReader(handle))
    return groups, galaxies


def disjoint_tf(groups, galaxies, native, fp_pgc, fp_t17):
    """No CF4-FP method or shared source T17 component enters this control."""
    t17_by_cfgroup = {}
    pgc_by_cfgroup = {}
    for row in galaxies:
        if not row["1PGC"]:
            continue
        group = int(row["1PGC"])
        if row["T17"] and int(row["T17"]) > 0:
            t17_by_cfgroup.setdefault(group, set()).add(int(row["T17"]))
        if row["PGC"]:
            pgc_by_cfgroup.setdefault(group, set()).add(int(row["PGC"]))
    skipped = dict(other_method=0, missing_measurement=0, fp_source_overlap=0,
                   nonnative_or_range=0)
    selected = []
    for row in groups:
        if not row["o_DMtf"] or int(row["o_DMtf"]) <= 0:
            continue
        if any(row[name] and int(row[name]) > 0 for name in OTHER_METHODS) or row["DMsnII"]:
            skipped["other_method"] += 1
            continue
        group = int(row["1PGC"])
        dm, error, vcmb = (number(row[name]) for name in ("DMtf", "e_DMtf", "Vcmb"))
        sgl, sgb = number(row["SGL"]), number(row["SGB"])
        if not all(np.isfinite(x) for x in (dm, error, vcmb, sgl, sgb)) or error <= 0:
            skipped["missing_measurement"] += 1
            continue
        if group not in native or not (15. <= vcmb/100. <= 180.):
            # The native set is the authoritative CF4 training/holdout and
            # position support. This velocity cut is only a broad prefilter;
            # the exact comoving-distance cut follows below.
            skipped["nonnative_or_range"] += 1
            continue
        if t17_by_cfgroup.get(group, set()) & fp_t17 or pgc_by_cfgroup.get(group, set()) & fp_pgc:
            skipped["fp_source_overlap"] += 1
            continue
        selected.append((row, t17_by_cfgroup.get(group, set())))
    return selected, skipped


def close_holdout(rows, native):
    """Promote TF groups sharing an observed T17 component to one split."""
    by_t17 = {}
    for i, (row, labels) in enumerate(rows):
        for label in labels:
            by_t17.setdefault(label, set()).add(i)
    holdout = {i for i, (row, _) in enumerate(rows) if native[int(row["1PGC"])]}
    while True:
        old = len(holdout)
        touched = set().union(*(labels for i, (_, labels) in enumerate(rows) if i in holdout)) if holdout else set()
        for label in touched:
            holdout.update(by_t17[label])
        if len(holdout) == old:
            break
    return np.array([i in holdout for i in range(len(rows))], dtype=bool)


def make_catalogue():
    groups, galaxies = source_rows()
    with np.load(NATIVE, allow_pickle=False) as f:
        native = dict(zip(map(int, f["CF4_pgc"]), map(bool, f["CF4_holdout"]), strict=True))
    with np.load(FP, allow_pickle=False) as f:
        fp_pgc = set(map(int, f["PGC"]))
        fp_t17 = {int(v[1:]) for v in f["source_group"] if str(v).startswith("T")}
    rows, skipped = disjoint_tf(groups, galaxies, native, fp_pgc, fp_t17)
    ztable = np.linspace(0., .2, 20001)
    dtable = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+ztable)**3+.69), ztable, initial=0.)
    zobs = np.array([number(row["Vcmb"])/299792.458 for row, _ in rows])
    distance = np.interp(zobs, ztable, dtable, left=np.nan, right=np.nan)
    valid = (np.isfinite(distance) & (distance > 15.) & (distance < 180.))
    skipped["exact_distance_range"] = int((~valid).sum())
    rows = [pair for pair, ok in zip(rows, valid, strict=True) if ok]
    hold = close_holdout(rows, native)
    if not rows or not np.any(~hold) or not np.any(hold):
        raise ValueError("TF-only source split empty")
    group = np.array([int(row["1PGC"]) for row, _ in rows], dtype=np.int64)
    if len(np.unique(group)) != len(group):
        raise ValueError("duplicated CF4 TF-only group")
    l, b = (np.deg2rad(np.array([number(row[key]) for row, _ in rows]))
            for key in ("SGL", "SGB"))
    directions = np.column_stack((np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)))
    packed = dict(group_pgc=group, member_count=np.array([int(row["o_DMtf"]) for row, _ in rows]),
                  modulus=np.array([number(row["DMtf"]) for row, _ in rows]),
                  modulus_error=np.array([number(row["e_DMtf"]) for row, _ in rows]),
                  observed_cz=np.array([number(row["Vcmb"]) for row, _ in rows]),
                  directions=directions, holdout=hold)
    return packed, skipped, dict(source_groups=len(groups), source_galaxies=len(galaxies),
                                 native_groups=len(native), fp_rows=len(fp_pgc), fp_t17=len(fp_t17),
                                 exact_distance_range_cMpc_h=[15., 180.])


def field_control(catalogue):
    with np.load(STATE, allow_pickle=False) as f:
        rho = jnp.asarray(f["rho"], dtype=jnp.float64)
        velocity = jnp.asarray(f["velocity_km_s"], dtype=jnp.float64)
    if rho.shape != (128, 128, 128) or velocity.shape != (3, 128, 128, 128):
        raise ValueError("archived PM state geometry changed")
    directions, cz, dm, error, held = (jnp.asarray(catalogue[k]) for k in
        ("directions", "observed_cz", "modulus", "modulus_error", "holdout"))
    ztable = np.linspace(0., .2, 20001)
    dtable = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+ztable)**3+.69), ztable, initial=0.)

    def run(order, amplitude, density_field, velocity_field):
        roots, weights = np.polynomial.legendre.leggauss(order)
        nodes = (roots+1.)*.5*(191.-.1)+.1
        weights = weights*.5*(191.-.1)
        zcos = np.interp(nodes, dtable, ztable)
        return tf_group_logratios(density_field, velocity_field*amplitude, directions, cz, dm, error,
            jnp.asarray(nodes), jnp.asarray(weights), jnp.asarray(zcos))

    evaluate257 = jax.jit(lambda amplitude, density_field, velocity_field:
                          run(257, amplitude, density_field, velocity_field))
    values = np.asarray(evaluate257(1., rho, velocity))
    finer = np.asarray(jax.jit(lambda density_field, velocity_field:
                               run(513, 1., density_field, velocity_field))(rho, velocity))
    zero = np.asarray(evaluate257(0., rho, velocity))
    if not all(np.isfinite(x).all() for x in (values, finer, zero)):
        raise FloatingPointError("nonfinite TF mark conditional score")
    train = ~np.asarray(held)
    def train_score(amplitude):
        return jnp.where(held, 0., evaluate257(amplitude, rho, velocity)).sum()
    value, derivative = map(float, jax.jit(jax.value_and_grad(train_score))(1.))
    eps = 1e-4
    difference = float((train_score(1.+eps)-train_score(1.-eps))/(2*eps))
    if not np.isclose(derivative, difference, rtol=3e-3, atol=1e-3):
        raise AssertionError("TF field-amplitude derivative mismatch")
    return dict(train_logratio=value, holdout_logratio=float(values[~train].sum()),
                zero_velocity_train_logratio=float(zero[train].sum()),
                velocity_amplitude_derivative=derivative,
                velocity_amplitude_finite_difference=difference,
                quadrature_max_group_abs_change=float(np.max(np.abs(finer-values))),
                quadrature_train_logratio_change=float(np.sum(finer[train]-values[train])),
                score_q257_min_max=[float(values.min()), float(values.max())],
                numerical_quadrature_certified=False)


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    catalogue, skipped, source = make_catalogue()
    numerical = field_control(catalogue)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "tf_only_groups.npz", **catalogue)
    report = dict(classification="CORRECTED_TF_ONLY_CONDITIONAL_MARK_DEVELOPMENT_NOT_CALIBRATED",
                  job_id=os.environ["SLURM_JOB_ID"], source=source,
                  skipped=skipped, tf_groups=len(catalogue["group_pgc"]),
                  train_groups=int((~catalogue["holdout"]).sum()),
                  holdout_groups=int(catalogue["holdout"].sum()),
                  tf_measurements_summarized=int(catalogue["member_count"].sum()),
                  min_max_modulus_error=[float(catalogue["modulus_error"].min()),
                                         float(catalogue["modulus_error"].max())],
                  numerical=numerical, state=str(STATE),
                  assumptions=dict(group_TF_Gaussian=True, source_TF_bias_correction_reapplied=False,
                                   TF_and_FP_observations_disjoint=True, selected_density_bias=1.,
                                   group_sigma_v_km_s=150., group_inclusion_calibrated=False,
                                   group_source_covariance_calibrated=False,
                                   output_is_posterior=False),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (GROUPS, GALAXIES, NATIVE, FP, STATE)},
                  code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), ROOT / "src/cf4_r2_tf_group_marks.py")},
                  runtime_seconds=time.monotonic()-started,
                  host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
