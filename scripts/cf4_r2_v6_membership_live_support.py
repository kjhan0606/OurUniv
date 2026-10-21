"""Exact v6 source-membership audit plus a bounded dynamic-support stress test.

Heldout distance marks are not used. The support test evaluates only 48
training, catalogue-ungrouped links across three velocity-scaled copies of one
unconditional N128 state; it is not a live-field posterior or calibration.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_r2_fp_distance import fp_log_likelihood_ratio
from cf4_r2_linked_fp_sparse_train import load_train_singletons, source_neighborhoods
from cf4_r2_marked_tracer_jax import (
    conditional_single_link_logfactor, intrinsic_biased_source_masses,
    intrinsic_lf_bin_fractions, predict_source_marked_radial_key_density,
)
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells

BASE = Path("/gpfs/kjhan/CF4/z0_density")
OUT = BASE / "r2_v6_membership_live_support_v1"
SPLIT = BASE / "r2_sky_closed_split_v6/split.npz"
POINTS = BASE / "r2_point_mark_manifest_v1/points.npz"
OBS = BASE / "r2_source_observation_assembly_v1/observations.npz"
GROUP = BASE / "r2_hierarchical_field_geometry_v1/geometry_q257.npz"
SOURCE = BASE / "r2_marked_source_geometry_v1/geometry.npz"
STATE = BASE / "r2_pm128_unconditional_v1/state.npz"
SDSS = Path("/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat")
CROSSMATCH = ROOT / "data/cf4_2mpp_crossmatch_v1.csv"
COSMO = ROOT / "config/cf4_r2_common_cosmology_v1.json"
C = 299792.458
N, BOX = 128, 384.
VELOCITY_SCALES = (0.5, 1.0, 1.5)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qabs(values):
    values = np.abs(np.asarray(values, dtype=float))
    return dict(median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                maximum=float(np.max(values))) if len(values) else None


def dynamic_support_check(labels, options, fp, point):
    """Rebuild periodic source neighborhoods for each bounded velocity state."""
    membership = {str(row["internal_label"]): row["membership_state"] for row in labels}
    eligible = [o for o in options if membership[o[0]] in (
        ["source_ungrouped_catalogue_present"], ["source_ungrouped_catalogue_absent"])]
    if len(eligible) != 429:
        raise ValueError(f"support training subset changed: {len(eligible)} != 429")
    eligible.sort(key=lambda o: (float(point["radius_cMpc_h"][o[2]]), o[0]))
    chosen = []
    for chunk in np.array_split(np.arange(len(eligible)), 48):
        chosen.append(min((eligible[int(k)] for k in chunk),
                          key=lambda o: hashlib.sha256(o[0].encode()).hexdigest()))

    from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax
    with np.load(STATE, allow_pickle=False) as f:
        rho, velocity = jnp.asarray(f["rho"], dtype=jnp.float64), jnp.asarray(f["velocity_km_s"], dtype=jnp.float64)
    with np.load(SOURCE, allow_pickle=False) as f:
        source = {k: jnp.asarray(f[k]) for k in f.files}
    cosmology = json.loads(COSMO.read_text())["common_cosmology"]
    rho_cell, velocity_cell = native_mass_momentum_to_count_cells(rho, velocity, BOX)
    fraction = jnp.sum(intrinsic_lf_bin_fractions()[1:4])
    mass = intrinsic_biased_source_masses(rho_cell, jnp.log(fraction), jnp.ones(5),
                                          reference_interval=(-25., -21.))
    source_velocity = jnp.moveaxis(velocity_cell, 0, -1).reshape(-1, 3)
    observer = jnp.full(3, BOX/2.)
    args = dict(observer=observer, box_size_cMpc_h=BOX,
        hubble_km_s_Mpc=cosmology["H0_km_s_Mpc"], little_h=cosmology["h"],
        radius_table_cMpc_h=source["radial_table"], modulus_table_h=source["modulus_table"],
        redshift_table=source["redshift_table"], grid_size=N, sigma_los_km_s=100.,
        radial_min_cMpc_h=5., radial_max_cMpc_h=180.)

    scorers = {}
    for pop in range(6):
        def make_score(population):
            def score(pos, vel, src_mass, angular, voxel, radius, dz, mean, std, alpha):
                density = predict_source_marked_radial_key_density(
                    pos, vel, src_mass, angular, population, voxel, radius, **args)
                rel = (pos-observer+BOX/2.) % BOX-BOX/2.
                eta = jnp.log10(dz/jnp.linalg.norm(rel, axis=1))
                mark = fp_log_likelihood_ratio(eta, 0., mean, std, alpha)
                return (conditional_single_link_logfactor(
                    density, jnp.zeros_like(density), jnp.broadcast_to(mark[None, :], density.shape)),
                    jnp.sum(density, axis=1))
            return jax.jit(score)
        scorers[pop] = make_score(pop)

    shifted_fn = jax.jit(lambda p, v: observer_centred_spherical_rsd_jax(
        p, v, observer, BOX, cosmology["H0_km_s_Mpc"], little_h=cosmology["h"], scale_factor=1.)[0])
    results = []
    for scale in VELOCITY_SCALES:
        started = time.monotonic()
        scaled_velocity = scale * source_velocity
        shifted = shifted_fn(source["positions"], scaled_velocity)
        shifted.block_until_ready()
        neighborhoods, _sigma, _radius = source_neighborhoods(
            np.asarray(shifted), chosen, point, cosmology["H0_km_s_Mpc"], cosmology["h"])
        width = 1 << int(math.ceil(math.log2(max(map(len, neighborhoods)))))
        errors = []
        for j, option in enumerate(chosen):
            label, _g, i, _row, train_i = option
            ids = neighborhoods[j]
            padded = np.full(width, int(ids[0]), dtype=np.int32)
            padded[:len(ids)] = ids
            active = jnp.asarray(np.arange(width) < len(ids), dtype=jnp.float64)
            ix, pop = jnp.asarray(padded), int(point["population"][i])
            voxel = np.asarray(np.unravel_index(int(point["flat_cell"][i]), (N,)*3), dtype=np.int32)
            vals = (voxel, float(point["radius_cMpc_h"][i]), float(fp["dz_row"][train_i]),
                    float(fp["eta_mean"][train_i]), float(fp["eta_std"][train_i]),
                    float(fp["eta_alpha"][train_i]))
            local = scorers[pop](source["positions"][ix], scaled_velocity[ix], mass[:, ix]*active[None, :],
                source["angular"][:, ix], *vals)
            full = scorers[pop](source["positions"], scaled_velocity, mass, source["angular"], *vals)
            lf, lb = float(local[0]), np.asarray(local[1])
            ff, fb = float(full[0]), np.asarray(full[1])
            ferr = abs(lf-ff)
            berr = float(np.max(np.abs(lb-fb)/np.maximum(np.abs(fb), 1e-30)))
            if ferr > 1e-8 or berr > 1e-8 or not np.isfinite([ferr, berr]).all():
                raise AssertionError(f"dynamic sparse/full mismatch at {label}, scale={scale}")
            errors.append((ferr, berr, len(ids)))
        results.append(dict(velocity_scale=scale, groups=len(chosen),
            candidate_count_p50=float(np.median([x[2] for x in errors])),
            candidate_count_max=int(max(x[2] for x in errors)),
            max_logfactor_abs_error=max(x[0] for x in errors),
            max_density_sum_relative_error=max(x[1] for x in errors),
            elapsed_seconds=time.monotonic()-started))
    return dict(status="PASS_FIXED_SUBSET_MECHANICS_ONLY", selection="48 radius-rank bins, stable hash within bin, no marks used",
                velocity_scales=list(VELOCITY_SCALES), scales=results,
                limit="Only velocity-scaled copies of one unconditional state; not posterior states, selection calibration, sampler support, or R2 completion.")


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU required")
    expected = os.environ.get("CF4_EXPECTED_COMMIT")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not expected or expected != head:
        raise RuntimeError(f"source commit is not pinned: expected={expected}, HEAD={head}")
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()

    with np.load(SPLIT, allow_pickle=False) as f:
        labels, roles, recno = f["fp_source_group"].astype(str), f["fp_role"].astype(int), f["point_recno"].copy()
    with np.load(OBS, allow_pickle=False) as f:
        pgc, source_group = f["PGC"].astype(int), f["source_group"].astype(str)
        states, richness = f["membership_state"].astype(str), f["source_richness"].astype(int)
    with np.load(GROUP, allow_pickle=False) as f:
        np.testing.assert_array_equal(labels, f["group_labels"].astype(str))
        row_group, anchor_group = f["row_group"].astype(int), f["anchor_group"].astype(int)
    with np.load(POINTS, allow_pickle=False) as f:
        np.testing.assert_array_equal(recno, f["recno"])
        point = {k: f[k].copy() for k in ("recno", "population", "flat_cell", "radius_cMpc_h", "vcmb_km_s")}

    # Construct an exact source-group -> unique PGC and v6 role map.
    group_rows = [[] for _ in labels]
    pgc_group, meta = {}, {}
    for r, g in enumerate(row_group):
        g, p = int(g), int(pgc[r])
        if p in pgc_group or source_group[r] != labels[g]:
            raise ValueError("duplicate PGC or source-group alignment error")
        group_rows[g].append(r)
        pgc_group[p] = g
        meta[p] = dict(role=int(roles[g]), state=str(states[r]), richness=int(richness[r]),
                       IDgroupT17=int(labels[g][1:]) if labels[g].startswith("T") else 0)

    # Secure count-point links and exact four-flag source manifest.
    rec_index = {int(r): i for i, r in enumerate(recno)}
    linked = {}
    with CROSSMATCH.open(newline="", encoding="utf-8") as stream:
        for edge in csv.DictReader(stream):
            if edge["match_class"] == "secure_joint_mark" and edge["twompp_recno"]:
                g, i = pgc_group.get(int(edge["PGC"])), rec_index.get(int(edge["twompp_recno"]))
                if g is not None and i is not None:
                    linked.setdefault(g, set()).add(i)
    nrow, nanchor = np.bincount(row_group, minlength=len(labels)), np.bincount(anchor_group, minlength=len(labels))
    role_name = {0: "train", 1: "heldout", 2: "buffer", 3: "unassigned_Tempel_GID"}
    manifest, linked_candidates = [], []
    by_role_state = {}
    for g, label in enumerate(labels):
        member_rows = group_rows[g]
        member_states = sorted(set(states[member_rows].tolist()))
        member_richness = sorted(set(map(int, richness[member_rows])))
        points = sorted(linked.get(g, set()))
        one_radius = float(point["radius_cMpc_h"][points[0]]) if len(points) == 1 else None
        flags = dict(train_role=bool(roles[g] == 0), exactly_one_secure_countpoint=len(points) == 1,
                     exactly_one_FP_row=int(nrow[g]) == 1, no_CF4_anchor=int(nanchor[g]) == 0)
        role = role_name.get(int(roles[g]), f"unknown_{roles[g]}")
        entry = dict(internal_label=str(label), role=role, membership_state=member_states,
                     published_IDgroupT17=[int(label[1:]) if label.startswith("T") else 0],
                     published_NgroupT17=member_richness, source_PGCs=[int(pgc[r]) for r in member_rows],
                     secure_linked_countpoint_recno=[int(recno[i]) for i in points],
                     FP_row_count=int(nrow[g]), CF4_anchor_count=int(nanchor[g]),
                     linked_countpoint_radius_cMpc_h=one_radius,
                     one_link_in_5_180_cMpc_h_factor_window=bool(one_radius is not None and 5 <= one_radius <= 180),
                     graph_singleton_flags=flags)
        manifest.append(entry)
        state = member_states[0] if len(member_states) == 1 else "mixed_or_empty"
        key = f"{role}|{state}"
        by_role_state[key] = by_role_state.get(key, 0) + 1
        strict = (len(points) == 1 and int(nrow[g]) == 1 and int(nanchor[g]) == 0
                  and one_radius is not None and 5 <= one_radius <= 180)
        if strict and role in ("train", "heldout", "buffer"):
            state_count_key = (role, state)
            # Count all strict graph-linked roles, while the likelihood pilot
            # below keeps only catalogue-ungrouped training rows.
            by_role_state[f"strict|{role}|{state}"] = by_role_state.get(f"strict|{role}|{state}", 0) + 1
        if strict and role == "train" and state in (
                "source_ungrouped_catalogue_present", "source_ungrouped_catalogue_absent"):
            i, r = points[0], member_rows[0]
            linked_candidates.append((str(label), g, i, r, int(pgc[r]), float(one_radius)))

    if len(linked_candidates) != 429:
        raise ValueError(f"strict v6 training ungrouped count changed: {len(linked_candidates)} != 429")
    expected_roles = {"strict|train|source_ungrouped_catalogue_present": 424,
                      "strict|train|source_ungrouped_catalogue_absent": 5,
                      "strict|heldout|source_ungrouped_catalogue_present": 73,
                      "strict|buffer|source_ungrouped_catalogue_present": 6}
    for key, count in expected_roles.items():
        if by_role_state.get(key, 0) != count:
            raise ValueError(f"v6 role/membership regression {key}: {by_role_state.get(key, 0)} != {count}")

    # Read original mark fields only for v6-train PGCs; never access the
    # heldout rows in the compressed eta arrays for this statistics table.
    train_meta = {p: m for p, m in meta.items() if m["role"] == 0}
    raw_md5, raw_sha = hashlib.md5(), hashlib.sha256()
    train_rows = {}
    with Path("/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat").open("rb") as stream:
        header = stream.readline(); raw_md5.update(header); raw_sha.update(header)
        names = header.decode("ascii").lstrip("#").split(); col = {n: i for i, n in enumerate(names)}
        required = ("PGC", "IDgroupT17", "NgroupT17", "zcmb", "zcmb_group", "deVMag_r",
                    "logdist_corr", "logdist_corr_err", "logdist_corr_alpha")
        if any(n not in col for n in required):
            raise ValueError("SDSS-PV source schema changed")
        for line in stream:
            raw_md5.update(line); raw_sha.update(line)
            vals = line.split()
            if not vals:
                continue
            p = int(vals[col["PGC"]]); expected = train_meta.get(p)
            if expected is None:
                continue
            group_id, rich = int(vals[col["IDgroupT17"]]), int(vals[col["NgroupT17"]])
            if (group_id, rich) != (expected["IDgroupT17"], expected["richness"]):
                raise ValueError(f"published membership mismatch for PGC {p}")
            train_rows[p] = dict(state=expected["state"], z=float(vals[col["zcmb"]]),
                zgroup=float(vals[col["zcmb_group"]]), mag=float(vals[col["deVMag_r"]]),
                eta=float(vals[col["logdist_corr"]]), sigma=float(vals[col["logdist_corr_err"]]),
                alpha=float(vals[col["logdist_corr_alpha"]]))
    if raw_md5.hexdigest() != "b5b6e31caf7ea469c2ac2cb775fa8d14" or set(train_rows) != set(train_meta):
        raise ValueError("SDSS source hash or training-only join changed")

    strata = {}
    bins = {}
    for p, row in train_rows.items():
        state = row["state"]
        category = ("present_singleton" if state == "source_ungrouped_catalogue_present" else
                    "grouped" if state == "source_grouped_catalogue_present" else
                    "catalogue_absent" if state == "source_ungrouped_catalogue_absent" else "unresolved")
        strata.setdefault(category, []).append(row)
        key = (int(C * row["z"] // 1000), int(row["mag"] // .5))
        bins.setdefault(key, {}).setdefault(category, []).append(row)

    def moments(rows):
        return dict(n=len(rows), eta_mean=float(np.mean([r["eta"] for r in rows])),
                    eta_std_mean=float(np.mean([r["sigma"] for r in rows]))) if rows else dict(n=0)

    def contrast(a, b):
        paired = [(min(len(x.get(a, [])), len(x.get(b, []))),
                   np.mean([r["eta"] for r in x[a]]) - np.mean([r["eta"] for r in x[b]]),
                   np.mean([r["sigma"] for r in x[a]]) - np.mean([r["sigma"] for r in x[b]]))
                  for x in bins.values() if x.get(a) and x.get(b)]
        if not paired:
            return dict(common_bins=0, matched_row_pairs=0, delta_eta_mean=None, delta_eta_std=None)
        w = np.asarray([x[0] for x in paired], dtype=float)
        return dict(common_bins=len(w), matched_row_pairs=int(w.sum()),
                    delta_eta_mean=float(np.average([x[1] for x in paired], weights=w)),
                    delta_eta_std=float(np.average([x[2] for x in paired], weights=w)))

    # Reuse the existing full-source fixed-state check; its sparse indices are
    # explicitly not treated as valid for a changing field state.
    prior_support = Path("/gpfs/kjhan/CF4/z0_density/r2_linked_fp_sparse_train_v6_fullcheck_v1/result.json")
    prior = json.loads(prior_support.read_text())
    if prior.get("job_id") != "407082" or prior.get("status") != "COMPLETED_V6_FIXED_STATE_FULL_SOURCE_TRAINING_COMPARISON_NOT_CALIBRATION":
        raise ValueError("pinned v6 full-source support result changed")

    sparse_options, sparse_point, sparse_fp = load_train_singletons(SPLIT)
    dynamic_support = dynamic_support_check(manifest, sparse_options, sparse_fp, sparse_point)

    OUT.mkdir(parents=True)
    with (OUT / "v6_group_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    inputs = (SPLIT, POINTS, OBS, GROUP, SOURCE, STATE, CROSSMATCH, COSMO, prior_support)
    input_hashes = {str(p): sha256(p) for p in inputs}
    (OUT / "v6_group_manifest_metadata.json").write_text(json.dumps(dict(
        role_codes=role_name, input_sha256=input_hashes, SDSS_source_sha256=raw_sha.hexdigest(),
        heldout_eta_numeric_fields_parsed=0), indent=2) + "\n")

    # Redshift residual is between two different source measurements; report
    # both rather than collapsing them to one exact value.
    candidates = {p: (label, i) for label, _g, i, _r, p, _rad in linked_candidates}
    fp_dcz, count_dcz = [], []
    with np.load(POINTS, allow_pickle=False) as f:
        vcmb = f["vcmb_km_s"].copy()
    for p, (label, i) in candidates.items():
        row = train_rows[p]
        fp_dcz.append(abs(C * (row["zgroup"] - row["z"])))
        count_dcz.append(float(vcmb[i] - C * row["z"]))
    abs_count_dcz = np.abs(count_dcz)
    report = dict(classification="R2_V6_MEMBERSHIP_STRATA_AND_DYNAMIC_SUPPORT_DIAGNOSTIC_NOT_POSTERIOR",
        status="COMPLETED", job_id=os.environ["SLURM_JOB_ID"], source_commit=head,
        Q_GOAL="Correct source semantics and support are prerequisites for the same-field CF4 present-state posterior; this bundle is diagnostic only.",
        Q_LEAN="One v6 manifest, a train-only descriptive screen, and reuse of the completed full-source check; no new posterior, simulation, or repeated path sweep.",
        v6_groups=len(labels), membership_group_counts=by_role_state,
        strict_ungrouped_linked_counts={"train": 429, "heldout": 73, "buffer": 6},
        train_rows_with_eta_parsed=len(train_rows), heldout_eta_numeric_fields_parsed=0,
        training_richness_strata={k: moments(v) for k, v in strata.items()},
        cz_bin_km_s=1000, apparent_r_mag_bin=.5,
        matched_bin_descriptive_contrasts={"present_singleton_minus_grouped": contrast("present_singleton", "grouped"),
                                           "catalogue_absent_minus_grouped": contrast("catalogue_absent", "grouped")},
        interpretation="Observed-cz/magnitude bins do not control true distance; these contrasts cannot establish eta independence, inclusion probability, or calibration.",
        train_ungrouped_FP_group_vs_individual_cz_abs_km_s={"median": float(np.median(fp_dcz)), "p90": float(np.quantile(fp_dcz, .9)), "max": float(np.max(fp_dcz))},
        train_ungrouped_2Mpp_minus_individual_cz_km_s={"signed_median": float(np.median(count_dcz)),
            "abs_median": float(np.median(abs_count_dcz)), "abs_p90": float(np.quantile(abs_count_dcz, .9)),
            "abs_max": float(np.max(abs_count_dcz)), "n_abs_gt25": int(np.sum(abs_count_dcz > 25)),
            "n_abs_gt50": int(np.sum(abs_count_dcz > 50)), "n_abs_gt100": int(np.sum(abs_count_dcz > 100))},
        fixed_state_support_reused={"job_id": prior["job_id"], "groups": prior["training_one_countpoint_one_FProw_groups"],
            "elapsed_s": prior["elapsed_seconds"], "max_logfactor_abs_error": prior["largest_full_source_discrepancies"]["conditional_logfactor_abs"][0],
            "max_density_sum_relative_error": prior["largest_full_source_discrepancies"]["trueK_density_relative"][0]},
        bounded_dynamic_support_stress=dynamic_support,
        live_posterior_support_status="Not tested: rebuild the RSD-shifted periodic neighborhood for every proposed posterior state; do not reuse fixed-state indices. The stress uses velocity-scaled copies of a single unconditional state.",
        selection_association_calibrated=False, FP_covariance_calibrated=False,
        field_fit=False, sampler=False, R2_posterior=False, N256=False, new_gravity_runs=0,
        MW_M31_M33="Same new evolved field only; MW/M31 roles stay ambiguous and M33 unresolved when unsupported; all observables constrain that field; native truth labels are evaluation-only.",
        input_sha256=input_hashes, SDSS_source_sha256=raw_sha.hexdigest(),
        elapsed_seconds=time.monotonic() - started,
        host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
