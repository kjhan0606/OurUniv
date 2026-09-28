"""Pre-registered bounded SDSS-PV mock subset; retain summaries only.

Select all eight observers of the first simulation realization in tar order,
plus the first catalogue from each of the next seven distinct realizations.
Selection depends on archive names/order only, never on eta or project holdouts.
This tests the SDSS individual-FP/host-velocity component, not Tempel or CF4
group selection, 2M++ FoG, or the R2 posterior.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tarfile
import time
from urllib.request import urlopen

import numpy as np
from scipy.stats import skewnorm

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4")
OUT = BASE / "z0_density/r2_sdss_mock_15subset_v1"
URL = "https://zenodo.org/api/records/6824749/files/mocks.tar.gz/content"
COMPRESSED_CAP = 512 * 1024**2
MEMBER_CAP = 128 * 1024**2
NOMINAL = ((.68, .16, .84), (.90, .05, .95), (.95, .025, .975))
Z_BINS = ((0., .03), (.03, .06), (.06, .2))
OBSERVER_XYZ = np.array([[250, 260, 300], [250, 260, 1200], [250, 1160, 300],
                         [250, 1160, 1200], [1150, 260, 300], [1150, 260, 1200],
                         [1150, 1160, 300], [1150, 1160, 1200]], dtype=float)


class BoundedReader:
    def __init__(self, stream):
        self.stream, self.count = stream, 0

    def read(self, n=-1):
        remaining = COMPRESSED_CAP - self.count
        if remaining <= 0:
            raise RuntimeError("512 MiB compressed stream cap reached; do not fall back to full archive")
        request = min(n if n >= 0 else 65536, remaining + 1)
        data = self.stream.read(request)
        self.count += len(data)
        if self.count > COMPRESSED_CAP:
            raise RuntimeError("512 MiB compressed stream cap exceeded; partial output is rejected")
        return data


def source_pdf_cdf(truth, mean, std, alpha):
    delta = alpha / np.sqrt(1 + alpha**2)
    scale = std / np.sqrt(1 - 2 * delta**2 / np.pi)
    loc = mean - scale * delta * np.sqrt(2 / np.pi)
    return skewnorm.cdf(truth, alpha, loc=loc, scale=scale)


def coverage(values):
    out = {}
    n = int(len(values))
    for level, lo, hi in NOMINAL:
        observed = float(np.mean((values >= lo) & (values <= hi))) if n else None
        se = float(np.sqrt(level * (1-level) / n)) if n else None
        out[str(int(level*100))] = dict(n=n, fraction=observed, nominal=level,
            binomial_se=se, within_2se=bool(abs(observed-level) <= 2*se) if n else None)
    return out


def summarize_member(name, raw):
    match = re.search(r"_R(\d+)\.(\d+)_err_corr$", name)
    if not match:
        raise ValueError(f"unexpected mock member name: {name}")
    box_id, observer = match.group(1), int(match.group(2))
    if observer not in range(8):
        raise ValueError(f"observer ID outside documented eight-view set: {name}")
    data = np.genfromtxt(io.BytesIO(raw), names=True, delimiter=",", encoding=None)
    required = ("z_true", "z_obs", "cen_flag", "parenthalomass", "x", "y", "z",
                "vx", "vy", "vz", "vxcen", "vycen", "vzcen", "logdist_true",
                "logdist", "logdist_err", "logdist_alpha")
    if data.dtype.names is None or any(k not in data.dtype.names for k in required):
        raise ValueError(f"mock schema changed for {name}")
    arrays = [np.asarray(data[k], dtype=float) for k in required]
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError(f"non-finite mock values in {name}")
    if np.any(data["logdist_err"] <= 0) or np.any(data["parenthalomass"] <= 0):
        raise ValueError(f"invalid uncertainty/mass in {name}")
    central = np.asarray(data["cen_flag"] == 1)
    if np.any(~np.isin(data["cen_flag"], [0, 1])):
        raise ValueError(f"invalid central flag in {name}")
    cdf = source_pdf_cdf(data["logdist_true"], data["logdist"],
                         data["logdist_err"], data["logdist_alpha"])
    if not np.isfinite(cdf).all():
        raise ValueError(f"invalid source-PDF CDF in {name}")
    p = np.column_stack([data[k] for k in ("x", "y", "z")]) - OBSERVER_XYZ[observer]
    direction = p / np.linalg.norm(p, axis=1)[:, None]
    dv = np.column_stack([data["vx"]-data["vxcen"], data["vy"]-data["vycen"],
                          data["vz"]-data["vzcen"]])
    relative_los = np.sum(dv * direction, axis=1)
    logmass = np.log10(data["parenthalomass"])
    medges = np.unique(np.quantile(logmass, [0., 1/3, 2/3, 1.]))
    strata = []

    def add_strata(label, mask, dimension, edges, edge_labels):
        for lo, hi, edge_name in zip(edges[:-1], edges[1:], edge_labels):
            sel = mask & (dimension >= lo) & (dimension < hi)
            idx = np.flatnonzero(sel)
            strata.append(dict(stratum=edge_name, n=int(len(idx)),
                eta_bias_mean=float(np.mean(data["logdist"][idx]-data["logdist_true"][idx])) if len(idx) else None,
                standardized_residual_mean=float(np.mean((data["logdist"][idx]-data["logdist_true"][idx]) /
                                                        data["logdist_err"][idx])) if len(idx) else None,
                coverage=coverage(cdf[idx])))

    populations = []
    for pop, mask in (("all", np.ones(len(data), dtype=bool)),
                      ("central", central), ("satellite", ~central)):
        q = cdf[mask]
        v = relative_los[mask]
        populations.append(dict(population=pop, n=int(mask.sum()), coverage=coverage(q),
            host_relative_los_km_s=dict(rms=float(np.sqrt(np.mean(v*v))),
                abs_p50_p90_p99=np.quantile(np.abs(v), [.5, .9, .99]).tolist(),
                empirical_abs_le_100=float(np.mean(np.abs(v) <= 100.)),
                empirical_abs_le_200=float(np.mean(np.abs(v) <= 200.)),
                gaussian_sigma100_reference=dict(abs_le_100=0.682689492,
                    abs_le_200=0.954499736))))
        if pop == "all":
            add_strata(pop, mask, np.asarray(data["z_true"]),
                       np.array([0., .03, .06, .2]), ["z_lt_.03", ".03_to_.06", "z_ge_.06"])
            add_strata(pop, mask, logmass, medges,
                       [f"hostmass_q{i+1}" for i in range(len(medges)-1)])
    return dict(member=name, realization=box_id, observer=observer, rows=int(len(data)),
        sha256=hashlib.sha256(raw).hexdigest(), log10_hostmass_edges=medges.tolist(),
        populations=populations, all_population_strata=strata,
        satellite_host_relative_los_abs_p50_p90_p99=np.quantile(np.abs(relative_los[~central]), [.5, .9, .99]).tolist(),
        satellite_host_relative_los_rms=float(np.sqrt(np.mean(relative_los[~central]**2))),
        central_host_relative_los_max_abs=float(np.max(np.abs(relative_los[central]))) if np.any(central) else None)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm required")
    expected = os.environ.get("CF4_EXPECTED_COMMIT")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not expected or head != expected:
        raise RuntimeError(f"source commit is not pinned: {expected} != {head}")
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    summaries, anchor_box, anchor_observers, other_boxes = [], None, set(), set()
    first_member = None
    with urlopen(URL, timeout=180) as response:
        stream = BoundedReader(response)
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith("_err_corr"):
                    continue
                match = re.search(r"_R(\d+)\.(\d+)_err_corr$", member.name)
                if not match:
                    continue
                box_id, observer = match.group(1), int(match.group(2))
                if anchor_box is None:
                    anchor_box = box_id
                    first_member = member.name
                choose = False
                if box_id == anchor_box and observer in range(8) and observer not in anchor_observers:
                    choose = True
                    anchor_observers.add(observer)
                elif box_id != anchor_box and box_id not in other_boxes and len(other_boxes) < 7:
                    choose = True
                    other_boxes.add(box_id)
                if not choose:
                    continue
                if member.size > MEMBER_CAP:
                    raise ValueError(f"selected mock member exceeds 128 MiB: {member.name}")
                fileobj = archive.extractfile(member)
                if fileobj is None:
                    raise ValueError(f"cannot stream selected member: {member.name}")
                raw = fileobj.read(MEMBER_CAP + 1)
                if len(raw) != member.size or len(raw) > MEMBER_CAP:
                    raise ValueError(f"truncated/oversize selected member: {member.name}")
                summaries.append(summarize_member(member.name, raw))
                del raw
                if anchor_observers == set(range(8)) and len(other_boxes) == 7:
                    break
    if len(summaries) != 15 or len(anchor_observers) != 8 or len(other_boxes) != 7:
        raise RuntimeError(f"pre-registered sample not reached within cap: files={len(summaries)}, anchor={len(anchor_observers)}, other_boxes={len(other_boxes)}")
    if len({row["realization"] for row in summaries}) != 8:
        raise ValueError("selection failed to produce eight distinct simulation boxes")

    by_box = {}
    for row in summaries:
        by_box.setdefault(row["realization"], []).append(row)
    box_coverage = []
    for box_id, rows in sorted(by_box.items()):
        for pop in ("all", "central", "satellite"):
            members = [next(p for p in row["populations"] if p["population"] == pop) for row in rows]
            for level, _, _ in NOMINAL:
                key = str(int(level*100))
                values = [x["coverage"][key]["fraction"] for x in members if x["coverage"][key]["fraction"] is not None]
                box_coverage.append(dict(realization=box_id, views=len(rows), population=pop,
                                         level=level, equal_observer_mean=float(np.mean(values))))

    # A descriptive, non-promoting acceptance summary: file-level binomial
    # screens plus replication in distinct simulation boxes.
    file_failures = []
    for row in summaries:
        for pop in row["populations"]:
            for level, _, _ in NOMINAL:
                item = pop["coverage"][str(int(level*100))]
                if item["within_2se"] is False:
                    file_failures.append(dict(member=row["member"], population=pop["population"], level=level))
    trend_checks = []
    for row in summaries:
        for stratum in row["all_population_strata"]:
            if stratum["n"] < 200:
                continue
            for level, _, _ in NOMINAL:
                item = stratum["coverage"][str(int(level*100))]
                trend_checks.append(dict(member=row["member"], realization=row["realization"],
                    stratum=stratum["stratum"], level=level, n=stratum["n"],
                    within_2se=item["within_2se"]))
    independent_box_count = len(by_box)
    # Pool observer views only within the same simulation box, then require
    # every redshift/host-mass stratum to pass in at least two distinct boxes.
    # This is a bounded screen, not a fit or a posterior-promotion gate.
    box_stratum_checks = []
    stratum_names = sorted({
        item["stratum"] for row in summaries for item in row["all_population_strata"]
        if item["n"] >= 200
    })
    for box_id, rows in sorted(by_box.items()):
        for stratum_name in stratum_names:
            matching = [item for row in rows for item in row["all_population_strata"]
                        if item["stratum"] == stratum_name and item["n"] >= 200]
            if not matching:
                continue
            n = sum(item["n"] for item in matching)
            for level, _, _ in NOMINAL:
                key = str(int(level*100))
                successes = sum(round(item["coverage"][key]["fraction"] * item["n"])
                                for item in matching)
                fraction = successes / n
                se = float(np.sqrt(level * (1-level) / n))
                box_stratum_checks.append(dict(realization=box_id, stratum=stratum_name,
                    level=level, n=n, fraction=fraction, binomial_se=se,
                    within_2se=bool(abs(fraction-level) <= 2*se)))
    checked_boxes_by_stratum = {
        name: len({item["realization"] for item in box_stratum_checks
                   if item["stratum"] == name}) for name in stratum_names
    }
    trend_failures = [item for item in box_stratum_checks if not item["within_2se"]]
    replicated_strata = all(count >= 2 for count in checked_boxes_by_stratum.values())
    coverage_screen = ("CONDITIONAL_PASS" if not file_failures and not trend_failures
                       and independent_box_count >= 2 and replicated_strata
                       else "REJECT_OR_INCONCLUSIVE")
    OUT.mkdir(parents=True)
    result = dict(classification="R2_SDSS_15_MEMBER_HOST_MEMBER_CALIBRATION_DIAGNOSTIC_NOT_GROUP_SELECTION",
        job_id=os.environ["SLURM_JOB_ID"], source_url=URL, first_regular_member=first_member,
        selected_members=summaries, compressed_bytes_read=stream.count,
        compressed_cap_bytes=COMPRESSED_CAP, full_archive_checksum_verified=False,
        raw_mock_members_retained=False, independent_boxes=independent_box_count,
        observer_selection="all eight observers of first archive realization; first member from next seven distinct realizations",
        pre_registered_selection=True, selected_using_marks_or_project_holdout=False,
        source_pdf_file_level_2SE_failures=file_failures,
        distance_and_hostmass_stratum_2SE_checks=trend_checks,
        independent_box_coverage=box_coverage,
        pooled_box_stratum_2SE_checks=box_stratum_checks,
        pooled_box_stratum_independent_counts=checked_boxes_by_stratum,
        pooled_box_stratum_2SE_failures=trend_failures,
        source_pdf_coverage_screen=coverage_screen,
        interpretation="The screen tests the SDSS individual-FP/host-member component only. Do not transfer the selected SDSS satellite mixture to 2M++ tracer FoG, Tempel group membership, group-COM scatter, or CF4/2M++ shared covariance. Even PASS does not promote an R2 posterior.",
        Q_GOAL="Calibrates a named individual FP/host-velocity component in the linked conditional factor, not the same-field density posterior.",
        Q_LEAN="At most 512 MiB sequential input, 15 fixed catalogues, summary-only output; no 10.6 GB archive retention, fit, sampler, or gravity run.",
        MW_M31_M33="No truth identities used for field candidate selection; future MW/M31 roles stay ambiguous and M33 unresolved when unsupported on the same evolved field.",
        R2_posterior=False, group_selection_calibrated=False, group_covariance_calibrated=False,
        elapsed_seconds=time.monotonic()-started,
        host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
