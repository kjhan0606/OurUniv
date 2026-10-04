# SDSS-PV mock component diagnostic — 2026-10-05

## Eligibility result

The active linked FP training cohort is a subset of the same public SDSS-PV
catalogue as the candidate mock release: all1,414 selected training PGCs
match exactly as a set. No heldout rows or outcomes were used. The training
cohort's Tempel17 group-size quantiles are `[1,1,3,4,33]` (q00/q25/q50/q75/q100).

One catalogue, `MOCK_HAMHOD_SDSS_v5_R19051.5_err_corr`, was inspected using a
bounded8MiB HTTP range; no archive or mock member was retained. It contains
33,881 selected galaxies, 17,266 composite host groups, 14,775 selected
centrals, and 2,491 host groups with no selected central. The host key
`(parenthalomass,z_obs_cen,vxcen,vycen,vzcen)` is collision-free and maps
one-to-one to the repeated `ID` values in this file. The official PDF labels
`ID` “Mock galaxy ID”, but the file repeats it across rows assigned to one
host. The full-ensemble runner therefore groups by the validated composite
key and checks the ID/key relation independently in every catalogue.

For `e=logdist-logdist_true` and the reported standard deviation `logdist_err`,
the single-catalogue approximate-Gaussian diagnostic is:

| Selected host richness | Galaxies | Groups | Mean `e/err` | SD `e/err` | 68% coverage | 95% coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12,627 | 12,627 | -0.1705 | 0.9518 | 0.7018 | 0.9572 |
| 2–4 | 7,947 | 2,690 | 0.0292 | 0.9951 | 0.6806 | 0.9527 |
| 5–9 | 10,872 | 1,738 | 0.0860 | 0.9918 | 0.6862 | 0.9512 |
| 10+ | 2,435 | 211 | 0.2276 | 1.0450 | 0.6534 | 0.9331 |

The first-pass group-mean variance ratios were centered only by the
catalogue-wide residual mean, so they mixed richness-dependent mean bias with
random group scatter. Those ratios are invalid as covariance estimates and
are intentionally excluded here. The corrected ensemble calculation first
removes each catalogue's mean separately within each richness bin. The
skew-normal `logdist_alpha` remains reported; the preliminary standardized
view does not fit a Gaussian replacement.

## Bundle decision and execution

The alignment is adequate for the narrower purpose of testing SDSS-PV FP
error and within-host covariance. It does **not** calibrate absolute group
inclusion, Tempel17 richness, the 2M++ count likelihood, or the full
multi-survey CF4 observation law. The published mock-selection model also
omits redshift-success effects. The 2,048-catalogue ensemble is therefore a
bounded calibration diagnostic only, not a route to posterior promotion.

The driver implements `scripts/cf4_r2_sdss_mock_stream_summary.py` and its
focused tests. A complete first pass parsed 2,048 catalogue files and wrote
the expected archive MD5, but Slurm job412581 was marked failed after the
summary was written because the final progress print passed `flush` to the
JSON encoder. This is an output-only error; a regression test now covers it.
The group-mean dispersion was also corrected to remove richness-bin means
before estimating residual scatter. The CPU-only Slurm script
`scripts/run_cf4_r2_sdss_mock_stream_summary.sbatch` downloads the official
10.6GB archive into a job-private temporary directory alongside the already
verified source under `/gpfs/kjhan/CF4/external/sdss_pv_6824749`, validates
the published byte count and MD5 while streaming the tar, verifies all
2,048 files /256 simulation boxes /8 observers per box and group-key
invariants, and writes only `result.json` under the CF4 z0-density results.
The temporary archive is removed only after a fully verified successful
summary. Requested memory is2GiB; the parser holds at most one ~34k-row mock
catalogue and its group accumulators. It uses one CPU on `a10`, no GPU GRES,
so scarce H100/H200/A100 GPUs are not reserved for a CPU-only parser.

Q-GOAL: aligned as one observed FP-error/covariance component for the
CF4-conditioned field, but not the field reconstruction itself. Q-LEAN: the
10.6GB source is streamed once after exact source overlap and schema were
verified; raw archive is temporary; no fit, heldout access, gravity, new group
census, or Tempel-finder rewrite. MW/M31 roles remain ambiguous and M33
unresolved; the future constraints still belong on the same NEW evolved field.

Primary sources: [Howlett et al. 2022](https://arxiv.org/abs/2201.03112),
[official SDSS-PV mock release, version1.1.0](https://zenodo.org/records/6824749).

## Corrected full-ensemble result and active-cohort support — job 412614

The table above is one schema fixture, not an ensemble estimate. The corrected
no-redownload rerun completed on the previously staged archive: Slurm
412614 `COMPLETED/0:0` in22:32, batch MaxRSS79,336KiB under2GiB requested.
It parsed2,048 catalogues across256 simulation boxes (eight observers per
box); streamed byte count10,643,218,721 and MD5
`9ba3e8876f6f08a2af00d30cbf1c6cd9` match the release. The output is
`/gpfs/kjhan/CF4/z0_density/r2_sdss_mock_stream_law_412614_20261005/result.json`
and pins source commit `e764a967ab4e49999db12213620ae740a5b841ec`. Its
job-private10.6GB tar and temporary directory were removed after success.
The earlier412581 job remains preserved as a failed attempt: its data pass
completed, but a final progress-print bug caused nonzero exit after JSON
write; its retained archive was reused here, not downloaded again.

| Selected mock-host richness | Galaxies | Host groups | Mean standardized residual | SD | 68% coverage | 95% coverage | Group-mean variance ratio after within-bin centering | 256-box q05 / q50 / q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25,374,667 | 25,374,667 | -0.1771 | 0.9612 | 0.6937 | 0.9551 | 0.9267 | 0.919 / 0.926 / 0.934 |
| 2–4 | 16,498,129 | 5,576,661 | +0.0286 | 0.9915 | 0.6867 | 0.9519 | 0.9816 | 0.966 / 0.981 / 0.999 |
| 5–9 | 21,943,938 | 3,512,807 | +0.0983 | 1.0083 | 0.6765 | 0.9470 | 1.0456 | 1.024 / 1.046 / 1.066 |
| 10+ | 4,320,876 | 378,266 | +0.2251 | 1.0431 | 0.6514 | 0.9338 | 1.1576 | 1.093 / 1.155 / 1.224 |

The active linked training cohort was reconciled against its existing frozen
1,414-row association ledger; its row-file SHA256 matches the saved result.
Tempel17 `NgroupT17` counts at the same numeric cuts are429,670,267,48
(30.34%,47.38%,18.88%,3.39%); quantiles are `[1,1,3,4,33]`. This establishes
numerical support overlap only. Mock selected-host richness is not Tempel17
richness, and there is no mapping between their membership/selection rules.
The first-catalogue group-mean ratios centered only on a catalogue-wide mean
remain invalid and are superseded by this within-richness-centered ensemble
calculation.

**Disposition:** the ensemble shows richness-conditional residual-mean shifts
and a modest richness trend in group-mean scatter relative to independent
reported errors. These statistics are not a covariance matrix and are not
transferred to Tempel groups; neither a likelihood correction nor covariance
inflation is adopted. The exact same-source PGC overlap does not calibrate
Tempel group inclusion, redshift-success,2M++ counts or the heterogeneous
CF4 selection law. No heldout values, field state, fit, posterior, native LG
truth identity or gravity evolution was accessed. R2 remains NO-GO.

Q-GOAL: this supplies a bounded SDSS-PV FP measurement/selected-host residual
component for the same field, not the z=0 density/velocity result. MW/M31
remain role-ambiguous and M33 unresolved; their observables must eventually
constrain those same roles on the same NEW evolved LG field at
`<=0.3 cMpc/h`, with truth identities limited to calibration/evaluation.
Q-LEAN: one full pass, one exact existing-cohort support check, and removal of
the temporary archive are sufficient. Do not repeat the archive pass, run a
Tempel group finder, or fit a correction from unmatched richness bins.
