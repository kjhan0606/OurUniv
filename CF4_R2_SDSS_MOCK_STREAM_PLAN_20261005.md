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
