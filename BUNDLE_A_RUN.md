# Bundle A — bounded prior comparison → actual-data preview

Authority: [CF4_MASTER_PLAN.md](CF4_MASTER_PLAN.md), approved 2026-09-07.
Plan: `config/cf4_bundle_a_prior_to_data_v1.json`.
Source submitted to Slurm: `a007513` (committed and pushed).
Outputs: `/gpfs/kjhan/CF4/z0_density/bundle_a_prior_to_data_v1`.

## Comparison closure and actual-data decision

All four mock fits and job333765 completed by 2026-09-07 17:57:20 KST.
All fits pass mechanics (divergences0, maximum Rhat1.02227). The quantile
extension improves support density RMSE by only1.1–1.4%, with correlation
gains .0041/.0037/.0098 and paired count predictive changes +.218/-1.158/-3.340.
Both truth fields fail the frozen adequacy rule. Improved curvature recovery
and one-point shape are not enough to adopt it. The prior-repair series ends.

The user instructed proceeding to the actual-data preview. Use the baseline
PM-calibrated lognormal density/velocity prior, with the original24 nuisance
parameters and no added curvature, for a **model-stress diagnostic**, not a
validated local density reconstruction. Plan:
`config/cf4_actual_data_preview_v1.json`. Reuse the sampler with actual-data
inputs and no truth metrics; no new PM simulation or additional mock fits.
This is still Bundle A.

## Actual-data execution

Source `bc645ac` committed and pushed before submission.
Output: `/gpfs/kjhan/CF4/z0_density/actual_data_preview_v1`.

| Job | Scope | State at submission record |
| --- | --- | --- |
| 333862 | Regression and actual-input preflight | COMPLETED in1m40s; all10 tests and input checks pass; peak RSS1272176 KiB |
| 333872 | One actual-data fit, four chains on one GPU | COMPLETED on syn07, 20m54s; ended 2026-09-07 18:43:14 KST |
| 333990 | Result aggregation/failure reporting | COMPLETED in8s; scientific status NO_GO_SAMPLER_NOT_VALIDATED |
| 334240 | Saved-chain/observation-model diagnosis only | COMPLETED in36s, exit0; peak RSS958896 KiB; no new fit |

CF4:19,313 retained observations (15,346 training /3,967 heldout), using
actual BGc velocities. 2M++:36,635 galaxies (29,257 training /7,378 heldout).
No synthetic velocity substitution or truth arrays. GPU job requests7373 MiB
(6144 estimate +20%), four CPUs, one GPU, two-hour limit. Preflight requests
3000 MiB; aggregation1200 MiB. Scientific success has not been established.

A recorded diagnostic warning: population0/3 observed totals are9617/15671,
while the homogeneous nuisance-prior-centre model predicts about2020/3026.
This is not a posterior predictive test or proof of a particular cause.
Inspect normalization/bias excursions and spatial/heldout residuals; do not
silently normalize away the discrepancy or call the map validated.

Final fit: sampling divergences0, mean acceptance .919, max Rhat1.325,
minimum bulk ESS11.1. H0 is the worst-mixing nuisance; density-RMS mixing
alone does not certify the field posterior. Held-out moment-residual SDs
are1.114 (velocity) and1.080 (counts), descriptive only, not truth recovery.
Mean/sample fields and posterior uncertainties exist but are not scientifically
promoted. Follow-up diagnosis found an imported magnitude-convention mismatch
and field-dependent H0 mixing: [ACTUAL_PREVIEW_DIAGNOSIS.md](ACTUAL_PREVIEW_DIAGNOSIS.md).
The user subsequently approved repair and one refit. Corrected preflight334344
passed; GPU fit334345 and afterany aggregation334346 are submitted with source
600c682. See [ACTUAL_CORRECTION_RUN.md](ACTUAL_CORRECTION_RUN.md) for changed
magnitude/selection semantics, independent calibration split and exact radial
coordinates. No new posterior result yet. The submission notes below are historical.

## Historical comparison submission/startup

| Job | Purpose | Dependency |
| --- | --- | --- |
| 333760 | Initial CPU regression check; failed before any input/fit creation | None |
| 333761 | Corrected 10-test regression suite and training-only input preflight | None |
| 333762 | Four GPU fits, maximum two concurrent | afterok:333761 |
| 333765 | Paired scientific comparison, including incomplete/failed-fit reporting | afterany:333762 |

The initial failure was a NumPy array passed directly into a JAX-traced test
call. Corrected in `a007513`; thresholds and science model unchanged. The
second regression suite passed all 10 tests in 73.98 s. Job 333761 completed
successfully in 3m09s, including saved-data identity, training-only prior,
initial gradient and heldout checks. Tasks 333762_0/1 are running and reached
chain0 warmup256 with zero divergences and acceptance about .90; tasks2/3
await the concurrency limit. Job333765 awaits the array. Nonfatal hwloc
binding warnings appear in startup stderr while sampling advances. No final
sampling/science verdict yet.

CPU preflight requests 3600 MiB (3000 estimate +20%); GPU tasks request
7373 MiB (6144 estimate +20%), four CPUs and one GPU each, two-hour limits.
Aggregation requests 2400 MiB (2000 estimate +20%). All use Slurm on syntax,
partitions a40/a100/h100/h200; GPU jobs exclude syn06. No manual node run.
No new PM simulation. Preserve the failed test log.

Frozen comparisons: two quantile-prior fits reuse Z9's native PM truth0
gamma=0/.2 data and saved baseline results; baseline/quantile fits on native
PM truth5 reuse identical Z6 data. Prior training uses only fields1–4.
Report paired physical spatial recovery and held-out predictions; histogram
or curvature improvement alone does not pass the frozen adequacy rule.

After comparison the driver reviews the result before using an actual-data
model. There is no automatic scientific promotion or further prior-repair
loop. Bundle A permits one subsequent actual-data diagnostic fit after its
model/input semantics are recorded; **it has not started**. Bundle B requires
user approval. Automated work currently covers these fits and comparison only.

For the actual-data handoff, existing inputs include 36,635 retained 2M++
galaxies (29,257 training / 7,378 holdout) in
`/gpfs/kjhan/CF4/z0_density/datum_bearing_twompp_v1/datum.npz` and
`data/cf4_clean.npz`. Reuse `cf4_linear_cr.prepare_bgc_catalog` to retain
actual `vobs`; `prepare_fixed_design` deliberately discards it. Preserve raw
exposure and the existing splits. The current BGc target cut is cz=1500–18000
km/s, so this is not a direct LG velocity catalog. Explicit LG observations
must enter Bundle B. Do not report mock truth metrics for real data.

Actual-data input/output preparation is recorded in `ACTUAL_DATA_PREVIEW.md`.
At the subsequent approval check, tasks0/1 had reached chain0 sampling192
with zero divergences; tasks2/3 and comparison were still pending. No actual
fit or next bundle was launched, and running scientific code was unchanged.
