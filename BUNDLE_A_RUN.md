# Bundle A — bounded prior comparison → actual-data preview

Authority: [CF4_MASTER_PLAN.md](CF4_MASTER_PLAN.md), approved 2026-09-07.
Plan: `config/cf4_bundle_a_prior_to_data_v1.json`.
Source submitted to Slurm: `a007513` (committed and pushed).
Outputs: `/gpfs/kjhan/CF4/z0_density/bundle_a_prior_to_data_v1`.

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
