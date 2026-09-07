# Z9 — approved controlled tracer-response experiment

## Completed result and next approved diagnostic

All four fits completed (20m29s to 20m40s each); aggregation job 333697 finished
2026-09-07 12:59:35 KST. All sampler gates passed, divergence fraction zero,
maximum Rhat 1.01944. **The extended model is not approved for promotion.**
In the null case gamma=0, its posterior median was 0.28957 (95% interval
0.15992–0.41560). In the gamma=0.2 case, median was 0.45186 (0.31953–0.58354).
Observed-support correlation improved only 0.61218 to 0.61388 in the perturbed
case, and RMSE worsened 0.53864 to 0.54070. Heldout count log-predictive differences
were -0.366 (null) and +7.850 (perturbed); velocity differences -1.116 and -2.436.
These sums alone do not certify predictive significance or physical recovery.

The user approved the next calculation: Z10 fixes the exact native mock density
and velocity while retaining all 25 observation nuisance parameters and the same
saved observations. Its plan is `config/cf4_z10_fixed_truth_plan_v1.json`.
This isolates whether the curvature displacement persists conditional on true
fields. It does not by itself distinguish a wrong field prior from other
field–observation coupling or establish calibrated real-data reconstruction.

Z10 implementation `3227f5b` was committed and pushed before submission.
Slurm array `333719` (tasks 0/1) runs on allocated GPUs, with four CPUs and
4916 MiB requested host RAM per task (4096 MiB estimate plus 20%), one-hour
limits. After-any aggregation job `333721` reports results or missing/failed
cases; it does not launch another bundle. Outputs are in
`/gpfs/kjhan/CF4/z0_density/z10_fixed_truth_v1/`. Inspect these exact job IDs,
`task_0/progress.json`, `task_1/progress.json`, and `comparison.json` for status.

Z10 initially completed at 14:25:01 KST but failed the full sampler gate through
radial nuisance 20. Its other 21 nonradial coordinates passed. User-approved
exact-block repair then completed as CPU Slurm job `333743` at 15:43:53 KST,
taking 1m10s. It retained every nonradial sample and drew the independent four-
dimensional Gaussian radial posterior exactly. Both repaired joint and retained
marginal pass the original gates: worst Rhat 1.02396, minimum bulk ESS 134.18.
Curvature intervals remain [-0.03326, 0.05823] for truth 0 and [0.14418, 0.22921]
for truth 0.2. This validates the conditional diagnostic, not real-data recovery
or the free-field model. Full closure: `config/cf4_z10_exact_radial_result_v1.json`.
No next bundle has started; user approval is required before that transition.

## Original execution record

Implementation commit: `3c370a5` (pushed). Project: OurUniv/CF4;
submission host: `syntax`; compute exclusively through Slurm.

| Job | Work | Dependency |
| --- | --- | --- |
| 333695 | CPU regression tests, corrected Z8 statistics, Z9 paired-data check | none |
| 333696 | Four GPU posterior fits, at most two concurrently | successful 333695 |
| 333697 | Report completed/failed fits and paired scientific comparisons | termination of 333696 |

This is a submission record, not a completion certificate. Read these fixed job
IDs and their named logs when checking status; do not add process-scan loops.

Verified startup: job 333695 completed in 2m39s, peak RSS 964644 KiB.
All eight regression tests and the full-size paired-data/nested-response checks
passed. Tasks 333696_0 and 333696_1 then started on syn05 under Slurm; both
reached chain 0 warmup step 128 with zero divergences and acceptance about 0.90.
Tasks 2/3 await the two-task concurrency limit. The GPU logs contain hwloc CPU
binding warnings; sampling nevertheless advanced. This is not a completion or
final convergence assessment.

Outputs: `/gpfs/kjhan/CF4/z0_density/z9_tracer_response_v1/`.
Logs: `/gpfs/kjhan/CF4/logs/cf4_z9_{tests,tracer,finish}_*.{out,err}`.
The detailed frozen plan is `config/cf4_z9_tracer_response_plan_v1.json`.
GPU RAM request is 7373 MiB per task (6144 MiB estimate plus 20%); each task
has a two-hour limit. New output budget is approximately 5 GiB.

## Scientific correction and decision boundary

Z8's reported 0.76 ratio was pointwise posterior uncertainty divided by truth
spatial SD, **not reconstructed density amplitude**. Its `logdensity` probe was
scalar log posterior, and `white_RMS` included all four latent blocks. Neither
probe established physical bias–density identifiability. The old smoothing /
missing nonlinear physics inference is withdrawn; original artifacts remain
preserved. Corrected statistics are written to the distinct
`z8_count_tracer_identifiability_corrected_v2` directory by job 333695.

Corrected count-only whole-box examples (truths 0 and 5): truth spatial SD
0.675/0.681; posterior-mean spatial SD 0.209/0.200; quadratic-mean individual-draw
spatial RMS 0.564/0.561; mean pointwise uncertainty 0.513/0.514. Individual draws
also have less amplitude than these truths, but these numbers alone do not
identify its cause, certify coverage, or establish missing tracer nonlinearity.

Z9 compares the existing nonlinear power-law tracer link to one additional
common log-density-curvature coefficient. Both the matched-null generator and
a specified perturbed generator are fitted with both models, using identical
data within each pair and independent heldout Poisson noise. The existing
PM-calibrated z=0 prior is unchanged. This is one reused development truth at
12 cMpc/h, not independent galaxy physics or an actual CF4 reconstruction.

Assess heldout prediction, physical density recovery and interval coverage
together. Distinguish posterior-mean spatial amplitude, individual-draw spatial
amplitude and pointwise uncertainty. Sampler gates alone cannot promote a model.
Automatic aggregation does not make the final scientific judgment. The driver
reviews the completed comparison; the next bundle requires user approval.
