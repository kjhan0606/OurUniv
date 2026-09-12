# Z11 — prior-compatible field control

## Closure — 2026-09-07

All four fits and aggregation completed (final job 333746 at 16:36:26 KST).
All four pass the declared mechanics gates; all injected curvature values
fall inside their 95% intervals. Median gamma values for tasks 0–3 are
0.01463, 0.16840, -0.03181 and 0.17664 (truth 0, .2, 0, .2).
Observed-support density correlations are approximately .554–.590.
This supports a field/observation coupling mismatch on native PM fields,
not its unique cause or actual-data calibration. No LG/IC milestone passed.
Follow `CF4_MASTER_PLAN.md`, Bundle A. The startup record below is historical.

## Historical submission/startup record

User-approved next bundle after Z10 exact-block closure. Source `832ccaf` was
committed and pushed before submission from syntax. All compute uses Slurm.

| Job | Scope |
| --- | --- |
| 333744 | CPU input freezing, reproducibility/gradient/holdout checks |
| 333745 | Four free-field GPU fits; after successful 333744, at most two concurrent |
| 333746 | After-any aggregation and comparison with existing PM results |

Verified startup: 333744 completed in 2m08s (batch peak RSS 1082068 KiB), with
all input/gradient checks passing. Tasks 333745_0/1 reached chain 0 warmup step
128 with zero divergences and acceptance about 0.90; tasks 2/3 await the two-
task concurrency limit. Logs include hwloc binding warnings but sampling is
advancing. Final convergence and science outcomes are not yet available.

Outputs: `/gpfs/kjhan/CF4/z0_density/z11_prior_control_v1/`.
Plan: `config/cf4_z11_prior_control_plan_v1.json`.
Read these fixed job IDs and their specific logs/progress files, not process
scans. Submission does not mean completion. GPU tasks request 7373 MiB host RAM
(6144 MiB estimate plus 20%), four CPUs, one GPU, and two-hour limits. Output
budget is 5 GiB; no new PM simulation or resolution increase.

Two frozen field seeds (2026091100/2026091101) are passed through the exact
existing z=0 prior; each is tested with gamma=0 and gamma=0.2. All 25 nuisance
parameters and the fields remain free during inference. Truth vectors never
initialize the chains. Reuse Z9 tasks 1/3 as the PM reference and Z10 as the
fixed-field conditional diagnostic. All field, nuisance, and noise priors and
sampler settings are unchanged. Z10's independent exact radial block is **not**
applied to these free-field fits.

Different mock datasets do not admit raw log-predictive score subtraction.
Compare curvature displacement, density recovery/coverage, and descriptive
physical-field statistics. Prior-compatible recovery would support a
field-model/coupling discrepancy on PM truth, not uniquely prove it. Failure
also on these idealized mocks would motivate inference/identifiability work.
Two fields and fixed nuisance truths do not constitute simulation-based
calibration or validate real galaxies. This still uses 12 cMpc/h development
fields, not the final LG-only 0.3 cMpc/h target.

Stop after the four fits and comparison. The driver must interpret results;
automatic aggregation cannot authorize production or another bundle. Await
user approval before the next bundle.
