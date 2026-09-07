# Z11 — prior-compatible field control

User-approved next bundle after Z10 exact-block closure. Source `832ccaf` was
committed and pushed before submission from syntax. All compute uses Slurm.

| Job | Scope |
| --- | --- |
| 333744 | CPU input freezing, reproducibility/gradient/holdout checks |
| 333745 | Four free-field GPU fits; after successful 333744, at most two concurrent |
| 333746 | After-any aggregation and comparison with existing PM results |

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
