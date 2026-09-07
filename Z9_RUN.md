# Z9 — approved controlled tracer-response experiment

Implementation commit: `3c370a5` (pushed). Project: OurUniv/CF4;
submission host: `syntax`; compute exclusively through Slurm.

| Job | Work | Dependency |
| --- | --- | --- |
| 333695 | CPU regression tests, corrected Z8 statistics, Z9 paired-data check | none |
| 333696 | Four GPU posterior fits, at most two concurrently | successful 333695 |
| 333697 | Report completed/failed fits and paired scientific comparisons | termination of 333696 |

This is a submission record, not a completion certificate. Read these fixed job
IDs and their named logs when checking status; do not add process-scan loops.
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
