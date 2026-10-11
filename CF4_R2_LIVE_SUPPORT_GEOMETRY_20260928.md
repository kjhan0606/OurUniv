# R2 linked-FP live-support geometry — 2026-09-28

## Finding

Astra's advisory review found that the proposed static angular-ray support
cannot be certified for the current radial-key operator. In
`observer_centred_spherical_rsd_jax`, coherent RSD is applied and the
minimum-image direction is recomputed from the shifted periodic position.
When the source crosses a box face, this direction can rotate; it is not
generally the original direction or its negative.

The regression case uses a 384 cMpc/h periodic box, observer at its center,
source relative coordinate `(190,30,0)`, and 2,000 km/s radial velocity. With
`H0=74.6 km/s/Mpc` and `h=0.746`, this is a 20 cMpc/h coherent shift. The
unwrapped shifted coordinate is approximately `(209.75,33.12,0)`; after
minimum-image wrapping it is `(-174.25,33.12,0)`. The changed ray contributes
at observed radius 177.37 cMpc/h, still within the 180 cMpc/h selection limit,
while both targets built from the original signed ray have exactly zero TSC
weight in that voxel. The focused regression
`test_periodic_coherent_wrap_can_leave_original_two_ray_support` compares a
two-source wrapped/unwrapped case: a local neighborhood rebuilt from the
actual shifted positions matches full-source five-bin densities and the
normalized mark factor; the original-ray list drops the wrapped source and
changes the factor. It passes.

Therefore:

- Do not use a static original-direction ray list as a universally exact
  live-field index.
- Do not impose an undocumented no-wrap restriction on the posterior or
  redefine the radial operator to keep the original direction.
- The existing 8-sigma sparse neighborhood is built from the actual shifted
  source positions of one archived state. It is a frozen-state approximation
  control only, not a live-field support construction.

## Fixed-state v6 comparison

Typed-H100 Slurm **407082** completed all 1,414 v6 training groups in 312 s
(4 GiB requested; host peak 1.33 GiB). Per-group state-frozen 8-sigma
neighborhood factors agree with the full 2,097,152-source reference within
`1.78e-15` absolute normalized log-factor error and `1.74e-13` maximum
relative five-bin density-sum error. Candidate counts are p50 612, p90 696,
p99 766, maximum 884. The v6 training identity set removes exactly three
v5 groups (`T60475`, `T66514`, `T81318`) and adds none, as expected from the
graph buffer. Heldout marks were not read.

The focused wrap regression passes after the synthetic two-source comparison.
`py_compile` also passes. The full marked-tracer unittest module was started
on the login node but stopped by the driver after three minutes while an
existing incomplete-gamma shape-gradient test was still computing; this was
not an assertion failure, and that broader suite is not claimed as passed.

This establishes fixed-state index mechanics only. The dynamic shifted-point
wrap regression validates a two-source perturbation, not an all-group live
field transition or sampler. The 8-sigma candidate truncation remains an
explicit approximation despite its tiny discrepancy on this saved state. No
field fit, sampler, heldout score, N256 calculation, or production posterior
was run.

The run must not perform a field fit, sampler, heldout prediction, N256
calculation, new gravity simulation, external archive download, or output
cleanup. Preserve failures and existing raw results.

## R2 status and limits

R2 remains **NO-GO** for a production present-state posterior. Even a passed
full-source mechanics comparison does not calibrate CF4/2M++ selection,
source incidence, group association, or shared redshift/FP covariance, and it
does not show sampler stationarity or heldout predictive performance.

**Q-GOAL:** validating the linked source-to-key mark factor is necessary for
the CF4-conditioned z=0 target and keeps the same evolved field observable to
future low-k/high-k and LG constraints. It is not itself the delivered z=0
posterior.

**Q-LEAN:** the one full-source comparison across v6 training links is complete;
no support-threshold sweep or new simulation is justified. The next necessary
work is not another mechanics sweep: it is a defensible selection/association
and shared group-redshift/FP covariance model, followed by a live-field
stationary posterior and untouched heldout prediction. Do not advance to R3
or label R2 complete without that science result.

**MW/M31/M33:** all remain latent roles on the same NEW field. MW/M31 identity
remains ambiguous and M33 can remain unresolved; their observables must
constrain this same field. Native truth identities are for evaluation only,
never candidate seeding or selection.

No email was sent.
