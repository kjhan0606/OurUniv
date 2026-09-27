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
`test_periodic_coherent_wrap_can_leave_original_two_ray_support` passes.

Therefore:

- Do not use a static original-direction ray list as a universally exact
  live-field index.
- Do not impose an undocumented no-wrap restriction on the posterior or
  redefine the radial operator to keep the original direction.
- The existing 8-sigma sparse neighborhood is built from the actual shifted
  source positions of one archived state. It is a frozen-state approximation
  control only, not a live-field support construction.

## Next bounded calculation

Run the existing source-linked continuous FP factor on the v6 graph-closed
training singleton set, comparing each fixed-state 8-sigma neighborhood
against the full 2,097,152-source calculation. Report all per-group
unnormalized five-bin source-density discrepancies and normalized FP-factor
differences. Compare v5/v6 training identities explicitly; do not read
heldout marks. This detects indexing errors on the archived state but does
not prove validity as the field changes.

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

**Q-LEAN:** one full-source reference comparison across the v6 training links
is proportionate; no support-threshold sweep or new simulation is justified.

**MW/M31/M33:** all remain latent roles on the same NEW field. MW/M31 identity
remains ambiguous and M33 can remain unresolved; their observables must
constrain this same field. Native truth identities are for evaluation only,
never candidate seeding or selection.

No email was sent.
