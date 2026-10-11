# R2 exact shell-geometry census — 2026-10-04

Status: completed by Slurm job 412034. Attempt 412025 ran four pytest checks,
then failed in the census script's own self-test because an outer-shell
tangency assertion expected `r_min < 180` at `r_min=180`; that test-only
condition and its regression assertion were corrected. Dispatch attempts
412032 and 412033 then failed before tests because of, respectively, a mistyped
expected commit and an unexported environment variable. They did no census
work. Job 412034 used the verified commit and completed the census. This
resolves the specific
gap in `CF4_R2_RAY_COST_PROFILE_20261004.md`: its training-key sample had no
voxel center below 13.66 cMpc/h, leaving observer-near and partial-shell cells
unprofiled. It does not run the full selection integral.

## Frozen scope and method

The geometry-only job enumerates the complete 128^3 grid analytically, with
the observer at the origin and cell spacing 3 cMpc/h. A cell is active exactly
when its box has positive-volume intersection with the radial shell
`5 < r < 180 cMpc/h`, using exact axis-aligned-box minimum and maximum radii.
The report partitions active cells into fully interior, inner-boundary,
outer-boundary, and (if present) both-boundary cases. It also totals
center-radius bins and conservative angular-cap pixel estimates for every
active cell at NSIDE 2048. Cap-area totals are estimates, not exact HEALPix
query counts.

For every cell attaining the maximum angular cap, the job separately runs the
actual inclusive RING `healpy.query_disc` and counts, in fixed-size chunks,
candidate pixel centers that hit the box and the 5–180 shell. This is the
bounded exact ray census for the worst observer-near geometry; it does not
trace every active cell. Four focused geometry tests run in the same Slurm
allocation. The script reads no catalogue, angular map, field, count, training
or holdout array.

## Q-GOAL / Q-LEAN and LG obligations

- Q-GOAL: exact shell support is needed to determine whether the map-aware
  exposure integral can cover the same CF4-conditioned z=0 field. This census
  creates no field, likelihood, posterior or LG constraint.
- Q-LEAN: one full-grid scalar geometry pass and exact rays only for the
  maximum-cap cells; no full exposure integration, fit, PM evolution or
  simulation.
- MW/M31 remain role-ambiguous and M33 unresolved. Their observables must
  ultimately constrain those same roles in the NEW evolved field at LG
  resolution <=0.3 cMpc/h; native truth identities remain
  calibration/evaluation-only. This R2 geometry work cannot identify them.

This is driver-reviewed as the consecutive technical follow-up after Astra's
shared-redshift factor audit; no duplicate external review is requested.

## Execution contract

Slurm runner: `scripts/run_cf4_r2_shell_geometry_census.sbatch`. The H200/H100/
A100 typed modes were checked; H200 is reserved and A100 is mixed/draining, so
the bounded job requests `gpu:H100:1`. It requests 2 CPUs, 4 GiB host memory
(well above 20% of the sub-GiB chunked geometry estimate), and 30 minutes.
Output is isolated at
`/gpfs/kjhan/CF4/z0_density/r2_ray_geometry_census_20261004_v1/result.json`.

## Result and disposition

The exact census found 938,128 voxels with positive-volume intersection with
`5 < r < 180 cMpc/h`: 870,528 fully interior, 56 inner-boundary partial,
67,544 outer-boundary partial, and none partial at both boundaries. The
partition closes exactly. The sum of conservative cap-area estimates is
6.901e9 NSIDE-2048 pixels; this is not an exact total query-disc count or a
runtime estimate. It is below the earlier 7.31e9 sample projection, but does
not validate that projection.

The eight voxels attaining maximum cap (`0.955318 rad`) each need about
10.644 million inclusive `query_disc` candidate pixels, of which about
6.291 million center rays intersect the box and only 16.7 thousand intersect
the narrow `5–5.196 cMpc/h` shell segment. The census separately reports 56
inner-boundary partial cells and 56 active cells whose conservative cap-area
estimate exceeds the ray profiler's 1.5-million candidate limit; their
cell-by-cell cross-tab was not saved. At least the eight maximum-cap cells
exceed that limit, so reusing `ray_geometry` unchanged for a full integral
would skip valid cells. The actual selection integrator must process these
candidates in bounded chunks (or provide a separately measured memory-safe
path); do not raise the cap blindly. The census itself took 18.51 s and
process peak RSS was 0.205 GiB under a 4-GiB request; Slurm MaxRSS3.5MiB is
inconsistent and not trusted.

Disposition: this closes the geometry census, not selection integration.
Before the projected 2.46–2.86 hour full NSIDE-2048 precompute, plan a
chunked, no-skip implementation and validate it on the eight maximum-cap and
all 56 inner-boundary cells. Compare against unchunked references where they
fit and preserve the pinned NSIDE-512 map values. This census does not certify
angular accuracy, calibrate selection/bias, add multi-member covariance, or
complete a shared-latent count/mark likelihood. R2 remains NO-GO; no posterior
or holdout score was made. A fresh independent holdout must be frozen only
after the integration method is fixed because the historical holdout is
contaminated.

## Fable5 plan audit and driver ruling

Fable5 (`claude --model opus`, read-only) returned **CONDITIONAL PASS** for
one fail-closed Slurm bundle that combines chunk checks, full NSIDE-2048
integration, and exact closure. Adopted: no candidate skip; chunked rays;
explicit population-by-radial-shell channels; whole-domain geometry and
selection closure; small chunked-versus-unchunked comparisons; and no separate
validation job. Fable also recommended an NSIDE-1024 whole-grid convergence
pass as a diagnostic, not a gate. The driver adopts that pass because the
previous small-cell sample omitted the observer-near region; no NSIDE-4096 or
adaptive follow-up is authorized absent a material 1024-to-2048 discrepancy.

Driver amendments: this N128/3-cMpc/h product is a count-grid selection
operator check, not the N256/1.5-cMpc/h global density field or the
<=0.3-cMpc/h LG map. No posterior or LG claim follows. The census recorded
equal totals (56 inner-boundary partial cells and 56 above-cap cells) but not
their exact cross-tab; the integrator will compute and report that cross-tab,
not assume Fable's inferred equality. Geometrically active cells with zero
2048 pixel-center hits will be reported as a resolution diagnostic, not
automatically treated as a science failure; any occupied training cell with
zero exposure remains a later support failure. The exact closure identity is
the decisive no-skip/double-count check. Full advice and the bundled plan are
recorded in `CF4_R2_EXPOSURE_PRECOMPUTE_PLAN_20261004.md`.

MW/M31 remain role-ambiguous and M33 unresolved. This numerical R2 bundle
identifies no LG members; their observables must constrain those same roles in
the same NEW evolved field at LG <=0.3 cMpc/h. Native truth identities remain
calibration/evaluation-only, never candidate selection.
