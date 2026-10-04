# R2 exact shell-geometry census — 2026-10-04

Status: planned, awaiting the bounded Slurm run. This resolves the specific
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

Pending Slurm execution. The result will determine whether the previous
2.46–2.86 hour NSIDE 2048 rough full-grid projection needs a material inner-
shell correction and whether the existing per-cell candidate cap can process
the worst rays without a streaming change. It will not certify angular
accuracy or justify wiring the resulting exposure into a field likelihood.
R2 remains NO-GO pending cellwise integration accuracy, selection/bias
calibration, multi-member covariance, and a shared-latent count/mark law. A
fresh independent holdout must be frozen after the integration method is
fixed because the historical holdout is contaminated.
