# R2 map-aware ray/cell cost profile — 2026-10-04

Status: bounded training-geometry benchmark in progress. This is not a full
selection integral, likelihood fit, posterior, heldout score, or simulation.

## Goal and fixed scope

The 2026-10-03 cellwise audit found that global effective-volume agreement
conceals large local exposure errors. Its next action is to measure whether a
ray/cell integral over the pinned NSIDE=512 maps is practical before computing
all selection exposures. This bundle uses only unique voxel locations derived
from the frozen `train_keys` array. It does not open training counts,
`all_keys`, `holdout_keys`, `holdout_counts`, or any field state.

For each bounded sampled training voxel, the profile traces equal-area
HEALPix pixel-center directions through the voxel, computes exact Cartesian
slab intervals, applies the corresponding pinned native-map pixel values as a
piecewise-constant angular field, and integrates each of the six existing
radial population factors over the interval. It does not smooth the maps or
substitute the voxel-centre pixel for angular integration. The run compares
NSIDE 512/1024/2048 where a conservative candidate-pixel cap allows bounded
memory and reports skipped controls explicitly.

The extrapolation is deliberately a rough resource projection: sparse
training-voxel samples are weighted by spherical-shell volume-equivalent
voxel counts. It is not a runtime guarantee or an accuracy bound. In
particular, it cannot authorize the full-grid integration by itself. No
posterior, fit, selection/bias calibration, shared-group covariance,
shared-latent count/mark law, or independent holdout is created here. The
previous archive's holdout arm is known to be contaminated; a fresh independent
holdout is required after the method is frozen.

## Q-GOAL / Q-LEAN and LG obligations

- Q-GOAL: accurate selection exposure is necessary to connect the 2M++ data
  to the same CF4-conditioned z=0 field; this cost measurement produces no
  field result.
- Q-LEAN: a small deterministic training-geometry sample measures geometry
  and six-channel work. No repeated global integration, training-count score,
  heldout read, fit, PM evolution, or simulation is included.
- MW/M31 remain role-ambiguous and M33 unresolved. Their observables must
  eventually constrain these roles in the same NEW evolved field at LG
  resolution <=0.3 cMpc/h; native truth identities are calibration/evaluation
  only. This R2 numerical task does not identify or score LG components.

Because the preceding important-result review was Astra-led, this consecutive
technical bundle is driver-reviewed; no second external reviewer is requested.

## Execution record

Source: `scripts/cf4_r2_ray_cost_profile.py` and
`scripts/run_cf4_r2_ray_cost_profile.sbatch`. The Slurm job uses a typed H100
request, two CPUs, 4 GiB host memory, and a 15-minute limit. The 4-GiB request
is above the estimated sub-GiB streaming peak by more than 20%; the fixed
candidate cap prevents a single inner voxel from materializing an unbounded
high-NSIDE ray set. Actual Slurm memory/time and any omitted radial bins will
be reported from the result, not inferred from submission.

Result path, selected source commit, and measured/extrapolated costs will be
added after the bounded job completes. R2 remains NO-GO until both numerical
selection integration and the separate observation/calibration barriers are
resolved.
