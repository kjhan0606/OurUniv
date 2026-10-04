# R2 map-aware ray/cell cost profile — 2026-10-04

Status: bounded training-geometry benchmark completed by Slurm 412001. This is
not a full selection integral, likelihood fit, posterior, heldout score, or
simulation.

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

## Result — Slurm 412001

Source commit: `f6c66acf5fcea1cab43d25b4ffd7646aef53b680`. H100 (`gpu:H100:1`),
2 CPUs, 4 GiB requested, 15-minute limit; completed/exit 0 in 11 seconds
(script-measured 4.12 seconds). The geometry self-tests passed 4/4. Eighteen
unique training-key voxel locations were sampled across the eight radial
center bins (center radii 13.66–181.21 cMpc/h); the two prior difficult-cell
controls were included. The training key hash matches the frozen input and
the run read neither training counts nor any holdout/all-key array.

For the 53 paired positive population exposures from those cells, the relative
change is p95/max `1.520%/1.732%` from NSIDE 512→1024 and
`0.576%/0.637%` from 1024→2048. There are no one-sided zeros in these 53 pairs.
The two fixed control exposures reproduce the previous values within
`4.9e-8` absolute. This is resolution sensitivity on a small selected sample,
not an angular-accuracy certificate. NSIDE 2048 refines angular integration of
the pinned NSIDE 512 piecewise-constant maps; it adds no observational detail.

The volume-equivalent spherical support contains about `904,759` spatial
voxels. The sample-weighted rough full-support projection is 469 seconds at
NSIDE 512, 1,118 seconds at NSIDE 1024, and 10,279 seconds (2.86 hours) at
NSIDE 2048. Replacing per-bin means by medians gives 8,843 seconds (2.46
hours) at NSIDE 2048, exposing timing noise in the tiny sample. The projected
NSIDE 2048 work is about `7.31e9` candidate pixels and `4.56e9` ray/voxel
intervals, with geometry reused across six radial channels. These are
extrapolations from two or three training-selected cells per bin, not measured
full-grid costs or time guarantees. They use volume-equivalent, not exact
intersecting-cell counts, and can undercount partial boundary voxels.

The process-reported peak RSS was `0.256 GiB`; the maximum per-cell streaming
array estimate among sampled NSIDE 2048 controls was `54.6 MB`. Slurm's
accounting reported only `3,520 KiB` MaxRSS, inconsistent with the in-process
measurement, so it is not used as a memory estimate. Importantly, the training
sample had no voxel centers inside `13.66 cMpc/h`; the nearest-observer empty
voxels and any larger candidate sets they may create remain unprofiled despite
the radial-bin-level projection. Thus the 4-GiB test request is not a
certified full-grid memory requirement.

Decision: the method reproduces the two bounded ray references and the
NSIDE 2048 full calculation looks operationally plausible, but the cost
projection is too rough to launch that full calculation without one explicit
inner-voxel/memory treatment and an exact active-cell census. The next bundle
should resolve those geometric boundary cases, then decide whether to run a
single full-grid NSIDE 2048 exposure precompute. Any resulting cube remains a
numerical selection artifact only; do not wire it into a quantitative field
likelihood until the separate selection/bias, multi-member covariance and
shared-latent count/mark limits are addressed. The contaminated historical
holdout still requires a fresh independent split after the method is frozen.

R2 remains NO-GO. MW/M31 remain role-ambiguous and M33 unresolved; they must
eventually constrain those same roles on the NEW evolved field at LG resolution
<=0.3 cMpc/h, with native truth IDs reserved for calibration/evaluation.
