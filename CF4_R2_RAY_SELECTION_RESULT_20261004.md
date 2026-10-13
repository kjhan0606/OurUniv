# R2 map-aware ray exposure — historical predecessor-catalogue diagnostic, 2026-10-04

Status: **not applicable to the active inclusive v6 target; no current-target
resolution gate is passed by this experiment. R2 remains NO-GO.**

## Applicability correction (driver disposition after Fable5 review)

The original training-impact interpretation was too broad. The training script
read `r2_common_catalogue_128_v1/counts_3_sparse.npz`: an older disjoint
catalogue with24,993 training counts and8,375 held-out counts. The active
target is the inclusive `r2_sky_closed_split_v6` split with47,121 training
counts and uses the shell-CDF/TSC source-to-key count kernel. The map-aware ray
exposure operator evaluated here is not connected to that active count target.
Consequently, every NSIDE1024-to-2048 statistic below is a valid diagnostic
only for the old disjoint support; it is not evidence for v6, likelihood
readiness, or an active-target NSIDE choice. No v6 ray rerun or NSIDE4096
escalation should follow from this artifact. The frozen files and job logs are
preserved, and the earlier promotion language below is superseded by this
correction.
The full-grid result is
`/gpfs/kjhan/CF4/z0_density/r2_ray_selection_n128_nside1024_2048_20261004_v1/result.json`.
The training-only result and frozen NSIDE1024 checkpoint are
`/gpfs/kjhan/CF4/z0_density/r2_ray_selection_n128_nside1024_trainingimpact_20261004_v1/`.
The HDF5 status remains `COMPLETE_NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY`.

## Execution and implementation checks

Typed-H100 Slurm job412060 ran on `syn08` for1:34:51, exit0, requesting16
CPUs and8GiB. Slurm MaxRSS was3,961,908KiB (3.78GiB); the result JSON's
24.78GiB worker-sum bound is a deliberately loose theoretical sum of per-child
peak RSS, not the observed concurrent job peak. The previous unchunked census
job was412059; its eight tests passed but preflight stopped before output
creation because it assumed the geometry-only legacy routine returned an
exposure. Commit `c46c693` reconstructed that independent reference and added
a regression; the corrected job passed all nine tests.

The complete active-cell count is938,128. The exact2×2×2 table establishes
that all56 cells above the former1.5-million cap are exactly the56 inner-shell
partial cells;67,544 are outer-shell partial and870,528 fully inside both
radial boundaries. At NSIDE2048, the actual inclusive-disc candidate total is
7,088,692,124, about2.7% above the6,901,444,420 cap-area estimate. NSIDE1024
used1,819,724,558 candidates. No active cell was dropped.

Nine Slurm regressions passed; all eight maximum-cap cells agreed across
65,536-,250,000-, and one-piece chunks to at most7.7e-15 relative difference.
Six existing fixed-reference checks passed. Both full-sky closure identities
are excellent: population-shell relative error2.01e-15 (NSIDE1024) and
5.26e-15 (2048), with pure-geometry error below8e-16.

## Scientific interpretation

Global closure proves the sum of the implemented voxel integrals recovers the
full-sky reference. It does **not** prove cellwise angular convergence. The
maximum binned p95 NSIDE1024-to-2048 relative change is33.2% (p99 62.3%, max
89.1%) for population5, radial shell1, center radii15–30 cMpc/h; several
outer bins have binned p95 changes12–24%, with p99/max reaching100% where one
quadrature has zero exposure. NSIDE1024 records160 active zero-ray cells;
NSIDE2048 records8. Therefore NSIDE2048 is a no-skip production artifact but
is not certified by all-grid maxima alone as an absolutely converged cellwise
operator; the training-only proxy decision is below.

Against the old order-six map, the all-grid summary counts25,773
population-shell-cell entries that were zero before and positive now, and
zero entries that were positive before and zero now. This is not training
support or a count-weighted score: the construction intentionally did not
read keys or counts. The subsequent training-only audit below determines
whether the discrepancy reaches observed training support.

The output is only for the N128/384-cMpc/h (3-cMpc/h) count-grid selection
operator. It adds no observational resolution, does not calibrate survival,
galaxy bias, group inclusion or covariance, and creates no count likelihood,
posterior, z=0 density/velocity field, or IC. R2 stays NO-GO.

## Historical training-only resolution result (old disjoint catalogue only)

Typed-H100 Slurm412142 completed all128 NSIDE1024 x-slabs and wrote a closed,
hashed, data-free operator (1,819,724,558 candidate rays; population closure
2.01e-15; pure-geometry closure3.76e-16). Its12 regression tests passed.
The job then failed in result assembly on a driver KeyError, after the
data-free checkpoint had been frozen and before a training result was written.
The artifact was preserved and verified in Slurm412202 before the sparse
training phase resumed; the expensive integral was not repeated.

The resumed audit read22,457 frozen training keys and24,993 counts only. There
were no occupied zero-exposure keys at the old order-six, NSIDE1024, or
NSIDE2048 resolution, and no old-zero/new-positive transitions in training.
All six population-specific predeclared NSIDE1024-to-2048 proxy gates passed.
The largest absolute count-weighted log-exposure sum is0.106 nats; population
p95 count-weighted `|log(E2048/E1024)|` is at most0.00102, no training cell
exceeds the `0.5` shot-noise-scaled threshold, and no training count is above
the `0.1` threshold.

The data-free NSIDE2048 pure-geometry check still has a sparse all-grid tail:
among870,528 fully interior cells, absolute deviation from unit exposure has
p95/p99/max0.0206%/0.0717%/1.204%, with10 cells above1%. At unique training
cells, NSIDE2048 p95/p99/max is0.0127%/0.0455%/0.613%; no training cell
exceeds1%, and63 galaxies occupy cells above0.1%. NSIDE1024 has four training
galaxies in cells above1%. The high-all-grid tail therefore remains recorded,
but is not occupied by the frozen training sample at the >1% level.

Pre-audit decision (now withdrawn for the active target): retain NSIDE2048 as
the numerical exposure candidate for the old frozen N128 disjoint training
support. This has no v6 implication. Even on that old support, do not call the
operator absolutely converged or likelihood-ready: 1024-to-2048 is only a
pairwise proxy, and this calculation does not measure how exposure errors
weight the nonuniform expected-count term of an inferred density field. No
such density field was supplied here.
Selection/survival/bias calibration and the shared multi-member count/mark
law remain the active R2 blockers.

## Scope and next R2 work

The training impact and zero-support categorization are complete. Its
thresholds were frozen in source before count access: per population,
`abs(delta log L) < 1 nat`, no cell with
`abs(log(E2048/E1024))*sqrt(Ncell) > 0.5`, and less than1% of training counts
above0.1 by that metric. These passing results concern training support only,
not the full nonuniform-field expected-rate term. Return next to the real R2
observation-law blockers: overlap-aware CF4/2M++ datum ownership,
selected-group inclusion/shared covariance, and calibrated survival/bias/
RSD-FoG behavior. Do not score heldout values or promote a field/posterior
based on the exposure check.

Q-GOAL: a selection-denominator diagnostic for the same CF4-conditioned R2
field, not a field/posterior result; the operator starts at5cMpc/h and does
not identify the LG. Q-LEAN: one data-free all-active-cell lower-resolution
artifact plus one sparse training-only pass, with no heldout exposure. R2
remains NO-GO. MW/M31 roles remain ambiguous and M33 unresolved; their
observables must constrain those same roles on the same NEW evolved LG field
at `<=0.3 cMpc/h`, with native truth IDs reserved for calibration/evaluation.

MW and M31 identities remain role-ambiguous and M33 remains unresolved. Their
observables must eventually constrain those same latent roles on the same NEW
evolved Local Group field at `<=0.3 cMpc/h`; native truth identities are
calibration/evaluation-only and may not seed or select candidates.
