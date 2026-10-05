# Active-v6 angular-selection local operator check — 2026-10-05

## Why this is the next bounded R2 check

The 2026-10-04 full-ray exposure artifact is a predecessor-catalogue geometry
integral, not the active inclusive-v6 likelihood. It uses fixed luminosity
selection and map rays but has no active field, velocity, RSD, TSC source-to-key
kernel or conditional FP normalization. Its NSIDE comparison cannot choose a
resolution for v6; no NSIDE escalation or v6 key replay follows from it.

Driver source audit of the actual v6 path:

- `src/cf4_r2_resolution_target.py` refines the latent source grid to N256 but
  replicates the pinned N128 two-map angular array onto each eight-child group.
- `src/cf4_r2_shell_cdf_count.py` keeps each supplied source angular value
  constant across its source-volume quadrature nodes. The active N256 target
  uses GL2 source-volume integration, then the same coherent spherical RSD,
  order-4/8 LOS count kernel, K/LF transfer and TSC deposition.
- The Poisson count factor applies that intensity once. Its training expected
  count includes empty exposed cells and excludes heldout/buffered geometry.
- The conditional FP numerator and denominator also receive the pinned
  angular source weight through `raw_field_logpdf` and `streaming_raw_mark`.
  This small count comparison does not recompute their normalized ratio and
  does not alter the mark measure.

This is the active target's angular approximation, so a small fixed-state
comparison is a relevant next step. It is not a full field solve or a reason
to multiply the older exposure cube into v6.

## Review and bounded design

Fable's CLI was unavailable due its usage limit. The prescribed Astra backup
review returned **CONDITIONAL PASS** for a local operator diagnostic and
rejected an immediate NSIDE4096/full-grid run or another chain/optimization
extension. The driver independently verified the source connection above.
Adopted constraints: geometry-selected controls only; reuse the saved
conditional diagnostic state; no PM replay, optimization, heldout outcomes,
global resolution ladder or automatic follow-up.

The Slurm job freezes 12 N256 source cells from map-boundary, radial-boundary
and smooth-interior geometry before loading training keys or field values.
For each cell it compares the existing parent-cell angular average against a
reference that looks up the same pinned NSIDE512 RING map at each of that
cell's eight GL2 volume nodes. Density, mean velocity, full-box-normalized
count rates, nuisance coordinates, radial limits, LOS/RSD quadrature and TSC
deposition are identical. It records each patch's contribution to the
training-only expected-rate integral over all exposed voxels (including
empty cells), plus one geometrically nearest occupied training key per
population. Counts themselves are never loaded. This remains a local operator
comparison, not a full Poisson likelihood or whole-field convergence result.

Resource bound: one typed H200, 4 CPU cores, 32 GiB host RAM, 45-minute Slurm
limit. The prior fixed-state N256 diagnostic measured about 15.3 GiB host peak;
32 GiB is above the required 20% headroom. H200 was checked with five of eight
typed devices unallocated at submission planning time. If this control set
does not finish inside the single cap, stop and report; do not increase map
resolution or launch a ladder.

Q-GOAL: this tests an angular-selection approximation inside the same
CF4-conditioned z=0 field target; it produces no posterior or delivered map.
Q-LEAN: one fixed state and 12 cells, no full-grid integral, posterior chain,
gravity simulation, heldout score, or new calibration framework.

MW/M31 remain role-ambiguous and M33 unresolved. This diagnostic does not
identify any of them. Their eventual observables must constrain those same
roles on the same NEW evolved LG field at `<=0.3 cMpc/h`; native truth
identities remain calibration/evaluation-only. R2 remains NO-GO.

## Execution result

Pending Slurm execution. The output directory is
`/gpfs/kjhan/CF4/z0_density/r2_active_v6_pixel_angular_local_20261005_v1/`.
