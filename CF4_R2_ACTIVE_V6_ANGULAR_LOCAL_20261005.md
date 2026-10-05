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
For the corrected replay, map-boundary and smooth-interior controls are each
stratified across four radial intervals (18–55, 55–95, 95–135, 135–168
cMpc/h), in addition to the four radial-boundary controls. Candidate-list
ranks are recorded separately from the C-order flattened N256 grid IDs used
to gather the field.
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

The first replay, Slurm job413637, completed successfully as a process in
1:17 (MaxRSS 2,013,924 KiB), but its **field-dependent measurements are
invalid and withdrawn**. The control selector returned candidate-list ranks;
the GPU runner mistakenly used those small ranks as flattened indices into
the full N256 density, velocity and tracer-rate arrays. For example, source
cell `[126,124,139]` has flattened index8,289,419, not candidate rank282.
Thus the saved source positions and angular geometry are valid, but their
field-weighted expected-count and nearest-key intensity values came from
unrelated cells. Do not interpret those deltas as same-field evidence. The
original artifact at
`/gpfs/kjhan/CF4/z0_density/r2_active_v6_pixel_angular_local_20261005_v1/`
is preserved unchanged for provenance.

The v1 geometry-only map contrasts remain descriptive. Its four
maximum-contrast and four smooth controls clustered near18 cMpc/h because the
selection rule did not stratify radius; this is a control-design limitation,
not a physical-field conclusion. The corrected code makes the distinction
explicit (`candidate_indices` versus `flat_ids`), checks each flattened ID
against its declared N256 cell before any field gather, and adds an
index-coded synthetic-field regression. Local syntax checks and six focused
tests pass.

The corrected replay, Slurm job413638, ran on `syn101` A100 with 4 CPUs and
32GiB requested memory. It completed in1:02, exit0, with MaxRSS1,880,360KiB;
all six focused tests passed in the allocation. The twelve controls now span
the intended radial strata, and the first control's saved N256 flat ID is
8,289,419, consistent with `[126,124,139]`.

For these twelve selected source-cell patches only, the cell-constant versus
per-GL2-node NSIDE512 map-weighted training expected-count subtotal changed
by `[+0.000167,+0.000643,+0.0000339,+0.000550,+0.003523,-0.000271]` for the
six tracer populations. Relative to each selected-patch subtotal, these are
`[+1.40%,+3.13%,+0.40%,+6.62%,+12.0%,-1.66%]`. These are local contributions
from geometry-selected cells—not whole-field changes, a likelihood delta,
posterior movement or a global error bound. The largest single positive
contribution is from `smooth_interior_95_135` at about120.1 cMpc/h, population
4 (`+0.00448` expected counts); the largest map-boundary contribution is
population1 at about60.6 cMpc/h (`+0.000652`). Thus the discrepancy is not
explained solely by within-child pixel-boundary variation: replication of a
3-cMpc/h parent angular average onto its 1.5-cMpc/h children can also matter.
These sparse controls justify a bounded local reference before any
production-operator change, but do not quantify a whole-field effect.

The saved nearest-*geometric* occupied-key samples are not a support-aware
key comparison: several are many cMpc/h from the displaced source and have
zero kernel intensity. Their values are retained in the artifact but are not
used as evidence about local key-level effects. A follow-up key diagnostic,
if needed, must select from actual positive support of the same RSD/TSC/LOS
operator and explicitly report populations with no supported training key.
The expected-count subtotal above uses the full training exposure window,
includes empty exposed cells, and does not load observed counts or heldout
values.

This remains a bounded GL2-node angular-map comparison on the saved incomplete
conditional diagnostic state—not a converged posterior, calibrated
likelihood, delivered density map or R2 completion. No PM replay, optimization
extension, NSIDE escalation or production IC was run. MW/M31 roles remain
ambiguous and M33 unresolved; their observables must ultimately constrain
the same NEW evolved field at LG `<=0.3 cMpc/h`, with native truth identities
used only for calibration/evaluation. R2 remains NO-GO. Result:
`/gpfs/kjhan/CF4/z0_density/r2_active_v6_pixel_angular_local_20261005_v2/result.json`.
