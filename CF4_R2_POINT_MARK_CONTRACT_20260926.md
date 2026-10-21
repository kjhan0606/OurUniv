# R2 individual-point / CF4-group observation contract — 2026-09-26

## Why this is the next core step

The intended first science product remains a CF4-conditioned posterior for
the *present* density and velocity field, evolved from a latent LCDM IC.
Neither the old CF4 BGc × disjoint 2M++ count product nor the inclusive
binned count diagnostic is a valid joint likelihood. Matched 2M++ galaxy
redshifts and CF4 group redshift/distance marks share objects; a six-population
cell count cannot retain the particular galaxy redshifts needed for a
conditional group mark.

This bundle binds the actual individual-point data to the frozen crossmatch
and native CF4 split. It does not create a new simulation or posterior.
`scripts/cf4_r2_point_mark_manifest.py` preserves every eligible 2M++
galaxy's source redshift, angle, magnitude, six-population label, N128 cell,
official-map and catalogue completeness values, and all non-unmatched CF4
association edges. It also retains CF4 group Vcmb, DMzp, e_DMzp, Ngal and
the native holdout label *without* treating local 2M++ GID or archived Tully
Nest as a CF4 `1PGC` identity. No uncertain match is forced secure.

`src/cf4_r2_point_process.py` supplies only the exact numerical identity for
a six-population piecewise-cell-modulated redshift-space Poisson process,
with pointwise survey selection `s_p(x)` and its cell-average integral `E_pc`:

`log p(Y|u,S) = -Σ_pc u_pc E_pc + Σ_i log[u_(p_i,c_i) s_(p_i)(x_i) / V_cell]`.

Here `u_pc` is an *unselected* expected cell count supplied by an evolved
field/galaxy model; the kernel does not estimate it from observed totals.
The formula fails closed if any observed point has zero map support or its
cell has zero integrated exposure. Other point/integral inconsistencies need
source and quadrature checks; they are not automatically detected by this
kernel. It is not yet a scientifically adopted likelihood. For a constant
`u_pc` within each cell,
conditioning on cell counts makes the within-cell point locations independent
of the field. Thus preserving individual points is necessary for shared
group-redshift ownership, but does **not** magically recover 0.3 cMpc/h
field information from a 3 cMpc/h N128 model. A truly sub-cell field or a
finer LG-specific observation model is needed for that goal.

## Scientific limits and next decision

The literal official-map model predicts zero probability for any observed
galaxy at zero pointwise selection. Source row/map disagreements and earlier
exception marks cannot be fixed by dropping rows while retaining the old
selection integral: that would be unmodelled, spatially varying thinning.
The output flag `quality` reproduces that *historical* rejection rule; it is
not a newly established physical truth label.
The present integral is order-six quadrature and the luminosity function is a
development model. Population bins use redshift-derived absolute magnitudes;
this approximation must eventually be reconciled with peculiar velocities
and redshift errors. CF4 group inclusion/measurement selection and the
joint covariance `p(V_group,DM_group | matched individual Y,F)` also remain
uncalibrated. The previously fitted two-stratum Student-t redshift law failed
heldout coverage, and the numerical Gaussian conditional cannot replace it.
No inclusive actual-data likelihood, N256 sampling, production density map or
IC is promoted by this bundle.

The next substantive action is to establish how the catalogue completeness
columns and ARES mask relate, resolve the one literal zero-support point with
a source-justified observation/discrepancy model, **and** formulate the CF4
group distance-selection/conditional-mark law, then evaluate both
under the same evolved state on actual heldout data. Do not cure a zero
selection probability with an arbitrary epsilon or uncalibrated diffuse term.

MW/M31/M33 identification remains a later readout of each **new** evolved
field: roles and the unresolved-M33 possibility must remain ambiguous and
their observables must constrain that same field. Native truth IDs are for
calibration/evaluation only, never for generated-field selection. This
manifest does not identify or insert any LG member.

Q-GOAL: preserving the complete actual observation and overlap structure is
required before an R2 present-field posterior can be credible. Q-LEAN: one
small source-bound data product and one exact observation kernel; no new
cosmological run, extra mock ladder, production claim or external audit.

## Execution result

The first typed-H200 Slurm submission **405612 FAILED/exit1** after the
2/2 numerical-kernel tests passed: the manifest builder incorrectly assumed
that every crossmatch `1PGC` has a row in the published CF4 group table.
The correction preserves such edges with an explicit missing-canonical-group
flag and undefined group marks; it does not drop or invent them. The bounded
retry **405614 COMPLETED/exit0** in six seconds, with both tests passing.

The resulting [source-bound report](/gpfs/kjhan/CF4/z0_density/r2_point_mark_manifest_v1/result.json)
and `points.npz`/`edges.npz` contain 57,238 eligible 2M++ points, 15,211
distinct crossmatched points, 15,239 matching edges, 10,595 distinct group
IDs, and 14,878 secure edges. Four secure edges lack a row in the canonical
CF4 group table and therefore have no group Vcmb or distance mark in this
contract. In the preserved edge array, 28 2M++ points have two CF4 edges;
seven of those point to *different* CF4 group IDs. A one-to-one assignment
cannot be assumed. The inclusive N128 count projection agrees exactly with the
preserved diagnostic count view. Exactly 442 objects have catalogue/map
completeness mismatches, including the already-known 102 matched and 340
nonmatched split; 316 are also in the prior exception list. A single
nonmatched, heldout point (2M++ recno 67100, population 4) has catalogue
completeness 0.5 but official-map completeness 0, despite positive
cell-integrated exposure 0.09919. Its HEALPix pixel has both zero and 0.5
nearest neighbours. This is evidence for a local source/map discrepancy,
**not** authorization to substitute a neighbouring value or an epsilon.

A second bounded typed-H200 check, **405619 COMPLETED/exit0** in three seconds,
found that only **98/442** catalogue/map disagreements have a neighbouring
HEALPix pixel agreeing with the catalogue mark within 0.05; the zero-support
point is among the 98. The remaining 344 are not explained by a one-pixel
shift. The [locality report](/gpfs/kjhan/CF4/z0_density/r2_map_discrepancy_locality_v1/result.json)
is therefore a **NO** to automatic nearest-neighbour replacement or blanket
angular smoothing.

There is also a semantic caution: the original 2M++ CDS ReadMe describes
`c11.5/c12.5` as redshift-incompleteness quantities at magnitude limits,
while the ARES example supplies its own HEALPix `maskdata` files. The
[2M++ source paper](https://academic.oup.com/mnras/article/416/4/2840/975884)
describes sky-dependent redshift completeness and final survey masks;
neither that source nor the inspected ARES example asserts exact pointwise
equality between these released products. Therefore **442 differences are
not, by themselves, proof of 442 bad galaxies**. The current mismatch-based
quality exclusion is historical and needs independent source/model
justification before reuse as a thinning law. This observation does not
remove the one zero-support contradiction under the literal ARES map.

Decision: the point/mark representation and algebraic kernel are accepted as
development infrastructure; the literal full-sample point-process likelihood
is **NO-GO** because of the observed zero-support point and unresolved
map/catalogue relationship. The actual CF4 conditional group-distance law is
still absent. No R2 posterior or high-resolution local density/IC was made.

Later same-day correction: an individual-point likelihood is **not mandatory**
for the N128 density factor. The ARES/BORG source uses voxel counts and
voxel-integrated selection. The source-coherent coarsened-count development
route and its remaining association/CF4-mark factors are recorded in
`CF4_R2_COARSENED_MARKED_COUNTS_20260926.md`. This addendum does not promote
the old binned-count × BGc product or the uncalibrated point kernel.
