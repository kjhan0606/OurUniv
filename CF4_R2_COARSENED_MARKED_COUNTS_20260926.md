# R2 coarsened count / shared-mark bundle — 2026-09-26

## Decision in relation to the goal

R2/5 still targets an actual-data-conditioned present-day density/velocity
posterior from one latent LCDM IC state. The previous individual-point
candidate was a **development model**, not a requirement to score every
within-cell coordinate. The [2M++ ARES/BORG source paper](https://arxiv.org/pdf/1509.05040)
defines a voxel-integrated target selection (its equation 1) and a
voxel-count observation likelihood (its equation 5). It does not require
the zero-support individual-point likelihood that our earlier candidate
attempted. A galaxy at a zero-valued HEALPix pixel can contribute to a
positive-exposure *voxel count* under that coarsened model; this is a model
resolution choice, not permission to insert a pointwise epsilon.

The appropriate narrow R2 candidate is therefore: score six-population
N128/384 **counts** with the cell-integrated ARES selection, retain each
2M++ galaxy's measured redshift/angle/recno as an observed covariate for
the eventual **conditional** CF4 mark law, and do not score those redshifts
again in a separate independent CF4 velocity factor. This does not make the
joint likelihood complete. If `C` denotes counts, `U` within-cell locations
and individual redshifts, `A` the observed association/selection graph, and
`G` CF4 group marks, a full generative law would require

`p(C,U,A,G|F,S) = p(C|F,S) p(U|C,F,S) p(A|C,U,F,S) p(G|A,C,U,F,S)`.

Only the **first** factor is evaluated in this bundle. Treating
`p(U|C,F,S)` as field-independent is an explicit coarse-resolution
approximation, not source-calibrated truth. The association/group-selection
factor and conditional CF4 distance/velocity mark factor remain absent.
Consequently the old inclusive count × BGc product is still **NO-GO**.

For later R3, MW/M31/M33 must be identified from each **new** evolved field,
with MW/M31 role ambiguity and unresolved M33 retained. Their observables
must constrain that same state. Native identity labels can calibrate or
evaluate only, never seed or select generated-field candidates. This bundle
does not claim an LG member identification or 0.3 cMpc/h information.

Q-GOAL: the coarsened count target preserves all eligible 2M++ density
information at the chosen N128 resolution without forcing an unsupported
pointwise selection law. Q-LEAN: one exact source join and one count-factor
control, no new gravity run, posterior sampling, pixel repair, or validation
ladder. The next work is the missing association/CF4 conditional mark law,
not another homogeneous count score.

## Source and computation

The pinned ARES example `2MPP.txt` has 67,224 rows. A coordinate/redshift/
magnitude bridge with fixed tolerances matched **67,222** uniquely to the
72,973-row VizieR source; 57,235 of the present 57,238 N128 eligible rows
appear in that ARES example. The one map-zero galaxy, recno **67100**, is
among them and has the ARES example's faint apparent magnitude and accepted
absolute magnitude. Thus a simple "wrong source catalogue" explanation for
its zero HEALPix value is rejected. The [bridge result](/gpfs/kjhan/CF4/z0_density/r2_ares_catalogue_bridge_v1/result.json)
and mapping preserve two unmatched ARES rows rather than forcing them.
Source bridge job **405649 COMPLETED/exit0** in 2 s.

`src/cf4_r2_coarsened_observation.py` implements the exact sparse Poisson
cell-count log PMF, including the expected count in *empty* cells and no
observed-count rate renormalization. Three focused tests check the full
integral, zero occupied-cell support and the difference between pointwise
zero and positive integrated-cell selection. The source-bound N128 control
projects all 57,238 points to **45,776** occupied population/cells exactly,
including recno 67100's occupied cell. That cell's published homogeneous
expected count is **0.0178894**, not zero; all occupied cells have positive
expectation. The homogeneous baseline's total expectation is 49,569.03
versus 57,238 observed and its count log PMF is −233,960.45. These are
*not* a rate/bias fit, model comparison, evidence ratio, posterior, or
effective-resolution measurement. Initial control **405668 COMPLETED/exit0**
in 3 s, tests 3/3. Its report mistakenly abbreviated the complete
factorization by omitting the association-selection factor; the calculation
itself was unchanged. Corrected metadata is in the separate
[v2 result](/gpfs/kjhan/CF4/z0_density/r2_coarsened_observation_v2/result.json)
from Slurm **405676 COMPLETED/exit0** in 2 s, with the same three tests and
identical numerical score; v1 remains preserved as a superseded
interpretation. The v2 report explicitly lists all three missing factors.

## Scientific boundary and next bundle

The only result promoted here is a **finite, normalized N128 count factor**
under its stated integrated-selection and published-rate baseline. It does
not show that a count-only analysis captures the actual within-cell angular
selection or the field dependence of CF4 matching. In particular, the
15,239 crossmatch edges include ambiguous/repeated points and four edges
without a canonical group row. The earlier two-stratum CF4 group-redshift
conditional undercovered the native holdout, so it is not imported as
`p(G|...)`. Published distance errors alone do not establish distance-
indicator selection, group inclusion, or cross-covariance.

The next substantive bundle must build/calibrate the **conditional CF4
distance-and-group-redshift observation law and association selection on
the same evolved field**, or establish a deliberately narrower conditional
target with its omitted information and bias bounds. No N256 sampler,
production density map or IC launch follows from a finite count PMF.

Fable5 returned a read-only **CONDITIONAL PASS** on the coarsened development
direction. The driver adopts its count-resolution argument but rejects two
unsupported suggestions: current CF4 group redshift is not deterministic
from the secure eligible 2M++ redshifts, and a selected-only CF4 holdout
cannot calibrate inclusion probability. The crossmatch-edge predicates and
MW/M31/M33 limits are reconciled in
`CF4_R2_COARSENED_FABLE_DISPOSITION_20260926.md`. The observed edge table
may be fixed as bookkeeping conditional on both catalogues; CF4 distance-
group *inclusion* and ambiguous links are separate unresolved factors.
