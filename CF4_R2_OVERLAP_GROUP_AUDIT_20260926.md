# R2 CF4–2M++ overlap: grouped-redshift source audit — 2026-09-26

The current N128 disjoint-tracer selection cannot be promoted to the R2
posterior: the previous calibration-mark audit found 3,120 of 3,183 failures
were CF4 crossmatches. **Simply putting those galaxies back into the binned
2M++ count factor and multiplying by the existing grouped BGc CF4 velocity
factor is not an established joint likelihood.** The frozen source contract
itself requires a shared-redshift latent if excluded targets are reintroduced.

`scripts/cf4_r2_overlap_group_audit.py` reads only the three source CSVs and
two preserved row/group lists, verifies source hashes, and makes no new
catalogue or inference product. The scoped regression test passes. Results:

| Source-level quantity | Value |
| --- | ---: |
| Full crossmatch unique 2M++ targets (all non-unmatched classes) | 17,007 |
| N128 eligible parent rows | 57,238 |
| Eligible crossmatch targets (all classes) | 15,211 |
| Eligible secure targets | 14,878 |
| Calibration-only crossmatch targets | 3,120 |
| Eligible secure targets in native CF4 groups | 14,026 |
| Eligible secure targets with no canonical CF4 group row | 4 |
| Canonical CF4 groups with multiple eligible secure 2M++ members | 1,635 |
| Calibration-only noncrossmatch failures | 63 (56 old metadata exclusions; 7 new map-rule failures) |

Across the eligible secure rows with a canonical group, the absolute
individual CF4-versus-2M++ `Vcmb` difference is median 9, p90 50, p99
206 km/s. More importantly, **CF4 group `V3k` versus individual 2M++ `Vcmb`**
differs by median 52, p90 437, p99 1,467 km/s (max 4,801 km/s). This is not
an estimate of redshift measurement error: it includes group/member velocity
differences and possibly group construction. It demonstrates that the
existing one-group-one-redshift simplification does not equal the individual
redshift driving each count-cell assignment. The four missing group keys
are quarantined from any automatic group-mark join.

## Likelihood ownership and scientific decision

Let `Y` be the *individual* 2M++ observed point catalogue (angles,
redshifts and selection-relevant magnitudes), `G` the CF4 group catalogue,
and `F(s)` the common evolved field. The safe target must be a declared joint
`p(Y,G | F(s), nuisance, membership)`—for example a point-process factor
`p(Y|F)` times **properly normalized conditional group-distance/redshift
marks** `p(G|Y,F,membership)`, with a shared member/group velocity model and
the observed CF4 group construction. Quarantined matches must be marginalized
under a predeclared association model or omitted with an explicit selection
condition; they cannot be called secure. This is a *target factorization*,
not a working implementation. Marginalizing `Y` down to six N128 count grids
loses individual within-cell redshifts needed for an exact conditional mark
construction; the current binned Poisson × BGc Gaussian product therefore
needs a calibrated composite-likelihood approximation or a new joint model.

## External advice and driver disposition

Fable5 returned **CONDITIONAL PASS** for a leaner candidate: include every
eligible 2M++ target in the count factor regardless of CF4 matching, retain
exactly one existing grouped BGc CF4 factor per group, and interpret the
product as a **partial/composite likelihood** conditioned on observed group
redshift. This is worth a bounded test because it removes the dominant
crossmatch-driven count thinning without per-member duplication of CF4
distance marks. It does *not* prove that the product is the exact joint
likelihood. In particular, its conditional-independence assumption
`D_group ⟂ Y_individual | V3k_group, F, membership, selection` has not been
checked; BGc reference-pool estimation, distance-derived field positions,
group/member mismatch and selection all remain approximate. The advice's
statement that omitting the selection and group-redshift factors **cannot
bias** inference is not established for this implementation, so the driver
does not promote that claim or a posterior on its basis.

Fable's proposed conversion of residual metadata failures to angle-only
masks is **not adopted**: the 63 noncrossmatch calibration failures comprise
56 source-bound individual metadata exclusions and seven newly checked
map-rule failures. An object-specific catalog discrepancy is not in general
an empty-sky angular mask. Nor does removing the Beta survival parameters
prove the new count model calibrated; it only removes the *particular*
importance-sampling collapse already observed.

Next bounded implementation: build a new, preserved **diagnostic-only**
inclusive 2M++ row/count view from frozen eligibility, treating imputed
ZoA/cloned-redshift rows and the 63-type metadata anomalies separately;
never replace the old catalogue in place. Check count/exposure support and
one grouped/individual-redshift-dependent mock under a single saved state.
Test conditional-mark score/calibration for the candidate partial likelihood,
including a multi-member negative control, before quantitative actual-data
sampling. If it fails, use the fuller joint point/mark likelihood; do not
rescue the score with an arbitrary octant field or a new phase search.

The first part is submitted as Syntax Slurm **405248** using an A100 typed
request, one CPU, 3 GiB and a 10-minute cap. H200/H100/A100 test submissions
all passed; A100 had the earliest estimated start. The script
`scripts/cf4_r2_inclusive_count_diagnostic.py` reproduces the old disjoint
counts exactly, then stores inclusive sparse counts and per-row exposure in a
new `/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1` root.
Job 405248 **COMPLETED/exit 0** in two seconds. The preserved result has
57,238 parent rows, with 34,301 train, 11,435 heldout and 11,502 calibration;
15,211 parent rows are crossmatched. It reproduces the old disjoint sparse
counts exactly. Every eligible row has **positive N128 integrated-cell
exposure**, including the 15,211 crossmatches and 340 noncrossmatched
metadata-anomaly rows in the full parent. This is a cell-level support check,
not proof that all 340 objects have positive pointwise angular selection or
that the count/CF4 partial likelihood is calibrated. The old 63 number refers
only to the 20% calibration slice. No likelihood or posterior was evaluated.

A source-bound **pointwise** follow-up, Slurm405253 (COMPLETED/exit0, seven
seconds), used the official HEALPix maps at every eligible galaxy coordinate
and exactly reconstructed the old survivor flags without using the CF4 match
in the quality rule. There are442 object-level mark/map discrepancies among
57,238 eligible rows:102 crossmatched and340 noncrossmatched. Only **one**
row has zero pointwise map value; all442 are mark discrepancies by the frozen
0.05 threshold. The old319-row exception manifest intersects316 of the
current eligible nonmatches, so it is not a complete new-model anomaly list.
The full340 nonmatch failures include the63 in the calibration slice. These
are observed-data inconsistencies, not an inferred sky-completeness mask.
The inclusive diagnostic's57,238 rows deliberately retain them for audit;
its zero-*voxel*-exposure result cannot certify their selection likelihood.

BGc `vobs` and its position are functions of CF4 grouped redshift and
distance, and the transform uses a training reference pool. The current
R2 implementation's overlap guard is a **global disjoint exclusion**, not a
shared-redshift model. The older `src/cf4_2mpp_joint_likelihood_local.py`
ownership guard prohibits an independent 2M++ redshift factor but does not
connect individual redshift-space count cells to the grouped BGc transform;
its mere presence does not resolve this R2 defect.

No heldout count row, IC phase, native halo identity, MW/M31/M33 assignment
or candidate selection was used. The old disjoint and count artifacts are
preserved. In particular, 17,007 is **not** the number of currently eligible
crossmatched count rows. No inclusive count catalogue, rate correction or
N128/N256 posterior is promoted by this audit.

Q-GOAL: resolving the shared-data/selection likelihood is required before a
CF4-conditioned z=0 posterior. LG role identification—including unresolved
M33 and ambiguous MW/M31/M33 components—must later use observables on the
*same generated field*, never native truth identities to seed candidates.
Q-LEAN: this one bounded source check avoids a premature inclusive-catalogue
refit, an arbitrary octant survival field and another gravity simulation.
