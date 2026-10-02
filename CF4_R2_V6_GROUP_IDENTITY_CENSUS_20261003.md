# R2 — v6 multi-member catalogue identity census (2026-10-03)

## Driver plan review

The frozen T10106 control has two secure members assigned to the same CF4
`1PGC` and the same 2M++ `GID`. That is a catalogue concordance for one group,
not a basis for extending the assumption to the other 271 multi-link groups.
This bundle checks the same source-ID relations across the complete
score-blind, training-only v6 multi-link set before implementing a shared
group factor.

- Q-GOAL: yes. The factor must use the observed association graph correctly
  when linking group distance marks to count-owned members on the same field.
- Q-LEAN: one census over 272 already eligible groups is sufficient for the
  structural question. Read IDs and frozen split roles only; do not score
  marks, fit covariance, inspect a field, or add a posterior gate.
- MW/M31 remain role-ambiguous and M33 unresolved. This source census does not
  identify generated components; their observables must later constrain the
  same NEW evolved field at LG resolution, with native identities reserved
  for calibration/evaluation.

Per the user's routing instruction, an immediately consecutive or duplicate
Fable audit is conducted by the driver. The driver therefore reviews the
cohort/source joins directly; no outside review is stacked onto the preceding
driver review.

## Bounded implementation

`scripts/cf4_r2_v6_group_identity_census.py` reuses the frozen v6 group
eligibility rule and the already selected training graph. It classifies each
eligible Tempel source group independently under the CF4 `1PGC` and 2M++
`GID` namespaces, and counts crossmatch-vs-CF4-member-table agreement. It
reads only IDs and split roles; no redshift values, FP marks, likelihoods,
heldout measurement values, field state, or PM evolution are used. Existing
source hashes pin all inputs. The T10106 result is included as a regression
control.

The one Slurm job runs a focused relation test and the census: typed H100,
one CPU, 2 GiB host memory (over 20% above the estimated <0.5-GiB working
set), and 10 minutes. H200/H100/A100 modes were checked; H100 was selected.
This answers only whether observed catalogue partitions align in the training
multi-link sample. It cannot establish physical membership, inclusion
probabilities, member-redshift/FP covariance, or a same-field likelihood.

## Result and driver assessment

Typed-H100 Slurm job 410960 COMPLETED/exit0 in five seconds; the focused test
passed and MaxRSS was 3.44 MiB under 2 GiB. All 828 secure crossmatch edges
agree with the CF4 member table's `1PGC` assignment. Across the 272 eligible
training groups, the CF4 and 2M++ within-catalogue relations are:

| CF4 `1PGC` relation | 2M++ `GID` relation | Groups |
|---|---|---:|
| shared | shared | 167 |
| shared | partly unassigned | 32 |
| shared | all unassigned | 54 |
| shared | distinct | 11 |
| distinct | shared | 4 |
| distinct | partly unassigned | 4 |

Thus 167/272 (61.4%) have a single shared catalogue assignment within each
catalogue; the other 105 require explicit unassigned/multiple-group handling.
This does **not** mean that CF4 `1PGC` equals 2M++ `GID`, nor that either
catalogue grouping is physical truth. T10106 is reproduced as two PGCs in
`1PGC=53982`, two count points in `GID=2887`, and two consistent crossmatch
edges. No velocities, FP marks, likelihood, heldout measurement, field state,
or PM evolution were read.

Driver assessment: Q-GOAL is yes because this determines observed group
ownership required by the shared same-field mark factor. Q-LEAN is yes because
one complete small ID census resolves the cohort structure with no score or
posterior gates. This is not covariance calibration. The next single source
bundle will read only training group/member redshift fields for these existing
272 groups, summarize shared-member and group-velocity overlap by the above
relation classes, and then decide whether a covariance can be specified from
available source definitions. It will not fit that covariance to produce a
posterior; group inclusion remains uncalibrated.
