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
