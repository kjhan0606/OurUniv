# R2 selected-mark holdout contract — 2026-10-21

Status: implementation contract only. No held-out measurement is extracted,
no predictive score is produced, and R2 remains NO-GO.

## Score-blind identity manifest

The v6 role-1 source cohort is a candidate set, not yet a certified evaluation
set. The metadata-only join found 197 single-mark candidates under the
existing membership rule. Existing materialized training assets cover 0/197
candidate PGCs. The role-aware manifest builder is
`src/cf4_r2_mark_holdout_manifest.py`; its driver is
`scripts/cf4_r2_mark_holdout_manifest.py`.

The builder reads only the v6 split IDs/roles, source-group/PGC membership and
ownership identifiers, secure crossmatch classes, and the existing parent/GID
identity links. It must fail closed for a non-unique source group, PGC or
linked point; anything other than one mark and one secure role-1 point; an
anchor; a disallowed membership state; an unresolved edge/collision; or a
recorded Tempel-parent, 2M++ GID, or CF4-group role outside role 1. It does
not read split count keys/counts, point positions/radii, photometry, raw
velocity marks, field states, population fits, or training-mixture weights.
It does not rehash whole NPZ/catalogue files because those containers also
hold embargoed outcomes; it records the frozen source digests already in the
v6 provenance plus digests of the identity/role fields actually consumed.

The local unit fixture checks role agreement, unmatched-edge handling,
crossmatch collisions, unresolved cross-fold links, parent/GID/CF4 role
mismatches, duplicate identities and multiple linked points. A metadata
manifest does not establish physical membership or make availability bias
ignorable.

## Frozen post-fit evaluator

Do not materialize or validate the 197 raw measurement rows until a valid
post-fit evaluation state exists. The bounded one-sweep ESS feasibility job
is not a posterior and does not release the embargo. The eventual extraction
must be identity-restricted to the certified manifest; it must report missing,
duplicate, or invalid rows rather than dropping them according to their
scores.

Reuse the active conditional-v6 operator and its conventions:

- Raw FP inputs: the published `r`, `s`, `i` marks and reported errors; the
  selected-row cut inputs (`deVMag_r`, `SIGMA_STARS`, aperture metadata, plate
  and magnitude/quality fields); source membership/richness; within-bin
  2M++ K magnitude; and the fixed population coding. The optical-error
  covariance and affine cut geometry are the existing prescriptions in
  `src/cf4_r2_raw_selected_fp.py`, not newly fitted error laws.
- The field-dependent source mixture must be rebuilt for every retained
  state from that same evolved density/velocity field, shared tracer and
  population state. Do not reuse `raw_source_mixture_v1` weights, which are
  frozen at a different training state.
- The **linked 2M++ point radius** conditions the source/RSD kernel. The CF4
  row keeps its own frozen `dz_row` reference in the eta/mark law. The point
  radius, point K magnitude and CF4 `dz_row` have separate operator ownership;
  do not substitute one for another, even where a singleton's numeric radii
  happen to coincide.
- Reuse `raw_field_logpdf` / `streaming_raw_mark`: the within-bin K density,
  normalized optical-cut denominator, shared population parameters and
  coordinate-density/Jacobian conventions remain unchanged. Do not add a
  second `f_n`, eta likelihood, group-redshift observation, or repeated count
  occurrence. Population fit, richness centering, nuisance scaling and other
  fitted preprocessing must come from the training fold only.

For each retained state, evaluate count and mark factors on the **same evolved
field and shared nuisance state**:

```text
ell_joint[s] = ell_count_holdout[s] + sum_i ell_selected_mark[i, s]
log p(D_holdout | retained states) = logmeanexp_s(ell_joint[s])
log p(D_marks | D_counts, retained states)
    = logmeanexp_s(ell_joint[s]) - logmeanexp_s(ell_count_holdout[s])
```

`src/cf4_r2_raw_mark_holdout_inputs.py::prepare_postfit_mark_observations`
defines the identity-keyed row alignment, existing optical-error covariance,
affine cut geometry, and training-centered richness transform. Its tests use
synthetic rows only; the helper is not called on actual holdout data before
post-fit. The helper `JointCountMarkHoldoutPredictive` implements predictive
aggregation and keeps repeated retained states and legitimate
negative-infinity likelihoods.
Do not add separately marginalized count and mark log scores: they generally
represent different mixtures. Joint-score Monte Carlo reliability must be
reported separately; a chain being called “valid” is not itself evidence that
the predictive log-mean is numerically stable.

This is a score for the frozen **selected/linked conditional-v6 cohort**, not
a calibrated full-survey CF4/2M++ likelihood. It cannot repair missing
eligibility, targeting, redshift-success, FP-quality, linkage or group
inclusion probabilities, nor can it establish calibrated field uncertainty.
It is validation of one needed component of the first z=0 posterior at 1.5
cMpc/h in the 384-cMpc/h box—not the <=0.3-cMpc/h LG deliverable.

MW/M31 remain role-ambiguous and M33 unresolved. Their observables must
ultimately constrain those same roles on the same **NEW evolved LG field at
<=0.3 cMpc/h**. Native truth identities are calibration/evaluation labels
only, never candidate seeds or selectors. No R2 exit audit, exit email, or R3
entry follows from this contract.
