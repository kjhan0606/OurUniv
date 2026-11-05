# R2 selected-mark denominator audit — 2026-10-05

## Question and result

Can the existing CF4, Tempel17, SDSS-PV and 2M++ catalogues identify the
probability that a CF4 distance group entered the observed catalogue? No. The
source-level denominator checks remain closed descriptively; no empirical
`S_group(d,F)` law or production CF4 group-inclusion correction is obtained.

There is, however, a narrower conditional estimand already represented by the
active v6 linked cohort: the FP distance marks for the 1,414 training rows,
conditional on their observed, securely matched 2M++ count points and on the
marks being present. Each row has one secure training-count link, one FP mark,
one distinct Tempel parent/singleton, and no unresolved association. This can
avoid modelling the *number of CF4 groups* only for this selected linked
subset. It is not the full CF4 survey likelihood, nor does it establish that
mark availability is ignorable.

## What the FP correction means

Howlett et al. §5.1 define `f_n` for each individual SDSS-PV galaxy and each
candidate distance. It normalizes the FP measurement likelihood over that
galaxy's stated magnitude and velocity-dispersion cuts (with the source's
redshift limits and distance convention); the published log-distance PDF
already includes this correction. It is not the probability that a Tempel or
CF4 group entered the catalogue, does not model redshift-success failures,
and does not supply a parent list of groups with no distance measurement.
The paper separately documents a richness-dependent residual and corrects it
using richness-specific FP fits. [Howlett et al. (2022), §§5.1–5.3](https://academic.oup.com/mnras/article/515/1/953/6611706).

The local implementation is consistent with that limited interpretation:
`src/cf4_r2_fp_distance.py` evaluates the published FP likelihood shape and
explicitly does not multiply another `f_n`; `src/cf4_r2_raw_volume_target.py`
normalizes the raw FP mark against the same source-selected count-point
kernel in numerator and denominator. The count Poisson occurrence remains
scored once. Do not add a second FP-selection factor or a separate group
incidence term to this conditional mark factor without changing the estimand.

## Why the wider denominator remains unavailable

- The SDSS-PV/Tempel identity join is not a complete DR8–DR14 targeting and
  redshift-success parent. The 418 catalogue-absent full-sample rows and 75
  eligible absent rows are source-ungrouped, not observed non-detections.
- The exact NYU-VAGC DR7 mask used for the published `in_mask` flag is not the
  full PV selection. The nearest-random mask proxy was rejected (43.390%
  false-positive rate on native `in_mask=0`); neither it nor a different
  raster may be substituted as an inclusion denominator.
- CF4, Tempel17 and 2M++ group definitions are not interchangeable. The
  observed association graph, including unlinked and ambiguous rows, is not a
  sample of known CF4 failures.
- The SDSS-PV mock ensemble diagnoses selected-host residuals, but lacks
  redshift-success effects and a demonstrated mapping from mock host
  richness to Tempel17 membership. Its residual shifts do not calibrate
  transfer corrections for CF4.

These conclusions are supported by the preserved source records
`CF4_R2_TEMPEL_PARENT_DENOMINATOR_20260928.md`,
`CF4_R2_GROUP_SELECTION_AND_CONDITIONAL_BUNDLE_20260926.md`,
`CF4_R2_SDSS_MOCK_STREAM_PLAN_20261005.md`, and the active v6 ownership
ledger/reconciliation. No catalogue was re-downloaded, no heldout value was
read, and no calculation or simulation was launched for this audit.

## Driver disposition and next boundary

Keep R2 **NO-GO for a full calibrated CF4 posterior**. Do not continue trying
to infer `S_group` from the selected-only catalogues. Retain the 1,414-row
linked mark term only as an explicitly conditional component alongside the
v6 training-count process; this does not yet make a stationary posterior or a
delivered z=0 map. The separate 2M++ selection/tracer-bias calibration,
redshift-success/mark-availability assumptions, posterior stationarity and
untouched heldout prediction remain unresolved.

The next material choice is whether to spend the next field-inference bundle
on a clearly labelled **conditional v6 diagnostic** (47,121 training count
rows plus the 1,414 linked training FP marks, with the mark-presence pattern
conditioned upon and the untouched v6 holdout reserved for a final predictive
check), or to defer field inference until external parent/redshift-success
calibration is obtained. Neither choice may be called a full CF4-calibrated
posterior. This audit does not authorize a large sampler run or change the
scientific target on its own.

Q-GOAL: the conditional linked-mark factor can constrain the same z=0 field,
but it does not establish the full CF4 distance-group incidence law or the
LG-scale result. MW/M31 remain ambiguous and M33 unresolved; all three must
ultimately constrain the same NEW evolved field at `<=0.3 cMpc/h`, with
native truth IDs reserved for calibration/evaluation.

Q-LEAN: one source-definition/code-path review; no repeated joins, archive
pass, denominator proxy, gravity run, or sampler tuning. The next bundle
should begin at the actual field-inference boundary, not another source census.
