# Selected FP marks: driver assessment of focused Fable advice

R2 remains incomplete. No active target was changed.408337 completed the
approved same-target sampler-mechanics comparison, not production inference.
Read-only Fable5 returned ADVISE MODIFY to the request in
`config/cf4_r2_selected_fp_advice_20260929.md`; no files/jobs were changed by it.

The primary source's brighter-r-subset warning makes selected-mark calibration
important, but does not prove bias of this specific K-selected subset. The
existing constant-association working assumption is explicit, not established
for the actual selection. That is an open science risk, not a demonstrated
arithmetic error. No repeat of the old one-mock marginal-PDF coverage check.

## Useful advice retained

Keep the source moment/skewness conversion and its original fn correction.
Do not add another fn, widen the zero prior, refit only a small intersection
without its selection law, or change the live target. Public optical FP
observables plus K photometry can support a joint selected-population model;
the selected eta summaries alone do not identify that model. Shared zero and
group/source covariance are separate remaining issues. No new TNG or mail.

## Substantive corrections/rejections, not invocation failure

1. Deterministic selection from ALL observed catalogue features does imply
   p(theta|all features,A)=p(theta|all features) when A is redundant. It does
   NOT establish that a published likelihood conditional on a SMALLER set
   of summaries and a different selection remains valid. The discarded or
   unmodelled K/color/group information is exactly at issue here.
2. It is not generally correct to put all extra mark selection into a
   positive association mixture weight while leaving the original mark PDF
   unchanged. An elementary counterexample fixes the law: let base candidate
   weights be w_j, normalized mark laws f_j(m), selection B a mark subset,
   and Z_j=integral_B f_j(m)dm. Then conditional-on-selection candidate weights
   are proportional to w_j Z_j, BUT selected mark laws are I_B f_j/Z_j.
   Their product cancels Z_j in the numerator; the final mixture denominator
   is sum_j w_j Z_j. Multiplying w_j by Z_j while retaining f_j double-uses
   selection. Real optical/NIR/group matching is more complicated, not a
   reason to skip deriving its joint law. Our likelihood RATIO is not itself
   a normalized sampling law over photometric data, so this example cannot
   be inserted into the code blindly either.
3. An observed P(link|apparent K,z) slope is descriptive. At fixed observed
   flux, changing latent distance changes INFERRED luminosity, not that
   observed flux. Converting the observed slope into d ln Z/d ln distance
   needs a luminosity/color/type population law. The proposed scalar fit
   does not identify that derivative, or bound it, merely by binning in z.
4. A distance-only tilt of a collapsed-K candidate mixture is a sensitivity
   experiment, not an exact upper bound on unknown K/mark-dependent selection.
   Comparing its score shift to the empirical spread of row logfactors is
   not a calibrated negligible-bias criterion. No arbitrary pass threshold.
5. Correctly conditioned K-selected FP refitting is not inherently double
   counting the Poisson count occurrence. That depends on the factorization
   and normalizers. The blanket 'yes, double conditioning' assertion is too
   strong; a naive refit may still fail to describe the required joint law.
6. Advice's1,836 rows is stale: the current target has1,414 single-mark links.
   It would not close the 'last' R2 limitation: shared source calibration,
   group/RSD discrepancy, mixing, resolution and heldout prediction remain.

Thus reject the suggested observed-slope -> physical-association calibration
and its score-spread gate as unsupported. This is NOT a failed Fable invocation
and is not permission to seek a more favorable verdict. The driver accepts
the warning but does not implement an unvalidated repair.

Existing official mock has optical magnitudes/FP observables and true eta,
but no K-band selection or recovered Tempel groups. Its previous marginal
source-PDF coverage result cannot calibrate this actual intersection. Repeating
brightness cuts on that fixture would illustrate the already-published risk,
not solve the missing optical/NIR joint law. Preserve it; no new download.

Q-GOAL: prevent a falsely precise CF4 current-state posterior, while retaining
the same field for future MW/M31/M33 identification and constraints. These
roles remain ambiguous/unresolved; no truth identities seed candidates.
Q-LEAN: no new slope-fitting pipeline, control simulations or arbitrary
selection-offset family follows. Resolve actual model identifiability and
target definition before any calibrated-posterior promotion.

## Actual raw inputs, not a new correction

408340/sourcef02ad1f completed6s: joined existing training optical FP columns
and uncertainties to2M++ K photometry for the SAME1,414 links; the existing
CF4-linked training parent contains8,901 rows, NOT all34,059 SDSS galaxies.
Result: `/gpfs/kjhan/CF4/z0_density/r2_raw_fp_inputs_v1/`.
The actual figure shows a brighter optical distribution in the selected
intersection and strong apparent-r/K association. This confirms the changed
observed sample, NOT the size or sign of a distance bias conditional on true
distance. No parameter was fitted, no heldout score or field run occurred.

Magnitude systems were preserved separately; the2M++ table used here has no
K measurement-error column. No zero error, diagonal optical covariance, or
physical colour conversion is silently inferred. Raw r is still the source's
group-redshift-based radius, not an independent true size/distance. The public
release [file list](https://zenodo.org/records/6824749) supplies no separate shared FP-fit covariance artifact;
this is not proof that no such information exists elsewhere.

Next priority is a defensible selected-mark/shared-calibration model using
these actual observables, not an automatic longer conditional chain. The
present coefficients/selection PDFs cannot simply be refitted on the narrow
intersection and reused as independent priors. A raw-observable likelihood or
a rigorously derived conditional alternative needs an explicit joint law.

The within-count-cell radial refinement is another unresolved approximation,
but it must not be 'fixed' just by appending a log radial density minus log
cell intensity: current TSC count response and measured-point radius have to
be coarsenings of the SAME observation law. The existing continuous mark
kernel also neglects periodic aliases represented by the count quadrature.
Before a refinement, define its measure/support/normalizer and demonstrate
that marginalizing the added observable restores the count law. A conditional
design approximation can be declared, but does not establish physical
calibration by itself. No new radial likelihood has been implemented.
