# Selected FP marks: driver assessment of focused Fable advice

R2 remains incomplete. No active target was changed.408337 continues the
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
