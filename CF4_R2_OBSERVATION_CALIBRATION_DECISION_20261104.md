# R2 observation-calibration decision — 2026-11-04

## Current route revision — 2026-11-07

The user approved updating the goal after reassessing this blocker. Follow
the current R2 objective and route revision in `CF4_MASTER_PLAN.md`.
The missing parent still prevents claiming calibrated selection for v6, but
is not a proof that every defensible CF4-conditioned inference is impossible.
Source-native conditional FP redesign and a bounded discrimination study may
proceed; the earlier requirement to obtain a parent or a supplied ignorability
specification before any redesign is superseded. Selection cancellation,
latent-distance weights, noisy conditioning variables, residual-dependent
cuts, shared group errors and count/mark ownership must be addressed explicitly.
Discarding source-design information does not automatically reproduce the
full joint posterior. Preserve the full R2 field/calibration/held-out goal.

The working v6 target remains unchanged and NO-GO for production or posterior
promotion. No held-out outcomes, repeated parent searches, arbitrary selection
prior or new sampling budget are authorized by this revision. Astra's
mid-course redesign advice is adopted; it is not R2 exit approval.

## Earlier decision for the frozen v6 model

Do **not** repeat the native-TNG RSD/count mock or the already completed
actual-state moment/observation-operator bundle. They do not identify the
selection process for the active observed CF4 marks. Do not launch another
sampler or read held-out outcomes to compensate for this missing observation
information. R2 remains open and NO-GO for a calibrated posterior.

The next decisive input is a source-matched selection/parent specification
for the active training marks, or a documented argument that a precisely
defined subcohort is ignorable conditional on variables already present in
the likelihood. Neither is currently in evidence. The only defensible
current mark interpretation is conditional on the selected linked marks
being present; that is not a full-survey CF4 selection calibration and does
not establish calibrated posterior uncertainty.

## Driver's evidence check

- `CF4_R2_NATIVE_RSD_MOCK_20261002.md` records the native-K/RSD mock and
  subsequent physical-moment and actual-state operator work (including jobs
  410544/410551, 410592/410600, 410628/410630/410632, and 410643–410888).
  The development spatial test retained roughly a 5.9% count excess. The
  single 75 cMpc/h TNG box, native-K rather than survey Ks, different
  cosmology/deposition, and imposed source window cannot calibrate the actual
  SDSS-PV/CF4 targeting, redshift-success, FP-quality, catalogue-linkage, or
  group-inclusion process. Its split regions are not independent universes.
- `config/CF4_R2_OBSERVATION_LAW_IDENTIFIABILITY_20261021.md` and
  `CF4_R2_SELECTED_MARK_DENOMINATOR_AUDIT_20261005.md` establish that the
  active v6 evidence comprises 47,121 training count rows and 1,414 observed
  linked SDSS-PV/CF4 FP marks. The marks have secure observed links and
  distinct selected Tempel parents, but the files do not contain the
  source-matched preselection/failed-target parent or the CF4/Tempel linkage
  denominator. The published FP `f_n` correction covers stated object-level
  magnitude and velocity-dispersion cuts; it is not a group-entry,
  redshift-success, or mark-availability law.
- The selected-mark and target notes explicitly leave mark-presence
  ignorability unestablished. A conditional likelihood for observed marks
  cannot estimate its own missing-mark process from selected marks alone.
  Secure linkage/singleton status removes some association complications for
  the selected rows; it does not establish the probability that eligible
  sources or groups entered that selected set.
- The active target code and conditional-map disclosure are consistent with
  that boundary: the raw FP term conditions on observed marks and their
  linked count points; the diagnostic records no calibrated group-incidence
  or mark-availability model. Its physical-variance readout is not posterior
  uncertainty. This is a separate limitation from sampler convergence.

This check adopts Astra's advice to reject a repeated mock bundle, after
independently checking the cited repository evidence. The advice is
mid-course guidance, not an R2 exit audit, posterior promotion, or R3 approval.

## Selection-factor cancellation audit — 2026-11-04

Astra's later read-only advice was **NO-GO** for treating a public DR14
reconstruction as closure of this blocker, but **CONDITIONAL GO** for using
the existing SDSS-PV mocks only to test the FP subfactor under their own
selection model. The driver checked the current factor wiring directly:

\[
\widetilde\pi_{v6}(F,\theta\mid C,Y_I;I)\;\propto\;
\pi(F,\theta)\,p(C\mid F,\theta_c)
\prod_{i\in I_{\rm linked}}q_{v6}(Y_i\mid C_i,T_i=1,F,\theta_p),
\]

where \(F\) is the evolved field, \(C\) is the observed 2M++ count
catalogue, \(A_i\) denotes entry into the linked CF4/Tempel/SDSS-PV mark
cohort, \(I\) is the fixed list of observed linked marks, and \(T_i\)
denotes the explicitly modeled per-object FP measurement cuts. The tilde
and \(q_{v6}\) explicitly denote a working target, not an established
selected-cohort likelihood. In code, the first factor is the single sparse Poisson
count score (`count_field_loglike`); the second is the source-weighted raw
mark numerator divided by its same-state cut-normalization integral
(`raw_field_logpdf` / `row_logpdf`). The latter includes the declared finite
K/magnitude support and FP cut rectangle. It does **not** supply
\(p(A\mid C,F)\): source eligibility, targeting, redshift success, usable
mark availability, and catalogue linkage into the 1,414-row cohort are not
represented by that denominator. Nor is another `f_n` factor appropriate;
the published object-level correction is already reflected in the source
likelihood definition and is not group inclusion.

This omission is not automatically fatal to every conditional likelihood.
If the true selected-mark conditional were known, its separate cohort-entry
factor could be dropped when field-independent after conditioning on the
variables actually held fixed. The implemented cut-normalized working factor
is not yet that true selected-mark conditional: field independence of an
overall entry rate alone would not establish the required cancellation
within its mark/source mixture. A sufficient conditional-independence
argument or an explicit joint missingness factorization is still needed.
If availability depends on latent distance, velocity,
environment, FP residual, or unconditioned group properties, the term need not
cancel. The frozen 47,121 count rows and 1,414 selected marks contain neither
the eligible non-mark roster nor a demonstrated set of sufficient observed
conditioning variables; the code does not establish the required
independence. Therefore the existing factorization remains an explicitly
conditional development target, not evidence that mark availability is
ignorable or a full-survey calibrated posterior.

Driver disposition: adopt Astra's distinction, but do not repeat the DR14
catalogue/mock sweep, request data by email, add an arbitrary selection prior,
or launch a sampler to compensate. Preserve the conditional-v6 boundary and
R2 NO-GO. If a source-matched denominator is unavailable, only a separately
justified selection-independence argument could change this decision; the
current selected-only evidence does not provide one.

**Q-GOAL:** this resolves which observation factor is presently represented,
not the required z=0 posterior, its uncertainty, or held-out prediction. MW
and M31 remain role-ambiguous and M33 unresolved; all three must ultimately
constrain those same components on the same NEW evolved LG field at
`<=0.3 cMpc/h`, with native truth identities reserved for calibration and
evaluation.

**Q-LEAN:** one code-to-likelihood factorization check using existing files;
no new catalogue extraction, archive pass, simulation, sampler screen, or
additional instrumentation. The result does not close observation
calibration.

## Unavailable-input disposition and conditional notation — 2026-11-07

The user confirmed that neither the source-matched parent/attempt/usable-FP/
linkage input nor a defensible selection-independence specification is
currently available. This is a scientific input dependency, not a running
job or a permission wait. Do not repeat the same request, public-parent
search, mock pass, or source census; do not send email. Reactivating the R2
goal does not supply those inputs or authorize an arbitrary selection law.

The driver rechecked `raw_field_logpdf` and the raw-FP normalization in
`row_logpdf`: the denominator integrates the stated measurement cuts, not
the additional availability/linkage event. The earlier displayed equation
was too strong in naming that working factor directly as
`p(Y | C,A=1,T=1,F)`. It is corrected above to `q_v6`; the code and previous
NO-GO verdict are unchanged. Conditioning the written target on an observed
index list is not itself a selection correction.

For example, if `q_F(y | C,T)` were the correct pre-availability mark law,
an additional effective selection probability `a_F(y,C,T)` would give

\[
 p(y\mid C,T,A=1,F)
 =\frac{q_F(y\mid C,T)\,a_F(y,C,T)}
 {\int q_F(y'\mid C,T)\,a_F(y',C,T)\,dy'}.
\]

Selection depending on unobserved distance or environment can make the
effective `a_F` field-dependent even without an explicit field argument in
the survey's targeting rule. As a purely algebraic illustration, two
preselection mark laws `(0.8,0.2)` and `(0.2,0.8)`, paired respectively with
selection probabilities `(0.25,1)` and `(1,0.25)`, both yield inclusion0.4
and selected marks `(0.5,0.5)`. This is not an actual CF4 mock or a prior:
it shows why selected marks, or even a total inclusion fraction alone, do
not generally identify the unknown mark-dependent selection. It does not
claim that the galaxy-count data contain no information about the field.

Driver disposition: no production or multi-chain target, held-out outcome
read, FP-dropping substitution, or claimed calibrated conditional posterior
follows. Preserve the full R2 objective. The remaining in-scope analytic
precision reconciliation is recorded in
`CF4_R2_CONDITIONAL_V6_POSTERIOR_RUN_CONTRACT_DRAFT_20261104.md`; it does
not resolve selection or make the conditional working model a science
delivery. Routine interpretation checks are driver-reviewed; this is not
an R2 exit audit or a new external-review request.

## Minimum source-matched information to change the decision

For the active training cohort only, seek documentation or a parent extract
that identifies each relevant stage and keeps failures/non-links in the
denominator:

1. **Parent and eligibility:** exact source release/program, footprint and
   preselection parent; stable identifiers and the variables/cuts that made
   an object eligible before targeting or measurement.
2. **Observation outcomes:** per-stage target/attempt status, observation
   coverage, redshift success/failure, and reasons for failure or exclusion.
3. **FP/mark outcomes:** FP-candidate and fit-quality flags, the actual
   magnitude/dispersion/other quality cuts, usable-mark versus rejected-mark
   status, and the selection variables available before each decision.
4. **Catalogue association:** crosswalks and zero/one/multiple-match status
   among SDSS-PV, CF4, Tempel and the linked 2M++ count/group record, including
   unresolved associations and non-links; specify group construction and
   membership rules.
5. **Selection law and uncertainty:** exact query/cut version, inclusion or
   completeness probabilities/weights if available, and a defensible
   calibration/transfer uncertainty for the selected source population.
   State which variables and stages are not represented.

The extract must be genuinely source-matched to the selected training marks;
catalogue rows with no match are not automatically failures. Do not infer
attempts from missing joins or use the TNG coefficients as CF4 priors. Keep
the frozen held-out marks, velocities, and outcomes out of the returned
training extract and all fitting until the sampler/observation law is valid.
Documentation or aggregate parent counts may be supplied first to establish
whether an object-level extract exists.

The published paper's data-availability statement identifies an
author-supplied route for catalogue supersets, but no request is sent here.
The user's standing instruction not to send email remains in force. This
file is an internal data-requirements specification, not an email or a
sent/drafted Gmail message.

## Acceptance rule and next work

Adopt new selection information only if its population and observation stages
match the chosen CF4 cohort, likelihood-factor ownership is explicit (no
duplicated count/mark term), selection uncertainty is propagated, and the
held-out split remains protected. A defensible conditional-independence
argument must state the conditioning variables and show they are present in
the active likelihood; selected-only residuals are not that argument.

If no source-matched input or such argument is available, retain the
explicitly conditional v6 estimand and its limitation, but do not call it a
full calibrated CF4 posterior. Continue the separate field-level MC-error /
resource-contract work from
`CF4_R2_EXACT_N256_POSTERIOR_FEASIBILITY_20261104.md`; do not launch the
multi-chain target until the observation scope and acceptable computational
precision/cost are both defensible. No held-out result is read by this
decision.

**Q-GOAL:** this resolves a critical prerequisite for the requested
CF4+galaxy-conditioned z=0 density/velocity posterior; it does not itself
produce that field, uncertainty, or held-out prediction. MW/M31 remain
role-ambiguous and M33 unresolved. Their observables must ultimately
constrain those same roles on the same NEW evolved LG field at
`<=0.3 cMpc/h`; native truth identities remain calibration/evaluation only.

**Q-LEAN:** reuse completed mock, provenance, and operator evidence. No new
simulation, repeated public-parent sweep, short sampler screen, held-out
read, arbitrary selection prior, or extra gate suite is justified by the
current record.
