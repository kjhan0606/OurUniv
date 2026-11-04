# R2 observation-calibration decision — 2026-11-04

## Decision

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
\pi(F,\theta\mid C,Y_{A=1})\;\propto\;
\pi(F,\theta)\,p(C\mid F,\theta_c)
\prod_{i\in I_{\rm linked}}p(Y_i\mid C_i,A_i=1,T_i=1,F,\theta_p),
\]

where \(F\) is the evolved field, \(C\) is the observed 2M++ count
catalogue, \(A_i\) denotes entry into the linked CF4/Tempel/SDSS-PV mark
cohort, and \(T_i\) denotes the explicitly modeled per-object FP
measurement cuts. In code, the first factor is the single sparse Poisson
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
The cohort-selection factor can be dropped for field inference only if its
probability is constant in \(F\) after conditioning on the variables actually
held fixed in this target (equivalently, a defensible conditional-independence
argument for \(A\)). If availability depends on latent distance, velocity,
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
