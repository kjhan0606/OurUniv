# R2 actual-field milestone: science/cost advice and driver decision

External review trigger: the first numerically usable actual-data field and
the decision whether a larger, <=3 H200 GPU-hour follow-up is justified.
This is not the reinstatement of per-stage external gates. Request:
`config/cf4_r2_first_field_science_advice_20260928.md`.

Fable5 returned **CONDITIONAL PASS**. Invocation used model
`claude-fable-5`, read-only plan mode, Read/Glob/Grep only, blocked
Edit/Write/Bash/Agent, high effort,600s bound. It reported reading the three
science documents, not the implementation files. This is science/plan advice,
NOT an independent code audit. No Astra fallback was needed or invoked.

## Faithful summary of advice (not a verbatim transcript)

- Q-GOAL: actual count/FP-conditioned descent and fitted-state numerical
  checks are meaningful R2 progress. They do not supply a stationary mode,
  uncertainty, prospective prediction, calibration or N256 resolution.
- Q-LEAN: inspect the already-saved objective gains, gradients and FP/count
  components first. If408084 is still descending, allow one bounded extension
  within3 GPU-hours; if it stagnates like407886, freeze rather than extend.
  Do not add kernel-tuning sweeps, new mock streams or a diagnostic ladder.
- It proposes at most three substantive bundles: the conditional extension,
  a single frozen-state heldout comparison with prior prediction, and a small
  HVP cost/sign feasibility check rather than a full Laplace calculation.
- Count improvement with worse FP than the near-uniform start is a tradeoff,
  not by itself proof of a defective model. The partial recovery during408084
  is relevant. Repeated FP worsening warrants model investigation.
- A nonstationary MAP or the optimizer's limited-memory inverse Hessian is
  not calibrated posterior uncertainty. Physical particle dispersion remains
  a separate valid readout. Keep the old ESS2–5 HMC line closed.
- MW/M31/M33 remain unidentified in the coarse new state; especially M33
  requires later substructure/boundness work. Their observables must constrain
  that SAME state; truth IDs cannot seed or select generated objects.

## Driver assessment: adopt, amend, defer

1. **Adopt** the bounded single-extension decision and component-wise readout.
   At26 additional accepted steps408084 has objective161252.61 versus175535.45
   at restart, FP−45.05 versus−50.94, and gradient_inf34.89 versus129.91.
   Recent gains remain hundreds, not the1e-6 stagnation of407886. Final results
   still decide; these interim values are not a convergence claim.
2. **Amend** any quasi-stationarity criterion based only on a hundred-fold
   drop in max gradient or tiny objective change. Numerical parameter scaling
   affects that gradient and failed line searches can make tiny steps without
   reaching a stationary solution. Preserve the actual stopping reason and
   test local curvature only as a feasibility calculation, not certification.
3. **Defer heldout consumption.** The accepted disposition orders uncertainty
   assessment before its prospective predictive evaluation. A nonstationary
   plug-in MAP score is not a posterior predictive score. Freeze the target,
   fitted state and uncertainty/prediction method before the reserved one-time
   readout; do not inspect it to tune subsequent MAP iterations. If only a
   plug-in diagnostic becomes feasible, explicitly decide/label that changed
   deliverable rather than silently treating it as the promised validation.
4. **Amend** the proposed many-direction HVP check to only the directions and
   numerical step-size comparisons necessary for a first cost/sign estimate.
   PMWD's custom VJP does not automatically provide a usable forward JVP or
   second derivative. Verify support before promising exact Hessian actions;
   finite differences of gradients, if used, must be labelled approximations.
   A few positive directional curvatures cannot prove positive definiteness.
5. **Qualify** the FP interpretation: monotonic worsening alone does not
   identify FoG or covariance as its cause. Bias, selection, association,
   dynamics and a normal joint-likelihood tradeoff remain alternatives.
   Also6,946 is the number of heldout count **keys**, not galaxies.
6. No N256, long HMC, arbitrary power rescaling, renewed TNG dependency, email
   or R2 completion follows from this advice. Shared source covariance,
   association and survey-matched count/FoG calibration remain unresolved.

The <=3 GPU-hour envelope is for the prospective next bounded work, not a
claim that R2 can be finished within that time or permission for unlimited
repeated extensions. Existing408084 and its already-queued readouts finish
first. Driver retains responsibility for accepting or rejecting advice.
