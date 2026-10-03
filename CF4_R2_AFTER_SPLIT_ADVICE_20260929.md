# Prospective R2 continuation: Fable advice and driver disposition

R1 -> **R2 in progress** -> R3 same-NEW-field MW/M31/M33 -> R4 -> R5.
Current pilot408296 was RUNNING when advice was obtained. Fable read a prefix
of7 proposals,6 accepted; not a terminal chain. Invocation was read-only
`claude --model claude-fable-5`,600s cap,plan permissions,Read/Glob/Grep only,
Edit/Write/Bash disallowed. Request:
`config/cf4_r2_after_split_advice_20260929.md`. This was advice about prospective
larger computation and the remaining science gap, not another routine gate.

## Advice (paraphrase, not a new authorization)

Fable returned ADVISE MODIFY: complete the current pilot; test longer
trajectories at the same target/metric before a large chain; investigate
zero/covariance and selection limitations; do not yet launch production/N256.
It proposed8+8 proposals of~8 integrations,~128 gradients or~94min force cost,
~2.5h allocation. It floated12–25GPU-hour production only conditionally on
an unverified bulk mixing assumption. No such production has been submitted.

## Driver accepts/amends/rejects with reasons

- ACCEPT completing the one live pilot and considering a bounded longer-
  trajectory experiment if its endpoint remains finite. Small-step acceptance
  alone is insufficient. Longer prior rotations are a plausible efficiency
  lever, not established actual-target mixing. Inspect the terminal first;
  no automatic extension or commitment to8+8/2.5h.
- CORRECT the rotation angle: it is L*epsilon*sqrt(C(k)), NOT uniformly
  L*epsilon. At finite grid high-k C is below1; low modes are much slower.
  Mode displacement divided by its nonzero mean amplitude is NOT a mixing
  diagnostic. Need displacement relative to fluctuation scale and autocorrelation
  once enough stationary data exist. Neither16 nor8 retained trial states can
  measure a reliable gradients-per-effective-sample number. Record jump-per-
  gradient as an efficiency proxy only, separating cold-start drift.
- REJECT the unconditional prediction that the current chain must end before
  relaxation:7-step power growth cannot determine the stationary power or
  burn-in endpoint. It is evidence of current drift, not a calibrated forecast.
  Never force the white IC power to1 to manufacture stationarity/LCDM agreement.
- REJECT an angular-per-shell selection refit justified solely by count
  residuals. The density field, bias, selection and latent velocity can all
  contribute; counts alone do not identify which caused a discrepancy.
  Such a refit could erase real structure. The quoted6252/5530 and1446/1801
  residuals are the OLD429 fit, not the new1414 fit: current approx5940/5530
  and1541/1801. No ad hoc selection correction or prior widening is authorized.
- AMEND the zero-SD sweep recommendation: changing .004 to .01/.02 could be
  an explicitly labeled sensitivity, not independent calibration or a repair.
  Existing fixed-prior response curves already show tension. A new sweep is
  not automatically the most useful next science action without a supported
  error model. No target/prior change made.
- AMEND source-covariance claims: the present target does not supply common
  fitted-FP covariance beyond one zero. Independent release information or a
  justified refit can constrain that uncertainty. We have not established that
  authors are its only possible source, that it is unavailable everywhere, or
  that all relevant effects are strictly unidentifiable under every joint model.
- REJECT quietly redefining R2 as a conditional working N128 posterior with
  caveats. Such a product may be disclosed as partial, but R2 still requires
  actual global present-state samples/uncertainty, quantitative model/numerical
  errors and heldout prediction at the1–2cMpc/h environmental target. MW/M31
  ambiguous,M33 unresolved; later LG observables constrain that same state.
- ACCEPT not scaling straight toN256.8x17.94GiB~143.5GiB is a warning, not a
  full compiler memory estimate. Streaming/working-array structure must be
  addressed before any costly high-resolution chain. Field summaries also
  require measured forward/I/O cost, not an unsupported blanket 'minutes'.

Q-GOAL: retain the actual posterior/physical-state destination, without
relabelling a cold conditional trace as fulfillment. Q-LEAN: one live pilot,
its saved-result report and one subsequent evidence-based decision; no new
Hessian suite, independent gravity reruns, arbitrary selection/zero tuning,
external-data dependency or5000-step production launch follows from this advice.
