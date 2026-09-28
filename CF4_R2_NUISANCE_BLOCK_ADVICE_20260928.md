# R2 nuisance-block advice and driver disposition

Request: `config/cf4_r2_nuisance_block_science_advice_20260928.md`.
Fable5 returned **CONDITIONAL PROCEED**, using the bounded read-only
`claude-fable-5` invocation. It inspected the previous advice, recent fitted-
tangent record, R2/R3 plan sections and selected optimizer/target-construction
lines. It did NOT inspect likelihood internals/catalogue transforms. This is
science/plan advice with limited optimizer inspection, not a likelihood audit.

## What the advice said

- A fixed-field, all-ten-nuisance optimization is proportionate if followed
  by the actual full IC+nuisance gradient and never used as a calibrated prior.
- Add a prior-mean versus fitted FP-zero comparison, a10x10 nuisance Hessian,
  and before/after radial-count profiles within the same allocation.
- It proposed gain~100, zero~3-prior-SD and radial-shape thresholds to choose
  another bounded joint optimization versus calibration replanning.
- It required fixing/testing the n_ic=0 optimizer path, full-target baseline
  agreement, and explicit conditional/partial labels; no LG identification,
  posterior covariance, heldout consumption or R2 promotion.

## Driver decisions and corrections

1. **Adopt** one conditional nuisance block, baseline value/gradient agreement,
   full-field gradient afterward, unchanged priors/weights/domain and fresh
   LOS-width support. Preserve all nuisance freedom for future joint inference.
   This is an optimization update at a fixed field, not calibration/posterior.
2. **Adopt with narrower interpretation** an FP-zero sensitivity comparison
   at the NEW fixed field, with all other parameters unchanged. Comparing
   fitted zero with prior mean isolates that final-state zero effect. Two
   evaluations do NOT decompose the v2->v3 FP-score change into independent
   field/zero causes: the field, other nuisances and their interactions changed.
3. **Adopt** the same training radial-count readout at block start/end, computed
   from the intensity already needed by the target, with no extra PM run.
4. **Defer the10x10 Hessian.** The statement that nuisance gradients cost
   "seconds" is unmeasured. Removing PM/field VJPs does not remove expensive
   luminosity/LOS integration and its nuisance derivatives. Measure cost first;
   do not append21 evaluations or a new identifiability gate to a20min budget.
   Near-zero curvature is also coordinate- and prior-dependent, not a complete
   identifiability result. Three ongoing directional checks are not SPD proof.
5. **Reject a conditional residual as proof of model impossibility.** An
   unconverged/wrong fixed field can retain radial residuals and an FP offset
   even if another jointly varied field fits. A nonconvex conditional solve
   cannot prove that *no* calibration setting works. Large gain/full IC-gradient
   changes motivate a conditioning/coupling interpretation, but do not uniquely
   identify it. Neither result alone establishes catalogue/selection failure.
6. **Reject automatic scientific thresholds~100 or3 SD** and the instruction
   to freeze a nonstationary state "as the partial-target MAP." Those thresholds
   are not calibrated decision rules. Record actual gain, gradients, residuals,
   nuisance displacement and stop reason. Retain an optimizer state as such,
   not a mode/posterior, and do not attach automatic alternating cycles.
7. **Correct terminology/cause.**1.12825 is a bias VALUE, not a physical bias
   gradient.4.09 prior SD is a tension, not a measured posterior significance;
   stochastic variation and remaining field nonconvergence also remain possible.
   The .004 prior itself is a development choice, not an externally certified
   scale for this partial target. Do not widen it to improve the image.
8. **Fix/test the empty IC block without overstating the failure.**0/0 does
   produce an invalid term, but Python's ordered max(1.,nan,...) does not always
   propagate that NaN. Silent corruption is not established. Explicitly omit
   the absent IC term and test conditional descent and a still-nonzero full
   gradient on a coupled quadratic.

## Bounded next action and decision (declared before execution)

After the existing curvature assessment, permit ONE conditional block:
up to20 accepted iterations/20min application,30min Slurm,H2002CPU12GiB
(host estimate<=10GiB with20% headroom). This is a new bounded optimization
method/decision, not an automatic extension of the previous175min envelope.
Keep enough application time for the mandatory full-gradient endpoint.
First reproduce the saved full target and nuisance derivatives and measure
nuisance-only derivative time. Stop on inconsistency or nonfinite derivative;
true zero-probability trial states may be rejected without a likelihood floor.
If the cost prevents completion, preserve the accepted point and report an
unfinished conditional fit rather than infer model failure. No sweep/extra
Hessian/indefinite repeat is authorized by this plan.

The next decision is whether actual nuisance equilibration/coupling supports
a redesigned joint inference step, or whether the evidence instead requires
observation-model/source-calibration work. A low gain with a significant
remaining conditional gradient is optimizer nonconvergence, not a physical
rejection. A small conditional gradient with a large full IC gradient is not
joint stationarity. Conditional radial residuals do not certify model failure.
Do not compute a Laplace posterior before a usable stationary state, or call
any plug-in heldout score a PPC. Heldout remains reserved until its target,
fitted state and uncertainty/prediction method are frozen.

Q-GOAL: reduce an evidenced obstacle to actual z=0 inference, not more surrogate
tests. Q-LEAN: reuse the target, saved field, optimizer and one allocation; no
new simulator, data download, broad test framework or costly Hessian. MW/M31
remain ambiguous and M33 unresolved; this block does not identify them. Their
later observables and generated candidates must constrain the SAME new field,
without native truth IDs. R2 is still incomplete, N256 is not authorized by
this conditional development result, and no email is sent.
