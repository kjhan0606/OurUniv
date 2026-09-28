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

## Implementation/submission

Source267398f submitted as408154,H2002CPU12GiB/30min,application20min,
up to20 conditional updates. Output `r2_v6_fixed_field_nuisance_v1`,restart
`r2_v6_shell_map_v3/accepted_checkpoint.npz`. Seven optimizer/rate/curvature
tests including the empty-IC coupled quadratic, plus the existing particle
moment test, precede computation in the allocation. No tests were executed
on the login node. Source commit was pushed to `agent/freeze-zoom-pipeline`.
The nuisance branch shares the full target's parameter wiring and checks
actual baseline/endpoint score and nuisance-gradient agreement. Radial
predictions reuse its intensity; FP-zero comparisons are fixed-field
sensitivities, not trajectory decompositions. Endpoint IC/nuisance L2/max
gradients and the derivative array are retained for later decisions, avoiding
an otherwise redundant PM adjoint.180s application time is reserved for the
mandatory endpoint full gradient/readout. Submission is not a test result.

## Outcome and one joint follow-up —2026-09-29

408154 COMPLETED19m13s. All7 optimizer tests and1 particle-moment test passed.
The conditional/full baseline differs by2.91e-11 in score and1.95e-14 in
nuisance gradient; the endpoint also agrees.5 accepted nuisance updates then
application-budget stop, NOT conditional convergence. Objective
151825.92350→149990.00403,gain1835.91947. Conditional max gradient42.6823→5.59751.
IC max3.65466→8.17888,L2 norm490.005→683.978; nuisance optimizer-coordinate
L2 norm48.600→8.506. The IC block already dominated the L2 norm BEFORE the
update; do not compare the blocks' max values as invariant conditioning proof.
There is recoverable nuisance suboptimality and coupling, not an identified
unique cause or a stationary joint solution.

FP score−25.5635→−18.9454 while its shared white zero4.08892→5.30160
(.0212064dex). At fixed field/nuisances, setting that final zero to0 gives
FP−56.0130,a37.0676-unit difference. The radial shape remains mismatched:
48–60 prediction4802.0 vs4417,168–180 prediction1406.1 vs1801. Both the
incomplete conditional optimization and the not-yet-equilibrated field remain
alternatives to model mismatch. No source prior is widened or recalibrated.

Fable's negligible-cost premise was NOT confirmed: first nuisance derivative
281.38s including compilation,warm derivatives75–76s,full endpoint76.66s.
Removing PM/field VJPs did not appreciably reduce wall time; the expensive
observation calculation dominates here. The full nuisance Hessian was rightly
not added. Python self peak8.96GiB,Slurm batch MaxRSS16758024KiB(~15.98GiB),
normal exit0. Without attributing the accounting difference to a cause, use
the larger observed figure for the next memory request:20GiB (>20% margin).

### Driver decision: reuse conditional secants for optimization, not covariance

The evidence supports the ONE bounded joint follow-up mentioned in Fable's
advice (substantial actual gain and increased full IC gradient), rather than
another fixed-field loop or an unchanged64-step restart. Reconstruct a small
10x10 SPD inverse L-BFGS metric from the existing accepted conditional
parameter/gradient differences. No new Hessian evaluations. Positive-curvature
secants are retained as in the existing optimizer; non-SPD output is rejected,
not made into a covariance by clipping eigenvalues. This is an optimizer
preconditioner only, not a statement that physical negative curvature is absent.

For joint L-BFGS, apply the usual scalar history scale on IC and this fixed
metric on the ten nuisances inside the initial inverse-metric action. The
joint secant updates still couple both blocks. Keep original white/scaled
coordinates, priors, support, likelihood weights, step caps and actual Armijo
acceptance. Reproduce the exact conditional endpoint score at startup and
require the metric source's accepted checkpoint. Two small regressions check
secant reconstruction/negative-pair rejection and the unchanged joint optimum
under a known block metric; existing tests precede execution in Slurm.

Bound:32 joint updates/60min application,75min Slurm,H2002CPU20GiB,plus
existing final field readout5min/2GiB and saved-field integration/radial readout
10min/6GiB. Requested caps90 GPU-minutes for this explicitly assessed follow-up;
not a silent extension of the earlier175min envelope. No automatic additional
joint/nuisance cycles after it. Inspect actual gradients, FP-zero tension,
radial shape and the field before choosing further inference/model work.
Do not promote a nonstationary endpoint or the unstable finite-gradient HVPs
to a Laplace posterior. Heldout is still not consumed.

Q-GOAL: address the measured coupled optimization obstacle within the actual
z=0 inference. Q-LEAN: ten-dimensional matrix from saved evidence, no costly
conditional-Hessian sweep, new solver, new data or routine external re-audit.
The preceding Fable advice covered one bounded joint follow-up; driver applies
the corrections above rather than importing its arbitrary thresholds. Still
N128/3cMpc/h development, not N256 or full calibrated CF4. MW/M31 ambiguous,
M33 unresolved; all future candidates/observations must concern this SAME NEW
field, no native truth IDs. R2 remains incomplete and no emails are sent.
