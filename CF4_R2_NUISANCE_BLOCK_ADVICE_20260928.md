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

Submitted on sourceeb877a6/pushed branch:408158 joint optimization
(`r2_v6_joint_secant_map_v1`),408159 afterany field readout,408160 afterok
saved-field integration/radial readout (`r2_shell_cdf_field_check_v10`).
Nine optimizer/rate/curvature/metric tests plus one particle-moment test run
before408158's main calculation. Same fresh-state adjoint and exact restart
score check remain. Only one joint follow-up is submitted, not a retry chain.

408158 FAILED35s on syn104 before main GPU execution: cuInit(0) returned
CUDA_ERROR_NOT_INITIALIZED. The9 pure optimizer tests passed; the particle
test fell back to CPU, which is NOT GPU validation. The main GPU requirement
correctly stopped inference; no objective/field/result was produced.408159
then failed8s because no source report existed.408160 was explicitly cancelled
while pending DependencyNeverSatisfied. Logs are retained; no data deleted,
no GPU reset/manual run, and no cause beyond CUDA initialization is asserted.

Same source/target retry408163 uses typedH100,2CPU20GiB/75min and the SAME
32-update/60min application bound, output `r2_v6_joint_secant_map_v2`. This
replaces a failed pre-inference allocation, not another completed science run.
Dependent readout408164 uses H1001CPU2GiB/5min; saved-field checker408165 uses
H1002CPU6GiB/15min (`r2_shell_cdf_field_check_v11`). The latter gets5min more
Slurm time for the different GPU/cache, not a new diagnostic or fit; request
caps now95min for the retry bundle. Reconfirm GPU and target startup before
claiming that inference resumed. Source remains eb877a6.

408163 FAILED6m30s on H100/syn08 after one full evaluation and zero optimizer
updates. Nine optimizer tests and the particle test passed with GPU available.
The exact restart guard detected objective149990.1429854582 instead of
149990.00403363275. Count and nuisance-prior terms match to rounding; the
entire0.13895182545 discrepancy is in the FP term (-19.0843320053 instead of
-18.9453801798).408164 readout completed;408165 pending DependencyNeverSatisfied
was cancelled. No tolerance was relaxed, no target/prior changed, no data deleted.
The cause is not yet established; PM uses float64, and count agreement does
not support casually attributing this to a float32 PM calculation.

Bounded diagnostic408178,source2e606db,H1002CPU8GiB/15min, compares the saved
H200 terminal and H100 initial arrays and scores their429 training FP rows.
No PM evolution, optimization, count-integration sweep, heldout or new data.
Q-GOAL: restore a reproducible actual-distance target before joint inference.
Q-LEAN: isolate the differing factor using existing saved states. MW/M31/M33
identification remains pending on this same NEW field, with no truth IDs used.

408178 completed1m40s. The IC/tracer/FP-zero arrays are bit-identical; density
and velocity differences are at most4.83e-13 and2.29e-11km/s. On H100 both
saved states reproduce FP=-18.94538017978733 in value-only and per-row code.
Thus changed scientific inputs or PM precision are not the explanation.

408180/source5624b74 completed13m52s,MaxRSS6.10GiB. Its nested-JIT isolated
FP value-and-gradient returns-19.084332005253486, reproducing the failure
without any count integration. The flattened full count/FP value-and-gradient
returns the correct-18.945380179787325 and reference nuisance gradients.
The submission used CF4_EXPECTED_COMMIT=HEAD; the runner resolved its source
check while HEAD was5624b74. This recording weakness is not target evidence;
subsequent submissions again pin the explicit hash. No persistent compilation
cache was used by this diagnostic, so a stale persistent cache is not necessary
for the reproduced mismatch.

408184/source2e47980 completed3m57s,MaxRSS2.62GiB. Compact per-observation
candidate batches reproduce all six FP population scores; primal values from
value-and-gradient agree with direct values to8.89e-16 per row. Local compact
support uses every active neighbor, not a smaller physical support or floor.
408185/source1286278 compares the exact same isolated function with autodiff
BEFORE the sole outer JIT, to test the proposed removal of redundant nested
JITs. H1002CPU12GiB/15min; no PM, optimization, heldout or science-model change.
This is an observed compiled-execution discrepancy, not yet an identified
upstream compiler defect. Do not infer that all nested JITs or H100s are wrong.

408185 completed7m38s,MaxRSS4.84GiB. The SAME isolated FP function and inputs,
with autodiff before the sole outer JIT, return-18.945380179787325 and
zero-gradient5.000946304146577, matching the reference and compact calculation.
The nested version returned-19.084332005253486 and4.998954868403673. This
establishes a concrete execution-path workaround, not the upstream defective
compiler pass. No scientific model, GPU precision, padding support or tolerance
was changed. Old accepted H200 steps had explicit full/score-only agreement;
this does not retrospectively show that they were invalid.

Driver applies the flat value/gradient compilation to the full and conditional
targets. Separate value-only JITs remain for line searches; the mathematical
functions are unchanged. At every fit startup compare direct and derivative
primal components on the SAME cached field/support before optimization, without
an extra PM evolution. Preserve the exact endpoint restart guard, whole-adjoint
finite-difference check and all accepted-trial score checks. The existing small
nonempty target regression now checks derivative primal components as well;
actual-size startup checking is needed because the small fixtures alone did
not reproduce this path-dependent error. No arbitrary likelihood tolerance is
enlarged. Resume only the already-assessed32-update/60min joint follow-up.
