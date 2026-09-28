# R2: completed joint follow-up and actual distance readout

R1 -> **R2 incomplete** -> R3 same-state LG -> R4 precision evolution -> R5 zoom IC.
This is a development MAP at N128/384 (3 cMpc/h), not the requested surrounding
1–2 cMpc/h posterior, and not an LG <=0.3 cMpc/h reconstruction.

## Completed evidence

| Slurm job | Result | Scientific interpretation |
|---|---|---|
|408188 / source3f04209|59m05s,32 accepted joint updates; all11 prerequisites passed|Iteration limit, NOT convergence|
|408190|6s, saved density/velocity comparison|Actual NEW field readout, not named-structure validation|
|408191|7m25s, final count integration/readout|Numerical count agreement, not calibration|
|408216 / source254b25a|2m30s,429 FP training-row predictions|Training plug-in readout, NOT heldout/PPC|

408188 objective149990.00403363275 ->146824.16396973614. Count log score
-144315.2348776422 ->-141770.41500053153 (gain2544.82); FP score
-18.94538017978734 ->-19.03477667725891 (slightly worse). About624.08 of the
total3165.84 improvement comes from shrinking the IC prior penalty, while the
nuisance penalty grows. Therefore total objective improvement does not mean
that CF4 distances improved. IC gradient L2 ends857.20 (started683.98), maximum
coordinate6.934; no stationarity or posterior-covariance claim.

Final tracer LOS width38.55 km/s; physical mass-weighted velocity dispersions
[67.55,67.06,73.17] km/s are distinct quantities. White-IC mean square.004785
is a MAP property, NOT an LCDM posterior realization. Do not restore power by
hand. The field has visible structure but MW/M31 remain unidentified/ambiguous
and M33 unresolved. Their later observables must constrain this SAME NEW field;
native truth IDs may not seed or select candidates.

408191 reproduces the count score exactly. CDF4x32 versus8x32 score difference
-0.00033708; exposure relativeL1=2.17e-7; scalar velocity derivative relative
difference7.65e-9. Total expected47006.39 vs observed47121 masks radial shape
residuals: predicted/observed6252.18/5530 at84–96,1445.66/1801 at168–180 cMpc/h.

408216 reproduces FP=-19.03477667725893 with the per-row sum. Shared zero is
5.0966 prior SD=+.0203865dex (prior SD.004); down from5.3016, not up. Predicted
eta before zero has mean-.0279605 and SD.0153028; source means have mean
.0176132 and SD.0812815. Mean source reported uncertainty.0951991dex. The
observed/predicted correlation is-.0775964, not demonstrated positive distance
agreement. Mean residual is.0455737 before zero and.0251872 after; RMS.0954508
and.0875687. This alone does NOT establish model failure: the field is not
stationary, measurements are noisy, and these are source PDF moments with
shared calibration, not an independently calibrated Gaussian sampling law.

## Readout figures (real results, not illustrative simulations)

- [Field comparison](/gpfs/kjhan/CF4/z0_density/r2_v6_joint_secant_map_v3/readout/field_comparison.png)
- [Training radial counts](/gpfs/kjhan/CF4/z0_density/r2_shell_cdf_field_check_v12/training_radial_counts.png)
- [Training FP distances and residuals](/gpfs/kjhan/CF4/z0_density/r2_v6_joint_fp_readout_v2/training_distance_prediction.png)

Every test included in a later nonexpert Korean PDF must have an example
figure. These readouts are available for that report; no PDF is claimed here.

## Narrow input checks and limits

The catalogue builder `scripts/cf4_r2_common_catalogue.py` uses observed
`Vcmb`; `src/cf4_twompp_disjoint_tracer_pilot.py:226` computes comoving distance
from Vcmb/c then multiplies by h. No Carrick flow-corrected distance is inserted
by that builder. A hypothesized double-RSD input error is NOT established.

[Howlett et al.2022, sec.5, eq.20](https://arxiv.org/html/2201.03112) describes
mean/std/shape to skew-normal location/scale conversion, matching
`src/cf4_r2_fp_distance.py`. It describes a flat eta prior and already-included
selection correction; skewness is small. Thus neither an extra Jacobian nor
another selection factor nor an unverified skewness explanation is justified.
The web PDF parser failed, but a streamed74.7kB PDF read with Ghostscript
subsequently confirmed the exact catalogue columns: logdist_corr and its error
are the mean and standard deviation, NOT location and scale; alpha is the
skew-normal shape. The corrected columns come from richness-specific FP fits.
The catalogue explicitly cautions that zero calibration is group-level, not
independent member-level. This supports retaining the grouped-row exclusion
until its shared model is implemented, not dropping group data forever.
Source: [release column description](https://zenodo.org/records/6824749/files/data_description.pdf).

Current fit uses47121 counts and429 strict ungrouped linked FP rows only;
985 grouped links and other CF4 methods are not in this live likelihood.
Selection/bias/association/shared covariance remain incomplete. The zero prior
is relative to CF3, not a full absolute-scale calibration. Heldout untouched.

## Workflow correction

408215/sourcec5a06ad was cancelled after39s. A static syntax check had failed,
but the driver's semicolon-separated shell continued to commit/submit anyway.
Fix254b25a passed static checks; fresh408216 completed. Preserve the partialv1
output, do not use it scientifically. Use `set -e` for validation followed by
commit/submission. No accepted fit was altered or deleted; no filesystem
mechanism is needed to address this driver error.

## Next decision

No automatic identical MAP extension or large posterior run. Seek one focused
science/cost consultation about the minimum substantive action that measures
and improves CF4 leverage, distinguishing noise, calibration, optimization and
model adequacy. The goal/order, unchanged likelihood weights/priors, training
split, no new TNG, and same-NEW-field MW/M31/M33 requirement remain in force.
Q-GOAL: actual CF4-conditioned current-state posterior. Q-LEAN: no diagnostic
ladder or arbitrary correlation threshold; use saved state where possible.

## Fable advice and independent driver disposition

Read-only Fable5 returned CONDITIONAL PROCEED after inspecting this record,
the linked-singleton kernel and saved-field readout. It recommended one
prior-consistent FP zero-response comparison, followed by985 grouped links
and a bounded joint rerun, before sampling; proposed combined cap4GPU-hours.
No files were edited or jobs launched by the advisor.

Adopt: no automatic identical MAP loop; compare current/pre-joint/homogeneous
benchmark using ONE shared zero with the SAME prior; broader group-aware data
is a substantive next implementation, not an optional cosmetic readout.

Amend/reject the following unsupported inferences:

- FP score movement is NOT its information fraction, and it does not prove
  that FP was ignored. The numerical "99.99%" posterior interpretation is invalid.
- Pearson significance1/sqrt(N), an asserted attainable-correlation cap, and
  ~11 effective constraints need assumptions not established for these
  correlated, selected, fitted observations; retain them only as unverified
  heuristics, not quantitative conclusions or thresholds.
- `load_train_singletons` already selects ONE FP row and ONE point per source
  group. The985 excluded Tempel-grouped links are NOT985 multi-FP systems whose
  measurement precision automatically improves by averaging. Their shared
  redshift/member law needs actual source evidence.
- Matching the source PDF formula does not calibrate tracer selection/bias,
  shared FP errors or association. Existing calibration limits do not vanish.
- No independent-row bootstrap: shared fit/systematics and sky correlations
  make it an unjustified significance scale. A plug-in score difference cannot
  establish that anti-correlation is merely noise or certify model adequacy.
- Do not assume a group-aware live likelihood is a trivial addition, nor
  submit a fit of it before its shared-data ownership is implemented correctly.

Driver's immediate bounded action: reuse saved current and pre-joint states,
plus a homogeneous zero-flow BENCHMARK (not another PM state). Keep each saved
tracer fixed; the benchmark uses final tracer. Extract exact count-conditioned
candidate eta weights once, refit only shared zero against the unchanged
N(0,.004dex^2) prior, and plot the conditional score curves. No PM, count fit,
heldout, field gradient or bootstrap is needed to answer this narrower question.
One H100 job,2CPU8GiB/30min; the previous readout's~2.2GiB RSS plus candidate
cache leaves margin. Unit regression verifies cached-mixture reconstruction
and that FP/zero does not contaminate its conditioning weights. Stop the
diagnostic after this comparison; it is not the substantive R2 endpoint.

Then connect grouped-source semantics to the existing group kernels, rather
than adding independent marks or importing old fitted nuisance priors. If
essential covariance/membership information is unavailable, state precisely
what sensitivity approximation is possible and what remains uncalibrated;
do not label an unspecified shared covariance physical calibration. MW/M31
ambiguity and unresolved M33 remain explicit same-NEW-field R3 obligations.

408217/sourcea76ee35 stopped in saved-reference parsing, after reproducing the
current and pre-joint FP scores. The conditional optimizer writes its terminal
full score under full_gradient_after_block, not the joint trace (empty here).
Driver fixes the reader and tests BOTH report shapes, resolving endpoint
references before GPU scoring. This is not a likelihood failure. Current-state
partial result is retained but the three-way comparison is not yet complete.
No checkpoint or target changed. One corrected run replaces this failed
readout; do not introduce a general provenance framework.

## Completed FP comparison and broader single-mark implementation

Corrected408218/source0fbd783 COMPLETED2m55s,2.55GiB batch RSS; two regressions
passed. No PM evolution. With the same shared-zero Gaussian prior, conditional
penalized FP scores are current=-32.02186,pre-joint=-32.97307,homogeneous=3.96855.
Current improves.95121 relative to pre-joint but trails the homogeneous
benchmark by35.99041. Best zeros are.020500,.020520,.008462dex respectively.
This establishes a descriptive distance-score deficit, not its statistical
significance, a Bayes factor, full-target preference or cause. Keep calibration
and conditioning-model uncertainty active; do not say the negative correlation
was "only noise". Figure:
[shared-zero comparison](/gpfs/kjhan/CF4/z0_density/r2_fp_zero_response_v2/zero_response.png).

Focused follow-up Fable advice agrees the blanket grouped-galaxy exclusion is
not mathematically required for the selected ONE-FP-per-source-group marginal.
Its fixed d(z_group) reference cancels from the field-dependent distance
residual, rather than acting as another independently scored velocity datum.
The exact likelihood representation is supported by Howlett sec2.2–2.3.
Retain one secure point, one observed FP, zero anchors and graph-closed training
roles. This does not justify independent factors for multiple observed members.

Driver corrections to that advice: (1) re-referencing eta AND its source mean
does not leave a fixed-reference-zero log ratio unchanged; it changes a
data-only constant. BETWEEN-FIELD log-score differences are invariant and are
what the regression tests. (2) Missing group-shared physics is NOT guaranteed
to be already included in the published marginal error just because only one
row is scored. (3) Current LOSwidth is38.55km/s, not the100km/s initialization;
environment-dependent virial errors remain a risk in the conditioning model.

Implement a separately labelled1414-row CONDITIONAL WORKING target (429
unchanged+985 physically grouped, each still only ONE scored FP). No new
independent group-redshift factor, group-average measurement, inferred host
identity or invented error reduction. Counts and marks retain the SAME global
LOS parameter, source priors and weighting. Source-fit covariance, association,
selection/bias and group-environment LOS remain uncalibrated; not production.
First reuse the saved-state comparison on that cohort, prove the original429
scores are unchanged, and record cohort-specific source-reference offsets and
row scores. Then assess one bounded joint fit, not a hierarchy or test ladder.
Q-GOAL: broaden actual CF4 distance conditioning. Q-LEAN: selected single-mark
marginal avoids unnecessary full-group construction without claiming to solve
its missing physics. MW/M31 ambiguous,M33 unresolved; same NEW field remains
the later LG-observable target. Heldout untouched.

408221/source36fd5f9 completed the1414-row saved-field comparison and all3
regressions. The original429 scores reproduce. Current/pre-joint/homogeneous
conditional zero-penalized scores=-44.93093/-43.94968/18.85248; current trails
the benchmark by63.78342. Current conditional best zero.0329084dex=8.227prior
SD, versus benchmark.0141368dex=3.534SD. More data exposes rather than removes
the calibration/flow tension. These are fixed-state plug-in comparisons, not
evidence ratios or reasons to widen priors.

For grouped985, |log10(d_group/d_point)|/reportedFPstd has median.0325,p90.1056,
max.3796 (ungrouped median.0030,p90.0164,max.0892). Thus a gross, many-errorbar
group-reference/point-distance mismatch is not present in this selected cohort;
this does NOT validate virial or shared-source errors. No independent group
average was added. Saved field FP total at ORIGINAL zero=-49.306358986928174.

Driver continues with ONE broadened joint fit, not a repeat of the429 target:
same acceptedIC/tracer/zero warm start, fresh L-BFGS,32updates/60min application,
H1002CPU24GiB/75min Slurm. Host request allows roughly3x previous6.1GiB plus20%
for larger linked arrays/derivatives. Existing count/FP/particle/optimizer tests
and actual startup primal/adjoint checks remain. Initial objective must equal
old objective +oldFP-newFP; counts and priors must independently reproduce.
This is not same-target continuation and no old metric is imported. Afterward
reuse field, count and1414-row distance readouts. No additional unchanged
optimization cycles are authorized by this local decision.

Submitted408222/source9ebd06b with the above32step/60min bound.408223 is the
afterany field readout;408224 the afterok count check (H1002CPU6GiB/15min), and
408225 the afterok1414-row distance readout (H1002CPU12GiB/20min). The existing
full-source comparison407082 already covered these1414 labels on a saved
state; no repeated2million-source reference sweep is needed. Submission is
not successful startup or completion. Await the actual numerical evidence.

For the next scientific decision, keep model error distinct from source-formula
correctness. [Jasche & Lavaux2019, sec3.4–3.5](https://arxiv.org/html/1806.11117)
use a more flexible nonlinear bias and explicitly discuss likelihood model
error in 2M++ inference, including a tempered working posterior. That is
precedent for taking model discrepancy seriously, NOT permission to copy their
temperature or bias-fit values into this different linked CF4 target. No
tempering, prior widening, noise floor or bias-model change has been made here.
