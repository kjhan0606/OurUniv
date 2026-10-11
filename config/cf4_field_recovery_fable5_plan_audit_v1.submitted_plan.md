# C — return to the present-field generator and observation target

2026-09-10. User approves DESIGN after closing member-budget338040.
This document does not authorize a numerical run. Fable5 plan review precedes
the next implementation request. C remains unfinished; D is not entered.

## Decision: one central repair, not another member diagnostic

The current model cannot yet produce an adequate fine matter field. V2's
strict-FP32/full-context numerical checks pass, but all16 generated cases fail
the frozen development morphology criteria. Decision337496 measures heldout
fine high-band power/native0.650 with true parents and0.390 in rollout; training
values0.676 and0.297. Neither numerical correctness nor a better NLL establishes
the required field distribution. See BUNDLE_C_FLOW_DECISION.md and
BUNDLE_C_FLOW_PILOT_RUN.md; no new calculation is needed to repeat this finding.

The observation-link work has meanwhile established its limits:

- Field-only identification337986: no distinct MW/M31/M33 triple in32 cases;
  M31/M33 share the nearest peak. This does not prove M33 has no information.
- Composite proxy337991: source-field host discrimination3/3, but no consistent
  incremental M33 discrimination. It is a selected-population kinematic proxy,
  not a physical member/field law or a calibrated real-data likelihood.
- Budget338040: all6 alternatives already fail host mass demands. Conditional
  M33 information is inconclusive. Necessary moment inequalities are not a
  likelihood, sufficient allocation, or a reason to refit this diagnostic.

Close these three diagnostics. Recommend ONE matched field-objective experiment
using the existing conservative conditional flow, with a terminal model-family
decision at its end. Do not expand the finite bank, vary M33 test radii, fit
another proxy, or require a successful separate M33 peak before doing this.
This advances a necessary field-generating component; it will NOT itself
deliver an actual-data LG posterior. That distinction is a scope statement,
not permission to call the central scientific goal achieved.

## The full destination and missing factors

Keep observations -> present-day posterior -> LCDM-compatible IC -> forward
validation -> phase-consistent LG zoom. Target384 cMpc/h domain, surroundings
1.5 and LG .1875 cMpc/h in a24 cMpc/h patch. These numerical cell sizes are not
claims of recovered small-scale phases or sufficient particle/force resolution.

Let B be the coarse present field, C its environment refinement, F the fine
total matter moments, O the observer condition, E the explicit LG population
selection, S the uncertain member state, and eta the shared nuisances. A target
with an unselected field prior needs, schematically,

    p(B) q_E(C|B) q_F(F|C) p(E|F,C,O) q_S(S|F,C,O,E) p(eta)
       * L_env(D_CF4,D_gal | C,F,eta) L_LG(D_LG | S,F,eta).

Here q_E denotes environment refinement, not the selection event E. An
explicit joint E-conditioned prior is an alternative, but cannot be obtained
by renaming an unselected flow. The32 selected positive fixtures do not supply
p(E|F,C,O). Actual conditioning must use observations once; the12-grid CF4
posterior is not an independent p(B) to multiply by CF4 again. Environmental
likelihood retains the survey selection, overlap/covariance and actual positions.

Three substantive pieces are not ready:

1. An adequate conditional spatial field law (next experiment).
2. Same-field joint member/selection law and observational discrepancy. Existing
   proxy/budget tools inform this design but do not replace it.
3. A normalized global1.5 environment law/inference. A local conditional model
   cannot be independently tiled across overlapping patches to supply it.

These are subsequent scientific deliverables, not three new audit frameworks.
Do not submit full384 inference or actual LG importance weighting while silently
omitting2/3. Prior adequacy, observation conditioning, and eventual dynamical
admissibility are different requirements. The TNG-calibrated present prior is
not automatically an LCDM IC posterior merely because density power looks right.

## Proposed executable bundle: controlled generative learning

Reuse the original v2 checkpoint/optimizer, training origins, retained origins,
normalization buffers, model architecture, strict FP32 and full spatial context.
Start both branches at the same checkpoint and give each exactly1000 further
Adam updates at2e-4 with identical native-record order. Control uses original
six-scale native NLL. Repair uses that NLL plus one fixed structural energy
score. This is the previously deferred objective experiment with a concrete
terminal decision, not authorization to restart the old six-stage programme.

Use two independent generated fields per scored condition. With fixed T and
native training Y, define a_i=||T(X_i)-T(Y)||, b=||T(X1)-T(X2)|| and

    ES = (a1+a2-b)/2.

T uses only: (i) log1p density in fixed local spatial blocks, (ii) existing
eight log-power bands, (iii) existing three bulk-velocity and three physical-
dispersion RMS summaries. Do not add a feature catalogue or per-run tuning.
Reuse the existing metrics' units, parent references and binning. During
implementation freeze the exact block indices and training-only scales in
one run config BEFORE the first update. Each feature group is normalized by
its training-only mean pairwise distance; a zero-scale group is unavailable,
not repaired with a heldout-dependent floor. Equal nonzero group weights.
Freeze a single structural coefficient from initial training-only NLL and ES
reference magnitudes, recording both; no heldout optimization of this weight.
Matching objective magnitudes is not a guarantee of matching gradient sizes.

Score one native-parent transition on each ordinary repair update, cycling
.75/.375/.1875. Every8th update replace this score with a complete1.5->.1875
rollout. Intermediate parents in rollouts must be generated, not native truth.
Native NLL keeps all six original scales; shared parameters must not be
misrepresented as finest-only parameters. Keep both branches' native NLL
exposure identical. Structural generated-data exposure differs deliberately.

The NumPy decoder/no-grad sampler cannot supply a pathwise ES gradient.
Do NOT merely add ES to a logged loss or use an undeclared straight-through
estimator. Proposed minimal implementation re-evaluates each detached generated
trace under the model, including ALL categorical masks and continuous terms:

    grad ES = .5*((a1-b)*grad log q(X1|C)
                  + (a2-b)*grad log q(X2|C)).

Sum over all generated nodes/scales, with generated parents held fixed in the
score-function evaluation. Fixed decoder coordinates are parameter-independent;
do not discard ancestor terms. A past-sample moving scalar baseline may reduce
noise; it must not depend on the current sample pair. This is an unbiased
estimator under the stated integrability/support assumptions, not a guarantee
of usable variance in a very high-dimensional field.

### One in-job feasibility stop, not a sequence of diagnostic jobs

Reuse existing roundtrip/precision/conservation tests. Add only a small mixed-
law analytic gradient test for the changed estimator and a bounded8-pair
training-only gradient/time check. Record finite/nonzero structural gradients,
norms and split-half direction agreement, not an invented scientific SNR pass.
Stop if the analytic test fails, gradients are nonfinite/zero, the two four-pair
mean gradients have nonpositive inner product, or timed equal-update completion
does not fit the remaining learning cap with20% time headroom. This conservative
noise stop may reject an otherwise learnable model; label estimator feasibility,
not impossibility of the science. No repeated batches until agreement improves.

If the estimator fails, terminate this trial without implementing a second
decoder, surrogate gradient or architecture within it. If the run cannot finish
both branches, call the comparison INCOMPLETE, not evidence for either branch.

## Evaluation and a real stop condition

Reuse the original16 development draws, fixed train/retained teacher-parent
and rollout comparisons, and original numerical/morphology thresholds. Record
all results at start/end; one midpoint native NLL only. Show individual density
slices, not just loss histories or the smoothed mean. Physical sigma_v remains
separate from uncertainty among field draws. No amplitude rescaling.

- Both new branches fail the original field criteria: close this objective/
  current-architecture repair line; no additional steps, seeds or loss weights.
- Control passes and repair does not clearly improve: do not adopt ES merely
  because it was proposed; retain the simpler branch as a development candidate.
- Repair passes and improves heldout one-step AND rollout structure beyond the
  equal-update control: retain it as a development candidate, not a posterior.
- Neither comparison is decisive: no scientific promotion. Report ambiguity
  and a bounded next scientific recommendation for user decision, not an
  automatic fit extension. One-box reused cases do not certify generalization.

Passing one metric is insufficient. Use existing full morphology criteria and
report all cases; do not choose a best draw. The paired check is not a formal
statistical superiority test with this small reused development sample.

If this repair line closes, the next plan must explicitly compare a different
representation/model strategy and its data cost against stopping fine-field
inference at the current capability. It must not relabel another near-identical
flow run or more M33 diagnostics as a new central milestone.

## MW / M31 / M33 and actual observation handoff

No halo identifiers are assigned to the generated fields in this bundle. Native
IDs supply calibration labels only. All3 roles remain required in the eventual
joint inference; M33 may share a density cell/peak with M31. MW/M31 candidate
ambiguity must be marginalized, and unknown/missing member support retained.
Do not set resolved_halos=True on a statistical field or paste native members.

Next observation-link work, if a field candidate becomes usable, must generate
or infer member state from the SAME new field/latent state, with uncertainties
in subcell centers, velocities and mass-proxy mapping; it may not just attach
independent marks. Positive component moments must sum to F, with overlapping
cell supports allowed and disjoint particle-member accounting. Host M200c,
stripped bound mass and aperture mass are distinct. Published-assumption masses
remain disabled until their actual likelihood semantics are specified.

The actual CF4 input has no row inside LG R2 (minimum15.8152 cMpc/h in this
selection), so CF4 supplies the environment, not direct M33 interior data.
Existing distance/LOS/PM conventions and shared MW/Solar discrepancies should
be reused once the missing joint law is ready. Budget inequalities remain
necessary diagnostics, NEVER independent multiplied likelihood terms.

The later deliverable is joint CF4/galaxy/LG-on versus LG-off field samples and
measured change in F, not merely sharper member labels. Until then, explicitly
report 'no actual high-resolution LG posterior'. This plan does not claim that
repairing q_F alone solves identification or authorizes that later inference.

## Cost and scope

One Slurm GPU job, partitions a40,a100,h100,h200, exclude syn06;2 CPUs,
host24 GiB (estimated20+20%), GPU target24 GiB (estimated20+20%),4h hard cap.
Learning/preparation ends by3h, reserving1h for original evaluations and outputs.
These are estimates/caps, not measured throughput for the new gradient method.
At most6 checkpoints, target<6 GiB output. All calculations/tests on Slurm;
no login/syn101 manual run, filesystem probes, process scans or watchdog stack.

No actual-data sampling, raw snapshot pass, new simulation, IC/PM execution,
Hong restart, finite-bank expansion, neural q_S fit or new standalone M33 job.
The present approval is for design. Finish Fable5 review, incorporate essential
corrections, commit/push, then request user approval for this one implementation
and bounded execution. No intermediate user approvals for our coding fixes.

## Plan audit requested

Q-GOAL: Is this the right immediate bottleneck toward the actual CF4/LG z=0
posterior and LG zoom, or does it merely prolong ML repair without a viable
observation path? Assess that criticism directly, not by restating the goal.
Q-LEAN: Is the one matched experiment proportionate? Cut machinery and do not
replace it with a ladder of tiny diagnostics. Assess estimator feasibility,
terminal decision, essential/deferred scope and explicit MW/M31/M33 handling.
Give GO / CONDITIONAL GO / NO-GO; if NO-GO identify a concrete smaller central
deliverable, without claiming an unavailable joint law already exists.
