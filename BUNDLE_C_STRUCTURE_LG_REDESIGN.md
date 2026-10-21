# C: structure-learning repair with an explicit LG connection

2026-09-09. User accepted withholding v1/v2 adoption and approved continuing
the redesign. This document is a proposal for the next execution bundle, NOT
approval of the new paired fits. No new job was submitted during redesign.
Subsequent user approval authorizes this bundle with Fable5 plan audit first,
explicit Q-GOAL/Q-LEAN questions, Astra fallback only on audit unavailability.
Implementation/execution must incorporate the audit, not preempt its verdict.
Fable5 returned CONDITIONAL GO; see BUNDLE_C_STRUCTURE_LG_PLAN_AUDIT.md.
The driver identified a circular field-to-member assumption in the proposed
Gaussian proxy and a shared-parameter issue in the suggested gradient fallback.
Do not treat either as an implemented fix. Execution is held pending the user
decision on prioritizing a field-only LG observational connection; no new job.
The target remains CF4/count/LG→present-field posterior→IC→forward/zoom, with
surroundings1–2 and LG<=.3 cMpc/h. Do not substitute TNG image realism for it.

## Decision from the completed evidence

Keep the lossless seven-moment representation, explicit atoms, strict FP32,
full-context evaluation and normalized flow density. Do not yet discard this
model family or promote the current fitted models. Confirmed deficiencies:
single-step small-scale power loss and additional multiscale rollout loss,
including training fields. Fixed native NLL improvement alone did not cure
structure. Underoptimization versus capacity remains unresolved.

Recommend ONE controlled objective experiment, with LG observation-interface
implementation/calibration in the same execution bundle. Do not simultaneously
enlarge architecture, introduce a new simulation source or change selection.
This is a feasibility experiment, not a prediction that the repair will work.

## A. Controlled structural learning, not amplitude correction

Fork the SAME v2 weights and optimizer state into two branches. Both use strict
FP32, full context, the same training-only cases/order and1000 additional Adam
updates at the existing2e-4 rate. Control retains native conditional NLL; repair
adds a distributional structural score. Both use the same final-checkpoint
rule. Equal added update count addresses the unresolved extra-training confound;
it does not isolate scored data exposure or guarantee equal wall-clock cost.
No sequential hyperparameter/seed search if either branch disappoints.

Proposed objective: original native NLL plus energy scores on (a) one-step
samples conditioned on native parents and (b) genuine1.5→.1875 rollouts
conditioned ONLY on the native1.5 start. Never teach native fine truth under
an incompatible generated intermediate parent; it lies outside its exact
conservation surface. Score complete sampled fields in physical moment space.
Retain original six-scale NLL exposure; rotate the three LG scale scores and
full rollout score deterministically between updates. Use two independent
samples per scored condition, not the posterior mean or best sample.

For a fixed field transform T, use

    ES_T = .5*(||T(X1)-T(Y)|| + ||T(X2)-T(Y)|| - ||T(X1)-T(X2)||).

X1/X2 are independent draws from the same conditional model, Y its native
training reference. The diversity term is mandatory; omitting it encourages
a deterministic compromise. Fixed local patches of log1p density supply
spatial structure, alongside the existing spectral summaries and velocity/
directional-sigma summaries. Normalize features with training-only, frozen
statistics, not per-generated-field rescaling or heldout power ratios. Use
positive score weights fixed from training-only reference magnitudes, not
selected on diagnostic improvements; freeze exact transforms/weights before
any optimizer update. Summary scores alone do not identify a complete field
distribution, which is why full normalized NLL remains in the objective.

Statistical basis: Pacchiardi et al., JMLR25(45),2024,
https://jmlr.org/papers/v25/23-0038.html uses proper scoring-rule training and
spatial/patch scores for probabilistic forecasts. This supports the objective
principle, NOT this3D cosmological application, its compute budget or success.
The NLL-plus-structural combination here is our proposed adaptation.

Implementation risk must not be hidden: the current sampler/decode is NumPy
and no-grad, with categorical atoms and state-dependent supports. Merely adding
ES to the logged loss does NOT train it. For a first implementation use a
likelihood-ratio estimator on detached complete generated traces, including
all mask probabilities and all node/scale log densities. Decoder maps are
parameter-independent at fixed trace; compute logq with the generated parent
states held fixed, rather than backpropagating through a different conditional.
If a_i=||T(X_i)-T(Y)|| and b=||T(X1)-T(X2)||, an unbiased score-function
gradient under the usual integrability conditions is

    .5*((a1-b)*grad logq(X1|C) + (a2-b)*grad logq(X2|C)).

This avoids a false straight-through gradient or silently freezing atom
probabilities. It may have prohibitive variance in millions of dimensions.
An in-job initial timing/gradient segment must check finite nonzero contribution,
an analytically tractable small mixed-law gradient control, and repeated
fixed-condition gradient noise. If not useful within the budget, STOP and
report the estimator limitation; do not launch a6000/60000-step hopeful run.
Do not claim a gradient check proves optimization feasibility. A differentiable
decoder/variance-reduced estimator is a separate redesign if needed, not an
automatic extra implementation series in this bundle.

## B. LG observation link: mandatory contract, bounded calibration

Reuse native `spatial_calibration_v1/spatial_components.h5` (93 member profiles,
32 total/remainder patches), the observational contract and COM/proxy source
products. No raw particle pass. Implement a statistical member-state readout
with three disjoint member components S_MW,S_M31,S_M33 plus remainder:

    F = S_MW + S_M31 + S_M33 + S_other  (all seven extensive moments).

Member mass/COM velocity derive from S. Position requires continuous subcell
location information or explicitly calibrated grid-COM discrepancy; never
claim a .1875 grid resolves every satellite/halo internally. Inclusive M200c
is NOT disjoint bound mass. Stellar-versus-member velocity offsets, Solar
nuisances, distance/LOS/PM covariance and mass proxies remain explicit. The
M33 satellite alternative must remain represented; do not drop it for a nicer
pair. Overlapping source patches are not independent calibration samples.

The general distribution required is normalized q_S(S,u|F,C), where u includes
any continuous center/proxy/nuisance variables. Three binary component splits
can reuse conservative moment coordinates (not spatial restriction); conditional
mask laws and any center prior must be normalized and properly conditioned.
Writing this factorization or fitting profiles alone does NOT supply q_S.
This bundle implements/calibrates the readout with native component states
and records its discrepancies; full q_S fitting/sampling is explicitly NOT
promised within the same4h field-objective experiment.

The current `cf4_lg_observation_contract.predict` requires resolved_halos.
Do not set that flag true on a statistical field or remove the native safety
guard. A separately labelled statistical-state adapter must expose its limited
semantics and feed the existing observable-frame/covariance calculations.
An LG likelihood attached to independently sampled marks cannot constrain F;
only a normalized joint S/F law makes that connection. The readout calibration
is a necessary interface deliverable, not an LG-conditioned map.

## C. Target and adoption conditions

    p(B) q_E(C|B) q_F(F|C) q_S(S,u|F,C) p(eta)
      * L_CF4(C,F,eta) L_counts(C,F,eta) L_LG(H(S,u,F),eta).

Use actual observation likelihoods once; the saved12-grid posterior is not a
new independent prior. Environment1.5/global384 normalized law and q_S remain
missing. A local24-cube likelihood cannot be tiled as overlapping independent
probabilities. All present results remain native-coarse conditional development.
LG posterior release requires those missing pieces, not just a passing P(k).

After paired fits, reuse all16 original evaluation draws plus fixed train
diagnostics, teacher/rollout comparisons and fixed native validation NLL at
start/mid/end (three checkpoints, not frequent monitoring). Original physical
and morphology gates remain; no relaxed power/boundary/connectivity criteria.
Require improvement beyond the equal-update control in both one-step and
rollout structure, report every metric and preserve uncertainty/diversity.
Numerical/morphological pass is only candidate-development permission; statistical
promotion needs independent calibration and LG/global integration.

Proposed compute envelope: one Slurm GPU,2 CPUs,24 GiB host (20+20%),4h total
including both branches, existing four regressions, focused score-gradient
control and readout calibration. Preparation/learning stops at3h to reserve
evaluation; if both branches cannot complete equal updates, comparison is
INCOMPLETE, not repair superiority. No new simulation, IC/PM run or Hong restart.
Cost/variance are unmeasured until the in-job segment; the limit is not a
runtime forecast. Readout-only outputs are small; retain at most6 model
checkpoints and final evaluation fields, target<6 GiB artifacts.

## Scope boundary

This redesign closes the prior1–4 analysis and proposes a specific falsifiable
repair and honest LG interface. It does NOT solve q_S, global inference or
underoptimization/capacity by declaration. Next execution needs user approval
of BOTH matched branches and the stated feasibility stop. If scope is approved,
complete exact training-only feature/weight definitions before submitting;
do not treat this design as an already running fit.
