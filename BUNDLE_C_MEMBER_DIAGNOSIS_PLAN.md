# Frozen member-readout diagnosis and observation-aware redesign

2026-09-12. User: proceed from the cause investigation. ONE evaluation-only
Slurm job, then a concrete next-learning proposal; no optimizer updates or
automatic next training bundle. Final route remains actual CF4+galaxy+LG ->
z=0 density/velocity posterior -> LCDM IC -> forward LG zoom, LG<=.3 and
environment1–2 cMpc/h. This is not a new field-prior attempt.

## Evidence and question

342129 completed13000 new updates,1000 per field, but fails member-readout
criteria. Restored fixture0 reproduced its5000-step success before learning;
afterward its L1 becomes2.816/.957/1.066. Across identical field/symmetry
combinations98.2% improve loss, first/last means4.714/1.733. Thus not simply
no learning. Final development20 M33 mass ratio.9474 but overlap.04453:
allocation/localization, not merely amplitude. Existing native partition,
restore and48 physical-feature symmetry tests pass.

Source selection uniformly chooses an eligible M31, then an eligible M33,
for a selected MW; inputs contain total moments and a coarse observer only.
This leaves an unobserved assignment choice. It is NOT proof that fixed13
training pairs are contradictory or impossible to memorize, or that this
alone explains the optimization failure. Distinguish this population-level
task issue from network transformation sensitivity and finite training.

## Single bounded calculation

1. Reuse native load_case, features, transform, metrics and strict FP32.
   Freeze checkpoint342086/source787497f (5000) and342129/source20253e1
   (18000). Require source/result/update match. No optimizer construction.
2. Fixed fixtures[0,10,15,16,17,20]: pretrained field, two overlapping
   training fields sharing M31 but selecting different M33, and ALL three
   previous development fields. This is diagnostic coverage, not independent
   validation or selection of a new model. Same six fields for both models.
3. Evaluate ALL48 signed permutations for each checkpoint/field:576 forward
   passes. Transform physical/observer features together; inverse-transform
   predicted fractional allocations before comparison to the unchanged native
   masses. Identity first. Reproduce saved identity metrics wherever available.
   Record per-role L1, mass ratio, overlap, centroid and conservation, plus
   prediction discrepancy from the same model's identity map normalized by
   native role mass. Report every orientation and uniform mean/min/max; do
   not choose a best direction/checkpoint or use an ensemble as a new model.
   A tiny inverse-transform roundtrip test supplements existing regressions.
4. In the SAME job, read the small existing population identity/branch/split
   tables. For each of the16 existing train/development fixtures count eligible
   M31 alternatives and M33 alternatives for its chosen host under the exact
   source selection. Read32 patch records to identify any identical-input
   (same origin/coarse observer) different-label cases; absence does not prove
   identifiability. No snapshot rescan/new member profiles/candidate learner.

Outputs: one compact per-view JSON, summary and selection-context JSON, plus
driver disposition and a short next-training proposal. No new volume dumps,
new morphology metrics, monitoring daemon or generic test framework. Existing
source files/results remain unchanged. Numerical checks run inside Slurm.

## Decision and observation connection

- On fixture0, the PRE-multi model's rotated versus identity predictions
  isolate sensitivity that existed BEFORE any new update. POST-multi measures
  what remains. This does NOT isolate the causal contribution of augmentation
  versus multi-field training: doing that requires a later controlled fit.
- Identity reproduction failure is an implementation mismatch to resolve,
  not scientific evidence. Successful reproduction with orientation spread
  establishes non-equivariance, not that removing augmentation fixes learning.
- Alternative source assignments substantiate an omitted selection variable,
  not measured posterior probabilities or an empirical identifiability bound.
- Next DESIGN must not equate a random native identity with a uniquely
  field-determined MW/M31/M33 label. Use hypotheses with explicit role/observer
  uncertainty and observational selection, including shared/unresolved M33.
  On NEW fields candidates must come from available field/state, never truth
  IDs/positions or nearest-truth initialization. Native labels are calibration
  only. No permanent M33 waiver and no demand for three distinct grid peaks.
- An eventual joint target must let actual LG distance/position/relative-
  velocity/mass observations reweight the SAME field/state, with COM/stellar
  discrepancies and selection accounted for. A conditional readout alone is
  not that posterior. Do not double-count observations in proposal/target.
  Missing q_F/global environment/continuous member law remain explicit.

## Q-GOAL / Q-LEAN / feasibility

Q-GOAL: settle a specific readout failure and remove an ill-defined assignment
target before further expensive learning toward the LG-conditioned z=0 map.
Not a prerequisite claim that deterministic segmentation must be perfect
before any observation-based reconstruction. Q-LEAN: one frozen evaluation
and small-table count in one allocation, existing primitives/tests, no fits,
closed peak/proxy/budget diagnostics, new simulations, or extra audit ladder.

1 GPU/2 CPUs/6GiB host/15min, a40,a100,h100,h200 excluding syn06. Host estimate
<=5GiB+20% headroom; earlier13-case training peak4.064GiB, this job streams one
case and two small models, retaining only identity and current-view maps.
Estimated2–5min execution, not guaranteed; application cap13min, incomplete
evaluation labelled incomplete. New member_mass_diagnosis_v5 output directory.
Fable5 primary plan audit before implementation/launch, Astra backup only on
invocation failure/no usable audit. Ask Q-GOAL/Q-LEAN, feasibility, essential/
deferred, all three roles and same-field observation contribution explicitly.
Report this bundle's result and wait for user approval before new training.
