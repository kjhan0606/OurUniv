# R2 discarded equilibration: mid-course review

Reviewer: existing Astra agent astra_cf4_science_audit_20261010.
Verdict: CONDITIONAL GO, not exit approval. Driver adopts supported advice.

418866 completed 45 exact evaluations, six adapted and four frozen
transitions. Fixed signed energy errors: -.502, -.590, +.475, +.807;
acceptance T,T,F,T. IC-prior NLL still increased after adaptation while
count likelihood improved. This establishes functioning corrected transport,
not equilibrium. Neither objective descent nor IC mean square forced to one
is a convergence criterion for the conditioned target. Small fundamental
mode jumps alone do not prove a faulty metric without posterior widths.

Continue terminal418866 with unchanged target, mass6000, nuisance metric,
RNG and frozen epsilon .11996093316338646. Allocation12h, app11.5h,
97 exact calls including replay, at most20 independent L=4/6 proposals.
All states discarded for inference; no heldout scoring or intervals.
48GiB host has nearly3x measured chunked application peak16.43GiB.
Keep complete-trajectory budget guards; partial endpoints are not rejections.

Driver checked and implemented explicit parent/end index30/budget configuration,
discard_for_inference in report/trace/checkpoint, and actual split-kernel
continuous-versus-resumed regression. Grammar1171376 passed all six tests.
Source462524d26b3f5847d55544f3583da1e2b0f82922; submitted419084 on
h200 with gpu:H200:1, 48GiB. Scheduler confirmed RUNNING on syn104.

Next evaluation: component/nuisance/mode drift, acceptance and signed energy
errors; do not automatically rerun failures or declare convergence after20
transitions. Add a few fixed-geometry present-density/velocity summaries
using fields already computed by the oracle, without extra evolution.
Commit summaries only on acceptance, repeat retained summaries on rejection.
Do not modify the source of the current running allocation for this addition.
If drift subsides, cost distinct-start independent-stream multichain sampling;
if not, decide explicitly between affordable equilibration and evidenced repair.

Q-GOAL: same NEW conditional-v6 N256/384 field. Nominal1.5cMpc/h cells do
not prove observational resolution. Counts/tracer/RSD, mark availability,
shared calibration, numerical sensitivity, heldout prediction and UQ remain
incomplete. MW/M31 roles ambiguous, M33 unresolved; eventual LG observables
must constrain these roles in that same evolved <=.3cMpc/h field. Truth IDs
must not seed/select candidates.
Q-LEAN: compact existing target/kernel/restart/readout. No separate RAMSES/TNG
run, scalar pilot ladder or expanded gating framework. R2 remains open;
no exit email or R3 approval.
