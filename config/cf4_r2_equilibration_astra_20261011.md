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

## Driver check of current heldout access paths

Static inspection of conditional_map and raw_field_profile confirms the
current count path requests train_keys/train_counts and geometric heldout
voxel/exclusion masks, not heldout counts. load_train_singletons is called
with include_fp_parameters=False; it selects roles==0 using structural group
IDs and point geometry. Photometry/cut arrays are indexed by selected training
PGCs before entering the observation target. Whole-catalogue membership/IDs
and point geometry are accessed, so "heldout measurements unused" does not
mean "all heldout metadata inaccessible". This is a code-path check only;
it does not prove historical development blindness, an independent FP
heldout cohort, or absence of upstream calibration leakage. The existing
prospective heldout protocol still requires those assessments before any
predictive validation claim. No heldout measurement values were scored here.

## Completed block 419084: driver disposition

Slurm COMPLETED, exit0:0, elapsed10:22:40, batch MaxRSS19428264KiB
(18.53GiB). All20 new transitions10--29 completed:15 accepted,5 rejected.
The final transition29 was accepted, signed energy error -.1296069473.
All states remain explicitly discarded for inference. No heldout values
were scored, no posterior intervals established, and R2 is not complete.

Across this block IC-prior NLL changed8247764.898 ->8337866.903,
count log likelihood -139570.732 ->-138367.522, and conditional FP log
likelihood5982.224 ->5977.826. The late prior value decreased on the last
accepted move, but this does not erase the much larger block-wide drift.
Fundamental cosine coordinates remain near(-9.29,-4.90,-9.37); the last
accepted jumps were(-.001331,-.003351,+.004462). Thus neither equilibrium
nor adequate exploration of slow modes is demonstrated. Acceptance75%
is evidence of functioning MH correction, not posterior quality.

Next in-scope implementation: connect compact present-field summaries to
the already evaluated oracle field and preserve them on MH rejection;
extend same-target terminal restart to this exact recorded lineage. Then
evaluate a bounded, explicitly costed equilibration/slow-block proposal.
Do not relabel these states as production draws or automatically launch
long multichain production. MW/M31 ambiguity and unresolved M33 remain;
truth labels must not choose generated-field candidates. No R2 exit email.

The oracle now attaches density moments and mass-weighted bulk/spatial
velocity summaries to its already computed field, only for sampling runs.
The transport commits that summary on acceptance and repeats the retained
summary on rejection. These summaries are not within-cell dispersion or UQ.
Terminal419084 was added to the exact path/source restart allowlist; full
target/component/gradient replay and the existing metric/RNG fingerprint
checks remain required. Grammar1172571 completed exit0:0: all six transport
regressions passed, including rejection-summary preservation and real-kernel
continuous/resumed equality. This is wiring validation, not CF4 convergence.
