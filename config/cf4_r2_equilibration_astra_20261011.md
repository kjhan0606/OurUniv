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

## Bounded longer-trajectory comparison: provisional driver plan

Astra inspected the actual trace and original metric rationale. Its interim
advice does not diagnose mass6000 as faulty from small fundamental jumps.
For419084 L6 accepted5/7 and L4 accepted10/13. Rejection-inclusive total
IC squared displacement per force was about.0408 versus.0298. This is a
whole-IC transport proxy, not low-k ESS or a proven stationary comparison.
It supports one bounded longer-trajectory experiment rather than another
scalar curvature ladder or unbounded identical repeats.

Continue terminal419084 with same target, epsilon, metric and RNG, but
predeclare an independent state-independent uniform mixture of L8/L12.
All resulting states remain discarded. At most10 new proposals,97 exact
calls including replay, app11.5h/Slurm12h,48GiB host (measured18.53GiB).
Complete-trajectory time/call guards may stop before10 proposals. Same-field
summaries use existing evaluations; no extra dynamics. Native truth roles
never initialize/select candidates; MW/M31 ambiguous and M33 unresolved.
This remains R2 conditional development, not LG reconstruction or exit.

Implementation exposes CF4_HMC_LENGTHS and records the fixed schedule;
lengths must be distinct integers1..12. Grammar1172573 completed exit0:0,
all eight transport tests passed, including long schedules, invalid input,
rejected summaries, budget guards and actual continuous/resumed equality.
The full CF4 checkpoint replay still has to pass in the next allocation.
Final Astra advice and driver disposition remain pending before submission.

Final Astra verdict: CONDITIONAL GO for this one L8/12 continuation, not
production or exit. Driver adopts it: the slowing prior drift but continuing
band/nuisance/count drift supports further discarded equilibration, while
the small nonstationary L6 efficiency advantage justifies a single bounded
length comparison, not an optimal-metric claim. The passing test1172573
supersedes the review's earlier reference to submitted-test status.
Record the new trajectory policy as a separate segment without rewriting
the historical trace. Stop on replay/gradient/numerical/capacity failure;
ordinary rejection repeats remain valid. Never auto-extend this allocation.
Afterward compare rejection-inclusive displacement per force, energy errors
and drift, separating IC modes/nuisances from field summaries. Whole-box
mass/bulk summaries can be conservation checks rather than mixing metrics.
If useful transport improves, cost subsequent equilibration/multichain work;
if not, close the length experiment and address an evidenced slow block or
oracle cost. No further tiny length ladder, automatic mass reduction, or
posterior promotion based on acceptance alone. Q-GOAL/Q-LEAN limits above
remain in force. No R2 exit approval, email or R3 entry.

Submitted419672 on h200/gpu:H200:1,48GiB,12h; scheduler confirmed RUNNING
on syn104. Source308b146d9e18031cd479750f6c3079ccaaeae9eb; output
/gpfs/kjhan/CF4/z0_density/r2_hmc_long_equilibrate_20261011.
The initial submission419671 was cancelled after10seconds to correct
Slurm's comma-separated --export parsing: CF4_HMC_LENGTHS='8,12' is now
passed through the submission environment with --export=ALL, not as a
comma-containing assignment inside the export list. No inference outcome
was produced by419671. Actual target replay and new transitions are pending.

## Terminal 419672 and user budget-change authorization

Slurm COMPLETED, exit0:0, elapsed09:30:08, MaxRSS20520928KiB
(19.57GiB), empty stderr. Exact evaluations89 of97; new transitions30--37
completed, seven accepted and one rejected. The next complete trajectory
could not fit the remaining evaluation budget. This is a protective stop,
not an OOM, crash, or evidence of stationary sampling.

Terminal count log likelihood is -137141.53874999422 and IC-prior NLL
8379756.490895055. Density mean square changed10.08372315639 at replay
to11.85118465847; continuing field drift excludes posterior promotion.
All states remain discarded; heldout scoring and UQ remain incomplete.

The user approved changing the calculation budget after this terminal
report. A bounded completion of the original ten-proposal comparison needs
only indices38--39: at most25 exact calls including replay, Slurm4h,
application3.5h, host48GiB, unchanged target/epsilon/metric and L8/12 policy.
This is a proposed replacement allocation, not a submitted job or an
authorization for unbounded repeats. Check terminal lineage and replay
wiring before submission. Automatic monitoring remains stopped unless
explicitly requested again. Evaluate the completed length comparison
before any larger equilibration allocation; do not claim R2 completion.

Restart lineage allowlist added for terminal419672/source308b146. Grammar
1173412 completed exit0:0, eight transport regressions passed. User requested
proceeding; submitted420128 on h200/gpu:H200:1,48GiB,4h, application12600s,
25 evaluations and end index40. Sourcec508d3921e5dd3747add2ca85eea5c4d91297a7e.
Output /gpfs/kjhan/CF4/z0_density/r2_hmc_long_finish_20261012.
Full actual target/gradient replay remains mandatory inside the allocation.
This bounded completion does not authorize automatic monitoring or further
allocations, posterior promotion, R2 exit email, or R3 entry.

## Completed longer-trajectory comparison: driver evaluation

420128 failed after12s before numerical execution: the driver committed its
submission record while the job was queued, violating the runner's exact
HEAD guard. No physics or memory failure. Corrected submission420759 used
HEAD1b7b364 without subsequent commits while queued/running. It completed
02:37:43, exit0:0, MaxRSS15502800KiB, empty stderr, actual checkpoint replay
passed. Indices38/39 (both L12) rejected/accepted;25 exact calls. Together
with419672 the original ten new proposals30--39 are complete, eight accepted.

Rejection-inclusive IC squared RMS displacement per force:
L4 .02979834 (13 proposals), L6 .04079802 (7),
L8 .07432852 (2), L12 .07800981 (8).
Fundamental-cosine squared displacement per force respectively
1.08971e-5,2.44071e-5,5.06414e-5,4.54120e-5.
Nuisance squared displacement per force respectively
.00018039,.00016520,.00019215,.00033757.
These are descriptive nonstationary transport proxies, NOT ESS, posterior
widths, an optimal-length determination, or a controlled stationary benchmark.
The longer segment required112 force calls and42463.53 trajectory seconds.

From state29 to39, count log likelihood improved -138367.52 ->-137011.76,
IC-prior NLL rose8337866.90 ->8378281.97, and conditional FP log likelihood
changed5977.826 ->5977.843. The k4--8 IC band changed .44540 ->.49804.
Replayed starting density mean square10.08372 increased to11.97078, rising
on every accepted new transition. This is continuing field drift, not
stationarity. All draws remain discarded; no heldout/UQ claim or R2 exit.

Driver closes the length experiment and provisionally retains the L8/12
mixture without another length ladder. A warranted Astra mid-course advice
request compares one costed larger discarded equilibration allocation
(<=24h Slurm,23h application,48GiB,193 evaluations,<=20 proposals) against
an evidenced slow-block/oracle-cost repair. This allocation is NOT submitted
or approved by the review yet. Automatic monitoring remains stopped.
Q-GOAL: same NEW field, MW/M31 roles ambiguous and M33 unresolved; their
observables must later constrain that same LG field, never truth-seeded
candidates. Q-LEAN: retain existing target/kernel/readouts and avoid a new
gate framework. Independent chains, heldout prediction and calibration
remain necessary; this recovery alone does not complete R2.

## Next discarded block: adopted Astra advice and user submission request

Astra CONDITIONAL GO for ONE24h block, not production or exit. Driver
checked the cited completed traces and adopts retained L8/12, unchanged
epsilon/metric/target, terminal420759 restart, end index60,193 calls including
replay,23h application/24h Slurm,48GiB. Expected16--20 complete proposals,
not guaranteed20. No automatic doubling if structural drift persists.
Compare first/second-half field, band, nuisance and likelihood behavior with
rejection repeats before costing independent chains or a specific repair.

Eight fixed spatial readouts added from the already computed field: mean
density and three mass-weighted velocities in R50/R100cMpc/h spheres,
cell centres(i+.5)*box/N and observer box/2, matching cell_centres geometry.
Empty diagnostic regions report null rather than invented velocities.
These are environmental diagnostics, not observations, UQ or LG identities.
Exact terminal420759 path/source1b7b364 added to restart allowlist; full
target/gradient replay remains mandatory. Grammar1177937 completed exit0:0,
all11 field/transport tests passed, including spatial mass weighting and
actual continuous/resumed equality. User explicitly requested submission.
Freeze HEAD after submission; record the job in the conversation until it
has passed the startup HEAD guard. Automatic monitoring remains stopped.

## Submitted block421553: verified terminal record

Submitted h200/gpu:H200:1 after syn104 showed8 configured/7 allocated GPUs;
48GiB host,24h Slurm,23h application,193 evaluations,end index60.
Source5016282433a375ac9dd89660a10b1ae53d25a370; output
/gpfs/kjhan/CF4/z0_density/r2_hmc_equil24h_20261013.
Slurm COMPLETED,exit0:0,elapsed19:09:01,batch MaxRSS19694680KiB
(18.78GiB). Checkpoint replay passed;185 evaluations completed.
All20 new transitions40--59 completed:16 accepted,4 rejected.
Terminal density mean square12.97806165699 versus11.97077577217 at
the previous retained state. This endpoint difference alone neither proves
nor quantifies stationarity. First/second-half regional,band,nuisance and
likelihood analysis remains pending before deciding another allocation.
Status remains CONDITIONAL_HMC_TRANSPORT_DIAGNOSTIC_NOT_POSTERIOR:
R2 incomplete,all states discarded,no heldout prediction or UQ delivered.
No follow-on calculation submitted; automatic monitoring remains stopped.

## Driver first/second-half assessment of421553

Existing JSON trace only; rejection repeats included, no new field evolution
or heldout access. Indices40--49 versus50--59 (10 states each):

| Summary | First half | Second half |
| --- | ---: | ---: |
| Accepted proposals | 7/10 | 9/10 |
| Mean density second moment | 12.402800 | 12.823131 |
| Mean IC band power k4--8 | .513047 | .536147 |
| Mean count log likelihood | -136850.121 | -136689.722 |
| Mean conditional FP log likelihood | 5979.353 | 5980.788 |
| Mean R50 density | 1.481909 | 1.482917 |
| Mean R100 density | 1.575212 | 1.577560 |

Among19 adjacent stored density-second-moment differences,12 increased,
3 decreased and4 repeated due to rejection. Both halves still move upward:
40->49 is12.12973->12.52501;50->59 is12.54286->12.97806.
The second-half endpoint decrease from its maximum13.01048 shows some
bidirectional movement but does not erase the comparable half-wide drift.
Large-radius mean densities are relatively stable; they average away the
smaller-scale structural changes and cannot establish field stationarity.
All24 nuisance coordinates were inspected; half means shift in several
coordinates (e.g.5: -.01879->.00546,16:.02335->.01356,
20:-1.50108->-1.49189). No posterior width or significance is inferred
from these short dependent nonstationary traces.

Driver decision: NO-GO for production draws, intervals or independent-chain
promotion on present evidence. Do not automatically double the allocation.
The next design must explain the persistent k4--8/structural equilibration
and cost before choosing further burn-in versus a targeted transport/cost
change; regional stability alone is insufficient. No new calculation or
automatic monitoring was started for this analysis. R2 remains incomplete.

## Cost and transport mechanism, next design question

421553 used184 forces in68308.07 trajectory seconds (~371s/force), with
14 L8 and6 L12 proposals. This is substantial cost for20 transitions,
not20 independent posterior samples. The implementation's Fourier inverse
mass is C(k)=k^2/(k^2+5999). Its exact Gaussian-prior rotation angle is
L*epsilon*sqrt(C(k)): for integer|k|=4 approximately .0495/.0743 radians
for L8/L12, and for|k|=8 approximately .0986/.1479. Full likelihood kicks
change this dynamics, so these prior-only angles do NOT establish posterior
correlation times or prove the metric is faulty. They do expose why large
whole-IC jumps (dominated by many high-k coordinates) cannot by themselves
justify adequate transport of the still-drifting k4--8 block.

Do not infer a prescription to reduce every Fourier mass: the preserved
matched600-vs6000 controls in CF4_R2_METRIC_SENSITIVITY_20261002.md found
loss of acceptance at600 for that earlier target/state/configuration.
Next design advice requested from the existing Astra reviewer compares a
target-preserving, justified slow-block/metric intervention or oracle-cost
repair against explicitly costed burn-in. No new allocation, metric change,
monitoring loop or posterior promotion follows from this analytical check.

Existing per-evaluation support timing provides a bounded cost attribution:
185 packing records total14861.91s, mean80.33s (range76.70--88.39s), about
21.6% of19:09:01 elapsed. Even eliminating all of that measured packing
time would provide only about1.28x end-to-end speedup under otherwise
unchanged costs. This is an idealized upper bound for this component, not
a promised implementation speedup. The uninstrumented remainder cannot be
assigned wholly to PMWD or likelihood derivatives from these records alone.
Do not pursue packing optimization as a substitute for the observed
mode-exploration problem without evidence of actual useful improvement.
The timing evidence was sent to Astra for the pending next-action advice.

## Adopted paired band-metric design

Astra CONDITIONAL GO for one common-state control, not another24h repeat.
Driver adopts its bounded design: terminal421553/state59; armA unchangedC,
armB4*C only on16<integer_k_squared<=64 (existing4<k<=8band).
Fundamental/otherIC/nuisance metric,fulltarget,epsilon remain unchanged.
Four predeclared matched Gaussian-momentum/uniform pairs, each arm starts
at the SAME state59; L8 botharms. Clone diagnostic RNG, preserve original
checkpoint/RNG, never choose or continue a preferred endpoint.
65exactcalls including shared replay,7.5happ/8hSlurm,48GiB; reserve a full
pair before starting it. Failed arms must not be silently omitted.

Report proposed and rejection-inclusive retained band displacement perforce,
plus acceptance-probability-weighted proposed displacement perforce.
Engineering spending rule (not statistical validation): B may be considered
for another bounded discarded run only if pooled weighted movement>=2*A
and improves in>=3/4pairs without numerical/capacity failure. Mixed results
are inconclusive; poor gain rejects this factor4proposal, not launches a
factor ladder. No posterior/UQ/exit claim follows even from clear gain.

Implemented only the fixed band-symbol transformation so far; it is not
called by existing transport and changes no existing run. Grammar1200173
completed exit0:0,10 band/transport tests passed, including exact band edges,
unchanged out-of-band/input and real-field symmetry. Paired diagnostic
driver/common-random/rejection-output wiring remains to implement before
any GPU submission. Q-GOAL/Q-LEAN and MW/M31/M33 limitations above apply.

Paired computational core now implemented in scripts/cf4_r2_band_control.py:
same-state A/B and diagnostic RNG clones, proposed/retained/acceptance-weighted
band displacement perforce. split_hmc_step accepts an optional readout callback
without changing its momentum, dynamics or acceptance draw. It never returns
a selected endpoint from the comparison. Grammar1200175 completed exit0:0,
12 tests passed: identical-arm common-random equality, original coordinate
preservation, rejection movement zero with nonzero proposed/weighted movement,
band boundaries and existing transport resume tests. These are mechanics
fixtures, not actual CF4 control results. Full Slurm entrypoint/budget and
checkpoint replay wiring are still pending; no GPU control submitted yet.

Full paired driver/budget and terminal421553 lineage replay are now wired.
All arms start state59, fixed diagnostic seed2026102001, original chain RNG
untouched. Each completed arm is saved before attempting the next; a failed
arm propagates failure and cannot be silently dropped. Reserve16 calls and
8120s before each pair. Four pairs maximum,65 total calls including replay,
27000s application/8h Slurm,48GiB. No endpoint chain checkpoint is written.
Complete-pair and insufficient-budget fixtures plus band/common-random/
rejection/transport regressions passed13 tests in grammar1200177,exit0:0.
The runner permits the explicit new entrypoint; shell syntax and diff checks
passed. Preparing one advised bounded submission, no automatic monitoring.

## Regional diagnostic coordinate correction identified after submission

Source inspection: particle_grid returns native nodes at i*dx, as explicitly
documented in src/cf4_r2_native_to_count_cells.py and conditional_map's
native_mesh_origin_fraction=0 metadata. present_field_readout receives that
native field directly, but its regional masks currently use(i+.5)*dx.
Thus R50/R100 regional readouts in421553 and the submitted paired control
are half-cell-offset diagnostics, NOT correctly observer-centred native-node
summaries. Do not use them for quantitative regional validation or claim
the earlier relative stability proves observer-centred environmental mixing.
The global density moments, band displacement, likelihood, target gradient,
MH decisions and paired engineering spending rule do not use these masks.
Those results are unaffected by this diagnostic-only geometry discrepancy.

Preserve submitted source6782f05/HEAD while429961 may still be queued:
no runtime patch or new commit during this interval. Correct the diagnostic
origin explicitly and add a native-node versus cell-centre boundary fixture
after the allocation is terminal (or otherwise source-safe). Do not rerun
the expensive experiment just to regenerate these secondary summaries.
No monitoring query or new calculation was performed for this finding.

User explicitly resumed autonomous work.429961 remained PENDING/QOSGrpGRES
with zero runtime. H200 was checked first; part_h200 GrpTRES caps GPUs at6.
H100 syn09 showed5 configured/4 allocated GPUs. Cancelled the unstarted
429961 to replace it on permitted h100/gpu:H100:1 without duplicate runs.
Applied native-node diagnostic origin correction; isolated equivalent copies
passed3 tests in grammar1200181. Comparing repository files with those tested
copies is required before launch. Full paired mechanism tests1200177 remain
valid; no prior/likelihood/kernel change for this resource replacement.

## Bounded paired band-metric control result — 2026-10-21 KST

Job429996 completed normally on `h100` with typed `gpu:H100:1`, node `syn08`,
source commit `666edbdff5702f7d525efec6461b7fe1cf3049a7`. Slurm reports
`COMPLETED`, exit0, elapsed5:41:03, MaxRSS16,684,480KiB (15.91GiB) against
48GiB requested. Output:
`/gpfs/kjhan/CF4/z0_density/r2_band_metric_control_h100_20261020/result.json`;
logs `cf4_band_h100_429996.{out,err}`. Checkpoint replay passed. The three
hwloc CPU-binding warnings were confined to startup; the result JSON has
`error:null` and no numerical/capacity failure.

The predeclared design asked for four complete same-state A/B pairs,65 exact
target calls including replay, and reserved16 calls plus8120s before a new
pair. This H100 run completed3 pairs using49 calls, then stopped with
`insufficient complete-pair budget`; the fourth pair was not started. The
overall status is `CONDITIONAL_BAND_CONTROL_DIAGNOSTIC_NOT_POSTERIOR`, not a
posterior or stationarity result. Per-force alpha-weighted squared band
displacement is the predeclared comparison:

| Pair | A accepted / alpha | A ΔH | A weighted jump/force | B accepted / alpha | B ΔH | B weighted jump/force | B wins |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 0 | yes / 0.4491 | 0.8005 | 0.634942 | no / 0.08138 | 2.5086 | 0.405678 | no |
| 1 | no / 0.3040 | 1.1906 | 0.450878 | no / 0.6264 | 0.4678 | 3.295806 | yes |
| 2 | yes / 0.2840 | 1.2588 | 0.396885 | yes / 0.3194 | 1.1412 | 1.591450 | yes |

Across the three completed pairs, pooled A=1.482705 and B=5.292935,
B/A=3.5698; B wins2/3. Acceptance counts are A2/3 and B1/3. Mean acceptance
probabilities are A0.3457 and B0.3424. Thus B has much larger proposed
band motion and a larger pooled alpha-weighted score, but it loses pair0 and
has only2 wins; B's first-pair alpha is particularly poor. The predeclared
spending rule requires B>=2*A and B wins at least3/4 complete pairs. Although
the pooled-movement inequality is met, only three of four pairs were completed,
so that rule is **inconclusive and not passed**. Do not promote factor4 as an
adopted metric, infer posterior mixing/ESS, or infer a stationary posterior.
Do not launch a factor ladder or silently count the incomplete fourth pair.

Driver disposition: retain the result as a bounded sampler-mixing diagnostic.
It does not change the observation target or current-field posterior, and it
delivers no held-out prediction, calibrated uncertainty, selection/bias or
multi-member group calibration, or MW/M31/M33 identification. R2 remains
incomplete/NO-GO. Q-GOAL: this can only be a sampler engineering input to the
CF4-conditioned same evolved z=0 field; it supplies no LG science result.
Q-LEAN: the run stayed within the original paired design and did not add a
metric sweep, simulation, or target change. MW/M31 remain role-ambiguous and
M33 unresolved; their observables must constrain those same roles on the same
NEW evolved LG field. Native truth identities remain evaluation/calibration
only, never candidate seeds or selectors.

## Astra advice and bounded pair-3 completion — 2026-10-21 KST

Astra's read-only post-run verdict is **inconclusive**, not a sampler failure
and not a passed metric-adoption rule. It recommends completing exactly the
missing fourth matched pair because that pair can change the predeclared
engineering decision; no fifth pair, factor ladder, or production metric
adoption follows automatically. Q-GOAL: this remains sampler engineering for
the same CF4-conditioned evolved z=0 field and provides no field, calibrated
posterior, mixing/ESS or UQ result. Q-LEAN: one missing pair is proportionate;
additional metric/step/trajectory sweeps are not. MW/M31 remain
role-ambiguous, M33 unresolved, and their observables must constrain those
same roles on the same NEW evolved LG field. Truth identities remain
calibration/evaluation only.

Driver evidence check: `split_hmc_step` draws one momentum vector of length
`256^3+24` and one acceptance uniform; the A/B controller clones the same
generator state and advances the diagnostic stream once per matched pair.
Replaying seed2026102001 with that exact draw size reproduced the recorded
decisions for pairs0–2, using shared uniforms
`0.378395465449627`, `0.6461830557339759`, and `0.06099715537565553`.
The next pair's expected shared uniform is `0.25203270157118596`. Thus the
missing pair is reproducible without repeating any of the first three target
trajectories. The driver adopts Astra's one-pair recommendation and adds
fail-closed RNG replay/acceptance checks; seven focused tests pass.

The continuation will replay the frozen state once and evaluate only pair3:
17 exact target evaluations in the new job (one checkpoint replay plus16
force evaluations), with a 2.75h application guard and3h Slurm cap. Since the
prior job used49 evaluations including its own replay, the auditable
cross-job total is66—one more than the original single-job65-call ceiling,
solely because the checkpoint must be independently replayed in a new job.
No original-chain RNG, target, likelihood, prior, selected state, or heldout
value is changed. The prior artifact remains immutable; the continuation
will merge its three completed rows only for applying the unchanged four-pair
rule. Based on measured MaxRSS15.91GiB, request20GiB host RAM (about26% headroom)
instead of repeating the prior48GiB request. No standalone simulation or
field promotion follows.

Pre-submit resource snapshot2026-10-21 KST: H200 `syn104` had7/8 GPUs
allocated and the `part_h200` QoS reports `GrpTRES=gres/gpu=6`; this user's
H200 allocation requests already consume that six-GPU cap, so no H200 slot is
eligible to this job despite one physical GPU remaining. H100 `syn08` was
5/5 allocated and `syn09` 4/5, leaving one eligible typed H100. A100
`syn101` and `syn102` were each8/8 allocated. Therefore select Slurm
`h100` with `--gres=gpu:H100:1`, 2 CPUs,20GiB RAM and a3h wall cap. No manual
node execution. Output path:
`/gpfs/kjhan/CF4/z0_density/r2_band_metric_pair3_20261021`.

## Pair-3 completion and spending-rule disposition — 2026-10-21 KST

Job430888 completed normally on `h100` / `gpu:H100:1`, node `syn08`, using
source commit `d58995e3ae03441143536f277ea5fb925c2ef7b1`. Slurm reports exit
0:0, elapsed1:56:51, and MaxRSS12,816,504KiB (12.22GiB) against20GiB
requested. The checkpoint replay passed, the prior three acceptance decisions
were reconstructed exactly, and pair3 completed. The result is at
`/gpfs/kjhan/CF4/z0_density/r2_band_metric_pair3_20261021/result.json`.

Pair3 A was accepted (alpha0.54012; weighted squared band jump/force
0.725324); B was rejected (alpha0.09562; weighted score0.462932), so B loses
this pair. Independently summing the four saved pair rows gives A=2.208029,
B=5.755867, B/A=2.606790, with B wins2/4. The pooled-score threshold
B>=2*A passes, but the predeclared >=3/4 pair-win threshold fails. Therefore
`engineering_gain_rule_passed=false`; the factor-four proposal is closed
without another pair, factor ladder, metric change, or endpoint adoption.
This is not evidence of sampler failure; it simply does not justify spending
more on this proposal under its stated rule. Across both jobs the exact target
evaluation count is66, with zero repeated prior target trajectories. The
separate-process replay accounts for the extra call beyond the original
single-job budget.

Astra's read-only post-calculation mid-course verdict agrees: continuation was
supported, the spending rule fails, and this proposal should close. Astra
verified the pair3 replay/acceptance, aggregate scores and the distinction
between a local engineering result and posterior evidence. The driver
independently recomputed the four-row aggregates from the saved result JSON
and adopts that narrow recommendation. Astra's advice also points back to the
observation-law boundary; it is not an R2 exit approval.

Slurm and JSON both show normal completion, no error/stop reason, four complete
pairs, and the intended conditional diagnostic status. Every endpoint remains
discarded for inference: `selected_endpoint=false`, `posterior_sample=false`,
`posterior_uncertainty=false`, `heldout_scored=false`, and `R2_complete=false`.
The result is neither a calibrated posterior nor a z=0 density/velocity map.
Astra's post-calculation R2 mid-course audit does not substitute for the
required future R2 exit audit.

Driver disposition and next R2 priority: stop sampler/metric tuning and return
to the observation-law blocker—what is observed, selected, linked, grouped,
and lost from the underlying CF4 population, with defensible shared
count/mark covariance and predictive calibration. Do not invent missing
survival/bias parameters or call the conditional linked-mark likelihood a
full-survey likelihood. The driver completed the source/identifiability table
by reusing the settled denominator, source-access and v6 graph audits; no new
catalogue census or heldout read was made. The table separates the observed
47,121-count/1,414-mark conditional component from unobserved parent/failure
denominators, group inclusion, and shared covariance. Its record is
`config/CF4_R2_OBSERVATION_LAW_IDENTIFIABILITY_20261021.md`.

The proposed DR14 `Target`/`TargetInfo` bridge was tested read-only against the
exact frozen 1,414-mark training cohort using eight public SkyServer SQL
requests, each capped at 200 object IDs. The local frozen PGC-to-SDSS-PV join
was 1,414/1,414 unique. All 1,414 SDSS `bestObjID`s exist in DR14
`SpecObjAll`; they yield 1,414 science-primary and 1,512 total spectral rows,
all with `zWarning=0`. However, the tested candidate
`SpecObjAll.targetObjID -> Target.targetID -> TargetInfo.targetID` join yields
zero `Target` and zero `TargetInfo` links. This closes that direct-key route
as a target-parent bridge; one representative object's `Target.bestObjID` and
`Target.specObjID` keys were also absent, but alternate keys were not checked
cohort-wide. No bulk catalogue was downloaded and no target/success rate is
estimated. Absence of a link is not evidence of a failed or unselected source.

The next boundary is an actual source-specific parent/failure sample, if one
becomes available, or a narrower conditional estimand with availability-bias
assumptions justified externally. Do not infer such assumptions from selected
rows, or launch another sampler or large simulation in this reconciliation.
Q-GOAL: this evidence is relevant to whether CF4 can constrain the same
z=0 field and eventual IC/history, but it does not produce that field. Q-LEAN:
one exact-cohort test closes the unsupported route without a new catalogue
census or repeated metric work. MW/M31 remain role-ambiguous and M33
unresolved; their observables must constrain those roles on the same NEW
evolved LG field, with native truth IDs used only for calibration/evaluation.
R2 remains NO-GO.

## 2026-10-21 update — bounded SDSS-I/II tiling provenance and parent probe

The exact frozen 1,414-mark cohort now has a partial historical targeting
bridge. Eight bounded DR14 SkyServer requests (at most 180 selected IDs per
request) found 1,400 selected spectra with `SpecObjAll.targetObjID` matching
`sdssTiledTargetAll.targetID` and `sdssTilingInfo.targetID`. The same-target
`SpecObjAll.tile = sdssTilingInfo.tile` comparison matched 1,385 selected
spectra; 15 more have historical tiling rows but no tile equality, and 14
have no historical target/tiling record. These are selected-spectrum
provenance categories, not success/failure categories. Official DR14
descriptions limit these tables to SDSS-I/II tiling (`targetID` is the CAS
DR7 object ID; `sdssTilingInfo` records each target/run).

Astra advised one bounded parent-feasibility aggregate. The canonical run was
fixed independently of the cohort as the first indexed `sdssTileAll` pair
(`tileRun=4`, `tile=73`); the final aggregate was correctly run over the
**entire indexed `tileRun=4`**, beginning with all `sdssTilingInfo` rows and
not conditioning on a specific assigned tile. A first exploratory query of
only `tileRun=4,tile=73` was discarded from inference because selecting a
tile conditions on assignment.

The full run-4 aggregate returned 28,633 distinct target/run inputs. By raw
`tiMask`, the rows were 0:23, 2:254, 4:1,636, 5:881, 6:60, and 7:25,779.
The TILED bit was set for 26,660; DECOLLIDED for 26,093; COVERED for 28,356;
REMOVED/CULLED for zero. A left join to `sdssTiledTargetAll` matched 26,360
of 28,633 IDs, leaving 2,273 unmatched (retained, not filtered). A left join
to `SpecObjAll` by `targetObjID` found 27,023 targets with any spectrum row,
26,621 with a same-tile spectrum row, 25,087 with a same-tile
`sciencePrimary` row, and 24,880 with a same-tile `sciencePrimary`,
`zWarning=0`, non-null redshift record. The last count is a narrow catalogue
record proxy, not a vetted FP-usable redshift or distance measurement.

This identifies a limited **post-target-selection tiling-run frame** for one
historical SDSS-I/II run, with assignment flags and some spectrum-record
status independent of the selected CF4 training IDs. It does not identify
photometric pre-target eligibility; the 2,273 target-table key mismatch
remains unexplained; it does not establish source-matched FP quality,
morphology, or downstream CF4/Tempel linkage; and a single run is not
representative of the 75 cohort-linked runs, let alone transferable to all
34,059 SDSS-PV marks or the heterogeneous CF4 sample. No selection/survival
probability is estimated or fitted. The 2,273 target-table mismatches remain
in-frame with unknown selection attributes; the 24,880 clean-warning, non-null
redshift records are not complete observation outcomes or FP-usable distance
measurements. The exact cohort
binding, SQL templates/fingerprints, aggregate table, and interpretation are
in `config/cf4_r2_dr14_tiling_cohort_probe_20261021.md`.

Driver disposition: retain this as partial SDSS-I/II provenance and a
post-selection stage feasibility result, but close it as a route to a full
CF4/SDSS-PV parent/failure denominator unless a source-matched parent and
remaining quality/linkage stages are independently supplied. Do not expand
to a global table scan, archive download, or sampler restart. Q-GOAL: this
constrains at most a narrow observation stage; it is not a z=0 map or
posterior. The active route remains joint latent-IC/evolved-state/history
inference, with z=0 posterior first. MW/M31 remain role-ambiguous and M33
unresolved; their observables must constrain those same roles on the same NEW
evolved LG field at `<=0.3 cMpc/h`, with native truth IDs used only for
calibration/evaluation. Q-LEAN: two exact-cohort query rounds and one
run-bounded aggregate; no large data transfer, simulation, or new gates.
R2 remains NO-GO; this is not the separate R2 exit audit.

Astra's final advisory accepts the parent-feasibility result narrowly and
recommends stopping exploratory coverage queries. It emphasizes that run4 is
a deterministic feasibility example, not representative of the 75
cohort-linked runs; the 2,273 target-table misses remain in-frame with
unknown attributes; and the spectrum/redshift-record proxies are not
attempt-complete outcomes. The driver adopts that advice. The run-level SQL
fingerprint is retained, but the exact submitted SQL text and raw response
were not saved; the evidence file labels its join description as a
reconstruction rather than a replayable query.
