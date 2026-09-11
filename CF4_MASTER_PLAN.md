# CF4 / OurUniv — active master plan

Effective 2026-09-07. The user approved registering the goal-alignment review
as the highest-level project plan and proceeding with short, substantive
bundles. This file supersedes CODEX_PLAN.md, PLAN.md, the Hong restart plan,
and cf4_science_route_v3.json as execution/priority authority. Preserve those
files and previous results as history; do not silently revive their routes.
Direct subsequent user instructions take precedence over this file.

## Scientific destination

Use actual CF4 observations, galaxy-density observations, and explicit local
structure constraints to infer a **present-day density/velocity posterior**;
construct compatible LCDM initial conditions; evolve them forward and verify
the observed local environment and MW–M31–M33 system. Deliver phase-consistent
zoom ICs for studying LG formation and evolution, not a unique historical
reconstruction of every small-scale phase.

- Surroundings: 1–2 cMpc/h numerical reconstruction scale is sufficient.
- LG: target <=0.3 cMpc/h numerical reconstruction scale, with explicit LG
  mass, position, distance and relative-velocity uncertainties.
- Finer IC particle/force resolution is a separate zoom requirement. AMR
  alone does not improve particle mass resolution. Fine modes not fixed by
  observations remain conditional/prior content, not recovered observations.
- Retain Virgo/Coma and Local/Bootes Void environment constraints. Separate
  cluster zooms, full-volume 0.3 phase recovery, and RT/stellar/AGN/dust work
  are not prerequisites for the LG deliverable.
- Push defensible observation/structure information toward high k, especially
  in the LG region. Measure information gain; do not arbitrarily choose a
  tiny low-k domain and call the goal achieved. Conversely, global frontier
  certification must not block a useful, explicitly qualified LG prototype.

## Four outcome-based bundles

| Bundle | Deliverable | Exit decision |
| --- | --- | --- |
| A — usable present-field model | One bounded prior comparison, then an actual CF4 + galaxy-data preliminary z=0 density/velocity product with uncertainty and held-out predictions | Declare model limitations and whether it is usable; no indefinite toy-model repair loop |
| B — LG and dynamics connection | Explicit LG constraints in the z=0 route and a small mock z=0 → IC → forward-z=0 demonstration | Check LG/environment conditions and field/velocity consistency after evolution; test this before expensive high resolution |
| C — multiresolution reconstruction | Surroundings at 1–2 and LG at <=0.3 cMpc/h, with coarse/fine coupling and measured observation/structure information | Show LG-conditioned information gain, resolution/cost feasibility and residual uncertainty; recalibrate scale-dependent priors as needed |
| D — zoom IC and evolution | Phase-consistent nested ICs from buffered Lagrangian particle membership and a forward-evolved LG ensemble | Check LG masses/separations/velocities, environment and contamination, then decide on production |

The scientific order remains observations → z=0 posterior → IC → forward
validation. Bundle B is an early end-to-end mock risk test, not permission to
bypass the present-field stage or restart the historical direct-CF4 route.
Do not double-count the same data by treating a data-derived z=0 posterior as
an independent likelihood. Any approximate inversion/proposal must disclose
its target and correction before calling its output an IC posterior.

## Current state and bounded immediate work

**Bundle A is closed as a diagnostic delivery, not scientific certification.
Bundle B is closed as a development/diagnostic delivery. Bundle C entry is
approved and implementation is in progress. D requires user approval.**
Current design: [BUNDLE_C_DESIGN.md](BUNDLE_C_DESIGN.md).

Bundle B: environment334398 and four LG interface tests334401 completed;
bounded dynamics retry334402 passed both fixed development cases in4m58s.
Density RMS residuals0.964/0.972 became0.118/0.118; velocity333/300 became
17.8/18.1 km/s, with conservative readout. These are regularized mock IC
candidates, NOT actual-data IC posterior samples. The actual map's Local
Void and precise cluster localization remain unresolved. The source-backed
LG interface still needs a resolved fine-field operator and covariance/model
discrepancy calibration before actual conditioning. No further N32 sampling
or bridge extensions: next work must bring LG observations into the fine z=0
field while retaining its coarse mass/momentum environment. C now begins with
native observations at1.5 cMpc/h, an LG0.1875 layout and conservative coupling;
these are not yet a high-resolution posterior or an evaluated LG operator.
Native-data334408 and conservative/JAX tests passed. Selected actual inputs
contain no direct CF4 or count rows inside LG R2 cMpc/h (CF4 minimum radius
15.8152). LG observations must supply that information explicitly. Selection
integration334409 completed, but order4 missed the positive angular footprint
of one occupied population/cell. Native row positions have valid source-mask
and LF support. Geometry-only repair334508 and preservation check334512
completed: occupied zero-support keys1→0;70260 unoccupied population/cells
also repaired; all prior positive entries unchanged. The v2 selection is a
development input, NOT precision-certified:120 geometry controls still find
33 tiny-support population/cell entries missed by2048-point cubature, with
maximum tested shell-L1 absolute error9.37e-4. Preserve v1 and this limitation.
No high-resolution posterior inference is running. Next substantive C work
is the physically specified fine-field prior and LG halo/subhalo operator,
then the bounded LG-on/off comparison, not another coarse sampler extension.
See the C run record for current products and limits.

Native resolved-operator jobs334521/334522 now completed. A real TNG FoF
fixture with989815 particles supplies mass/mean-velocity/physical-dispersion
fields at0.1875 and1.5, with conservative restriction and particle/SUBFIND
mass/COM agreement. This is a simulated FoF component, NOT a reconstructed
LG or full matter map. The catalogue readout is valid for that unmodified
particle realization; arbitrary changes to fine cells cannot inherit its
halo catalogue. A full-matter conditional fine prior and actual joint LG
conditioning remain outstanding. No new high-resolution posterior is running.

The next C implementation is a finite whole-patch coarse-summary conditional
prior, plus a total-matter source including diffuse/non-FoF matter. It retains
native field/catalogue consistency and separates physical sigma_v from
posterior uncertainty in mean velocity. This finite support baseline does
NOT yet define a continuous LG posterior or exact full-parent conditioning.
Two-file timing334524 and5 regressions passed; full source job334528 submitted
(2 CPUs,9600 MiB,4h cap, estimated2–3h). It builds native400^3/50^3 moments
and18 train/9 nonoverlap check patches from existing TNG. No new simulation,
Hong retraining, actual CF4/LG weighting or automatic downstream inference.
See BUNDLE_C_RUN.md. Do not broaden prior kernels or narrow the science target
just to disguise insufficient support from a small finite patch collection.

Update2026-09-08:334528 completed03:27:20 KST in1h58m04s. All448 files,
11935938442 massive particles/cells processed; conservation passes, absolute
native cosmological mass error4.8093e-5. The18-component prior fails support
on all9 heldout targets (ESS1.00–2.06). This is not a CF4/LG posterior.
User authorized the next C step: ONE bounded support comparison using dense
native translations and24 exact proper rotations, unchanged32-dimensional
conditioning and prior bandwidth, strict heldout spatial exclusion. Report
raw, distinct-anchor and source-spatial-group concentration separately; do
not count correlated/rotated copies as independent universes. If grouped
support still fails, close finite-bank expansion and move to a continuous
joint matter/halo-model design, not repeated kernel/seed tuning.

That bounded comparison335875 has now completed:23275 native anchors and24
proper rotations (558600 correlated hypotheses). Raw component ESS rises to
231.6–2439.3, but native spatial-group concentration ESS is only1.70–5.35;
4/9 heldout targets still fail ESS>=4/max-group-weight<=0.5. Conditions and
bandwidth were unchanged; original results reproduced and the transformed
actual fine field passed its summary check. Status
NO_GO_FINITE_BANK_SUPPORT_CLOSE_THIS_REPAIR. CLOSE this finite-bank expansion.
The next C design must vary matter and halo state jointly and continuously,
with a defined diffuse component and calibrated physical/model discrepancy;
no further bank densification, kernel widening, N32 extension or fake LG map.
This result does not prove that every finite prior fails, nor that a continuous
model will automatically succeed. No actual CF4/LG fine inference is running.

Continuous-state implementation335878 now completed in24s: a native total-
matter patch with three disjoint member components supports21 continuous
position/member-mass/COM-velocity marks, coupled positive remainder mass and
momentum compensation, and correct second-moment transport. Conservation and
one noiseless same-generator inverse control pass. This is a KINEMATIC
OPERATOR, not a calibrated continuous LCDM prior, resolved transformed halo,
independent validation or actual LG posterior. Close the toy-control step.
The next substantive requirement remains the physical joint distribution of
environment, halo marks/profiles and remainder response with COM/model
discrepancy. The21 local marks cannot by themselves fix support on all32
coarse environmental features. Do not promote the precise numerical inverse
error to astronomical accuracy or restart bank expansion. Actual LG-on/off
information comparison remains undelivered; see BUNDLE_C_RUN.md.

Actual LG MARK conditioning has now run (335879,42s):51971 native two-primary/
third-object triples from5601 distinct observer candidates; separate M33
satellite and independent-primary alternatives. A continuous transformed-mark/
shell-summary distribution and measured stellar-minus-halo COM proxy connect
actual distances/LOS/proper motions, including shared MW nuisances and required
probability Jacobians. This is NOT a CF4-conditioned .1875 matter-field prior
or spatial map. Review335880 exposes separate-primary extrapolation despite
high proposal ESS; that alternative is NO-GO for scientific adoption, not
proof of physical impossibility. Gaussian leakage below the1000-DM particle
mass floor was fixed and existing draws restricted/re-normalized in335881,
preserving original outputs. Corrected satellite-conditional marks are usable
as a development input, with broad prior-dominated masses/environment, not a
calibrated LG reconstruction. Current products: `resolved_support_samples.h5`
and `physical_summary_v2.json` in `bundle_c_v1/lg_population_v1`.

Next substantive requirement is the SPATIAL profile/remainder-field model and
its joint connection to actual CF4/galaxy environment, not more mark-only
fits. Native membership/force resolution, missing mass/LMC/stellar-disk priors,
one-box covariance calibration and the32 environmental-feature support problem
are not solved by the marked Gaussian. The desired spatial LG-on/off map and
information gain remain undelivered. No job from this step remains active.

User authorized the next execution bundle, C-spatial, within the unfinished
master C. Scope/first calculation: BUNDLE_C_SPATIAL_DESIGN.md. Start with
native disjoint member profiles plus remaining total matter for16 training
and16 retained spatial cases; then define the continuous spatial distribution
and joint CF4/LG target. Do not reinterpret this as entry to D or completion
of the .1875 spatial deliverable. Source calibration alone does not close the
new execution bundle.

C-spatial first source calculation335916 completed4m08s:93 native member
profiles from16638729 massive rows,32 full member/remainder decompositions
at0.1875 with1.5 restriction and native mass/COM checks passing. These are
spatial CALIBRATION INPUTS, not a fitted conditional field or bundle closure.
Current record BUNDLE_C_SPATIAL_RUN.md. Whole source patches can overlap even
when halo IDs are disjoint; spatial fitting must prevent heldout-voxel leakage.
Next remains the continuous spatial distribution and actual observation-space
connection, not IC generation or another mark-only fit. No current job.

C-spatial conditional remainder candidate335968 completed2m46s, two focused
tests and all12 realizable/conservative draws passed numerical checks. A
geometry-only split provided13 training cubes and three disjoint retained
cubes without training/test voxel leakage. However all12 draws fail the
predeclared small-scale power criterion, both with and without the fixed
native member halos. Status NO_GO_CONDITIONAL_REMAINDER_MORPHOLOGY. This
stationary five-channel Gaussian-copula candidate is CLOSED, not adopted
or patched by amplitude/seed tuning. Native coarse7 moments and halos were
oracle conditions; no actual CF4/LG spatial inference was run. The fitted
profile regression is separate and not a joint halo/field law. Conserving
mass/momentum/second moments does not certify cosmological spatial structure.
See BUNDLE_C_SPATIAL_RUN.md for quantitative results. No active or downstream
job remains. C-spatial is not complete: the next design must supply missing
nonlinear environment/halo–matter spatial dependence before observation
conditioning. Do not close C or enter D on these diagnostic generated fields.

Current execution record: [BUNDLE_C_RUN.md](BUNDLE_C_RUN.md).

User-approved cause separation336263 completed42s; no new fit/draws. Native
density encode/decode is exact to2.6e-16 L1, and quantile+conservation roundtrip
power changes<=0.0944%, unlike the large stochastic generation failure. This
localizes the problem to the generative path/conditioning interaction without
uniquely identifying random phases as cause. Separate confirmed defect: scalar
fine sigma loses directional variance, changing mean velocity by3.4–16.8 km/s
per-axis RMS even for native input. Retain three fine variances in the next
representation. Details BUNDLE_C_SPATIAL_DIAGNOSIS.md and its comparison PNG.
Proposed next implementation BUNDLE_C_CONDITIONAL_FLOW_DESIGN.md: one spatially
conditioned multiscale flow with lossless conservative moment coordinates,
explicitly identifying the missing1.5 environment prior/inference and joint
member-field readout. This is DESIGN, not validated ML or an actual map. Current
approval ends at diagnosis/design; replacement pilot awaits approval. No job.
Subsequent user approval authorizes the implementation/one-GPU4h pilot.
Implementation/source1dc835b submitted as336268: shared multiscale spatial
conditional flow, explicit atom/continuous branches and conservative binary
moment coordinates retaining all directional variances. Frozen configuration,
within-fit spatial split/history limits and gates: BUNDLE_C_FLOW_PILOT_RUN.md.
One job includes native roundtrip, fixed learning and environment/fine mock
generation/evaluation. q_S, actual1.5/CF4/LG posterior and global384 inference
remain outside this pilot; no automatic follow-up science job.
336268 has now completed6000 steps in4m23s: native seven-moment/directional-
variance roundtrip passes, all16 generated cases fail development gates. No
actual-data adoption. User requested continued work2026-09-09. One identified
implementation mismatch is cropped24-parent training versus full64-parent
generation context. BUNDLE_C_FLOW_PILOT_RUN.md now freezes checkpoint-based
paired diagnosis and, ONLY if confirmed, one full-context6000-step correction
with unchanged model/seeds/gates. This is not an automatic longer-training
or prior-family series. All v1 products remain preserved. No actual CF4/LG or
global384 posterior launch; source/representation success is not C completion.
The gated repair337195 stopped after39s before training: crop-context mismatch
is confirmed, but trained-flow inverse/logdet errors exceed numerical limits.
User now authorizes cause separation and necessary correction/reverification:
one frozen same-input default-FP32/strict-FP32/FP64 comparison, not additional
training or relaxed morphology criteria. See BUNDLE_C_FLOW_PILOT_RUN.md.
337268 completed41s: all strict-FP32/FP64 cases pass, all default-FP32 cases
fail. TF32 convolution is supported as the inversion error source, not the
cause of all morphology failures. User now approves strict-FP32 paired gate
and, only on pass, the previously blocked single full-context6000-step fit.
Keep existing numerical/science thresholds and preserve all earlier outputs.
The current repair changes context and precision; no actual CF4/LG inference.
337279 completed9m30s,6000 steps, all numerical gates pass but all16 generated
fields still fail development morphology gates (fine high-band ratio0.235–0.593).
User approved ONE frozen analysis covering before/after quality, train/heldout
fixed likelihood, teacher-parent versus rollout losses, and likelihood/quality
alignment; driver then judges model viability AND actual CF4/LG connection.
Scope BUNDLE_C_FLOW_DECISION.md. No new training or replacement model authorized.
337496 analysis completed2m43s: v2 heldout fine high-band mean0.650 with true
parent versus0.390 in rollout; training0.676 versus0.297. Both one-step and
accumulated deficiencies remain. User accepted withholding current-model
adoption and authorized continued redesign. Candidate proposal:
BUNDLE_C_STRUCTURE_LG_REDESIGN.md, matched-update native-NLL control versus
structural-score repair plus a separately labelled LG member-state readout.
This is DESIGN, not a new fit authorization or a completed q_S/global law.
No actual LG posterior, direct-CF4 IC restart or amplitude repair is allowed.
Subsequent user approval authorizes entry to this next bundle, with a renewed
plan audit FIRST: Fable5 primary, Astra backup only if Fable audit cannot be
completed. Ask Q-GOAL (final-goal contribution) and Q-LEAN (excessive versus
necessary instrumentation/gates), feasibility, and essential/deferred scope.
An adverse scientific verdict is evidence to address, not an invocation failure
to bypass by seeking approval elsewhere. Preserve existing bundle boundaries.
Fable5 plan audit completed normally with CONDITIONAL GO (Q-GOAL conditional,
Q-LEAN broadly proportionate). Driver review found the suggested Gaussian
proxy circular unless its g(F) can infer members from total field alone; native
component inputs are not that operator. See BUNDLE_C_STRUCTURE_LG_PLAN_AUDIT.md.
No Astra fallback/no new fit. Ask before prioritizing the field-only LG proxy
connection ahead of the approved paired learning experiment; do not silently
claim that a Gaussian covariance supplies the missing joint physical q_S.
User now approves prioritizing field-only LG identification/readout before
paired learning. Every future bundle plan AND audit request must explicitly
cover MW/M31/M33 identification, ambiguous/unresolved cases and field-observation
connection. Current bounded implementation plan: BUNDLE_C_LG_IDENTIFICATION.md.
Native labels are for evaluation/calibration after blind candidates are frozen,
not inference inputs. Fable5 plan audit first, Astra fallback on invocation failure.
Fable5 completed the identification plan audit normally: CONDITIONAL GO, with
Q-GOAL direct/Q-LEAN proportionate. All four disclosure conditions (support
source, boundary truth counts, velocity/residual convention, role completeness/
shared peaks/aperture overlap) are implemented in the single CPU job. No extra
auditor/gate, new training or proxy likelihood is authorized by that result.
Identification job337986 completed1m13s, tests2/2, all32 cached native fixtures.
All32 show a shared M31/M33 nearest peak; distinct three-way matching0/32.
Two heldout M33 matches replace M31 on the shared peak, not a third detection.
No boundary truth loss; M33 training calibration unavailable. This closes the
bounded diagnostic, NOT q_S or an observed LG posterior. See the identification
report for counts and limitations. Recommend an explicit unresolved-member/
assignment observation-link design (or justified finer LOCAL information),
with Fable5 plan audit and user bundle approval before implementation. Do not
resume paired learning or new simulations automatically.
User subsequently approved the unresolved-member/observation-link DESIGN bundle.
Current proposal: BUNDLE_C_UNRESOLVED_MEMBER_DESIGN.md, with Fable5 plan audit
requested before any numerical implementation. It must address MW/M31 role
ambiguity as well as M33, normalized same-field component budgets and actual
field information, not repeat native-known-member transport or mark-only fits.
No new training, simulation, actual LG weighting or q_F promotion is implied.
Fable5 returned DESIGN CONDITIONAL GO (normal completion105402ms). Driver
adopts its Q-LEAN recommendation: a14-parameter composite-aperture kinematic
proxy first, not a high-dimensional cell-member learner. Exact finite pair
mixture, joint MW/M31/M33 uncertainty and fixed heldout field cross-scoring are
specified in BUNDLE_C_COMPOSITE_PROXY_PLAN.md. This is the NEXT proposed CPU
implementation, awaiting approval; no job submitted. Cell-member anchor measure
and p(E|F,O)/E-conditioned field prior remain unresolved and explicitly deferred,
not declared solved by conditional covariance or deterministic anchor encoding.
Fable5 follow-up on the concrete14-coefficient composite design returned GO.
Three required disclosures/tests (K spectrum/condition and sample counts,
consistent physical scale determinants, explicit unsupported-context labels)
are incorporated. Design bundle COMPLETE; proposed single2-CPU/1200MiB/10min
implementation now awaits user approval. No active job or new fit from this
design turn. No additional audit micro-stage for these accepted report items.
User now approved the concrete composite-proxy implementation and single Slurm
CPU diagnostic. Sources: src/cf4_lg_composite_proxy.py and
scripts/cf4_bundle_c_composite_proxy.py. Three focused tests run in the same
job before the one14-coefficient fit and frozen13/3 native evaluation. No new
neural training, simulation, actual-data weighting or extra audit is requested.
Composite diagnostic337991 now COMPLETED11s, tests3/3,13 training pairs and all
9 cross-field scores valid. Full/host-only scores prefer the source field in
all3 retained cases; incremental conditional-M33 contrasts versus other-field
means are[+.479,-1.358,-.023] nats and never rank the original field first.
Thus host field dependence is demonstrated diagnostically, but consistent
incremental M33 information is NOT established. No observed posterior/physical
member partition is promoted. See BUNDLE_C_COMPOSITE_PROXY_PLAN.md for full
results/limitations. Close this14-coefficient trial; no automatic refit or
neural model. Next design/approval must address the missing M33 mass/subcell
or finer-local-field connection and retain unresolved q_F/selection caveats.
User approved the next M33 mass/subcell DESIGN. BUNDLE_C_MEMBER_BUDGET_PLAN.md
now specifies one no-fit necessary mass/momentum/diagonal-second-moment budget
test over seven support unions, preventing host/satellite double counting.
Fable5 DESIGN CONDITIONAL GO received normally; four essential conditions are
incorporated (identical cell masks, congruent PSD scaling, unavailable contexts,
common BOX velocity convention). Native supplied mock positions/COMs/masses
are explicit stronger conditioning, not recovered identities or actual masses.
If hosts exclude every alternative, report inconclusive M33 conditional power,
not no M33 information. No posterior/support likelihood or sufficient physical
decomposition is claimed. Design complete; single2-CPU/1200MiB/10min diagnostic
implementation awaits approval. No new numerical job/ML/simulation launched.
User now approves that exact member-budget implementation/one CPU execution.
Sources src/cf4_member_budget.py and scripts/cf4_bundle_c_member_budget.py:
two focused tests then cached fixed16/17/20 mocks and both radii in one job.
No new plan audit, fit, actual-data weighting or automatic downstream task.
Member-budget338040 completed11s, tests2/2,18 comparisons,783 cached cell reads.
Both fixed radii and mass/moment modes: own-field controls3/3 pass, all6
alternatives already fail host-only mass requirements, unavailable0. Decision
INCONCLUSIVE_NO_HOST_COMPATIBLE_ALTERNATIVES, NOT no M33 information. The
necessary union-budget operator is implemented but not a posterior or physical
allocation. Close this one-shot test without further contexts/radii/refits.
Recommend next planning return to the missing present-field prior/inference
with MW/M31 conditions and explicit unresolved M33, rather than more standalone
M33 diagnostic variants. No next bundle is authorized by this result; no job
remains active. See BUNDLE_C_MEMBER_BUDGET_PLAN.md for results and limitations.
User approved the next central-work DESIGN on2026-09-10. Proposed plan:
BUNDLE_C_FIELD_RECOVERY_PLAN.md. It closes standalone member diagnostics and
asks Fable5 whether ONE bounded matched generative-objective repair is the
right immediate bottleneck, explicitly retaining the missing member/selection
and global environment laws. No actual-data fine inference or numerical run
is authorized by this planning approval. Fable5 review completed normally in
137035ms: CONDITIONAL GO, Q-GOAL/Q-LEAN accept one terminal matched experiment.
Three essential corrections are incorporated: training-gradient-based frozen
weight, worst-case full-rollout gradient/memory screen before training, and
one-step/rollout reporting even on failure. Driver does not adopt the audit's
overstrong causal classification or its coarse-posterior/prior-fine-IC fallback
as an automatic route change. See BUNDLE_C_FIELD_RECOVERY_AUDIT.md. DESIGN
complete; one1000-update-per-branch/4h Slurm implementation awaits approval.
No numerical job submitted, no high-resolution observed LG posterior exists.
User now approves the exact field-recovery implementation and one Slurm GPU
experiment. BUNDLE_C_FIELD_RECOVERY_RUN.md records source/configuration:
all-trace energy-score gradient, bounded8-pair screen, equal1000-step branches,
unchanged morphology gates and terminal comparison. No new Fable request or
standalone M33 diagnostic; no actual-data fine posterior or automatic follow-up.
Implementation source43692a1 pushed and submitted as Slurm338194. Initial
state PENDING(Resources), no node/tests/training yet. One job runs the7 focused
regressions, fixed feasibility segment, matched learning and final evaluation.
Execution status belongs in BUNDLE_C_FIELD_RECOVERY_RUN.md, not inferred from
submission success. No new scientific output or next-bundle authorization.
338194 subsequently finished both1000-step branches and saved checkpoints,
but failed at evaluation entry on2026-09-10 02:04:45 KST: FP64 native split
coordinates were passed to FP32 convolution. Seven tests and the operational
screen passed; morphology remains unevaluated. User requests this code fix.
The correction adds the missing network dtype conversion and one regression,
plus evaluation-only resumption using unchanged checkpoints/criteria into a
new output directory. No retraining, new model trial or additional plan audit.
Details BUNDLE_C_FIELD_RECOVERY_RUN.md; evaluation-only execution via Slurm.
Evaluation-only338389 now COMPLETED4m54s,8 tests pass, both saved-model inverse
checks pass, additional training updates0. Intermediate338388 test cleanup
defect is corrected/preserved in the run record. Fixed evaluation source18b84f8.
Both branches pass0/16 original morphology cases. Retained .1875 true-parent
P/native control/repair0.654972/0.664406; rollout0.436332/0.462160. ES modestly
improves both summaries but remains below acceptance: CLOSE_THIS_REPAIR_LINE_
BOTH_FAIL. Close this objective/current-architecture repair without more
steps/seeds/weights. No actual high-resolution LG posterior, no next-bundle
launch or automatic prior-fine-IC fallback. Detailed results and preserved
checkpoints: BUNDLE_C_FIELD_RECOVERY_RUN.md. Evaluation code error is resolved.
User now approves designing the recommended conditional3D U-Net diffusion
replacement, with augmentation and an explicit same-field LG connection plan.
BUNDLE_C_DIFFUSION_PLAN.md is the next proposed implementation, not a launch.
Fable5 plan audit completed normally121178ms: CONDITIONAL GO, Q-GOAL/Q-LEAN
accept ONE bounded field-prior attempt. Five conditions are incorporated:
pre-submission sizing, legal-branch identity before optimization, invalid draws
count as failures, budget-short runs are inconclusive, and no further field
training after failure/inconclusive until native-data member identifiability/
learnability is addressed. Disposition BUNDLE_C_DIFFUSION_AUDIT.md. The member
q_S, selection, global environment and efficient actual-data inference remain
unimplemented; neither morphology nor a normalized diffusion sampler solves
them. No repeated closed member diagnostics or direct-IC fallback. DESIGN
complete; proposed1-GPU/4-CPU/48GiB/24h,30,000-update implementation awaits
user approval at the next bundle boundary. No numerical job submitted.
User now approves this implementation and single Slurm diffusion experiment.
Execution record BUNDLE_C_DIFFUSION_RUN.md; source/model config and static
sizing are implemented.34,095,557 parameters; concrete bounded-cache sizing
reduces host request from provisional48GiB to15GiB (12GiB estimated peak+20%,
rounded), within the approved envelope. One allocation runs8 focused tests,
native identity,32 real largest-scale operational updates, then the same fit
and fixed evaluation. Only the8 original fine draws are comparable here; the
old8 environment draws are explicitly out of scope, not declared passing.
No new audit, q_S learner, observed posterior or automatic follow-up launched.
Implementation341b419 pushed; Slurm338402 submitted2026-09-10 09:03:02 KST.
Initial state PENDING(Priority), no allocation/tests/training yet. One job
will execute the approved test/fit/evaluation sequence when resources arrive.
Actual run state/results belong in BUNDLE_C_DIFFUSION_RUN.md; submission is
not a numerical pass or scientific result. No separate monitoring daemon.
Previous B delivery: [BUNDLE_B_RUN.md](BUNDLE_B_RUN.md).
Update2026-09-10 18:26 KST:338402 COMPLETED/exit0 after9h02m57s;
8 tests passed and30k updates finished, but all8 retained draws are
GENERATION/SUPPORT FAILURES, not valid fields with measured bad morphology.
All12 rollout cases fail first refinement; all native-parent scales fail too.
Continuous loss1.000238 is consistent with a zero predictor, not proof of
its cause. No fine-field or observed-LG promotion. User 'next proceed'
authorizes ONE20min Slurm frozen-checkpoint diagnosis, zero optimizer steps.
Fable5 CONDITIONAL GO; plan and incorporated conditions:
`BUNDLE_C_DIFFUSION_DIAGNOSIS{,_AUDIT}.md`. No verifiable original step0
exists: do not trust seed reconstruction as proof of weight updates.
Check actual normalization, simple denoising baselines, raw/EMA gradients
and unchanged failed reverse chain. No further field training before
member learnability is addressed and a new concrete plan is approved.
Frozen diagnostic implementation3240750 pushed; Slurm338746 submitted
2026-09-10 18:41:37 KST, initial PENDING(Priority), no numerical result yet.
1 GPU/2 CPUs/6GiB host/20min; no automatic additional training or bundle.

338746 completed2026-09-10 18:42:44 KST in36s, zero optimizer steps. On both
native tested scales raw/EMA epsilon RMS~.005, MSE~1 and near-zero correlation;
at t100 the algebraic reference MSE~2.4e-7. Reverse latent RMS rises1->2032 and
fails fraction support. Normalization agrees; continuous gradients are nonzero
and checkpoint/plain backward agrees. Failed denoising is established, but its
unique architectural/optimization cause is not. No new morphology or actual LG
product. Frozen diagnosis CLOSED, not another monitoring/diagnostic series.
User requested next repair PROPOSAL2026-09-11; `BUNDLE_C_REPAIR_PROPOSAL.md`
is design only, with Fable5 plan review. No new implementation/learning job
authorized or submitted. Next bundle requires user approval.
Fable5 repair-plan review returned CONDITIONAL GO111090ms. Current proposed
implementation specification is `BUNDLE_C_REPAIR_DISPOSITION.md`; submitted
plan preserved separately. Next proposed bundle is ONE native field-only
MW/M31/M33/remainder mass-allocation learner,13/3 development fixtures,
<=2000 updates/90min Slurm. No further total-field-prior learning in that job.
Driver corrects the proposed class-balanced CE to per-field/per-role normalized
map-L1 to avoid biasing small-member mass fractions. Pilot is not a calibrated
member posterior or a member-velocity model. Stable v-prediction and separated
category gradients remain a deferred repair hypothesis, not code or a job.
Await user approval; do not revive closed peak/proxy/budget tests or launch a
long density fit automatically. A pass leads to joint-member/denoiser design.
User approved the single native member-mass implementation/pilot. Source and
fixed choices are recorded in `BUNDLE_C_MEMBER_MASS_RUN.md`. One530804-parameter
U-Net,13/3 fixed fixtures, equal-role normalized map-L1, <=2000 updates/70min
within one90min Slurm job,1GPU/2CPU/6GiB. Three focused/reused tests run in the
same allocation; no new audit ladder. Mass-readout only, no member velocities,
observed fine field, diffusion repair/training or automatic next bundle.
Implementationee1c8e7 pushed; Slurm341713 submitted2026-09-11 14:57:49 KST,
initial PENDING(Priority), no allocation or numerical test/fit result yet.
Execution record `BUNDLE_C_MEMBER_MASS_RUN.md`; no separate polling daemon.

Previous diagnostic delivery: [BUNDLE_A_RUN.md](BUNDLE_A_RUN.md).

Completed A history, not instructions to rerun: the four-fit comparison
failed the frozen quantile-prior adoption rule. That repair series closed.
The original24-nuisance PM-calibrated model then supplied an actual-data
model-stress diagnostic, followed by the approved observation-model correction
and length check below. None scientifically promotes the baseline.

Actual-data preflight333862 passed; fit333872 and aggregation333990 completed.
The actual-data result is NO_GO_SAMPLER_NOT_VALIDATED (max Rhat1.325,
minimum bulk ESS11.1), not a usable posterior. Saved-chain diagnostic334240
completed without refitting. It confirms field/H0 coupling and identifies a
physical-Mpc versus Mpc/h magnitude convention mismatch in the imported
2M++ population/selection setup. See
[ACTUAL_PREVIEW_DIAGNOSIS.md](ACTUAL_PREVIEW_DIAGNOSIS.md).
The user subsequently approved the focused observation-model correction,
field-aware sampler adjustment and ONE corrected actual-data refit.
Active correction plan: `config/cf4_actual_data_corrected_v2.json`, documented
in [ACTUAL_CORRECTION_RUN.md](ACTUAL_CORRECTION_RUN.md). Preserve the old fit.
The disjoint sample is retained; calibrate its survival from a separate20%
parent mark sample, retain the old20% heldout, and use60% for field fitting.
This supersedes the earlier one-fit cap only for this approved refit, not for
further prior-family experiments. No Bundle B execution is active.

Corrected fit334345 completed in22m22s. Max Rhat1.034 passes, but H0 and
its field-dependent conditional mean have bulk ESS97.1/85.1 (<100); retain
NO_GO_SAMPLER_NOT_VALIDATED. The user now approved ONE sampling-length check:
`config/cf4_actual_data_longer_v3.json`, four fresh chains with2048 samples
each,512 warmup, unchanged model/data/gates. Reuse frozen actual inputs,
preserve old chains, and do not claim exact continuation or pool the runs.
No automatic further extension or Bundle B launch.

Final length-check job334358 completed2026-09-07 22:24:15 KST in36m33s;
aggregation334359 completed22:24:23. All predefined sampler gates pass:
max Rhat1.03183, min bulk ESS113.59, min tail ESS316.88, divergences0.
Heldout count/velocity moment-residual SDs .99628/1.11173 are diagnostic,
not scientific calibration. Product remains12 cMpc/h and contains no
resolved LG. Stop adding samples. The user accepted preparing the next
bundle: actual environment comparison, explicit LG observational likelihood
contract, and a bounded coarse z=0→IC→forward-z=0 mock bridge. No B jobs yet.

Z4–Z11 were N32, 12 cMpc/h development experiments, not actual-data maps.
Z8's amplitude interpretation was erroneous and has been withdrawn. Z10
fixed-field and Z11 prior-compatible controls recover the injected tracer
curvature; Z9 free-field PM cases do not. This supports a field/observation
model mismatch but neither isolates its sole cause nor proves a quantile
transform will fix it. Z11 completed all four fits; its results are at
`/gpfs/kjhan/CF4/z0_density/z11_prior_control_v1/comparison.json`.

The following is the historical Bundle A scope, now completed at diagnostic
level; it does not authorize another comparison or actual-data fit:

1. Finish ONE training-only non-Gaussian-prior comparison against the saved
   native-PM controls, with at most four new mock fits. Reuse existing
   simulations, likelihood, sampler and reports. Require spatial field
   recovery and held-out predictions, not just a nicer density histogram or
   recovered nuisance coefficient. Exact cases/resources/decision rule:
   `config/cf4_bundle_a_prior_to_data_v1.json`.
2. Driver judges the comparison. No automatic additional prior families,
   nuisance terms or new simulations if it fails. State the model limitation
   and decide its suitability for a preliminary observation-space diagnosis.
3. Within A, implement and run at most one actual-data preliminary four-chain
   fit after its input semantics and chosen model are recorded. Use actual
   CF4 velocity data and prepared 2M++ counts, preserving selection, errors,
   overlap treatment and heldout separation. No pseudo-truth or truth-based
   accuracy/coverage numbers for the real universe. If model suitability
   fails, the product is a model-stress diagnostic, not a validated posterior.
4. Deliver actual-data mean/sample density maps, mean velocity and posterior
   velocity uncertainty, predictive residuals and limitations. Distinguish
   uncertainty in the mean field from physical velocity dispersion/FoG;
   current population FoG parameters are not a resolved sigma_v(x) field.
   Close A with a concrete Bundle B design and request approval.

The real-data preview remains 12 cMpc/h and cannot certify the target LG
scale. Independent galaxy/selection mocks and physically relevant validation
are required before scientific promotion, but not an unbounded prerequisite
to viewing a clearly labelled diagnostic map. Reused development fields and
shared generator/inference code are never independent validation.

## Working rules

- Prefer substantive scientific outputs over more generic validation code.
  Reuse tests; add only checks necessary for the changed computation.
- A clean sampler is necessary, not scientific success. Separate posterior
  mean smoothing, individual-draw structure, uncertainty and phase recovery.
- Syntax is a Slurm login server. Numerical jobs use Slurm; no manual syn101
  or login-node calculations. Current GPU partitions: a40,a100,h100,h200;
  exclude syn06 for these fits. Request estimated peak memory plus ~20%.
- GPFS is ordinary shared storage. Read/write scoped project artifacts;
  do not implement storage/inode/renameat2 probes or process-scan monitoring.
- Use fixed job IDs and final artifacts for bounded checks; no pgrep loops.
- The driver plans, implements, runs, evaluates and commits. New user policy
  reinstates plan/design audits even for an Astra driver: Fable5 primary;
  Astra backup if the Fable audit invocation fails or produces no usable audit.
  Every plan audit must explicitly answer Q-GOAL and Q-LEAN plus feasibility
  and essential/deferred scope. Do not turn these two questions into a new
  gate framework or per-step audit series. Prior Astra closure-audit waiver
  remains unless changed by the user; this instruction concerns plan audits.
  Explicitly audit the MW/M31/M33 identification plan in every bundle; require
  no oracle component access on new fields and honest unresolved-M33 handling.
- Commit/push coherent changes. Preserve unrelated user work and all failed
  scientific results. Keep run summaries current; do not confuse submission,
  sampler pass, scientific acceptance and final-goal completion.
- At every bundle boundary report the result, goal contribution, unresolved
  risk and next deliverable. When approval is needed say **승인해주세요**.
