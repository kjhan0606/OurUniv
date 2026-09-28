# Fitted-state descent: evidence and driver decision

R1 → **R2 in progress** → R3 same-state LG → R4 → R5 zoom IC.
R2 is not complete. No heldout observation was scored, no posterior promoted.

## Measurements

407840 (5m11s) tested the v2 accepted state, not an unconditional fixture.
Normalized descent reverse derivative −403325.84 versus objective finite
differences −32.82, −4173.25, +105005.17 at eps .01, 1e-4, 1e-6. Both sides
increased the objective at these scales. This initially leaves derivative
error, very strong curvature, and observation discontinuities unresolved.

407881 (3m59s;7.60GiB Python peak) extends the SAME direction/state:

| eps | objective FD | observation VJP with field FD | relative reverse/FD difference |
| --- | ---: | ---: | ---: |
| 1e-7 | −371999.54 | −375423.16 | .07767 |
| 1e-8 | −403443.35 | −403442.77 | .0002913 |
| 1e-9 | −403378.57 | −403378.26 | .0001307 |
| 1e-10 | −403318.55 | −403316.67 | .00001807 |

At1e-7 and below, the plus direction actually decreases the objective.
Occupancy of native cells does not change between the two perturbed fields.
Nearly all the signed velocity-tangent contribution comes from native
rho>=1e-3; bins0<rho<1e-6 contribute~1.8e-7 and1e-6<=rho<1e-3 about−.0012,
versus total~−400000. This signed attribution does not exclude cancellation,
conditioning in count-cell moments, or other directions. It does not support
blaming nearly-empty native nodes as the demonstrated primary cause.

407880 was cancelled immediately to correct Slurm comma-separated export
parsing, replaced by407881 with the epsilon list inherited through ALL.
No output was deleted or treated as a scientific failure.

## External advice and independent driver assessment

Read-only `claude --model claude-fable-5 --permission-mode plan` returned
CONDITIONAL PASS; Q-GOAL and Q-LEAN support same-state diagnosis, not another
blind long fit. No edits, jobs or simulations were delegated. Advice to
separate field and observation tangents and avoid conservation-only claims
is adopted. Proposed direct mass/momentum wiring remains a conditional
alternative, not an evidence-backed fix adopted here.

Specific unsupported inferences are NOT adopted:

- The score reproduction discrepancy6e-10 was **absolute**, not relative;
  the claimed6e-4 objective noise floor and dismissal of eps<=1e-9 do not
  follow. The measured small-epsilon results are informative.
- Equal maximum neighbor counts do not establish identical candidate sets.
- Monotone approach over three eps is suggestive, not proof of a correct
  derivative. The extra measured agreement is stronger local evidence.
- An IC coordinate cannot be identified with the same Eulerian grid cell
  after nonlinear evolution; the proposed location interpretation is invalid.
- A transpose test using mutually derived AD rules cannot by itself prove
  forward correctness. PMWD nbody uses custom VJP, with no direct JVP path.
- Conservation alone does not exclude Jacobian conditioning, but neither
  does a rho*(momentum/rho) intermediate prove the observed discrepancy;
  its algebraic cancellation and fitted-state evidence must be checked.

## Narrow continuation (unchanged physical target)

The existing12-halving search cannot reach the locally valid scale from its
IC RMS .1 trial. There is no evidence here requiring a PM-adjoint patch or a
likelihood floor. Reuse the accepted state with initial coordinate-norm cap
1e-7 (measured descending), grow after successful steps, use safeguarded
quadratic backtracking up to40 trials, and evaluate gradients only at accepted
score-only candidates. Verify score-only and value/gradient agree before
acceptance. This changes optimizer steps, NOT prior support, survey cuts,
likelihood, or fields. Three small optimizer regressions precede the retry.

One20-iteration/25min application/30min Slurm retry, one typedH100,2CPU12GiB
(measured9.52GiB peak plus~20% margin). Initial random-direction derivative
check uses eps1e-8; larger eps at this conditioned state were not reliable.
Report progress and cost; do not keep spending allocations on negligible
descent. No Laplace uncertainty without a usable stationary solution.

Q-GOAL: recover an actual data-conditioned density/velocity state with the
latent-IC connection. Q-LEAN: same target/operator/cache, no new simulator,
new observational data, or validation framework. MW/M31 remain ambiguous,
M33 unresolved on this NEW state; their observables must constrain it in R3,
never native-truth-selected candidates. Surroundings1–2cMpc/h and LG<=.3
remain the final resolutions; N128/384 is only3cMpc/h development. No email.

## Retry disposition and next cause test

407886 passed all3 optimizer tests and the actual-state random-direction
check (prior-subtracted relative difference .0001309). Score-only and full
objectives agree. Accepted decreases then became negligible: total decrease
about.122 from1,039,923.9003, recent decreases~1e-6, maximum gradient still
~11,770. Other tiny candidate steps repeatedly jump upward by~.525. The
driver cancelled this allocation early for scientific stagnation, NOT OOM,
test failure or convergence. Preserve its accepted checkpoints and trace;
no final-state snapshot was promised after forced cancellation. Readout407887
is afterany, so missing final state must not be interpreted as a map success.

407901 is the queued15min/H1002CPU12GiB cause test on the SAME v2 accepted
state used for the derivative comparison. It counts, separately for every
GH15 node, source radial-cut crossings under eps1e-6 and1e-8, and reports
which have nonzero angular exposure. No fitting, changed mask, smoothing,
heldout score or new simulation archive. Zero crossings would exclude this
particular discrete-cut explanation at those trials; nonzero counts alone
would not quantify their contribution to the objective jump. Any subsequent
integral repair must preserve the actual observational cut and be supported
by a count-factor comparison, not an arbitrary likelihood floor.

407901 completed3m29s: at eps1e-6 one sky-active source-node crosses the
radial cut on the plus side and two on the minus side; at1e-8 there are zero
crossings. This identifies actual finite-GH selection switches, but does not
by itself attribute the entire count jump or all field stiffness to them.
407886 was CANCELLED by the driver at14m19s;10 accepted steps, total objective
decrease.121594473, final accepted max gradient11770.6683. Its readout407887
completed4s, explicitly no final map. Its historical status STARTED is a stale
application marker, not a running job; future readouts now label that case
INCOMPLETE_RUN_NO_FINAL_STATE without inventing the scheduler exit cause.

## Boundary-fitted integral repair candidate

`src/cf4_r2_shell_cdf_count.py` integrates the same source-selected LF/K/TSC
integrand over ray intersections with periodic observer shells5–180. Gaussian
probability-coordinate quadrature removes the hard GH radial-selection atoms.
27 disjoint observer images retain wrapped aliases; both signed LOS branches
and the inner5 cut are included. The existing8*sigma_radius<box/2 live-support
bound suffices for those images. Tails beyond8sigma are explicitly omitted
(unconditional Gaussian mass<1.3e-15); probabilities are NOT renormalized to
the selected shell. Positive tails use survival probabilities. Inactive
square roots/quantiles are guarded before nonlinear operations, not hidden
with a likelihood floor. Only identically zero interval weights are skipped.
Integrated Schechter interval fractions are continuous at moving K-bin
intersections; no apparent-K point indicator is newly introduced.

Fable conceptual follow-up returned CONDITIONAL PASS, Q-GOAL/Q-LEAN aligned.
Adopt normalization, tail, signed-image, finite-gradient and numerical-error
controls. Reject its assumed0.3125cMpc/h TSC scale: the current count grid is
3cMpc/h. Also reject globally parameter-independent geometry/smoothness:
post-wrap rhat can depend on velocity; tangencies and other field nonlinearities
remain. Do not claim this fixes every fitted-state pathology. The driver uses
the explicit probability-transform derivation; unverified literature details
suggested by the adviser are not recorded as established citations.

Four focused controls precede any full-state use: analytic cut probability
and derivative, signed inner-exclusion mass, periodic-image tail mass, and
the full LF/K/TSC source integral at16/32 nodes with velocity finite difference.
One H1002CPU6GiB/15min allocation (estimated host<=5GiB with20% margin).
This candidate is not yet wired into a fit or declared production-ready.

H200407913 completed2m28s,4/4 tests pass (126.6s test time,2.44GiB RSS).
Next is one saved-field comparison, not another gravity run: recompute the
GH15 training-count factor on the v2 final state, compare CDF16/CDF32 fields,
their training-exposure L1 and occupied-cell log errors, and a CDF16 coherent
velocity-scale derivative against finite difference. No heldout counts or
marks loaded. Preserve the original sigma/LF/bias,5–180 cut and v6 exposure.
H2002CPU12GiB/25min; estimated host<=10GiB includes prior9.52GiB actual-state
peak and new compilation buffers,~20% requested margin. Keep three count
fields only in memory and write compact JSON; no new raw simulation output.

Two quadrature orders are a numerical comparison, not an exact oracle or
calibrated posterior. A usable engineering candidate requires the original
GH score reproduced within1e-7, exposure-L1 difference<.001, count-score
difference<.5 and scalar derivative discrepancy<.001; failing any criterion
does not authorize fitting or loosening it. Even passing does not establish
all-IC derivative conditioning, survey calibration or R2 completion.

Saved-field407919 failed before CDF evaluation on float32/float64 scan carry;
f92530e repairs accumulator promotion and adds mixed-precision coordinates.
Retry407922 completed7m16s, four tests passed first. GH15 reproduces the old
count score to~2e-9. CDF16/CDF32 exposure-L1 difference is1.51e-4, but count
score difference−4.28487 exceeds.5 and maximum occupied log-intensity
difference1.9963 is large. The CDF16 velocity-scale derivative is internally
consistent (relative difference7.02e-7). This numerical candidate therefore
does NOT qualify for fitting; good gradients alone do not mean an accurate
integral. Host Python peak5.05GiB. No PM evolution, optimization or heldout.

Narrow improvement: stratify each shell interval in physical signed distance
before its probability-coordinate quadrature. Each subinterval retains its
exact unconditional Gaussian probability; no low-weight interval is dropped.
A compact five-sigma TSC-tail reference explicitly probes what a single
global probability rule can miss despite small total L1 error. Compare4-node
rules with16 versus32 physical subintervals on the SAME saved state, preserving
the previously stated.5 score and.001 L1/derivative criteria. Five small tests
precede it in one H2002CPU12GiB25min allocation. Expected host<=10GiB including
the measured5.05GiB and larger node buffers;12GiB includes20% margin. This is
a finite numerical repair attempt, not a search over unrelated models or a
longer field fit. M33 and all R2 calibration/uncertainty limits remain open.

407946 stopped in the new toy-tail test, before any full-state evaluation:
16 physical subintervals with8 within-interval nodes gave .7519% relative
error, exceeding the unchanged .1% reference requirement. Refine that
reference to32 physical intervals; do not relabel the failed result a pass.
407950 then passed5/5 tests. The five-sigma TSC integral is1.3563983584e-5;
globalCDF16 misses it entirely, 4x16 gives1.1842245372e-5, and the8x32
reference gives1.3563961971e-5 (relative1.59e-6). Thus coarse-tail accuracy is
explicitly not certified just because it is now nonzero.

Its saved-field4x16/4x32 comparison reduces total-score difference to.1727
and exposure L1 to1.31e-5, but maximum occupied log difference remains.04547.
Prefer verifying4x32 against8x32 before fitting, rather than using the cheaper
4x16 solely because the aggregate criteria pass. No new gravity/heldout.

Cost repair for this same integral: monotonic Schechter survival lets the
LF/K transfer evaluate six varying boundary arrays once, then reuse min/max
intersections; the old definition evaluated60 gamma arrays. This is algebraic
reuse, not a new LF approximation. Retain the original expression as a small
regression reference. Check values and all four parameter derivatives at
three LF shapes, plus equality-boundary subgradients. The next single
H2002CPU12GiB25min job runs those2 tests, the5 integration tests, then the
saved-field4x32/8x32 check. Reproduce the previous4x32 count score before
accepting the optimization; only then assess integration error and derivative.
Q-GOAL: usable same-state current-field inference; Q-LEAN: reuse analytic
boundaries rather than raising simulation resolution or adding a solver.
MW/M31 roles remain ambiguous, M33 unresolved; their observables constrain
the same NEW field in R3, not native-truth-selected components.

407950 completed the scalar adjoint (relative3.53e-6),13m02s application,
6.80GiB Python peak. Pending407971 was replaced by407975 before execution to
preserve the original strict empty-LF-interval derivative convention.
407975 passed2 LF and5 integral tests, and reproduces old GH15/4x32 scores
exactly.4x32/8x32 score delta is−.00154245, exposure L1=1.494e-6, max occupied
log delta=.00107253. The reused-boundary4x32 forward took47.7s including
compilation versus206.7s previously (not an isolated warm-kernel benchmark).

However, its4x32 derivative FAILED on GPU workspace allocation96.92GiB,
not host RAM or a failed value-accuracy check. Slurm elapsed6m29s. The next
repair streams one physical interval at a time, checkpointing each interval,
rather than constructing all image/interval/node/source arrays on the reverse
path. It preserves quadrature nodes, weights and accumulation order. Repeat
the saved-field score and derivative check; inspect compiled device memory
with20% margin before attempting the derivative allocation. This is a direct
response to measured OOM, not a filesystem/resource-scanning framework.

The combined count/singleton target now has an explicit optional shell-CDF
branch (default historical GH unchanged), awaiting its nonempty-count
reference test and the full-state derivative result. The optimizer can reuse
its already-checked initial gradient and perform its directional finite
differences with score-only calls. No fit has been launched on the new branch.

When numerics are usable, a small-perturbation initial optimizer state is
available explicitly, not as a new prior: white_IC=.01 times the predeclared
unranked seed draw. It avoids an exactly uniform CIC-knot starting point.
The Gaussian IC prior, cosmological power, likelihood and parameter domain
remain unchanged. This is NOT a cosmological prior draw or posterior sample;
readout labels distinguish it. Do not continue the already-stalled v2/v3
state merely to spend more iterations. A fresh bounded fit still requires
the actual new initial-state directional check; no posterior/IC promotion.

Streaming408010 COMPLETED7m02s,2 LF+5 quadrature tests pass. Saved GH15 and
CDF4x32 scores reproduce exactly; CDF4x32 scalar adjoint discrepancy2.83e-6.
Compiled derivative temporary memory14.01GiB plus.80GiB current device use,
against104.85GiB effective allocator limit, replaces the failed96.92GiB
single allocation. Coupled-target408011 passed4 optimizer tests and the
nonempty-count GH/CDF+unchanged-FP/prior/heldout wiring test (139.2s).

## New bounded actual-data fit

Use the explicit shell-CDF4x32 count branch with429 strict singleton FP marks
and47,121 training counts on the frozen v6 graph. Same384 box/N128 grid,
cosmology, prior, LF/bias/width nuisance law, shared FP zero and live support.
Start white_IC at.01 times seed2026092702, white nuisance coordinates zero,
without restarting/ranking prior fitted ICs. This is an optimizer start only;
it is neither a prior realization nor posterior uncertainty. No physical
parameter bounds, likelihood floor, new smoothing or heldout score.

One H2002CPU12GiB/45min Slurm allocation:20 accepted iterations maximum,
35min application cap, adaptive initial coordinate-step norm1.0. The actual
start must pass the prior-subtracted directional check at eps2e-5. A combined
count+FP derivative compile-memory check now precedes allocation, retaining
20% device margin; host estimate<=10GiB with12GiB requested. Save only initial,
last accepted and final fields/parameters, below.5GiB; no RAMSES dumps.

Judge count and FP changes, gradient, meaningful field response and cost.
Do not confuse a budget stop or a low MAP score with a posterior. If there is
real progress, continue within bounded R2 work; if another pathology occurs,
retain evidence and fix it instead of extending a blind run. Calibration,
shared source covariance, posterior uncertainty and untouched heldout
prediction remain required before R2 closure or N256. MW/M31 roles stay
ambiguous and M33 unresolved at this coarse stage; R3 must identify them
from the SAME new state and attach their actual observables, not truth IDs.

Live408032 has passed its initial observation-only directional check
(absolute discrepancy4.39e-5, normalized by max(1,|derivatives|)). Seven
accepted updates reduce objective231875.999 to203571.746 and maximum gradient
756.69 to546.91. Counts improve while FP score is lower than the initial
near-uniform start; do not equate joint-objective decrease with improvement
of every observation or a calibrated map.

Reuse the existing saved-field integral check once on this run's terminal
field: CDF4x32 versus8x32 and the existing scalar derivative, without PM
evolution, fitting or heldout measurements. An explicit state-path option
replaces the old hardcoded fitted state; the old-score reproduction assertion
still applies ONLY to its original reference field. Submit afterok408032,
H2002CPU12GiB/15min (previous actual check7m02s, host3.84GiB). This checks
that the numerical accuracy used for inference survives the field update;
it is not another model-selection ladder or R2 completion gate framework.

408032 COMPLETED35m59s on the application budget, not stationarity:17 accepted
updates,18 full evaluations. Objective231875.999→175535.453, count
−231771.137→−174395.939, FP.0859→−50.9406, maximum gradient756.69→129.91.
IC quadratic penalty increased from104.948 to1088.293;
the improvement is not solely prior shrinkage as in
the failed older run. Mean density remains1; white-IC mean-square.000100086→
.001037877. White rate coordinate.459742 implies2.508x rate normalization,
but the field also changes (density range .947–1.169→.0093–497.55).
408033 created the actual density/velocity illustration. Visible band-like
features and large flows are not proof of physical filaments, nor evidence
of a specific defect without a matched control. No named object is identified.

The next same-target continuation, contingent on terminal integral evidence,
is bounded to32 updates/70min application, H2002CPU12GiB/80min Slurm. Restart
the accepted coordinates with fresh L-BFGS history and initial step-norm cap16;
this is not exact history continuation or a new prior. Keep4x32, all429/47121
training inputs and all observational priors fixed. Repeat the actual-start
directional check, no heldout. Save native-node physical particle velocity
variance/validity in the existing final forward readout, not by adding a
second simulation. Existing second-moment conservation fixture checks its
mass*(variance+mean^2) identity; optimizer tests also precede the run. Budget
metadata is now recorded before computation rather than after the first check.
Storage remains below.5GiB/run; no new snapshots or high-resolution launch.
Q-GOAL: usable same-state present-field inference; Q-LEAN: refine a descending
fit with existing code, not another diagnostic ladder. R2 scientific limits
remain, MW/M31 roles are ambiguous, M33 unresolved, and R3 must constrain
these components from the SAME new field without truth-ID seeding.

408078 completed its terminal-state comparison:4x32 reproduces the fit
count score to3e-11,4x32/8x32 delta−.000862329, exposure L1=1.40715e-6,
max occupied log delta=.00104405. Scalar velocity derivative discrepancy
1.934e-6. Python peak4.10GiB, calculation286.2s plus seven small tests.
No new PM evolution or heldout score. These results support the bounded
same-target continuation above, not calibration or posterior certification.

Continuation submitted as408084/sourceece86ee, output
`/gpfs/kjhan/CF4/z0_density/r2_v6_shell_map_v2`; afterany readout408085.
The source-state mean density, rate, count/FP components and gradient remain
the baseline for its evaluation. No repeated heldout peek or R2 promotion.

For the v2 terminal field, reuse the same saved-state integral readout once,
after successful process completion. Its already-computed intensity also
supplies a 2M++ training observed/predicted radial-count illustration. Radius
is the observed voxel-centre radius, binned at12cMpc/h for display only; this
is not true-distance calibration, a change of reconstruction resolution,
posterior predictive uncertainty or heldout validation. No extra forward
evolution or fitted parameters are needed for the figure.

One bounded cost check408099/sourceec46f05 did NOT justify a deposition
rewrite. Six-population contiguous-axis TSC agrees with six separate scatters
to3.56e-15 in values and5.33e-15 in tested gradients. On2,097,152 synthetic
non-lattice sources, separate/batched forward medians are.00746/.01717s and
derivative medians.01626/.01033s. The small127-source case looks faster but
does not predict production-size forward cost. These are isolated kernels,
not end-to-end fits; no speedup of the actual likelihood is established.
Keep the existing production scatter unchanged; do not start a tuning sweep.
The fixture contains no survey calibration, PM evolution or heldout data.

Driver frame-path review: the 2M++ parent coordinates are built from Vcmb in
`scripts/cf4_r2_common_catalogue.py`; FP source `zgroup` is `zcmb_group` in
`scripts/cf4_r2_sdss_fp_source_link.py`, and the hierarchical geometry builder
interpolates that `zgroup` into `dz_row` in
`scripts/cf4_r2_fp_group_marginal_control.py`. The live fit supplies the PM
peculiar velocity directly. This is consistent with a CMB-rest reference
observer, not an uncorrected heliocentric/LG redshift target. The low-level
kernel's older `observer_subtracted` convention name is not evidence that a
second subtraction of the physical local observer velocity is required.
No numerical change or extra simulation follows from that name alone;
these code paths do not separately certify the original catalogue transforms.

## Second fit and one bounded next extension

408084/sourceece86ee COMPLETED57m06s (application3403.8s),32 accepted steps,
on the iteration limit, NOT convergence. Objective175535.453→159462.615;
count−174395.939→−156978.834; FP−50.9406→−43.9472. Gradient_inf129.91→73.96,
with an intermediate low~31 and a late rebound; do not report monotonic
gradient convergence. Last accepted gain262.91 is far from407886 stagnation.
Python peak5.91GiB.408085 saved the final image and physical particle velocity
variance; mass-weighted per-axis dispersion is95.73,96.48,100.31km/s. It is not
posterior uncertainty or a calibration of the tracer width100.13km/s.

The current terminal4x32 count readout reproduces−156978.8338641434 and
predicts50818.597368226 galaxies versus47121 observed. This identifies the
largest gradient: with rate white coordinate t=.5784511721 and numerical
coordinate x=100t, d(-log target)/dx=[2(E−N)+t]/100=73.95773188, exactly
the recorded maximum. This is evidence about the gradient's source, not
proof that all other optimization or science limitations are solved.

Before the next joint refinement, make ONE scalar conditional-rate MAP
initialization. Solve2[E exp(2(t_new−t_old))−N]+t_new=0, retaining the Gaussian
prior. All count intensities share this amplitude; it cancels from the
conditional FP factor. Check the predicted full-objective change against
the actual joint implementation and check its rate derivative before using
that accepted state. This is neither rate marginalization nor profiling
throughout the fit, no likelihood reweighting, and no altered field/prior.
The normal joint optimization, including that rate variable, follows.

Following `CF4_R2_FIRST_FIELD_ADVICE_20260928.md`, the prospective budget is
one <=3 H200 GPU-hour envelope: joint continuation up to64 updates/120min
application (130min Slurm), then three-direction curvature feasibility
(25min application/30min Slurm), plus5min terminal readout. Total requested
GPU allocation limits165min, not an unbounded series of restarts. Same
N128 data/target,2CPU12GiB per numerical job (host estimate<=10GiB), no
heldout, N256, HMC or posterior promotion. Initial step-norm cap16 remains
an optimizer control only. The actual restart adjoint must still pass.

Curvature uses central finite differences of full gradients at eps1e-3 and
3e-4, along the scaled rate coordinate, one IC fundamental mode and one
predeclared normalized random IC direction. The scalar rate curvature has
the analytic reference(4E+1)/10000. Save step-size dependence, signs, norms
and cost; never infer positive definiteness, covariance or coverage from
three directions. No unsupported PMWD forward-JVP promise. A quadratic
fixture preserves both positive and negative curvature. Per-coordinate
gradient summaries distinguish IC and nuisance contributions with negligible
storage, rather than adding a new diagnostics framework.

Q-GOAL: reach a usable actual-field state and decide how its uncertainty can
be computed. Q-LEAN: one evidence-supported extension and a small cost/sign
check, not indefinite MAP repetition or a mock/tuning ladder. MW/M31 remain
ambiguous and M33 unresolved; the same NEW state must later supply their
candidates and receive their actual observational constraints, without truth
IDs. The broader calibration/covariance/R2 resolution limits remain open.

408093 completed the v2 terminal readout:4x32/8x32 score delta.00209658,
exposure L1=1.43687e-6,max occupied log delta=.00040595, scalar derivative
discrepancy1.5645e-6. Its training radial-count image shows excess at middle
distances and a deficit at the far end; a single rate correction cannot
remove this shape discrepancy. It remains training evidence only.

The next envelope is submitted on source474939c:408119 joint refinement
(`r2_v6_shell_map_v3`),408120 afterany readout,408121 afterok curvature
(`r2_v6_shell_map_v3_curvature_v1`). The first job must verify the full-target
rate update and initial adjoint. Six small optimizer/rate/curvature tests and
one existing particle-moment test precede each numerical job. The final
curvature readout is not an automatic science approval or posterior product.

Terminal numerical/readout addition:408150/source7f2a928 runs the EXISTING
saved-field checker afterok408119 on the v3 final state, output
`r2_shell_cdf_field_check_v9`. It compares4x32/8x32 integration and the scalar
velocity derivative and plots training radial counts from that same readout.
No new PM evolution, fit, heldout score or implementation. H2002CPU6GiB/10min;
the previous checker peak4.11GiB plus20% fits this host request. The envelope's
requested GPU allocation caps now total175min, still below3GPU-hours.
Reason: the substantially changed terminal field needs its own numerical
accuracy and radial-residual readout; the v2 comparison is not a v3 result.

### Bounded v3 fit finished, not stationary

408119 COMPLETED/exit0 in1h51m08s (Python6641.32s,peak5.91GiB),64 accepted
joint updates,66 full-gradient evaluations,70 score-only trials. Stop reason
is **iteration limit**, not convergence. Objective159462.614936→151825.923500;
count−156978.833864→−146150.539323; conditional FP−43.947229→−25.563497.
The initial scalar-rate change−137.92777589 matched the analytic prediction
to~1.5e-12; its residual scaled derivative was−9.5e-13. It was a one-time
MAP initialization, not rate marginalization. The restart adjoint passed
with normalized discrepancy2.56e-5.

Final maximum gradient42.6823 is still optimizer coordinate2097154, i.e.
white tracer index2 (bias for true K interval[-25,-23.6666666666667]). IC
maximum3.65466. These coordinate-scaled values are not directly comparable
physical uncertainties. Last-step objective gain53.6857; real descent, but
neither stationarity nor permission for another identical restart.

The FP shared white zero moved1.66710→4.08892, corresponding to
.00666839→.01635569dex with the fixed .004-dex prior scale. Thus the improved
FP score must NOT be attributed to the density/velocity field alone. A
4.09-prior-SD fitted zero is a model/calibration tension to investigate, not
proof of an implementation error or a license to widen that prior. Its
remaining objective derivative is−1.81574 in white-zero coordinates.

408120 completed15s and produced the final comparison image. Native mean
density1,range0–252.521; white-IC mean-square.00537995, not a posterior draw
or a restored LCDM spectrum. Physical mass-weighted node dispersions are
100.75,100.26,108.67km/s, separately defined from tracer width99.729km/s
and posterior uncertainty. Visible coherent structures and bands remain
unvalidated; MW/M31/M33 are not identified. Curvature408121 and saved-field
integration/radial readout408150 remain to be assessed. Heldout is untouched.

408150 subsequently COMPLETED5m51s. It reproduces the v3 count score;
4x32/8x32 delta.00119886,exposureL1=1.46911e-6,max occupied log delta.00072355.
Scalar velocity derivative reverse528.2393764 versus FD528.2393831,
relative discrepancy1.27e-8,Python peak4.11GiB. Expected training count
47586.02 versus47121, but the shape mismatch persists: predicted/observed
4979.7/4417 at48–60cMpc/h and1366.6/1801 at168–180. It is not just overall
normalization. The image `r2_shell_cdf_field_check_v9/training_radial_counts.png`
is a training readout, not independent validation. No heldout was read.

408121 completed the three-direction finite-gradient curvature feasibility
check (13 full gradients;Python1077.82s,peak5.91GiB). Rate-axis curvature
19.0345070956/19.0345070945 matches analytic19.0345070942; vector change
3.90e-10. IC fundamental-x directional curvatures6036.486/5879.556 have
14.32% relative vector change. Fixed random-IC directional curvatures
1.165796/1.169114 look similar, but vector norms60.088/31.823 and relative
vector change1.02080 do NOT support a stable Hessian action at the tested
epsilons. Each two-gradient action costs~151s. No Hessian/covariance is
certified and no larger finite-difference Hessian calculation follows.
This scale dependence does not by itself identify an incorrect first derivative
or a unique cause. No negative direction was seen in these THREE probes;
that is not an SPD test. The unchanged-gradient checks remain valid within
their stated scope. Next conditional-block decision and Fable corrections:
`CF4_R2_NUISANCE_BLOCK_ADVICE_20260928.md`.
