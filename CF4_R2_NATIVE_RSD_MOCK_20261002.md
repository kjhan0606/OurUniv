# R2 native galaxy RSD/count calibration mock — 2026-10-02

R1 complete -> R2 current/incomplete -> R3 LG -> R4 precise validation -> R5
zoom IC. R2 must still deliver a validated actual384-box N256/1.5 z=0
density/velocity posterior, uncertainties and actual untouched prediction.

## Driver design and purpose

Previous source-support bundle410484/410505 supplies native K, positions,
velocities and a preserved1.5 total-matter field. It demonstrates relevant
luminosity support, but a native real-space beta fit cannot calibrate the
selected observed-K/RSD count operator. This bundle feeds native galaxy
locations/velocities/luminosities through the SAME radius, apparent/absolute
K and redshift-correction definitions as the active count law, then tests
that law's nine tracer coordinates on a known native field.

No galaxy mocks are drawn from the tested Poisson model: catalogue galaxies
come from the existing independent hydro simulation, including their actual
clustering and velocity residuals. The matter field is known calibration
input, not an inferred IC/current universe or a new gravity calculation.

## Finite source-window geometry

Native75-cMpc/h material cannot cover all six apparent-K/absolute-K populations
from a central observer. Split its x direction at37.5. Translate the whole
cube by(154.5,154.5,154.5), and add84 in x for the far half. The fixed observer
is(192,192,192) in the384 bookkeeping box. Matter cells and native galaxies
use the identical map; each native cell/galaxy occurs once. The near half
supplies close bright populations; the far half samples roughly84–133
cMpc/h and supplies the fainter apparent sample. Translation changes radial
projection but does not alter native local positions, luminosities, density
or velocity vectors inside each half. This explicitly defined source window
contains no galaxies outside those two volumes in either the mock or its
intensity model. It is not a physical384-cMpc/h simulated universe.

The full native75-box responses rho^beta have unit mean before the source
window mapping. The common rate is per3-cMpc/h cell; each1.5 source carries
1/8 of that volume. Native K_h=M_K-5log10(h), h=.6774, Om=.3089. Flat distance
tables use native matter/Lambda cosmology. Observer peculiar velocity is0.
RSD uses h/H0=.01, recomputed observed minimum-image radius, and the active
redshift-dependent K correction. Observed objects satisfy5<=r<180, K<=12.5,
and redshift-derived -25<=M_h<-21. NGP floor(position/3) matches catalogue
counts; the active model retains TSC. No row selection uses a fitted score.

Train/test are the negative/positive y sides with a12-cMpc/h central buffer,
applied after RSD to counts AND predicted intensities. Observed counts and
empty exposed cells enter once. No actual CF4/2M++ outcome is read. Earlier
native-source diagnostics have consumed this same universe, so this is a
development prediction check, not a pristine independent-validation claim.
The two native parts are also correlated, not independent universes.

## Fit, verification and outputs in one allocation

Four small observation/placement/count tests run first. Start beta=1,
alpha=-1,Mstar=-23.28,sigma_LOS=300km/s (predeclared, not truth-estimated).
Use the active nine standard-normal development tracer priors once.
An analytic training-only rate warm start preserves this target; confirm its
mean and rate derivative. A flattened compiled derivative must match direct
primal and one fixed finite-difference direction before optimization.
Bound L-BFGS at24 iterations,40 suggested/48 hard evaluations and75min
application time, including readouts. A stop at budget or optimizer failure
is not convergence; save the last accepted finite coordinates.

Fit GL2 source-volume/LOS4x8 on the training counts only. At the fixed final
point, report training/test population-radius means, full sparse count scores,
observed population migration and stellar-resolution counts. Compare source
GL4, LOS4x16 and diagnostic NGP without refitting. Save mock sparse counts,
initial/final predicted means and an actual population/radius PNG. These
are necessary physical observation-model evidence, not more sampler tuning.

Native K is IR/Palomar K(Vega), not calibrated2MASS Ks. Dust/aperture,
all-matter versus stellar COM velocities, hydro/PM density response, native
NGP moments, cosmology and finite-volume limits remain explicit. Fitted
coefficients cannot automatically become R2 priors. Failure of prediction
does not uniquely identify bias, FoG, LF shape or stochastic dependence.
No extra LF likelihood on the same counts, unsupported floor, actual IC
sample, native LG truth assignment or R2 posterior promotion.

Q-GOAL: tests the unresolved luminosity/selection/RSD tracer law needed for
the first actual z=0 field delivery. MW/M31 remain ambiguous and M33 unresolved
on the NEW inferred field; their observables must constrain that same field
later at<=0.3. Native IDs only label this external calibration input.
Q-LEAN: reuses the existing forward/count machinery on125000 source cells;
one9-coordinate fit plus three fixed endpoint controls, no PM adjoint,
simulation, download, generic gate framework or closed600-metric branch.
Routine bounded driver review under the current external-review policy.

One typed H200/H100/A100 Slurm mode after checking availability. Host estimate
<=8GiB for native arrays, several100MB means, JAX compilation/workspace and
existing environments; request10GiB (>20% margin),2CPU,1GPU,90min Slurm.
Output `/gpfs/kjhan/CF4/z0_density/r2_native_rsd_mock_fit_20261002_v1/`.
Tests/driver/numerical results and actual memory are pending submission.

Source c5f0085 submitted as410544 at2026-10-02 16:08:42 KST. It RUNS on
H100/syn08 through Slurm, typed `gpu:H100:1`,2CPU/10GiB/90min. Four mock
observation tests pass. Native source extraction/placement selects6050 unique
galaxies: six observed populations614/1799/1862/54/812/909; training2656,
test2360,buffer1034. Every selected galaxy has>=300 stellar particles.
The156 selected true-faint-bin objects therefore have resolved stellar input;
the earlier unresolved all-faint population does not describe these actually
selected rows. It does not certify native luminosity/dust/baryonic accuracy.
Initial target compilation/gradient is in progress; no fitted law or predictive
result exists yet. Logs/artifacts use the fixed410544/v1 paths above.

## Terminal result and bounded stationarity continuation

410544 COMPLETED exit0 in2m44s. Application peak host memory2.012GiB;
four tests pass; directional derivative relative error4.45e-6. L-BFGS
stopped at24 iterations, gradient infinity norm2.453: NOT converged.
Training score improves -6168.625 to-5954.164; development test score
improves -5358.851 to-5160.472. However test counts grow from2454.926
to2503.112 versus2360 observed, and population/radius aggregate L1 worsens
222.629 to285.717. Neither score improvement nor exit0 proves adequacy.

GL4 relative intensity L1=.000413; LOS16=.000823. These controls barely
change population totals; sparse likelihood changes are nevertheless finite.
Diagnostic NGP changes intensity by36.6%, improves test sparse score, but
has TWO occupied training cells with zero support. Do not select a new law
using the already consumed development test, or fix support with a floor.
The image population_radius_prediction.png is the actual count comparison.

Next action: restart the same GL2/TSC training objective from the final
coordinates with fresh L-BFGS history and disclosed training-rate reprofiling;
200 iterations/240 hard evaluations maximum. Runtime expectation~12min from
2.47s/evaluation plus compilation/controls, within existing90min allocation.
Repeat derivative/rate checks, retain the same populations/exposures/prior,
and report stationarity only if optimizer success AND gradient_inf<=.001.
No independent validation claim: test outcomes have already been inspected.
No simulation, CF4 posterior promotion, NGP switch or new calibration prior.
Output v2 in the same fixed native mock family. Request10GiB again: adequate
measured headroom, unchanged input sizes, not an enlarged field calculation.
Q-GOAL/Q-LEAN and NEW-field MW/M31 ambiguity/M33 unresolved remain as above.

410551 COMPLETED/exit0 in3m57s:48 iterations/53 evaluations, gradient_inf
.0005615, proper optimizer gradient convergence. Host peak2.010GiB.
Training score-5954.1345; test-5160.7389, expected2504.1049 versus2360.
Test aggregate L1=285.6283. Completing optimization does NOT fix prediction.
sigma_LOS=182.2236km/s, alpha=-.939502, Mstar=-23.341070; conditional
point estimates only, no posterior/prior injection. Close fit extension.

Next diagnostic keeps these coordinates fixed, with NO refit: NGP source
GL2/LOS8 baseline, GL2/LOS64 and GL4/LOS64. Examine all occupied zero keys,
their observed galaxy counts and repaired intensities, training/test means
and scores. NGP's discontinuous voxel indicator can miss small regions with
finite quadrature; two baseline zeros do not prove physical absence of
support. Conversely, a nonzero refined value is not proof of convergence.
One existing input allocation,2CPU/10GiB/typed GPU; expected<=5min, cap30min.
No smoothing floor or NGP adoption. These outcomes discriminate numerical
support loss from a physical stochastic-count/selection discrepancy before
modifying the R2 observation law. Q-GOAL/Q-LEAN and latent roles unchanged.

410555 COMPLETED/exit0 in54s. GL4/LOS64 still gives zero at training key
10098880 and4.15e-15 at11246902. No useful support restoration or numerical
convergence proof follows. Next short source-row readout uses exact two
native IDs (labels only) to compare galaxy LOS velocity with its native
matter-cell mean and fitted sigma182.2236. Also report all6050 selected
absolute standardized residual quantiles and beyond8-sigma count. This
tests local Gaussian-tail/geometry explanations; a local residual cannot
alone prove absence of support from other source locations. No fit, field
change, prior injection or repeated full integration. Same10GiB/2CPU Slurm
GPU runner,30min cap, expected seconds. Source windows and roles unchanged.

410561 COMPLETED/exit0 in9s: native ID100 residual1777.7945km/s=9.7561sigma,
ID96511 residual1558.8406km/s=8.5545sigma; both source positions near window
edges. All selected absolute residual percentiles50/90/95/99/100 are
.17335/1.60549/2.54101/4.63396/9.75612sigma;8/6050 exceed8sigma. These are
calibration labels, NOT inferred-field MW/M31/M33 IDs. Evidence identifies
real non-Gaussian-tail/geometry concerns, not uniquely absent total support.
Important finding warrants one bounded read-only Fable5 design consultation
on minimal normalized FoG law and count-operator closure. Explicitly asks
Q-GOAL/Q-LEAN, scientific claim limits, same-field member roles and essential
versus deferred requirements. No further optimization or law adoption while
advice is pending. Invocation uses claude-fable-5, plan permission mode,
Read/Glob/Grep only, Edit/Write/Bash denied, medium effort, timeout600.
Driver independently decides on usable advice; substantive disagreement is
not tool failure. Prompt in config/cf4_r2_native_velocity_tail_fable_20261002.txt;
output /gpfs/kjhan/CF4/logs/r2_native_velocity_tail_fable_20261002.txt.

Fable advice returned CONDITIONAL PASS. Adopt: characterize native training
residuals, compare narrow+broad Gaussian mixture with Gaussian and finite-
variance Student-t, measure interpolation effect, prefer Gaussian mixture
for compatibility with periodic Gaussian CDF machinery. NO immediate prior
injection/count refit. Amend: all6050 previously summarized galaxies include
test and observation selection; they are NOT a clean unconditional training
sample. New fit uses source true y<-6, >=300 stars, intrinsic -25<=K_h<-21
BEFORE apparent/radius/observed-K selection. Observation-selected comparison
is explicitly diagnostic only. Narrow/broad are not central/satellite labels.

Reject advice's strong claims that TSC quadrature is fully converged or the
NGP zeros uniquely proven truncation artifacts: endpoint integral controls
do not establish all derivatives or exact continuous support. A tail-scale
number alone does not locate its physical source. Other source rays remain
possible. Reject treating between-half spread as an independently calibrated
prior width; correlated halves cannot certify cosmic/calibration coverage.
Keep the6% mismatch unresolved rather than declaring sample variance proves
adequacy. Exact NGP integration is a scientifically motivated design option,
but selection/mark weights vary along each ray, so CDF differences alone
cannot exactly integrate the FULL marked count law. No silent operator swap.

410569 submitted Slurm H100 mode: short residual characterization on native
training sample and >=3 versus<3Mpc source-window-edge distance. Compare
native cell mean with TSC-interpolated mass/momentum THEN ratio, not mean of
velocities. Save residual arrays/IDs/rho/K/edge distance, zero-mean Gaussian,
ordered-width two-Gaussian mixture and finite-variance Student-t MLE/AIC,
quantiles and rho split. Two fixed optimizer starts only stabilize the
three-scalar mixture likelihood, not count-target scans. Empirical fit is
conditional on the stated native stellar-resolution/intrinsic-K population,
not luminosity-independent universal calibration. Tail uncertainty and
native-to-CF4/Ks bridge remain open. No observed-test model selection, native
role assignment, new simulation, field fit or R2 promotion. Native IDs label
calibration only. Q-GOAL: inform normalized LOS closure on the R2 critical
path. Q-LEAN: one cheap readout reusing known files; no generic gates.
Runner2CPU/10GiB/30min, expected<2min; resource peak recorded by application.
Exact reachability/operator redesign follows measured residual evidence,
not the unsupported categorical assertions in the advisory.

410569 COMPLETED/exit0 in8s (application2.065s). Preselection training3785
galaxies: cell-mean residual Gaussian sigma204.32 versus mixture widths
21.13/303.05km/s and broad weight.4519. Mixture AIC46569.18 versus Gaussian
51013.49, restricted finite-variance t47390.03. The t fit hits its nu>2
lower boundary (nu2.000045); do not claim an unrestricted t comparison.
Mass/momentum TSC reference mixture31.53/317.80km/s,weight.4416, likewise
strongly preferred to Gaussian. This is training descriptive evidence only.
No central/satellite assignment, independent calibration, or count refit.

Interior>=3Mpc mixture16.29/188.39km/s versus boundary24.56/400.81km/s,
weights.4590/.5427: source geometry/environment heterogeneity remains
material. Selection-only comparison shifts widths19.88/284.55, confirming
the importance of the preselection population. Density below native mean
has only ONE training galaxy; a rho<1 versus>=1 split cannot calibrate
density dependence or void tracers. Also this K/stars selection does not
cover the entire five-true-K latent process or its unbounded tails.

Driver next design: normalized narrow+broad LOS Gaussian mixture is supported
as a development candidate, not a universal R2 coefficient. Before the single
count refit, implement consistent voxel-count integration with boundaries
explicitly integrated, keeping varying luminosity/selection weights inside
the integral. Establish geometry reachability on the two actual rays and
training-only population/environment sensitivity of mixture parameters.
Do not resume count fitting by silently retaining the materially mismatched
TSC operator, injecting the fitted mixture as certified prior, or selecting
based on the consumed test. Native-K/Ks, full true-bin/tail coverage and
calibration uncertainty remain R2 prerequisites; no new simulation yet.

Next implementation is a scalar voxel-CDF branch in the existing marked
source-volume count engine (TSC production default unchanged). Intersect
each ray with the half-open observed voxel and periodic-image shell, THEN
integrate the varying luminosity-transfer weights using Gaussian-CDF nodes.
Not simply a CDF probability multiplied by a cell-centre mark. Five geometry
tests cover parallel/half-open rays, periodic/negative direction, Gaussian
cell-partition mass conservation, finite inactive intervals and derivatives.
The two previously failed occupied training keys are explicit probe cases,
not a new likelihood fit or test-selected candidate list.

Probe Gaussian182.2236 and narrow/broad21.1317/303.0455 components with
training weight.451921. Each component uses its own8sigma/27-image bound,
unconditional probabilities; omitted Gaussian mass<1.3e-15 per component.
No epsilon floor or selected-region renormalization. Source GL2/mark4 versus
GL4/mark8 tests remaining integration sensitivity; finite-difference check
of broad log-width derivative. Physical boundaries are exact along LOS;
source-volume interpolation and mark integration remain numerical. No claim
of complete-field NGP implementation or global calibration from two keys.
Memory<=8GiB planning bound,10GiB request,2CPU/typed H100,30min cap, estimated
<10min including compilation and derivatives. Output fixed native voxel
closure v1. No new field evolution, actual CF4 outcome or R2 prior injection.
Q-GOAL: separate supported velocity-tail closure from observation-bin
smoothing before R2 field inference. Q-LEAN: existing engine plus scalar
boundary clip, five focused tests and two keys, no new framework. Same NEW
field MW/M31 roles remain ambiguous, M33 unresolved, native IDs calibration
labels only. Next full-count implementation depends on these readouts.

410574 COMPLETED/exit0 in1m17s, application65.335s/host1.181GiB. All4
native+5 voxel geometry tests pass. Broad-width FD relative errors1.05e-6
and8.76e-7. SourceGL4/mark8 single Gaussian values0/4.3717e-15; mixture
2.3871e-7/1.5811e-6 at the two keys. Mixture removes zero support but still
assigns extremely low expectations to observed single galaxies. Refinement
changes mixture by6.3%/-8.0%, so no complete marked-integral convergence
claim. A globally fitted two-Gaussian velocity law is NOT declared adequate.

Next cheap training readout410575 uses already available native second
velocity moments: diagonal variance E[v_i^2]-E[v_i]^2, preserving mean AND
dispersion. Compare same three-parameter mixture with broad width
sqrt(core^2 + b^2 * diagonal-LOS-variance-proxy) against global widths, on
the SAME preselection3785 rows. For both native cell and interpolated
mass/momentum/raw-second variants. Nonnegative variance asserted (roundoff
only clipped), no arbitrary scatter floor beyond fitted positive core law.
Diagonal moments do not contain cross-axis covariance: explicitly a proxy,
not exact LOS covariance. Free b accommodates galaxy/all-matter discrepancy
but is NOT a hydro/PM/Ks cross-calibration. Two fixed small-MLE starts, no
count refit or heldout tuning. This adds an existing physical predictor, not
global sigma inflation or another native count-optimizer scan. Same2CPU/
10GiB/typed H100,30min cap, expected seconds; output native-velocity v2.
Q-GOAL/Q-LEAN and NEW-field role limitations remain unchanged. No new
simulation or R2 prior injection; do not select the law on two rare keys.

410575 COMPLETED/exit0 in20s;9tests pass. On3785 preselection training rows,
conditional matter-dispersion mixture AIC43968.17 versus global46569.18,
with SAME three parameters; core7.989km/s, matter-dispersion scale.76622,
broad weight.75607. Broad sigma median92.48/p99448.59/max450.64km/s.
TSC-moment variant likewise improves AIC45268.52 versus47833.52, core8.704,
scale.74406,weight.83618. Both fixed MLE starts agree. This favors an
environment-conditioned velocity closure over global widths in training,
not independent validation or a calibrated universal satellite fraction.
The mixture labels remain mathematical narrow/broad components, not galaxy
identity assignments; diagonal covariance and one-box/K/resolution limits
persist. Next: connect mean AND native dispersion proxy consistently to
the voxel-boundary marked integral and check conditional count prediction,
before a full R2 use. No stochastic-floor rescue or widening by hand.

Next connection bundle: optional source diagonal-variance input to the
existing marked integrator, recomputing projected width for EVERY shifted
source-volume ray. Core and broad probabilities integrated independently,
then normalized training mixture weights combined. No cached central-ray
variance alias. Caller verifies nonnegative moments and conservative maximum
width within8sigma_radius<192 (27-image domain); image possibility uses any
source width, not a scalar-only predicate. Default constant-width production
route remains unchanged. Seven focused geometry/projection/derivative tests
plus four observation tests run inside Slurm before native two-key readout.
GL2/mark4 versusGL4/mark8, fixed global-versus-conditional comparison and
conditional log-scale FD. Existing tail keys are training diagnostics, NOT
the criteria for fitting/choosing the conditional law. No fit or prior
injection.10GiB/2CPU/typed GPU/30min, expected<5min; same input sizes plus
native diagonal variance. This is the physical mean+dispersion connection
before full-count boundary integration, not yet a complete R2 observation
likelihood. Q-GOAL/Q-LEAN and MW/M31 ambiguity/M33 unresolved unchanged.

30c77c3 connection410587 COMPLETED/exit0 in1m49s;11tests pass. Conditional
GL4/mark8 key means.000304812/.00282951 versus global2.387e-7/1.581e-6,
improvement~1277/~1790 times. Conditional-scale derivative errors2.37e-7/
1.47e-7. Not adequacy or convergence: source/mark refinement still changes
values, and two rare training cells are not a calibration population.

Next full-grid extension partitions each finite Gaussian ray at ALL observed
voxel-plane crossings, integrates varying mark weights inside each interval,
and deposits interval mass to its single voxel. No sampled NGP indicator or
TSC smoothing. A fixed grid_size+2 plane budget PER axis is sufficient from
the declared ray span<box (8sigma_radius<box/2); padding duplicates endpoints,
not truncates physical crossings. Oblique/negative ray tests verify all
breakpoints against explicit planes and conserve Gaussian interval measure.
Eight focused tests plus four mock tests precede native full-grid prediction.

One fixed conditional closure, sourceGL2/mark4, all six populations and both
training/development-test exposures including empty cells. Check exact
agreement with410587 scalar values at the SAME quadrature rule; report all
occupied zero keys (never floor), count scores, means and radial tables.
Store compressed conditional means, no field fit or closure/count refit.
This is full-count operator evidence, not a valid CF4 posterior or pristine
heldout test. Native inputs/windows/roles and diagonal/K/Ks limits unchanged.
Array bounds: one streamed source-volume node,125000sources x392 crossings
~392MB doubles, plus sort/scatter/compilation; plan host<=8GiB and device
<=12GiB, request10GiB host/2CPU/one typed GPU/30min cap. Expected<10min;
actual peaks/cost determine any next extension, no automatic larger fit.
Q-GOAL: physically consistent observation operator over whole calibration
population before R2 inference. Q-LEAN: existing source/mark integrator and
one boundary partition, two component forward passes, no simulation or
new validation framework. MW/M31/M33 same-NEW-field constraints unchanged.
Source63324d6 submitted410592 typed H100,2CPU/10GiB/30min. Initial Slurm
state PENDING Priority; missing log before allocation is not execution
failure. No resubmission/cancellation follows merely from queued state.

410592 now COMPLETED/exit0 in32s;12tests pass, native application17.892s,
host peak1.220GiB. Full-grid/scalar key means agree exactly/to2.67e-16.
All occupied training AND development-test keys have positive support.
Train score-5135.0783,mean2653.8367 vs2656,L1=256.4940; test score-4434.0938,
mean2499.0907 vs2360,L1=253.4782. Fixed conditional closure/operator improves
sparse score versus old TSC/Gaussian but retains~5.9% test excess. No count
refit, test-selected closure, independent calibration or posterior promotion.

Next in the same bundle: refine full prediction to sourceGL4/mark8 and check
training logscale score derivative at this fixed physical closure against
FD. This tests the FULL empty-cell-inclusive objective, not just the two
scalar values. Scalar reference uses matching410587 GL4/mark8. Keep exposures,
galaxy rows, rates and all coefficients fixed; no score optimization. Save
primal report before derivative so a failed reverse pass does not erase
forward evidence, and make a real population/radius comparison PNG.
Same10GiB/2CPU/typed GPU/30min cap; forward peak measured1.22GiB, refined
streamed arrays same size; gradient/compilation planning bound<=8GiB host
and<=24GiB device. Runtime estimate<10min, bounded by actual Slurm cap.
No broad native count-optimizer restart. Q-GOAL/Q-LEAN and same-NEW-field
MW/M31 ambiguity/M33 unresolved remain as above. Actual CF4 heldout untouched.
