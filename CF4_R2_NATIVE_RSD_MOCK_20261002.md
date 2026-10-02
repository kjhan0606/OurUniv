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
