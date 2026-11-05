# Conditional-v6 posterior run contract — revised draft, 2026-11-04

**Status: design proposal only. No Slurm job is authorized by this draft.**
The first version received Astra's mid-course **NO-GO as written**. This
revision adopts the supported persistence and diagnostic corrections below;
the production runner, final resource estimate and scientific scope still
require implementation/reconciliation before submission.

## Estimand and boundary

This proposal targets only the frozen conditional-v6 observation model:
47,121 training 2M++ count rows plus1,414 observed linked SDSS-PV/CF4 FP
training marks. It does not estimate missing FP-mark/group inclusion, redshift
success, or the full CF4 survey selection. Any field from this target must be
described as conditional on the selected observations and the frozen
observation law—not a full-survey calibrated CF4 posterior. The potential
individual-TF factor in
`CF4_R2_TF_FORWARD_SOURCE_FACTOR_SPEC_20261104.md` is not included.

Held-out count/mark outcomes remain sealed until all chain/field Monte Carlo
criteria below pass and the target is frozen. Afterward, held-out log
predictive scores and their Monte Carlo errors are reported; they are not used
to tune the target or sampler.

## Proposed sampling design

- Four chains must start from independent draws of the complete frozen
  standard-normal canonical prior `q=[white_IC(N^3), 9 tracer whites, 15
  population whites]`, with independent RNG streams. No chain may inherit
  state60, another chain's trajectory, or a training-fitted initial field.
  Each chain may test at most four prior draws for finite target support;
  record every attempt and its reason. A genuine zero-likelihood start may be
  replaced within this bound, but a numerical failure aborts that chain. Do
  not add a likelihood floor or initialize from a selected “best” field.
- Warmup is capped at128 transitions per chain, with step-size adaptation only
  in warmup and the metric/force law fixed before each chain. Freeze the
  kernel before retention. The executable contract must state the numerical
  step-size bounds, target acceptance, trajectory length, adaptation equation,
  and a warmup-end viability rule. `128` is a spending cap, **not** evidence
  of equilibration; chains failing the predeclared viability review stop
  before retained sampling rather than automatically spending the remaining
  budget.
- At most512 unthinned retained transitions per chain. The maximum is a cap,
  **not evidence that the effective sample size will be adequate**.
- Use the unchanged full training target for acceptance. A fixed approximate
  integration force is permissible only if its proposal map is reversible and
  volume-preserving and the full target supplies the Metropolis correction;
  name and validate it as such, not as exact-gradient HMC. No surrogate or
  approximate target may define retained states.

## Minimum Monte Carlo evidence for a conditional field product

For scalar nuisance parameters, training log-likelihood components, and
predeclared observables require four-chain rank-normalized **and folded**
split-\(\hat R\le1.01\), bulk ESS\(\ge400\), and tail ESS\(\ge400\). The
observable list must include selected low-k real/imaginary density and
mass-weighted momentum (\(\rho v\)) modes, band powers, and fixed regional
density/flow summaries; shell powers alone are insufficient. Name the implementation and ensure it handles
the independent chain starts and unequal/incomplete chains without silently
truncating diagnostics to a favorable suffix.

The scalar thresholds are the predeclared R2 screening rule, not a guarantee
of global exploration. Rank normalization, folded split-Rhat, bulk/tail ESS,
and Geyer's initial-monotone autocorrelation sequence follow the
[Stan Reference Manual posterior-analysis definitions](https://mc-stan.org/docs/reference-manual/analysis.html).

For the actual N256 density contrast, each velocity component, and velocity
occupancy, compute chain-separated online first/second moments and fixed-batch
means from post-Metropolis states only. Rejected proposals contribute the
repeated retained state, never the trajectory endpoint. Predeclare the full
box and observation-informed reporting regions, two batch lengths and minimum
number of complete batches. Unexpected nonfinite cells and zero-variance
cells must be explicitly counted and cause a stop where they imply a stuck or
invalid map; they may not disappear from the percentile denominator. Report
cross-chain spatial disagreement. Check both mean-map precision and posterior
SD-map stability; an MCSE criterion for the mean alone does not validate the
uncertainty map. Any percentile gate must be reported separately for the full
box and each observation-informed region, with its valid-cell denominator.

## Spatial precision-gate correction — 2026-11-05

The first-version regional point-estimate rule
`p95 MCSE(mean)/posterior_SD <=0.10` is **not a qualified production gate**.
The spatial synthetic run at job450009 found substantial marginal
false-passing against its declared velocity-y reference in a 147-cell
synthetic central region: for batch64, 1.17%, 7.42%, and 23.83% at spatial
correlation lengths0, 1, and2 cells; batch32 gives25.78%, 39.06%, and48.05%.
At length2/batch64, the Wilson95% interval is19.02–29.41%. These are
per-observable/per-scale rates, **not** the joint false-acceptance rate of the
complete gate. The velocity reference is asymptotic, and the generator's
spatially varying AR process does not establish joint spatial stationarity.
The archived study also did not preserve paired replicates needed to evaluate
the conjunction. See
`CF4_R2_SPATIAL_MCSE_CALIBRATION_PLAN_20261104.md`.

Therefore keep0.10 as the maximum scientifically tolerated relative MC error,
not as a calibrated point-estimate cutoff; do not pass production chains using
the current rule and do not tune an inflation factor or replacement threshold
against the same development run.

The **proposed joint field-precision event** is now explicit. Let `G_field`
pass only when the following three field conditions hold together:

1. At both batch sizes 32 and 64, for both the full box and frozen training
   reporting region, the regional p95 of `MCSE / posterior_SD` is at most 0.10
   for density mean, density posterior SD, occupancy, and each velocity
   component's conditional mean and posterior SD.
2. For those same two domains and all density/velocity/occupancy maps, the
   regional p95 of between-chain mean-map disagreement / within-chain
   posterior SD and between-chain posterior-SD-map disagreement / mean
   posterior SD is at most0.10.
3. Every required map cell is finite and defined. Nonfinite ratios, an
   unexpected undefined cell, or a zero posterior SD with nonzero MCSE fails
   closed; expected physically undefined cells must be enumerated and resolved
   before defining the production reporting domain, not removed from a p95.

The separately frozen scalar-chain gate `G_scalar` still requires
rank-normalized and folded split-`Rhat <=1.01`, bulk ESS>=400, and tail
ESS>=400 for every predeclared scalar and field observable. Overall chain
qualification requires `G_field AND G_scalar`. The field calibration below
does not by itself calibrate the false-acceptance probability of this entire
combined sampler gate.

The working false-acceptance tolerance is **at most 5%**: in each predeclared
alternative/stress scenario where at least one known-law or finite-sample true
field-precision regional-p95 ratio violates its 0.10 requirement, the
probability that the *entire* observed `G_field` conjunction nevertheless
passes must have a one-sided 95% binomial upper confidence bound <=0.05.
Realized cross-chain disagreement is an observed diagnostic within the gate,
not a latent precision truth and must not define whether a replicate is a
false acceptance. This is a driver-set field-gate risk criterion, not a result
of job450009. Evaluate paired outcomes across metrics, scales, and regions; do
not infer the joint probability from marginal rates. A single stress scenario
can qualify only that alternative and does not establish that the gate accepts
genuinely precise fields or calibrate all relevant posterior dependence.

### Cross-chain map disagreement is not pooled MCSE — 2026-11-05

The implemented cross-chain map statistic is the across-chain sample SD of
individual chain means (or posterior-SD maps), normalized by the within-chain
posterior SD (or mean posterior SD). It is not the Monte-Carlo standard error
of the equally weighted pooled posterior map. For `m` independent,
equal-length, comparably mixed chains, if `s_B` is the across-chain SD of
their means, the corresponding pooled-mean standard error is approximately
`s_B/sqrt(m)`. Thus with four chains, a `s_B / within_SD <= 0.10` condition
acts roughly like a `pooled_MCSE / within_SD <= 0.05` condition, before
regional-percentile effects. Unequal autocorrelation, chain variances, or
nonstationarity weaken this approximation; the two quantities must not be
interchanged.

The 2026-11-05 fixed stationary stress result is consistent with this
distinction: the known velocity-x/z pooled precision ratios are about0.0534,
yet their individual-chain mean-disagreement p95 medians are about0.174 in
the full box and0.17 in the training region. This is not evidence of
nonstationarity; it shows the0.10 cross-chain cutoff imposes a stricter
separate chain-agreement requirement than the pooled precision limit. Its
scientific purpose and acceptable sampling-cost implication must be made
explicit before any positive-control run or production qualification. Do not
relax it from this single negative-control result. The paired outcome and
limitations are in `CF4_R2_JOINT_FIELD_MCSE_STRESS_RESULT_20261105.md`.

Driver disposition: retain this as a separate **chain-local agreement and
map-stability guard**, not as a second estimate of pooled MCSE. In the ideal
IID/equal-chain case, `s_B / within_SD` is approximately
`1/sqrt(ESS_per_chain)`; the0.10 cutoff corresponds roughly to100 effective
states per chain, or400 pooled effective states across four balanced chains.
That is consistent in scale with the existing scalar ESS>=400 screen, but is
only a heuristic correspondence: autocorrelation, finite chain count and the
regional p95 matter, and the posterior-SD-map ratio is a separate stability
statistic. Keep the predeclared cutoff unchanged for now and label its extra
per-chain requirement explicitly. The stress result does not show
nonstationarity or justify loosening it. Defer a positive control until the
chain-local requirement is frozen, and do not let it displace the
source-matched selection work; it is not needed to reinterpret the completed
negative control.

Before any further calibration, use a generator with a verified joint
stationary spatial law (or a separately validated finite-sample reference),
preserve paired gate outcomes, distinguish exact finite-length references
from asymptotic ones, and enforce the fail-closed validity rule. A spatially
varying persistence field requires its joint invariant covariance to be
derived and checked. No new batch-length, threshold, or correlation sweep is
implied by this contract amendment. If `G_field` cannot be shown to meet the
5% risk bound at the available sample count, report insufficient Monte Carlo
precision and do not promote the maps.

Accumulate posterior moments and fixed-batch map summaries online; full
density/velocity map samples need not be written for every state. However,
**persist every retained canonical state `q` for every chain, including exact
rejection repeats, in a lossless representation** with chain, iteration,
target value, RNG/checkpoint and checksum metadata. The held-out outcomes remain inaccessible
during sampling and all convergence assessment. Only after the predeclared
chain and field-MCSE criteria pass may a separate scorer load held-out
outcomes and compute the joint held-out predictive factor for each persisted
state. Report Monte Carlo error for the final log posterior-predictive score.
Without persisted states or a separately validated exact replay, the sealed
held-out evaluation cannot be performed; online moments alone are
insufficient.

The state-storage requirement is material: for `q` with `256^3+24` float64
coordinates, four chains times512 retained states are approximately256 GiB
before metadata/checkpoint overhead. A lower-precision or compressed format
is not assumed valid; any such substitution needs a target/field precision
replay before production. Do not persist full field maps at every state
unless a later, explicit need justifies the additional storage. The final
report must distinguish posterior spread from Monte Carlo error.

## Current implementation readiness (updated 2026-11-04)

The proposed schedule and diagnostics are **not executable in the current
runner**:

- `scripts/cf4_r2_n256_joint_pilot.py` still runs one chain from one supplied
  initializer/restart. Its optional `long_chain` mode requires
  `CF4_R2_AFFINE_FORCE=1`; `src/cf4_r2_chain_schedule.py` defaults to48
  proposals,12 warmup, then fixed4/8-step trajectories—not the proposed
  128/512 schedule. The tested `CF4_R2_START_POLICY=independent_prior` branch
  now draws the complete `N^3+24` standard-normal state and uses four-indexed,
  separate prior/audit/transition RNG streams. Only an exact `-inf` count
  component with finite raw-mark component may retry, at most four times;
  `CF4_R2_MASTER_SEED` and `CF4_R2_CHAIN_INDEX` select the deterministic stream
  family, while `CF4_R2_CHAIN_ID` may name its retained-state store. Support
  and numerical failures abort. Seven focused start-helper tests and the
  targeted 67-test CPU regression set pass under the runner's Python 3.11.
  The canonical state store is format v2 and now stores/checksums the
  post-transition NumPy RNG state with each retained state, including exact
  rejection repeats; a synthetic restore test reproduces the next RNG draws.
  The accepted-state checkpoint records all current RNG streams, but no runner
  recovery path yet restores the state and reconciles it with field-MCSE
  sufficient statistics. `src/cf4_r2_retained_diagnostic_replay.py` now
  rebuilds posterior moments and field-MCSE from every checksum-verified
  retained q, including rejection repeats; a synthetic regression matches
  direct accumulation exactly. It reads no likelihood or held-out data, but
  is not yet wired into runner resume and has no N256 cost/memory measurement.
  This remains one chain per runner invocation and does not implement the
  proposed four-chain supervisor or sampler restoration from retained RNG
  metadata. The legacy 48/12 schedule now records its diminishing
  Robbins–Monro log-step equation and, after the complete 12-transition
  warmup, validates transition records and step bounds; zero accepted
  transitions stop before retention. A nonzero acceptance result only permits
  the retained diagnostic to proceed and explicitly does not establish
  equilibration or stationarity. Seven focused schedule tests pass. This is
  not the proposed 128/512 warmup/production contract. Legacy format-v1 state stores
  remain readable and checksum-verifiable but cannot be extended without RNG
  history. The lossless state store is wired
  into this legacy long-chain path and redundant every-eighth full-field dumps
  were removed; this still does not turn the diagnostic into the proposed
  production pipeline. The runner now records retained-state low-k density
  and mass-weighted momentum modes, density band powers, and training-region
  density/occupancy/flow summaries. Their N256 wall-time overhead has not been
  measured.
- `src/cf4_r2_posterior_moments.py` accumulates posterior sample moments and
  occupancy, but no fixed-batch field means or map-level MCSE. Its output
  explicitly sets `posterior_MC_error_estimated=False`.
- `src/cf4_r2_chain_diagnostics.py` now computes rank-normalized folded
  split-Rhat and Geyer initial-monotone bulk/tail ESS for equal-length scalar
  chains. Controlled IID/AR(1) regressions cover the ESS estimator. The
  retained-state runner records low-k density and mass-weighted momentum
  modes, density band powers, and training-region density/occupancy/flow
  summaries; its extractor refuses missing or nonfinite observables.
  `four_chain_scalar_diagnostics` checks four independent chain indices,
  distinct recorded prior-start states with only the permitted zero-count
  retry history, shared target/schedule metadata, complete equal-length
  retained traces, and the frozen scalar thresholds without trimming or
  promotion. `scripts/cf4_r2_four_chain_report.py` runs this gate on exactly
  four distinct chain-report files, records input hashes, defaults to stdout,
  and refuses to overwrite an output file. Synthetic CLI-path, fail-closed,
  and non-promotion tests pass; a 4,096-state-per-chain IID control exercises
  the diagnostic PASS path without representing CF4 inference.
  `scripts/cf4_r2_four_chain_field_mcse_report.py` now reads four completed
  chain reports, verifies the four chain-indexed RNG streams and shared target
  contract, computes chain-separated map disagreement before pooling, and
  combines persisted sufficient statistics one source chain at a time. The
  map comparison streams one chain's selected map members through online
  Welford summaries instead of retaining all four chain maps at once; two new
  tests match its scalar/vector regional summaries to the original
  in-memory comparator within `1e-12`. It
  reads only training keys from the supplied split and never reads held-out
  outcomes. Four small synthetic NPZ/report tests pass. This report explicitly
  does not evaluate the scalar convergence gate or promote a field. It is not
  an integrated chain supervisor/restart or production qualification
  pipeline. Field-MCSE thresholds/spatial calibration, held-out scoring, and
  N256 memory/I/O remain unmeasured or incomplete.
- The production implementation must therefore be built and regression-
  tested before this is an executable run proposal. It must preserve the
  exact target (an approximate force is acceptable only with a verified
  reversible proposal and exact full-target Metropolis correction), retain
  rejected states, and keep held-out outcomes sealed until qualification.

This is an implementation-readiness finding, not a new sampler experiment.
No Slurm calculation has been submitted under this draft.

### Astra review and driver disposition — 2026-11-04

Astra's read-only verdict was **NO-GO as written; remediable design gaps**.
The driver independently checked the named repository evidence and adopts
these findings:

1. Retaining only moments cannot score held-out data after the fit while
   keeping those outcomes sealed. The lossless per-transition canonical-state
   store is implemented in `src/cf4_r2_canonical_state_store.py`, wired into
   the existing one-chain long-chain diagnostic, and tested with exact
   rejection-repeat/resume cases. Its v2 index also persists/checksums
   transition RNG state; eleven canonical-store tests pass. Legacy v1 stores
   remain inspectable but are not resume-compatible. The store is not yet
   connected to an independent-start production driver or runner resume path.
2. The candidate field criterion is incomplete. The implementation must
   define reporting regions, batch lengths/count, nonfinite and zero-variance
   handling, cross-chain map disagreement, and posterior-SD-map precision;
   do not use a whole-box p95 of mean MCSE as the sole pass.
3. The tested start primitive now covers complete independent prior draws,
   separate purpose-specific RNG streams, bounded finite-support retries and
   fail-closed numerical-error handling. The four-chain launcher/replay path,
   warmup adaptation, trajectory, freeze point and early-stop review remain
   unimplemented.
4. Specify held-out joint score as the posterior `logmeanexp` of per-draw
   joint count+mark log likelihoods, with chain-aware Monte Carlo error.
   Held-out scoring evaluations must be included in the cost and remain a
   post-qualification action.
5. The proposed3,200 GPU-hour ceiling is **not yet a usable cap**:2,924 hours
   is only the two-transition extrapolation for HMC transitions; it excludes
   up to2,048 held-out evaluations, implementation-specific setup/replays,
   diagnostics, and state output. Do not ask the user to accept this number as
   though it covered the full deliverable. Re-cost after the scorer and chain
   pipeline exist; enforce any final ceiling across all segments and chains.

The conditional-v6 chain can at best deliver an assumption-conditional
posterior. It cannot by itself complete the active R2 goal's observation
calibration: the source-matched FP/group availability and failure parent
remain unidentified. Keep that as an independent required science item; do
not promote the conditional product as full-survey calibrated. Astra's advice
is mid-course guidance, not an R2 exit audit. Its external report is available
in the conversation record; no separate output file or job was created.

If any chain diagnostic or field-map MCSE target fails at the cap, report
`INSUFFICIENT_MONTE_CARLO_PRECISION`; do not promote a MAP, a short trace, or
the terminal state as a posterior. No automatic continuation is included.

### Field-map MCSE primitive — 2026-11-04

`src/cf4_r2_field_batch_mcse.py` now provides chain-local fixed-batch
accumulators for normalized density (reporting contrast relative to a declared
reference), conditional mean/SD velocity, and occupancy. Its multiscale
wrapper independently finalizes each predeclared batch length and refuses to
use an incomplete trailing batch. It counts repeated post-Metropolis states,
retains partial-batch sufficient statistics losslessly, rejects
nonfinite/overflowing inputs before counting them, and explicitly reports
undefined and zero-posterior-SD cells. The batch-delta MCSE formulas are
checked against direct synthetic covariance calculations. The new
`src/cf4_r2_cross_chain_field_diagnostics.py` compares chain-specific mean and
posterior-SD maps, supports unequal chain lengths, and marks a cell undefined
if any chain lacks a valid value instead of silently omitting it. Seventeen
field-MCSE tests, five cross-chain tests, and the three existing
posterior-moment tests pass.

These tested primitives are now wired through
`src/cf4_r2_retained_field_mcse.py` into the existing one-chain
`long_chain` output path. The adapter receives post-Metropolis retained fields,
including the unchanged prior field after a rejected transition, and records
accepted/rejected transition counts. At normal chain completion, the runner
now writes `field_mcse_sufficient_statistics_unassessed.npz` for later
post-processing and records its raw-array byte estimate in the report; it
refuses to overwrite an existing file. This is not an interruption-safe
checkpoint and is not written periodically. At N256 it is estimated at
9.25GiB of raw arrays per chain (about37GiB for four), excluding small NPZ
headers. No N256 file, read/write time, or peak RSS has been measured, and a
file-based four-chain field-MCSE report is now implemented and has focused
synthetic archive tests, but it has not been run on N256 files and its peak
RSS/I/O are unmeasured. Its merge holds at most two raw accumulators, an
18.5GiB floor at N256 before finalization temporaries; retain the required
20% headroom from a measured production-size peak before any such run. The
independent-start production driver and field-MCSE checkpoint/replay recovery
in the chain runner also remain unimplemented. One accumulator holds37
float64 full-grid
arrays:4.625GiB at N256 before finalization temporaries; two batch scales
therefore require at least9.25GiB of persistent accumulator state, in addition
to the field, JAX host buffers, checkpoints, and finalization temporaries. No
high-N memory profile or production job was run; the eventual Slurm memory
request must retain the required20% headroom based on measured peak RSS.

The one-chain diagnostic requires explicit
`CF4_R2_FIELD_MCSE_BATCH_SIZES` (exactly two sizes) and
`CF4_R2_FIELD_MCSE_MINIMUM_BATCHES`; it rejects a retained capacity that ends
inside either batch or cannot meet the declared minimum. No production batch
sizes or minimum were selected. A synthetic stationary-AR(1) study explored
32/64 states and eight batches per chain at the 512-draw cap; it does not
calibrate spatially correlated or actual sampler fields. Keep these values
provisional. Each scale can be finalized separately to avoid holding both
full map-output sets simultaneously. Checkpoint restore adopts serialized
arrays directly, avoiding another zero-filled accumulator allocation per
scale. Finalization peak RSS remains unmeasured; no GPU job was submitted.

#### Synthetic batch-scale calibration — 2026-11-04

`scripts/cf4_r2_mcse_ar1_calibration.py` ran 5,000 CPU-only replicates of
four independent stationary Gaussian AR(1) chains with 512 retained states
per chain. The candidate pooled batch-means estimator used 32- and 64-state
scales, 16/8 batches per chain, and compared MCSE/posterior-SD with the
provisional 0.10 threshold. At `phi=0.95`, the exact pooled ratio is 0.1353;
the 64-state estimator median is 0.1158 and only 5.56% of scalar replicates
put both point estimates below 0.10. At `phi=0.90`, exact is 0.0954 while the
64-state estimate p95 over scalar replicates is 0.1025. This reveals the
larger scale's sensitivity to slow correlation but also its conservative
behavior near the threshold. Scalar replicate quantiles are not a spatial
regional-p95 calibration; actual field cells are correlated and
heterogeneous. `MultiScaleFieldMCSEAccumulator.merge_independent_chains_inplace`
now pools complete equal-length chain sufficient statistics without copying
full maps; the 17-test field-MCSE suite includes an exact AR(1) estimator
comparison. This helper is not yet integrated with the file-based chain
runner/report, and no N256 memory or I/O profile exists. The detailed result
and limitations are in
`CF4_R2_FIELD_MCSE_AR1_CALIBRATION_20261104.md`. No R2 target, held-out data,
or GPU was accessed; R2 remains NO-GO.

#### Reporting-domain decision — 2026-11-04

The field-map summaries use the full periodic box plus one explicitly named
training-data-bearing region. `src/cf4_r2_field_reporting_regions.py` builds
that region from only the frozen population-major `train_keys`: union their
observed 128-grid count voxels over tracer populations, then lift each
3 cMpc/h coarse cell to its aligned 2x2x2 block of 1.5 cMpc/h native PM nodes.
This uses the shared box origin and the native node convention at coordinate
zero; held-out and buffered keys are not included. The helper rejects duplicate
or out-of-range keys and non-integer refinement, with five focused geometry
regressions in `tests/test_cf4_r2_field_reporting_regions.py`.

This mask is a reporting domain for cells that contain training count data; it
is **not** a claim that those PM nodes exhaust all likelihood influence. RSD,
volume integration, and interpolation can transmit information beyond the
same-coordinate coarse cells. The full-box summary remains mandatory, and the
region must be reported with its explicit cell denominator. The mask is wired
into the existing one-chain long-chain diagnostic; it does not read held-out
outcome values or generated-field/native-truth identities.

**Q-GOAL:** this makes field-level uncertainty measurement more executable,
but produces no posterior or validated z=0 map. MW/M31 are still
role-ambiguous and M33 unresolved; their observables must constrain those
roles on the same NEW evolved LG field at`<=0.3 cMpc/h`, with native truth
identities used only for calibration/evaluation. **Q-LEAN:** this is directly
required to report uncertainty rather than only posterior means; do not add a
GPU run merely to demonstrate the helper. Next, build the independently
initialized chain path, its warmup/restart semantics and the cross-chain map
comparison, and finish observation-
calibration and held-out cost contracts before considering any large
submission.

## Resource ceiling proposal

The latest measured exact-HMC cost is1.1422 H100 GPU-hours per transition
(job447829; only two transitions, so this is a planning estimate). The
maximum design is2,560 transitions total, before held-out scoring:

\[
4\times(128+512)\times1.1422\simeq2,924\text{ H100 GPU-hours}.
\]

The first-version3,200 allocated-GPU-hour ceiling is **withdrawn as
incomplete**, not replaced by a larger assumed budget. Its nominal transition
cost alone corresponds to about30.5 days on four continuously occupied
H100s; this omits the post-qualification held-out pass and implementation
costs. Re-estimate from the implemented scorer and chain pipeline before
setting a hard cap. Any final cap must cover every concurrent chain,
restart/replay, diagnostics and predictive evaluation; it must be enforced
without borrowing unused per-chain budget or silently extending after a stop.
H200/H100/A100 dispatch follows the typed-GRES policy (H200 first), and a
slower mode may exhaust the same cap earlier. Even an accepted resource cap
would be a bounded posterior attempt, **not a guarantee of convergence or
completion**.

## Decision and required review

**Q-GOAL:** a converged conditional-v6 posterior would be the first z=0
density/velocity field product at1.5 cMpc/h, but it would **not** alone meet
the active R2 objective while observation-selection calibration remains
unresolved. It is not the final high-resolution Local Group result. MW/M31
remain role-ambiguous and M33 unresolved. Their observables must ultimately
constrain those same roles on the same NEW evolved LG field at
`<=0.3 cMpc/h`; native truth identities remain calibration/evaluation labels
only.

**Q-LEAN:** one retained-state implementation and the necessary field
uncertainty diagnostics directly support the requested deliverable. Do not
repeat another sampler screen, surrogate ladder, gravity run, or uncalibrated
TF implementation. Preserve the raw field samples only if needed; compact
canonical states are required for the sealed held-out predictive pass.

The requested Astra contract review is complete. Before any large
submission, finish the production runner and field/score diagnostics, settle
the remaining source-matched observation-calibration route without reading
held-out outcomes, calculate the total run+prediction cost, and obtain the
user's explicit acceptance of both the conditional scientific scope and the
final resource risk. Until those conditions hold, do not read held-out
outcomes, launch chains, or change the active target.
