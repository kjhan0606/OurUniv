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
velocity modes, band powers, and fixed regional density/flow summaries; shell
powers alone are insufficient. Name the implementation and ensure it handles
the independent chain starts and unequal/incomplete chains without silently
truncating diagnostics to a favorable suffix.

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

The first-version `p95 MCSE(mean)/posterior_SD <=0.10` remains only a
candidate threshold, not yet a complete field criterion: batch lengths,
regional domains, zero/invalid-cell policy, cross-chain disagreement and
posterior-SD precision must be fixed in the tested implementation. If these
cannot be estimated stably at the proposed draw count, report insufficient
Monte Carlo precision rather than promote the maps.

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

## Current implementation readiness (read-only code inspection)

The proposed schedule and diagnostics are **not executable in the current
runner**:

- `scripts/cf4_r2_n256_joint_pilot.py` still runs one chain from one supplied
  initializer/restart. Its optional `long_chain` mode requires
  `CF4_R2_AFFINE_FORCE=1`; `src/cf4_r2_chain_schedule.py` defaults to48
  proposals,12 warmup, then fixed4/8-step trajectories—not the proposed
  128/512 schedule. It does not construct independent chain starts or
  implement the stated warmup-end viability review. The lossless state store
  is now wired into this legacy long-chain path and redundant every-eighth
  full-field dumps were removed; this improves persistence but does not turn
  the diagnostic into the proposed production pipeline.
- `src/cf4_r2_posterior_moments.py` accumulates posterior sample moments and
  occupancy, but no fixed-batch field means or map-level MCSE. Its output
  explicitly sets `posterior_MC_error_estimated=False`.
- `src/cf4_r2_chain_diagnostics.py` supports scalar traces only, while
  `scripts/cf4_r2_two_chain_report.py` reads two existing chains and has no
  held-out scoring stage. Neither implements the proposed four-chain,
  all-field contract.
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
   rejection-repeat/resume cases (7/7 tests pass). It is not yet connected to
   an independent-start production driver.
2. The candidate field criterion is incomplete. The implementation must
   define reporting regions, batch lengths/count, nonfinite and zero-variance
   handling, cross-chain map disagreement, and posterior-SD-map precision;
   do not use a whole-box p95 of mean MCSE as the sole pass.
3. Four starts must independently draw all `N^3+24` standard-normal
   coordinates. Specify RNG streams, bounded finite-support retries and
   numerical-failure handling. The warmup adaptation, trajectory, freeze
   point and early-stop review remain unimplemented.
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
