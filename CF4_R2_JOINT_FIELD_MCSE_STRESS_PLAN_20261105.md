# R2 joint field-MCSE stress test — result, 2026-11-05

## Decision question

Test one predeclared, spatially correlated stationary alternative whose true
finite-sample velocity-mean error ratio exceeds0.10, preserving paired gate
outcomes across all map quantities, regions and both MCSE batch sizes. This is
a case-specific stress test, not a general production-gate calibration, a
batch/threshold/correlation sweep, or a production-chain simulation.

The preceding result (Syntax job450009) is closed as a qualified negative
control, not calibration. Astra identified that it did not preserve paired
gate events, used asymptotic velocity references, and did not establish joint
spatial stationarity. The amended contract is in
`CF4_R2_CONDITIONAL_V6_POSTERIOR_RUN_CONTRACT_DRAFT_20261104.md`.

## Frozen stress design

- Grid16^3; four independent chains;512 retained synthetic states per chain;
 256 independent replicates; batch sizes32/64; minimum8 complete batches per
  chain. Synthetic central reporting sphere is fixed at radius0.2 box lengths
  and contains147 cells. The full16^3 box is the second reporting region.
- Use one fixed spatial Gaussian correlation length of2 cells, with the
  periodic Fourier covariance already used by the current generator. Use a
  **constant temporal AR coefficient over all cells for each Gaussian field**:
  density0.20 and velocity components(0.20,0.90,0.20). Start each path with a
  draw from the same filtered Gaussian law as its subsequent innovations.
  Since each AR coefficient is scalar over the grid, this yields a jointly
  stationary spatial Gaussian field. Draw density and velocity latent fields
  independently.
- Use spatially homogeneous occupancy probability0.20 and a constant
  two-state Markov persistence0.20. Initialize each cell from Bernoulli(0.20)
  and use independent cell/time uniforms. This is a stationary product process,
  independent of the Gaussian fields. Set
  `rho=O*exp(0.02*X-0.0002)` and use occupancy `O` as the velocity validity
  mask. The combined synthetic state is jointly stationary in space and time
  on the periodic grid.
- Under this fixed design, density-mean and occupancy mean ratios use their
  exact finite-length references. Density-SD and velocity-SD references are
  explicitly asymptotic influence-function references and descriptive only.
  For each velocity-mean component, estimate an independent finite-sample
  oracle for the pooled occupied-state ratio estimator. Conditional on four
  chain occupancy histories, define
  `Q=sum_chain sum_t sum_s O[chain,t] O[chain,s] phi_v**abs(t-s)` and
  `M=sum_chain,t O[chain,t]`; conditional variance is `Q/M**2` because the
  latent velocity SD is1. The target reference is
  `sqrt(E[Q/M**2 | M>0])`, not `E[sqrt(Q/M**2)]`. Report the exact model
  probability `P(M=0)`; any sampled `M=0` is a validity failure and makes the
  oracle inconclusive. Estimate the expectation from exactly250,000 independent
  histories using bounded chunks and the linear-time recurrence for `Q`, with
  streaming mean/variance only (no pairwise covariance matrices). Use an
  independent frozen oracle RNG stream. Define the0.5% requirement as the
  delta-method relative Monte-Carlo standard error of the square-root
  reference, and require the one-sided97.5% lower confidence bound for the
  velocity-y reference to exceed0.10 before generating gate replicates.
  Otherwise stop as inconclusive without auto-extension. Report the same
  reference for both regions: under the homogeneous generating law the true
  cellwise precision map is constant, so the true regional p95 is equal in
  the full box and central sphere; their estimated-p95 distributions still
  differ because of region size and spatial dependence.
- At every replicate, retain paired numerical regional summaries and the
  Boolean result for every constituent of `G_field`. MCSE conditions are
  evaluated at both batch sizes and in both regions. The cross-chain
  mean/SD-map disagreement is batch-size-independent for these same complete
  retained states and is evaluated once per region, not duplicated across
  batch scales. `G_field` passes only if all MCSE and cross-chain conditions
  pass and every required map cell is defined and finite. Report the MCSE-only
  pass event separately so an all-reject result cannot be mistaken for useful
  gate discrimination. The scalar-chain gate is separate and is not calibrated.
- The fixed alternative is precision-inadequate when its true finite-sample
  velocity-y mean-error regional-p95 ratio is greater than0.10, established
  before gate replicates by the oracle lower bound above. A false acceptance
  is any replicate where the complete observed `G_field` passes. Do not define
  truth using realized cross-chain disagreement: that is an observed gate
  diagnostic, not a latent precision truth. Invalid/nonfinite maps are gate
  failures and remain in the denominator. Report the exact numerator and all
  256 trials. Report the exact one-sided95% Clopper-Pearson upper bound required
  by the contract and, separately, the97.5% bound used with the97.5% oracle
  lower bound for a nominal Bonferroni familywise95% statement; the oracle's
  Student-t/delta-method coverage is approximate, not an exact finite-sample
  confidence guarantee. The contract risk
  criterion passes only if its95% upper bound is<=0.05. No
  marginal-probability multiplication or independence assumption is permitted.

## Execution boundary

One CPU-only Syntax `a10` Slurm job, no GRES, no manual node execution. Use a
fresh job-ID output directory, pin the pushed source commit, and run focused
tests before calculation. Static live-array accounting gives about64MiB for
the four Gaussian paths,9.25MiB for all four chains' two-scale accumulators,
and under3MiB of chunked oracle arrays. Allowing for Python/NumPy, summaries
and test-process overhead, expected peak RSS is conservatively bounded below
1GiB; job450009's measured 225MiB peak is supporting context, not the new-job
measurement. A2GiB request leaves at least20% headroom under this estimate.
The 2-hour cap exceeds the prior 36-minute three-scenario generator run and
includes the new oracle, with no automatic extension. Persist paired
per-replicate outcomes
and constituent flags so a canceled/failed job leaves auditable partial
evidence; do not automatically retry or extend. Save oracle uncertainty,
references, job/accounting data and logs. No target evaluation, CF4 input,
held-out outcome, N256 chain, map promotion or gravity simulation.

## Interpretation and decision

A successful risk-bound result bounds false acceptance only for this one
synthetic stationary alternative. It does not establish the gate's ability to
accept genuinely precise fields and an all-reject result is not evidence of
useful discrimination. It does not calibrate other posterior dependence
patterns, the scalar-chain gate, production N256 precision, observation
selection, convergence or held-out prediction. A failed or inconclusive result
keeps `G_field` NO-GO; do not tune a replacement threshold from these same
outcomes. R2 remains NO-GO until the actual observation law, sampler and full
precision contract are independently defensible.

**Q-GOAL:** this tests a required uncertainty gate for the requested
CF4+galaxy-conditioned z=0 posterior at1.5cMpc/h; it produces no posterior or
held-out prediction. MW/M31 remain role-ambiguous and M33 unresolved. Their
observables must ultimately constrain those same roles on the same NEW evolved
LG field at<=0.3cMpc/h, with native truth IDs used only for calibration and
evaluation.

**Q-LEAN:** one fixed, directly motivated correlated stress case, reusing the
existing report and estimator; no parameter ladder, target/sampler call,
public-data search, email, held-out read, GPU calculation or simulation.

Status: **Syntax Slurm job450919 completed, exit0, in11:04** on `a10`/`syn04`
from source commit `c2053cf3f6433f0e7f4746f655698389f7521175`; MaxRSS was
262,012KiB under the2GiB request. All five focused startup suites passed
(43 tests total). The oracle processed250,000 histories and qualified the
predeclared alternative: the velocity-y finite-sample reference is0.1090724,
with a one-sided97.5% Student-t/delta-method lower bound0.1090667>0.10. No
zero-occupancy histories occurred. Across256 paired replicates there were
0 false acceptances; exact one-sided Clopper-Pearson upper bounds are1.1634%
at95% and1.4306% at97.5%. This meets the proposed risk bound for this
alternative only.

Crucially, `G_field`, the MCSE-only event, and the cross-chain event each
passed0/256 replicates. Both batch scales failed the velocity-y mean-precision
condition in both regions on every replicate. The cross-chain map statistic
also failed for all velocity components. This is an all-reject negative
control: it bounds false acceptance here but does not measure acceptance of
adequately precise fields or establish useful discrimination. The report and
raw artifacts are `CF4_R2_JOINT_FIELD_MCSE_STRESS_RESULT_20261105.md` and
`/gpfs/kjhan/CF4/z0_density/r2_joint_field_mcse_stress_20261105/450919/`.

Astra's read-only result audit is **PASS for this fixed negative-control
result only; NO-GO for production-gate qualification or R2 exit**. It notes
that the cross-chain comparator measures spread among individual-chain means,
not pooled-map MCSE; for four comparable independent chains, that spread is
approximately twice pooled MCSE. Therefore its0.10 cutoff is an additional,
stronger per-chain-agreement condition, not the0.10 pooled-precision
requirement. Do not call its rejection evidence of nonstationarity or infer
overconservatism from an all-reject result. A positive-control calculation is
deferred until the scientific relationship between these two requirements is
explicitly reconciled; no threshold or persistence sweep is authorized.

The result confirms no posterior, held-out data or R2 target was accessed.
The observation-selection denominator remains unresolved and R2 stays
NO-GO. No sampler or production qualification follows from this study.
