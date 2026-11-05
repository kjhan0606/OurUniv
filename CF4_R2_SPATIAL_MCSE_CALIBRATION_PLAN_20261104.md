# Spatial field-MCSE synthetic calibration — 2026-11-04

## Purpose and boundary

Calibrate the existing two-scale field batch-means estimator under stationary
spatial maps whose temporal persistence varies with location. This addresses
the gap left by the scalar AR(1) study: spatial cells are correlated, so a
regional p95 of cellwise MCSE ratios does not have the scalar-replicate
distribution. This is estimator-method work only. It does not evaluate the
CF4 target, establish stationarity or convergence of any R2 chain, access
held-out outcomes, produce a posterior, or promote a density/velocity field.

The calculation will call the production
`MultiScaleFieldMCSEAccumulator` at every retained synthetic state. Four
independent chains of 512 states will be generated per replicate on a periodic
`16^3` grid. Three innovation correlation lengths (0, 1, and 2 grid cells)
will change spatial dependence without changing each cell's known temporal
law. Density uses a positive mean-one lognormal variate gated by a stationary
two-state Markov occupancy process; each velocity component is Gaussian and
conditionally observed on the same occupancy mask. Occupancy probability
varies spatially from 0.2 to 0.8. Density, occupancy, and velocity temporal
AR coefficients vary smoothly across the box. Every chain starts in its exact
stationary marginal law.

The diagnostic compares fixed-batch scales 32 and 64 (eight batches per chain
at the larger scale) at the provisional 0.10 MCSE/posterior-SD threshold. It
uses 256 independent four-chain replicates per correlation length. The full
box and a fixed central synthetic reporting sphere (radius 0.2 box lengths)
are reported with explicit cell counts. For each observable it reports the
true regional p95 ratio, the replicate distribution of estimated regional
p95, and false-pass/fail frequencies with Wilson intervals. Density-mean and
occupancy references use exact finite-length autocovariance sums. Density-SD
and conditional velocity mean/SD references use stationary long-run
influence-function/delta-method variances; they are explicitly asymptotic
references. This does not settle the production threshold or batch sizes.

## Bounded execution contract

- Run one CPU-only Slurm job on Syntax `a10`, one CPU, 2 GiB RAM, four-hour
  wall cap; no GPU/GRES and no manual node execution. The modeled live arrays
  are below 1 GiB; the 2 GiB allocation is at least 20% above that conservative
  expected peak. Slurm MaxRSS will be reported as an observed resource result,
  not inferred from the array estimate.
- Pin the committed source revision and run the focused synthetic regression
  tests in the allocated environment before the study.
- Use a fresh output directory keyed by Slurm job ID. Refuse to overwrite.
  Save JSON and Slurm stdout/stderr; no N256 production artifacts are read or
  written.
- A source/hash mismatch, focused-test failure, nonfinite estimator output,
  or resource/time stop ends the job. No automatic retry, parameter ladder,
  threshold retuning, or interpretation as an R2 chain is permitted.

## Startup correction — 2026-11-04

The first Syntax submission, job450007, failed six seconds after start because
the pinned `circle` Python environment does not install `pytest`. The synthetic
study never started and no tests ran in that allocation; its job-specific
output directory is preserved empty. The local regression uses only the
standard-library `unittest` framework, so the wrapper is corrected to run that
same focused test file with `unittest discover`. One source-pinned retry of the
identical design and resource cap is appropriate: this is an environment/test-
runner wiring failure before any calculation, not a scientific/resource-gate
failure. No correlation length, batch scale, threshold, replicate count,
target, or memory/time cap changes.

## Scientific relation and limitations

Q-GOAL: calibrating the regional MCSE machinery is necessary to quantify
uncertainty for the requested CF4+galaxy-conditioned z=0 field at 1.5 cMpc/h,
but creates no such posterior or held-out prediction. MW/M31 remain
role-ambiguous and M33 unresolved; their observables must ultimately constrain
those same components on the same NEW evolved LG field at `<=0.3 cMpc/h`,
with native truth identities used only for calibration/evaluation.

Q-LEAN: one spatially correlated synthetic design directly tests the known
regional-percentile weakness; it does not add another CF4 target evaluation,
mock catalogue, sampler screen, gravity run, or broad gate suite. The source-
matched CF4/Tempel/SDSS-PV availability/group-selection law remains unresolved.
The separate N256 I/O/peak-RSS profile of the four-chain file report remains
required before production-sized field artifacts can be processed. R2 remains
NO-GO until observation scope, sampler qualification, field MCSE, and held-out
prediction are all defensible.

## Completed result and interpretation — 2026-11-05

Syntax Slurm job 450009 completed on `a10` with exit 0 in 36:01 (2 GiB
requested, MaxRSS 224,956 KiB). It is a reproducible synthetic-estimator study only. The
machine-readable result is
`/gpfs/kjhan/CF4/z0_density/r2_mcse_spatial_calibration_20261104/450009/result.json`.

For the 147-cell synthetic central reporting region, the declared velocity-y
reference p95 is 0.110984, above the provisional 0.10 ratio threshold. Per-scale
false-pass fractions relative to that declared reference are:

| Spatial correlation length (cells) | Batch 32 | Batch 64 |
|---:|---:|---:|
| 0 |25.78%|1.17%|
| 1 |39.06%|7.42%|
| 2 |48.05%|23.83%|

At length 2 / batch 64 the Wilson 95% interval is 19.02–29.41%. Larger batches
helped in this case but did not make this individual regional point-estimate
criterion dependable. These are **marginal rates for one metric, scale, and
region**, not false-acceptance probabilities for the conjunctive production
gate; the saved output does not preserve paired replicates needed to compute
that joint event. Do not multiply marginal rates or assume independence.

This is a qualified negative-control result, not a calibration of the
production precision gate. In particular:

- Density-mean and occupancy references use finite-length sums, but the
  density-SD and conditional velocity references are influence-function /
  asymptotic references. Label the rates above as false passes relative to the
  declared asymptotic velocity-y reference, not exact finite-sample error.
- The generator gives each cell its stationary temporal marginal, but its
  spatially varying AR coefficients do not in general make the *joint spatial
  map* stationary. The impact on regional-p95 distributions is unmeasured.
- `_regional_p95` discarded nonfinite cells before taking percentiles, and its
  finite-cell counts were not propagated into the saved result. The saved
  result cannot establish whether any invalid cell was present, so the
  promised fail-closed gate was not verified. The current source has since
  been hardened to stop on any nonfinite regional cell; this source correction
  does not retroactively validate the archived JSON.

Astra's read-only mid-course review conditionally accepted the run as a
synthetic diagnostic, rejected using it to calibrate the proposed production
gate, and recommended closing it without a batch/threshold/correlation
sweep. The driver independently checked the result table, reference types,
generator covariance and finite-cell path, and adopts that disposition. Keep
regional-p95 point estimates descriptive; do not derive an inflation factor
or lower threshold from these development results. Future precision
qualification must specify the actual joint acceptance event, its tolerated
false-acceptance probability, exact versus asymptotic reference, and
fail-closed treatment of undefined/nonfinite cells before any necessary new
calibration.

Q-GOAL: this prevents an unqualified precision gate from promoting the
requested CF4+galaxy-conditioned 384 cMpc/h, N256 / 1.5 cMpc/h z=0 posterior. It
creates no posterior, held-out prediction, or observation-selection
calibration. MW/M31 roles remain ambiguous and M33 unresolved; their
observables must constrain those same roles on the same NEW evolved LG field
at <=0.3 cMpc/h, with native truth IDs reserved for calibration/evaluation.

Q-LEAN: reuse this result; no new sampler, simulation, threshold sweep, or
diagnostic ladder follows. The separate one-CPU N256 report memory/I/O profile
remains a resource measurement only and does not repair this precision gate.
R2 remains NO-GO.
