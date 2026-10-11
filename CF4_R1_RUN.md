# R1 — particle/observation connection and computation feasibility

User2026-09-13 approves the internal-IC joint state/history route and R1.
The first science delivery remains actual-data z=0 posterior maps; failed
ML repair lines stay closed. No new external audit of this routine entry.
Parent plan: [end-to-end replan](CF4_END_TO_END_REPLAN_20260913.md).

## First executable entry, not R1 closure

One Slurm GPU job runs focused regressions, a common-IC PM force/time ladder,
likelihood-only adjoint checks, then two short non-truth-initialized HMC chains.
Small periodic12 cMpc/h box with32^3 particles,2LPT,a=1/64→1. Baseline32^3
force mesh/max delta-a1/32; paired64^3 force mesh then delta-a1/64. Same
LPT positions, velocities, particle mass and white phases are checked.
Common32^3 readout is0.375 cMpc/h, not the actual LG0.3 deliverable.

Source: [particle forward](src/cf4_r1_particle_forward.py),
[runner](scripts/cf4_r1_particle_entry.py),
[configuration](config/cf4_r1_particle_entry_v1.json).
Use existing PMWD compatibility guards and BlackJAX chunked HMC, not a new
sampler framework. Installed PMWD's transfer is its EH approximation; this
is not a production transfer-function adoption or normalization fit.

Gaussian aperture operators return mass, centroid offset, mean peculiar
velocity and physical directional velocity variance from the SAME particles.
No cell-based identity labels, native halo IDs, best-seed selection or density
painting. Three predeclared probes are not identified MW/M31/M33. Their
overlapping masses are NOT a disjoint member partition, M200c or bound mass.
The explicit-weight CIC test includes mesh/particle ratio2; PMWD's default
scatter density normalization must not be mixed with unscaled momentum.
Weighted periodic force tests mass-split equivalence, common-mesh long/short
force addition and a position derivative. These are NOT padded nested-gravity
or independent-random-field covariance tests; no multi-resolution claim.

Synthetic data:21 Gaussian probe observations (3 log masses,9 offsets,9
velocity components) generated at the baseline; noise fixed in configuration.
Likelihood gradient checks exclude the white prior so a correct prior
derivative cannot mask a bad dynamics adjoint. Two directions/two epsilons,
unchanged2% engineering tolerance. Mismatch stops sampling but preserves
fidelity results. Two independent prior initializations;64 warmup+128 retained
steps each,8 leapfrog steps, fixed sampling step after warmup. Short ESS/unsplit
Rhat only for five diagnostic quantities; no mixing/coverage certification.
All white modes are free, no optimizer, no optimization-as-posterior claim.

The force/time ladder is the SAME solver at unchanged particle resolution.
It measures sensitivity; it is NOT the independent higher-fidelity calibration
required before R2. No same-generator mock is claimed as independent truth.
Saved products include phases, particle states, density/u/physical variance,
valid-cell masks, likelihood gradients, timings, chain states and limitations.

## Actual LG observation contract and unresolved work

Retain [the existing source-backed contract](config/cf4_lg_observation_contract_v1.json):
M31/M33 distance moduli, heliocentric LOS and two proper-motion components,
shared distance ladder terms and caller-supplied MW/solar nuisances. The four
existing contract regressions run here. Unknown COM/PM correlations, stellar
versus halo velocity offsets, LMC reflex and tracer overlap remain unresolved.
Existing assumed mass priors remain DISABLED. No new source values or diagonal
covariance are silently promoted by this engineering test.

Required later in R1: use calibrated particle observables at actual LG scale;
define new-state candidate/assignment support and unresolved-member cases;
link an appropriate M33 bound/enclosed-mass and motion operator without
assuming its identity or letting a host aperture stand for its existence.
TNG/SUBFIND truth can calibrate/evaluate, never seed generated candidates.
The bound/substructure machinery and verified local/coarse dynamics path are
still required. This entry implements the continuous moment portion only.
The source-backed filtered-observable strategy is motivated by
[Wempe et al.2024](https://arxiv.org/html/2406.02228v1); their calibrated halo
likelihood, backend, uncertainties and computational performance are not ours.

## Resources, decision and continuation

1GPU/4CPU/10GiB/1h via a40,a100,h100,h200, exclude syn06. Host peak estimate
8GiB (small particle/mesh arrays, compilation and existing library/test memory,
two32MiB chain arrays); +20%=9.6GiB rounded to10GiB. Application cap50min,
Slurm hard cap60min; maximum1GPU-hour of R1's cumulative4GPU-hour envelope.
No new large snapshot, RAMSES run, download, GPFS tests or manual launch.
New output root `/gpfs/kjhan/CF4/z0_density/r1_particle_entry_v1/job_JOBID`;
expected<0.2GiB. Existing products and unrelated work preserved.

Job automatically executes tests→paired evolution→adjoint→short chains→report.
Partial results persist on application cap; Slurm failure is not science GO.
Driver next reads the fixed job/result, judges physics and cost, and proceeds
within remaining R1 budget. Do not launch R2 on this entry alone. No whole-R1
completion or automatic large simulation claim. A hard failure is investigated,
not cured by loosening scientific tolerances or adding a new training sweep.

## Submission

Slurm **354568**, submitted2026-09-13 21:34:05 KST from committed/pushed
source `8f2aa0d9d8d3e2c67508164ecdd974ba89d11ccb`. Initial state
PENDING(Resources); no GPU node allocated and no numerical test result yet.
Python syntax compilation, Bash syntax and diff whitespace checks passed.
Numerical tests execute only in the allocation, not on the login server.
Requested1GPU/4CPU/10GiB/1h as above. Source unchanged after submission.

Result:
`/gpfs/kjhan/CF4/z0_density/r1_particle_entry_v1/job_354568/result.json`.
Progress: same directory `progress.json`.
Logs: `/gpfs/kjhan/CF4/logs/cf4_R1_particles_354568.{out,err}`.
Scheduler start estimates are provisional, not a promised start time. No
external polling/process-scan daemon or automatic subsequent science job.

Update2026-09-14: user requests `a100_pcie`. Partition is UP, includes
syn103 and permits this account. While354568 was still PENDING, updated
that SAME job to `a100_pcie,a40,a100,h100,h200` with `scontrol update`.
No cancellation, duplicate submission, resource increase or source changes.
Job ID, original submit time,1GPU/4CPU/10GiB/1h and syn06 exclusion retained.
The queued script/source pin is unchanged; this is a scheduler-side partition
override. Include a100_pcie in subsequent applicable GPU submissions.

## Entry result and authorized correction — 2026-09-14

354568 **COMPLETED/exit0** on syn103/a100_pcie,01:25:13–01:31:05 KST
(5m52s). Nine regressions pass. Likelihood-only directional derivative maximum
relative error0.0292%. Two chains retain128 draws each; sample divergences0,
acceptance .941/.922. However the five unsplit diagnostics have maximum
Rhat1.1504/minimum ESS6.36: **mixing is not established**. One warmup divergence
occurred in chain1. Warm gradient evaluations ~27–53ms are not cost per
independent posterior draw.

Same-particle force mesh32→64 changes the three fixed aperture masses by
−5.62%,−40.85%,−15.21%; then halving delta-a changes them by
+1.18%,+3.35%,−0.60%. This is resolution sensitivity, not measured error against
truth. No physical halo or M33 is identified and R1 is not complete.
Slurm MaxRSS9721416K (~9.27GiB) exceeds the Python-only5.38GiB report; use
the Slurm peak for follow-up memory sizing. Entry used352 of14400 allowed
GPU-seconds. There was no automatic downstream job after its completion.

User now authorizes addressing the important unresolved issues. One bounded
follow-up reuses this exact mock data and PM model and changes the numerical
comparison/sampler only. Sources:
[runner](scripts/cf4_r1_resolution_mixing.py),
[configuration](config/cf4_r1_resolution_mixing_v2.json).

- Same32^3 particles and LPT phases; three predeclared seeds, force meshes
  32^3/64^3/128^3 at max delta-a1/128, plus128^3 at1/256. Keep radius.75,
  centers, physical mass, observation units and noise unchanged. Compare
  mass/COM/velocity differences in physical units and the FIXED mock sigma.
  This tests convergence trend, not independent calibration, particle
  convergence or license to rescale masses. Finer PM is not ground truth.
- Sample the UNCHANGED original32^3-force/max delta-a1/32 likelihood with
  four independent prior starts per arm.128 warmup; fixed8 leapfrog/2048
  retained versus uniformly random16–48/512 retained. Expected retained
  integration work is65536 steps per arm; actual work/timing is recorded,
  warmup costs are separate and not matched. Same step-size adaptation,
  target .8 and cap .1 in both arms. The old cap was .3, so comparison to
  the old entry is not an isolated trajectory-only causal experiment.
  State-independent length mixing preserves the Metropolis HMC target;
  the initial hypothesis is insufficient travel, not a diagnosed adjoint bug.
  [BlackJAX dynamic HMC](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/dynamic_hmc/index.html).
- Use rank-normalized/folded split Rhat and library bulk/tail ESS on48
  predeclared summaries: logL, white norm/three coordinates, DC and12
  Fourier projections,21 physical observables and9 physical dispersions.
  Engineering criterion Rhat<=1.05, ESS>=100 and no retained divergences is
  NOT production convergence or coverage; disclose every failing summary.
  [Rank/folding diagnostics](https://arxiv.org/abs/1903.08008).
  Keep FP64 final restart states; archived draw white arrays are FP32 and
  are explicitly not exact restart states. No best-seed selection or pooling
  with the old short chains.

The short fixed8 trajectory lengths in the entry were .512/.612. Enlarging
them is a testable remedy for weakly constrained white-mode correlation; high
acceptance alone is not evidence of independent samples. Longer/randomized
trajectories may instead expose nonlinear gradient/energy problems; retain
that outcome rather than silently dropping failures or changing the target.

This directly addresses numerical reliability needed before connecting real
MW/M31/M33 data, but the probes remain unnamed apertures. New-state candidate
assignment, bound M33 readout and the local/coarse dynamics path remain
explicitly unresolved. No R2/actual-data inference or new neural fit here.

Resources:1GPU/4CPU/**16GiB/2h**, a100_pcie,a40,a100,h100,h200, exclude syn06;
application cap110min. Estimate13GiB peak (observed9.27GiB plus finer-mesh
executables, diagnostic imports and <=~1GiB live trace/save buffers), +20%
rounded16GiB. Clear no-longer-used JAX executable caches between fidelity
and sampling, not storage-system caches. Output estimate<2GiB. Even the full
2h allocation plus entry is below R1's4GPU-hour cap. Numerical tests run only
inside this allocation. Preserve every entry artifact and unrelated work.
The same job performs tests→fidelity→both sampler arms→diagnostic report;
bounded incomplete execution is reported as such, not automatically extended.

Follow-up **358369** submitted and started2026-09-14 **11:26:08 KST** on
**syn103/a100_pcie** through Slurm, source `32ae8c51c309db8115f0e20fdab2a16703a05891`
committed and pushed beforehand. Initial RUNNING is not a numerical pass.
Allocation ends by13:26:08 KST; application cap is110min from Python setup.
Result/progress: `/gpfs/kjhan/CF4/z0_density/r1_resolution_mixing_v2/job_358369/`.
Logs: `/gpfs/kjhan/CF4/logs/cf4_R1_resmix_358369.{out,err}`.
No separate polling daemon or automatic R2 job; the fixed calculation and
postprocessing are included in this single allocation.

## Comparison358369 completed; particle accuracy is next

358369 COMPLETED/exit0 on2026-09-14 12:43:16 KST, elapsed1h17m08s;
tests8/8, Slurm MaxRSS5031884K (~4.80GiB). The12 force/time cases, baseline
reproduction and both four-chain sampler arms finished. Sampler transitions
and physical readouts are separate evaluations, not part of the12-case count.
The numerical report's status is COMPLETE_DRIVER_JUDGMENT_REQUIRED, not an
automatic science pass.
Fixed8: max rank/folded split Rhat1.00947, minimum bulk/tail ESS820.34/1468.45.
Random16–48:1.01292,419.33/543.85. Both48-projection engineering criteria
pass, warmup/retained divergences0. Retained integration steps65536 versus
65590; seconds per minimum bulk ESS2.18 versus4.18. Keep fixed8 with enough
draws as the small-model baseline. Longer trajectories are not required by
these data; changed warmup/step cap/chain length preclude attributing the
original short-run miss uniquely to integration length. Neither arm certifies
full-field convergence, actual-data coverage or fine-PM mixing.

At three seeds/fixed delta-a1/128, force32→64 versus64→128 differences have
21-observable RMS .69456→.21444 times FIXED mock sigma; maximum aperture-mass
changes56.63%→10.40%. Halving delta-a at force128 gives RMS .03466 sigma and
maximum mass change3.10%. These are differences, not known truth errors.
Same-data baseline observables reproduced to7.1e-15. R1 used352+4628=4980
Slurm GPU-seconds (1h23m);2h37m of the4h numerical envelope remains.

User now authorizes the recommended particle/force/observable accuracy work.
The next single comparison is [particle-resolution v3](config/cf4_r1_particle_resolution_v3.json),
implemented in [the runner](scripts/cf4_r1_particle_resolution_run.py) and
[supplied-state forward](src/cf4_r1_particle_resolution.py).

| Test | Particle grids | Force grid | max delta-a |
| --- | --- | --- | --- |
| Particle sampling | 32³→64³→128³ | 128³ fixed | 1/256 |
| Force | 128³ fixed | 128³→256³ | 1/256 |
| Time | 128³ fixed | 256³ fixed | 1/256→1/512 |

Same three seeds, box12 cMpc/h, original aperture radius/centers/noise. The
existing initial **displacement and velocity**, not wrapped coordinates, are
periodically Fourier-resampled onto nested Lagrangian grids. Original coarse
nodes, mean displacement/velocity and total mass must agree. Real even-grid
Nyquist modes use the library's split-bin convention, tested with a constant,
a sinusoid, a Nyquist wave and nested-node recovery.
[SciPy resampling convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html).

This holds ONE interpolated 2LPT state fixed, including its original LPT
discretization/aliasing error. **No fresh high-k modes are added.** It is a
particle quadrature control, NOT a high-resolution LCDM prior, production
zoom-IC method or differentiable fine-IC parameterization. Recomputing fine
2LPT or adding high-k would change the initial state and confound this test.
Particle masses are4.537e9,5.671e8,7.089e7 Msun/h; fine force cell.046875.

The supplied-state forward uses the identical PMWD nbody_init/nbody_step in
lax.scan, avoiding hundreds of unrolled steps. It is NOT a new independent
solver or a validated fine adjoint. A small regression compares it to ordinary
PMWD; all three N32 endpoints must reproduce358369 before drawing physical
conclusions. Mean/physical variance maps use a common64³ readout (.1875),
explicit particle mass weights and conserved total mass/momentum.

Two additional plane-wave controls at128³ particles, force128³/256³ compare
against the exact pre-shell-crossing Zel'dovich trajectory, not another PM
reference. Wavenumber4, final linear amplitude.3 gives minimum Jacobian.7.
They share the background growth table, so this validates spatial force/time
evolution and units in a restricted regime, not the background ODE or collapsed
halos. Compare particle displacement/velocity and the SAME aperture readout.
[Plane-parallel exactness](https://arxiv.org/abs/1502.07389).

New-field MW/M31/M33 identification is still unresolved: these fixed probes
must not inherit halo identities, and M33 boundness cannot be established by
positive aperture mass or effective particle count. The goal contribution
is determining whether same-state physical predictions are numerically stable
before real LG likelihoods, not bypassing those identification requirements.
An independent collapsed-structure solver comparison remains required. Existing
legacy GRAFIC export recomputes velocity rather than exporting this exact state,
so it is NOT invoked as an automatically matched RAMSES comparison. Preserve
exact FP64 finest initial displacement/velocity and final particles to support
the later verified transfer contract; no RAMSES launch is included here.

One Slurm1GPU/4CPU/16GiB/1h, application cap50min, eligible a100_pcie,a40,
a100,h100,h200, exclude syn06. Conservative host peak estimate13GiB including
larger FFT/compiler/readout/interpolation arrays and snapshots, +20% rounded
to16GiB. Largest single256³ real FP64 mesh is128MiB, several vector/complex
workspaces coexist; do not extrapolate only particle-array bytes. GPU memory
is measured, preallocation disabled. Estimated output<1.5GiB; no large new
snapshot download, GPFS diagnostics or process-scan monitoring. Full allocation
would put R1 at2h23m, still below4h. No HMC refit, new ML, R2 or independent-
solver calibration is automatically claimed. Report numerical differences
without changing observation errors, radius or acceptance criteria.

Submitted **359000** at2026-09-14 **13:17:10 KST**, source
`62ccea176d5ed237b373fd1a921ca86f3e89493b` committed/pushed before launch.
Initial **PENDING(Resources)**, no assigned node or numerical test result yet.
1GPU/4CPU/16GiB/1h, a100_pcie,a40,a100,h100,h200, exclude syn06. Static
Python/Bash syntax and diff checks pass; numerical tests run in the allocation.
Result/progress: `/gpfs/kjhan/CF4/z0_density/r1_particle_resolution_v3/job_359000/`.
Logs: `/gpfs/kjhan/CF4/logs/cf4_R1_particles_v3_359000.{out,err}`.
The one job performs tests→three-seed particle/force/time comparisons and two
analytic controls→common-field snapshots and report. No duplicate/manual run,
polling daemon, automatic R2 submission or promotion is attached.

359000 **FAILED/exit1** at2026-09-14 13:40:03 KST,5s after allocation on
syn103/a100_pcie. Driver import failed BEFORE main/tests/numerical cases:
`scripts/cf4_r1_particle_resolution.py` shadowed the same-named src library
because the executing script directory comes first on Python's search path.
This was a driver naming defect, not a physical-resolution failure, GPFS issue,
memory exhaustion or the incidental hwloc warning. Failed logs are preserved.

Technical recovery within the already-authorized calculation: rename only
the executable driver to `cf4_r1_particle_resolution_run.py`, update the Slurm
entry point and add a search-order regression. The particle, force, timestep,
IC, seed, observable and resource choices stay unchanged; cumulative resource
accounting adds the failed5s (4985s before retry). A standard-library-only
module-resolution check is permitted on the login node; numerical imports and
all seven focused tests still execute only in Slurm. Syntax compilation alone
did not catch this import failure and is not claimed as a runtime validation.

Retry **359203** from pushed source `90bae45b9f0b48e2fa1c2f90055922e00e861351`
started2026-09-14 **14:13:01 KST** on syn103/a100_pcie and COMPLETED/exit0 at
**14:15:04**, elapsed2m03s. All7 regressions pass; all15 three-seed numerical
cases and two analytic controls finish. Import failure is resolved. Slurm
MaxRSS2345516K (~2.24GiB), Python peak2.52GiB; retain both measurements.
Same-state N32 endpoint errors<=1.07e-11 cMpc/h and1.97e-8 km/s versus358369.
Total R1 Slurm GPU allocation time is352+4628+5+123=5108s, leaving9292s
(2h34m52s) of4h. Fast forward-only scan timings do not establish fine-adjoint
or effective-posterior-sample cost.

| Matched comparison | Max probe-mass change | Max centroid change (cMpc/h) | Max mean-velocity change (km/s) |
| --- | ---: | ---: | ---: |
| Particles32³→64³, force128³ | 31.40% | .20571 | 21.236 |
| Particles64³→128³, force128³ | 15.81% | .11628 | 7.923 |
| Force128³→256³, particles128³ | 2.902% | .04404 | 2.989 |
| Halve timestep at particles128³/force256³ | .1240% | .00167 | .103 |

These three-seed differences decrease, but particle sensitivity remains.
More importantly the new **planar analytic control has a large discrepancy**:
force128³ gives displacement/velocity relative RMS .13367/.13568; force256³
gives .80493/.82675 at the same128³ particles. This is not a passing analytic
accuracy test even though the7 implementation regressions passed. Large
displacement/velocity differences coexist with broad-aperture residuals only
.0242/.0883 times mock sigma; smooth observables alone cannot certify dynamics.

Driver decision: **HOLD numerical-accuracy promotion / no R2 launch**. Do not
infer a unique PMWD defect, a physical resolution floor, or a false analytic
reference without checking. Next bounded cause separation must verify the
planar initial state/reference and units, initial force response and growth
history/particle-lattice dependence before choosing any correction or more
resolution. Independent collapsed-halo calibration and actual LG membership
remain unresolved. No further job or external review has been launched by
this completion record; important verified findings follow the advisory policy.

Final artifacts: `/gpfs/kjhan/CF4/z0_density/r1_particle_resolution_v3/job_359203/`.
`result.json` contains the complete quantitative comparisons and unresolved
flags; `reference_particles_seed*.npz` and common readout fields are preserved.
Logs: `/gpfs/kjhan/CF4/logs/cf4_R1_particles_v3_359203.{out,err}`.

User-approved next cause separation is frozen in
[CF4_R1_PLANAR_DIAGNOSIS.md](CF4_R1_PLANAR_DIAGNOSIS.md). It does not change
the production dynamics or promote the prior accuracy comparisons.
