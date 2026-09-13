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

Implementation prepared; submission and numerical outcome recorded below.
