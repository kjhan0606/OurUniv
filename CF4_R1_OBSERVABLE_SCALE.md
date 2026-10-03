# R1 observable-scale decision bundle — 2026-09-15

360338 completed in1h58m13s, both RAMSES endpoints at a=1. Initial-state
handoff passes. AMR8/9 density differences by ascending bands are .12%,.52%,
5.67%,15.30%; probe masses differ <=.136%, velocity components <=.214km/s.
Physical variance changes ~10–17%. This is not fine-scale accuracy closure.

Next bounded work reuses those endpoints and CIC/TSC, with the SAME three
predeclared centers and Gaussian widths .3,.75,1.5 cMpc/h. Report mass,
centroid, vector mean velocity, sigma AND variance, effective particle counts.
Reproduce the old .75 variance results before accepting the new readout.
Widths are not numerical map resolution. No new simulation, no wider window
selected to disguise unresolved fine structure, no covariance fitted to one seed.
CPU4/8GiB (estimated ceiling6.5GiB plus20%, rounded),10min, no GPU; only small
JSON output, no snapshots. Driver evaluates routine results; no external audit.

Q-GOAL: separate mean-flow applicability from unresolved dispersion before
actual environment/LG inference. Q-LEAN: reuse final particles and existing
moments, one small readout, no assignment/mesh sweep or new framework.
MW/M31/M33 remain UNIDENTIFIED in these probes. The following substantive
development must identify generated-state host candidates and ambiguous roles,
then explicitly assess whether a bound/deblended M33 candidate is measurable.
HOP hosts alone cannot provide that subhalo constraint. Existing LG contract
mass priors remain disabled; Gaussian mass is not M200c or bound mass.
Do not launch R2 production or claim R1 complete from these diagnostics.

## Completed361450 and driver disposition

Source35e8eff; four endpoints/three widths completed, old .75 variance
reproduction passes. Results /gpfs/kjhan/CF4/r1_observable_scale/job_361450/result.json.
At width .3, AMR9-TSC maximum mass difference3.876%, mean-velocity vector
difference .3673km/s, sigma difference5.54%; corresponding AMR9-CIC values
8.347%,1.4804km/s,9.41%. AMR9-AMR8 mass difference .170%, velocity .0719km/s,
sigma4.37%. At .75 and1.5, AMR9-AMR8 sigma differences8.19% and6.13% persist.
The dispersion sensitivity is not monotonically larger at smaller windows;
it cannot be attributed solely to the chosen aperture width. Fine force
resolution changes the readout, but its asymptotic convergence is unproven.

Driver decision: TSC is the better-supported candidate for these mass/mean-
flow probes, NOT a globally certified kernel or calibrated LG likelihood.
Stop aperture/assignment-order sweeps. Do not erase sigma by broadening
windows or conflate it with uncertainty on mean velocity. Existing one-seed
interpolated-IC control cannot establish errors for full-power LG structures.
Next substantive target is generated-state host/substructure measurability:
reuse the existing HOP interface on archived states, compare host identity,
mass and velocity across solvers, report role ambiguity and M33 limitations.
No assumption that this unconditioned fixture contains an actual LG analogue;
absence is a result, not license for best-seed selection. Bound M33 and the
multiresolution/production-adjoint route remain explicit unresolved work.
