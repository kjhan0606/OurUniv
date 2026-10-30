# R2/5 same-field selected-group radial measure

The source identity issue is closed. This bundle addresses an actual missing
connection: previous group-distance integrals used a homogeneous r^2 radial
measure despite predicting velocities from a spatially varying evolved field.
Implement `dd*d^2*rho_F(d)^b*S_group(d,F)` as an explicit caller-supplied
selected-tracer measure in BOTH numerator and denominator of the conditional
FP group factor. The field density and velocity must come from the SAME state.
Distance-independent group amplitudes cancel; shape and inclusion need not.

This is a power-law tracer assumption, not physical calibration. S_group is
group/sample inclusion, NOT another copy of the individual FP fn correction.
Its effective form conditional on redshifts/associations must eventually be
derived from the joint observation model. Normalizing this factor alone does
not supply that model. Zero density/selection stays zero; invalid inputs are
not silently floored. Empty total support cannot be promoted to finite evidence.

One mechanics execution reuses all10,020 source-semantic FP rows,6,821 source
groups, the saved unconditional N128 density/velocity and existing correlated
redshift integrator. b=1,S_group=1 are explicitly uncalibrated controls; retain
the previous single sigma150/zero.004 setup rather than repeat sigma sweeps.
Compare with the homogeneous radial measure, refine distance/zero quadrature
257/129 to513/257, and test both velocity-amplitude and radial-bias derivatives.
No fitting, new gravity run, source rematching, N256 sampling or posterior.

Q-GOAL: density structure now affects the distance conditional as well as
the count response; this is necessary for actual z=0 joint inference.
Q-LEAN: extend the existing module, driver and tests, one <=10min allocation,
no new framework. H200/H100/A100 checked; choose typed H200,2CPU,16GiB
(<=13GiB conservative JAX/array peak estimate plus20%, rounded), one GPU.

MW/M31/M33 remain NEW-field candidates with latent MW/M31 roles and explicit
unresolved M33; their observables must constrain that same evolved field in
R3. Observed group identities here never seed generated LG identities.

Unresolved after successful numerics: selected-group inclusion/bias, physical
COM/member discrepancy, the joint count/position/mark observation law and
literal point-selection support. Do not label arbitrary control settings a
calibrated prior or use this job to authorize a production posterior.

## Execution result and driver disposition

H200 Slurm406119 COMPLETED1m29s/exit0 on syn104. Three focused/reused tests
pass; batch MaxRSS2,530,968KiB. The16GiB conservative request was larger than
the measured host need; a comparable follow-up should use measured peak+20%
rounded to4GiB unless its arrays change. No gravity calculation was launched.
Result: `/gpfs/kjhan/CF4/z0_density/r2_fp_same_field_radial_v1/result.json`;
per-group/global-zero readout is `group_factors.npz` in the same directory.

All10,020 FP rows /6,821 source groups participate. Same-state density enters
both conditional integrals; all evaluated radial nodes have positive density.
Refinement changes train velocity log ratio by0.000225857 (all-group0.000195850).
The density-weight versus homogeneous train-factor difference is-0.156162 at
the fine quadrature (coarse-0.157267): nonzero but NOT a fit improvement or
evidence of information recovery. Saved PM phases remain unconditional.

Velocity-amplitude derivative-46.470107504 agrees with finite difference
-46.470107303. Radial-bias derivative-1.214230776 agrees with-1.214230777.
These are scalar derivatives of the observation component, not an IC-adjoint
test or demonstrated density-field posterior. Shared zero point is integrated
once after group products; no extra FP selection correction was applied.

Accept the same-field radial-measure implementation. Do not promote the b=1,
S_group=1 fixture, provisional covariance or present-field posterior. The
previous r^2-only assumption is now replaceable through explicit model inputs,
not empirically solved. Next essential work is a source-constrained inclusion/
tracer model and group velocity distribution feeding this interface, jointly
consistent with observed counts and shared redshifts. Existing SDSS randoms/
mocks are candidates to inspect for that calibration; their galaxy selection
cannot simply be relabelled a group inclusion law. No further scalar derivative
or homogeneous-vs-density control series is needed after this pass.
