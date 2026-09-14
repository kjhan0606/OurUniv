# R1 planar discrepancy: bounded cause separation

User2026-09-14 requests the cause of359203's13.57%/82.68% velocity errors
against a pre-shell-crossing analytic plane wave. This is diagnosis, not a
production-force correction or a new inference/model selection exercise.
Existing data, failed logs and numerical comparisons remain preserved.

## Questions and controlled interventions

1. Verify the flat matter+Lambda growth function AND its logarithmic derivative
   against independent quadrature at eight scale factors. For this cosmology,
   `D(a)=(5 Omega_m/2) E(a) integral_0^a da/[a E(a)]^3`, normalized to D~a
   early. Compare the unnormalized PMWD table directly, not a fitted amplitude.
2. Construct the same single Fourier mode through PMWD's2LPT initializer;
   compare initial displacement and peculiar velocity to the analytic formula.
   In the plane-parallel case the second-order contribution is zero. This
   checks sign, growth normalization and the canonical-to-km/s conversion.
3. Reproduce BOTH prior endpoints while recording initial force response and
   growth history with the same PMWD kick/drift factors and KDK sequence.
   Diagnose initial-force gain and shape residual also at .001 times the
   original displacement (no evolution or changed science target in this check).
4. Shift the particle lattice by HALF A FORCE CELL and sample the SAME physical
   plane wave at the shifted Lagrangian positions. This changes mesh alignment,
   not the continuum solution or a tunable cosmological phase.
5. At force256, project the deposited density onto its transverse average and
   keep the longitudinal PM/CIC operator. Separately remove axial modes at/above
   the particle Nyquist scale. These distinguish transverse and axial sampling
   aliases, not an authorized force-filter fix for the real three-dimensional
   problem. No high-k threshold is adopted for the science model.
6. Replace ONLY the force by the known planar-sheet acceleration
   `a_PM=(3/2) Omega_m (x-q-mean(x-q))` in PMWD's acceleration units. Same
   time integration and initial state. Known Lagrangian labels q are legal only
   for this exact pre-crossing control, never generated-halo assignment.

The continuum planar trajectory is exact before crossing; the experiment has
final linear amplitude .3 and minimum continuum Jacobian .7.
[McQuinn & White](https://arxiv.org/abs/1502.07389). Particle-lattice dynamics
need not exactly equal a continuous fluid, so this is a hypothesis to test,
not proof that PMWD is broken. [Discrete perturbation theory](https://arxiv.org/abs/astro-ph/0601479).

## Frozen computation

Source: [diagnostic driver](scripts/cf4_r1_planar_diagnosis.py),
[configuration](config/cf4_r1_planar_diagnosis_v4.json).
128³ particles in12 cMpc/h, wavenumber4, a=1/64→1, maximum delta-a1/256;
two original force meshes, two half-cell alignments, projected plane force,
projected/low-passed force and exact sheet force. Seven evolutions, no new
random phases, no chain extension, no full snapshots. Initial force metrics
and compact histories/endpoint comparisons are the outputs.

The installed PMWD and production forward/sampler are NOT patched. Library
private kick/drift functions are reused locally; agreement with359203 is
checked before interpreting the changed-force controls. Pure-mode LPT and
growth errors abort rather than blaming the force kernel. No generic new
validation framework, filesystem test, process scanner or manual node run.

One Slurm GPU/4CPU/8GiB/15min, application cap12min, a100_pcie,a40,a100,
h100,h200, exclude syn06. Estimate6GiB host peak from previous2.52GiB plus
force/history/constant buffers and compilation; +20% rounded8GiB. Clear only
obsolete JAX executable caches between cases; expected files<1MiB. Allocation
budget before this job5108s/14400s; even the full900s request is within R1.

Q-GOAL: isolate a potentially serious dynamics error before conditioning real
CF4/LG fields. This does not identify MW/M31/M33 or supply an M33 bound-mass
operator; those requirements remain. Q-LEAN: one seven-control diagnostic
uses the same small state and existing integrator, not another fine-mesh or
sampler sweep. Verified important conclusions receive external advice under
the current policy; no external opinion substitutes for numerical evidence.
