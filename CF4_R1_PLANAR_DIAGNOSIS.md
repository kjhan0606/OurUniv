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

## Result359206 and driver localization — 2026-09-14

Source `977852c9cd01ae93f619b0c38ec4afc562786b52`, Slurm359206 ran on
syn103/a100_pcie14:34:59–14:36:09 KST, COMPLETED/exit0 in70s. All seven
controls finish. Growth D/dD comparison max relative4.345e-6; pure-mode LPT
displacement/velocity max differences1.12e-17 cMpc/h /4.99e-15 km/s. Both
previous PM endpoints reproduce. Source-level sign/unit conventions and the
background reference are not responsible for the large discrepancy.

| Controlled force/mesh setup | Initial force relative RMS | Final velocity relative RMS |
| --- | ---: | ---: |
| Original PM128 | 9.271% | 13.568% |
| Original PM256 | 57.728% | 82.675% |
| PM128, half-cell lattice alignment | .707% | 4.396% |
| PM256, half-cell lattice alignment | 4.793% | 25.635% |
| PM256, transverse-average plane density | 23.491% | 30.157% |
| Same plane density, axial modes below particle Nyquist | 4.706% | 6.315% |
| Exact planar-sheet force, SAME time integration | 1.59e-13 | 2.88e-14 |

The last row is a fractional numerical error, not a percentage. The exact
force control checks this restricted growing-mode trajectory and does not
prove the time step adequate for generic nonlinear dynamics.

**Confirmed localization:** the large discrepancy in this plane benchmark
comes from the spatial particle-mesh force operator, not the analytic growth
normalization, initial-condition sign, peculiar/canonical velocity conversion,
or the KDK time integration. It is already present in the weak initial state,
not generated only by late nonlinear evolution or a sampler/learning failure.

**Supported mechanism:** particle-lattice/force-mesh alignment and assignment/
sampling aliases introduce spurious spatial-force patterns. In the infinitesimal
displacement test, nodal PM128 force gain is .99358684, but residual after
removing that gain is9.518%; shifting half a cell leaves the same gain but
only2.98e-6 residual. For PM256, nodal gain1.074502/residual61.694% becomes
gain1.030601/residual3.69e-5 after the shift. This is primarily a shape/alias
effect, not a single amplitude-normalization discrepancy. Transverse projection
and axial filtering reduce different contributions but do NOT solve the full
three-dimensional inference problem or authorize suppressing physical high-k.

Source-backed mechanism to distinguish from a numerical fit: for infinitesimal
1D displacements u_i of particles initially at mesh vertices, exact CIC
deposition gives

`delta_i = -(u_(i+1)-u_(i-1))/(2 dx)
           + (|u_(i-1)|-2|u_i|+|u_(i+1)|)/(2 dx)`.

The additional term comes from changing the deposition side at a CIC corner;
it is not a physical continuum density perturbation. At fixed resolution it
does not vanish relative to u when its amplitude is reduced. At half-cell
alignment the infinitesimal deposition/gather stays on one smooth branch.
The measured .99358685 fundamental gain matches sin(k dx)/(k dx) for32
particle samples per wavelength; remaining nodal shape errors are not removed
by correcting this gain. This explains the unshifted PM128 behavior and the
importance of lattice alignment; the entire PM256 residual is not claimed to
be quantitatively decomposed by this1D formula alone.

**Limits/decision:** no universal83% error claim for random3D cosmologies,
no proof of a unique erroneous PMWD source line, and no independent collapsed-
halo calibration. Axis-aligned, perfectly regular plane particles expose a
specific numerical vulnerability. Hold production accuracy/R2 promotion; do
not resume longer HMC or choose finer meshes as a substitute for a force fix.
No production force, observational noise, amplitude or high-k boundary was
changed. An important-finding read-only Fable review was requested with the
exact evidence and Q-GOAL/Q-LEAN; its advice is recorded separately on return.

Result: `/gpfs/kjhan/CF4/z0_density/r1_planar_diagnosis_v4/job_359206/result.json`;
seven compact history NPZ files accompany it. Logs:
`/gpfs/kjhan/CF4/logs/cf4_R1_planar_diag_359206.{out,err}`.
R1 used5178/14400 Slurm GPU-seconds; remaining9222s (2h33m42s).

## Fable advice and driver disposition

Read-only `claude-fable-5` advice completed successfully, with no permission
denials. The initial CLI response was not retained across a truncated tool
response; one identical retry supplied the retained response below. This is
one substantive advisory opinion, not two independent reviews. No calculation
was repeated for that recovery.

- [Exact request](config/cf4_r1_planar_cause_fable_prompt.md)
- [Retained CLI response](config/cf4_r1_planar_cause_fable.response.json)

Fable finds the force localization sound, independently re-derives the nodal
CIC formula, and finds no invalidating sign, unit, growth or KDK defect. It
recommends a locally isolated, consistent higher-order scatter/gather trial
(TSC or PCS), with a force-gradient check across an assignment boundary.
Q-GOAL: prerequisite dynamics risk reduction, NOT an actual CF4/LG posterior.
Q-LEAN: the70s diagnostic is proportionate; no chain extension or broad new
validation framework is warranted.

Driver accepts this core advice with three specific qualifications:

1. Fable calls the13.6%/83% values "worst-case bounds". They are NOT established
   bounds: these are measured errors of one coherent plane fixture. Neither
   random3D error nor its upper bound has been measured by this experiment.
2. The controls support transverse lattice content and axial sampling aliases,
   but do not uniquely or quantitatively decompose all PM256 error. If a
   higher-order scheme leaves a residual, that does not undo the confirmed
   spatial-force localization or the exact1D CIC kink derivation. It would
   limit the claim that a particular assignment order cures the whole problem.
3. A higher-order kernel needs its OWN correct support/weights and consistent
   scatter/gather derivatives; reusing CIC's2-per-axis stencil or backward
   rule is not sufficient. Generic3D force/gradient checks and independent
   nonlinear validation remain necessary before science promotion. They may
   be deferred for the isolated assignment experiment, not waived. M33 and
   generated-state MW/M31/M33 identification likewise remain open.

Recommended next corrective experiment, NOT implemented or launched in this
diagnose-only turn: an isolated TSC deposit+gather force using the same FFT
Poisson kernel and frozen planar state, compared with CIC at both force meshes
and alignments, plus one bounded derivative check. Keep all physical modes;
do not correct the power amplitude, permanently filter high-k, lengthen HMC,
or promote production on a plane-only improvement. No installed PMWD or
production force code has been changed.
