# R2 actual-data joint IC-gradient control — 2026-09-26

## Scope and results

The common `h=0.746` N128 2M++ counts and directly integrated selection were
conservatively restricted to N32/12 cMpc/h **only for a cheap mechanics
control**. A fixed unconditional 384-cMpc/h PMWD IC evolves to one density
and mean-velocity state. Six-population spherical RSD/FoG count intensities
and the actual CF4 radial prediction both read this same state. The 2M++
training likelihood uses 24,993 galaxies; 8,375 holdout galaxies are excluded.
The CF4 likelihood uses 15,346 training radial rows; 3,967 holdouts are
excluded. Four shared CF4 bulk/H0 Gaussian nuisance coordinates are
analytically marginalized with a Woodbury 4x4 solve. An eight-row direct
covariance calculation checks that marginal likelihood before PM evolution.

Syntax Slurm405224 completed in 2m55s, max host RSS 3.86 GB. Both training
and heldout occupied population/cells had positive intensity, and the joint
objective plus IC gradient were finite. Its **single** central-difference
step of `epsilon=0.005` gave `+0.102` versus autodiff `-0.943`; that step is
therefore **not** accepted as a derivative validation. Preserve the failure.

The same fixed IC, direction, data and model were retested with only a
finite-difference step ladder in Slurm405225 (2m32s, max host RSS 3.79 GB).
The directional autodiff derivative remained `-0.943245`; the central
difference at `epsilon=2e-5` was `-0.947450`, relative difference 0.42%.
Across `epsilon=0.04` to `0.0001`, differences oscillated and even changed
sign. The count term's small-step directional difference was `+171.883`, CF4
`-165.037`, and white-prior `-7.793`, so the total is a cancellation; using
only the total derivative at a broad step is especially misleading. This is
consistent with crossing nonsmooth particle/mesh or deposition boundaries,
but **does not uniquely localize** the source or certify stable finite-step
HMC dynamics. The test establishes a local directional derivative at one
IC/direction, not a global adjoint or inference acceptance gate.

Machine outputs:
`/gpfs/kjhan/CF4/z0_density/r2_joint_ic_gradient_n32_v1/result.json` and
`/gpfs/kjhan/CF4/z0_density/r2_joint_ic_gradient_n32_v2/result.json`.

## Science decision

No R2 posterior or N128/1.5-cMpc/h map was produced. The fixed published
rates/biases, Beta posterior-mean survival, and inherited FoG widths are
**development choices**, not a jointly calibrated observation model; their
uncertainty was not marginalized. The IC phase was unconditional. An objective
value on that random sky is not a CF4/2M++ fit, and a clean small-step
derivative does not justify starting a long chain.

Next: define and test a minimal six-population rate/bias/survival/FoG nuisance
law without using observed totals as a hidden normalization, then run one
bounded N128 full-IC-gradient and memory/cost control. A larger inference
requires finite-step gradient/HMC behavior and heldout prediction checks, not
an arbitrary acceptance-threshold adjustment. The LG roles MW/M31/M33 remain
latent features of any **generated** state; M33 can be unresolved. No native
truth identities are inputs to this likelihood, and the local member
observations have not yet been added. Q-GOAL: a real two-channel likelihood
is now differentiated through one shared gravity state. Q-LEAN: this was one
N32 mechanics control plus a same-state step-size diagnosis, not another
simulation, ML retraining or gate framework.
