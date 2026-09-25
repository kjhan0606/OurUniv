# R2 continuous-tracer operator feasibility — 2026-09-25

## Result and scope

The six-population operator in `src/cf4_r2_continuous_tracer.py` reads one
evolved density/velocity grid, uses a positive normalized `rho^bias` tracer,
observer-centred spherical coherent RSD, three-node Gaussian LOS FoG/redshift
quadrature, periodic TSC, and then the survey exposure. Its rate is a model
expectation, **not** normalized to the observed galaxy total. One explicit
unresolved/contaminant component is spatially uniform before selection. This
component guarantees positive predicted intensity where exposure is positive
despite the finite GH/TSC stencil; it is a *model parameter*, not a hidden
Poisson-log numerical floor. Its trial fraction **0.03 is arbitrary and not
calibrated**. An actual-data posterior is forbidden until its meaning and
prior are externally constrained or replaced by a validated positive-support
continuous response.

Syntax Slurm 404199 passed two operator tests, then evaluated an archived
N32/384 PM density/velocity state with the corrected actual 2M++ selection.
There were zero occupied cells with zero intensity. The minimum intensity on
positive-exposure cells was `3.41e-8` with the **placeholder** unselected
rate of one galaxy per cell. This is a support/gradient check, not a count fit
or physical bias/rate estimate. The same job benchmarked an **analytic,
non-PM** N128/384 field to establish operator memory/compile feasibility.
An initial uniform-total gradient was near zero by mass conservation, so a
nonuniform spatially weighted gradient was rerun in Slurm 404201. On GPU:
forward compilation 9.35 s, warm forward 0.044 s, gradient compilation
23.63 s, warm scalar-parameter gradient 0.049 s, batch MaxRSS 2.62 GB.
The weighted prediction derivative was `3.945e5`; no zero-gradient inference
was drawn from the earlier conserved total. These are **operator-only** times,
not full PM/IC adjoint or posterior costs. Both jobs requested 32 GB host
memory (well above the measured peak), 1 GPU, 2 CPUs, bounded time.

Machine outputs:
`/gpfs/kjhan/CF4/z0_density/r2_continuous_feasibility_v1/job_404199_n32.json`
and `job_404201_n128.json` in the same directory. Slurm 404197 failed
before numerical work because the default Python lacked JAX; it was replaced
with the project's JAX environment. No simulation or posterior fit ran.

## Scientific disposition

This closes the **mechanics/cost** part of a positive-support response, not
the R2 observation model. The N128 benchmark used an analytic field, uniform
selection and fixed nuisance values. The only existing 384-box native PM
fields used here are N32/12 cMpc/h. The actual corrected selection array is
also N32; simply interpolating that exposure or an N32 PM field does not
create independent 3 cMpc/h information. The artificial diffuse fraction,
rate, bias scale, FoG and survival uncertainty still need a calibrated
joint treatment. A full-field IC gradient and chain ESS have **not** been
measured by this scalar-parameter adjoint.

Next substantive bundle: obtain a genuine 384-box N128 PM state and an
observation-preserving N128 count/selection representation, calibrate or
replace the diffuse-support term with disjoint mocks/marks, then time a full
IC-to-count gradient and compare held-out counts/CF4 velocities. Do not
launch N256 posterior sampling from this benchmark alone. This remains the
R2 present-field route; the eventual MW/M31/M33 roles are latent readouts of
the same evolved state, M33 still unresolved, with no native truth IDs passed
to generated-field inference. Q-GOAL: the operator connects actual R2 survey
geometry to a future PM state. Q-LEAN: one operator, two small Slurm probes,
no new TNG or simulation, no promotion gate beyond the missing calibration.
