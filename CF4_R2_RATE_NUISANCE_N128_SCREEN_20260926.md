# R2 uncertain-rate and N128 IC-gradient screen — 2026-09-26

## Result

The six 2M++ count rates now have an explicit independent Gamma prior, with
means from the published cell-rate reference and a deliberately broad,
**uncalibrated** shape of one (coefficient of variation one). The likelihood
integrates those six rates analytically instead of centering or renormalizing
them on the observed totals. A direct two-cell quadrature, exact zero-support
case, and finite-difference derivative all pass. The unit test exposed and
corrected a JAX precision issue: applying `gammaln` directly to integer count
arrays used float32 even in an x64 run. Counts are now cast explicitly to the
likelihood dtype.

Slurm **405229** completed one actual CF4+2M++ joint objective and full IC
gradient at N128/384 cMpc/h (3 cMpc/h cells; 2,097,152 white-noise IC
coordinates), on the same random-phase PM state as the prior N128 control.
It took 1m34s, MaxRSS 3,701,832 KiB. The objective and gradient were finite;
22,457 occupied training keys and all occupied holdout keys had positive
intensity in the saved-state support preflight. The joint likelihood used
15,346 CF4 training velocity rows and 24,993 2M++ training galaxies;
3,967 CF4 and 8,375 galaxy holdouts stayed out of the objective. The four
shared CF4 bulk/H0 nuisance modes remained analytically marginalized.

Slurm **405230**, source commit `f781840` (its result JSON records the
submitted `HEAD` alias rather than its resolved hash), repeated the same
state with a directional finite-difference ladder. It completed in 1m51s,
MaxRSS 4,592,784 KiB. Compile-included gradient evaluation was 64.3 s;
same-process warm evaluation was 0.336 s. Autodiff gave -3527.524 for the
fixed direction; central differences over 1e-4 through 1e-5 gave -3526.115,
-3529.041, -3528.435 and -3528.291. The smallest-step relative difference
was 0.000217 (0.022%). The prior, count, and CF4 components were recorded
separately. This is one local directional check; the step dependence still
does **not** certify global smoothness, stable HMC integration or acceptance.
The source was then corrected to resolve a submitted symbolic commit to its
full SHA in future reports; the completed output was not rewritten.

The saved-state Gamma-shape sensitivity (shape 0.5, 1, 2) changes the count
log likelihood from -143182.03 to -143179.73 to -143177.85. It is **not**
evidence that any shape is calibrated or preferred by independent data.
Machine results are under
`/gpfs/kjhan/CF4/z0_density/r2_rate_nuisance_n128_screen_v1/` and `_v2/`.

## Science decision and next work

The N128 joint IC adjoint is technically feasible for one state and one
direction. It is not an inferred posterior, a 1.5-cMpc/h N256 global map,
or a <=0.3-cMpc/h LG field. The six published linear biases and inherited
FoG widths remain fixed development inputs. The 36 independent Beta
posteriors from disjoint survival marks are preserved, but only their means
entered this screen; environment/sky-dependent selection uncertainty, bias
transfer and RSD/FoG physical mock calibration remain unresolved. The
unclustered component is fixed to zero, not established absent. No new TNG
data were needed. The random IC phase was neither selected nor conditioned
on CF4 and 2M++.

Next, use the saved same-state responses and independent calibration marks
to define a *bounded* joint survival/rate sensitivity and physically testable
bias/FoG discrepancy law, then assess heldout predictive behavior. Do not
promote a long N128/N256 posterior or high-resolution IC solely from the
successful derivative. MW/M31/M33 identities must later be inferred among
generated-field components with role ambiguity and unresolved M33 allowed;
the native truth IDs cannot seed or select candidates. The local-member
observables still need to constrain the **same** inferred field.

Q-GOAL: this establishes an actual-data joint IC gradient at 3 cMpc/h on
the route to the R2 z=0 posterior, without diverting to simulation or ML
training. Q-LEAN: two short Slurm screens plus three focused unit tests;
no new simulation output, catalogue, audit framework, or production chain.
