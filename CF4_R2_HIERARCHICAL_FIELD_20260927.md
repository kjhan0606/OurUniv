# R2/5 — live selected-population and shared-error bundle

R1 mechanics -> **R2 actual present-field inference, unfinished** -> R3
LG<=0.3cMpc/h -> R4 precise forward checks -> R5 zoom IC ensemble.

## One substantive model extension

Previous406350 proves bounded live IC/mark transitions, not model calibration.
Do not simply lengthen its chains. Replace fixed selected-density slope1 and
independent-within-group FP residuals with the following explicit conditional
development target, evaluated from EVERY new PM state at native mesh origin0:

`w_g(d) = dd d^2 rho_F(d)^b_eff (d/100)^kappa`

`L_g = sum_d w_g K(z_group,z_2mpp_members | d,F)
          product_i L_FP_i(eta_i(d)+beta+u_g)
          product_j L_other_j(mu(d)+offset_method)
        / sum_d w_g K`.

Here `dd` is already included in w, so the implementation quadrature does
not multiply it twice. All source distance marks in a group see ONE distance
and ONE shared FP excess offset u_g. The non-FP marks do NOT share u_g.
The original raw observations and existing source skew-PDF shapes enter once.
Global beta/method offsets remain shared across groups. No fitted offsets,
relative-REML scatter, mock truth roles or combined CF4 DM enter as priors.

Use noncentred u_g=tau*z_g, z_g~N(0,1) for the5,330 training groups only.
Previous six development nuisance priors unchanged. Add
log(b_eff)~N(0,.5^2) and log(tau/.02dex)~N(0,.7^2). These are explicit proper
REGULARIZING priors, not externally measured bias, inclusion or covariance.
b_eff combines tracer response and environment-dependent selection; no
separate identifiability claim. Tau represents additional FP-group discrepancy
beyond supplied individual PDFs, not recovered source-fit covariance or pure
physical depth. Its lognormal prior excludes exactly zero; sensitivity to it
and possible overlap with source errors must remain scientific limitations.
This is a testable expanded partial model, NOT resolution of every R2 risk.
The150/100/50km/s velocity covariance, finite radial support, arbitrary non-FP
zero scales and missing cross-group source calibration remain provisional.

### Legitimate galaxy data ownership

Reuse the3,062 uniquely associated eligible2M++ member redshifts in2,413
groups as covariates INSIDE the same group-distance kernel. Geometry export
also saves the source-group-redshift-only kernel and member counts. At the
initial same state report full-minus-group-only conditional score and IC
gradient. This measures wiring/model-conditional influence, NOT independent
kinematic validation or calibrated information gain. Both kernels use their
corresponding proper denominator; no extra independent redshift factor.
The galaxy count/within-cell point likelihood remains unimplemented jointly.
Do not multiply historical inclusive counts/BGc, or interpret this model as
the entire CF4+2M++ posterior. Group-membership/Covariance assumptions remain.

## Execution and checks in one allocation

Four small tests: zero-offset recovery; shared-offset integral against exact
correlated Gaussian (including unchanged non-FP marks); gradient/closed-holdout
ownership; heldout new-offset integration. Reuse native PM registration test.
Run a bounded N128/384 live pilot with two fresh prior IC/group starts, each
32 warmup+32 retained transitions, same prior whitening and4–8 step HMC.
No previous random-field calibration estimates or score-selected seeds.
Check the initial MARK-ONLY full-IC+nuisance adjoint after subtracting analytic
prior tangent; stop sampling if scaled difference exceeds .02. Record the
two-step FD values, not just pass/fail. This is a local check only.

Save initial and fixed final states (3, not every transition), their white
IC/group/hyper coordinates, density, mean velocity, physical diagonal velocity
variance and scalar traces. No credible/mean posterior maps or convergence
claim from64 transitions. Group-white summaries are not independent draws.
At the final chain1 state compare distance Q257/513 on its sampled training
offsets. For heldout groups integrate NEW offsets with9/17 Gaussian nodes at
Q513; never reuse a fitted group offset or zero-offset heldout score. A
quadrature difference>.05nat withdraws that heldout factor diagnostic; it
does not license a quadrature sweep or fitting to holdout data. Even passing
is only a conditional factor, not field-averaged predictive density or an
independent validation of upstream full-source FP fitting.

One H100 job (current H200 draining),4CPU,14GiB host (planning peak<=11.5GiB
including extra compiled adjoint, plus20%),25min Slurm cap; application1200s.
Expected ~10–15min based on406350, not a guaranteed finish time. Absolute
Python, module purge, thread caps, all numeric work via Slurm. Existing data
preserved, no storage probes or cleanup; new compact output budget~320MB.

Q-GOAL: same actual observations now propagate selected-density-response and
shared-group discrepancy uncertainty through the SAME IC/current field.
This removes two fixed assumptions but cannot establish calibrated selection
by nuisance flexibility. R2 actual useful posterior remains first delivery.
Q-LEAN: reuse source geometry, PM dynamics, source PDFs and HMC; four focused
checks and one short calculation. No survey acquisition, new solver, external
audit, separate monitor or generic gate infrastructure.

MW/M31/M33: this3cMpc/h surrounding-field pilot does not identify them. R3
must obtain candidates from each NEW generated particle state, retain MW/M31
role ambiguity and unresolved M33, and constrain that SAME state using their
observables plus a defensible role/detection law. No native truth identities,
best-match shortlist, seed promotion or historical40349 halo reuse.

## Execution/result

Source da6c5ff committed/pushed. Syntax Slurm406356 started on syn08/H100
with the above14GiB/4CPU/25min allocation. All6 checks (two reused, four new)
pass. Source geometry export9.576s, unchanged5,330/1,491 closed group split,
3,0622M++ members in2,413 groups; no ambiguous member associations in this
subset. No new fitted-data priors or field-dependent geometry cache.

Initial mark-only directional derivative: AD92.528816; FD92.524785 at2e-5
and92.518952 at1e-5, scaled discrepancies0.00436% and0.01066%. The analytic
prior tangent1832.667034 was removed. Pass the bounded local derivative
screen, not a general HMC stability claim.

At this SAME initial state/group nuisance realization, adding member2M++
redshifts versus the group-only kernel changes the training conditional
factor by+0.301197nat and IC mark gradient L2 by1.136202, versus full mark
gradient L2=31.297144 (3.6304%). This establishes nonzero conditional wiring,
not externally calibrated information, count-data use, or posterior recovery.
The completed results below supersede the initial running status.

### Completion and interpretation

H100406356 COMPLETED8m54s/exit0. All6 tests, two chains and endpoint
diagnostics finish in the same allocation. Main-driver runtime445.38s,
first joint gradient compile/evaluation43.83s, process peak7.443GiB;
Slurm sampled MaxRSS7,447,796KiB. No OOM. The14GiB request was conservative;
an unchanged repeat would need about9GiB for measured peak+20%, rounded to
10GiB. This is not a memory estimate for N256 or a new count model. The same
nonfatal hwloc CPU-binding warnings as406350 remain; no system probes added.

| Quantity | Chain0 | Chain1 |
| --- | ---: | ---: |
| Warmup / retained transitions | 32 / 32 | 32 / 32 |
| Retained mean acceptance | .871677 | .946268 |
| Retained divergences | 0 | 0 |
| Frozen retained step | .01259252 | .01175776 |
| IC RMS change from own start | .568148 | .566779 |
| Endpoint b_eff | .652396 | .739602 |
| Endpoint excess tau, dex | .00321904 | .00299864 |
| Endpoint rho mean | 1 | 1 |
| Endpoint rho RMS | 1.959988 | 1.947604 |
| Endpoint mean-velocity RMS, km/s | 252.389 | 236.102 |

These are short-trajectory ENDPOINTS, not posterior means, credible intervals
or source calibration. In particular, movement of tau from its starting
.02dex to~.003dex is NOT a measurement of negligible common source error.
Noncentred group/field/hyperparameter equilibration has not been established;
the added lognormal discrepancy prior and missing source covariance remain.
Do not fix these endpoint bias/tau values for the next run, turn them into
independent priors, or infer improved reconstruction from higher mark scores.
No prior-only matched chain was run, so not all IC motion is attributable to
observations. No individual halo or observed density truth was supplied.

At the fixed final chain1 state, Q513-minus-Q257 training logfactor is
0.002457884nat. Heldout NEW-offset GH9 and GH17 conditional factors agree
to1.14e-13nat per group (summed difference-1.04e-13nat). Accept these endpoint
numerical checks, not an all-state quadrature bound. The reference-dependent
heldout logfactor1879.775152 is not an absolute predictive density or a
field/hyperparameter-averaged posterior prediction, and cannot be compared
naively to the previous frozen/zero-offset values as a quality improvement.

Outputs: `/gpfs/kjhan/CF4/z0_density/r2_hierarchical_field_pilot_v1/`
contains `result.json`, initial and two predetermined endpoint state NPZs,
`transition_summaries.npz`, and `endpoint_factor_checks.npz`.
The three state files total208,207,805bytes and include the same-state IC,
hyperparameters, training-group whitened offsets, density, mean velocity and
physical diagonal velocity variance. Physical variance is NOT uncertainty in
the mean field. Geometry files in `r2_hierarchical_field_geometry_v1` total
85,249,666bytes. Seven numeric artifacts total293,538,495bytes (~294MB).
No existing raw output was removed and no per-step full fields were stored.

Driver decision: **PASS_BOUNDED_HIERARCHICAL_WIRING_ONLY**. The fixed b=1 and
zero extra common-offset assumptions are no longer hardwired into this new
partial target; field dependence, source sharing, non-FP ownership and
heldout-offset integration work. CLOSE this short-transition bundle without
claiming a full R2 delivery or extending it blindly.

Next substantive work remains a coherent galaxy-count/point + CF4 conditional
mark law, with selected-group/velocity/source discrepancy exposed and tested
as part of that model. The present member-redshift conditional does not use
the bulk2M++ density information. Do not replace the missing likelihood with
more fixed-field calibration, historical inclusive-count/BGc multiplication,
an apparent nuisance fit, or an N256 production launch. The next implementation
must explicitly state its observation factorization and which dependence is
retained/approximated before using counts to update these same live fields.
MW/M31/M33 and R3–R5 restrictions above remain unchanged.
