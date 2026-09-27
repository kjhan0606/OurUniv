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

Implementation in progress; no numerical pass yet.
