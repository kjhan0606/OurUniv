# R2 conditional CF4 group mark — bounded model entry, 2026-09-26

## Scientific role and data ownership

R2 first needs a real-data-conditioned, dynamically connected z=0 density and
velocity posterior. This bundle does **not** deliver one. It addresses one
specific obstacle: CF4 group redshift/distance marks and the matched 2M++
individual redshifts are not independent observations. The intended
factorization is `p(Y_2M++ | F, selection) × p(G_CF4 | Y_2M++, F, selection,
association)`, with `F` the same evolved state. The second factor must be
normalized *conditional on the individual point data*. The old binned-count
times grouped-BGc product is not this factorization: the count grid loses
individual within-cell redshifts, and BGc does not model group/member source
dependence. No source or old posterior is silently replaced.

For the later LG work, MW/M31/M33 roles must be found from each **new**
evolved state with ambiguity and unresolved M33 retained. Their observations
must constrain that same state. Native identity labels cannot seed or select
candidate structures. This R2 group-mark model does not identify LG members
or treat that requirement as already solved.

## Actual-data conditional calibration

`scripts/cf4_r2_group_redshift_conditional.py` uses only the frozen CF4 group,
2M++ individual, and secure crossmatch sources plus the N128 eligible point
list. It assigns groups to the preserved native CF4 train/holdout split,
then models the residual `CF4 group Vcmb − mean(secure matched individual
2M++ Vcmb)` separately for one versus multiple eligible secure members.
The Student-t degrees of freedom are fixed at four; location/scale use only
the training split. This is **conditional on the observed secure-match
selection**, not a CF4/2M++ survey-selection or field model. It uses neither
the archived 2015 2M++ cross-ID as the current local GID nor an average of
CF4 distance contributors as the published group velocity.

Syntax typed-H200 Slurm **405550 COMPLETED/exit0** in 10 s. The preserved
[result](/gpfs/kjhan/CF4/z0_density/r2_group_redshift_conditional_v1/result.json)
contains 9,754 selected groups; the fitted/test counts are 6,563/1,629 for
one secure member and 1,214/348 for multiple. On the heldout groups, the
Student-t beats a fitted Gaussian in mean log score in both strata. But its
nominal 90%/95% intervals cover only **82.0%/86.1%** for one-member groups;
multiple-member coverage is **88.2%/92.8%**. These are not independent sky
volumes, and neither fit accounts for the 442 pointwise mark/map anomalies.
Decision: **NO-GO** for this two-stratum Student-t as an accepted group
redshift law. Better average log score does not cure tail undercoverage.
Do not tune coverage by an arbitrary global scale or insert the fitted
Student-t into an actual-data IC posterior.

## Numerical conditional-mark kernel

`src/cf4_r2_conditional_group_mark.py` computes a properly normalized
Gaussian `p(V_group, DM_group | V_matched_members, F)` from an externally
supplied joint mean and positive-definite covariance. It uses a Schur
complement, checks duplicate member IDs, and never adds a second Gaussian
factor for the member redshifts. The supplied means must come from the same
evolved field. The covariance is **not** calibrated by the failed Student-t
fit; in particular, a Gaussian kernel cannot be adopted merely because its
algebra is correct. The pure numerical tests compare its log density with
`log p(joint) − log p(members)` and check no-member and invalid-covariance
cases. Syntax typed-H200 Slurm **405551 COMPLETED/exit0** in 6 s;
all **3/3 tests pass**. That is an algebraic implementation check, not
empirical calibration of its covariance or a scientific likelihood pass.

## Next modelling requirement, not a gate ladder

The necessary next science computation is a **source-aware full point/mark
candidate**: retain 2M++ individual redshift-space points; handle its
object-level selection anomalies; specify CF4 distance-mark selection and
group/member association; then calibrate or marginalize the group-redshift
conditional with heldout coverage and residual dependence by distance/sky.
It must use a field-dependent distance/velocity prediction from one latent IC
state. Merely multiplying the present empirical redshift law, the old binned
counts and BGc would not be a normalized joint model. No N128/N256 sampling,
new simulation, LG seed search or production density/IC claim follows here.

Q-GOAL: this work directly targets the observation dependence blocking R2.
Q-LEAN: one source-bound 10-second calibration and one small algebraic kernel;
no more abstract stress arms, new cosmological run, or posterior fit.
