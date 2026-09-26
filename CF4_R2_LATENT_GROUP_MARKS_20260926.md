# R2/5 — shared group marks and latent central identity

Implement one missing piece of the current selected-group conditional model:
central identity is uncertain, and member FP marks may share a group offset.
This is NOT calibration or permission for production posterior sampling.
No new gravity run, mock download, or nuisance fit is required.

For each group sum `w0 * product(Lsat) + sum_i wi * Lcen_i * product_j!=i(Lsat_j)`.
The zero-central branch means the central is not in the observed FP subset.
Normalize these hypothesis weights jointly per group. Integrate ONE shared
group offset after multiplying member factors, then integrate the group
distance using the same selected `dd*d^2*rho^b*S_group` measure and correlated
redshift kernel in numerator and denominator. Integrate the existing ONE
global zero point after multiplying the group factors. Use the same source
eta=0 reference for every role; do not duplicate source Sn/fn corrections.

Assumptions/limits: single physical host per group; role/mark-offset independent
redshift kernel; fixed conditional role weights for the selected observed
subset. Tempel FoF grouping can violate these assumptions. Interlopers, joint
role-dependent FoG/group selection, and shared source-fit error are NOT solved.
The additional scatter could overlap the published uncertainty: no automatic
variance inflation or fitted calibration is authorized by this experiment.
Mock truth labels do not define roles in the real or generated catalogue.

Q-GOAL: replace known member-independence/known-role shortcuts in the R2
observation operator used by the SAME physical field; do not claim a new map.
Q-LEAN: add one exact marginal operator to the existing group module and
reuse its actual-source driver. Three small necessary checks (enumeration and
zero support; zero-effect identity; shared Gaussian covariance/derivative)
plus one bounded full-source execution. No external review for routine wiring.

Use all10,020 existing FP rows/6,821 source groups on the saved unconditional
N128/384 state (3 cMpc/h, not the1.5 target). Compare257/513 distance nodes,
129/257 global-zero nodes,5/9 Gaussian group-offset nodes, and the unchanged
baseline on the same integration grids. Illustrative, deliberately UNCALIBRATED
inputs: observed-central probability0.5, equal conditional member weights,
central/satellite offsets-0.005/+0.005 dex, group-offset SD0.005 dex. These
are mechanics perturbations, NOT the truth-label means in the selected mock.
Retain old150 km/s member/redshift law, bias1/inclusion1 radial fixture and
global-zero SD0.004 dex solely for controlled comparison. No fit/seed choice.
Report quadrature sensitivity, finite source factors and conditional heldout
scores with the same global zero. Such scores are not independent validation.

One Slurm typed H200/H100/A100 allocation:1GPU/2CPU,8GiB host RAM
(estimated peak<=6GiB plus20%, rounded),10min. Sequential quadrature avoids
holding all member-by-distance-by-zero-by-group-offset tensors at once;
rematerialize nested mark kernels for future derivatives. Small derivatives
are tested; full IC-adjoint memory/cost with this extension is NOT measured.
Keep outputs in a new scoped GPFS directory; no filesystem diagnostics.

MW/M31/M33: inference must identify candidates from each NEW field, marginalize
MW/M31 role ambiguity and keep unresolved M33 explicit; their observables must
constrain that SAME state. This generic FP group-role law is not an LG component
finder or a mass/velocity/boundness model. No native identity seeds generated
candidates. R3<=0.3 LG, R4 precise forward validation and R5 phase-coherent zoom
remain outstanding. R2 actual posterior also remains undelivered.

After this bundle, stop synthetic covariance variations. The next deliverable
must determine what actual selected-group calibration can be supported by the
existing source/mock (which lacks recovered Tempel group IDs), then connect
the defensible conditional observation factor to count/field inference. Do not
equate these provisional nuisance values with resolution of that requirement.

## Execution

Syntax Slurm406155 on typed H200 COMPLETED1m49s/exit0. All three existing
group tests and three new latent/shared-mark tests pass. Full-source driver
time30.32s, batch MaxRSS1,740,316KiB (~1.66GiB); this is not a full IC-adjoint
benchmark. Output:
`/gpfs/kjhan/CF4/z0_density/r2_fp_latent_group_v1/{result.json,group_factors.npz}`.
No external audit, parameter fitting, extra download or new simulation ran.

All10,020 rows/6,821 groups completed:5,331 training/1,490 heldout groups;
3,062 unique overlapping2M++ redshifts condition2,413 groups. Exact mixture
enumeration (including an impossible satellite branch rescued by a central),
zero-effect reduction to the old operator, normalization invariance, shared
Gaussian covariance, and small-problem autodiff/finite differences pass.
All-zero support remains zero, not an added probability floor.

At the finer513/257/9 distance/global/group-offset rule, the training
log-factor contrast of saved-state velocity versus zero velocity is-23.523453;
the all-group contrast is-32.572105. Conditional heldout contrast is-9.048652.
Coarse/fine changes are0.00032217 training and0.00036257 all groups.
Latent-minus-old-mark log-factor changes are+7.064312 training/+12.894747 all,
with coarse/fine difference at most0.00020288. These are fixed-fixture
sensitivity results, NOT model evidence, a fitted improvement or posterior
validation. A random unconditional velocity state need not beat zero velocity.
Simultaneous quadrature refinement supplies a sensitivity check, not a rigorous
global error bound. No new parameters or field phases were inferred.

Driver decision: accept and close the **numerical latent/shared-mark interface**.
Do not accept the illustrative0.5 central probability, role offsets, group
scatter, bias1/inclusion1 or150 km/s redshift covariance as physical calibration.
The source PDF's good individual width does not fix these group correlations.
The complete count/association/selected-group likelihood and actual R2
posterior remain undelivered; the same source-fitted FP holdout is not an
independent calibration sample. No additional synthetic scatter sweep follows.

Next science delivery: source-supported constraints on the JOINT selected
group law (role/FoG, group inclusion, shared FP-fit error), separating what
can be inferred from actual observations from what needs an external parent
or recovered-group mock. The existing selected mock has neither recovered
Tempel group IDs nor pre-selection parents: more numerical marginalizers or
more rows of that same schema cannot supply the missing information. Reuse
this operator when that law is specified; do not silently elevate provisional
constants or launch a large N256 posterior on them.
