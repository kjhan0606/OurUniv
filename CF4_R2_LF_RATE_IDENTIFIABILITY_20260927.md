# R2/5 — observable-rate coordinate for the 2M++ source model

The R2 z=0 density/velocity posterior remains **NO-GO**. This change is a
count-model parameterization repair, not an LF fit, survey calibration,
heldout prediction, sampler pass, or a 1.5-cMpc/h map. No email was sent.

## Problem and exact repair

The first source-selected count control defined one intrinsic rate per
3-cMpc/h cell for *all* Schechter galaxies from infinite brightness to
infinite faintness. Its five true-K bin fractions were divided by
`Gamma(alpha+1)`. The observed six K bins instead cover redshift-derived
absolute K from -25 to -21. At fixed total rate, the intrinsic fraction in
that interval is 0.095961 at `alpha=-0.94`, but 0.016337 at `alpha=-0.99`
(`Mstar=-23.28`). Thus moving the unobserved faint-end slope toward -1
suppresses bright galaxies by a factor 5.87 before the rate can compensate.
This is an avoidable amplitude/shape correlation caused by an unsupported
infinite-faint normalization. It does not prove that the LF shape itself is
unidentifiable from all observed K/radial information.

The actual [2M++ source paper](https://arxiv.org/html/1105.6107) reports
finite-range LF fits, including `[-25,-17]` for the imported
`alpha=-0.94,Mstar=-23.28`; it also explicitly notes a faint-side shape
change near -21 and bright-end deviations below -25. The published fit does
not determine an all-faint galaxy abundance for this source-selected model.

`intrinsic_lf_reference_weights` now divides each true-bin LF measure by
the finite `[-25,-21]` LF measure. In the joint control, the rate coordinate
means the average *intrinsic reference-interval* count per cell. The new
zero-coordinate centre is the old centre's reference fraction, so the
physical predicted intensity at the development LF centre is unchanged.
For arbitrary LF parameters, transforming the rate by the corresponding
reference fraction reproduces the old five-bin masses exactly; the focused
unit test checks this also near `alpha=-0.99`. The new prior on the reference
rate remains a **development regularizer**, not a source-derived prior. The
existing prior-IC control406585 and its scores remain historical outputs of
the old rate coordinate; no previous result is silently relabelled.

Typed-H100 Slurm **406587 COMPLETED/exit0** (2m10s, peak host1.34GiB
under5GiB request): all seven focused marked-count tests pass, including
JIT evaluation of the reference-rate path and old/new mass equivalence at
two LF slopes. The changed joint module also passes static Python
compilation. This does not repeat the full PM IC adjoint, infer nuisance
parameters, or test posterior predictive calibration.

The unbounded fifth true-K bin still requires `alpha>-1`; the present
parameterization does **not** justify extrapolating the faint LF to infinity
or cover the alternative published `1/Vmax` slope below -1. A future model
should replace this tail by an explicit finite luminosity/selection law or
integrate only the finite source magnitudes able to enter the selected
observed bins. Do not treat an arbitrary cutoff as a calibrated repair.

## Consequence for the R2 route

Use the graph-closed v5 octant-2 split for any new actual-data fit. First
calibrate or stress-test the selected galaxy response and CF4 group
inclusion/covariance using source-backed mocks/heldout predictions; do not
restart the nonstationary identity-mass HMC or promote N256 from this
algebraic repair. Public 2M++ mock literature derives mocks from CosmoSim
semi-analytic galaxy catalogues, but the raw database alone is not an
off-the-shelf validated mock of our joint CF4/2M++ selection and grouped
marks. The [Manticore-Local 2M++ analysis](https://arxiv.org/html/2505.10682)
supports considering flexible bias and overdispersed counts as alternatives;
its parameters/posteriors are not independent calibration for this different
CF4 joint model.

Q-GOAL: removes an artificial nuisance geometry in the first science target,
the CF4+galaxy-conditioned present field. Q-LEAN: a small exact
reparameterization and focused regression, no new simulation or gate ladder.
MW/M31/M33 remain latent roles identified from each **new** evolved field;
M33 can remain unresolved, and native truth IDs cannot choose candidates.
Their observations must constrain that same state in R3.
