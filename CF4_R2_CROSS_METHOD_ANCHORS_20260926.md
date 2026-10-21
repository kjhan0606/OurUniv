# R2/5 — observed cross-method distance links and identifiable calibration

Implement and run one source-bound cross-method bundle. Reuse the10,020
SDSS FP source rows, their Tempel/singleton groups, and the existing CF4
individual-galaxy catalogue. No new gravity, simulation data or scatter sweep.

## Source and statistical contract

CF4 table2 describes T17 as the Tempel2017 group ID. Its method-specific
distance columns are on arbitrary zero-point scales; the combined DM is an
MCMC combination of methods. Therefore use ONLY non-FP method columns
(TF, SN Ia, SBF, SN II, TRGB, Cepheid, maser), with FREE relative method zero
points. Never use combined DM, DMfp or a CF4 group-average DM as an independent
anchor. Source: https://arxiv.org/html/2209.11238, sectionXII/table2.
The SDSS zero point itself used CF3 overlap, and richness-corrected FP fits
share calibration: https://arxiv.org/html/2201.03112, sections5.3–5.4.
These anchors are additional measurements, NOT independent calibration data.
CF3 flags are recorded but are not a proof of method-level independence.

Link by an exact same PGC or explicit matching positive T17. A conflicting
same-PGC/T17 association is reported, not silently accepted. Do not infer a
Tempel group from1PGC or a spatial nearest neighbour. Preserve all original
FP rows and catalogue sources. Close the inherited holdout over newly linked
CF4 groups before fitting; retain the shared source-fit caveat regardless.

Deliver a reusable row-level anchor artifact: recno/PGC/method, original
modulus/error, source group, source CF4 group, association type, CF3 flag,
and a closed split. This attaches real observations to the SAME group-distance
latent already used by the field likelihood; no independent field or fixed
mock-truth distance is introduced.
Add an optional non-FP mark to the existing shared-distance integral (both
baseline and latent-role kernels). Its method-zero vector is caller supplied
and shared; FP-specific offsets do not shift the non-FP marks. The caller
must predict a consistent luminosity modulus from that same state/distance.
This Gaussian conditional interface does not certify source-fit independence.

Within-group FP contrasts alone remove a common group shift identically.
An explicit contrast-matrix check demonstrates this, without pretending that
the common variance has been measured. Other methods create relative-distance
information but their free zero points leave an absolute calibration gauge.

One bounded CONDITIONAL calibration diagnostic: approximate source FP eta
summaries by their published moments, form inverse-variance group means, and
compare to linked non-FP moduli. For comparison use
`mu_FP* = 5 log10[(1+z_group) D(z_group)/h] +25 -5 mean_eta`, h=.746,Om=.31.
This fixes the observed group redshift in the luminosity-distance factor;
it is not an exact Doppler-aware luminosity-distance likelihood. The raw
anchor moduli remain untransformed in the reusable artifact.

For residual vector r_g within group g use
`r_g = method_offset + noise`,
`C_g = diag(e_DM_anchor^2) + (var_mu_FPmean + tau_relative^2) 11^T`.
Fit method offsets and one nonnegative total relative group-excess SD using
training-only Gaussian REML, then report fixed-fit heldout conditional scores.
Keep the covariance normalization and the shared FP mean term; do NOT count
all anchor-minus-FP pairs as independent. Fit in variance units with a fixed
search ceiling1mag^2; report a boundary rather than automatically widening it.
No offsets are estimated using the heldout rows. No refit after viewing scores.

This is a conditional Gaussian MOMENT diagnostic. Selection, physical group
depth, association errors, population trends and shared source fitting can all
contribute to tau_relative. It cannot identify a pure FP shared variance,
central/satellite probabilities, FoG/COM discrepancy or group inclusion.
Do NOT copy tau_relative/5 into the previous group-offset prior or scale source
errors. Absolute zero point and field response still require joint inference.
Do not use this fitted diagnostic as an independent prior AND reuse its rows.

Q-GOAL: replace invented calibration information with actual cross-method
observations and identify what they do/do not determine before an R2 posterior.
Q-LEAN: one assembly plus one small covariance-aware fit, no parameter family
search, new simulation, large archive or validation framework. Three focused
checks: correlated GLS against dense algebra, and exact group/method ownership
with common-shift cancellation; non-FP marks inside the SAME distance integral
against direct quadrature/gradient. Reuse existing source/model interfaces.
One Slurm H200/H100/A100 compatible allocation,1GPU/2CPU/2GiB (estimated
peak<=1.6GiB plus20%, rounded),5min; data fit is CPU-only, JAX wiring test on GPU.

Complete the SAME bundle by attaching the actual assembled anchors to the
existing N128 source driver: one5min/1GPU/2CPU/3GiB allocation (previous
same-field host peak~2.4GiB plus20%, rounded). Freeze the training diagnostic
method offsets only for this conditional connection; do NOT multiply their
fitted likelihood as a prior, fit heldout offsets or copy the relative excess
scatter into the FP group kernel. Use the saved closed split and consistent
observed-z luminosity convention. Compare257/513 distance and129/257 global
zero nodes; no new scalar derivative series, nuisance fit or gravity run.
Scores at these fixed estimated offsets are wiring diagnostics, not calibrated
evidence, independent validation or posterior samples. The full field inference
will need to infer/marginalize shared method calibrations jointly using each
observation once. Do not freeze them permanently at these diagnostic estimates.

MW/M31/M33: this is not a component identification method. R3 candidates
must come from each NEW same evolved field, retain MW/M31 role ambiguity and
unresolved M33, and let their actual observables constrain that SAME state.
No native truth identity enters any generated-field candidate selection.
R2 actual posterior is still outstanding, followed by R3<=0.3 LG, R4 precise
forward validation, and R5 phase-consistent zoom ICs.

## Observed bridge and conditional fit

H200 Slurm406182 COMPLETED25s/exit0. Three focused numerical/source tests
pass; source assembly/fit takes0.38s. Slurm's3,396KiB RSS reading misses the
short-lived JAX test and is NOT a trustworthy process peak. Result and raw
reusable inputs:
`/gpfs/kjhan/CF4/z0_density/r2_cross_method_anchors_v1/`.

There are114 usable non-FP measurements in96 source groups:23 same-PGC
links and91 explicit T17 links. No inferred1PGC-to-Tempel assignment is used.
This is the coverage of these strict, available catalogue links, not proof
that no other physically corresponding observations exist. Source-membership
conflicts at PGC2179170 and94030 are recorded; original FP/catalogue data
are unchanged. Closing the split promotes one FP row to holdout, leaving
5,330 training/1,491 heldout FP groups. The anchor sample has72 training
measurements in65 groups and42 heldout measurements in31 groups.

| Method | Training / heldout measurements | Fitted relative offset (mag) |
| --- | --- | --- |
| SN Ia | 60 / 28 | -0.0433 ±0.0503 |
| TF | 6 / 6 | +0.1176 ±0.2128 |
| SBF | 4 / 8 | -0.1069 ±0.1206 |
| SN II | 2 / 0 | -0.1131 ±0.3808 |

Errors are conditional Gaussian offset SDs at the fitted excess variance,
not total source/systematic errors. No linked TRGB/Cepheid/maser rows exist
within this strict selected subset.58 anchors carry CF3=1 and56 CF3=0;
neither set is an independent source-calibration validation sample.

The fitted total relative group-excess SD is0.17851mag, with an interior
variance optimum after17 optimizer evaluations. Training REML NLL changes
36.13841→34.43624. Heldout log predictive density changes-16.87131→-16.54040
(only+0.33092nat), integrating training-only method-offset uncertainty while
holding the fitted variance fixed. Heldout quadratic changes53.9826→42.0415
over42 measurements. This modest conditional improvement does NOT validate
the covariance model or establish the origin of the excess.

There are1,158 multi-FP groups and3,199 within-group contrast dimensions,
but their common-shift direction is exactly null. The0.17851mag cannot be
identified as pure FP common noise, divided by5 and silently substituted for
the previous0.005dex mechanics scatter. It also includes possible depth,
selection, association, population and non-FP/source-calibration effects.
Close this one-fit diagnostic without holdout tuning or a parameter sweep.

## Same-state connection and decision

H200406183 COMPLETED55s/exit0; three existing group regressions pass, for
six focused/reused tests across this bundle. Batch MaxRSS1,939,900KiB
(~1.85GiB); source control25.78s. No new gravity, posterior fit or simulation
output was created. Results:
`/gpfs/kjhan/CF4/z0_density/r2_cross_method_same_field_v1/`.

All114 real non-FP marks were included at their source group distances in the
same selected radial/velocity integrals as10,020 FP rows. There is no separate
non-FP distance integral multiplied afterward. Training-estimated method
offsets are fixed conditional inputs in this run, NOT independent priors;
their posterior uncertainty is not propagated in this wiring comparison.
The fitted relative excess scatter is NOT inserted as an FP shared error.
The kernel interface leaves method offsets as explicit caller parameters for
the joint field/calibration inference. The source group's closed split is used.

Finer distance/global-zero rule513/257 gives saved-state-vs-zero-velocity
log-factor contrasts-26.773182 training/-47.427341 all groups. Coarse/fine
changes are0.00073726/0.00011531. The anchor-added mark-factor refinement
changes by at most0.00067951. These are numerical sensitivities at the saved
unconditional N128 state, not a posterior or scientific model comparison.
Anchor-on/off log factors use a fixed35mag data reference and are NOT evidence
for adding data or a measure of recovered information. Field contrasts cancel
that reference. The raw catalogue measurements are preserved unchanged.

Driver decision: accept the observation bridge and same-distance implementation;
close this bounded diagnostic fit. There is genuine relative-distance input,
but96 anchored groups out of6,821 and weak non-SN-Ia training coverage do not
certify the full population law. The JSON's `independent_FP_contrasts` counts
linear contrast dimensions, NOT statistically independent measurements.
Do not add further same-data residual/variance sweeps. No calibration prior
from this fit may be multiplied with its own reused measurements.

Next essential model work: retain these raw observations once, jointly infer
method zero points and shared source-fit/FP uncertainty with the SAME field,
and specify the selected-group radial/inclusion/FoG law. The structural
measurement/selection dependencies cannot be replaced by a fitted scalar
excess, the old0.5 central probability or a declared zero inclusion uncertainty.
Physical central/satellite calibration and absolute-scale anchoring are still
unresolved; production R2/N256 remains unpromoted. Do not hide these limits by
calling this small conditional fit the final posterior or a new IC product.
