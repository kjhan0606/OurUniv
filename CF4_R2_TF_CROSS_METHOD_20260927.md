# R2/5 — TF cross-method source consistency screen

The live count+FP+TF factor is numerically connected but its TF Gaussian
group-summary errors, shared zero point and inclusion law are not yet
validated. Use the frozen CF4 group source to compare TF distance moduli
with independent *measurement methods* within the same catalogued group
(FP, SNIa, optical and infrared SBF), separately from the disjoint TF-only
training set. Inspect robust relative offsets and broad redshift/sky splits.
Do not use the combined `DMav/DMzp` as an independent method. Do not tune
the joint target on these residuals or call them truth: CF4 methods underwent
joint calibration, and their cross-method covariance is unknown. This is a
source-consistency screen, not a selection/mock or posterior calibration.

One A100/2CPU/4GiB/10min Slurm allocation, no gravity or sampler. Q-GOAL:
tests whether the newly connected all-sky TF summaries are at least internally
compatible with other CF4 distance methods before investing in source
selection and sampler work. Q-LEAN: one predeclared overlap table and coarse
splits, no residual-driven parameter fit or threshold ladder.

MW/M31/M33 are not inferred here. Their future ambiguous and possibly
unresolved roles must be identified from each NEW generated field; observed
LG quantities must constrain that same field. Native truth identities cannot
seed/select generated candidates.

## Result and decision

The A100 request406486 was cancelled while pending for resources; the same
script ran on typed-H100 **406487 COMPLETED/exit0**. Pinned result:
`/gpfs/kjhan/CF4/z0_density/r2_tf_cross_method_screen_v1/result.json`.
The source has38,053 groups;10,035 have positive TF measurements, and9,533
meet this screen's broad cz/error range. TF-only training/heldout medians in
cz are5917/5739 km/s. The8,502 TF-only groups have median and90th-percentile
TF contributor count **one**; their quoted modulus-error median is0.41mag.
Thus a fixed150-km/s *group* velocity width is especially provisional for
this mostly-singleton sample.

Within the *same source groups*, TF-minus-FP has669 overlaps, median
+0.016mag, MAD0.354mag. TF-minus-SNIa has264 overlaps, median-0.1375mag,
MAD0.2275mag. Optical/infrared SBF overlaps number61/42 and have medians
-0.038/-0.0665mag. The TF/FP median varies from-0.031 to+0.0435mag
across supergalactic hemispheres; these broad subdivisions are descriptive,
not fitted gradients. Formal zero-covariance normalized differences are
included in the JSON only as a screen; they cannot estimate a source-error
inflation because the CF4 method calibrations are shared.

Decision: no obvious large TF-vs-FP source mismatch, but the method-specific
SN offset, mostly-singleton TF sample and unknown cross-method covariance
prevent fixing the TF zero/width or declaring a calibrated likelihood from
this check. Do not tune on these same overlaps, double-use them as independent
validation, or launch N256. The next model must distinguish singleton/group
redshift scatter and carry method-calibration uncertainty while testing
selected-sample sensitivity and joint-field sampler equilibration.
