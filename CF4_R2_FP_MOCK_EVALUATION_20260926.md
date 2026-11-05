# R2/5 selected-mock measurement assessment

Evaluate the one fixed official SDSS mock already acquired. No further
download, fitted nuisance, error inflation or gravity calculation. Compute
the source skew-normal CDF at known truth using its published MOMENTS,
separately for central/satellite and fixed z_true bins0-.03,.03-.06,>=.06.
Report mean eta bias, standardized residual mean/SD and truth containment
of central68/90/95% source-PDF intervals. Both original individual-redshift
and converted host-redshift coordinates must give identical probabilities.
Use a Gaussian special-case check and source-row alignment in the same job.

These are diagnostic source-PDF intervals, NOT the posterior after the
selected-group radial prior/velocity law. All rows share a fitted FP; one
mock box and member correlations forbid treating33881 rows as independent
universes. No binomial significance gate, automatic coverage correction or
permanent central/satellite cut. Native labels only stratify evaluation.

Q-GOAL: establish whether the source measurement component itself displays
gross bias/miscoverage before investing in full field inference. Q-LEAN:
one compact existing-data evaluation, no new testing framework or fit series.
One Slurm CPU on a typed H200 allocation after H200/H100/A100 check,1GiB
(<=0.8GiB estimate+20%, rounded),5-minute cap. Preserve all earlier results.

MW/M31/M33 remain ambiguous candidates in each NEW field; unresolved M33
must remain explicit and their observables constrain the SAME field. Truth
labels in this mock never seed/select generated LG components. R2 remains
the current stage; R3 LG, R4 precise forward check and R5 zoom are outstanding.

Passing does not validate Tempel group recovery, richness-corrected FP fits,
group inclusion or COM-vs-PM discrepancy. If measurement shape is reasonable,
close this component check and focus on the actual group selection/covariance
bottleneck rather than retune the source measurement PDF on one mock.

## Completed assessment

Typed-H200406150 COMPLETED3s/exit0. Both embedded coordinate/Gaussian checks
pass. Result:
`/gpfs/kjhan/CF4/z0_density/r2_fp_mock_evaluation_v1/result.json`.
No parameter fitting, extra download or gravity run occurred. Short scheduler
RSS sampling does not establish an accurate peak memory measurement.

Across33,881 galaxies, source-PDF central68/90/95% intervals contain truth
at68.540/90.325/95.257%. Standardized residual mean-0.01270,SD0.99047;
mean eta bias-0.002603dex. The reference-variable conversion changes CDF
values by at most2.22e-16. Retain the source skew-normal moment conversion;
this fixture supplies no reason to inflate its individual measurement errors.

Population structure remains: centrals have mean eta bias-0.008168dex,
satellites+0.001701dex; below z_true=.03 they are-0.010719 and+0.009566dex.
Their aggregate nominal95% containment is95.316% and95.211%. Good marginal
width/containment does not eliminate a population-dependent zero-point or
selection effect, nor certify grouped products of correlated measurements.
Do not calibrate two new offsets from truth labels or discard centrals/
satellites to improve this score. These labels are not observed for every
real FP galaxy. Shared source fitting and one correlated mock limit inference.

Driver decision: accept individual source-PDF shape as a development component
with explicit population-bias caveat; close this one-mock width check. This is
not independent end-to-end validation, a richness-corrected FP calibration,
or a full selected-distance posterior. Next implementation must address the
group-level dependence/selection and latent population uncertainty in the
actual same-field model. Repeating individual-error checks or simply adding
more mock rows would not solve the current scientific bottleneck.
