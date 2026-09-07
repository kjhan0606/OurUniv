# Bundle A: approved focused correction and one actual-data refit

Authority: user approval after [ACTUAL_PREVIEW_DIAGNOSIS.md](ACTUAL_PREVIEW_DIAGNOSIS.md).
Plan: `config/cf4_actual_data_corrected_v2.json` (inherits unchanged baseline
settings from V1). Output: `/gpfs/kjhan/CF4/z0_density/actual_data_corrected_v2_run1`.
Old sources/data products/results remain available; no historical builder edits.

## Scientific changes fixed before inspecting the new fit

1. Catalog magnitudes use M_phys-5log10(h); Schechter selection uses luminosity
   distance in Mpc/h. The same six numerical bins, Mstar and h=.6711 now agree.
   Regenerate the counts and globally order6-integrated raw exposure, without
   matching observed totals or admitting positive counts outside model support.
2. Keep CF4 crossmatch exclusions; do not silently restore same-object counts.
   Keep frozen metadata rejects and apply the same published-map/mark <=.05
   consistency rule to newly admitted magnitude rows. Estimate survival
   probabilities from calibration-only parent membership labels. Fixed strata:
   six luminosity/apparent populations x radial edges5,30,60,90,120,150,180 cMpc/h.
3. Use the existing recno hash, with <.2 heldout unchanged, [.2,.4) reserved
   for selection calibration, and >=.4 for the count likelihood. Calibration
   rows are never density-fit or held-out rows. Their counts condition a mark
   likelihood only; no total is used to recenter density/rate priors.
   The field-training count fraction changes from.8 to.6; predictions retain
   the.2 held-out fraction. Comparisons with V1 are not paired score tests
   because magnitude-selected membership has changed.
4. Independent Beta(1,1) stratum probabilities updated with calibration-only
   survivor/reject labels enter as uncertain selection parameters. Sample their
   exact logit-Beta densities (including Jacobian), not fixed best-fit weights.
   Integration sums raw exposure by shell BEFORE applying the uncertain survival
   factors. Keep the original full-parent density/bias prior centres; their
   compatibility with the PM field/link remains approximate, not certified.
5. Reparameterize the four Gaussian radial nuisance coordinates as
   q=mu(field,y_train)+chol(C)*epsilon. This is a triangular, constant-Jacobian
   transform of the same joint field/radial target, not resampling saved H0.
   Epsilon decouples exactly; nuisance storage remains in physical standardized
   q coordinates. The field-dependent conditional means are explicitly included
   in Rhat/ESS gates. Keep the same4 chains,512 warmup,1024 sampling steps.

The survival model assumes conditional ignorability within each luminosity/
radius stratum; sky/environment/assembly dependence, LF uncertainty and
selection correlations remain limitations. It is a bounded correction to a
model-stress inference, not a claim that these effects are absent or that
calibration is now guaranteed. Using a disjoint calibration subset avoids
using the same survivor counts to estimate completeness and fit density.

## Execution and checks

Slurm only. CPU preflight:2 cores,3500 MiB estimated +20%=4200 MiB,20min.
GPU refit:1 GPU,4 cores,7000 MiB estimated +20%=8400 MiB,2h limit,
partitions a40/a100/h100/h200, exclude syn06. CPU aggregation:1200 MiB,5min.
Previous fit took21min; this adds selection parameters and coordinate transforms,
so its runtime is not guaranteed to match. No new PM run or mock fit.

Targeted regression checks: magnitude/selection equivalence to independently
converted physical units; preservation of heldout and disjoint calibration;
mark-calibration insensitivity to noncalibration labels; uncertain-selection
forward factor and derivative; exact Gaussian radial energy, field derivative,
and exclusion of heldout velocities. Reuse the existing physical/model/data
tests. Full actual-data preflight also checks finite gradients and exact
radial-block energy before a GPU submission.

Initial CPU job334332: all14 regressions passed in86.2s. Input preparation
then failed before catalog loading because the JAX/circle environment lacks
healpy. Fix: use the already established Python3.13 astronomy environment
for catalog/selection preparation and circle/Python3.11 for inference checks.
No package installation or scientific-model change. Preserve the initial
failed log and empty output directory; retry uses the new run1 output above.

Environment-corrected CPU job334344 (source600c682) completed in3m32s,
exit0, peak RSS1202496 KiB. All14 regressions passed again; full corrected
input, finite-gradient, heldout exclusion and radial energy checks passed.
An Astropy log-distance warning occurs on source rows with invalid/zero
redshift; these are rejected by the pre-existing finite/positive-redshift
catalog cuts and do not enter the field input.

Corrected parent:57249 eligible rows. Calibration-only parent sample:11504
rows,8321 survivors. Field counts:24998 training and8378 heldout (33376 total),
population totals[5895,6338,1455,6264,10607,2817]. CF4 remains19313 rows,
15346 training and3967 heldout. No calibration or old heldout row is used
as a training density count. Previously rejected CF4/metadata objects remain
excluded from the density likelihood. Empty calibration strata retain the
uncertain Beta(1,1) prior, never an invented measured completeness.

| Job | Purpose | State at submission |
| --- | --- | --- |
| 334344 | CPU regression + corrected actual-input preflight | COMPLETED/PASS |
| 334345 | One corrected actual-data four-chain GPU fit | Submitted afterok:334344, source600c682 |
| 334346 | Result aggregation, including failed/incomplete fit reporting | Submitted afterany:334345 |

The corrected initial homogeneous full-survivor rate expectations are
[4623.5,7339.9,1699.5,4437.8,13483.5,4161.2]. Compare with the observed
training+heldout totals only after multiplying these predictions by.8
(the separate.2 calibration subset is not in those observed totals).
The brightest populations' observed/reference ratios are now about1.59/1.76,
not the previous4.76/5.18. This is descriptive, not a controlled single-cause
test or fitted-posterior result; magnitude membership and calibration split
changed. Remaining discrepancies must be evaluated in the actual fit.

Status: one corrected actual-data fit submitted after preflight success.
No next-bundle launch, no automatic additional fit if scientific gates fail.
