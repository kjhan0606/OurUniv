# Bundle A: approved focused correction and one actual-data refit

Authority: user approval after [ACTUAL_PREVIEW_DIAGNOSIS.md](ACTUAL_PREVIEW_DIAGNOSIS.md).
Plan: `config/cf4_actual_data_corrected_v2.json` (inherits unchanged baseline
settings from V1). Output: `/gpfs/kjhan/CF4/z0_density/actual_data_corrected_v2`.
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

Status: implementation prepared; tests and input preflight not yet submitted.
One corrected actual-data fit is authorized only after preflight success.
No next-bundle launch, no automatic additional fit if scientific gates fail.
