# Bundle A — actual-data preview input/output contract

2026-09-07. Implementation preparation within the approved Bundle A.
Authority: `CF4_MASTER_PLAN.md`. This document does not claim a fit has run.
Choose and record the field model after job333765's paired comparison;
do not launch another mock-repair series or advance Bundle B automatically.

Model decision after completed comparison: retain the original PM-calibrated
lognormal joint prior and24 nuisance parameters, without curvature, solely
for a model-stress preview. The quantile candidate failed adoption criteria;
no claim that the baseline is calibrated. Execution plan:
`config/cf4_actual_data_preview_v1.json`.

## Inputs and likelihood

- CF4: `data/cf4_clean.npz`, through the existing
  `cf4_linear_cr.prepare_bgc_catalog(fixed.frozen_args(...), catalog)`.
  Retain its actual `vobs`, row identities, positions, directions, variances,
  four nuisance-design columns and fixed holdout mask. Do not substitute
  synthetic noise or use `prepare_fixed_design` alone: it discards `vobs`.
- Existing BGc settings include target cz=1500–18000 km/s, window801,
  error_scale=.9, bulk prior150 km/s, H0-offset prior3 km/s/Mpc, and
  train-only reference rows. Record selected row counts and units at runtime.
  These are retained approximations, not newly validated distance-error
  physics. This cut does not provide direct MW/M31/M33 measurements.
- Counts: `/gpfs/kjhan/CF4/z0_density/datum_bearing_twompp_v1/datum.npz`.
  Use `counts_train`, `counts_holdout` and `raw_selection_exposure` directly.
  `counts_all` is only an input-consistency total, not extra training data.
  Existing split: 29,257 training and 7,378 heldout of 36,635 galaxies.
- Preserve six population definitions, their published external density/bias
  priors, the raw exposure, spherical RSD/FoG and native field/count-grid
  coordinate convention. Never renormalize the prior or exposure to observed
  totals. Examine population-normalization and radial/angular residuals:
  published priors/selection need not describe this restricted catalog well.
- The existing 17,007 same-object exclusions avoid direct catalog reuse, not
  all shared selection/systematic covariance. The deterministic count split
  is a predictive partition, not an independently randomized Poisson trial.
  These limitations must accompany any conditional-Poisson likelihood result.

## Minimal implementation and checks

Reuse the existing HMC/field-summary driver with an explicit actual-data
branch. No fake truth arrays: set truth absent, omit truth correlation,
RMSE and coverage, and plot observations/mean/sample/uncertainty instead of
a fictitious truth comparison. Keep existing mock behavior unchanged.

One scoped Slurm input check verifies row/data binding, units/shapes,
finite gradient, positive-count support and heldout exclusion. Preserve
unrelated work and running-job code. At most one four-chain actual-data fit,
with the existing convergence thresholds and a recorded model/source commit.
Use Slurm and estimate peak memory +20%; no new generic test framework.

## Product and interpretation

- Density posterior mean, uncertainty and selected individual samples;
  the smooth posterior mean is not an individual simulated universe.
- Mean three-component velocity and its posterior uncertainty. Label this
  uncertainty separately from observational error and physical dispersion;
  population FoG widths do not supply a resolved sigma_v(x) map.
- Saved observed/predicted count and radial-velocity residuals, population
  and spatial summaries, heldout predictions, nuisance and chain diagnostics.
- Exposure/information limitations for unobserved regions; no global or LG
  observational-resolution claim. The preview spacing is still12 cMpc/h.

A mock adequacy pass allows only a preliminary model-based product, not
scientific calibration. If the tested correction fails, record whether a
baseline actual-data run is still useful as a clearly labelled model-stress
diagnostic; if not, report the specific block. Never call sampler convergence
alone a successful reconstruction. Review this actual-data outcome and design
the LG/dynamic-connection bundle before requesting Bundle B approval.
