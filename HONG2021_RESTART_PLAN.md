# Hong et al. TNG100 reproduction: restart audit and plan

Date: 2026-08-01

## Executive diagnosis

The first run must not be discarded as if its weights learned nothing.  The
epoch-200 network evaluated with mini-batch statistics gives an unaugmented
training MSE of `0.00613`, consistent with the logged minimum training loss
`0.00615`, with no cube above MSE `0.05`.  The same weights evaluated with the
saved BatchNorm running statistics give MSE `0.02262`; 39 of 432 training cubes
exceed `0.05` and some outputs saturate at `-1` or `+1` over a large volume.
The first P0 problem is therefore inference-time normalization, not absent
weight learning.  This is not checkpoint corruption: epoch 200 reproduces its
logged validation MSE in inference mode.  It is a train/eval function mismatch
on the heterogeneous training cubes, amplified by the biased spatial split.

Three other effects must be kept separate:

1. The frozen validation split is much denser than the training split.  Median
   galaxy counts are 1,299 versus 990 per cube, and median occupied-cell radial
   velocity dispersions are 389 versus 247 km/s.  About 90% of pairs of
   validation cubes overlap one another, compared with 22% for training cubes.
2. The current validation figure compares a one-voxel-thick slice of normalized
   log density.  Hong et al.'s visual comparison uses a 5-Mpc/h-thick projection
   of linear density.  The visual claim of failure is not yet an apples-to-
   apples paper comparison.
3. A deterministic MSE network estimates a conditional mean.  Small filaments
   not determined by the sparse galaxy/velocity input are expected to be
   smoothed.  Hong et al.'s own prediction is visibly smoother than its truth.
   Exact high-k filament recovery requires a stochastic residual stage even
   after the literal Hong baseline is reproduced.

## What is already correct

- TNG100-1 snapshot 99, `75 Mpc/h` box, and all `1820^3` dark-matter particles.
- `240^3` global grid, `64^3` sub-cubes, `20 Mpc/h` cubes, and
  `0.3125 Mpc/h` voxels.
- Center numerical cut `4e10 < SubhaloMassType[:,4]*1e10 < 1e11`, which gives
  the paper's reported 988 candidates.  Applying the documented extra `1/h`
  gives 1,552 and therefore does not reproduce the reported count.
- Target luminosity cut `M_B < -15`, B-band index 1, center-relative radial
  peculiar velocity, `|b|<10 deg` mask, and the two published input channels.
- Target `y=log10(rho_dm/rho_mean)/4.5`, 24 augmentations, full channel widths,
  MSE, Adam hyperparameters, batch 6, and 200 epochs.
- The encoder/decoder sizes, skip connections, reflection padding, convolution
  kernels, nearest-neighbor upsampling, and output `tanh` match the published
  diagram at the documented level.

## Missing or underspecified items

### P0: must resolve before another 200-epoch run

1. **BatchNorm inference fidelity.**  Determine the exact Keras normalization
   axes, operation order, moving-stat update, and inference behavior.  Require
   batch-size/order-invariant inference and agreement between logged and
   re-evaluated training loss before accepting a checkpoint.
2. **Representative spatial split.**  The unpublished 525 IDs cannot be
   recovered from the paper.  Generate all 988 candidate features first and
   choose cross-split-nonoverlapping 432/93 sets while matching the distributions
   of galaxy count, target mean/variance, velocity dispersion, and center mass.
   Keep several valid split realizations to measure split uncertainty.
3. **2pCF normalization.**  Freeze whether `delta` uses the cosmic mean,
   validation-ensemble mean, cube mean, or another convention, and whether
   sub-cube faces wrap.  The first raw calculation mixed spatial correlation
   with the different one-point means, `1.921` in truth and `3.024` in the
   prediction.
4. **Paper-matched visualization.**  Add 5-Mpc/h projections of linear density,
   identical color limits, density histograms, Fourier cross-correlation, and
   transfer functions.  Do not judge filament recovery from the current single
   log-density slice alone.

### P1: data-selection ablations

The paper gives no stellar-mass lower limit for target galaxies; its published
selection is `M_B < -15`.  A new lower mass cut must therefore be an explicitly
labeled numerical-resolution ablation, not silently inserted into the literal
baseline.

Current snapshot-99 counts are:

| Target selection | Number in `(75 Mpc/h)^3` |
| --- | ---: |
| `M_B < -15` (current) | 48,296 |
| plus `SubhaloFlag == 1` | 41,927 |
| plus `Nstar >= 100` | 39,475 |
| plus `Mstar >= 1e8` in stored mass units | 37,897 |

The current target sample contains 6,369 (`13.2%`) objects with
`SubhaloFlag=0`.  Its stellar-mass range extends down to about `6.9e5` in the
stored numerical convention; a small number contain only one star particle.
Among `SubhaloFlag=1` targets, 5.8% have fewer than 100 star particles.

Run the following labeled variants on identical balanced splits:

- `L0-paper`: `M_B<-15` only.
- `L1-flag`: `M_B<-15` and `SubhaloFlag==1`.
- `L2-res100`: L1 plus `Nstar>=100`.
- `L3-mstar1e8`: L1 plus `Mstar>=1e8`.

More restrictive samples contain less positional and velocity information and
may smooth filaments further; their purpose is to test numerical contamination,
not to assume that a higher mass cut is automatically better.

Other P1 ablations are NGP versus CIC dark-matter assignment, exact observer-
centered versus global-grid-aligned cubes, and raw versus explicitly documented
input scaling.  These must be changed one at a time.

## Restart sequence

### Stage 0: preserve and diagnose v1

- Keep v1 data, checkpoints, predictions, and metrics immutable.
- Implement a BatchNorm audit that reports training mode, inference mode,
  recalibrated mode, batch-size invariance, saturation fractions, and per-cube
  outliers for every checkpoint.
- Build the paper-matched 5-Mpc/h projection and Fourier diagnostics from the
  existing predictions.
- Resolve the primary 2pCF definition before using it as a gate.

Stage-0 results on the Ada GPU are now frozen in
`evaluation/tng100_v1/bn_audit/epoch009.json` and `epoch200.json`:

| Checkpoint and mode | Train MSE | Validation MSE | Train cubes with MSE > 0.05 |
| --- | ---: | ---: | ---: |
| epoch 9, saved inference | 0.01134 | 0.01263 | 0 |
| epoch 9, batch statistics | 0.01122 | 0.01270 | 0 |
| epoch 200, saved inference | 0.02262 | 0.01514 | 39 |
| epoch 200, batch statistics | 0.00613 | 0.01580 | 0 |
| epoch 200, cumulative recalibration | 0.01445 | 0.01605 | 17 |

Epoch 9 passes the 10% inference-fidelity criterion (ratio `0.993`).  Epoch 200
fails it (ratio `3.678`), while its batch-statistics loss agrees with the logged
training loss (ratio `0.997`).  A simple one-pass recalibration is therefore not
an accepted repair.

The paper-style 5-Mpc/h projections are frozen as
`paper_visual_minimum_validation.png` and
`paper_visual_minimum_training.png`.  They show that both checkpoints recover
the dominant nodes and filaments.  Predictions are smoother than truth and
lose small branches, as expected for a deterministic conditional-mean model
and qualitatively similar to the published figure.  The earlier interpretation
that the whole network failed to learn is withdrawn.

### Stage 1: rebuild metadata and splits, not density particles

- Reuse the validated `240^3` dark-matter count grid; rereading `1820^3`
  particles is unnecessary unless testing CIC.
- Build a catalog table for all 988 observers containing center mass, flag,
  central/satellite status, local target count, target flag/resolution counts,
  velocity moments, and density moments.
- Generate at least three cross-split-nonoverlapping 432/93 splits with matched
  feature distributions.  Freeze these manifests before training.

Stage 1 is complete for the literal `L0-paper` catalog.  The reusable metadata
for all 988 observers is in
`derived/hong2021_v2/center_metadata.h5`.  The preregistered matching features
are center stellar mass, local galaxy count, occupied-cell count, occupied-cell
velocity dispersion, target mean, and target standard deviation.  Merely
changing the compact validation-region seed improved the old maximum absolute
standardized mean difference (SMD) from `1.024` to only `0.741`.  Joint spatial
and feature optimization was therefore required.

The frozen three-split manifest is
`derived/hong2021_v2/joint_balanced_splits_v1.json`.  Its maximum absolute SMDs
are `0.086`, `0.089`, and `0.091`; maximum one-dimensional KS distances are
`0.148`, `0.158`, and `0.154`.  All pass the preregistered `|SMD|<0.25`
criterion.  Their minimum cross-split L-infinity separations are `20.049`,
`20.071`, and `20.039 Mpc/h`, so no 20-Mpc/h training cube overlaps a
validation cube.  Validation-set pairwise Jaccard similarities are 0.62--0.72,
below the frozen 0.8 diversity ceiling.

### Stage 2: cheap discriminating pilots

- First reproduce the current `L0-paper` data with the corrected normalization
  implementation.
- Run 15--20 full-width epochs because the first run reached its minimum
  validation loss at epoch 9.  Do not spend another 200 epochs until inference
  invariance passes.
- On the same split, run L1, L2, and L3 pilots.  Change no architecture or loss
  in this comparison.
- If the literal Keras BatchNorm behavior remains uncertain, implement a small
  TensorFlow/Keras reference and compare layer outputs and moving statistics on
  identical batches before choosing the production implementation.

The 20-epoch balanced-split `L0-paper` pilot is complete.  Epoch 7 passes BN
inference fidelity (`0.994` re-evaluated/logged training-MSE ratio, zero
outliers), whereas epoch 20 fails (`2.083`, 11 training outliers).  Late-epoch
BN train/eval divergence therefore persists even after split balancing; epoch
7 is the only accepted literal-baseline checkpoint.

A frozen-weight layer and batch-composition probe localizes the epoch-20
failure.  Saved-stat inference has 11 unaugmented training cubes above MSE
`0.05` (worst `0.788`), while live batch statistics put every cube below
`0.0103`.  At the deepest encoder BN there are only 48 activation values per
channel and the most extreme batch mean and variance differ from the saved
statistics by `61.9` saved standard deviations and a factor of `4.46e4`.
The exact 24-fold augmented audit likewise gives MSE `0.01661` and 208 failed
augmentations with saved statistics, versus `0.00896` with shuffled live batch
statistics and `0.00932` when transforms of the same cube are grouped.  A
shuffled augmented cumulative batch-moment recalibration still leaves two
catastrophic unaugmented cubes, so it does not rescue a fixed inference model.
This recalibration averages per-batch moments; it must not be mislabeled as an
exact pooled population variance because between-batch mean variance is absent.

The controlled successor changes only BatchNorm to GroupNorm and keeps the
split, inputs, target, 24 augmentations, widths, batch, optimizer, learning
rate, seed, and lack of input standardization fixed.  Its 20-epoch Ada pilot is
tracked in `config/hong2021_tng100_v3_groupnorm_pilot.json`.

That controlled GroupNorm pilot is complete.  It eliminates the normalization
failure: both epoch 14 and epoch 20 have zero MSE-`0.05` outliers and exactly
identical train/eval outputs.  Epoch 14 minimizes validation MSE at `0.012326`,
slightly better than the accepted BN epoch-7 value `0.012532`; 2pCF selection
instead prefers epoch 20.  The nominal batch-1/6 maximum-difference gate is
missed at `1.48e-4`, but the fixed-eval BN baseline has the same CUDA
batch-shape numerical effect (`1.43e-4`), so it is not GroupNorm coupling.

The scientific resolution gate still fails.  The predicted/truth mean-density
ratio improves from `1.895` to `1.508` but exceeds the frozen 20% bias limit.
Raw cosmic-mean 2pCF KS is `0.738/0.455/0.324` over the three scale ranges,
and log-density `(T,r)` is only `(0.385,0.338)` at `k=3--6 h/Mpc` and
`(0.238,0.165)` at `k=6--10 h/Mpc`.  These high-k correlations are slightly
worse than BN epoch 7.  The 5-Mpc/h slabs remain visibly smooth and omit the
truth's minor filaments.  GroupNorm is therefore the correct deployment
normalization, but this deterministic Hong architecture is not a sufficiently
resolved present-density estimator and must not advance to CF4 or a 200-epoch
production run.

Post-result review identified one invalid gate in that decision.  The linear
ratio of ensemble mean densities is not a clean bias statistic when the
log-density residual scatter is about 0.5 dex; exponentiation and its
correlation with truth density strongly affect that ratio.  Epoch-20
GroupNorm's log residual is `0.009 +/- 0.506 dex`, close to Hong Table 2's
`-0.014 +/- 0.543 dex`, even though its linear mean ratio is `1.508`.
The original gate failure remains recorded for audit, but the successor gate
in `config/hong2021_evaluation_gates_v2.json` uses an absolute mean log bias of
0.05 dex and treats the linear ratio as diagnostic only.  Cosmic-mean 2pCF is
the frozen physical convention; absolute comparison with Hong remains
diagnostic because their finite-cube convention is unpublished.

A zero-training-cost falsification used the v1 epoch-200 checkpoint with live
mini-batch BN statistics.  Although log-density transfer at `k=3--6 h/Mpc`
rose to `0.661`, phase correlation fell to `0.248` (and to `0.155` at
`k=6--10 h/Mpc`), versus `0.356/0.190` at epoch 9.  Long training therefore
adds mostly uncorrelated small-scale power and does not recover the missing
phases.  A 200-epoch deterministic continuation is rejected independently of
the revised density-bias gate.

The target-density audit rules out an accidental void-heavy selection.  In
2,000 random 20-Mpc/h TNG cubes, the mean density is `1.022` cosmic units and
the mean volume fractions below `0.1/0.5` cosmic density are `0.465/0.819`.
The actual training cubes are denser (`1.509`) and less void-rich
(`0.425/0.778`) because they are centered on MW-mass observers.  Train and
validation remain matched in these added environment statistics.  The more
serious data limitation is reuse: 432 training cubes sum to 8.19 box volumes
but cover only 64.3% of the TNG box, equivalent to 33.9 unique cube volumes;
covered voxels appear in 12.7 cubes on average.

Uniform voxel MSE also misaligns with the desired structures.  On validation,
cells below half the cosmic density occupy 78.1% of the volume and contribute
54.0% of squared error.  Cells above 10 times cosmic density contain 69.1% of
the mass but contribute only 6.8% of the loss.  The GroupNorm prediction
overestimates `rho<0.1` cells by `+0.40 dex` and underestimates
`rho=2--100` cells by `-0.66` to `-0.71 dex`, directly demonstrating
regression toward the volume-weighted mean.  Thus the density file is valid,
but spatial redundancy and the uniform objective are plausible causes of the
smooth filament reconstruction.

The final standardized-GroupNorm falsification is also complete.  It used
training-only occupied-cell scaling and a 20-epoch cosine learning-rate decay;
the precommitted primary checkpoint is minimum-validation epoch 7.  Relative
to unstandardized GroupNorm, validation MSE is effectively unchanged
(`0.012335` versus `0.012326`).  Log-density phase correlation rises only from
`0.3369` to `0.3523` at `k=3--6 h/Mpc` (delta `0.0154`, below the required
`0.05`) and from `0.1666` to `0.1736` at `k=6--10 h/Mpc`.  Its log-density
transfer is `0.395/0.253` in those bands, so the slight amplitude recovery does
not recover the missing phases.  Mean log bias (`-0.0154 dex`), outlier count
(zero), and exact train/eval mode invariance pass their gates.

An automatic 2pCF-only selector preferred epoch 20, but it is not substituted
for the preregistered checkpoint: its validation MSE is worse (`0.013018`) and
its high-k phase correlations (`0.3528/0.1789`) still fail the same gate.
Density-stratified residuals remain a regression to the mean: the epoch-7
model overpredicts `rho<0.1` by `+0.371 dex` and underpredicts
`rho=2--100` by `-0.694` to `-0.700 dex`.  Standardization therefore falsifies
the last uniform-MSE deterministic remedy.  This branch is frozen; the next
model-development stage is a conditional stochastic residual model, not a
200-epoch extension or CF4 deployment of this checkpoint.

The successor stochastic-residual pilot is now running on LagEunha.  It freezes
the v4 epoch-7 prediction as the conditional mean and learns only the residual
through a smooth `k=2--4 h/Mpc` high-pass transition.  Training uses all 48
signed cube-axis isometries: 24 orientation-preserving rotations and 24
orientation-reversing mirror transformations.  Validation is unaugmented, and
translations are excluded because the radial-velocity observable is tied to
the observer at the cube centre.  The compact GroupNorm diffusion model has
3.60 million parameters; its first epoch reached train/validation epsilon MSE
`0.540/0.426` in 22 seconds with about 6.3 GiB GPU memory.  Acceptance will be
based on ensemble calibration and density/P(k)/2pCF/peak/void statistics, not
on matching an individual random high-k phase realization.

The 50-epoch run and equal-seed ensemble evaluation are complete.  Sixteen
representative validation environments were selected by farthest-point
sampling in observer/input/target feature space, and 16 realizations were
drawn per environment from epochs 21 and 50.  Unbounded cosine-DDPM ancestral
sampling initially exploded because its final beta is nearly one; a broad
training-support bound of eight residual RMS units stabilized the reverse
chain without clipping to the target's nominal `[-1,1]` range.

The stochastic stage is a real improvement but does not yet pass.  It preserves
the frozen low-k mean to below `4e-17` of residual power.  At `k=3--6/6--10
h/Mpc`, epoch 21 raises total log-density power from the deterministic
`0.167/0.066` of truth to `0.740/0.809`; epoch 50 gives `0.707/0.808`.
The density-PDF total-variation distance improves from `0.411` to
`0.193/0.185`, and 2pCF and peak/void counts also move toward truth.  However,
the residual RMS ratios are only `0.871/0.859`, with voxel coverages
`0.528/0.789` and `0.535/0.796` for nominal 68/95-percent intervals.  Both
ensembles are therefore underpowered and underdispersed, missing the frozen
10-percent power gate by 9--19 additional percentage points.  Epoch 21 is the
preferred refinement checkpoint because its high-k power and peak counts are
closer to truth, but neither checkpoint is accepted.  The required HOP gate
was consequently not run.

V5 is not used as evidence that conditional generative modelling itself has
failed.  It received only 3,600 optimizer updates (21,600 examples), because
random isometry augmentation did not multiply the DataLoader epoch length.
It also lacked EMA and fixed-noise validation, ended its cosine-DDPM schedule
at beta 0.999, treated open cubes as periodic, and applied the same soft
high-pass filter both when defining the target and after sampling.  The last
operation squares the transition response and suppresses precisely the
intermediate scales that were reported as deficient.  A posteriori A(k)
rescaling is rejected because it would hide rather than test these failures.

V6 is therefore a fair same-network comparison between EDM-preconditioned
denoising and rectified flow.  It replaces the periodic target with a
reflect-boundary Gaussian-Laplacian residual (sigma 2 cells), projects out only
the exact DC mode after sampling, and never filters a generated residual a
second time.  Both methods use the same 3.60-million-parameter GroupNorm U-Net,
20,000 optimizer updates, batch 6, EMA 0.999, a fixed validation noise bank,
and 40-step Heun sampling.  The train/validation residual RMS values are
0.091686/0.091890, so there is no normalization mismatch.  GPU train and sample
smoke tests and all 24 unit tests pass.  The precommitted evaluation remains
16 identical held-out environments with 16 equal-seed realizations per method;
HOP remains downstream of the statistical field gate.

V6 is complete.  Both methods pass the preregistered ensemble field gate.  In
the `3--6/6--10 h/Mpc` bands, the total log-density power ratios are
`0.924/0.999` for EDM and `0.932/0.982` for flow; their residual RMS ratios are
`0.975/0.971`.  For a 16-member ensemble, sample-quantile intervals have
finite-ensemble expected coverages of `0.600/0.838`, rather than the nominal
continuum `0.68/0.95`.  The measured EDM coverages are `0.593/0.843` and the
flow coverages are `0.589/0.840`, so neither is underdispersed at the available
ensemble resolution.

The downstream grid-HOP audit evaluated all 544 truth, deterministic, EDM, and
flow fields.  Paired environment bootstrap intervals for EDM include unity for
all tested abundance statistics: its all-group, mass-above-`1e13`, and
mass-above-`3e13 M_sun/h` ratios are respectively `0.868`, `0.877`, and
`0.907`.  Flow fails because the corresponding 95-percent intervals exclude
unity, with ratios `0.838`, `0.812`, and `0.735`.  EDM is therefore accepted
and flow is rejected; further steps are not added merely because flow's
objective validation loss was still decreasing.  This grid-HOP calculation is
an Eulerian field-morphology gate with one pseudo-particle per voxel, not a
gravitationally bound particle halo catalog.  The accepted EDM model must next
pass independent EAGLE validation and forward PM/RAMSES plus particle-HOP
testing before any physical halo claim or application to CF4.

Epoch 7 improves the raw cosmic-mean 2pCF KS over v1, from
`0.781/0.590/0.489` to `0.685/0.453/0.348` in the `0--1/1--3/3--10 Mpc/h`
ranges, but it does not reproduce the paper.  Normalizing truth and prediction
by their separate validation-ensemble means gives
`0.520/0.192/0.233`; this is a diagnostic, not an accepted correction, because
the predicted ensemble mean density is `1.895` times the truth mean.

Fourier diagnostics explain the visual result.  For log density, the
mode-weighted `(T,r)` values are `(0.393,0.358)` at
`k=3--6 h/Mpc` and `(0.259,0.176)` at `k=6--10 h/Mpc`; `r(k)` first falls
below 0.5 at `k=3.78 h/Mpc` (wavelength `1.66 Mpc/h`).  Linear density instead
has excessive high-k power, reaching `T=3.84` and `5.59` in those bands while
phase correlation declines.  The excess comes from compact predicted peaks,
not correctly phased filaments.  Thus the literal deterministic Hong baseline
does not pass the high-resolution present-density gate despite its
`0.3125-Mpc/h` output voxels.

An identical diagnostic on the v1 epoch-9 checkpoint separates split effects
from model resolution.  V2 substantially improves raw 2pCF KS, but log-density
phase correlation is unchanged at `k=3--6 h/Mpc` (`0.356` to `0.358`) and
worsens at `k=6--10 h/Mpc` (`0.190` to `0.176`).  Linear-density high-k
transfer increases from `3.29/4.45` to `3.84/5.59`.  Split balancing is
therefore necessary for honest validation but does not recover missing
small-scale information; it makes the compact-peak artifact more evident.

Pilot acceptance requirements:

- Re-evaluated train MSE in inference mode within 10% of logged train MSE.
- No train or validation cube with MSE above `0.05` from normalization failure.
- Prediction invariant to inference batch size and sample order within floating-
  point tolerance.
- No large-volume `tanh` saturation; report both tails explicitly.
- Validation/train feature standardized mean differences below a preregistered
  threshold, initially 0.25 where feasible.
- Better 5-Mpc/h projected morphology, `r(k)`, transfer function, residual PDF,
  and correctly normalized 2pCF than v1.  Model selection cannot use MSE alone.

### Stage 3: one full baseline

- Train only the best justified literal-paper configuration for 200 epochs on
  the Ada GPU, with atomic checkpoints and complete normalization metadata.
- Evaluate minimum-validation, minimum-training, and last checkpoints on all
  frozen splits.  Report split-to-split uncertainty.
- Pass held-out TNG100 before running the independent EAGLE gate.

### Stage 4: recover statistically valid small scales

Even a successful Hong baseline will not identify every true small filament.
Use its output as the conditional mean present-day density and learn or sample
the TNG residual field conditioned on the observables and mean prediction.
Require the ensemble to recover the TNG one-point PDF, 2pCF/power spectrum,
cross-correlation, void/peak statistics, and stochastic variance without
claiming that unconstrained phases are observed.  This is the component needed
before the present density can constrain a high-resolution IC.

Only after Stages 0--4 pass should the model be applied to CF4 and coupled to a
PM/PMWD inverse stage.

### Independent-simulation gate after V6

EAGLE RefL0100N1504 particle access remains unavailable because the account
activation was not returned.  The SQL galaxy catalogue cannot supply the
particle-level dark-matter truth, so it was not used as a substitute.  A
provisional, explicitly weaker cross-code test was instead frozen on 16
independent CAMELS-SIMBA CV boxes.  It used one mass-selected observer per box,
TNG-only normalization, a TNG-derived stellar-mass proxy for the `M_B<-15`
number density, and exact mass-preserving resampling of the public `256^3` CMD
dark-matter grids to the Hong `0.3125 Mpc/h` voxels.  Coordinate orientation,
file completeness, and box-mean preservation were verified before inference.

The frozen EDM model does not pass this test.  Total log-density power ratios
are `1.092/2.004` at `k=3--6/6--10 h/Mpc`; generated residual RMS is `1.172`
times truth.  Rank-histogram TV is `0.0925`, and finite-ensemble coverage is
`0.699/0.910` versus expected `0.600/0.838`, demonstrating overdispersion.
The density-PDF TV improves from deterministic `0.397` to `0.083`, and all
three 2pCF bands improve, but high-k variance is excessive and generated voids
fragment strongly.  This is not a single normalization error: the `3--6` band
is near the gate while `6--10` has twice the truth power.  Grid-HOP was skipped
by the preregistered rule, and forward PM/RAMSES is not entered.

SIMBA also exposes a real domain shift: occupied tracer cells average `607`
per cube versus `778` in TNG validation, occupied-cell radial velocity scatter
is `195` versus `430 km/s`, and target log-density scatter is `0.127` versus
`0.146`.  The next defensible model experiment must keep SIMBA CV 0--15 locked
and untouched, learn a multi-simulation residual distribution using other
realizations only, and then rerun this same gate.  Tuning a posteriori `A(k)`
or a residual scalar on these test truths remains disallowed.

### EAGLE gate completed after particle access

The earlier EAGLE-access blocker is obsolete.  The complete RefL0100N1504
snapshot 28 archive was validated and all `1504^3` dark-matter particles were
streamed directly from its 256 HDF5 tar members.  Following Hong et al., the
EAGLE targets use a stellar-mass rank cut matching the TNG100 `M_B<-15` number
density, not an EAGLE B-band conversion.  Exact `20 Mpc/h`, `64^3` cubes were
formed without changing the `0.3125 Mpc/h` voxel size; 172 observer cubes pass
the truth-blind exact-grid geometry cut.

The frozen TNG V6 EDM EAGLE gate completed on 16 truth-blind, spatially
dispersed observers.  Seven of eight field checks passed: the total-density
power ratios are `0.927/1.052` at `k=3--6/6--10 h/Mpc`, residual RMS ratio is
`1.004`, coverage and rank histogram pass, and PDF plus all 2pCF bands improve
over the deterministic field.  The conjunctive gate nevertheless fails one
marginal environment condition: the `rho>100` volume fraction is `0.001454`
versus truth `0.001220`, slightly worse in absolute error than deterministic
`0.001416`.  The preregistered threshold is not relaxed, so grid-HOP and
forward dynamics remain skipped.

### V7 audit and V8 development protocol

V7 already tested source-balanced TNG plus SIMBA development fine-tuning
without simulation labels.  Its step-500 checkpoint retained the TNG field
gate but failed locked SIMBA: total log-density power was `1.066/1.840` in the
`3--6/6--10 h/Mpc` bands and residual RMS was `1.139`.  A separate stationary
Fourier density-likelihood experiment also failed its TNG mock
self-consistency test, so it is removed from the IC path rather than being
retuned on independent truth.

V8 is the next predeclared experiment.  It augments the same EDM U-Net with a
global context MLP driven by eight rotation/mirror-invariant quantities that
exist for observations: occupied cells, tracer-count sum and occupied-cell
count moments, occupied radial-velocity moments, and deterministic mean-field
moments.  There is no simulation label and no target-density feature.  Feature
moments give equal weight to TNG train and SIMBA CV16--23 development train.
The context output layer is initialized to zero, making the step-zero model
exactly equal to the V7 parent.

Checkpoint selection is frozen before training completion.  Steps
500/2000/5000/10000 are screened on the established TNG representatives and a
truth-blind observable-feature selection from SIMBA CV24--26.  The score is the
worst absolute log deviation from unity over both high-k power ratios and the
residual-RMS ratio in both domains.  Only the selected checkpoint receives the
full 16-member, 40-step development gate.  It must pass all eight checks in
both domains before the historically inspected SIMBA CV0--15 stress test is
run.  CV0--15 is not independent for V8 because its V7 metrics motivated the
new architecture; it is never used to choose V8 weights or checkpoints.  A
stress-test pass permits the sealed EAGLE confirmation-32 test, which is the
new independent V8 evidence.  No fallback checkpoint is tried on EAGLE.

### V8--V11 development results (2026-08-05)

V8 completed 10,000 source-balanced updates and improved the SIMBA
development power and residual calibration substantially.  The unchanged full
field gate still failed: TNG was marginally low in the `6--10 h/Mpc` band
(`0.8943`, below `0.90`), while SIMBA overproduced `rho<0.1` volume and
underproduced `rho>100` volume and peaks.  Repeating the same diagnostic on
SIMBA training representatives gave the same tail errors, so this is model
underfit rather than a fortunate void-heavy validation selection.  Neither the
historical SIMBA stress set nor sealed EAGLE was opened.

V9 tested a preregistered train-only density-tail weighting.  It restored all
eight TNG checks but increased uncorrelated SIMBA small-scale variance: the
SIMBA `6--10 h/Mpc` power ratio rose from `1.221` at step 1000 to `1.428` at
step 5000, while coverage and environment checks failed.  Rare-voxel weighting
therefore changes amplitude without learning the required coherent structures
and is rejected; no independent data was opened.

V10 added a deterministic, zero-DC 3-D correction for the smooth component
omitted by the frozen mean, while retaining the V8 Laplacian stochastic
component.  Correction validation loss decreased monotonically through the
predeclared terminal step 5000.  All three full-fidelity candidates at steps
1000/3000/5000 nevertheless failed the development conjunction.  TNG remained
fixed near `0.896` in the highest Fourier band, as expected because the
stochastic parent was unchanged.  SIMBA remained over-voided (generated
`rho<0.1` volume about `0.446--0.447` versus truth `0.345`) and deficient in
`rho>100` peaks (about `59--61` versus truth `89.8`).  This shows that a better
conditional mean cannot calibrate draws from a residual distribution centered
on the old mean.

V11 is consequently a centered two-component test, frozen in
`config/hong2021_v11_recentered_edm.json`.  It fixes the V10 step-5000
correction, defines the stochastic target as the complete non-DC field
`truth-corrected_mean`, and retrains the observable-context EDM with uniform
voxel likelihood and equal source balance.  Unlike V8--V10, no Laplacian or
Fourier filter is applied to this new residual.  The train-only residual RMS is
`0.10721` in TNG and `0.09642` in SIMBA, giving an equal-source scale of
`0.10196`.  Cache generation, a four-step train smoke test, sampling, and all
51 Hong unit tests pass.  The 10,000-step Ada run completed and all three
predeclared candidates failed the dual-domain conjunction.  At step 10000,
TNG passes seven checks and misses only finite-ensemble coverage by `0.0041`.
SIMBA passes power (`0.957/0.969`), RMS (`0.957`), coverage, rank histogram,
PDF, 2pCF, and DC checks, but fails the peak/void aggregate: generated
`rho<0.1` volume is `0.456` versus truth `0.345`, and generated `rho>100`
peak count is `62.2` versus truth `89.8`.  No checkpoint is selected and no
historical SIMBA or EAGLE target is opened.

The V11 residual audit shows a train/validation-consistent asymmetric target
whose extreme positive tail is shortened by sampling.  SIMBA train and
validation residual RMS values are `0.09642/0.09625`, skewness values are
`0.758/0.753`, and 99.9-percent quantiles are `0.4194/0.4187`; the split is not
the cause.  V12 therefore applies a single train-only monotone Gaussianization
before EDM training and its frozen inverse after sampling.  The empirical CDF
uses exactly 50-percent TNG train and 50-percent SIMBA development-train weight,
131,072 bins, 8,193 knots, and latent support `[-5,5]`.  It uses no validation,
historical SIMBA, or EAGLE values.  Source latent skewness falls to `-0.064`
for TNG and `+0.098` for SIMBA; SIMBA train/validation latent standard
deviations are `0.9408/0.9395`.  Cache, train, inverse-sampling smoke tests, and
the 10,000-step run completed.  V12 still failed development: step 5000 passed
all TNG checks, but every SIMBA candidate failed the peak/void aggregate.  The
sealed EAGLE set remained unopened at this point.

### V13 train-only DC correction and independent result (2026-08-05)

The decisive V12 diagnostic was not another high-k amplitude error.  Every
stochastic member has exact zero cube DC, but the frozen conditional mean has
a stable positive truth-minus-mean DC of `0.00218` in TNG training and
`0.02598` in SIMBA development training.  This shifts the nonlinear density
thresholds even when centered power and residual calibration pass.  V13 froze
an eight-feature, target-free observable ridge correction before fitting.  It
used only TNG training and SIMBA CV16--23 training with equal source weight;
five-fold train-only CV with fold-local scaling selected `lambda=1e-6`.  The
frozen model SHA-256 is
`c743611fda5715222bd42db99e65359518f8c7b4e168453b55de52991577d5c3`.
It adds one predicted scalar to the conditional mean and every existing V12
member, so stochastic residuals, random numbers, coverage, and rank statistics
are unchanged.

V13 passed all eight unchanged checks in both development domains at the
predeclared step 5000.  In SIMBA development, generated `rho<0.1` volume moved
from `0.4572` to `0.3541` (truth `0.3452`), `rho>100` volume moved from
`0.000778` to `0.001045` (truth `0.001060`), and the high-k power ratios stayed
at `0.956/0.978`.  The historical SIMBA CV0--15 stress set then also passed
8/8, with high-k power `0.967/0.970`, residual RMS ratio `0.961`, and
`rho<0.1` volume `0.3449` versus truth `0.3552`.

Those passes authorized the one-time EAGLE confirmation-32 opening.  It failed
five of eight field checks: high-k power was `0.868/0.629`, residual RMS ratio
was `0.893`, 68-percent coverage was low by `0.0532`, and rank-histogram TV was
`0.0598`; the peak/void aggregate also failed.  Density-PDF and all 2pCF bands
still improved over the deterministic mean, and exact DC projection passed.
The failure is large-scale cross-code underdispersion, not a threshold-edge
decision.  Grid-HOP was skipped.  Per the sealed protocol, no EAGLE-driven
retuning, alternate checkpoint, seed, amplitude, or threshold is permitted.
Further development requires additional non-EAGLE training simulations and a
new untouched independent simulation family; the remaining EAGLE cubes are
not a replacement independent gate.

### V14 cross-code data firewall (frozen 2026-08-05)

Before reading any new simulation truth, the next-cycle roles were frozen in
`config/hong2021_v14_data_firewall.json`.  TNG100 and SIMBA remain development
domains.  Public CAMELS Swift-EAGLE CV0--19 becomes a new training domain and
CV20--26 becomes development validation.  It provides an EAGLE-like domain
shift without reopening the failed RefL0100N1504 target.  All 27 CAMELS Astrid
CV realizations are reserved for a one-time independent gate and were absent
locally at freeze time; only public HTTP file metadata was queried.

The preferred Magneticum alternative cannot currently be executed because the
official CAMELS access table marks it private.  The executable protocol thus
keeps Astrid completely unopened until a conditional multiscale location-scale
V14 model and every artifact hash are committed.  Astrid then uses one frozen
stellar-mass-only observer per realization, ensemble 16, 40 sampling steps,
seed 28777, and the unchanged eight-check field gate.  A failure is terminal
for V14 and grid-HOP is skipped.  See `NEW_INDEPENDENT_GATE.md`.

The original V14 firewall was superseded before downloading Swift-EAGLE or
opening Astrid.  The audit had missed a target-operator confound present since
V7: TNG and EAGLE used direct 0.3125-Mpc/h particle assignment, while SIMBA
used the public CMD adaptive 32-neighbour density followed by 256-to-80
resampling.  On already-development SIMBA CV16, raw NGP had 3.69% zero cells
and a half-particle floor produced 2.12 times the CMD high-k power; raw CIC had
no zero cells, log-field correlation 0.987 with CMD, and 1.08--1.12 power
ratios.  V14 firewall v2 therefore excludes CMD and freezes exact periodic,
cell-centred CIC from raw particles for every development suite and sealed
Astrid.  No pseudocount or post-hoc transfer is allowed.
