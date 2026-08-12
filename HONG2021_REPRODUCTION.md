# Hong et al. (2021) present-density reproduction

Reference: Hong, Jeong, Hwang & Kim, *ApJ* **913**, 76
([arXiv:2008.01738](https://arxiv.org/abs/2008.01738),
[DOI](https://doi.org/10.3847/1538-4357/abf040)).

The post-v1 diagnosis and staged restart are frozen in
[`HONG2021_RESTART_PLAN.md`](HONG2021_RESTART_PLAN.md).

## Current status

The latest completed model path is the sealed V72 conditioning-stratified
spatial quantile transport.  Its single fresh stage-A screen failed on the
TNG100 physical tail, Q4, sub-Mpc two-point comparison, and maximum-energy
ordering; stage B, Astrid, and independent EAGLE remained unopened.  The V72
verdict and artifact hashes are bound in
[`config/hong2021_v72_result_record.json`](config/hong2021_v72_result_record.json).

Before spending another fresh partition or training another generator, V73
measured the attainability of the unchanged sixteen-query gate using 10,000
cluster-aware fit-train truth-oracle trials per domain.  The all-domain joint
pass probability is only `0.7633`, below the preregistered `0.8` floor.  TNG100
is the bottleneck: the q99.999, Q4, and all-scale two-point pass probabilities
are `0.896`, `0.868`, and `0.8559`, while high-k power (`0.9996`) and residual
RMS (`1.0`) are stable.  The current gate therefore has a measured `0.2367`
truth-oracle false-rejection probability and requires prospective
null-calibrated redesign.  This finding does not rescue or reopen V72.

The domain-blind stress also fails four of six ordered source-target pairs,
all involving TNG100, whereas SIMBA and Swift transfer in both directions with
absolute-core probabilities `0.85` and `0.8784`.  TNG within-box spatial
heterogeneity does not cross the frozen materiality limits.  In the already
consumed V72 maximum-energy comparison, TNG100 is decisively worse than the
independent control (paired mean `+0.07425`, 95% interval
`[+0.04280,+0.10623]`); SIMBA and Swift remain underpowered.  Complete results
are in
[`config/hong2021_v73_result_record.json`](config/hong2021_v73_result_record.json).
No new Hong candidate or gate change is authorized pending explicit approval.

V74 then prospectively audited the two V73-identified redesigns without
reading raw truth again.  It independently reproduced the sixteen-query joint
pass probability (`0.7661` versus V73 `0.7633`) and found that 32 grouped
queries are the minimum tested count that makes the unchanged physical and
morphology conjunction attainable.  At 32 queries the all-domain joint pass
probability is `0.91975` with 95% Wilson interval
`[0.91590,0.92344]`; TNG100 absolute-core probability is `0.95075`.

The strict maximum-energy winner was replaced prospectively by a
truth-oracle-calibrated, three-domain Bonferroni no-detectable-inferiority
rule.  Its independent 32-query family-wise false-rejection probability is
`0.04735`, with Wilson upper bound `0.05038`.  The consumed V72 TNG100 result
still fails the prospective margin (`+0.07425` versus `+0.03962`), so V72 is
not rescued.  V74 is recorded in
[`config/hong2021_v74_result_record.json`](config/hong2021_v74_result_record.json).
The complete prospective gate remains unfinished: rank histogram and voxel
coverage need a separate exact-conditional-null audit before any new Hong
candidate or fresh gate is authorized.

V75 completed that audit.  It showed that the old fixed rank-TV and coverage
cutoffs are invalid: their null behavior changes drastically with within-field
spatial dependence, and the old finite-ensemble coverage expectation is not
distribution-free.  The replacement pools the sixteen generated residual
fields and truth field per query and randomizes which of the seventeen is
designated truth.  This exact field-label null preserves spatial dependence,
the marginal distribution, quantile interpolation, and ties.  Across four
exchangeable synthetic scenarios its preregistered domain test rejected
`1.22--1.686%` at nominal `1/60`, and it had `0.87` and `1.0` power against
the two frozen alternatives.

The V75 scalar combination itself is **not promoted** to the complete gate.
Post-result logical inspection found that
`max(rank_tv/0.05, coverage_deviation/0.03)` can let the wider null component
hide an extreme conditional p-value in the other component.  The decisive
already-consumed example is V72 candidate TNG100: rank has conditional
`p=0.00001`, coverage has `p=0.97439`, yet the scalar composite also has
`p=0.97439` and passes.  V76 must therefore retain the exact label null but
protect rank and coverage separately over three domains at per-test
`alpha=1/120`.  No complete gate, new candidate, or fresh partition is
authorized yet.  The immutable V75 result and this correction boundary are
recorded in
[`config/hong2021_v75_result_record.json`](config/hong2021_v75_result_record.json).

V76 prospectively replaced only that masking-prone combination.  Rank and
coverage now receive separate exact-label conditional p-values in each of
TNG100, SIMBA, and Swift; all six must be strictly greater than `1/120`.
Bonferroni therefore bounds family-wise type-I error by `0.05` without assuming
independence.  Reusing the immutable V75 null arrays, individual false-rejection
rates were `0.547--0.887%`, and the three-domain family diagnostic was
`3.62--4.88%` over the four spatial-dependence and tie scenarios.  All frozen
calibration limits passed.

The independent V76 power draw detected location bias in `86/100` replications
(95% Wilson lower `0.7786`) and underdispersion in `100/100` (lower `0.9630`).
The deterministic nonmasking audit passed all 36 p-value pairs and detects
rank-only and coverage-only failures.  The separate exact-label rule is thus
selected.  V72 remains failed and sealed; no V72 ensemble, fresh partition,
checkpoint, or model sample was accessed.  The result is bound in
[`config/hong2021_v76_result_record.json`](config/hong2021_v76_result_record.json).
Only a complete prospective 32-query gate specification may now be frozen;
executing it or constructing a new candidate still requires explicit approval.

V77 found that such a complete gate still cannot be frozen.  V74's 32-query
attainability result used four/eight/twenty fit-train groups for
TNG100/SIMBA/Swift, whereas the untouched validation pool has four/three/seven
groups and cannot execute the same quotas.  A prospectively frozen compatible
design reserved 32 metadata-only indices per domain after excluding all
historical development and consumed V72 stage-A indices; no input or target
voxels were read.

On 20,000 independent compatible-design verification trials, the unchanged
physical/morphology conjunction passed all three domains only `0.7311` of the
time (95% Wilson lower `0.72491`), below the `0.8` requirement.  SIMBA is the
main named bottleneck: its strict all-scale two-point improvement passes
`0.82075`, and its full domain joint passes `0.7974`.  The energy rule remains
well calibrated (`0.04835` family false rejection; Wilson upper `0.05141`).
After adding the V76 rank/coverage error budget, the conservative complete-gate
pass lower bound is only `0.6235`.  Even perfect morphology would not suffice,
because the all-domain absolute-core Wilson lower is `0.86045`.

The result is therefore a sealed gate-design failure, not a model failure.
No complete gate or new candidate is authorized.  A next attempt must replace
the collection of sampling-sensitive hard/conjunctive checks with a separately
frozen conditional-null design and one global error budget; changing only the
query count, energy rule, or two-point row cannot meet the target.  Full hashes
and the untouched index reservation are in
[`config/hong2021_v77_result_record.json`](config/hong2021_v77_result_record.json).

The published network can run on the local NVIDIA A10 hardware.  A full-width
TNG100 forward pass has been executed at `64^3`: the transcription has
461,024,955 parameters and returns the required `64^3` density grid.
The paper batch size of six also completes a full forward/backward/Adam step on
one local A10, peaking at about 8.62 GiB allocated and 12.22 GiB reserved.
Compute memory is therefore not a blocker.

The full TNG100-1 snapshot 99 (1.7469 TiB) and group catalog (4.18 GiB) were
downloaded to `/scratch/kjhan/IllustrisTNG/TNG100-1` on `syntax`.  On 2026-07-30,
`src/tng_validate.py` passed all 448+448 chunks: manifest byte sizes, HDF5
headers, required field shapes, and global object counts agree.  The subsequent
density pass read every one of the `1820^3 = 6,028,568,000` dark-matter
coordinates, checked finiteness and box bounds, and conserved the particle
count exactly.

`src/hong2021_prepare_tng.py` produced 432 training and 93 validation cubes in
`derived/hong2021` under that scratch tree.  A deep pass over every prepared
voxel found no non-finite values, non-integer counts, non-zero velocities in
empty cells, or targets outside `[-1,1]`.  The full 200-epoch paper-width run is
complete.  It ran through epoch 97 on the `syntax` A10 and resumed without lost
epochs on the `LagEunha` RTX 5000 Ada.  The portable training products and
prepared cubes are under `/gpfs/kjhan/IllustrisTNG/TNG100-1`.

The first held-out evaluation is also complete.  It uses the 93 unaugmented
spatially held-out cubes and compares the unique minimum-validation-loss
(epoch 9) and minimum-training-loss/last (epoch 200) checkpoints.  In agreement
with the paper's selection procedure, the minimum-training-loss checkpoint has
the lower mean 2pCF KS statistic.  It does **not**, however, reproduce the
published TNG100 2pCF statistics:

| Metric | This run, epoch 200 | Hong et al. TNG100 |
| --- | ---: | ---: |
| `log10(rho_pred/rho_truth)` | `+0.028 +/- 0.553` | `-0.014 +/- 0.543` |
| 2pCF KS, `0-1 Mpc/h` | `0.781 +/- 0.189` | `0.263 +/- 0.035` |
| 2pCF KS, `1-3 Mpc/h` | `0.590 +/- 0.112` | `0.175 +/- 0.087` |
| 2pCF KS, `3-10 Mpc/h` | `0.489 +/- 0.080` | `0.130 +/- 0.042` |

The voxel residual width is close to the paper, but rare predicted cells hit
the `tanh` upper limit and dominate the linear-density 2pCF: the median
zero-lag correlation is about `3.87e4` in the epoch-200 prediction versus
`1.50e3` in truth.  The raw calculation used
`delta=rho/rho_cosmic_mean-1`; over the finite validation ensemble the mean
density ratios are `1.921` (truth) and `3.024` (prediction), so the statistic
mixes spatial correlation with a large one-point normalization offset.  Using
each field's validation-ensemble mean reduces the periodic mean KS over
`0-10 Mpc/h` from `0.512` to `0.345`, confirming that normalization is a major
part of the discrepancy.  Per-cube mean normalization and `xi/xi(0)` do not
give the same correction, and the paper does not publish which finite-cube
convention generated Table 2.  As a diagnostic only, clipping prediction
values at the truth `99.999` percentile reduces the raw periodic mean KS to
`0.258`; post-hoc clipping is not accepted as a scientific correction.

The correct conclusion is therefore that the 2pCF gate is **inconclusive until
its normalization is frozen**, not that the trained network has already failed
the TNG gate.  CF4 inference remains disabled until that ambiguity is resolved.

The subsequent BatchNorm audit separates checkpoint behavior from weight
learning.  Epoch 9 reproduces its logged training and validation losses with
the saved inference statistics and has no cube above MSE `0.05`.  At epoch 200,
mini-batch statistics reproduce the logged training loss (`0.00613` versus
`0.00615`), but saved inference statistics give `0.02262` and 39/432 training
outliers.  Its inference validation MSE still reproduces the logged value
(`0.01514`), so this is not a damaged checkpoint.  It is a late-epoch
train/inference mismatch exposed by the heterogeneous training distribution;
one-pass cumulative recalibration only improves the training MSE to `0.01445`
and leaves 17 outliers.  The machine-readable audits are in
`evaluation/tng100_v1/bn_audit/epoch009.json` and `epoch200.json`.

Paper-matched 5-Mpc/h projections of linear density were also generated for
both checkpoints.  The network recovers the principal nodes and filaments but
smooths fine branches and high-k texture, qualitatively like the published
TNG prediction.  This withdraws the earlier claim that the entire model failed
to learn; normalization instability and conditional-mean smoothing are two
different limitations.

The split audit has also been resolved for the v2 pilot.  Metadata for all 988
observer candidates was built from the group catalog and the already validated
`240^3` dark-matter grid; no particle reread was needed.  The old split has a
maximum absolute SMD of `1.024` across six frozen observer/input/target
features.  Three jointly optimized 432/93 splits reduce this to `0.086`,
`0.089`, and `0.091`, with maximum one-dimensional KS distances of `0.148`,
`0.158`, and `0.154`.  Every split retains a minimum cross-split L-infinity
separation greater than `20 Mpc/h`.  The exact candidate indices and subhalo
IDs are frozen in
`derived/hong2021_v2/joint_balanced_splits_v1.json`.

The balanced-split pilot completed at 20 epochs.  BN auditing accepts the
minimum-validation checkpoint at epoch 7 and rejects epoch 20.  The accepted
checkpoint recovers the main nodes and filaments in the paper-style 5-Mpc/h
projection, but removes fine branches and adds compact peak artifacts.  Its
raw cosmic-mean 2pCF KS values are `0.685`, `0.453`, and `0.348` over
`0--1`, `1--3`, and `3--10 Mpc/h`, an improvement over v1 but not a
reproduction of the published table.  Separate ensemble-mean normalization
improves the latter two ranges, but hides a factor-`1.895` prediction/truth
mean-density bias and is therefore diagnostic only.

The Fourier result rules out interpreting voxel size as information
resolution.  In log density, transfer and phase correlation fall to
`(T,r)=(0.393,0.358)` at `k=3--6 h/Mpc` and `(0.259,0.176)` at
`k=6--10 h/Mpc`.  In linear density the model has excess rather than missing
high-k power, but that power has poor phase correlation and is carried by
compact peaks rather than the missing filamentary texture.  The complete
machine-readable result and plots are under
`evaluation/tng100_v2_split00_l0_bn/epoch007/spectral_v2`.

The same spectral estimator applied to the v1 epoch-9 checkpoint confirms that
the balanced split mainly improves the 2pCF distribution, not information
resolution.  V1 and v2 log-density phase correlations are `0.356` and `0.358`
at `k=3--6 h/Mpc`, then `0.190` and `0.176` at `k=6--10 h/Mpc`.
Meanwhile the v2 linear-density high-k excess is larger.  This comparison is
stored in `evaluation/tng100_v1/spectral_v2_minimum_validation`.

Two reproduction details not specified sufficiently by the paper are frozen
and recorded in every output:

- The reported 988 observer candidates are reproduced only by applying
  `4e10 < SubhaloMassType[:,4]*1e10 < 1e11` without an additional division by
  `h`.  Dividing by `h`, as the public catalog unit label suggests, gives 1,552
  instead.
- The paper does not publish its 525 subhalo IDs or split seed.  We select a
  deterministic spatial split with 93 clustered validation observers and 432
  training observers, requiring no periodic 20-Mpc/h cube overlap across
  splits.  The minimum cross-split L-infinity separation is 20.0377 Mpc/h and
  the exact IDs are stored in the HDF5 files.

## Frozen paper specification

- TNG100-1, snapshot 99, `(75 Mpc/h)^3`, `1820^3` dark-matter particles.
- Center galaxies: `4e10 < Mstar/Msun < 1e11`; 988 candidates reported.
- Each sample: `20 Mpc/h`, `64^3`, hence `0.3125 Mpc/h` per voxel.
- Target galaxies: absolute `B` magnitude `M_B < -15`.
- Input: galaxy count and mean radial peculiar velocity relative to the center.
- Mask: `|b| < 10 deg`.
- Target: `log10(rho_dm/rho_dm_mean)/4.5`.
- Non-overlapping samples: 432 train and 93 validation.
- Augmentation: three cyclic axis permutations times eight flips, producing
  10,368 train and 2,232 validation samples.
- Five reflective-padding, `5^3`, stride-2 encoder convolutions with
  128, 256, 512, 1024, and 2048 channels.
- Nearest-neighbor upsampling, encoder skip concatenation, batch normalization,
  reflective-padding `3^3` convolutions, and a final `tanh`.
- MSE, Adam `(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7)`, batch 6, 200 epochs.
- The paper ultimately selected the minimum-training-loss checkpoint after
  comparing the validation two-point-correlation distributions.

The source is implemented in `src/hong2021_model.py`,
`src/hong2021_data.py`, and `src/hong2021_train.py`.  The frozen parameters are
in `config/hong2021_tng100_v1.json`.

## Velocity uncertainty extension for CF4

The paper has two input channels in total: `Ngal` and mean `Vpec`.  It propagates
distance uncertainty by drawing 1,000 distance-modulus realizations, not by
supplying a velocity-error channel to the CNN.

For CF4 we will compare that baseline against:

1. `Ngal`, inverse-variance-weighted mean `Vpec`, and the propagated error of
   that mean;
2. the same three channels plus within-voxel velocity dispersion.

The measurement-error channel and the sample-dispersion channel must not be
called the same `sigma_v`.  At `0.3125 Mpc/h`, most occupied cells contain only
one observed galaxy, so within-cell sample dispersion is usually zero.  The
propagated measurement error remains informative.  Definitions and acceptance
tests are frozen in `config/hong2021_cf4_uncertainty_v1.json`.

A CF4 diagnostic using the 40-Mpc/h cube and `|b|>=10 deg` confirms this:
2,750 galaxies occupy 2,563 of the `128^3` voxels, and 95.9% of occupied voxels
contain exactly one galaxy.  Only 4.1% can define a within-cell sample
dispersion.  In contrast, the propagated velocity-error median is about
267 km/s (5--95% range about 7--621 km/s), so the observational-error channel
contains useful information.  These numbers are diagnostic only because the
current table has neither the paper's B-magnitude cut nor its V_GSR frame.

## What “Hong resolution” means

`0.3125 Mpc/h` is the output voxel, not uniform information resolution.  Hong
et al. measured distance-dependent covariance scales of approximately 0.26,
0.68, 0.92, 1.06, and 1.18 Mpc/h in successive radial shells.  Success requires
reproducing these covariance/2pCF tests on held-out TNG and independent EAGLE,
not merely writing a `64^3` array.

## CF4-specific input gap

The local `cf4_galaxies.csv` has distance moduli and their errors, so the
velocity-error grid can be constructed.  It does not contain the two fields the
paper used to define the observational sample: LEDA absolute B magnitude and
velocity in the Galactic Standard of Rest.  It contains `Vcmb` only.  These
must be added by a documented external cross-match/frame conversion before a
CF4 density map can be labeled a Hong-method application.

## Completed run and next gate

The completed baseline used the frozen 432/93 files, all 24 paper augmentations,
batch 6, Adam at `1e-3`, and 200 epochs.  Checkpoints were written atomically
once per epoch and best-model names are hard links.  Evaluation products are in
`/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_v1`; `metrics.json` records
the estimator convention, both checkpoint results, and model selection.

Before any CF4 application, first freeze the exact density-contrast and
finite-volume normalization used for the 2pCF.  The next pilot must use matched,
cross-split-nonoverlapping observer samples and reject any checkpoint whose
inference-mode training loss differs from its logged training loss by more than
10%.  The literal target selection remains `M_B<-15`; `SubhaloFlag`, star
particle count, and stellar-mass cuts are separately labeled ablations, not
silent changes to Hong et al.  A model assessed under the corrected statistic
must reproduce the held-out TNG100 table before testing EAGLE RefL0100N1504 at
`z=0`.  Only after both gates pass should the uncertainty-aware CF4 extension
be interpreted; the present CF4 table still lacks the paper's LEDA B-band
magnitude and V_GSR fields.
