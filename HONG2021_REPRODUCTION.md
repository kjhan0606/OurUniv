# Hong et al. (2021) present-density reproduction

Reference: Hong, Jeong, Hwang & Kim, *ApJ* **913**, 76
([arXiv:2008.01738](https://arxiv.org/abs/2008.01738),
[DOI](https://doi.org/10.3847/1538-4357/abf040)).

## Current status

The published network can run on the local NVIDIA A10 hardware.  A full-width
TNG100 forward pass has been executed at `64^3`: the transcription has
461,024,955 parameters and returns the required `64^3` density grid.
The paper batch size of six also completes a full forward/backward/Adam step on
one local A10, peaking at about 8.62 GiB allocated and 12.22 GiB reserved.
Compute memory is therefore not a blocker.

The full TNG100-1 snapshot 99 (1.7469 TiB) and group catalog (4.18 GiB) were
downloaded to `/scratch/kjhan/IllustrisTNG/TNG100-1`.  On 2026-07-30,
`src/tng_validate.py` passed all 448+448 chunks: manifest byte sizes, HDF5
headers, required field shapes, and global object counts agree.  The subsequent
density pass read every one of the `1820^3 = 6,028,568,000` dark-matter
coordinates, checked finiteness and box bounds, and conserved the particle
count exactly.

`src/hong2021_prepare_tng.py` produced 432 training and 93 validation cubes in
`derived/hong2021` under that scratch tree.  A deep pass over every prepared
voxel found no non-finite values, non-integer counts, non-zero velocities in
empty cells, or targets outside `[-1,1]`.  The full 200-epoch paper-width run is
active in tmux session `hong2021_train`; its products are under
`training/tng100_v1`.

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

## Current run and next gate

The active baseline command uses the frozen 432/93 files, all 24 paper
augmentations, batch 6, Adam at `1e-3`, and 200 epochs.  Checkpoints are written
atomically once per epoch and best-model names are hard links, avoiding
duplicate multi-GiB writes.  A stopped run can continue with `--resume`.

After training, reproduce the paper's density-residual and 2pCF-KS table on
held-out TNG100.  EAGLE RefL0100N1504 at `z=0` remains the independent
simulation gate.  Only after those tests pass should the uncertainty-aware CF4
extension be interpreted; the present CF4 table still lacks the paper's LEDA
B-band magnitude and V_GSR fields.
