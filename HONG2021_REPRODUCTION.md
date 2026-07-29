# Hong et al. (2021) present-density reproduction

Reference: Hong, Jeong, Hwang & Kim, *ApJ* **913**, 76
([arXiv:2008.01738](https://arxiv.org/abs/2008.01738),
[DOI](https://doi.org/10.3847/1538-4357/abf040)).

## Result of the first audit

The published network can run on the local NVIDIA A10 hardware.  A full-width
TNG100 forward pass has been executed at `64^3`: the transcription has
461,024,955 parameters and returns the required `64^3` density grid.
The paper batch size of six also completes a full forward/backward/Adam step on
one local A10, peaking at about 8.62 GiB allocated and 12.22 GiB reserved.
Compute memory is therefore not the blocker.

A scientifically faithful training cannot start yet because the server does
not contain TNG100-1 snapshot 99 or its group catalog.  The apparent
`/scratch/jaehyun/TNG100` directory is only a small collection of logs and
parameter files.  The 1.5-TB local TNG300 directory contains merger trees,
offsets, hydrogen post-processing, and an IDL galaxy catalog, but no snapshot
particles.  Particle coordinates are indispensable: the target is dark-matter
density, not a halo- or galaxy-smoothed proxy.

The official full TNG100-1 snapshot is about 1.7 TB.  A field-restricted
download only needs the dark-matter coordinates for the target plus the
snapshot-99 group catalog for galaxy position, velocity, stellar mass, and
photometry; access still requires an IllustrisTNG API key.  No key is present
in the current environment.

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

## Next executable gate

Obtain authenticated access to:

- TNG100-1 snapshot-99 dark-matter `Coordinates`;
- TNG100-1 snapshot-99 group catalog fields needed for centers and target
  galaxies;
- preferably EAGLE RefL0100N1504 at `z=0` for the independent test.

After these arrive, prepare exactly 432/93 non-overlapping cubes, train the
frozen baseline, reproduce the paper's density-residual and 2pCF-KS table, and
only then run the uncertainty-aware CF4 extension.
