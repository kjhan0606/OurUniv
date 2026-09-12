# EAGLE independent gate

EAGLE RefL0100N1504 at snapshot 28 is a locked cross-simulation test.  It is
not part of training, input normalization, hyperparameter selection, or gate
threshold tuning.  Hong et al. trained on TNG100 and applied that fixed model
to EAGLE.

## Frozen galaxy selection

Hong et al. did **not** use EAGLE `B`-band magnitudes for target selection.
They matched the number density of the TNG100 `M_B<-15` sample with an EAGLE
stellar-mass cut.  The local TNG preparation has 48,296 targets in a
`(75 Mpc/h)^3` box, giving `0.1144794074 (h/Mpc)^3`.  The EAGLE box therefore
contains 35,632 rank-selected target galaxies.  The catalogue query uses the
recommended 30-pkpc aperture stellar mass and freezes ties by `GalaxyID`.

The center definition is a central galaxy (`SubGroupNumber=0`) with
`4e10<Mstar/Msun<1e11`.  The paper reports 478 EAGLE centers.  The current
public catalogue count is recorded without moving either mass boundary to
force agreement.

## Exact cube geometry

The snapshot stores a `67.77 Mpc/h` periodic box, while the required voxel is
`0.3125 Mpc/h`; their ratio is 216.864.  A periodic `217^3` grid would silently
change the voxel size.  Version 1 instead constructs exact 0.3125-Mpc/h cells
on `[0,67.5)^3` and uses only observer cubes fully contained in that regular
region.  The selection uses positions and fixed geometry only, before the
dark-matter field is inspected.  Every emitted cube is exactly `20 Mpc/h` and
`64^3`.

## Commands

```bash
bash scripts/query_eagle_hong_catalog.sh
python src/hong2021_prepare_eagle.py --audit-only
python src/hong2021_prepare_eagle.py
```

The preprocessor reads HDF5 files directly inside the uncompressed tar and
does not create a second 470-GiB extracted copy.  Products are written under
`/gpfs/kjhan/EAGLE/RefL0100N1504/derived/hong2021_v1/`.

After preparation, the deterministic TNG checkpoint and V6 EDM checkpoint are
run with their existing TNG normalization.  The same preregistered field gate
is evaluated first.  Grid-HOP and forward dynamics remain forbidden if that
field gate fails.

## Completed result (2026-08-05)

The particle pass read all 3,402,072,064 DM particles and produced 172 exact
geometry-safe test cubes.  A truth-blind farthest-point selection fixed 16
representative cubes before inference.  The frozen V6 EDM passed seven of the
eight field checks.  It passed both high-k power bands (ratios 0.927 and
1.052), residual RMS (ratio 1.004), finite-ensemble coverage, rank-histogram,
density-PDF, 2pCF, and DC-projection checks.

The conjunctive field gate failed only because the `rho>100` volume fraction
was marginally farther from truth than the deterministic mean: truth
0.001220, deterministic 0.001416, and EDM 0.001454.  The other selected
peak/void statistics improved, including the `rho>100` local-peak count.  The
threshold is not changed after inspecting EAGLE, and grid-HOP was therefore
skipped.

## Post-gate seal

Before any V8 development, the prepared EAGLE file, the 16-object selection,
the V6 ensemble and metrics, its decision, and the V6 checkpoint were recorded
with SHA-256 hashes in
`config/hong2021_eagle_confirmation_seal_v1.json`.  The already inspected 16
cubes are permanently excluded from future model selection.  The other 156
cubes cannot be used for training, normalization, feature fitting,
architecture or checkpoint selection, or gate tuning.

A 32-object confirmation subset was selected from those 156 using observer
positions and GalaxyIDs only, before V8 results existed.  Its immutable object
list and inference settings are in
`config/hong2021_eagle_confirmation32_v1.json`.  It may be opened exactly once,
and only after the TNG/SIMBA development gates and the historically inspected
SIMBA CV0--15 stress test pass.  CV0--15 is not described as an independent V8
test because its V7 results were already known.  Failure is not followed by
EAGLE-driven iteration.

## Confirmation-32 outcome (2026-08-05)

V13 was the first post-seal method to pass both development domains and the
historical SIMBA stress prerequisite.  It therefore opened confirmation-32
once, using the frozen V12 step-5000 checkpoint, ensemble size 16, 40 EDM
steps, seed 27777, and the train-only observable DC model recorded in
`config/hong2021_v13_dc_fit_record.json`.

The independent field gate failed.  Generated/truth total log-density power is
`0.868` at `k=3--6 h/Mpc` and `0.629` at `k=6--10 h/Mpc`; residual RMS is
`0.893` of truth, 68-percent finite-ensemble coverage is low by `0.0532`, and
rank-histogram TV is `0.0598`.  The peak/void aggregate also fails, although
the density PDF, every 2pCF band, and exact DC projection pass.  Grid-HOP was
not entered.  This confirmation is permanently classified as failed and may
not be followed by EAGLE-based model, checkpoint, amplitude, seed, or gate
tuning.
