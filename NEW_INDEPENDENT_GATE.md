# V14 cross-code firewall and new independent gate

The one-time V13 EAGLE confirmation failed and remains a sealed historical
failure.  Its values cannot select another model, checkpoint, normalization,
random seed, amplitude, or threshold.  Additional RefL0100N1504 cubes are the
same simulation family and cannot become a fresh independent test.

The next cycle therefore assigns data roles before any new truth is read:

- TNG100 and the existing SIMBA partitions remain development data.
- CAMELS Swift-EAGLE CV0--19 is new training data and CV20--26 is new
  development validation data.  It supplies EAGLE-like subgrid physics through
  a different hydrodynamics implementation, without reusing the failed EAGLE
  target.
- All 27 CAMELS Astrid CV realizations are the new one-time independent gate.
  No Astrid file was present locally, downloaded, memory-mapped, or inspected
  when this role was frozen.  Only HTTP size and modification metadata were
  queried.

The authoritative machine-readable contract is
`config/hong2021_v14_data_firewall_v2.json`.  The original v1 contract was
superseded before any Swift-EAGLE or Astrid truth was read: it would have mixed
the adaptive 32-neighbour CAMELS Multifield Dataset target with the direct
particle-assignment TNG target.  A raw SIMBA CV16 development audit measured
that operator difference at roughly the ten-percent Fourier-gate scale.  V14
therefore uses the same raw-particle, cell-centred CIC operator for TNG100,
SIMBA, Swift-EAGLE, and Astrid; CMD grids are excluded.  In particular, Astrid uses one
stellar-mass-selected observer per independent CV realization, 16 ensemble
members, 40 sampling steps, and seed 28777.  It must pass the same eight field
checks as V13.  Grid-HOP runs only after all eight pass.  Failure is terminal
for this V14 claim: there is no alternate checkpoint, seed, subset, correction,
or relaxed gate.

## Why Swift-EAGLE development and Astrid testing

The current CAMELS documentation lists first-generation Swift-EAGLE and
Astrid as public, while Magneticum is private.  Swift-EAGLE's public z=0 raw
snapshots and corrected SUBFIND catalogues permit a target grid to be derived
locally.  Astrid's public raw snapshots and catalogues permit an exact
one-shot raw-CIC test after model freeze.  Astrid also changes both simulation code
and subgrid model relative to the TNG, SIMBA, and Swift-EAGLE development
mixture.

V14 development is not a post-hoc Fourier transfer correction.  The permitted
model family is a conditional multiscale location-scale likelihood learned
only from development truth, with target-free observable context at inference.
The exact architecture and artifact hashes must be committed before any
Astrid download.

Official data references:

- <https://camels.readthedocs.io/en/latest/data_access.html>
- <https://camels.readthedocs.io/en/latest/description.html>
- <https://camels.readthedocs.io/en/latest/codes.html>
- <https://camels-multifield-dataset.readthedocs.io/en/latest/access.html>
