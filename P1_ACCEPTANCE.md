# P1 preregistered Local-Universe acceptance test

This file and `config/p1_targets_v1.json` freeze the P1 test before any of the
16 accepted parent constrained realizations are evolved to \(z=0\).

## Scope

P1 tests whether the coarse \(192^3\), \(L=384\,h^{-1}\mathrm{Mpc}\) parent
realizations put the required large-scale environments at their observed
positions. It does not select a Milky Way--M31--M33 system and it does not claim
precision halo masses at a particle spacing of \(2\,h^{-1}\mathrm{Mpc}\).

All members are evolved with the same PM solver and cosmology. The \(z=0\)
density is Gaussian-smoothed by \(4\,h^{-1}\mathrm{Mpc}\). Density percentiles
are evaluated against cells in a radial shell about the observer, which avoids
turning the strong radial selection function of CF4 into a structure score.

## Frozen hard gates

- Virgo: positive density at the M87 position, target-cell radial-shell
  percentile at least 70, and a peak within \(5\,h^{-1}\mathrm{Mpc}\) whose
  radial-shell percentile is at least 90.
- Coma: the same test at NGC 4874, with an \(8\,h^{-1}\mathrm{Mpc}\) search
  radius.
- Local Void: use the four distinct minima reported by Tully et al. (2019),
  rather than a single approximate sphere. At least three probes must have
  negative mean density within \(6\,h^{-1}\mathrm{Mpc}\), their joint mean must
  be negative, and their median radial-shell percentile must not exceed 35.
- Boötes void: use the B1950 centre and redshift from Kirshner et al. (1987).
  The smoothed centre must be at or below the 35th radial-shell percentile, and
  cumulative mean density must be negative at both 12 and
  \(24\,h^{-1}\mathrm{Mpc}\). The reported \(31\,h^{-1}\mathrm{Mpc}\) empty
  sphere is measured as an advisory profile point because a matter field need
  not reproduce a galaxy-empty sphere literally.

A member passes P1 only if all four gates pass. Thresholds will not be relaxed
after seeing the ensemble. If no member passes, the result is evidence that
the present velocity-only posterior ensemble is insufficient for the requested
named structures; the remedy is a revised observation model or explicit,
uncertainty-calibrated structure constraints, not post-hoc threshold tuning.

## Blind secondary cosmography

Every member also receives the same non-gating measurements at Fornax, Hydra,
Centaurus, Norma, Perseus, and the Shapley core. These anchors guard against
selecting a field that happens to match only the four requested locations.
They are reported but do not affect the v1 hard pass. Extended structures such
as Perseus--Pisces, the Great Attractor basin, and the Local Sheet require
topology and velocity-field comparisons; representing them as extra point
constraints would be physically misleading.

## Deferred measurements

Virgo/Coma \(M_{200c}\), Local-Group relative velocities, halo contamination,
and the final zoom masks are deferred to a higher-resolution run. Those runs
will use the user's compiled HOP implementation and direct spherical-overdensity
post-processing.

## Sources and coordinate conventions

- Tully et al. (2019), *Cosmicflows-3: Cosmography of the Local Void*,
  arXiv:1905.08329.
- Kirshner et al. (1987), *A Survey of the Boötes Void*, ApJ 314, 493,
  doi:10.1086/165080.
- Virgo is centred on M87 and Coma on NGC 4874. Equatorial positions are
  transformed to de Vaucouleurs supergalactic coordinates.

The exact machine-readable coordinates and thresholds are in
`config/p1_targets_v1.json`.
