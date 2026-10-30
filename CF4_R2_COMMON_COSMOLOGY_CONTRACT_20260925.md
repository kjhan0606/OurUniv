# R2 common-cosmology datum contract — 2026-09-25

## Why the earlier N128 connection was not a quantitative likelihood

The saved CF4 radial design and R1/PMWD dynamics use `h=0.746`,
`Omega_m=0.31`, whereas the original 2M++/ARES galaxy catalogue and shell
selection used `h=0.6711`, `Omega_m=0.3175`. This was acceptable for operator
wiring but not a single quantitative joint observation model. CF4 distances
and derived peculiar velocities are bound to the original distance-ladder
`H0`, so moving only CF4 points is not a valid correction.

Syntax Slurm404255 compared the same 72,973 source catalogue records at both
cosmologies without changing saved data. Of 57,249 original eligible rows,
11 become ineligible at the CF4/PM fiducial. Among 33,376 formerly used
counts, 8 are removed, 11 change one of the six populations and 329 move N128
cells. The maximum source-row radial change is only 0.062 cMpc/h because
the dominant `H0` factor cancels in comoving Mpc/h. Conversely, treating the
fixed CF4 physical-distance rows as if their `h` were 0.6711 would shift
their radial coordinates by roughly 1.6–16.3 cMpc/h. This is a coordinate
counterfactual only, not a recalculated CF4 velocity catalogue. Machine
result: `/gpfs/kjhan/CF4/z0_density/r2_cosmology_contract_audit_v1/result.json`.

## Chosen common basis and what is actually rebuilt

The R2 development basis is now the archived CF4/R1 cosmology in
`config/cf4_r2_common_cosmology_v1.json`: `h=0.746`, `Omega_m=0.31`,
`Omega_b=0.05`, `A_s=1.63e-9`, `n_s=0.96`, 384 cMpc/h box. This keeps the
already frozen CF4 radial datum and actual N128 PM state in their own basis.
It is a modelling choice, not a measurement of cosmological parameters.

Slurm404257 rebuilt the *source-catalogue-derived* N128 2M++ cells and
six-population/radial-shell survivor counts at that basis: 57,238 eligible
parent rows, 33,368 observed count rows (24,993 train, 8,375 held out), and
11,502 disjoint calibration marks. It preserved the original observer-
independent survivor and split marks by unique source `recno` and rejected
any newly eligible row lacking such a mark. Output:
`/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1/`.

The external mean-count and linear-regime bias references come from
[Lavaux & Jasche 2016, Table 1](https://arxiv.org/pdf/1509.05040): the paper's
population identifiers 3,4,5,0,1,2 map to this code's bright-first order.
The published mean counts per 2.34375-cMpc/h voxel were converted to a
3-cMpc/h voxel by volume ratio `(3/2.34375)^3` only. Those numbers are a
*source prior*, not a calibrated N128 galaxy rate: the CF4 crossmatch
exclusion, disjoint calibration sample, environment-dependent survival and
new PM response can change the effective rate and bias. No actual count
total was used to centre these prior numbers.

## Direct selection result and rate-prior diagnostic

The prior h=0.6711 N128 exposure must **not** be paired with these corrected
counts. Slurm404259 directly integrated the official angular maps and
Schechter luminosity fractions in six shells on the N128 grid at h=0.746,
using order-6 cell quadrature rather than interpolating or reweighting the
old map. All 29,100 observed population/cells have positive exposure. The
radial luminosity lookup error was `1.83e-10` (absolute); the largest
effective-volume change relative to the old N128 map was 0.0235% across
nonzero population/shell bins. This comparison is a sanity check, **not** an
angular quadrature or selection-calibration certificate. Machine result:
`/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1/`.

Using the Table-1 rates scaled only by cell volume, Beta posterior-mean
survival in each bin, this integrated selection, and the predetermined 80%
non-calibration count fraction, a *homogeneous-field* reference predicts
28,574 selected galaxies against 33,368 observed. In local population order,
observed/reference ratios are `[1.595, 1.080, 1.072, 1.766, 0.983, 0.847]`.
This is a **prior-transfer diagnostic**, not a goodness-of-fit test: the
actual density field, cosmic variance, environment dependence of exclusions,
and uncertain rates are omitted. The two bright-bin excesses especially
forbid treating the published rates as exact N128 constants or silently
normalizing the forward model to observed totals.

Slurm404269 applied the recomputed counts/selection and those source priors to
the existing *unconditional* N128 PM state with zero diffuse fraction. The
29,100 occupied cells all retained positive finite intensity (minimum
`2.79e-6`). Predicted population totals were `[4414, 7805, 1894, 4193,
13082, 4676]`, versus observed `[5895, 6337, 1455, 6266, 10599, 2816]`.
The sky phase is random; these residuals are **not** a CF4 fit, bias estimate,
posterior predictive check or failure of the common-cosmology mapping. Output:
`/gpfs/kjhan/CF4/z0_density/r2_common_pm_support_v1/result.json`.

## Promotion limit and next action

Even after numerical support succeeds, the mask/LF integral and Beta
survival marks alone do not
calibrate six nonlinear PM biases, RSD/FoG, environment-dependent exclusion,
or the field-model discrepancy. A random-phase sky is not a likelihood fit.
The first physical R2 delivery remains CF4+2M++ conditioned z=0 density and
velocity posterior samples from latent ICs; neither this datum, source rates,
nor selection calculation alone delivers them.

For later LG inference, MW/M31/M33 must be identified as uncertain roles
from each generated present state. M33 may remain unresolved; native mock
truth identities may label calibration/evaluation but cannot seed generated
candidate selection. Their observations must constrain the same evolved
state. Q-GOAL: one coordinate/cosmology contract is required before a valid
CF4+galaxy likelihood. Q-LEAN: source rows and the existing PM basis were
reused; no TNG download, new RAMSES simulation, filesystem diagnostic, or
additional gate framework was introduced.
