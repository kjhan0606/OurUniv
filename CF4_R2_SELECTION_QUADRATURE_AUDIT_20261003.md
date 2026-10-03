# R2 3 cMpc selection-quadrature audit — 2026-10-03

Status: the pinned angular maps are available, but the current cellwise
selection integral is **NO-GO for a quantitative count likelihood**. No field
fit, posterior, heldout score, or simulation was run.

## Global agreement concealed cellwise error

The saved order-six full N128 integral and a bounded order-eight full-grid
recalculation have very similar *global* effective volumes: the maximum
relative difference over positive population/shell totals is `4.26e-4`
(0.043%), and the 95th percentile is `2.30e-4` (0.023%). This aggregate
agreement does not establish accurate per-voxel exposure.

A corrected training-only cell audit compared 22,457 training
population-cell keys (24,993 galaxies) against the saved order-six cube.
There are no zero-exposure training keys at either order, but the order-six to
order-eight relative exposure difference has a 95th percentile of 0.712% and
a maximum of 89.3%. The fixed-density Poisson log-intensity contribution
`sum(n_cell * log(E8/E6))` changes by `+2.965` nats, with absolute per-cell
contributions summing to `48.858` nats. This is only the `n log(lambda)`
selection contribution at fixed density, **not** a full Poisson score or fit.

Two training cells were selected by maximum relative and maximum absolute
order-six/order-eight discrepancy. The equal-node-budget split-layout check,
scrambled Sobol estimates, and map-pixel ray sums are:

| Training key | Count | GL6 | GL8 | 2x2x2 subcells, GL4 | Sobol 2^18 mean | HEALPix rays NSIDE 512 / 1024 / 2048 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3153974 | 1 | 0.1598683 | 0.1220155 | 0.1481018 | 0.1326226 | 0.1327339 / 0.1326210 / 0.1326252 |
| 5070912 | 2 | 0.0018879 | 0.0035743 | 0.0016122 | 0.0034212 | 0.0033372 / 0.0033882 / 0.0034077 |

The Sobol values use eight independently scrambled replicates; their reported
standard errors at `2^18` are `2.34e-5` and `3.37e-6`. The ray reference sums
equal-area HEALPix pixel-center directions, maps each ray to the pinned
NSIDE=512 completeness pixel, and integrates its radial selection over the
exact geometric ray/cell interval using a 200,001-point cumulative radial
table. For key 5070912, the NSIDE 512 ray estimate differs
2.45% from Sobol; NSIDE 1024 and 2048 differ 0.96% and 0.39%. The NSIDE 1024
to 2048 change is 0.57%, so this is a promising geometry-aware reference, not a
certified production integral. These two cells are not a whole-grid error
bound.

Relative to the Sobol `2^18` means, GL6 differs by about 20.5% and 44.8%, GL8
by about 8.0% and 4.5%, and the equal-work split GL4 layout by about 11.7% and
52.9%. Increasing tensor order alone or accepting GL8 would therefore be an
unsupported repair. The small global-volume delta reflects cancellation and
cannot certify local Poisson intensity.

## Heldout integrity and quarantine

The first order-eight full-grid diagnostic also calculated cellwise summaries
over `all_keys`, which included both training and holdout cell locations. Its
all-key local maxima/percentiles and support statistics are quarantined and
must not be used for model choice or evaluation. In a follow-up inspection I
loaded the archive's complete `holdout_counts` array to retrieve one selected
cell value (key `6645091`, count 1). No likelihood, fit, or predictive score
used those values, but the archive's holdout arm is no longer pristine and
must not be described as untouched. Future predictive validation needs a
fresh independent holdout after the observation/integration method is frozen.

The global effective-volume comparison is data-independent and remains
usable. Subsequent local checks were changed to read only `train_keys` and
`train_counts`; the final training-only ray/Sobol result is
`/gpfs/kjhan/CF4/z0_density/r2_selection_quadrature_20261003_v8/result.json`.
The quarantined first result remains preserved at
`/gpfs/kjhan/CF4/z0_density/r2_selection_quadrature_20261003_v2/result.json`.

## Decision and next R2 work

Do not promote order-six or order-eight cellwise exposures into the joint
count likelihood. The full angular map is present, but pixel-resolving
cell-volume integration is not established. Next design a map-aware exposure
integrator using the exact angular map values and radial cell intersections;
benchmark it on training-only cells and measure full-grid cost before any
full precomputation. A rough native-NSIDE512 estimate is 3.15 million rays
and about 88 voxel intervals per ray between 5 and 180 cMpc/h, or roughly
2.8e8 interval contributions before six radial tracer factors. NSIDE 2048
would use 50.3 million rays, so measure streaming cost before a full-grid run.
Do not smooth or nearest-pixel substitute the map, and do not run another
blind tensor-order ladder. The 442 pointwise map/catalogue
discrepancies, selection/bias calibration, group covariance and shared-latent
count/mark conditional remain separate blockers. No posterior is quantitative
or promotable; R2 remains NO-GO.

Q-GOAL: accurate selection exposure is necessary to connect 2M++ counts to the
same CF4-conditioned z=0 field; this audit supplies no field result.

Q-LEAN: one full-grid order-eight comparison was sufficient to falsify
cellwise stability; only two training cells received independent references.
No repeated global integral, model fit, heldout score, or gravity run was added.

MW/M31 remain role-ambiguous and M33 unresolved. Their observables must later
constrain those same roles in the NEW evolved field at LG resolution
`<=0.3 cMpc/h`; native truth IDs remain calibration/evaluation-only.
