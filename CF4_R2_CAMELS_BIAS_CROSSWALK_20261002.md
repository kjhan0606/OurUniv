# R2 CAMELS bias-prior compatibility — 2026-10-02

## Decision

External CAMELS bias evidence exists, but the six legacy numerical priors are
not calibrated for the active R2 bias parameters. Do not inject them
unchanged, reverse their order, or promote the old v8 holdout as current-R2
calibration. This is a source/provenance correction only: no likelihood,
prior, field, posterior, selection law or heldout result changed.

A separate literature-prior issue is now resolved: the six bias exponents in
[Lavaux & Jasche (2016), Table 1](https://academic.oup.com/mnras/article/455/3/3169/2892571),
are **not independent external calibration**.
Their ARES first step inferred luminosity-dependent bias from the 2M++ data;
the authors then held those estimates fixed in the BORG reconstruction of
that same catalogue. Table 1 fixes one observed subsample to bias 1, so the
other five values are relative estimates. Reusing them as an informative prior
while the same 2M++ counts enter this project's likelihood would reuse
catalogue information without carrying its dependence and can make the bias/
density constraint spuriously tight. The values are external to the CF4
distance marks, but not independent of the 2M++ count catalogue. They remain
useful as historical reference points, not independent prior information.
Exact row-by-row overlap between the paper input and the locally pinned
catalogue was not re-derived here; both are nevertheless the same named 2M++
survey catalogue.
There is also a model-step approximation: the paper derives a linear-regime
bias with ARES and carries it into BORG's power-law bias exponent using
`b ≃ alpha`; the authors explicitly note this equality is not exact and may
bias the reconstructed density.

This distinction applies to old six-observed-bin development code that reads
`published_prior` and multiplies its six bias centres by lognormal nuisance
factors (for example `scripts/cf4_r2_all_method_sampler.py`). That partial
sampler includes the same 2M++ count datum and explicitly records
`source_calibrated=False` and `R2_delivery=False`; its output is not an
independent validation result or a promoted posterior. A local heldout score
is prospective only if the bias estimate was built without using those heldout
rows; the paper's full-catalogue ARES estimate does not establish that
independence for a within-2M++ split. The current N256 five-true-K exact-GL2
path uses five unit-centred broad white-coordinate regularizers for the five
bias responses and does not inject the six Lavaux--Jasche values. Do not
confuse the historical six-bin pilot with that active target.

This bias distinction does **not** make every active tracer prior independent
of the 2M++ catalogue. The active target's additional LF-shape coordinates are
`Mstar=-23.28 + 0.2*u8` and
`alpha=-1 + 0.06*exp(0.5*u7)`, with standard-normal `u7,u8`; their reference
comes from Lavaux & Hudson (2011), Table 2, `|b|>10, K<11.5`. The log-rate
coordinate has scale 2 and is centered on the finite-reference-window LF
fraction evaluated at those same LF defaults. Thus the Mstar
regularizer is weak relative to that row's quoted 0.01-mag error, while alpha
is materially centered at -0.94, restricted to alpha>-1, and has local width
about 0.03 versus the row's quoted 0.02 error. The source paper estimates its
LF from a subset of the 2M++ catalogue, and the active target also uses 2M++
counts; exact row-level overlap was not recomputed. Treat these LF coordinates
as same-survey empirical regularization, not independent external calibration.
The paper's different CMB-frame `-25<M<-21`, `5000<cz<20000 km/s` row is
`alpha=-0.73, Mstar=-23.17`; its alpha value is about three white-coordinate
standard deviations from the active reference.

## What the source actually estimates

`scripts/cf4_camels_external_bias_calibration.py` forms six equal-number bins
from the training-realization quantiles of positive
`SubhaloMassType[:,4]`. Its population index increases with stellar mass. In
each CAMELS realization it fits the slope of log subhalo counts against log
dark-matter CIC density over occupied cells only, on an N=80 grid in a
25 cMpc/h box (0.3125 cMpc/h cells). Its own interpretation says this is not a
direct CF4 luminosity bias.

The legacy 2M++ calibration labels observations as
`population = appbin*3 + absbin` in
`scripts/cf4_joint_tracer_calibration_v1.py`. The apparent-K index distinguishes
the K<=11.5 and 11.5<K<=12.5 samples; within each, the three absolute-K edges
are ordered bright-to-faint. The v8 program injected CAMELS value `p` into
observed-bin index `p`. Thus the old six-to-six mapping paired the lowest
stellar-mass sextile with the brightest observed-K bin and the highest
stellar-mass sextile with the faintest observed-K bin. Reversing the index
would not repair this: apparent-K is a separate selection/distance axis, not
a luminosity ordering.

The active source model in `src/cf4_r2_marked_tracer_jax.py` has five latent
true-K intervals `(-inf,-25)`, `[-25,-23.6667)`, `[-23.6667,-22.3333)`,
`[-22.3333,-21)`, and `[-21,+inf)`. Its five positive bias responses are
normalized over the full source box and then mixed through the LF/selection
transfer into six observed populations. `src/cf4_r2_source_sky_joint.py`
labels these as development regularizers, not calibrated priors; the active
coordinates are `exp(0.5*white_tracer[1:6])` under the broad white-coordinate
Gaussian penalty. This five-bin target does not consume the six-vector
`external_bias_prior`; older six-observed-bin partial targets do, as noted
above, but are uncalibrated development artifacts.

The discrepancy is not only an ordering problem. CAMELS has stellar-mass
sextiles without a K-band luminosity or M*/L_K crosswalk; it estimates a
real-space, occupied-cell log-OLS response on 0.3125 cMpc/h cells. R2 has
five true-K responses, including two unbounded tails, normalized over the
periodic source field, then used in its own redshift-space observation law.
The old CAMELS calibration/v8 program used h=0.6711; the current fixed-field
N256 sensitivity path uses h=0.746. A follow-up driver source check resolved
the convention within the current path: `scripts/cf4_r2_common_catalogue.py`
converts physical absolute magnitude to `M_h=M-5 log10(h)` using
`src/cf4_actual_selection.py:magnitude_h`, while
`scripts/cf4_r2_marked_source_geometry.py` forms its modulus as
`5 log10(D_L*h)+25`; the common-selection integral also supplies `D_L*h` to
the same modulus calculation. The current label-generation and transfer
source code are internally consistent in the h-scaled convention; this source
audit did not independently verify the provenance/hash of the on-disk products.
The extant catalogue, geometry and common-selection result metadata each
record the active h=0.746 cosmology; the geometry/selection artifacts remain
explicitly `NOT_CALIBRATED`, and their result records do not bind the
generator-source commit. This supports the configured-convention statement,
not a full byte-to-source reproduction claim. By contrast, the old v8
classifier called the physical distance modulus directly and did not apply
`magnitude_h`. Thus identical numerical K edges in old v8 and current R2
refer to different selected rows. For the h-only conversion,
`5 log10(0.6711)=-0.866` mag: the old physical edges `[-25,-21]` map
nominally to `[-24.13,-20.13]` in the h-scaled convention, before small
cosmology-shape differences. More importantly, the literature
`Mstar=-23.28` is specified at H0=100, while old v8 evaluated that same number
against physical distance moduli at h=.6711. Its LF knee was therefore
nominally about0.87mag too faint relative to the data. This issue applies to
the shared legacy v1-v8 classifier/model source path; exact offsets depend on
each configured cosmology. The primary 2M++ catalogue paper states the H0=100
convention in §§2.2 and 2.6 and lists `Mstar=-23.28` in its K<11.5 LF row:
[Lavaux & Hudson (2011)](https://academic.oup.com/mnras/article/416/4/2840/975884).
This is a further reason not to transport the old holdout result; it does
**not** establish an h-convention bug in the current R2 path. The N256 source
spacing is 1.5 cMpc/h and the count deposition grid is N128/3 cMpc/h. Any
future crosswalk must preserve this explicit magnitude convention along with
population, scale, selection and estimator correspondence.

The separate legacy CAMELS FoG artifact also has six stellar-mass-bin widths;
the active R2 source model has one shared line-of-sight width. No validated
six-to-one reduction is recorded, so those values are likewise not an active
R2 calibration.

## Scope of the old v8 “holdout pass”

`config/cf4_joint_tracer_calibration_v8_decision_v1.json` records six positive
scores for the N32/384 cMpc/h development model and explicitly disallows a
production IC. The split in the source is an interleaved voxel-index holdout.
The v8 decision reports positive heldout increments relative to its
implemented null under that old model and index mapping; it does not establish
a physical CAMELS stellar-mass-to-CF4-K correspondence or calibrate the active
R2 five-bin parameters. In the source, the comparison null also sets both the
Carrick density covariate and radial nuisance to zero, and the fitted
positive-bias bound is strictly positive by construction. These details make
the old decision's broad “degeneracy removed” wording stronger than this
evidence alone supports. Preserve the historical decision artifact unchanged;
use this note as the later scope qualification.

## Independent review and driver disposition

Fable5 read-only audit: **CONDITIONAL PASS** for this provenance correction;
**REJECT** transfer of the legacy prior unchanged or merely reversed. It
answered Q-GOAL positively: a misassigned bias prior could distort the
inferred z=0 density and velocity/environment field. Q-LEAN recommends this
short source/provenance correction only; no new experiment, optimizer or
gate ladder now. The driver independently verified the population indices,
active true-K edges, prior application sites, old holdout construction, and
the fixed-field N256/N128 grid settings, and adopts that scope. Fable flagged
the h convention as unchecked and could not access the paper; the driver
subsequently verified the source convention from the primary article and
resolved both current-path consistency and the legacy mismatch above. This
review is advisory, not a posterior or promotion decision.

Fable5 read-only follow-up on Lavaux & Jasche (2016): **CONDITIONAL PASS** for
this documentation-only clarification; **NO** to treating the six Table 1
values as an independent prior or injecting them into the active five-bin
model. It verified the same-catalogue ARES-to-BORG sequence, the fiducial
normalization, the historical six-bin sampler use, and the active five-bin
unit-centred code path. Q-GOAL and Q-LEAN both pass for a concise provenance
correction; no code, configuration, target, fit, or calculation change is
needed. The driver independently checked the primary paper and source wiring.
Fable could not directly fetch the MNRAS page/PDF and exact paper-input-to-local
row overlap was not reproduced, so those points are not claimed as verified.

Fable5's 2026-10-02 read-only follow-up on the active LF-shape coordinates
returned **CONDITIONAL PASS**. The driver independently verified the exact
transforms, Gaussian-coordinate prior application, and same-catalogue LF
lineage against the primary 2M++ paper. The audit distinguishes the weak Mstar
regularizer from the more consequential, support-restricted alpha prior; it
does not invalidate the fixed-field diagnostics or require a job now. Before
any future sampling, revise alpha together with the finite faint-end law and
the radius-dependent selection/transfer model. Do not add an independent LF
likelihood from the same 2M++ catalogue. The Lavaux--Jasche bias issue remains
separate: those six exponents are absent from the active target, while the
Lavaux--Hudson LF shape enters it. Full reasoning and Q-GOAL/Q-LEAN disposition
are recorded in `CF4_MASTER_PLAN.md`.

## Next valid use, if needed

Do not fit a crosswalk until it is needed by the active likelihood. A valid
future mapping would require a calibrated conditional link from stellar mass
to K-band luminosity (including scatter) or direct K-band marks, projection
onto the active five true-K bins including both tails, the active resolution
and normalization convention, and matching selection/redshift-space
estimands. Train-only construction and an actually prospective, independent
CF4-disjoint prediction would then be needed before that calibration could
inform R2. This note authorizes none of those calculations.

Likewise, Lavaux--Jasche Table 1 values alone cannot fill the active R2
calibration requirement: they are same-catalogue plug-in estimates for a
different six-observed-bin law. A future reuse would need an explicit joint
likelihood or a training-only estimate with its dependence and prospective
validation handled; they must not be presented as independent external
calibration.

No legacy CAMELS or v8 result identifies MW, M31 or M33. Later role inference
must use LG observables on the same new z=0 field at zoom resolution, retain
the MW/M31 label ambiguity and allow M33 to remain unresolved. Native truth
identities may label calibration/evaluation only, never seed or select a
generated-field candidate. R2 remains NO-GO for posterior promotion at
1.5 cMpc/h; the eventual LG target remains <=0.3 cMpc/h.
