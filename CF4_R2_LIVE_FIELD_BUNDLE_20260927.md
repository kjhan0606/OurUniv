# R2/5 — live-field/source-mark coupling, with native PM grid coordinates

The next outcome is a working joint IC/field/calibration transition, not
another fixed-field nuisance fit. Full R2 production remains unapproved by
the current model evidence. Use only the source-bound10,020 FP/114 non-FP
distance marks conditional on the existing group/member redshifts. Do NOT
multiply the old BGc factor or inclusive galaxy counts into this pilot.

## Coordinate finding and targeted correction

Source inspection found that `particle_grid` calls PMWD scatter with default
offset0, hence its mesh samples sit at i*dx. The recent group/source controls
used `read_centred` with default origin_fraction0.5. This reads the native
mesh at x-dx/2 rather than x: a1.5cMpc/h shift per axis at N128/384, not a
physical displacement or a new survey-origin convention. Confirm against
PMWD gather on explicit particles in the same Slurm job. Preserve old outputs;
their numerical/quadrature tests do not certify their spatial registration.

The new live operator explicitly reads origin0; do not change particle
positions, the observer[192,192,192], or the PM gravity kernel. Keep the
source group's radial quadrature and rebuild density weight AND redshift
kernel from each evolved state. A frozen geometry may contain directions,
distance nodes/weights, zcos and redshift sufficient statistics, not frozen
rho or radial velocity. One saved-state comparison isolates this origin
change before joint updates. Other historical consumers are not silently
recomputed or scientifically promoted.

## Defined partial/development target

`p(s,nu|marks,z,selection-model) proportional N(s;0,I) N(nu;0,I)
  product_train_g L_conditional_g(marks|z,F_PM(s),theta(nu))`.

theta has exactly the preceding six coordinates/zero-centred development
prior widths: FP eta zero.004dex, four method modulus offsets1mag, selected
radial tilt2. No posterior-fitted zero point or covariance is used as a prior.
Each group's radial weight is dd*d^2*rho_F(d)*(d/100)^kappa in BOTH
numerator/denominator, and its conditional redshift kernel uses v_F(d) from
that SAME evolved state. Covariance150/100/50km/s, bias1 and observed-z
luminosity convention stay provisional. Kappa is not an identified inclusion
probability. Source-richness fit covariance/absolute calibration and the
complete group/point/count selection law remain missing.

No new role law is invented here. MW/M31/M33 must eventually be identified
from each NEW evolved field, with MW/M31 ambiguity and unresolved M33 kept,
and their actual observables must constrain that SAME field. Native truth
labels and best-seed selection are prohibited. N128/3cMpc/h cannot resolve
these components or satisfy the LG<=0.3 target.

## One bounded implementation/run

- Reuse PMWD N128/384 forward, LCDM parameters and time stepping unchanged.
  Export field-independent source geometry once; retain the closed5,330/1,491
  train/heldout groups. The marks are not the full CF4 catalogue.
- Two focused checks: native scatter/gather origin and live/frozen likelihood
  identity plus field/nuisance derivatives on a small synthetic case. Reuse
  prior calibration algebra checks. One initial actual N128 joint directional
  check at two fixed small steps is essential for the newly connected adjoint,
  not a new diagnostic ladder. Subtract the exact Gaussian-prior tangent
  from both AD and central differences; stop sampling when mark-only error
  divided by max(1,|mark AD|,|mark FD|) exceeds.02. Report absolute error too.
- Two short HMC chains,64 transitions each (32 warmup,32 retained), fixed
  seeds2026092501 and2026092702; all2,097,152 IC coordinates plus six
  nuisances move. Random4–8 leapfrog steps, initial step.01, cap.05,
  target acceptance.8; reuse existing wrapper. This is a transition/cost
  pilot, NOT sufficient posterior equilibration or uncertainty estimation.
- Preserve the16th/32nd retained state per chain: IC plus corresponding z=0
  density, mean velocity and physical velocity variance. No posterior mean/
  credible map from these short chains. Keep scalar transition summaries and
  source hashes; no per-step full-state archive. Planned output<0.7GiB.
- One Slurm H200/H100/A100-compatible GPU,4CPU,12GiB host memory (planning
  peak<=10GiB plus20%; prior N128 gradient batch peak4.38GiB),20min hard cap,
  900s application cap. No RAMSES dumps, TNG download or production expansion.

Q-GOAL: make real source observations change the same dynamically generated
present state and its calibration, while correcting its actual coordinate
contract. Keep this partial-model pilot distinct from the required full R2
posterior and later LG/zoom products.
Q-LEAN: one coupled model, two short chains, four retained physical states;
no new gravity solver, calibration sweep, gate framework or added data source.
The coordinate finding warrants one Fable5 advisory review (Astra backup only
on an unusable invocation); routine implementation checks remain driver-owned.

Remaining route: R2 actual current-state posterior -> R3 LG constraints ->
R4 precise forward validation -> R5 phase-consistent zoom ICs. This bundle
does not revise the goal or eliminate the galaxy-count/model-calibration work.

## Advisory disposition before launch

Fable5 read-only CLI request with the recorded prompt returned no text before
timeout300s/exit124. This is an unusable invocation, not a substantive adverse
verdict. Required Astra backup completed a read-only source audit: CONDITIONAL
PASS for this bounded development pilot, not production. It independently
confirms the native node/half-cell mismatch, same-state kernel/measure,
training ownership and state-independent HMC trajectory randomization.
Q-GOAL direct; Q-LEAN proportionate. No numerical execution by the adviser.

Driver adopts its one numerical correction: the already planned two-point
joint derivative check must expose the mark-only tangent separately, because
the millions-dimensional Gaussian prior can dominate the full-target tangent.
The implementation subtracts the analytic quadratic-prior derivative using
the SAME existing evaluations; no extra job or diagnostic ladder is added.
The small-tangent denominator is bounded by1 and absolute error is saved.

Retain the advice's limitations: finite radial window/Q257 define the working
target; one endpointQ513 check is not a global error bound;32 retained draws
per chain do not establish equilibrium;900s budget is cooperative between
chunks while Slurm20min is the hard limit. No calibrated selection/source/FoG
or full R2 posterior claim, and no MW/M31/M33 identification or M33 waiver.

## Execution record

H100406349 failed in48s during the new synthetic test, before geometry export
or field inference. The native scatter/gather origin test passed. The second
test incorrectly demanded a density response from a z-only multiplier while
both selected rays lay in the same z plane: this is a radial-constant weight
and must cancel in the conditional ratio. Correct the fixture to vary along
both rays (cos(x_index+y_index)); do not change the likelihood or its correct
normalization. Preserve the failed log. No sampling or science result existed.

Corrected H100406350 COMPLETED8m42s/exit0. Both new tests and both reused
calibration tests pass (4 total). Source geometry export took9.04s; coupled
driver runtime433.85s. The driver's measured process peak is7.56GiB, below
12GiB requested; Slurm's sampled MaxRSS7,064,188KiB (~6.74GiB) misses that
peak and should not replace it for later resource sizing. Nonfatal runtime
NUMA binding warnings appeared; no numerical failure followed.

## Measured results

The half-cell legacy scores reproduce the preserved Q257 cache to maximum
group error1.14e-13. Changing ONLY the native-node read gives train/heldout
log-factor differences-1.843043/+0.066214 at zero nuisance. Median/p90/p99
absolute group log-factor changes are.004514/.025244/.093375. This confirms
and quantifies the registration error; it does not show that it caused every
prior model failure or invalidate unrelated simulation products.

The initial full N128 IC/nuisance adjoint took42.81s including compilation.
Total AD tangent-859.386076 decomposes into analytic Gaussian-prior
-747.531964 and mark tangent-111.854112. The mark-only finite difference at
1e-5 is-111.851291: absolute difference.002821, scaled error2.522e-5
(0.00252%). The2e-5 step agrees comparably. This is one local consistency
check; it does not establish all-trajectory dynamics/gradient accuracy.

| Chain | Warmup / retained | Fixed retained step | Mean acceptance | Retained divergences | IC RMS change from start |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 32 / 32 | .013610 | .93589 | 0 | .63118 |
| 1 | 32 / 32 | .014781 | .89963 | 0 | .61555 |

All2,097,152 IC coordinates and six nuisances were free to move. Gaussian
prior and likelihood were applied once; no MAP power suppression, explicit
power renormalization or best-seed selection. The short transitions are NOT
equilibrated posterior samples. IC RMS near1 is not phase-recovery evidence.

For chain0, the initial and final saved train mark log factors are1542.4773
and1633.3805; heldout factors1863.3354 and1880.9588. Its density RMS changes
1.95311→1.98017 and global mean-velocity RMS244.265→241.515km/s. Thus the
physical field, not only calibration, really changed. The two chains' final
kappa values-1.2670/+1.0772 also make their short-run differences explicit.
No likelihood score here is an absolute evidence, predictive calibration or
demonstrated information-gain estimate. There was no matched prior-only
experiment; do not attribute all field movement uniquely to observations.

At the predeclared last state of chain1, Q513-minus-Q257 mark log-factor
changes are-0.002329 train/+0.001147 heldout. This is endpoint sensitivity,
not a bound over either chain or a validated continuum likelihood.

## Products, decision and next work

`/gpfs/kjhan/CF4/z0_density/r2_live_field_pilot_v1/` contains result.json,
scalar transition summaries, the initial_chain0 state, and four predeclared
states `chain{0,1}_retained{16,32}.npz`. Each state contains white IC modes,
the nuisance vector and its SAME evolved rho/mean velocity/physical velocity
variance/valid mask. These are NOT GRAFIC zoom files. Mesh origin0 is explicit.
Physical variance is not posterior uncertainty of mean velocity.

Five state bundles total346,900,261bytes; Q257/Q513 field-independent geometry
adds85,102,508bytes, approximately432MB total plus tiny summaries. No per-step
full-state archive, RAMSES dump or existing-output deletion. Field geometry
and source hashes are preserved in `r2_live_field_geometry_v1`.

Driver accepts and closes live-field coupling/transition implementation.
This is substantive progress beyond frozen-field zero-point inference, but
does NOT complete R2: the field is3cMpc/h, the catalogue is a source subset,
chains are deliberately short, and selected-group/source-fit/FoG calibration
plus full count/point/redshift ownership remain unresolved. No N256 run,
longer-chain extension, posterior mean/error map or R3/LG promotion follows.

The next substantive bundle must address the remaining selected-group/shared-
error model and its legitimate galaxy-data connection in this LIVE operator;
do not return to frozen-field offset sweeps or the rejected inclusive-count
times BGc product. A longer run of the current partial model alone cannot
settle those scientific omissions. Retain the same-state MW/M31/M33 contract
for R3 and the subsequent R4/R5 validation/zoom deliverables.
