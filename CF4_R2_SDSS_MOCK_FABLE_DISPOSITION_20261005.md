# R2 SDSS-PV ensemble audit disposition — 2026-10-05

## Audit and driver verdict

Fable5 completed a read-only review of the corrected 2,048-mock diagnostic:
**CONDITIONAL PASS**. It confirmed the richness-bin centering, box-level
ensemble construction, checksum and reported values, and agreed that neither
the exact PGC overlap nor equal numeric richness cuts justify transferring
mock results to Tempel17. Q-GOAL is partial: this diagnoses an FP observation
component but produces no field. Q-LEAN is proportionate as a one-off; Fable
recommended no archive rerun, no redshift-stratified sweep and return to the
active expected-count/selection-law limitation.

The driver independently checked the proposed corrections and dispositions:

- The within-bin group-mean ratio is
  `sum_g (mean(e_g)-mean_bin)^2 / sum_g sum_i(sigma_i^2/n_g^2)`. It is not a
  covariance estimate by itself. The reported singleton ratio0.9267 agrees
  closely with its per-galaxy standardized residual variance0.9612²=0.9240.
  Across the richer mock bins, dividing by the pooled per-row residual
  variance gives only a rough1.003/0.999/1.029/1.064 comparison. Because the
  pooled galaxy weighting differs from the group denominator and errors are
  heteroscedastic, the driver does not adopt Fable's back-of-envelope
  exchangeable `rho` values as measurements.
- Howlett et al. (2022), §5/Eq.20, explicitly define `sigma_eta` as the
  skew-normal posterior standard deviation and `omega` as its scale
  parameter. The source convention verifies that `logdist_err` is an SD;
  the current moment-to-location/scale conversion is consistent with it.
  [Primary article](https://academic.oup.com/mnras/article/515/1/953/6611706).
- A hash-verified join of the exact active1,414 association rows to
  `SDSS_PV_public.dat` found1,414 distinct Tempel parent groups/singletons.
  `IDgroupT17=0` rows were treated as separate singletons. No two marks in
  this active conditional cohort share a Tempel group, so within-Tempel
  multi-mark covariance is absent from this factor set. Global FP zero-point
  uncertainty, full group inclusion, and groups outside the conditional
  cohort remain unresolved.
- The mock richness is selected mock-host row count; the paper's richness
  trend uses all Tempel group members, not only SDSS-PV-selected members.
  A possible redshift/selection mixture remains untested here; this is not a
  reason to refit or correct the mock results.

The driver adopts the audit's wording correction: the result **diagnoses**
SDSS-PV FP residuals and selected-host group-mean scatter; it does not
“calibrate” transferable CF4 covariance. The archived `result.json` remains
immutable and its broad legacy `limits[0]` wording is superseded by this
disposition. The source string is corrected for any future diagnostic run.

## Next work

Do not rerun the SDSS archive. Return to the active v6 2M++ expected-count
term and its selection/bias assumptions. The count factor must continue to
integrate the nonuniform field intensity over training-exposed cells,
including empty-cell expectation, while excluding heldout and buffered keys.
Static source review confirms that this Poisson structure is present; it
does **not** establish calibrated angular/radial selection, tracer bias,
redshift-success, or group inclusion. Any next numerical comparison must use
an already saved same-target state, training keys/counts only, and no extra PM
evolution. Do not reuse the predecessor split's ray exposure as a v6 gate.

Q-GOAL: a narrower conditional mark factor is now supported for the active
1,414-row cohort, but R2 still has no calibrated full selection law or
stationary z=0 posterior. MW/M31 roles remain ambiguous and M33 unresolved;
their observables must constrain those same roles on the same NEW evolved LG
field at `<=0.3 cMpc/h`, with native truth identities used only for
calibration/evaluation.

Q-LEAN: exact-source join and literature check closed the only necessary
mock applicability questions. No mock rerun, covariance fit, Tempel finder,
heldout score or gravity run is warranted by this result.
