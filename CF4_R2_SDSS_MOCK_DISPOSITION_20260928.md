# R2 SDSS-PV mock decision — 2026-09-28

## Decision after read-only Fable review

Do not download or stream the proposed 15-member SDSS mock subset. The
dependent Slurm job **407269 was cancelled while pending**; no mock data were
read or retained. Fable's verdict was **CONDITIONAL PASS** on the conditional
individual-mark route, but **REJECT** on using the SDSS host-member mocks to
calibrate the model's 100 km/s LOS displacement. The driver independently
checked the source wiring and adopts that distinction.

The current linked-FP kernel uses
`sigma_radius = h * sigma_los / H0`; at `h=0.746`, `H0=74.6`, and
`sigma_los=100 km/s`, this is **1 cMpc/h**. It enters the radial source/count
transfer, not the FP measurement PDF itself. A typical 0.1-dex distance
uncertainty at 80 cMpc/h is about 18 cMpc/h, so this blur is a small part of
the individual distance-mark width. The SDSS mock's central/satellite
host-relative velocities measure its approximate HOD member offsets, not the
residual of a source velocity from a TSC-smoothed 3 cMpc/h PM field. They do
not calibrate this kernel or the six-population 2M++ FoG law. The old proposed
per-file gate also made 135 two-sigma checks and rejected on any failure;
that is not a sound promotion rule.

Howlett et al. define the individual FP log-distance-ratio PDF with a
candidate-distance-dependent `f_n` normalization. In §5.1, each proposed
distance changes the cosmological redshift/distance used for that selection
normalization. Thus the supplied FP mark already carries its individual
selection correction; do not apply a second one. This supports conditioning
on the observed selected FP object for the individual-mark likelihood only.
It does not model group incidence, field-dependent association, the
CF4/2M++ shared observation graph, or shared multi-member errors. The full
SDSS mock selection also does not reproduce the full DR8–DR14 redshift-success
law.

## Completed bounded R2 bundle — Slurm 407305

The source-pinned replacement for pending-only 407268 completed on the pinned
commit `7e0071670d342d8d8ee6c0a61254a9da8c421051` in 2m51s (exit 0, peak RSS
1.81 GiB). It ran the v6 source-membership/dynamic-support screen and the
scale-matched TNG velocity-residual readout, using existing products only.
No new mock rows, gravity run, fit, or posterior were produced.

- The v6 audit parsed 8,901 training eta rows: 6,805 grouped, 2,022
  catalogue-present singletons, and 74 catalogue-absent rows. Within matched
  observed-cz/magnitude bins, singleton-minus-grouped eta contrast is
  `-0.00788 ± 0.00203` dex (2,018 row pairs); catalogue-absent-minus-grouped
  is `-0.03367 ± 0.00271` dex (74 pairs). These bins do not control true
  distance, so neither contrast identifies eta independence or an inclusion
  probability. No heldout eta values were parsed.
- For the 429 strict training singleton links, 48 mark-blind, radius-stratified
  groups were checked at velocity scales 0.5/1/1.5. Rebuilt periodic RSD
  neighborhoods match the full-source likelihood to `<=1.8e-15` in log factor
  and `<=3.8e-16` in density sum for this fixed-state stress. This verifies
  bounded-subset mechanics only; it does not certify support over posterior
  field states. The separate 1,414-link fixed-state reference also remains a
  mechanics comparison, not inference.
- In one existing 75 cMpc/h TNG box, the 3 cMpc/h TSC-smoothed total-matter
  velocity residual has isotropic 1D dispersion 122.3 km/s for 144,044
  resolved centrals and 307.0 km/s for 33,949 resolved satellites. Thus
  100 km/s is of central-residual order in this particular sample, but is far
  below the satellite value. The different cosmology, subhalo selection and
  single box prohibit transferring either number as a CF4/2M++ calibration or
  assigning an observed galaxy to a central/satellite class.

Disposition: **diagnostic complete; R2 posterior still NO-GO**. Group
selection, distance-dependent association, shared FP/group covariance and
live-field posterior support remain uncalibrated. Full machine-readable
outputs are `r2_v6_membership_live_support_v1/result.json` and
`r2_tng_tsc_velocity_residual_v1/result.json` under
`/gpfs/kjhan/CF4/z0_density/`.

## Next bounded R2 bundle

The next candidate is a clearly labelled **partial-target N128 field fit and
prospective evaluation**, not production/R2 promotion. It must use the frozen
v6 graph, keep buffered count exposure out, distinguish linked singleton and
Tempel-grouped FP terms, and explicitly retain the uncalibrated selection and
covariance assumptions. Do not interpret a MAP/Laplace product as a calibrated
posterior. Keep the 6,946 heldout count keys and heldout singleton FP marks
out of fitting and use them once for prospective evaluation.

1. Use the frozen v6 training count graph and exclude its buffered keys from
   Poisson exposure. Couple those counts to the existing one-link FP mark
   factor; keep the 429 strict training singleton FP rows distinct from the
   985 Tempel-grouped rows among the 1,414 one-link groups. Add TF rows only
   after their source mechanics match this factor.
2. State the approximation that association is field-independent conditional
   on observed position/redshift and bound the residual distance-dependent
   inclusion effect across a predeclared ±3σ true-distance range. This is a
   sensitivity bound, not a selection calibration.
3. Replace arbitrary shared LOS widths only where source-backed quantities
   apply: test Tempel's published group dispersion for grouped FP members;
   bound the ungrouped field-galaxy residual against the 3 cMpc/h PM velocity
   using the existing TNG moment/catalogue products. The completed TNG screen
   gives 122.3 km/s for centrals and 307.0 km/s for satellites; use only as
   explicitly labelled sensitivity scales, never as a transferred CF4/2M++
   FoG calibration or latent-role label. Do not launch new gravity or download
   mocks for this.
4. Use the existing N128 adjoint for MAP, then a Laplace/linear-response
   uncertainty approximation. Do not restart the previous identity-mass HMC
   path with ESS of only 2–5.
5. Run one prospective validation: predictive score for the 6,946 untouched
   heldout count keys versus prior predictive, plus coverage/log score for
   untouched heldout singleton FP marks. No gate ladder or N256 escalation.

Q-GOAL: this is the shortest route to a same-evolved-field z=0 density and
velocity estimate with a prospective check; it is still partial and cannot
establish the final LG solution. At N128 the inner 5 cMpc/h cut leaves the LG
cell unconstrained directly. MW/M31 remain ambiguous roles and M33 unresolved;
their observables must later act on the same new evolved field, and native
truth identities remain evaluation-only.

Q-LEAN: dropping the weakly matched 15-member stream avoids a low-value
external archive test. Reuse v6, the existing sparse factor and adjoint, and
only one heldout comparison. Group selection/covariance and calibration of
the count FoG remain open; no N256 or production posterior is promoted.

References: [Howlett et al. 2022](https://arxiv.org/abs/2201.03112),
[official SDSS-PV v1.1 data/mock release](https://zenodo.org/records/6824749).
