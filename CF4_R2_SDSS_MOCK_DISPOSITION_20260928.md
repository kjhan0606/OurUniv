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

## Next bounded R2 bundle

Single-purpose Slurm407268 was cancelled while still pending; it produced no
output. Its replacement is one source-pinned GPU job that runs the v6 support
check followed by the scale-matched TNG velocity-residual readout. This avoids
a source-pin change between two parts of the same bundle. It reads only
existing TNG moment and native-catalog products and writes aggregate
statistics, not mock rows.

After those bounded diagnostics, proceed toward a clearly labelled
**partial-target N128 posterior**, not production/R2 promotion:

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
   using the existing TNG moment/catalogue products. This one-box auxiliary
   readout is not a CF4/2M++ FoG calibration. Do not launch new gravity or
   download mocks for this.
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
