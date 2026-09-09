# C: field-only LG candidate identification and proxy calibration

2026-09-09 user approved this priority before the paired learning experiment.
The question is whether a NEW .1875 seven-moment field supplies distinguishable
MW/M31/M33 candidates and useful observational proxies, without known members.
Fable5 audits this plan with Q-GOAL/Q-LEAN and explicit identification scrutiny.

## One deliverable, one CPU job

Implement `identify(field, dx, observer_cell)` with NO catalogue/true positions/
native membership inputs. Use the cached32 native24-cube fixtures for testing
only, not a fresh survey or independent universes. The fixtures already condition
on an observer lying in coarse cell[12,13.5)^3; disclose this source selection.
The observer coarse cell is an explicit external location condition available
to the algorithm, not the exact MW position. Do not count this as blind discovery
of the MW anywhere in a384 box. Actual global deployment remains unimplemented.

- Find positive26-neighbour local density maxima on the supplied grid, no
  smoothing or catalogue-dependent threshold. Collapse tied adjacent maxima
  deterministically by lowest flat index. Exclude the outer two cells so both
  aperture proxies have complete support; log that boundary exclusion.
- Each maximum yields grid center and two fixed spherical-grid apertures,
  radii dx and2dx, integrated directly from TOTAL seven moments. Return aperture
  mass, mass-weighted mean velocity and three physical dispersions. Overlapping
  apertures are correlated proxies, NOT disjoint bound members or M200c.
- MW role candidates: peaks in the specified observer coarse cell. For each
  MW candidate retain M31 candidates within the existing native support ceiling
  3 physical Mpc, and M33 candidates within1.5 physical Mpc of each M31. Expand
  ceilings only by one grid-cell diagonal for quantization, fixed before results.
  Do not apply uncertain aperture-M200c mass cuts or pick the brightest triplet.
  Enforce three DISTINCT candidate identities; retain ambiguity, not best LG.
- Do not materialize huge triplet arrays: save candidate lists/role adjacency
  and counts. An empty MW set or absence of a distinct third peak is an explicit
  unresolved outcome, NOT permission to use the true catalogue's missing peak.

Freeze candidate lists first. THEN evaluate against native MW/M31/M33 labels
from existing calibration metadata, using minimum-total-distance one-to-one
matching with dummy unmatched slots and one-cell matching radius. Report all
nearest distances, shared-nearest-peak cases and role adjacency as well as
match fractions. The radius is a diagnostic convention, not a halo-finder
certification or scientific GO gate. Never invent a subcell M33 to pass it.

For matched candidates only, compare aperture masses/mean velocities and grid
positions against native bound mass/member COM velocity/catalogue center.
Training-only unique object IDs supply role-specific mean and covariance of
[log(bound/aperture mass), center residual xyz in kpc, COM velocity residual xyz].
Use the dx aperture as the frozen primary proxy,2dx as descriptive context;
no aperture-radius tuning on holdout. Report raw7x7 covariance, rank and counts;
if fewer than8 unique matches, do not invent full covariance or precision.
Report original heldout residuals using training means. Repeated profiles count
once per role/split, original32 patches and spatial overlaps remain disclosed.
The matched-only covariance omits detection/assignment failures, so it CANNOT
be used as the complete field-conditioned LG law. No inverse-covariance weight
or Gaussian LG posterior is automatically produced.

## Connection and stop decisions

g(F) now exists as a FIELD-ONLY CANDIDATE/PROXY CATALOGUE, not yet three identified
physical components. The future observation law must marginalize role assignments
and include a calibrated missed/unresolved branch; q_S or an explicitly labelled
proxy likelihood remains necessary. Keep existing resolved_halos guard intact.
Observed distances/LOS/PM and Solar nuisances would enter once through candidate
position/velocity/proxy likelihood; native truth never enters those candidates.

If distinct M33 is typically absent at .1875, this directly constrains the next
design: retain a subgrid satellite latent variable or introduce finer LOCAL
information, rather than demanding an impossible third grid peak. That choice
requires a new plan/approval. Do not discard M33 or silently identify it with M31.
If candidates are useful, next bundle builds a normalized assignment/missingness
proxy law before actual LG conditioning. This job itself is not that posterior.
No additional threshold/model/finder comparison automatically follows a failure.

Q-GOAL: resolve the circular component-identification assumption before LG data
can constrain a field. Q-LEAN: one finder, two fixed apertures, reused native
fixtures, two focused tests (field-only interface/conservation and merged/unmatched
case) plus one output report/map; no generic gate framework or new ML fit.
Use Slurm2 CPUs/4800 MiB (4000+20%)/30m, NO GPU; existing allowed CPU requests
on a40/a100/h100/h200, exclude syn06. No raw snapshot pass or GPFS diagnostics.
Outputs<20 MiB, sequential fields in memory, no full3D copied outputs.

## Fable5 plan audit disposition (2026-09-09, before execution)

`config/cf4_lg_identification_fable5_plan_audit_v1.response.json` returned
CONDITIONAL GO; Q-GOAL direct and Q-LEAN proportionate. All four in-scope
disclosure conditions are incorporated, with no new approval/gate/framework:

1. Distance ceilings 3/1.5 physical Mpc come from the frozen population supports
   in `scripts/cf4_bundle_c_lg_population.py`; expand by sqrt(3)*.1875 cMpc/h.
   They are population prior support, not individual object positions.
2. Count truth objects in the excluded outer two cells, per fixture and role.
3. Velocities are BOX-frame peculiar km/s for both truth and proxies. Residuals
   are native minus proxy (log native/proxy for mass); position residuals are
   physical kpc at z=0, using h=.6774. No Hubble/Solar correction is introduced.
4. Report role completeness, shared-nearest-peak cases and the discrete aperture
   intersection/minimum-volume fractions for every matched pair at both radii.

Tied-maximum group counts are also reported. Native one-to-one distance matches
are evaluation associations, NOT oracle-free role classifications: report the
matched triplet's membership in the field-only role adjacency separately.
Matching minimizes distance with dummy cost1.001dx; it does not force maximal
completeness. Fixed first train/heldout maps use the observer z slab; red truth
positions are projected evaluation-only overlays, not candidate inputs.
