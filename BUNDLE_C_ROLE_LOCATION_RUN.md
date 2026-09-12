# Approved probabilistic role-location pilot

2026-09-12: user explicitly approved implementation and one Slurm fit after
the driver-reviewed plan. Audit is advisory under the user's latest policy;
the unusable Fable preamble is NOT an audit pass. No additional audit invoked.

Implementation: `src/cf4_role_locations.py`,
`scripts/cf4_bundle_c_role_locations.py`, `config/cf4_role_locations_v1.json`,
and three focused tests in `tests/test_cf4_role_locations.py`.
Parameter sizing72,417; raw training cache728MiB; sequential conditional
backward passes. Three new tests cover center-cell periodic labels,48 signed
symmetries, normalized references/gradients/nonoracle rollout. Reuse one
existing physical-moment symmetry test. Static Python/shell syntax checks pass;
numerical tests are not yet claimed and will run in the pilot allocation.

Fixed computation:

- Fresh MW -> M31 -> M33 joint center-cell law on .1875 cMpc/h,128^3 native
  fields. No previous member-mass weights; no unique-cell exclusion.
-13 training fields,6500 joint updates =500 exposures each; strictFP32,
  AdamW1e-4/.01, clip10. Before/after full128^3,48-view probability check.
- Three reused development fields, all per-field proper joint/role scores;
  geometry, geometry*density, training-center-cell same-M31 reference.
  Conditional scoring uses native parents legitimately, not as a detection claim.
-64 nonoracle triples per development field, frozen before native-target
  comparisons; plots show NATIVE total density as background, not recovered mass.
- One final endpoint only; incomplete learning budget is inconclusive,
  technical failure is not a scientific no-go. No best-checkpoint selection.

Resources: Slurm1GPU/2CPUs/6GiB/4h, a40,a100,h100,h200 excluding syn06.
Host peak estimate5GiB +20%; GPU reserved-memory envelope20GiB. Learning cap
190min, application230min. First13 real updates provide timing/memory inside
the same fit. No manual node run or filesystem/process diagnostic.

Output: `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/role_locations_v1/`;
refuse an existing directory and preserve all older runs. Artifacts include
request/source/tests, pre/post symmetry, full model+Adam checkpoint/history,
samples/plots, proper scores and final result. No extra full density snapshots.

Status: implementation/static review complete, preparing source-pinned Slurm
submission. Job ID and initial numerical status will be recorded here.

Boundary: location-component FEASIBILITY ONLY, even if score criteria pass.
This does not certify physical halos, masses/COM velocities, calibrated q_S,
q_F, observed LG density/velocity posterior, or compatible LCDM ICs. Report
the endpoint and wait for approval before the next bundle.
