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

Source commit `50841a2` committed and pushed. Slurm job **347007** accepted
and RUNNING on **syn05**, start **2026-09-12 15:52:02 KST**,4h allocation.
Logs: `/gpfs/kjhan/CF4/logs/cf4_C_role_locations_347007.{out,err}`.
Initial execution: **4/4 numerical tests pass**. All13 training fields loaded.
Before-fit whole128^3,48-view check passes for all3 roles: maximum probability
L1 differences[3.509e-6,1.402e-6,2.107e-6], below1e-4. Native labels use
SubhaloPos center cells as planned, not previous member density peaks.

Job347007 COMPLETED/exit0 at2026-09-12 18:28:35 KST,2h36m33s. All6500
updates,500 per field, final symmetry and all192 autonomous triples completed.
Final native-fixture symmetry L1 maxima[5.195e-10,1.701e-9,3.519e-9] pass.
Host application peak2.714GiB/GPU reserved2.469GiB.

**NO_GO_ROLE_LOCATION_AT_FIXED_BUDGET.** Training mean joint NLL .0000387;
development mean63.0029 vs geometry24.8393/density10.5120/shared10.1410.
Per-development joint model/density NLL:16=155.215/9.940,
17=23.312/11.151,20=10.481/10.445. All frozen aggregate criteria fail.
Autonomous role median cell-position errors for16=[2.079,.265,.419],
17=[1.779,1.769,2.0625],20=[0,1.993,2.226] cMpc/h. Each64-draw set has only
8–10 distinct triples. Normalized probabilities/symmetry do not imply reliable
uncertainty or generalization. Close this fit; preserve all results, no extension.

User approved the recommended population-weighted SMALL location-model
comparison next. Plan `BUNDLE_C_POPULATION_LOCATION_PLAN.md`; it fixes the
inherited13-profile-subset limitation using existing catalogue positions and
field volumes. No actual high-resolution posterior is promoted by either task.

Boundary: location-component FEASIBILITY ONLY, even if score criteria pass.
This does not certify physical halos, masses/COM velocities, calibrated q_S,
q_F, observed LG density/velocity posterior, or compatible LCDM ICs. Report
the endpoint and wait for approval before the next bundle.
