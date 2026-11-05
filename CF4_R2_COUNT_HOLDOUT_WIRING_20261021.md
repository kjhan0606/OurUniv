# R2 count holdout prediction wiring — 2026-10-21

This is evaluator plumbing only; it is not a field result or a posterior.

- `src/cf4_r2_count_predictive.py` exposes the active conditional-v6
  present-field-to-six-population count-intensity map at the frozen observed
  grid, retaining native cell-volume scaling, whole-box bias normalization,
  source chunking and the existing LOS/RSD settings.
- `ConditionalCountHoldoutPredictive.update_field` sends one supplied field
  and tracer state through that forward map, then records one complete joint
  held-out Poisson log likelihood. The equal-weight log-mean-exp preserves
  repeated retained states and zero-probability states.
- `update_coordinates` accepts the canonical `[white_IC(N^3), tracer(9),
  population(15)]` draw, evolves its IC block through an injected unchanged
  PMWD field callback, and extracts only the tracer block for the 2M++ count
  factor; population coordinates belong to the CF4 mark factor.
- The scorer requires six populations, disjoint train/held-out exposure
  masks with nonempty held-out exposure, unique sparse keys, and
  role-consistent occupied keys. It does not certify that inputs are posterior
  draws or that the observation law is calibrated.
- Validation: CPU-only synthetic tests pass (5 forward-map, 12 scorer tests).
  A read-only check of `r2_sky_closed_split_v6/split.npz` loaded only
  `heldout_flat_voxels`, graph-buffer exclusion keys and training keys; it did
  not load held-out keys or counts. It found 262,144 held-out voxels, disjoint
  train/held-out exposures, and all 37,951 training keys exposed.
- No evolved N256 field, held-out outcome, posterior sample, predictive score,
  uncertainty or calibration result was computed. The held-out arm remains
  measurement-wise sealed for future post-fit evaluation. A separate,
  score-blind CF4 mark identity manifest now records split roles, membership,
  identifiers and graph ownership only; see
  `CF4_R2_MARK_HOLDOUT_EVALUATION_CONTRACT_20261021.md`.

Q-GOAL: enables the required same-field z=0 count prediction once a valid
posterior exists; does not deliver the requested posterior or the LG
`<=0.3 cMpc/h` constraints. MW/M31 remain role-ambiguous and M33 unresolved;
their observables must ultimately constrain those same roles on the same NEW
evolved field, with native truth identities used only for calibration or
evaluation.

Q-LEAN: no new data search, simulation, sampler run or held-out scoring was
added. Slurm job `431345` remains queued under the user's eight-GPU QOS cap;
changes are uncommitted until its pinned source commit has passed startup.
