# Independent member-repair planning comparison

2026-09-11. User requested independently constructed driver/Fable plans and
comparison. Planning only: no model patch, new training, science job or adopted
change to final criteria. Preserve both original drafts rather than rewriting
them into agreement after seeing the other answer.

## Independence and evidence

- Driver draft `MEMBER_REPAIR_DRIVER_INDEPENDENT.md` was frozen BEFORE the
  new Fable call. SHA256 before AND after receipt:
  `24311d0e1d0b25f6d38b4f549f31e54d8c625251649539c33d2da8ef5fea3656`.
- Fresh Fable5 CLI, safe-mode (CLAUDE.md/customizations disabled), tools empty,
  strict MCP, no session continuation/persistence. It received only neutral
  task/goal constraints, existing failed source/config and saved result/data
  summaries. No driver new proposal or preceding recommended fix was sent.
- Fable completed normally123782ms. Main response model claude-fable-5;
  auxiliary Haiku routing call is recorded, not claimed to be the planner.
  Exact request/response `config/cf4_member_independent_fable5_v1.*`;
  readable verbatim proposal `MEMBER_REPAIR_FABLE_INDEPENDENT.md`.
- This is independent PROPOSAL formation from shared evidence, not independent
  validation of data or code. Driver has seen earlier Fable reviews as project
  context; the new Fable call has not seen the driver's current draft.

341713: tests3/3,2000 updates complete; every training/development criterion
fails. Retained M33 mass ratios9.50/4.63/8.44 and overlaps8.16e-6/1.23e-6/
1.66e-6 show both mass excess and severe spatial misallocation. Scalar
identity e_L1=P/T+1-2*overlap makes this clear without another numerical probe.
Neither a small mass correction nor unchanged longer training is justified
as a sufficient remedy by these facts. Unique cause remains unestablished.

## Side-by-side original proposals

| Decision | Driver (before Fable answer) | Fable independent answer |
| --- | --- | --- |
| Working diagnosis | Sparse components, mass/shape learning and optimization are poorly conditioned; causal uncertainty retained | Hostile initialization and gradient scale dominate; attribution made more strongly than evidence permits |
| Starting prediction | Integrated TRAINING mass fractions initialize biases; no spatial template | Spatial TRAINING-average baseline plus residual network logits |
| Allocation | Foreground sigmoid times3 role shares, remainder complement | Keep four-way softmax |
| Training loss | Per-role log integrated-mass ratio squared + spatial shape KL | Per-role log1p(normalized map-L1), clip1,100-step warmup |
| First check | Fixed fixture0 learning for300 steps; if successful continue SAME fit on13 fields | Step0 reproduces baseline, then all13 fields for2000 updates |
| M33 | Retain mass/map overlap, no separate peak; require improvement beyond zero-mass null | Demote M33 map to report-only, recommend permanent waiver if co-occupancy exceeds roughly half |
| Budget |1GPU/2CPUs/6GiB/90min,2000 total across internal segments | Same reference envelope and2000 updates |
| Final claim | Mass-map feasibility only; velocities/joint posterior missing | Mean-function candidate only; velocities/joint posterior missing |

Common substantive conclusion: a simple attainable starting state and evidence
of actual spatial learning are needed before another long fit. Software pass,
conservation and a falling average loss are not member identification. Both
exclude restarting the failed field-prior model, new simulations and oracle
component labels at new-field inference.

## Driver evaluation: accept useful ideas, correct overclaims

Fable's baseline-residual approach is attractive for minimal code changes and
starting from a useful geometry predictor. But the written formula and claims
cannot be implemented literally:

1. softmax(log(b+epsilon)) gives (b+epsilon)/(sum b+4epsilon), NOT b exactly.
   Without epsilon, baseline-zero cells have -infinity logits and can never
   gain member mass by any finite residual. A spatial baseline thus requires
   an explicit support-opening distribution, with changed step0 maps measured
   separately from the original baseline. An arbitrary epsilon followed by
   claiming exact equality is not acceptable. This matters especially when
   a small leaked fraction of a huge background competes with tiny M33 mass.
2. Zero network output reproducing a chosen start map does NOT guarantee that
   the trained residual vanishes on mean-field/OOD inputs. No claimed bounded
   OOD error follows from this construction. Also log1p(L1) compresses the
   loss but is mathematically unbounded, not a bounded objective. It keeps
   each scalar ordering but changes tradeoffs in an average across roles.
3. A shared nearest peak is NOT a measurement of shared cell mass fraction.
   Even measured co-occupancy>50% is not a proof that spatial/velocity context
   cannot statistically inform M33. Reject the proposed permanent M33 waiver.
   M33 can remain unresolved/probabilistic, but must stay explicit in the final
   goal; neither planner can declare it recovered or impossible from this trial.
4. Failure of one short fit does not falsify learnability of ALL seven-moment
   models. Training pass/development fail does not uniquely identify sample
   size/selection. Likewise the exact degree of initial equal allocation is
   not measured simply by noting a random four-way softmax. Treat these as
   plausible hypotheses, not confirmed causal deductions from two log entries.
   In particular4.45 is the last update on fixture9, not the final mean over
   all13 fields; first and last updates use different fixtures/symmetries.

Weaknesses of the DRIVER draft also matter:

- Hierarchical sigmoid/softmax has no greater expressive support than flat
  softmax and may add another saturation bottleneck; no demonstrated advantage.
- Three simultaneous changes would not identify which caused an improvement.
- The300-step single-patch cutoff is an engineering budget, not a proof of
  sufficient optimization time. It can stop too early. Passing it can be pure
  memorization; neither outcome certifies observational identifiability.
- Shape KL/mass loss has a correct zero-error native optimum but changes the
  objective; there is no empirical proof it will optimize or generalize better.
  A deterministic conditional mean cannot replace uncertain member hypotheses.

## Recommendation after comparison — not yet an approved execution plan

Prefer a SMALLER combination, not concatenating both complete proposals:

1. Retain existing spatial backbone and four-way allocation initially. Defer
   the driver's hierarchical architectural change and Fable's spatial residual
   baseline because neither is necessary to try the loss-conditioning hypothesis.
2. Start the output at measured TRAINING integrated mass fractions via explicit
   head initialization. This borrows Fable's attainable-start principle without
   inheriting zero-support cells from a spatial template. Keep original spatial
   geometry baseline for comparison; do NOT call the starting field that baseline.
3. Use the driver's separated integrated-mass/shape objective. This addresses
   both the large mass-ratio gradient scale and almost absent native overlap;
   log1p on L1 alone can reduce gradients without supplying as direct a spatial
   target. This is a reasoned preference, not a measured superiority claim.
4. Within ONE bounded fit, first attempt one fixed training field; only then
   continue on the fixed13/3 split if the preset mass AND spatial criteria pass.
   Keep the total budget bounded and report inconclusive rather than impossible
   on a failed learning screen. Exact implementation/criteria need finalization
   before approval; no new parallel architecture/loss sweep is recommended.
5. Preserve explicit M33 evaluation and distinguish mass calibration, spatial
   overlap, assignment uncertainty and eventual same-field observation influence.
   No auto-removal of M33 based on a shared peak or this learner's performance.

Q-GOAL: this is still a necessary practical attempt at member-state information,
not a completed z0 posterior or a mathematical prerequisite that every possible
CF4/LG reconstruction must pass. The final CF4+galaxy+LG->z0->LCDM IC->forward
LG zoom route remains unchanged, LG<=.3/environment1–2. Both the fine-field
prior and member joint velocity/selection/uncertainty links remain missing.
Q-LEAN: reuse source/backbone/maps, one short internal learning segment and
one potential continuation, no extra diagnostic infrastructure or new data.

This comparison is NOT Fable approval of the subsequently combined plan: Fable
independently proposed its own option. A concrete next bundle needs its normal
plan decision/user approval before execution. No calculation submitted here.
