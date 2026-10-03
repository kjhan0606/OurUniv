# Fable5 original independent proposal (verbatim)

2026-09-11. New tool-disabled safe-mode session; no driver repair draft provided.
This is a proposal, not accepted instructions, an implementation audit or verified
causal conclusions. Driver comparison/corrections: MEMBER_REPAIR_COMPARISON.md.
Raw provenance: config/cf4_member_independent_fable5_v1.response.json.

# Independent Next Plan: Member-Mass Learnability After NO_GO Pilot 341713

**Verdict up front:** the pilot did not test learnability — it tested whether a badly conditioned objective could be optimized from a hostile initialization, and it could not. The failure is a training-side pathology, not yet evidence that member structure is unextractable from the moments. I recommend one lean rerun with a residual-on-baseline parameterization and a bounded loss, with M33's per-cell map demoted to report-only, before any go/no-go on the science.

## Confirmed facts vs. hypotheses

**Confirmed from the supplied artifacts:**
- This is a *training* failure, not a generalization failure. Training-mean member L1 finished at 13.4 / 6.3 / 17.6 (MW/M31/M33) versus baseline 0.94 / 0.97 / 0.95. The model could not fit its own 13 training fields to baseline level.
- The final mean loss (~4.45) is far above the trivial "everything is remainder" solution, which scores ~0.75 on this loss (member L1 = 1 each, remainder ≈ 0). Two thousand AdamW steps at lr 1e-4 never found a solution a constant head bias would give.
- The cause is visible at step 1: softmax initialization puts ~25% of patch mass in each role; since true member masses are ~1e-4–1e-3 of patch mass, initial member role-L1 is in the thousands (M33: 39891) and the gradient norm is 1.8e5 against a clip of 10. The optimizer spends the whole run descending orders of magnitude of over-allocation and plateaus with members still 7–10× over-massed and overlaps of ~1e-4 (mass in essentially all the wrong cells).
- The mean-field OOD ablation degrades severely (L1 50–90), so the network does read the field — but only to modulate a globally wrong allocation.
- Prior evidence (shared M31/M33 peak in all 32 cases) stands: at 0.1875 cMpc/h, M33 shares cells with M31.

**Hypotheses (untested):**
- H1: with the objective conditioned properly, this architecture and 13 fields can beat the geometric baseline for MW and M31.
- H2: the per-cell *M33* map is intrinsically unidentifiable at this resolution regardless of optimization, because the field moments in shared cells cannot attribute mass between M31 and M33.
- H3: 13 positively selected fields are too few even for a well-conditioned learner. Untestable until H1's confound is removed.

## Concrete changes

Three small edits to the readout, no new system:

1. **Residual-on-baseline parameterization.** Predicted logits = log(baseline per-cell fraction + ε) + network output, softmax over roles as now. Zero network output then *exactly* reproduces the geometric baseline, so training starts at L1 ≈ 1 per member role instead of ~10³–10⁴, and the network learns deviations from the baseline rather than absolute allocation. Side benefit: under the mean-field OOD ablation, output collapses toward the baseline instead of exploding — bounded out-of-distribution behavior. The baseline fractions are already computed in `prepare()`; the frozen baseline tensor becomes a model input, not an oracle (it is a training-set average, available identically on new fields).
2. **Bounded per-role loss.** Train on mean over roles of log1p(role L1), keeping the existing role L1 as the *evaluation* metric. This preserves monotonicity in the metric while damping the 1/true_role_total gradient amplification that made M33 dominate. Lower the gradient clip to 1 and add ~100-step linear warmup.
3. **Step-0 sanity gate.** Before training, evaluate the untrained model and require its L1 to match the baseline within 1e-6 on every role and field; abort as INCOMPLETE_EXECUTION otherwise. This proves the parameterization, cheaply.

Additionally, **re-specify what M33 is asked to be**: quantify, per fixture, the fraction of M33 truth mass lying in cells that also hold M31 truth mass. If (as the shared-peak evidence suggests) this is large, per-cell M33 map L1 is not a valid pass/fail target; M33 is judged on total mass relative error and centroid error only, reported without gating this round. This is a criteria scope change relative to the original pilot criteria — I am recommending it on identifiability grounds, not treating it as approved.

## One lean bounded experiment

Same data, splits, seed policy, 2000-update cap, augmentation, tests, and eval harness; only the three edits above plus the M33 co-occupancy statistic. Budget fits the reference envelope with headroom (pilot used 2.2 GiB GPU, 949 s learning): 1 GPU (a40 sufficient), 2 CPU, 6 GiB, ≤90 min, Slurm excluding syn06.

**Numerical success criteria:**
- Step-0 gate passes (residual parameterization reproduces baseline exactly).
- Training: MW and M31 mean map L1 ≤ 0.8× baseline; remainder L1 not worsened by >10%. A pass must be corroborated by overlap *above* baseline and mass relative error *below* baseline — L1 alone is not identification.
- Development: MW and M31 map L1 below baseline in ≥2 of 3 fields with medians improved; M33 scalar mass and centroid errors reported against baseline, ungated.

**Stop criteria:**
- If, starting *at* the baseline, the learner still cannot reach 0.9× baseline on training MW/M31 within budget: declare per-cell member allocation from seven-moment fields at 0.1875 cMpc/h NO-GO (H1 falsified in-distribution), and route member constraints to the posterior stage instead (below). Do not retry with more steps, larger models, or new fixtures.
- If training passes but development MW/M31 fail: attribute to sample size/selection (H3), record, and stop — scaling fixtures is a separate decision, not an automatic follow-up.
- If M33 co-occupancy with M31 exceeds ~half its mass in all development fields: record per-cell M33 maps as unidentifiable at this resolution and never gate on them again.

**Q-LEAN:** does the pilot's failure reflect objective conditioning (residual model beats baseline on training fields) or task unlearnability (it cannot, even in-distribution)?

**Q-GOAL:** can any field-conditioned readout produce MW/M31/M33 mass hypotheses on *new* fields, with no oracle masks/IDs/positions, that beat the geometric prior — or must local-member information enter the pipeline only at the posterior level?

## What a mass-only readout cannot deliver (explicit)

- **No member COM velocities.** Velocity moments are inputs only. In cells shared by M31 and M33, one cell velocity mixes both members; mass-softmax weights must not be used to synthesize distinct member COM velocities, and I do not propose any velocity readout here.
- **No posterior.** The output is a deterministic point map with no uncertainty, no likelihood, and no selection model. The 32 positively selected fixtures cannot calibrate p(E|F,O), false-positive rates on fields lacking an LG analog, or q(S|F,O,E). A passing readout is a mean-function candidate, nothing more.
- **No role semantics on new fields.** "MW/M31/M33" are conventions internalized from selected fixtures (observer-cell occupancy, mass ranking, satellite proximity); on new fields these are hypotheses under that convention, not detections. No stellar/COM correspondence, no M200c.

## Alternatives rejected or deferred

- Class-reweighted CE: already withdrawn (biases physical allocation); not reopened.
- More steps / LR sweeps on the unchanged parameterization: rejected — the failure is structural.
- Two-stage detect-then-allocate, or higher-resolution targets/new simulation: deferred/rejected under current constraints; only revisit if the residual pilot passes training but confirms M33 map unidentifiability.
- Reviving diffusion/flow field priors: separate failed track, out of scope.
- Any large new validation framework: not proposed; the existing harness plus one gate and one statistic suffices.

## Link to the final objective

The CF4→z0 posterior→IC→LG-zoom pipeline needs local member constraints as *explicit additional observations*, since actual CF4 rows are empty in the LG R2 region. This experiment decides where those constraints can live: if the readout beats the baseline in-distribution, it can seed a future emulator/likelihood for member observables inside the posterior; if it cannot, member masses should be treated as latent variables with a physical forward model at the posterior-inference stage, and no further field-readout training should be authorized. No science execution is requested here — this is the plan and its decision rule.
