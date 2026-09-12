# One member-mass conditioning repair

2026-09-11. User approves the final smaller combination in
`MEMBER_REPAIR_COMPARISON.md`. This concrete plan is submitted to Fable5 before
launch; its independent proposal was not approval of this combined plan.

## Purpose and bounded change

Final goal: CF4 + galaxy/LG observations -> present density/velocity posterior
-> LCDM-compatible ICs -> forward validation and LG zoom. LG <=.3 cMpc/h,
environment1–2. This is a native-data mass-readout feasibility attempt, NOT a
fine observed field, member phase-space posterior or IC calculation.

341713 completed2000 updates/17m34s, tests3/3, but all training/development
criteria failed. M33 development mass ratios9.50/4.63/8.44 and overlaps about
1e-6 show mass excess AND incorrect placement. Unique cause is not identified.

Reuse existing530804-parameter U-Net, flat four-way softmax, ten total-field/
observer features, full128^3/.1875-cMpc/h native patches, fixed13 training and
three development fixtures16/17/20. No hierarchical head or spatial-template
residual. Two changes only:

1. Compute pi_r = mean over TRAINING fields of integrated role mass / total
   field mass. Set final head weights zero and biases log(pi_r). Thus initial
   prediction is pi_r times the local total mass, NOT the old spatial baseline.
   All pi_r must be positive. No development target enters initialization.
   Initial backbone gradients are zero on the first update; the focused
   backward test checks subsequent nonzero learning as well as finite gradients.
2. Per-field/per-role loss = log(P_r/T_r)^2 + KL(t_r/T_r || p_r/P_r), equal
   weight across all four roles, coefficient1. P/T are integrated predicted/
   true masses. Use log-softmax and logsumexp on positive-total cells. Zero
   total mass stays zero; no physical mass floors or binary target conversion.
   Native equality has zero loss; exact zero fractions are softmax limits.

## One fit, two internal learning segments

- Same seed911101, AdamW1e-4/weight-decay.01, gradient clip10, no tuning sweep.
- First300 updates on fixed TRAINING fixture0, no augmentation. Save step0
  and step300 mass/shape metrics. At300, every MW/M31/M33 must have mass ratio
  in[.8,1.2], overlap>=.8 and normalized map-L1<=.3. Require nonnegative maps,
  positive integrated remainder and conservation<=1e-6; no member erasure.
- On screen failure save checkpoint, actual maps and criteria and STOP. Label
  INCONCLUSIVE_SINGLE_FIELD_LEARNING;300 steps are an engineering budget, not
  proof of sufficient optimization or general scientific unidentifiability.
- On pass continue the SAME model and Adam optimizer for1700 further updates
  on all13 training fields with48 rotations/mirror symmetries. Do not reset,
  change seeds/criteria, or request another between-segment audit. Report
  fixture0's extra exposure. No new independent fit or automatic retry.

Final evaluation reuses the old spatial training-average baseline, native
maps, mass/L1/overlap/centroid metrics and secondary OOD mean-field ablation.
Report original criteria unchanged: training each role mean L1<=.8 baseline;
development each M33 L1<baseline; each host median L1<=baseline. Also require
each development M33 L1<.8*min(baseline_L1,1) and each host median overlap>=.5.
All must pass for MASS_MAP_FEASIBILITY_ONLY. Otherwise NO_GO for this candidate.
Incomplete total updates are INCONCLUSIVE_BUDGET, not a completed failure.
These reused development fields are NOT independent validation.

## Identity, observations, essential/deferred

Only total mass, mean velocities, three physical velocity dispersions and a
coarse observer coordinate enter a NEW field. Native IDs/masses/centers are
target/evaluation information only. Roles MW/M31/M33 remain explicit, including
shared-cell/unresolved M33; there is no distinct-peak requirement or permanent
waiver. Deterministic maps are hypotheses, not unique galaxy identifications.

A later joint field/member model must let LG masses, positions and velocities
constrain the SAME total field, not merely relabel its unchanged density. That
observation-conditioning link, stochastic assignments, COM/stellar velocity
offsets, selection p(E|F,O), fine/global priors and actual CF4/LG inference are
missing and deferred. Pass here cannot certify that link. Neither outcome
authorizes another density fit, ICs, simulations or the next science bundle.

Q-GOAL: bounded effort toward spatial member information needed for local
observational constraints; not a universal prerequisite for all reconstruction.
Q-LEAN: reuse one runner/model/metrics; two loss/head changes, one internal
learning screen, no new validation framework or diagnostic series.

## Execution envelope

One Slurm GPU,2CPUs,6GiB host,90min; partitions a40,a100,h100,h200, exclude
syn06. Previous host peak3.99GiB/GPU2.20GiB. Retain conservative expected host
peak5GiB plus20% request6GiB; stable log-space intermediates fit the existing
20GiB GPU envelope (24GiB with20% margin). Same-sized network/cache; no CPU
science on syntax. Static syntax checks locally, focused numerical regressions
in the allocation. Max2000 total updates, learning70min, application87.5min.
Rough full-run estimate20–45min, not a measured ETA. Preserve all v1 outputs;
new output `member_mass_repair_v2`. One report/checkpoint and component maps.
