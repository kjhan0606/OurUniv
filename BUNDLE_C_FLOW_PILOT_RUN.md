# C conditional-flow pilot

User approved implementation plus one GPU pilot,4h maximum,2026-09-08.
This approval does not include actual CF4/LG inference, q_S, global384-grid
deployment or new simulations. Design BUNDLE_C_CONDITIONAL_FLOW_DESIGN.md.

## Implemented bounded configuration

`cf4_split_moments.py`: seven binary splits of an eight-child octet, retaining
all directional variances. This replaces the design's interior sphere chart
with an equivalent49-interior-degree chart easier to handle at boundaries.
For mass weights w,1-w, per-axis relative-velocity coordinate r=tanh(t), and
internal-energy share a, split means differ by sqrt(V/[w(1-w)])*r; child
variances are V*(1-r^2)*a/w and V*(1-r^2)*(1-a)/(1-w). Every factor conserves
M/P/Qdiag. Explicit empty/cold/end-point branches have categorical probabilities;
no positive-density floor. Cold detection only removes64-epsilon cancellation
relative to raw second moments. Test native roundtrip, do not infer its accuracy.

`cf4_conditional_split_flow.py`: one shared six-coupling3D affine flow,32/64
hidden channels, no BN or global normalization. Boundary probabilities factor
as mass-state, three velocity-states conditional on mass-state, three internal-
variance states conditional on those states, with physically mandatory inactive
states. Continuous flow includes only active coordinates and is conditioned
on all states. Its density is in split coordinates, not unconstrained physical
cell coordinates. Parent and root density/mean velocity/three variances plus
node and physical scale condition spatial convolutional networks.

Seven autoregressive spatial split-field factors constitute an octet, and six
factor2 physical scales share the model. This is not seven independent mixture
templates. Node-level context is available from prior ancestral splits at
generation, not a fine-truth leakage. Flow layers couple neighbouring spatial
positions through unchanged coordinate channels; inverse/Jacobian regression
is integrated before the data/fit. This is a finite patch law, not a global
law manufactured by multiplying overlapping crop densities.

Data: twelve fixed-seed uniformly drawn24-cubes from native x<=48, with native
periodic y/z. Test cubes have lower faces(49.5,0,0),(49.5,25.5,25.5), both
side24, no train/test or mutual test voxel overlap. Use all7 total-matter
moments, not halo-only or LG-selected sources. The source has been used in
historical work, so report WITHIN-FIT SPATIAL HOLDOUT / REUSED-SOURCE DEVELOPMENT,
not fresh or independent-universe confirmation. Coordinates/split are frozen
before outcomes. No raw source pass or filesystem diagnostics.

Single fixed fit:6000 Adam steps,lr2e-4, batch1, parent crops<=24 per axis,
equal exposure to child scales6,3,1.5,.75,.375,.1875. Final checkpoint only,
not best-heldout selection. Standardize continuous chart units from training
only, no physical amplitude changes. No augmentation family/seed search.
Native exact7-moment and mass-weighted velocity/sigma roundtrip must pass on
all14 source cubes and all six scales BEFORE fitting. Moment relative error
<=1e-9; per-axis velocity/physical-sigma RMSE<=1e-4 km/s. These are numerical
representation requirements, not observational tolerances.

Evaluate four draws for each of two cases and two regimes: native12→1.5
environment and native1.5→.1875 fine. Save native references and full generated
7-moment fields. Fixed random-seed0 draw also receives a deliberately permuted
root-context intervention while actual conservation parents are unchanged;
measure response, but do not count that altered-context field as a prior draw
or treat sensitivity as correct conditioning. Neither regime is actual-data
inference; no fixed native halo is carried into generated fields.

Predeclared development criteria: each regular draw must have seven-parent-
moment errors<=1e-8, and ratios within[.5,2] for the upper four of eight power
bands, top1%cell mass fraction,6-neighbour hot connectivity, per-axis bulk
velocity residual RMS, physical-sigma RMS and boundary/interior density-gradient
contrast. All16 regular draws must pass for a DEVELOPMENT DIAGNOSTIC PASS,
never scientific certification or posterior calibration. Report all results,
including failures, per regime. No amplitude repair, longer training or seed
selection follows a failure automatically.

Slurm partitionsa40/a100/h100/h200, exclude syn06, GPU1, CPUs2, host24 GiB
(estimated20+20%), GPU allocated-memory ceiling20 GiB with24 GiB headroom
target,4h total. Preparation and learning stop at3h elapsed to reserve evaluation
time; finite tests/preparation may fail before training. No guarantee of
convergence within the cap. Estimated artifacts<3 GiB in
`bundle_c_v1/conditional_flow_v1`. One job runs two focused tests, native
representation checks, fitting and final evaluation; no extra audit stages.

Current state: implemented, numerical verification awaits the one Slurm job.

Submitted336268, source1dc835b committed/pushed. Logs:
`/gpfs/kjhan/CF4/logs/cf4_C_flow_pilot_336268.{out,err}`.
Output progression: request.json→representation.json→status.json/training.jsonl
→checkpoint.pt→mock_fields.h5/result.json. A failure before the main program
(e.g. regression) is in Slurm stderr; main-program failures write failure.json.
No downstream inference, independent audit or automatic replacement job queued.

## V1 outcome and authorized continuation2026-09-09

336268 COMPLETED on H100 NVL,2026-09-08 21:22:03–21:26:26 KST,4m23s,
exit0,6000 steps complete. All representation checks passed: maximum moment
error3.45e-15, velocity RMS1.11e-12 and physical-sigma RMS2.80e-12 km/s.
Both focused tests passed. Numerical directional-variance loss is resolved.
But all16 environment/fine regular draws failed development morphology gates.
Fine high-band power ratios0.256–0.558; environment additionally shows poor
connectivity/boundary contrasts. Status NO_GO_CONDITIONAL_FLOW_DEVELOPMENT.
Products/checkpoint preserved. Four hours was a cap, not actual training time;
low GPU memory use is not proof of a GPU problem or of sufficient optimization.

User requested continued work after this report. Code review identifies a
training/evaluation context mismatch: training crops parent fields to24^3,
but finest generation evaluates64^3. Zero-padded convolution contexts differ.
This is a concrete implementation mismatch, not evidence that it is the ONLY
failure cause. Scope now: frozen-checkpoint diagnosis and ONE conditional
same-pilot correction if the paired numerical evidence confirms the mismatch.

Frozen audit: first original train and first original heldout cube, finest
nodes0/3. Compare likelihood on identical target indices20:44 in native64^3,
isolated24^3 crop and48^3 context with12-cell margins. Require inverse relative
error<1e-4, forward/inverse logdet error<1e-3, padded/full same-cell maximum
logp difference<1e-3, but isolated-crop mean absolute difference>1e-3 on all
four pairs. These are numerical cause-separation rules, not revised scientific
gates. Also record native train/held NLL, inverse-latent moments, permuted-root
NLL and one finest truth-parent refinement. They do NOT alone distinguish
underoptimization from model capacity. Save a native/one-step/old-rollout plot.

If that audit exits0, use full parent extents at every scale during one fresh
6000-step fit. Architecture, initialization/data/sample seeds, record sequence,
learning rate and scientific gates are unchanged. Old crop RNG draws are still
consumed so record order is identical. Full fields change context AND number
of scored cells per update; improvement would not isolate padding from data
exposure. No optimizer continuation, longer fit, best checkpoint or amplitude
repair. Preserve v1, write `conditional_flow_v2_full_context`. Abort without
fitting if paired evidence does not justify the correction.

Single Slurm job runs tests→frozen audit→conditional correction/evaluation,
GPU1/CPU2/host24 GiB (estimated20+20%),4h total with original3h preparation/
training stop. Audit outputs<5 MiB, corrected run<3 GiB. Pipeline audit dir
`flow_checkpoint_audit_v1`. No GPFS/inode checks or process-scanning monitors.
No actual CF4/LG inference or further correction series is authorized here.

Submitted337195, sourcead9fc7a committed/pushed. Logs:
`/gpfs/kjhan/CF4/logs/cf4_C_flow_context_337195.{out,err}`. Stage evidence in
`flow_checkpoint_audit_v1/result.json`; corrected fit artifacts appear only
if audit permits it. No follow-up job outside this bounded pipeline queued.
