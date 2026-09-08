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
