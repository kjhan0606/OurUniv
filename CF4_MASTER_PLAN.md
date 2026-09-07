# CF4 / OurUniv — active master plan

Effective 2026-09-07. The user approved registering the goal-alignment review
as the highest-level project plan and proceeding with short, substantive
bundles. This file supersedes CODEX_PLAN.md, PLAN.md, the Hong restart plan,
and cf4_science_route_v3.json as execution/priority authority. Preserve those
files and previous results as history; do not silently revive their routes.
Direct subsequent user instructions take precedence over this file.

## Scientific destination

Use actual CF4 observations, galaxy-density observations, and explicit local
structure constraints to infer a **present-day density/velocity posterior**;
construct compatible LCDM initial conditions; evolve them forward and verify
the observed local environment and MW–M31–M33 system. Deliver phase-consistent
zoom ICs for studying LG formation and evolution, not a unique historical
reconstruction of every small-scale phase.

- Surroundings: 1–2 cMpc/h numerical reconstruction scale is sufficient.
- LG: target <=0.3 cMpc/h numerical reconstruction scale, with explicit LG
  mass, position, distance and relative-velocity uncertainties.
- Finer IC particle/force resolution is a separate zoom requirement. AMR
  alone does not improve particle mass resolution. Fine modes not fixed by
  observations remain conditional/prior content, not recovered observations.
- Retain Virgo/Coma and Local/Bootes Void environment constraints. Separate
  cluster zooms, full-volume 0.3 phase recovery, and RT/stellar/AGN/dust work
  are not prerequisites for the LG deliverable.
- Push defensible observation/structure information toward high k, especially
  in the LG region. Measure information gain; do not arbitrarily choose a
  tiny low-k domain and call the goal achieved. Conversely, global frontier
  certification must not block a useful, explicitly qualified LG prototype.

## Four outcome-based bundles

| Bundle | Deliverable | Exit decision |
| --- | --- | --- |
| A — usable present-field model | One bounded prior comparison, then an actual CF4 + galaxy-data preliminary z=0 density/velocity product with uncertainty and held-out predictions | Declare model limitations and whether it is usable; no indefinite toy-model repair loop |
| B — LG and dynamics connection | Explicit LG constraints in the z=0 route and a small mock z=0 → IC → forward-z=0 demonstration | Check LG/environment conditions and field/velocity consistency after evolution; test this before expensive high resolution |
| C — multiresolution reconstruction | Surroundings at 1–2 and LG at <=0.3 cMpc/h, with coarse/fine coupling and measured observation/structure information | Show LG-conditioned information gain, resolution/cost feasibility and residual uncertainty; recalibrate scale-dependent priors as needed |
| D — zoom IC and evolution | Phase-consistent nested ICs from buffered Lagrangian particle membership and a forward-evolved LG ensemble | Check LG masses/separations/velocities, environment and contamination, then decide on production |

The scientific order remains observations → z=0 posterior → IC → forward
validation. Bundle B is an early end-to-end mock risk test, not permission to
bypass the present-field stage or restart the historical direct-CF4 route.
Do not double-count the same data by treating a data-derived z=0 posterior as
an independent likelihood. Any approximate inversion/proposal must disclose
its target and correction before calling its output an IC posterior.

## Current state and bounded immediate work

**Bundle A is user approved and active. B–D require user approval before
starting each bundle.** Routine fixes/tests within A do not need a new
approval. No automatic next-bundle launch.

Current execution record: [BUNDLE_A_RUN.md](BUNDLE_A_RUN.md).

Z4–Z11 were N32, 12 cMpc/h development experiments, not actual-data maps.
Z8's amplitude interpretation was erroneous and has been withdrawn. Z10
fixed-field and Z11 prior-compatible controls recover the injected tracer
curvature; Z9 free-field PM cases do not. This supports a field/observation
model mismatch but neither isolates its sole cause nor proves a quantile
transform will fix it. Z11 completed all four fits; its results are at
`/gpfs/kjhan/CF4/z0_density/z11_prior_control_v1/comparison.json`.

1. Finish ONE training-only non-Gaussian-prior comparison against the saved
   native-PM controls, with at most four new mock fits. Reuse existing
   simulations, likelihood, sampler and reports. Require spatial field
   recovery and held-out predictions, not just a nicer density histogram or
   recovered nuisance coefficient. Exact cases/resources/decision rule:
   `config/cf4_bundle_a_prior_to_data_v1.json`.
2. Driver judges the comparison. No automatic additional prior families,
   nuisance terms or new simulations if it fails. State the model limitation
   and decide its suitability for a preliminary observation-space diagnosis.
3. Within A, implement and run at most one actual-data preliminary four-chain
   fit after its input semantics and chosen model are recorded. Use actual
   CF4 velocity data and prepared 2M++ counts, preserving selection, errors,
   overlap treatment and heldout separation. No pseudo-truth or truth-based
   accuracy/coverage numbers for the real universe. If model suitability
   fails, the product is a model-stress diagnostic, not a validated posterior.
4. Deliver actual-data mean/sample density maps, mean velocity and posterior
   velocity uncertainty, predictive residuals and limitations. Distinguish
   uncertainty in the mean field from physical velocity dispersion/FoG;
   current population FoG parameters are not a resolved sigma_v(x) field.
   Close A with a concrete Bundle B design and request approval.

The real-data preview remains 12 cMpc/h and cannot certify the target LG
scale. Independent galaxy/selection mocks and physically relevant validation
are required before scientific promotion, but not an unbounded prerequisite
to viewing a clearly labelled diagnostic map. Reused development fields and
shared generator/inference code are never independent validation.

## Working rules

- Prefer substantive scientific outputs over more generic validation code.
  Reuse tests; add only checks necessary for the changed computation.
- A clean sampler is necessary, not scientific success. Separate posterior
  mean smoothing, individual-draw structure, uncertainty and phase recovery.
- Syntax is a Slurm login server. Numerical jobs use Slurm; no manual syn101
  or login-node calculations. Current GPU partitions: a40,a100,h100,h200;
  exclude syn06 for these fits. Request estimated peak memory plus ~20%.
- GPFS is ordinary shared storage. Read/write scoped project artifacts;
  do not implement storage/inode/renameat2 probes or process-scan monitoring.
- Use fixed job IDs and final artifacts for bounded checks; no pgrep loops.
- The Astra driver plans, implements, runs, evaluates and commits. The user
  waived Fable pre-audits and Opus closure audits while Astra is the driver.
  Other drivers follow the user's applicable audit policy.
- Commit/push coherent changes. Preserve unrelated user work and all failed
  scientific results. Keep run summaries current; do not confuse submission,
  sampler pass, scientific acceptance and final-goal completion.
- At every bundle boundary report the result, goal contribution, unresolved
  risk and next deliverable. When approval is needed say **승인해주세요**.
