# Bundle B — local structure, LG constraints and a small dynamics bridge

2026-09-07. **Execution approved by the user; Bundle B in progress.**
Authority: CF4_MASTER_PLAN.md and the user's acceptance of the recommendation
after the2048-sample result. Astra driver: no external audit call is required.
This is one outcome-based bundle, not a new series of per-step approval gates.

## Starting point and purpose

Bundle A supplied actual CF4+2M++ z=0 diagnostic mean/sample density, mean
velocity, posterior uncertainties and heldout predictions. Job334358 passed
all predefined sampler gates (max Rhat1.03183, min bulk ESS113.59), but the
12 cMpc/h approximate field/selection model is NOT scientifically calibrated.
End the sampling-length/prior-repair series. Do not call this a validated
LG parent posterior or claim MW/M31/M33 recovery.

Bundle B asks: **does the current coarse field place the local environment
sensibly; what additional LG observations must constrain the fine field;
and can a coarse present-day target be connected to LCDM ICs dynamically?**
Preserve the direction observations → z=0 posterior → IC → forward check.
Actual high-resolution inference remains Bundle C; no direct-CF4-to-IC restart.

## Three deliverables

### 1. Actual-map environment comparison

Reuse `/gpfs/kjhan/CF4/z0_density/actual_data_longer_v3/task_0` and the original
frozen input/support files. Evaluate128 predetermined retained fields:
indices8,24,...,504 from each of the four chains, never select on appearance.
Use stored ensemble mean for display and individual fields for window
uncertainty; do not estimate window variance by adding voxel variances.

- Annotate Virgo/M87, Coma/NGC4874, all four historical Local Void probe
  positions, and Bootes Void. Also show the six existing secondary anchors
  (Fornax, Hydra, Centaurus, Norma, Perseus, Shapley), so success is not
  judged only at a few requested locations. Positions initially come from
  `config/p1_targets_v2_observer.json`; cite their provenance/uncertainty,
  do not import that file's old hard thresholds or screening policy.
- At12 cMpc/h, report aperture means at radii12 and24 cMpc/h, uncertainty,
  P(delta>0)/P(delta<0), and native-grid peak offsets inside the fixed24
  cMpc/h search region. These are coarse environment statistics, not halo
  masses, measured void boundaries, or subcell localization. Preserve all
  outcomes and the known correlation between overlapping apertures.
- Report survey support fraction and boundary truncation. In particular,
  the historical Bootes radius31 at distance155 cMpc/h extends beyond the
  current180 cMpc/h tracer cut. Do not interpret unobserved cells as empty
  space or wrap a survey boundary periodically for comparison.
- The native density origin is(i+.25)dx while counts use(i+.5)dx. Use
  `cf4_z0_physical_field` coordinates; the old `DensityScorer` assumes.5
  and cannot be used unchanged. Keep h=.746 for physical-distance landmark
  conversion, distinct from the2M++ LF's h=.6711 convention.
- Optional contextual overlay: existing Carrick density cube and README at
  `/gpfs/kjhan/CF4/observations/twompp_carrick_v1`. Rotate Galactic to
  supergalactic coordinates and account for i-fastest storage. Show only
  comparably coarse morphology, recording native cell windows and4 cMpc/h
  published smoothing; do not claim interpolation makes transfer functions
  identical. This is luminosity-weighted galaxy contrast, NOT matter delta.
  No amplitude-RMSE gate, independent-truth claim or extra likelihood factor.

Output: one annotated figure and one structure table, separating supported,
opposite-sign, uncertain and insufficient-support results. A95% sign interval
excluding zero may label a strong *model-conditional* sign, not observational
calibration. Do not invent an all-structures-pass scalar or select a seed.
The [Carrick et al. reconstruction](https://arxiv.org/abs/1504.04627) shares
2M++ data; the [Local Void reference](https://arxiv.org/abs/1905.08329) uses
Cosmicflows-3. Neither is guaranteed independent of our observational inputs.

### 2. Explicit LG observation/likelihood contract

Produce one source-backed MW/M31/M33 table with distance/sky direction,
relative line-of-sight velocity, proper-motion components/covariance,
mass definitions and uncertainty, and environment/isolation assumptions.
Separate measurements, astrophysical priors and numerical model discrepancy.
Use primary-source tables and document solar/reflex-motion transformations;
do not multiply correlated derived distances, speeds and proper motions as
independent measurements. Convert total radial motion to peculiar motion
with the correct Hubble term, rather than using the two interchangeably.

Existing `src/cf4_lg_z0_likelihood.py` supplies useful relative-velocity algebra,
not a current approved likelihood. Its historical configuration's80 km/s
velocity widths,0.30 Mpc/h separation width, equal1.2e12 Msun/h masses and
free midpoint offset are PM-screening assumptions, not measurement errors.
Do not restore its old seed bank, best-pair selection or direct-IC workflow.
Keep MW/M31 identities/directions, rather than an anonymous pair somewhere
near the observer. M33 is a required target, not an optional scoring bonus.

Starting references, to be checked against later primary measurements during
contract preparation, include [van der Marel et al.](https://arxiv.org/abs/1805.04079)
for component motions/systematic errors and [Brunthaler et al.](https://arxiv.org/abs/astro-ph/0503058)
for M33 masers. [Patel et al.](https://arxiv.org/abs/1609.04823) is a dynamical
interpretation, not an independent duplicate mass or proper-motion measurement.
Do not freeze a numerical LG likelihood from abstracts or legacy screen cuts.

Define a callable observational likelihood and test units/covariance/velocity
frames on a tiny synthetic catalog. Attach metadata indicating which fields
require resolved halos/subhalos. N32 cannot evaluate MW/M31/M33 membership:
do not paint three peaks into its single LG voxel or run HOP on the N32 grid.
M200c, FoF, stellar mass, virial mass and stripped subhalo mass are not
interchangeable. M33 requires an appropriate substructure/peak measurement.

Output: measurement table + likelihood interface for the multiresolution
z=0 model, with explicit coarse/fine mass and momentum consistency. In Bundle C
the fine component must be updated by these observations, not just prior-filled.
No claim of actual LG conditioning until a field/catalog resolves the operator.

### 3. Bounded z=0 → IC → forward-z=0 development test

Reuse the two known development PM targets (indices0 and5 via
`scripts/cf4_z6_native_physics.py:read_native`), not new simulation/training
ensembles. Fit each target's coarse density AND mass-weighted velocity using
the existing N64 PMWD LCDM forward model and conservative mass/momentum
aggregation to N32, preserving native origin. Extract the existing JAX forward
core from `build_pmwd_truth` as needed; do not differentiate through its NumPy
wrapper. Keep the existing cosmology, LPT/time-step/mesh conventions.

One fixed unconditioned initialization per target, at most200 objective/gradient
evaluations per target. No access to generating IC phases as an initializer.
Minimize a declared coarse-field discrepancy plus standard-normal IC prior
penalty. Before either fit, freeze separate density/velocity normalizations
from the existing training fields1-4, not by tuning weights to fitted targets.
Record all evaluations and prior penalty/power; do not enforce P(k) by rescaling
the resulting IC. Take one independent final forward evaluation of each fit.

Development success rule: finite positive unit-mean density, conservative
mass/momentum readout, and both density and velocity RMS residuals reduced
by at least50% relative to that target's fixed starting forward field. Report
absolute residuals and physical spatial correlations as well. This is an
engineering progress threshold, not evidence of observational precision.
Failure within the cap is informative; do not relax weights/gates or add seeds.

This produces regularized IC *candidates*, not samples of an IC posterior,
unique recovered phases, independent validation, or an actual LG realization.
The targets were previously used for development; the PM generator is shared.
Before later actual IC inference, specify the correct joint/conditional target
or proposal correction. Never multiply the already data-derived z=0 posterior
by CF4/2M++ likelihoods again, or assume its empirical field prior is the
dynamically induced prior. An arbitrary positive density/velocity draw need
not admit an exactly matching growing-mode LCDM IC.

## Cost, sequencing and boundary

Implement/evaluate the three outputs as one bundle, with no external per-step
audits while Astra drives. Use existing loaders, maps, PM and scoring algebra;
no generic validation/provenance framework or filesystem work. Run all numerical
work through Slurm, never manually on syntax/syn101.

- Map analysis and focused tests:2 CPUs, estimated3000 MiB +20%=3600 MiB,
  at most30min/job. Reuse the two established astronomy/JAX environments.
- PM compatibility prototype: one GPU,4 CPUs, estimated10000 MiB +20%=12000
  MiB, total cap4 GPU-hours across both targets (including compilation and
  tests). Partitions a40/a100/h100/h200, exclude syn06. If measured memory
  or adjoint cost exceeds the cap, stop and report; no automatic larger run.
- Persist final/coarse fields and scalar progress only, <=2 GiB new products;
  no RAMSES snapshots, halo bank, fine-mode search or production zoom.

Before Bundle C approval, present the annotated actual map, LG contract and
mock bridge result together. Missing coarse evidence is not fixed by choosing
a prettier seed; failure of the dynamics bridge is not hidden by sampler pass.
Conversely, acknowledge diagnostic adequacy without reopening endless prior
repairs. State what contributes to the final LG<=0.3 cMpc/h, surroundings1-2
cMpc/h objective, what is unresolved, and the concrete multiresolution test.

Execution and fixed job IDs are recorded in BUNDLE_B_RUN.md.
**Bundle C still requires user approval.**
