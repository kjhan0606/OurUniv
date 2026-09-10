# C spatial diffusion — approved implementation and execution

2026-09-10. User approved implementation and one training/evaluation allocation
after Fable5 CONDITIONAL GO. No further plan audit or downstream bundle added.

## Frozen implementation

- `src/cf4_spatial_diffusion.py`:301-channel input (49 real,49 five-state
  one-hot categories,7 root conditions),48/96/192/384 spatial U-Net, two residual
  blocks per level/path, GroupNorm, timestep/scale conditioning, bottleneck
  attention and checkpointed residuals.34,095,557 parameters by static layer
  count; the actual model must match before optimization. No old flow weights.
-100-step cosine Gaussian DDPM noise prediction plus absorbing discrete
  diffusion. Discrete survival1-t/T, reverse reveal1/t; uniform-time weighted
  masked CE is T/t per site. No auxiliary CE/loss search. Both heads condition
  on both noisy modalities; fixed codes remain fixed in the reverse chain.
  Padded unused continuous targets are zero and ignored physically at decode.
- Full legal-branch table/version `cf4_mixed_joint_canonical_v1` is the
  `canonical` function/docstring. Native identity/FP64 moment roundtrip must
  pass before any update. Corrections and invalid draws remain visible.
- Native training slab x<256 fine cells cached once (2.136GiB); random origins
  x0..16,y/z0..49 coarse cells and all48 signed axis permutations. Native
  moments transform before encoding, no arbitrary rotations/cross-moment
  assumptions. Eight training-only normalization cubes, seed902102; uniform
  origins stay disjoint from the two historical retained cubes. Correlated
  one-box development, not independent-volume validation or rare-LG coverage.
- One fit, seed902101,30,000 updates, AdamW2e-4/weight_decay.01, gradient cap10,
  EMA.999. First32 updates use largest64^3 parents for measured feasibility;
  subsequent steps balance three scales. They are real updates in the same
  optimizer run. Last EMA only, no heldout model selection or automatic retry.
-22h learning cap from job start (including tests/preparation),24h Slurm cap,
  application deadline23h55m. Budget-incomplete learning cannot be promoted.
  Recovery checkpoints at10k/20k/30k and a budget-stopped final update; native
  fine fields are saved only for fixed evaluation, not every training epoch.

## Evaluation and failure handling

Reuse old morphology functions and tolerances, old retained origins and seed
99803+100*case+draw. This model trains ONLY the three LG refinements, not the
old environment cascade. Thus the direct historical comparison is **8 fine
draws**, not the old16 containing8 additional environment draws. There is no
claim those untested environment criteria pass. Same factor-two ratios and
seven-channel1e-8 conservation criterion remain. No evaluator threshold tuning.

Also retain old two training contexts and2 draws per training/retained case
for true-parent versus rollout at.75/.375/.1875, paired RNG where valid.
Stage1 shares the actual same draw; stages2/3 restore pre-rollout RNG for the
true-parent comparison. Fine unresolved/invalid draws count as failures, no
redrawing. Runtime/time failures do not receive a scientific success verdict.
New maps show native/control/repair/diffusion on the same projection/color scale;
JSON and plots include power/connectivity/velocity/directional sigma and loss.

The generated law is a sampler with explicit failure outcomes, not an evaluable
field likelihood or an actual observed posterior. No native member IDs assigned
to new fields; q_S, selection,1.5 global environment and actual LG conditioning
remain missing. No CF4 data is newly fitted here. Failed/inconclusive outcome
requires addressing member-state learnability before further field training,
with a new concrete plan/user approval. Closed diagnostics stay closed.

## Memory, execution and outputs

Static written sizing: `config/cf4_spatial_diffusion_sizing_v1.json`.
Parameter+gradient+Adam+EMA storage0.635GiB; GPU engineering peak20GiB includes
both modalities, checkpointed activations, convolution workspaces and allocator
headroom. Host engineering peak12GiB includes the2.136GiB native slab,
temporary moment/encoding buffers, HDF5/evaluation/library/cache allowances.
These are estimates, not measured peaks; in-job32-update record compares them.
Peak host RSS or GPU reserved beyond these allowances stops this run honestly.

The design-stage48GiB host request was provisional. The concrete bounded cache
and layer sizing support a SMALLER request:12GiB x1.2=14.4GiB, rounded to
**--mem=15G**. This reduces the approved resource envelope; no larger run or
model change. GPU20GiB x1.2=24GiB fits the requested partition hardware.
Worst-scale update estimate2–12s, balanced1–6s, NOT measured throughput or
convergence prediction.30k updates may exceed the22h cap; record INCONCLUSIVE_
BUDGET if so. Evaluation52 refinements x100 steps, estimated15–108min; partial
results remain if application time is exhausted. Queue delay is additional.

Runner `scripts/run_cf4_bundle_c_spatial_diffusion.sbatch`: Slurm only,
partitions a40,a100,h100,h200, exclude syn06,1 GPU,4 CPUs,15GiB,24h.
One allocation runs8 focused tests, native identity, sizing comparison,
learning, evaluation and result recording. No login/manual-node numerical run.

Outputs `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/spatial_diffusion_v1` (new only),
estimated<6GiB. Logs `/gpfs/kjhan/CF4/logs/cf4_C_diffusion_JOBID.{out,err}`.
No GPFS/inode/rename/process-scan diagnostic or monitoring framework.
Submission/job ID and measured results will be recorded below, separately
from this implementation description. Source syntax checks passed; numerical
tests have not yet run at the time this initial record is written.

## Submitted

Pushed implementation source **341b419**, Slurm **338402**, submitted
2026-09-10 09:03:02 KST. Bounded scheduler check: **PENDING(Priority)**;
no allocation, numerical tests or learning yet. ReqTRES confirms1 GPU,
4 CPUs,15GiB,24h; excluded syn06. Backfill estimate at09:03:06 was13:01 KST
with SchedNodeList=syn101; this is an estimate, NOT an allocation/start promise.
If Slurm assigns syn101, it is an ordinary scheduled job, not manual execution.

After allocation the same job performs8 tests, native identity, the32-update
operational segment, bounded learning and fixed evaluation. There is no
separate polling daemon and no automatic launch of a subsequent science bundle.
Current source syntax/shell checks pass; numerical results remain UNAVAILABLE
until this scheduled job actually executes. No observed LG product is claimed.
