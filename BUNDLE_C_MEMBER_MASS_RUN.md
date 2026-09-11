# Native field-only member mass readout: one approved pilot

2026-09-11. User approved `BUNDLE_C_REPAIR_DISPOSITION.md` scope1 only.
Fable5 conditional plan review and driver loss correction are already recorded;
no additional external audit, field-prior retraining or downstream job.

Implemented `src/cf4_member_mass_readout.py` and
`scripts/cf4_bundle_c_member_mass.py`, fixed configuration
`config/cf4_member_mass_pilot_v1.json`. This is a deterministic mass-allocation
learner, not a calibrated component posterior, new observed LG, resolved halo
finder or a physical member-velocity model. Outputs keep these limitations.

## Fixed scientific and numerical choices

13 training/3 development native fixtures as approved, no split/seed changes.
Mass/mean velocity/directional sigma features use the existing fixed physical
scales1e11 dx^3 Msun,300 km/s and100 km/s. Observer coarse cell is[8,8,8] in
the1.5 grid, whose center is[68,68,68] on the fine grid; no true subcell MW or
M31/M33 position is read by feature preparation. The coarse observer condition
and selected three-member E are supplied conditions, not inferred selection.

Native halo IDs and cached sparse member moments build TRAINING/EVALUATION
targets only. Features/forward accept no target identifiers or component masks.
All128^3 support retained; cached native remainder plus3 component masses
must reconstruct total mass to1e-9 relative-max (source roundoff recorded).
Training inputs and targets are FP32; native checks and reported metrics use
FP64. Global training mass unit1e12 Msun cancels in normalized map-L1 loss.
Final predicted fractions multiply the supplied total, conserving mass to
FP32 precision (1e-6 relative-max check), with no mass floor or target override.

Model530804 parameters:10 input channels, GN8/SiLU two-convolution blocks,
widths16/32/64, two stride downs and nearest-neighbor+convolution ups, no batch
normalization/dropout/attention, four softmax outputs. Residual checkpointing
is not required: activation checkpointing is used around each block. AdamW
1e-4/weight decay.01, gradient clipping10, seed911101, shuffled equal-field
epochs, one uniformly drawn signed-axis symmetry per update. Velocity and
observer-coordinate vectors and directional dispersions transform correctly.
One fit<=2000 updates/70min, final checkpoint only. No best-validation selection.

Baseline is unrotated equal-field average of native training mass fractions.
It never sees retained labels. Fixed mean-field ablation replaces only7 field
channels by training global means, keeping observer/grid; label OOD/secondary.
Loss, role-wise metrics and fixed20% training/three-case M33/host development
rules implement the disposition, with no threshold search. Include zero-mass
null L1=1, predicted/true masses and empty predictions so a weak engineering
criterion pass cannot be misreported as an identified/accurate galaxy.
No separate M33 peak required. All components may occupy the same cell.
Soft role maps are hypotheses; deterministic outputs do not quantify assignment
uncertainty. Never multiply total moment channels into inferred member velocities.

## Tests, resources and outputs

Two focused tests: actual-model parameter count/allocation/backward/zero-loss
identity and all48 signed feature/observer transformations. Reuse the prior
augmentation/restriction test:3 tests in this SAME Slurm allocation, before
any fitting. No login numerical tests or separate preflight calculation.

Concrete static sizing `config/cf4_member_mass_sizing_v1.json`:530804 parameters,
model/gradient/Adam only.008GiB; largest single feature array32x128^3=.25GiB.
Block checkpoint/skip/recompute plus workspace/allocator engineering bound
20GiB GPU+20%=24GiB. Host5GiB+20%=6GiB request. To avoid rereading the large
native arrays each update, preparation reads one native patch at a time and
caches only13 PREPARED FP32 cases (15 channels each,1.523GiB total), not13
FP64 seven-moment patches or the complete400^3 source. This refines the earlier
one-patch-cache wording without changing data or the approved memory envelope.
The engineering estimates are not measurements; same-fit first update and
every50th update compare actual peaks with sizing. No silent model downsizing.

Slurm runner `scripts/run_cf4_bundle_c_member_mass.sbatch`:1GPU,2CPUs,6GiB,
90min; partitions a40,a100,h100,h200, exclude syn06. Any syn101 use is ordinary
Slurm allocation, never manual. Application5250s hard budget,70min learning;
budget-truncated training is inconclusive even if incomplete fit looks good.

New output `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/member_mass_readout_v1`:
final checkpoint; native/predicted/baseline/ablation3D mass maps for cases16,
17,20; four-role comparison PNGs with shared per-role color scales, loss curve,
and compact metadata/history/results. Plot log-scale lower limit is DISPLAY
only, never a physical mass floor. Save metrics incrementally; no polling
daemon, raw particle pass, new simulation or storage/process probes.

Complete this one pilot, then report result and wait at the next bundle
boundary. A pass allows proposing joint-member-state/denoiser repair, not
automatically executing it. Failure does not establish scientific impossibility.
Submission and measured result will be recorded separately below.
