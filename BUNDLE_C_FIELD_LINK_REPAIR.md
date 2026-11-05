# One matched fine-lattice / physical-budget correction

2026-09-13, user requested implementation. Driver-reviewed routine bounded
repair, not a major discovery, goal revision or large production calculation:
oneGPU/2CPU/6GiB/2h maximum, below the prior generator's4h allocation. No
external call or routine approval wait. No unbounded parameter sweep.

## Evidence and intervention

Scale comparison347160 completed57s, tests4/4,16 valid native-parent draws,
zero learning. At.1875 native-parent mean bulk RMS ratio1.185 versus rollout
1.567; physical sigma .949 versus.799. Bulk fraction of the SAME1.5 variance
budget is native.210, teacher.288, rollout.494. Already at.75 it is native.094
versus generated.235. Variance decomposition identities pass. Boundary
roughness remains even with native parents (finest log boundary/internal
mean1.209 versus native1.006). High-band power means near1 hide case scatter.
Thus there is both one-step error and propagation, not just a wrong coarse
input or a proven unique architecture defect. Close scale diagnosis.

Implement TWO related changes as one explicit correction, without claiming
that this two-arm test isolates their individual causal contributions:

1. Add a small continuous-denoiser residual path on the FINE lattice. An
   orthonormal8x7 binary-tree basis lifts each of seven groups of seven split
   coordinates to eight leaf sites, seven channels at2x spatial resolution.
   A16-channel3D network with five3x3 convolutions and timestep conditioning
   connects cells across parent boundaries, then the adjoint basis returns
   49 coordinate corrections. This is a LINEAR coordinate embedding, not a
   physical density/velocity decoder. Existing parent convolutions already
   communicated between parents; this changes inductive bias, not the first
   possible information path. Zero head means exact pretrained output at
   initialization. Added37991 parameters; total1528397. No physical smoothing.
2. Weight continuous v-target MSE by parent-only mass/variance importance.
   For mass channel use B=M; relative-velocity and allocation channels use
   B=M*sigma_axis^2. Weight=(1+log1p(B/mean(B))), normalized to spatial mean1
   per channel, repeated for seven split nodes. All-zero B gets weight1.
   Weights are strictly positive and depend ONLY on available parent state,
   not native fine targets. At fixed noisy input/category/context/time,
   multiplication by this positive diagonal metric leaves the ideal
   conditional-mean minimizer unchanged. It changes finite-capacity learning
   emphasis, not the desired ideal score or mass/velocity output amplitudes.
   Log compression limits rare-peak dominance without adding physical floors.

The physical seven-moment decoder, atom canonicalization, reverse schedule,
normalization,100 sampling steps and full80/inner64 geometry remain unchanged.
No direct shrinkage of bulk velocities, transfer into sigma, or density A(k)
repair. Such output correction would modify the prior without learning it.

## Fair bounded comparison and endpoint

Two branches from the SAME final24000 EMA: (a) unweighted original continuous
model; (b) fine path + budget weights. BOTH use fresh AdamW lr1e-4/decay.01,
6000 new updates, EMA.999 and clip10. This is not exact optimizer continuation.
The original categorical model is frozen and bitwise checked in BOTH arms;
it still responds to each arm's generated noisy state during sampling.
Same1388 observers, same shuffle/48 augmentations, balanced3 scales, identical
step-specific noise/times/category masks. Saved native normalization reused.
No extra model/seed/LR/weight sweep or best-checkpoint selection.

All numerical regressions run in the same allocation before fit: existing4,
plus3 for lift isometry/gradient, zero initialization/cross-parent path, and
positive unit-mean/cold-limit loss weights. Host estimate4.5GiB plus20% rounded6.
One80^3 16-channel FP32 fine activation is31.25MiB; a conservative12-map
allowance375MiB plus stem inputs/lift buffers is additional to the old~1.2GiB
GPU reservation; total ceiling8GiB. Source slab remains1.65GiB; stream other
fields, don't cache multiple raw source cubes. Learning cap50min per branch,
110min TOTAL including evaluation,120min Slurm. Incomplete fit is inconclusive.
Save model/EMA/optimizer at2000/4000/6000 and on interrupted training; no
automatic restart on a science miss. Output<1GiB (two sets of8 saved fields,
checkpoints/plots and metadata), no new native particle data.

After EACH branch, reuse the original fixed denoising and16 full rollout
draws, ALL old morphology criteria, frozen same-field LG position-law/actual
likelihood readout, and first8 fields/projections. No oracle halo parents or
native fine halo is supplied on generated-field identification. Add the
existing variance-budget and log-boundary diagnostic to stored first draws;
compare arms AND original frozen model, not just their training losses.
No discarded invalid draws or replacement acceptance criterion. Actual
promotion still requires original feasibility and scientific qualification;
modest improvement alone is a repair result, not an observed LG posterior.

Q-GOAL: improve the total z=0 field's small-scale spatial/kinematic consistency
needed by same-field LG conditioning, without reverting to direct-CF4 IC.
Q-LEAN: small added path, reuse data/checkpoints/tests/evaluator; one matched
comparison tests improvement beyond extra steps. MW/M31/M33 remain probabilistic
center cells inferred from each generated field with no true candidates.
Unresolved M33, physical member masses/COM, archive-E selection assumptions
and actual1.5 CF4 environment posterior remain limitations. No fake member
mass map, actual LG on/off inference, IC or simulation is launched here.

## Execution

Source4d66631 committed/pushed; Slurm347264 submitted2026-09-13 11:42:31 KST
and RUNNING on syn05. All7 numerical regressions pass in.620s, including
the37991-parameter addition, lift isometry/gradient, initial exact equivalence,
cross-parent sensitivity and parent-only weight properties. This validates
implementation controls, not generated-field improvement. Same allocation
continues control6000 -> evaluation -> correction6000 -> evaluation, subject
to the fixed time/memory/completeness limits. Output
/gpfs/kjhan/CF4/z0_density/bundle_c_v1/stable_field_link_repair_v1/;
logs /gpfs/kjhan/CF4/logs/cf4_C_link_repair_347264.{out,err}.

Completed2026-09-13 12:25:29 KST,42m56s,exit0. Both6000-update fits complete,
7/7 tests,16/16 valid per arm. Individual morphology0/16 control and1/16
linked; ensemble0/8 both. Bulk RMS/native1.5523->1.5203, sigma .8063->.8219;
mean log-boundary1.1098->1.0681(native1.0061). Small improvement, both NO-GO.
Close this learning extension; next BUNDLE_C_SPLIT_ATTRIBUTION.md uses saved
states only to examine the first conservative split, not another blind fit.
