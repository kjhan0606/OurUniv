# R2: longer trajectories on the same conditional target

R1 -> **R2 in progress** -> R3 same-NEW-field MW/M31/M33 -> R4 -> R5.

408296 completed32 proposals in58m42s:27 accepted overall,15/16 after discarded
adaptive warmup. Frozen step0.07502935785806791, two integration steps.
Initial direct/gradient primal and canonical-coordinate seam checks agree;
all independent proposal-endpoint target comparisons passed. Initial IC white
mean-square0.00474704, final0.33691468; this is still a cold-start trace, not
stationary posterior uncertainty. Final FP factor-12.90897 and zero+.03094645
dex are conditional diagnostics, not proof of better calibrated predictions.
Count/IC-prior changes make the full potential larger; sampling is NOT MAP
minimization. Neither forcing white power to1 nor selecting the lowest-potential
sample is valid here. Saved-field/PDF readouts408331/408305 require inspection.

## Driver decision: one bounded continuation, not production

Keep the identical1414 FP/47121 count target, N128/384 geometry, priors,
proposal metric arrays and full Metropolis correction. Start from the accepted
terminal state, freeze the same step, use8 integration steps per proposal and
8 proposals with no new adaptation. New random seed2026092904; the previous
trace is not appended or promoted into stationary samples. This implements
the useful direction of Fable advice in CF4_R2_AFTER_SPLIT_ADVICE_20260929.md,
but reduces its prospective128-force experiment to64 forces (~47min measured
force time),75min application/90min Slurm,H1002CPU24GiB. This is a NEW bounded
bundle, not a retroactive extension of the original4GPU-hour control budget.

The previous run used5.56GiB host peak and17.94GiB device temporary storage;
24GiB host request remains conservative and H100 has ample measured device
headroom. No N256 launch, source-prior change, heldout use, TNG dependency or
extra independent gravity simulation. Larger calculations require a new cost
and science decision. All numerical work and tests stay in Slurm.

Compare accepted squared movement per force, energy errors and continuing
IC/zero/long-mode drift. Different sequential starting states mean this is
not a controlled causal efficiency experiment, and8 points cannot establish
ESS. A higher acceptance fraction alone is not the success criterion. Save
all scalar diagnostics/rejections, rolling checkpoint and one terminal field;
reuse illustrated Korean report and field readout, with no per-step large
snapshots. Assess terminal evidence before another continuation.

Code changes: support a completed same-target HMC restart; explicitly preserve
the frozen proposal metric/step; reject nonfinite Hamiltonian arithmetic at a
finite target instead of risking NaN acceptance. True zero-density support
still produces an ordinary rejection. Small Gaussian mechanics and restart
tests run in Slurm before the real target; no new general validation framework.

Q-GOAL: establish whether the actual conditioned field can be explored at a
usable cost; R2's 1–2cMpc/h map/uncertainty/calibration/prediction remain open.
MW/M31 roles are ambiguous, M33 unresolved. R3 must identify candidates from
this SAME generated field and constrain its physical components with LG data;
truth identities never provide candidates or select favorable realizations.
Q-LEAN: one changed trajectory length,64 force evaluations, existing readouts;
no Hessian sweep, arbitrary zero broadening, shell refit or new audit gate.
