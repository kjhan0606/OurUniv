# N256 larger-sampling advice and driver disposition

R1 -> R2 actual present-state posterior (ONGOING) -> R3 same-field LG ->
R4 precise evolution -> R5 phase-consistent zoom IC. R2 is NOT complete.

Read-only Fable5 CLI completed normally, exit0, session80277. Request:
`config/cf4_r2_n256_sampling_advice_20260929.md`. This file is a SUMMARY and
driver disposition, not a verbatim transcript. Trigger: advice before a
potentially large two-chain GPU allocation, not routine per-step auditing.
No larger production allocation has been submitted.

## Advice

CONDITIONAL: proceed after408412 supplies actual N256 transition acceptance,
fine-value cost and memory evidence. At review time those were pending.
Q-GOAL positive: conditional1414 raw FP/K marks plus47121 counts, actual
N256/1.5 history-linked posterior is a legitimate first R2 delivery, not LG,
all-CF4 or external absolute-scale calibration. Q-LEAN: retain the current
observed128 keys/exposure, abandon the unneeded compact-backend gate, no
additional proxy/cohort ladder or repeated audits.

Fable found no mathematical error in the inspected split-HMC/target wiring:
deterministic reversible prior-rotation/likelihood-kick proposals, fine
Hamiltonian Metropolis correction, accepted-state rejection cache, state-local
support rebuilding and1/8 physical count-source rate normalization.

For larger sampling: select longer trajectories from measured transport per
force evaluation and Hamiltonian errors during discarded warmup, then freeze
proposal settings. Consider a bounded2x24GPU-hour first allocation only after
the pilot costs are known. Use independent random streams and genuinely
different starts where feasible; report common ancestry. Track inherited
low-band power, fundamental projections, all nuisance variables, count/raw
scores and a few declared density summaries, including rejections. Measure
ESS/MC error and drift; acceptance alone is insufficient. Same-grid finer
quadrature and weak common-scale-prior sensitivity can use reweighting ONLY
with adequate overlap/weight ESS. Freeze before one heldout assessment.

Observed3-cMpc cells vs1.5-cMpc source field: defensible if reported honestly.
Separate expected-count exposure, FP source-support occupancy and posterior
vs prior band variance; none alone is a measured resolution/identifiability
proof. Do not casually resplit or call all fine-grid structure observed.
MW/M31 ambiguity and unresolved M33 persist; all later identification and
observables act on the SAME NEW field, without truth-ID seeding. N256 particle
mass2.90e11Msun/h is environment inference, not resolved LG halos.

## Driver checks, corrections and decisions

1. Adopt the finite longer-trajectory/two-chain/sensitivity/prediction sequence,
   conditional on measured408412 outcomes. No new automatic external gate.
2. Correct the claim that streaming moments and all necessary traces are
   already implemented. `cf4_r2_n256_joint_pilot.py` saves a final accepted
   field, checkpoint and short scalar trace; it does NOT accumulate posterior
   moments or record every nuisance/mode. These must be implemented before
   longer science sampling. Rejected states count as repeated samples.
3. Correct a notation slip: c is inverse momentum mass. The implementation
   draws Var(p_hat)=1/c, NOT c, with kinetic .5*c*|p_hat|^2. The code is right.
4. Do NOT adopt the review's unproved 'almost surely measure-equivalent'
   coarse/fine support assertion. Generic corrected_split_step can reject a
   coarse +inf endpoint, but the CURRENT physical oracle raises on nonfinite
   energy/support failures and stops the job; it does not silently proceed
   on a newly restricted posterior. No general coarse/fine support equality
   has been proved. Keep finite-support/numerical failures explicit.
5. Two chains do permit split-Rhat with adequate trace length; they do not
   certify exploration of all modes. Shared starts and short traces weaken
   the evidence. The suggested O(10) effective draws per chain is exploratory
   evidence, not a universal scientific UQ threshold; report actual MC error.
6. Do not blindly redraw all24 nuisances and the entire universe from broad
   priors for an expensive run. That can create a very poor data-supported
   start or exceed the declared LOS/source workspace. Prefer a declared,
   unselected alternative initialization with independently drawn small-scale
   modes and different available low-band state, documenting shared ancestry.
   A prior-conditioned band refresh may diversify starts; it is initialization
   ONLY, never a fixed science low/high boundary. All modes remain live.
7. Fine GL4 atN256 must respect the raw-component memory ceiling. Naively
   packing the entire cohort could exceed40million components; use row-streamed
   exact accumulation if needed, never drop components or call a smaller
   quadrature the promised sensitivity. This is a required computation, not
   already implemented or passed merely because it appears in a plan.
8. The observed grid/completeness transfer is a declared limitation. Source
   support occupancy is coverage, NOT information gain. Prior/posterior
   variance comparison and predictive evidence must accompany any influence
   map; no invented information-resolution claim from voxel size alone.

After advice completed,408412's full PM+24-nuisance FD passed:
AD321802.89445, FD321780.67501, relative6.90468e-5 (predeclared limit.002).
This does not yet supply acceptance, stationarity, covariance or heldout
prediction. The pilot continues; final larger-run lengths/settings remain
conditional on its measured outcomes.
