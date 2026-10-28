# R2 block-ESS pilot result — 2026-10-28

## Driver disposition

**Close this ESS branch.** Slurm job438631 exited successfully, but the
science pilot did not finish its predeclared three-block sweep before the
application walltime limit. Do not extend or repeat the run, tune its bracket
or Fourier boundary, or promote the terminal state. This was a one-anchor
feasibility test, not posterior inference. R2 remains **NO-GO**; R3 has not
started.

The current contract explicitly says to stop this branch if its budget or
target guard is hit rather than tune it in place. Astra independently
recommended closure at this turning point, with no automatic rerun or
multi-start from these data. The driver checked that recommendation against
the contract and adopts it. No simulation, sampler, heldout scoring, or field
promotion follows. A separate, tightly capped scalar-evaluator cost check is
being prepared after code inspection found that the old score-only path
computes observation gradients and discards them; it does not reopen ESS.

## Authoritative run evidence

- Slurm: `COMPLETED`, exit `0:0`, elapsed `03:01:05`; the job is terminal.
- Application: `ESS_PILOT_INCOMPLETE_BRACKET_NOT_TRANSITION`, stopped in the
  nuisance block because the `10,800s` application ceiling was reached.
- Exact target evaluations:15 including the anchor; maximum contract was25.
- Completed blocks: `ic_low_fourier`, `ic_high_fourier`; nuisance incomplete.
- Peak application host RSS:14.06GiB; memory was not the stop condition.
- No posterior sample or uncertainty, no field output/promotion, and no
  heldout values or scores. The run report says `R2_complete=false`,
  `posterior_sample=false`, `posterior_uncertainty=false`,
  `heldout_scored=false`.

Per-block results:

| Block | Exact proposals | Elapsed | Accepted angle | Prior-normalized RMS move | Interpretation |
|---|---:|---:|---:|---:|---|
| Low Fourier, dimension1,097,917 |7|4,304s|−8.2419×10⁻⁵|8.2456×10⁻⁵|Only a negligible move from this one anchor; not a posterior-width or mixing estimate.|
| High Fourier, dimension15,679,299 |5|2,926s|−0.099817|0.099783|A measurable move, but not evidence that the observation-constrained low modes are explored.|
| Nuisance, dimension24 |2|2,846s|—|—|Stopped by the global time ceiling; not a completed bracket or evidence that the block cannot move.|

Timing correction: the roughly70s recorded in support summaries is support
packing, not a complete likelihood evaluation. Dividing block elapsed time by
proposal evaluations gives about615s per low-block evaluation and585s per
high-block evaluation. The nuisance block's two evaluations consumed2,846s
in total. Slurm reported batch MaxRSS `14120848K`; the application report's
peak is14.06GiB.

The job ran the unchanged conditional-v6 target:47,121 training count rows
(37,951 keys) and1,414 linked raw-FP marks at N256 /384 cMpc/h, with the
heldout values not accessed by this run. The field grid spacing is1.5
cMpc/h; that sampling interval is not proof of observational resolution.
The mark-presence/selection limitation remains: selected marks alone do not
identify their availability law or a full-survey calibrated CF4 posterior.

Run stdout/result record:
`/gpfs/kjhan/CF4/logs/cf4_r2_ess_438631.out`.

## Astra turning-point advice and driver judgment

Astra advised closing the current prior-ellipse ESS branch: no time extension,
same-run restart, or multi-start expansion. The low-mode move is practically
negligible, but because it is one anchor it cannot quantify posterior width
or mixing. The high-mode move does not substitute for low-mode exploration.
The nuisance timeout is a budget stop, not a block failure. Astra also
corrected the cost interpretation above.

Before any new sampler, one bounded Slurm diagnostic is planned to compare
the old matched-gradient scalar score with the direct value-only primal at the
same anchor, test exact repeatability at that anchor, and record target/
support timings. It uses one full anchor replay plus one matched scalar and
three direct scalar evaluations; one nuisance coordinate is perturbed only
to invalidate the same-point cache between direct repeats. The 2h30m job
requests 18GiB host RAM (about28% above the observed14.06GiB peak). It produces
no field or posterior and reads no heldout values. Even a faster direct-value
route would address evaluator cost only, not the low-k movement failure.

Job439009 was submitted on `h100` / `gpu:H100:1` after a typed-GPU
snapshot (H200:1 free of8, H100:2 free, A100:1 free). It pinned code commit
`a456d997a0049e8a533bd0389ca51e76c2427a8c`; while it waited, a later
documentation commit changed `CF4_MASTER_PLAN.md`, which is explicitly in
the runner's startup diff guard. The driver cancelled 439009 before allocation
to avoid a guaranteed pre-calculation source-guard failure. It performed no
evaluation and consumed no GPU. Submit at most one corrected copy of the same
bounded diagnostic, pinned to the then-current committed HEAD after a fresh
H200-first typed-GPU check; do not change its target, cap or partition policy.

The following sampler bundle remains design-only until evidence is ready:

1. Keep the current conditional estimand explicit and state that conditioning
   on observed linked-mark presence does not establish ignorability. Do not
   infer missing failures or fit selection probabilities from selected rows.
2. Design a training-only proposal for correlated, observation-constrained
   low-k directions. It must state its full-target correction and how the
   complementary field and nuisance coordinates move; do not merely adjust
   the ESS angle, cutoff, or bracket.
3. Cost basis construction and exact target evaluation, set a capped and
   falsifiable movement/compute criterion, and identify a hard-stop result
   before any sampler pilot. No posterior/heldout claim follows from a design.

The proposal is not yet selected: the evidence does not establish a best new
sampler. If no exact, costed proposal can be justified, do not launch another
inference job. External denominator or justified sensitivity information is
still required for any full-survey selection claim.

## Gaussian-reference ESS design advice — 2026-10-28

The driver requested Astra's advice because choosing a replacement sampler is
a material R2 turning point. Astra judged the Gaussian-reference ESS math
valid only for one fixed full-rank reference with the exact residual
`log pi - log g`; a state-dependent plug-in reference is not covered by
that invariance argument. The complete unchanged target must score every
transition. A joint ESS in fixed Gaussian-whitened coordinates is the
simplest proposed move; if separate blocks are later used, define them in
those coordinates or use exact Gaussian conditionals. Full rank does not
guarantee useful movement, especially for the complementary modes or the
field–population coupling.

The count Fisher is exactly zero for the 15 population coordinates because
the count intensity does not depend on them. A separate, fixed-anchor
15-dimensional raw-FP curvature block is a reasonable proposal component;
include prior precision once, and treat PSD clipping as proposal geometry
only. The finite VJP sketch has the correct unbiased matrix expectation, but
its sample eigenvalues can be severely inflated (for example, identity
information in dimension d with r probes gives nonzero sketch eigenvalues of
order d/r). Use discovery probes only to propose a subspace, then independent
training-only probes to check projected information, residual information,
uncertainty and stability. These are still not evidence of full posterior
rank.

Driver disposition: adopt these corrections in the design draft, but do not
implement or submit this sampler yet. The reverse-pass count, useful
movement criteria, memory/time caps and falsification thresholds are not
numerically specified, and an adjoint may cost roughly a full expensive
N256 evaluation. If these cannot be costed within a small fixed cap, close
this sampler candidate. This review does not reopen the closed prior-ellipse
ESS branch or change R2's NO-GO disposition.

Q-GOAL: ESS feasibility work was relevant to the first z=0 conditional field
posterior, but achieved none of the posterior, uncertainty, or heldout
requirements. The eventual local-group requirement remains MW/M31 role
resolution and M33 handling through observables on the same NEW evolved LG
field at<=0.3 cMpc/h; native truth IDs remain evaluation/calibration only.

Q-LEAN: the fixed pilot was a bounded and relevant test. Continuing with
automatic repeats, an ESS parameter ladder, heldout scoring without valid
draws, extra gravity simulations, or field promotion would be unnecessary.
