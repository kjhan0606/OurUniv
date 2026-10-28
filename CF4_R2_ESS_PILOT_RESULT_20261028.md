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

Before any new sampler, one bounded Slurm diagnostic will compare the old
matched-gradient scalar score with the direct value-only primal at the same
anchor, test exact repeatability at that anchor, and record target/support
timings. It uses one full anchor replay plus one matched scalar and three
direct scalar evaluations; one nuisance coordinate is perturbed only to
invalidate the same-point cache between direct repeats. The 2h30m job requests
18GiB host RAM (about28% above the observed14.06GiB peak). It produces no
field or posterior and reads no heldout values. Even a faster direct-value
route would address evaluator cost only, not the low-k movement failure.

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

Q-GOAL: ESS feasibility work was relevant to the first z=0 conditional field
posterior, but achieved none of the posterior, uncertainty, or heldout
requirements. The eventual local-group requirement remains MW/M31 role
resolution and M33 handling through observables on the same NEW evolved LG
field at<=0.3 cMpc/h; native truth IDs remain evaluation/calibration only.

Q-LEAN: the fixed pilot was a bounded and relevant test. Continuing with
automatic repeats, an ESS parameter ladder, heldout scoring without valid
draws, extra gravity simulations, or field promotion would be unnecessary.
