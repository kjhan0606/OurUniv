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

Before any new sampler, one bounded Slurm diagnostic compared the old
matched-gradient scalar score with the direct value-only primal at the same
anchor and measured repeatability plus target/support timings. It used one
full anchor replay, one matched scalar evaluation, and three direct-value
evaluations; one nuisance coordinate was perturbed only to invalidate the
same-point cache between direct repeats. It produced no field or posterior
and read no heldout values. Even a faster direct-value route addresses
evaluator cost only, not the low-k movement failure.

Job439009 was submitted on `h100` / `gpu:H100:1` after a typed-GPU
snapshot (H200:1 free of8, H100:2 free, A100:1 free). It pinned code commit
`a456d997a0049e8a533bd0389ca51e76c2427a8c`; while it waited, a later
documentation commit changed `CF4_MASTER_PLAN.md`, which is explicitly in
the runner's startup diff guard. The driver cancelled 439009 before allocation
to avoid a guaranteed pre-calculation source-guard failure. It performed no
evaluation and consumed no GPU. The one corrected copy, 439023, was pinned to
the then-current committed HEAD after a fresh H200-first typed-GPU check and
completed as recorded below.
The corrected submission, job 439023 on h100 / gpu:H100:1 pinned to
`eb068b4b1d6ec5f99d276f5ddfb99dda52d4af7e`, completed on syntax with exit 0
in 25:19; Slurm batch MaxRSS was 10.64 GiB under the 18 GiB request. The fixed
anchor replay, matched scalar score, and three direct-value scores all
completed. The matched score evaluation took 400.46 s; direct-value evaluations
took 178.01, 142.38, and 144.50 s (mean 154.97 s, ratio 0.387, about 2.58x faster).
Direct-minus-matched objective was 0; only the FP term differed by
`-9.09e-13` nat. The aggregate q0 and repeated scores are identical, but the
strict termwise repeat flag is false because the FP reduction differs by one
float64 ULP (`9.09e-13` nat). Treat this as roundoff-level non-bitwise
agreement, not a scientific mismatch; preserve the emitted
`VALUE_ONLY_REPEAT_MISMATCH_NOT_POSTERIOR` status rather than rewriting the
historical artifact. This supports direct `obs.value` for cost feasibility,
not low-k movement or sampler efficacy. No field, posterior transition,
uncertainty, or heldout score was produced. Full Slurm output:
`/gpfs/kjhan/CF4/logs/cf4_r2_valueprobe_439023.out`.

The first submission of the already specified training-only count-intensity
VJP, job 439311 on H200, failed at anchor metadata validation after 20 seconds.
The preflight required a top-level `likelihood_changed` field that the retained
anchor's result does not write; the actual recorded target contract, source
revision, checkpoint replay, and target orders match. No forward map or reverse
pass ran, and no scientific result was produced. The validator now checks the
recorded target rules, training counts, holdout non-access, replay and target
orders directly. Its two regression tests pass, and it accepts the actual
retained anchor with the expected coordinate shape. The next action is one
corrected, committed/pinned resubmission after a fresh H200-first typed-GPU
snapshot. Keep the 20 GiB host request and 2 h application / 2.5 h Slurm caps;
this remains a feasibility measurement only and authorizes neither sketches
nor a sampler. R2 stays NO-GO.

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

A further read-only source check confirmed the train exposure mask includes
zero-count cells, and the count intensity can be exactly zero for some
exposed outputs because of the source-side selection/deposition operators.
Astra advised that \(\lambda=0,J=0\) does not establish zero Fisher
information: \(\lambda(t)=t^2\) has a positive-side \((\lambda')^2/\lambda\)
limit of4. The driver therefore rejects the “full Poisson Fisher” label for
this sketch. The only current candidate is a fixed-anchor,
positive-intensity-row PSD surrogate used solely to construct the Gaussian
proposal; all train-exposed rows, including zero and newly activated cells,
remain in every exact target/residual evaluation. No arbitrary intensity
floor and no full structural-zero census. This does not guarantee useful
mixing; nonfinite/over-cap geometry stops the candidate rather than being
regularized invisibly.

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

## Addendum — corrected count-intensity VJP and bounded next check (2026-10-30)

The corrected training-only N256 count-intensity VJP, job439314, completed on
H200 (`gpu:H200:1`) with exit0 in4:31 (batch MaxRSS5.95GiB). Result:
`/gpfs/kjhan/CF4/z0_density/r2_intensity_vjp_feasibility_20261029/439314/result.json`.
At the fixed conditional-v6 anchor, the one finite pullback had white-IC norm
957.57 and tracer norm230.06. The map covered 11,009,805 training-exposed
population-cells: 1,993,850 positive and 9,015,955 zero. Minimum positive
intensity was1.9562e-37 and maximum inverse-square-root weight2.26094e18.
Peak device allocation was43.78GiB. This shows one reverse pass is
computationally feasible on this H200, not that tiny-intensity rows dominate,
that a useful spectrum is stable, or that a sampler can mix. No heldout
outcome/score, posterior, uncertainty or field was produced.

Astra's mid-course response and the driver's bounded disposition are recorded
in `CF4_R2_VJP_TAIL_SCREEN_ASTRA_REVIEW_20261030.md`. The prepared one-hour
screen uses two fixed Gaussian seeds and the lowest-intensity1% positive-row
partition, with full/tail/rest VJPs per seed and one repeated full VJP (seven
pullbacks total). It measures decomposition/repeat error and per-coordinate
gradient concentration, without changing the target, applying an intensity
floor, or accessing heldout outcomes. Its pass only permits consideration of
a separately budgeted sketch experiment; it is neither a sampler nor R2
closure. The implementation's five pure NumPy tests and static checks pass;
the screen has not yet run. R2 remains NO-GO.
