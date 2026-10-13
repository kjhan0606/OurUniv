# E-conditional stable-field generator experiment

2026-09-12 autonomous continuation. Position-model and observation-link work
closed with bounded component passes347085/347087; neither is a density map.
This experiment returns to the physically conservative total density/velocity
generator. Plan/review: BUNDLE_C_STABLE_FIELD_{PLAN,REVIEW}.md. Fable5 GO;
driver corrects its80-vs40 network-grid memory estimate and receptive-field
units, adopts explicit invalid-draw/t1/metadata/evaluation-order conditions.

New1,490,406-parameter joint model: separate continuous v and categorical
branches, full-resolution dilated residual convolutions, direct time-dependent
noisy-coordinate gains, explicit transformed observer coordinates. v is a
diffusion target, NOT peculiar velocity. Prior failed field diffusion already
used dense translations: this is a changed objective/model/observer-conditional
law, not an unmodified longer fit or a claimed unique-cause diagnosis.

Existing1388 native train observers,80^3 buffered full matter at.1875,
1.5 native parent10^3 ->20^3 ->40^3 ->80^3, inner64 used by the frozen location
law/observation likelihood with GENERATED fine buffer. No native member roles
are supplied on generated fields. Uniform observer weighting, native slab
guard and correctly transformed observer under48 augmentations.64 fixed
training observers/augmentations normalize the chart; test data unused in fit.

One fresh24,000-step fit, AdamW2e-4/decay.01, clipping10 per branch, EMA.999,
100 reverse steps. Three numerical tests, native roundtrip, learning, final
checkpoint, fixed denoising,16 generated fields and128 role triples in SAME
Slurm job. Old fields/checkpoints preserved; no automatic refit after a miss.
Per-draw morphology and two-draw ensemble morphology are distinct reports;
no invalid draw discarded or retried, no amplitude/seed correction.

1GPU/2CPU/6GiB/4h; host estimate4.5GiB+20% rounded6, GPU ceiling8GiB;
learning3h inside application3h50. FP32, no manual syn101 or login-node
numerics. Checkpoints and first8 full-field draws/projections expected<.5GiB.
Output new /gpfs/kjhan/CF4/z0_density/bundle_c_v1/stable_field_v1/.

Source0bfd832533c439528faa97c676d93780998e8866 committed/pushed; Slurm347088
started2026-09-12 20:19:41 KST on syn05/A40.1GPU/2CPU/6GiB/4h allocated;
scheduler end limit2026-09-13 00:19:41 KST. All3 numerical tests pass in.380s,
including exact parameter count and isolated gradient paths. All64 native
normalization observers/3scales processed:192 native roundtrips total.
Buffered data guard passed for1388 train /8 test observers.

Training started at application52.73s. First200 updates elapsed~40.5s; step400
at131.07s. Steady200-step segment39.23s suggests roughly80min total learning,
plus generation/evaluation: provisional80–95min from startup, not a guarantee.
Step200 t59 v-MSE .7450 versus zero-v .9854; step400 t17 .9220 versus.9957.
These are DIFFERENT native training cases/times and only evidence of an active
learning path, not validation or successful generated structure. Peak host
3.030GiB and GPU reserved1.176GiB, within requested/envelope limits.

Status: RUNNING, source-frozen learning then final denoising/draws/role readout
automatically continue inside this same allocation without user approval.
No separate daemon/process scans or automatic extra training. Current artifacts
are /gpfs/kjhan/CF4/z0_density/bundle_c_v1/stable_field_v1/{status,history,
representation,request,tests,population}.json; stdout/stderr
/gpfs/kjhan/CF4/logs/cf4_C_stable_field_347088.{out,err}.
Next check fixed job347088 and final result.json, not a broad process search.
Driver evaluates the completed/incomplete outcome before another experiment;
autonomous authority remains active, so do not request redundant approval.

Conditional on native coarse field and archive satellite eligibility E;
not q(coarse|CF4), calibrated member mass/COM/existence/observer selection,
an actual LG posterior, or IC. No field-generation success claimed yet.

## Evaluation-only recovery

347088 FAILED2026-09-12 21:40:25 KST after1h20m44s. All24000 updates and
the final model/EMA/Adam checkpoint completed; training4785.96s. Evaluation
never started because `dict(**RESULT, status='EVALUATING')` duplicated the
existing status key. This is a driver implementation error, not evidence of
field-generation failure. Preserve stable_field_v1 and its failed result.

Correction updates the status once and adds evaluation-only entry. It loads
the exact final EMA, saved normalization, unchanged config/seeds and fixed
eight cases, checks completed-step/source consistency, and executes the SAME
denoising/16-draw/readout endpoint with ZERO optimizer steps. One focused
checkpoint-contract regression joins the three existing tests. New output
stable_field_eval_v2; no overwrite, refit, extra model or threshold change.
One Slurm GPU/2CPU/6GiB/30min; prior measured host3.11GiB plus20% fits6GiB.
Tests and numerical evaluation run in that allocation, not on the login node.

Correction source c04c3c4 committed/pushed; evaluation-only Slurm347108
submitted with30min walltime. Initial PENDING(Priority); no numerical recovery
pass inferred from submission. Logs cf4_C_stable_eval_347108.{out,err}.

347108 COMPLETED2026-09-13 01:46:44 KST on syn07,2m23s, tests4/4 and
zero added optimizer steps. Sixteen numerically valid/conservative fields;
terminal latent RMS .926–1.077, inner moment conservation max8.37e-16.
All fixed denoising scale/time mean MSE ratios improve on zero-v (.360–.905).
But individual morphology0/16, two-draw summary means2/8 pass: boundary ratio
fails16/16, high-k power5/16, connected fraction2/16. Status remains NO_GO.
Generated role samples and actual-position diagnostic scores completed,
not actual LG conditioning, physical members or independent validation.

User-approved no-fit boundary diagnosis347159 now completed8s. Existing
ratios reproduce; raw metric is extremely rare-peak sensitive (10 edges
carry median85–87% of boundary squared-gradient energy), while log-density
still supports modest grid-periodic roughness. Mixed evidence, not a pure
metric bug or permission to promote the generator. Details/results:
BUNDLE_C_BOUNDARY_DIAGNOSIS.md. No job from this bundle remains active.
