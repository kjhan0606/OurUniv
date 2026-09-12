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

Status: implementation/static Python/shell checks prepared, no numerical test
or field-generation outcome claimed yet. Conditional on native coarse field
and archive satellite eligibility E; not q(coarse|CF4), calibrated member
mass/COM/existence/observer selection, an actual LG posterior, or IC.
