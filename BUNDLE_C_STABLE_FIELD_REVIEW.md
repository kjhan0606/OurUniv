# Driver disposition of stable-field Fable5 advice

2026-09-12. Actual verdict GO with corrections, primary response preserved
in config/cf4_stable_field_fable5_v1.response.json. No code-execution audit
claimed. Q-GOAL/Q-LEAN accept one changed experiment, not a hidden repair loop.

Adopt explicit A0=1/deterministic last step, predeclared invalid-draw rules,
checkpoint -> fixed denoising -> generated fields -> role readout ordering,
incremental outputs and E-conditional labels. No physics claim from stable
latent coordinates alone. Full source and outcome remain bounded by the plan.

Correct the adviser's activation arithmetic:80^3 is the generated FINE field,
not the network lattice. The largest PARENT network input is40^3; other scales
20^3/10^3. The warning used80^3 and is8x too large for these tensor volumes.
Exact parameter count from layer arithmetic is1,490,406 (continuous743586,
categorical746820), verified again in the in-allocation test. Largest64-channel
FP32 map15.625MiB;245-channel FP32 onehot59.8145MiB (temporary int64 onehot
119.629MiB);304-channel concatenated input74.21875MiB. A conservative64-map
activation allowance across both branches is1000MiB, plus input/category
temporaries, output/gradient tensors, optimizer and backend workspace.
GPU8GiB envelope comfortably exceeds this arithmetic estimate; actual backend
workspace cannot be exactly certified from shapes, so first real largest-scale
update measures it. Keep strictFP32; no in-run mixed-precision/model switch.
Host one184x400x400x7 FP64 slab1.648GiB plus stream/latent/runtime buffers
fits estimated4.5GiB; request6GiB (20% margin rounded up).

Correct receptive-field units too:14 PARENT cells per axis, so finest radius
14*.375=5.25cMpc/h, not14*.1875. At coarser levels it spans larger physical
distances. Finite learned receptive field and outside-cube padding are still
limitations, but not proof of any future morphology failure's unique cause.

Invalid draw, frozen before launch: nonfinite latent/prediction/moment values,
out-of-chart or unrevealed category, failure of the existing conservative
decoder/realizability check, or full80/inner64 parent-moment conservation error
>1e-8. Native-chart finite-support errors count as invalid, never clipped or
resampled. CUDA/resource/time failures instead mean incomplete technical or
budget evaluation, not physical rejection of a successfully generated draw.
Morphology .5–2 ratios do not determine numerical validity. Report individual
failures even if a two-draw ensemble mean passes. Candidate feasibility is
not an independent calibration or acceptance of an actual LG posterior.

Generated buffer is part of the random field; no true fine halo is supplied
to role inference. Store first draw per8 observers as full80 seven moments
(~.214GiB uncompressed), plus checkpoints (~.1GiB), compact metadata/plots.
Target output<.5GiB; no snapshot duplication or raw new particle extraction.
The E-conditional generator, physical masses/COM and coarse CF4 field prior
are distinct contracts. No claim that the current archive E is an observed
selection law or that q_E can be reused as an unconditional cosmological prior.
