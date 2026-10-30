# Seed 40349 early-nonlinear gate audit disposition

## Primary review

The read-only Fable5 request is preserved in
`config/cf4_s40349_zoom_early_fable5_prompt_v1.txt` and its response in
`config/cf4_s40349_zoom_early_fable5_response_v1.json`.  The CLI completed
normally but returned only an intention to read the drafted files.  It gave no
verdict or answers to Q-GOAL/Q-LEAN, so it is **NO VERDICT**, not a pass.

## Astra driver backup audit

**Verdict: GO for this one bounded early-nonlinear gate.**

- Q-GOAL: PASS. The two-step gate never populated a level above L12. Evolving
  the same conditional IC to a~0.05 directly tests nonlinear AMR stability
  before contemplating a much larger z=0 run. It does not repair or bypass the
  unresolved present-field, parent-promotion, or MW/M31/M33 questions.
- Q-LEAN: PASS. One target epoch and one final dump are the minimum useful
  calculation because the log supplies numerical/refinement evidence while
  the dump is needed for density and contamination inspection. No initial,
  periodic, checkpoint, halo-finder, or z=0 output is included.
- Resources: defensible. The measured historical startup-to-early memory ratio
  predicts 79.9 GiB; 96 GiB supplies 20.1% margin. Eight times the L8 global
  particle count offset by eight times as many cores predicts roughly the old
  22-minute wall time under ideal scaling; two hours leaves substantial
  communication/I/O margin. The historical 3.5 GiB final dump scaled by eight
  predicts 28 GiB, below the declared conservative 32 GiB, versus 327 TiB free.
- Capacity: the current 19.77 million loaded grids plus the historical scale of
  early extra refinement fits the 24 million initial grid capacity; runtime
  growth remains enabled. Particle capacity is 190 million for 138.63 million
  loaded particles. A failure remains a failure, not authority to silently
  enlarge the run.

No essential pre-submission correction remains after the effective-namelist,
single-output, convergence, boundary, refinement and resource checks in the
runner. MW/M31/M33 identification, M33 deblending/ID tracking, z=0 halo finding,
contamination, environment agreement, seed ensemble robustness, and parent
promotion are deferred science and must not be claimed by this gate.
