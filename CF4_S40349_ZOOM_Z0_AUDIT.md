# Seed 40349 trace-only zoom z=0 plan audit

Date: 2026-09-23

Auditor: Claude Fable 5, read-only planning consultation

Driver disposition: accepted with the two essential corrections implemented

## Verdict

`CONDITIONAL PASS`

The auditor found that the single z=0 forward run materially tests whether one
conditional fine realization survives to the present epoch without exceeding
the claim ceiling.  It also found the plan lean: one new final dump, no
intermediate checkpoint, no extra gate ladder, and a separate halo/structure
decision after numerical completion.  The explicit trace-only status,
unpromoted parent, unresolved M33, dual GalaxyFinder/HOP analysis, and allowed
NO-GO outcome correctly prevent runtime success from becoming scientific
promotion.

## Essential findings and disposition

1. **Mechanically enforce the same-rank restart. — Accepted.**  The wrapper
   now parses `ncpu` from the source `info_00002.txt` and requires it, the
   Slurm task count, and the fixed contract all to equal 32.
2. **Pin the runtime MPI/compiler environment. — Accepted.**  The wrapper now
   purges inherited modules and loads the exact Intel MPI 2021.17, Intel
   compiler 2025.3, and GNU 13.2 stack verified on grammar.  The binary has no
   unresolved dynamic libraries under that stack.

These corrections convert the conditional pass to driver `GO` for submission.

## Deferred observations

- Actual final dump size will be recorded; the large free-space margin makes
  an estimate error non-blocking.
- A midpoint checkpoint remains unnecessary unless a late runtime failure is
  actually observed.
- The wrapper's timeout leaves about 30 minutes inside the allocation; a
  timeout will be an operational failure, not a scientific NO-GO.
- `feedback_mode='legacy'` is inherited inert metadata in this DMO run; the
  successful a=0.10 checkpoint exercised the same block with `hydro=false`.
- The final science report must state that any M33 analogue comes from random
  conditional high-k phases and is not CF4-recovered information.  This is now
  explicit in the plan.
