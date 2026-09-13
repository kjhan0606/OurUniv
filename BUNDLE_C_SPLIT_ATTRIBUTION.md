# Close linkage repair; attribute the first physical split error

2026-09-13, next bundle authorized. Driver-reviewed small diagnostic;
no external-review trigger or new approval wait.

347264 completed42m56s,7 tests pass,6000 updates per arm. Both generate16/16
valid fields; individual quality control0/16, linked1/16, ensemble both0/8.
Bulk RMS/native1.5523->1.5203, sigma .8063->.8219, log-face ratio1.1098->1.0681
(native1.0061). Neither arm is accepted. Close this extra-training/fine-path
repair: no longer run or weight sweep on these results.

Next deliverable: ONE saved-field physical-decoder intervention table at the
first1.5->.75 split, where the generated and native parent are demonstrably
the same. Both completed arms, eight preserved first draws, four modes each:
native and generated lossless roundtrips, native legal codes with generated
continuous values, generated legal codes with native continuous values.
Reuse pack/unpack, conservation and physical metrics. No model training,
fresh sampling, evaluation-threshold changes or amplitudes adjusted.

This tests sensitivity to the two parts of the physical representation,
NOT unique causal attribution to a neural branch. Re-encoded legal codes
are not raw network codes. Hybrid states break joint dependence; inactive
coordinates can become active and canonicalization can change with inherited
budgets. Report decoder corrections and category counts, never interpret
hybrids as a usable prior or physically recovered field. Do not inject native
categories or coordinates into a future observed reconstruction. Do not
extend swaps to fine levels with different conditioned parents.

Decision: substantial hybrid improvement motivates a specifically justified
joint/code or continuous-representation design; lack of improvement rules
out simply substituting that piece as a sufficient remedy. Neither outcome
authorizes an automatic new training sweep. Driver must inspect physical
effect sizes, not just a pass count; preserve original NO-GO.

Q-GOAL: locate the unresolved matter/velocity partition defect preventing a
usable z=0 prior and same-field LG observational conditioning. MW/M31/M33
remain the existing probabilistic location readout of a NEW generated field,
with unresolved M33 and missing physical member mass/COM law. This diagnostic
does not seed roles with true identities or advance actual LG inference.
Q-LEAN: existing two saved output files and source cubes, one short CPU job,
no new framework/tests/model.32 inline roundtrip controls directly validate
the intervention.2 CPUs, estimated peak<=2.5GiB (imports plus streamed cubes),
3GiB requested including20% headroom,5min allocation/4min application.
Output small JSON only; no new full density cubes or GPFS diagnostics.

Execution: source c49c73305c188ea63003ab6a96b7410946300713 committed/pushed;
Slurm352595 submitted. Static compilation/bash syntax/diff checks pass.
Output /gpfs/kjhan/CF4/z0_density/bundle_c_v1/stable_field_split_attribution_v1/.
Submission is not a completed numerical or scientific result.
